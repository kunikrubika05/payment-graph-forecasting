"""Benchmark: temporal neighbor sampling across Python / C++ / CUDA backends.

Measures latency and throughput for sample_neighbors and featurize operations
across different batch sizes and K values.

Usage:
    source venv/bin/activate

    # CPU-only benchmark (Python + C++ if compiled)
    PYTHONPATH=. python scripts/bench_sampling.py

    # Include CUDA (requires GPU + compiled CUDA extension)
    PYTHONPATH=. python scripts/bench_sampling.py --cuda

    # With real graph data from Yandex.Disk
    YADISK_TOKEN="..." PYTHONPATH=. python scripts/bench_sampling.py --cuda --real-data

Output: table of (backend, scenario, operation, median_ms, p95_ms, speedup_vs_python)
"""

import argparse
import time
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BenchScenario:
    """Benchmark scenario configuration."""
    name: str
    batch_size: int
    num_neighbors: int
    num_nodes: int
    num_edges: int


SCENARIOS = [
    BenchScenario("graphmixer_small",  batch_size=512,  num_neighbors=20,
                  num_nodes=100_000, num_edges=1_000_000),
    BenchScenario("dygformer_small",   batch_size=512,  num_neighbors=512,
                  num_nodes=100_000, num_edges=1_000_000),
    BenchScenario("dygformer_medium",  batch_size=2048, num_neighbors=512,
                  num_nodes=100_000, num_edges=1_000_000),
    BenchScenario("dygformer_large",   batch_size=2048, num_neighbors=512,
                  num_nodes=1_000_000, num_edges=10_000_000),
    BenchScenario("real_scale",        batch_size=512,  num_neighbors=512,
                  num_nodes=5_000_000, num_edges=20_000_000),
]


def generate_synthetic_graph(num_nodes, num_edges, seed=42):
    """Generate a synthetic temporal graph for benchmarking."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    dst = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    timestamps = np.sort(rng.uniform(0, 1_000_000, size=num_edges)).astype(np.float64)
    edge_ids = np.arange(num_edges, dtype=np.int64)
    node_feats = rng.standard_normal((num_nodes, 26)).astype(np.float32)
    edge_feats = rng.standard_normal((num_edges, 2)).astype(np.float32)
    return {
        "num_nodes": num_nodes,
        "src": src,
        "dst": dst,
        "timestamps": timestamps,
        "edge_ids": edge_ids,
        "node_feats": node_feats,
        "edge_feats": edge_feats,
    }


def generate_queries(num_nodes, batch_size, max_time=1_000_000, seed=99):
    """Generate random query batch."""
    rng = np.random.default_rng(seed)
    nodes = rng.integers(0, num_nodes, size=batch_size).astype(np.int32)
    times = rng.uniform(max_time * 0.3, max_time * 0.9, size=batch_size).astype(np.float64)
    return nodes, times


def benchmark_one(sampler, nodes, times, num_neighbors, warmup=5, repeats=20):
    """Benchmark sample_neighbors + featurize.

    Returns dict with median/p95 for each operation.
    """
    sample_times = []
    featurize_times = []

    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    for _ in range(warmup):
        nbr = sampler.sample_neighbors(nodes, times, num_neighbors=num_neighbors)
        sampler.featurize(nbr, query_timestamps=times)
        if has_cuda:
            import torch
            torch.cuda.synchronize()

    for _ in range(repeats):
        if has_cuda:
            import torch
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        nbr = sampler.sample_neighbors(nodes, times, num_neighbors=num_neighbors)
        if has_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        sampler.featurize(nbr, query_timestamps=times)
        if has_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        sample_times.append((t1 - t0) * 1000)
        featurize_times.append((t2 - t1) * 1000)

    total = [s + f for s, f in zip(sample_times, featurize_times)]
    return {
        "sample_median_ms": float(np.median(sample_times)),
        "sample_p95_ms": float(np.percentile(sample_times, 95)),
        "featurize_median_ms": float(np.median(featurize_times)),
        "featurize_p95_ms": float(np.percentile(featurize_times, 95)),
        "total_median_ms": float(np.median(total)),
        "total_p95_ms": float(np.percentile(total, 95)),
    }


def run_benchmarks(backends: List[str], scenarios: Optional[List[BenchScenario]] = None):
    """Run all benchmarks and print results table."""
    from src.models.temporal_graph_sampler import TemporalGraphSampler

    if scenarios is None:
        scenarios = SCENARIOS

    results = []

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario.name}")
        print(f"  batch={scenario.batch_size}, K={scenario.num_neighbors}, "
              f"nodes={scenario.num_nodes:,}, edges={scenario.num_edges:,}")
        print(f"{'='*70}")

        print("Generating synthetic graph...", end=" ", flush=True)
        graph = generate_synthetic_graph(
            scenario.num_nodes, scenario.num_edges
        )
        nodes, times = generate_queries(
            scenario.num_nodes, scenario.batch_size
        )
        print("done.")

        py_total = None

        for backend in backends:
            print(f"  Backend: {backend:8s} ... ", end="", flush=True)
            try:
                sampler = TemporalGraphSampler(
                    **graph, backend=backend,
                )
                metrics = benchmark_one(
                    sampler, nodes, times, scenario.num_neighbors
                )

                if backend == "python":
                    py_total = metrics["total_median_ms"]

                speedup = (py_total / metrics["total_median_ms"]
                           if py_total and metrics["total_median_ms"] > 0
                           else 1.0)

                print(
                    f"sample={metrics['sample_median_ms']:8.2f}ms  "
                    f"feat={metrics['featurize_median_ms']:8.2f}ms  "
                    f"total={metrics['total_median_ms']:8.2f}ms  "
                    f"(p95={metrics['total_p95_ms']:8.2f}ms)  "
                    f"speedup={speedup:5.1f}x"
                )

                results.append({
                    "scenario": scenario.name,
                    "backend": backend,
                    **metrics,
                    "speedup_vs_python": speedup,
                })

                del sampler
            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "scenario": scenario.name,
                    "backend": backend,
                    "error": str(e),
                })

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<22s} {'Backend':<8s} {'Sample(ms)':>10s} "
          f"{'Feat(ms)':>10s} {'Total(ms)':>10s} {'Speedup':>8s}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['scenario']:<22s} {r['backend']:<8s} {'ERROR':>10s}")
        else:
            print(
                f"{r['scenario']:<22s} {r['backend']:<8s} "
                f"{r['sample_median_ms']:>10.2f} "
                f"{r['featurize_median_ms']:>10.2f} "
                f"{r['total_median_ms']:>10.2f} "
                f"{r['speedup_vs_python']:>7.1f}x"
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark temporal neighbor sampling backends"
    )
    parser.add_argument("--cuda", action="store_true",
                        help="Include CUDA backend")
    parser.add_argument("--cpp-only", action="store_true",
                        help="Only benchmark Python + C++")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run specific scenario by name")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real stream graph data (requires YADISK_TOKEN)")
    args = parser.parse_args()

    backends = ["python", "cpp"]
    if args.cuda:
        backends.append("cuda")
    if args.cpp_only:
        backends = ["python", "cpp"]

    scenarios = None
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s.name == args.scenario]
        if not scenarios:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {[s.name for s in SCENARIOS]}")
            sys.exit(1)

    print(f"Backends: {backends}")
    print(f"Scenarios: {[s.name for s in (scenarios or SCENARIOS)]}")

    run_benchmarks(backends, scenarios)


if __name__ == "__main__":
    main()
