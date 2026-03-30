"""Benchmark: common neighbors computation across Python / C++ / CUDA backends.

Measures latency for computing CN on batches of random node pairs across
four scenarios that span sparse and dense graph regimes.

Key insight: CUDA bitset intersection dominates on dense graphs (avg degree >= 100).
C++ sorted-merge wins on sparse graphs (avg degree < 20) due to kernel overhead.

Usage:
    source venv/bin/activate

    # CPU backends only (Python + C++ if compiled)
    PYTHONPATH=. python scripts/bench_cn.py

    # Include CUDA backend (requires GPU + compiled CUDA extension)
    PYTHONPATH=. python scripts/bench_cn.py --cuda

    # Single scenario
    PYTHONPATH=. python scripts/bench_cn.py --cuda --scenario dense_medium

Output: table of (scenario, backend, median_ms, p95_ms, speedup_vs_python)
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import sparse


@dataclass
class BenchScenario:
    """Benchmark scenario configuration."""
    name: str
    n_nodes: int
    avg_deg: int
    batch_size: int
    description: str


SCENARIOS = [
    BenchScenario(
        name="sparse_like_bitcoin",
        n_nodes=500_000,
        avg_deg=6,
        batch_size=1024,
        description="Sparse real-world regime (avg_deg=6) — C++ expected to win",
    ),
    BenchScenario(
        name="breakeven",
        n_nodes=100_000,
        avg_deg=50,
        batch_size=1024,
        description="Transition zone (avg_deg=50) — shows where CUDA starts to pay off",
    ),
    BenchScenario(
        name="dense_small",
        n_nodes=50_000,
        avg_deg=200,
        batch_size=512,
        description="Dense small (social network scale) — CUDA starts to win",
    ),
    BenchScenario(
        name="dense_medium",
        n_nodes=100_000,
        avg_deg=500,
        batch_size=1024,
        description="Dense medium — primary CUDA showcase",
    ),
    BenchScenario(
        name="dense_large",
        n_nodes=200_000,
        avg_deg=1000,
        batch_size=2048,
        description="Dense large — maximum CUDA advantage",
    ),
]


def generate_adj(n: int, avg_deg: int, seed: int = 42) -> sparse.csr_matrix:
    """Generate a random symmetric binary CSR adjacency matrix."""
    rng = np.random.default_rng(seed)
    num_edges = n * avg_deg // 2
    src = rng.integers(0, n, size=num_edges).astype(np.int32)
    dst = rng.integers(0, n, size=num_edges).astype(np.int32)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sum_duplicates()
    A.sort_indices()
    return A


def benchmark_backend(
    cn,
    src: np.ndarray,
    dst: np.ndarray,
    warmup: int = 5,
    repeats: int = 20,
) -> dict:
    """Time cn.compute(src, dst) with warmup and repeats."""
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available() and cn.backend == "cuda"
    except ImportError:
        pass

    def sync():
        if has_cuda:
            import torch
            torch.cuda.synchronize()

    for _ in range(warmup):
        cn.compute(src, dst)
        sync()

    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        cn.compute(src, dst)
        sync()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "median_ms": float(np.median(times)),
        "p95_ms":    float(np.percentile(times, 95)),
    }


def run_benchmarks(
    backends: List[str],
    scenarios: Optional[List[BenchScenario]] = None,
) -> List[dict]:
    """Run all benchmarks and print results."""
    from src.models.graph_metrics import CommonNeighbors

    if scenarios is None:
        scenarios = SCENARIOS

    results = []

    for sc in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {sc.name}")
        print(f"  {sc.description}")
        print(f"  nodes={sc.n_nodes:,}, avg_deg={sc.avg_deg}, "
              f"batch={sc.batch_size}")
        print(f"{'='*70}")

        print("  Building adjacency... ", end="", flush=True)
        adj = generate_adj(sc.n_nodes, sc.avg_deg)
        rng = np.random.default_rng(77)
        src = rng.integers(0, sc.n_nodes, sc.batch_size).astype(np.int64)
        dst = rng.integers(0, sc.n_nodes, sc.batch_size).astype(np.int64)
        actual_deg = adj.nnz / sc.n_nodes
        print(f"done. actual avg_deg={actual_deg:.1f}")

        py_median = None

        for backend in backends:
            print(f"  {backend:8s} ... ", end="", flush=True)
            try:
                cn = CommonNeighbors(adj, backend=backend)
                metrics = benchmark_backend(cn, src, dst)

                if backend == "python":
                    py_median = metrics["median_ms"]

                speedup = (py_median / metrics["median_ms"]
                           if py_median and metrics["median_ms"] > 0
                           else 1.0)

                print(
                    f"median={metrics['median_ms']:8.2f}ms  "
                    f"p95={metrics['p95_ms']:8.2f}ms  "
                    f"speedup={speedup:6.1f}x vs python"
                )
                results.append({
                    "scenario": sc.name,
                    "backend":  backend,
                    **metrics,
                    "speedup_vs_python": speedup,
                })
                del cn
            except Exception as exc:
                print(f"FAILED: {exc}")
                results.append({
                    "scenario": sc.name,
                    "backend":  backend,
                    "error":    str(exc),
                })

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<25s} {'Backend':<8s} "
          f"{'Median(ms)':>10s} {'p95(ms)':>8s} {'Speedup':>9s}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['scenario']:<25s} {r['backend']:<8s} {'ERROR':>10s}")
        else:
            print(
                f"{r['scenario']:<25s} {r['backend']:<8s} "
                f"{r['median_ms']:>10.2f} "
                f"{r['p95_ms']:>8.2f} "
                f"{r['speedup_vs_python']:>8.1f}x"
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark common neighbors computation backends"
    )
    parser.add_argument("--cuda", action="store_true",
                        help="Include CUDA backend (requires compiled extension)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a single scenario by name")
    args = parser.parse_args()

    backends = ["python"]
    try:
        from src.models.graph_metrics import has_cpp, has_cuda
        if has_cpp():
            backends.append("cpp")
        if args.cuda and has_cuda():
            backends.append("cuda")
        elif args.cuda and not has_cuda():
            print("WARNING: CUDA extension not available, skipping cuda backend.")
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    scenarios = None
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s.name == args.scenario]
        if not scenarios:
            names = [s.name for s in SCENARIOS]
            print(f"Unknown scenario '{args.scenario}'. Available: {names}")
            sys.exit(1)

    print(f"Backends : {backends}")
    print(f"Scenarios: {[s.name for s in (scenarios or SCENARIOS)]}")

    run_benchmarks(backends, scenarios)


if __name__ == "__main__":
    main()
