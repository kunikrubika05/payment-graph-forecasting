"""Plot comparison charts for exp_005 (baseline) vs exp_006 (CUDA).

Generates:
  1. GPU utilization over time (baseline vs CUDA)
  2. Epoch time bar chart
  3. Training loss / val MRR curves (overlaid)

Usage:
    PYTHONPATH=. python scripts/plot_cuda_comparison.py \
        --baseline-dir /tmp/exp_005_results \
        --cuda-dir /tmp/exp_006_results \
        --gpu-log-baseline /tmp/gpu_log_baseline.csv \
        --gpu-log-cuda /tmp/gpu_log_cuda.csv \
        --output /tmp/comparison_plots

Requires: pip install matplotlib (already in most envs)
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        raise SystemExit(1)


def load_training_curves(base_dir):
    """Load training_curves.csv from experiment directory."""
    pattern = os.path.join(base_dir, "**", "training_curves.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        pattern = os.path.join(base_dir, "training_curves.csv")
        files = glob.glob(pattern)
    if not files:
        return None

    rows = []
    with open(files[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_gpu_log(csv_path):
    """Load GPU utilization CSV from monitor_gpu.sh."""
    if not csv_path or not os.path.exists(csv_path):
        return None

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_str = row["timestamp"].strip()
                ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
                rows.append({
                    "time": ts,
                    "gpu_util": float(row["gpu_util_pct"].strip()),
                    "mem_util": float(row["mem_util_pct"].strip()),
                    "mem_used_mb": float(row["mem_used_mb"].strip()),
                    "power_w": float(row["power_draw_w"].strip()),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_final_results(base_dir):
    """Load final_results.json."""
    pattern = os.path.join(base_dir, "**", "final_results.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def plot_gpu_utilization(gpu_baseline, gpu_cuda, output_dir):
    """Plot GPU utilization over time for both experiments."""
    plt = _try_import_matplotlib()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, data, label, color in [
        (axes[0], gpu_baseline, "Baseline (C++ sampling)", "#d62728"),
        (axes[1], gpu_cuda, "CUDA sampling", "#2ca02c"),
    ]:
        if data is None:
            ax.text(0.5, 0.5, f"No GPU log for {label}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        t0 = data[0]["time"]
        seconds = [(r["time"] - t0).total_seconds() for r in data]
        gpu_pct = [r["gpu_util"] for r in data]

        ax.fill_between(seconds, gpu_pct, alpha=0.3, color=color)
        ax.plot(seconds, gpu_pct, color=color, linewidth=0.8)
        ax.set_ylabel("GPU Utilization (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{label} — mean GPU util: {np.mean(gpu_pct):.1f}%")
        ax.axhline(y=np.mean(gpu_pct), color=color, linestyle="--",
                    alpha=0.7, linewidth=1)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    path = os.path.join(output_dir, "gpu_utilization.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_epoch_times(curves_baseline, curves_cuda, output_dir):
    """Bar chart of epoch times."""
    plt = _try_import_matplotlib()

    if curves_baseline is None or curves_cuda is None:
        print("  Skipping epoch_times plot (missing data)")
        return

    epochs = [int(r["epoch"]) for r in curves_baseline]
    t_base = [r["epoch_time_sec"] for r in curves_baseline]
    t_cuda = [r["epoch_time_sec"] for r in curves_cuda]

    x = np.arange(len(epochs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, t_base, width, label="Baseline (C++)",
                   color="#d62728", alpha=0.8)
    bars2 = ax.bar(x + width/2, t_cuda, width, label="CUDA",
                   color="#2ca02c", alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Epoch Time: Baseline vs CUDA Sampling")
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    mean_base = np.mean(t_base)
    mean_cuda = np.mean(t_cuda)
    speedup = mean_base / mean_cuda if mean_cuda > 0 else 0
    ax.text(0.98, 0.95,
            f"Mean: {mean_base:.1f}s vs {mean_cuda:.1f}s ({speedup:.1f}x speedup)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "epoch_times.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_training_curves(curves_baseline, curves_cuda, output_dir):
    """Overlaid loss and MRR curves for both experiments."""
    plt = _try_import_matplotlib()

    if curves_baseline is None or curves_cuda is None:
        print("  Skipping training_curves plot (missing data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_b = [r["epoch"] for r in curves_baseline]
    epochs_c = [r["epoch"] for r in curves_cuda]

    ax1.plot(epochs_b, [r["train_loss"] for r in curves_baseline],
             "o-", color="#d62728", label="Baseline", markersize=4)
    ax1.plot(epochs_c, [r["train_loss"] for r in curves_cuda],
             "s--", color="#2ca02c", label="CUDA", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss (should overlap)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_b, [r["val_mrr"] for r in curves_baseline],
             "o-", color="#d62728", label="Baseline", markersize=4)
    ax2.plot(epochs_c, [r["val_mrr"] for r in curves_cuda],
             "s--", color="#2ca02c", label="CUDA", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val MRR")
    ax2.set_title("Validation MRR (should overlap)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot CUDA comparison charts")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--cuda-dir", required=True)
    parser.add_argument("--gpu-log-baseline", default=None)
    parser.add_argument("--gpu-log-cuda", default=None)
    parser.add_argument("--output", default="/tmp/comparison_plots")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print("Loading data...")

    curves_baseline = load_training_curves(args.baseline_dir)
    curves_cuda = load_training_curves(args.cuda_dir)
    gpu_baseline = load_gpu_log(args.gpu_log_baseline)
    gpu_cuda = load_gpu_log(args.gpu_log_cuda)

    print(f"  Baseline epochs: {len(curves_baseline) if curves_baseline else 0}")
    print(f"  CUDA epochs: {len(curves_cuda) if curves_cuda else 0}")
    print(f"  GPU log baseline: {len(gpu_baseline) if gpu_baseline else 0} samples")
    print(f"  GPU log CUDA: {len(gpu_cuda) if gpu_cuda else 0} samples")

    print("\nGenerating plots...")
    plot_gpu_utilization(gpu_baseline, gpu_cuda, args.output)
    plot_epoch_times(curves_baseline, curves_cuda, args.output)
    plot_training_curves(curves_baseline, curves_cuda, args.output)

    res_b = load_final_results(args.baseline_dir)
    res_c = load_final_results(args.cuda_dir)
    if res_b and res_c:
        summary_path = os.path.join(args.output, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("CUDA Sampling Comparison Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{'Metric':<25s} {'Baseline':>12s} {'CUDA':>12s} {'Speedup':>10s}\n")
            f.write("-" * 60 + "\n")
            bt = res_b["timing"]["training_sec"]
            ct = res_c["timing"]["training_sec"]
            f.write(f"{'Training time (sec)':<25s} {bt:>12.1f} {ct:>12.1f} {bt/max(ct,0.1):>9.1f}x\n")
            f.write(f"{'Best val MRR':<25s} {res_b['best_val_mrr']:>12.4f} {res_c['best_val_mrr']:>12.4f}\n")
            f.write(f"{'Test MRR':<25s} {res_b['test_metrics']['mrr']:>12.4f} {res_c['test_metrics']['mrr']:>12.4f}\n")
            f.write(f"{'Test Hits@10':<25s} {res_b['test_metrics']['hits@10']:>12.4f} {res_c['test_metrics']['hits@10']:>12.4f}\n")
        print(f"  Saved: {summary_path}")

    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
