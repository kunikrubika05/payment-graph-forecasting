"""Collect all experiment results from Yandex.Disk and generate a report.

Downloads summary.json from every experiment, parses metrics,
and outputs a single Markdown file suitable for LLM analysis.

Usage:
    export YADISK_TOKEN="..."
    PYTHONPATH=. python scripts/collect_results.py [--output results_report.md]
"""

import argparse
import json
import logging
import os
import sys
import tempfile

from src.yadisk_utils import list_remote_files, download_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENTS_BASE = "orbitaal_processed/experiments"
EXPERIMENT_DIRS = [
    "exp_001_link_pred_baselines",
    "exp_002_graph_level_baselines",
    "exp_003_heuristic_baselines",
]


def collect_summaries(token: str) -> dict:
    """Download all summary.json files from YaDisk experiments."""
    results = {}

    for exp_dir in EXPERIMENT_DIRS:
        remote_path = f"{EXPERIMENTS_BASE}/{exp_dir}"
        sub_experiments = list_remote_files(remote_path, token)

        if not sub_experiments:
            logger.warning("No sub-experiments found in %s", remote_path)
            continue

        results[exp_dir] = {}

        for sub_exp in sub_experiments:
            summary_remote = f"{remote_path}/{sub_exp}/summary.json"
            config_remote = f"{remote_path}/{sub_exp}/config.json"

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                if download_file(summary_remote, tmp_path, token):
                    with open(tmp_path) as f:
                        summary = json.load(f)
                    results[exp_dir][sub_exp] = {"summary": summary}
                    logger.info("  [OK] %s/%s", exp_dir, sub_exp)
                else:
                    logger.warning("  [SKIP] %s/%s (no summary.json)", exp_dir, sub_exp)
                    continue
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                if download_file(config_remote, tmp_path, token):
                    with open(tmp_path) as f:
                        config = json.load(f)
                    results[exp_dir][sub_exp]["config"] = config
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    return results


def format_lp_results(results: dict) -> str:
    """Format link prediction results as Markdown table."""
    lines = [
        "## Link Prediction (exp_001)",
        "",
        "| Period | Window | Agg | Mode | LogReg MRR | CatBoost MRR | RF MRR |",
        "|--------|--------|-----|------|------------|--------------|--------|",
    ]

    for sub_exp, data in sorted(results.items()):
        summary = data.get("summary", {})
        config = data.get("config", {})

        period = config.get("period_name", sub_exp)
        window = config.get("window_size", "?")
        agg = config.get("aggregation", "?")
        mode = config.get("mode", "?")

        models = summary.get("models", {})
        if not models:
            models_alt = summary.get("model_summaries", {})
            if models_alt:
                models = models_alt

        logreg_mrr = _extract_mrr(models, "logreg", summary)
        catboost_mrr = _extract_mrr(models, "catboost", summary)
        rf_mrr = _extract_mrr(models, "rf", summary)

        lines.append(
            f"| {period} | {window} | {agg} | {mode} | {logreg_mrr} | {catboost_mrr} | {rf_mrr} |"
        )

    return "\n".join(lines)


def _extract_mrr(models: dict, model_name: str, summary: dict) -> str:
    """Extract MRR from various summary formats."""
    if model_name in models:
        m = models[model_name]
        if isinstance(m, dict):
            mrr = m.get("mean_mrr", m.get("mrr", m.get("test_mrr")))
            if mrr is not None:
                return f"{float(mrr):.4f}"

    for key in ["ranking_metrics", "test_metrics", "metrics"]:
        if key in summary:
            metrics = summary[key]
            if isinstance(metrics, dict) and model_name in metrics:
                mrr = metrics[model_name].get("mean_mrr", metrics[model_name].get("mrr"))
                if mrr is not None:
                    return f"{float(mrr):.4f}"

    return "—"


def format_heuristic_results(results: dict) -> str:
    """Format heuristic baseline results as Markdown table."""
    lines = [
        "## Heuristic Baselines (exp_003)",
        "",
        "| Period | Window | CN MRR | Jaccard MRR | AA MRR | PA MRR |",
        "|--------|--------|--------|-------------|--------|--------|",
    ]

    for sub_exp, data in sorted(results.items()):
        summary = data.get("summary", {})
        config = data.get("config", {})

        period = config.get("period_name", sub_exp)
        window = config.get("window_size", "?")

        heuristics = summary.get("heuristics", {})
        cn = heuristics.get("common_neighbors", {}).get("mean_mrr", "—")
        jc = heuristics.get("jaccard", {}).get("mean_mrr", "—")
        aa = heuristics.get("adamic_adar", {}).get("mean_mrr", "—")
        pa = heuristics.get("pref_attachment", {}).get("mean_mrr", "—")

        cn = f"{float(cn):.4f}" if cn != "—" else "—"
        jc = f"{float(jc):.4f}" if jc != "—" else "—"
        aa = f"{float(aa):.4f}" if aa != "—" else "—"
        pa = f"{float(pa):.4f}" if pa != "—" else "—"

        lines.append(f"| {period} | {window} | {cn} | {jc} | {aa} | {pa} |")

    return "\n".join(lines)


def format_graph_forecast_results(results: dict) -> str:
    """Format graph-level forecasting results."""
    lines = [
        "## Graph-Level Forecasting (exp_002)",
        "",
    ]

    for sub_exp, data in sorted(results.items()):
        summary = data.get("summary", {})
        config = data.get("config", {})
        period = config.get("period_name", sub_exp)

        lines.append(f"### {period}")
        lines.append("")

        if "models" in summary:
            for target_var, model_results in summary["models"].items():
                lines.append(f"**{target_var}:**")
                if isinstance(model_results, dict):
                    for model_name, metrics in model_results.items():
                        if isinstance(metrics, dict):
                            mae = metrics.get("mae", metrics.get("test_mae", "—"))
                            rmse = metrics.get("rmse", metrics.get("test_rmse", "—"))
                            mape = metrics.get("mape", metrics.get("test_mape", "—"))
                            lines.append(f"- {model_name}: MAE={mae}, RMSE={rmse}, MAPE={mape}")
                lines.append("")
        else:
            lines.append(f"```json\n{json.dumps(summary, indent=2, default=str)[:2000]}\n```")
            lines.append("")

    return "\n".join(lines)


def generate_report(all_results: dict) -> str:
    """Generate full Markdown report."""
    sections = [
        "# Baseline Experiment Results — ORBITAAL Bitcoin Graph",
        "",
        f"Total experiments: {sum(len(v) for v in all_results.values())}",
        "",
        "---",
        "",
    ]

    exp_001 = all_results.get("exp_001_link_pred_baselines", {})
    if exp_001:
        sections.append(format_lp_results(exp_001))
        sections.append("")

    exp_003 = all_results.get("exp_003_heuristic_baselines", {})
    if exp_003:
        sections.append(format_heuristic_results(exp_003))
        sections.append("")

    exp_002 = all_results.get("exp_002_graph_level_baselines", {})
    if exp_002:
        sections.append(format_graph_forecast_results(exp_002))
        sections.append("")

    sections.extend([
        "---",
        "",
        "## Raw JSON (for programmatic analysis)",
        "",
        "```json",
        json.dumps(all_results, indent=2, default=str),
        "```",
    ])

    return "\n".join(sections)


def main():
    """Collect results and generate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="results_report.md")
    args = parser.parse_args()

    token = os.environ.get("YADISK_TOKEN")
    if not token:
        logger.error("YADISK_TOKEN not set")
        sys.exit(1)

    logger.info("Collecting experiment results from Yandex.Disk...")
    all_results = collect_summaries(token)

    total = sum(len(v) for v in all_results.values())
    logger.info("Collected %d experiment results", total)

    report = generate_report(all_results)

    with open(args.output, "w") as f:
        f.write(report)

    logger.info("Report saved to %s (%d chars)", args.output, len(report))


if __name__ == "__main__":
    main()
