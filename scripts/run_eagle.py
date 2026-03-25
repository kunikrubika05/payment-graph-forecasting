"""Full EAGLE pipeline: tests -> sanity check -> HPO -> training -> TPPR.

Usage:
    YADISK_TOKEN="..." PYTHONPATH=. python scripts/run_eagle.py
    PYTHONPATH=. python scripts/run_eagle.py --skip-hpo --skip-tppr
    PYTHONPATH=. python scripts/run_eagle.py --period early_2012q4 --window 14
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def elapsed_str(start: float) -> str:
    """Format elapsed time as 'Xm Ys'."""
    diff = time.time() - start
    return f"{int(diff // 60)}m {int(diff % 60)}s"


def run_cmd(cmd: list[str], description: str) -> int:
    """Run a subprocess command with real-time output."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=os.environ.get("PROJECT_DIR", "."))
    if result.returncode != 0:
        logger.error("ОШИБКА на шаге: %s (exit code %d)", description, result.returncode)
    return result.returncode


def detect_device() -> str:
    """Detect CUDA/CPU device."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info("GPU: %s (%.1f GB)", name, mem)
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


def step_header(step_num: int, name: str):
    """Print a step header."""
    logger.info("=" * 60)
    logger.info("  ШАГ %d: %s", step_num, name)
    logger.info("=" * 60)


def load_hpo_params(hpo_dir: str) -> str:
    """Load best params from HPO results as CLI arguments string."""
    results_path = os.path.join(hpo_dir, "hpo_results.json")
    if not os.path.exists(results_path):
        return ""

    with open(results_path) as f:
        results = json.load(f)

    params = results.get("best_params", {})
    args = []
    for key, value in params.items():
        flag = key.replace("_", "-")
        args.extend([f"--{flag}", str(value)])

    logger.info("HPO лучший MRR: %.4f", results.get("best_mrr", 0))
    logger.info("HPO params: %s", params)
    return args


def main():
    parser = argparse.ArgumentParser(
        description="EAGLE: полный пайплайн обучения и проверки"
    )
    parser.add_argument("--period", type=str, default="mature_2020q2")
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hpo-trials", type=int, default=30)
    parser.add_argument("--hpo-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--skip-tppr", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--output", type=str, default="/tmp/eagle_results",
    )
    parser.add_argument(
        "--hpo-output", type=str, default="/tmp/eagle_hpo",
    )
    parser.add_argument(
        "--tppr-output", type=str, default="/tmp/eagle_tppr",
    )
    parser.add_argument(
        "--data-dir", type=str, default="/tmp/eagle_data",
    )

    args = parser.parse_args()

    script_start = time.time()
    amp_flag = ["--no-amp"] if args.no_amp else []
    python = sys.executable

    log_dir = "/tmp/eagle_logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"eagle_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("  EAGLE — полный пайплайн")
    logger.info("=" * 60)
    logger.info("  Период:     %s", args.period)
    logger.info("  Окно:       %d", args.window)
    logger.info("  Эпохи:      %d", args.epochs)
    logger.info("  HPO trials: %d", args.hpo_trials)
    logger.info("  HPO эпохи:  %d", args.hpo_epochs)
    logger.info("  Seed:       %d", args.seed)
    logger.info("  AMP:        %s", "OFF" if args.no_amp else "ON")
    logger.info("  Лог:        %s", log_file)
    logger.info("=" * 60)

    if not os.environ.get("YADISK_TOKEN"):
        logger.warning(
            "YADISK_TOKEN не задан — загрузка на Яндекс.Диск будет пропущена"
        )

    device = detect_device()
    logger.info("Device: %s", device)

    step = 0

    if not args.skip_tests:
        step += 1
        step_header(step, "Тесты")
        step_start = time.time()

        rc = run_cmd(
            [python, "-u", "-m", "pytest", "tests/test_eagle.py", "-v", "--tb=short"],
            "pytest tests/test_eagle.py",
        )
        if rc != 0:
            sys.exit(rc)

        logger.info("Тесты пройдены за %s", elapsed_str(step_start))
    else:
        logger.info("[SKIP] Тесты пропущены (--skip-tests)")

    if not args.skip_sanity:
        step += 1
        step_header(step, "Sanity check")
        step_start = time.time()

        rc = run_cmd(
            [python, "-u", "scripts/eagle_sanity_check.py"],
            "eagle_sanity_check.py",
        )
        if rc != 0:
            sys.exit(rc)

        logger.info("Sanity check пройден за %s", elapsed_str(step_start))
    else:
        logger.info("[SKIP] Sanity check пропущен (--skip-sanity)")

    if not args.skip_hpo:
        step += 1
        step_header(
            step,
            f"HPO (Optuna, {args.hpo_trials} trials × {args.hpo_epochs} epochs)",
        )
        step_start = time.time()

        rc = run_cmd(
            [
                python, "-u", "src/models/eagle_hpo.py",
                "--period", args.period,
                "--window", str(args.window),
                "--n-trials", str(args.hpo_trials),
                "--hpo-epochs", str(args.hpo_epochs),
                "--max-val-edges", "3000",
                "--seed", str(args.seed),
                "--output", args.hpo_output,
                "--data-dir", args.data_dir,
                *amp_flag,
            ],
            "eagle_hpo.py",
        )
        if rc != 0:
            sys.exit(rc)

        logger.info("HPO завершён за %s", elapsed_str(step_start))
    else:
        logger.info("[SKIP] HPO пропущен (--skip-hpo)")

    step += 1
    step_header(step, f"Обучение EAGLE-Time ({args.epochs} эпох)")
    step_start = time.time()

    train_cmd = [
        python, "-u", "src/models/eagle_launcher.py",
        "--period", args.period,
        "--window", str(args.window),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--output", args.output,
        "--data-dir", args.data_dir,
        *amp_flag,
    ]

    if not args.skip_hpo:
        hpo_params = load_hpo_params(args.hpo_output)
        if hpo_params:
            logger.info("Используются лучшие параметры из HPO")
            train_cmd.extend(hpo_params)
        else:
            logger.info("HPO результаты не найдены, используются параметры по умолчанию")
    else:
        logger.info("Используются параметры по умолчанию")

    rc = run_cmd(train_cmd, "eagle_launcher.py")
    if rc != 0:
        sys.exit(rc)

    logger.info("Обучение завершено за %s", elapsed_str(step_start))

    exp_name = f"eagle_{args.period}_w{args.window}"
    exp_dir = os.path.join(args.output, exp_name)
    results_path = os.path.join(exp_dir, "final_results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        m = results["test_metrics"]
        logger.info("  Test MRR:     %.4f", m["mrr"])
        logger.info("  Test Hits@1:  %.4f", m["hits@1"])
        logger.info("  Test Hits@3:  %.4f", m["hits@3"])
        logger.info("  Test Hits@10: %.4f", m["hits@10"])
        logger.info("  Best epoch:   %d", results["best_epoch"])
        logger.info(
            "  Total time:   %.1f min", results["timing"]["total_sec"] / 60
        )

    if not args.skip_tppr:
        step += 1
        step_header(step, "TPPR структурный бейзлайн")
        step_start = time.time()

        rc = run_cmd(
            [
                python, "-u", "src/models/tppr.py",
                "--period", args.period,
                "--window", str(args.window),
                "--topk", "100",
                "--alpha", "0.9",
                "--beta", "0.8",
                "--output", args.tppr_output,
                "--data-dir", args.data_dir,
            ],
            "tppr.py",
        )
        if rc != 0:
            logger.error("TPPR failed, continuing...")

        logger.info("TPPR завершён за %s", elapsed_str(step_start))
    else:
        logger.info("[SKIP] TPPR пропущен (--skip-tppr)")

    step += 1
    step_header(step, "Итоги")

    total_elapsed = elapsed_str(script_start)
    logger.info("Полное время: %s", total_elapsed)

    eagle_mrr = "N/A"
    if os.path.exists(results_path):
        with open(results_path) as f:
            eagle_mrr = f"{json.load(f)['test_metrics']['mrr']:.4f}"

    tppr_mrr = "N/A"
    tppr_exp = (
        f"tppr_{args.period}_w{args.window}_k100_a0.9_b0.8"
    )
    tppr_results_path = os.path.join(
        args.tppr_output, tppr_exp, "tppr_results.json"
    )
    if os.path.exists(tppr_results_path):
        with open(tppr_results_path) as f:
            tppr_mrr = f"{json.load(f)['test_metrics']['mrr']:.4f}"

    logger.info("")
    logger.info("Результаты:")
    logger.info("  EAGLE обучение:  %s/", exp_dir)
    if not args.skip_hpo:
        logger.info("  HPO результаты:  %s/", args.hpo_output)
    if not args.skip_tppr:
        logger.info("  TPPR результаты: %s/", args.tppr_output)
    logger.info("  Лог:             %s", log_file)
    logger.info("")
    logger.info("  ┌──────────────────┬──────────┐")
    logger.info("  │ Модель           │ Test MRR │")
    logger.info("  ├──────────────────┼──────────┤")
    logger.info("  │ EAGLE-Time       │ %-8s │", eagle_mrr)
    logger.info("  │ TPPR (structure) │ %-8s │", tppr_mrr)
    logger.info("  │ CN (baseline)    │ 0.46-0.73│")
    logger.info("  └──────────────────┴──────────┘")
    logger.info("")
    logger.info("EAGLE пайплайн завершён.")


if __name__ == "__main__":
    main()
