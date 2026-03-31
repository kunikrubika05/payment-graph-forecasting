from payment_graph_forecasting.experiments.results import (
    build_dry_run_result,
    build_final_results,
    history_best_epoch,
    history_best_val_mrr,
)


def test_history_best_helpers_handle_empty_history():
    assert history_best_epoch({"val_mrr": []}) is None
    assert history_best_val_mrr({"val_mrr": []}) is None


def test_history_best_helpers_extract_best_metrics():
    history = {"val_mrr": [0.1, 0.3, 0.25], "train_loss": [1.0, 0.8, 0.7]}

    assert history_best_epoch(history) == 2
    assert history_best_val_mrr(history) == 0.3


def test_build_dry_run_result_standardizes_payload():
    result = build_dry_run_result(
        experiment="exp1",
        output_dir="/tmp/out",
        parquet_path="/tmp/data.parquet",
    )

    assert result == {
        "mode": "dry_run",
        "experiment": "exp1",
        "output_dir": "/tmp/out",
        "parquet_path": "/tmp/data.parquet",
    }


def test_build_final_results_standardizes_summary():
    history = {"val_mrr": [0.2, 0.4], "train_loss": [1.0, 0.5]}
    result = build_final_results(
        experiment="exp2",
        model="GraphMixer",
        history=history,
        timing={"total_sec": 3.5},
        args={"seed": 42},
        device_info={"device": "cpu"},
        extra={"test_metrics": {"mrr": 0.35}},
    )

    assert result["experiment"] == "exp2"
    assert result["model"] == "GraphMixer"
    assert result["best_epoch"] == 2
    assert result["best_val_mrr"] == 0.4
    assert result["total_epochs"] == 2
    assert result["test_metrics"] == {"mrr": 0.35}
