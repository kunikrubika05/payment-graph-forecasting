from argparse import Namespace

from payment_graph_forecasting.experiments.runners.pairwise_mlp import run_pairwise_mlp_experiment
from payment_graph_forecasting.models.pairwise_mlp import PairwiseMLPAdapter


def test_pairwise_runner_dry_run(tmp_path):
    result = run_pairwise_mlp_experiment(
        Namespace(
            period="period_10",
            output=str(tmp_path),
            data_dir="/tmp/pairmlp_data",
            precompute_dir="/tmp/pairmlp_precompute",
            exp_tag="smoke",
            loss="bpr",
            features=["cn_uu", "aa_uu"],
            feature_indices=None,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            weight_decay=1e-4,
            dropout=0.1,
            patience=2,
            seed=42,
            n_negatives=30,
            upload=False,
            dry_run=True,
            no_amp=False,
        )
    )

    assert result["mode"] == "dry_run"
    assert result["n_negatives"] == 30
    assert result["selected_features"] == ["cn_uu", "aa_uu"]


def test_pairwise_adapter_builds_runner_kwargs():
    from payment_graph_forecasting.config.base import (
        DataConfig,
        ExperimentMetadata,
        ExperimentSpec,
        RuntimeConfig,
        SamplingConfig,
        TrainingConfig,
        UploadConfig,
    )

    spec = ExperimentSpec(
        experiment=ExperimentMetadata(name="pairwise_smoke", model="pairwise_mlp"),
        data=DataConfig(period="period_10", data_dir="/tmp/data"),
        sampling=SamplingConfig(n_random_neg=35, n_hist_neg=15),
        training=TrainingConfig(epochs=5, batch_size=32, lr=1e-3, patience=3, seed=9),
        runtime=RuntimeConfig(dry_run=True),
        upload=UploadConfig(enabled=True),
        model={"features": ["cn_uu"], "loss": "bce"},
    )

    payload = PairwiseMLPAdapter().build_runner_kwargs(spec)

    assert payload["period"] == "period_10"
    assert payload["n_negatives"] == 50
    assert payload["features"] == ["cn_uu"]
    assert payload["loss"] == "bce"
    assert payload["upload"] is True
