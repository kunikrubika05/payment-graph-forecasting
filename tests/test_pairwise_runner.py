from argparse import Namespace
from pathlib import Path

import src.models.pairwise_mlp.run as pairwise_run

from payment_graph_forecasting.experiments.runners.pairwise_mlp import run_pairwise_mlp_experiment
from payment_graph_forecasting.models.pairwise_mlp import PairwiseMLPAdapter
from src.models.pairwise_mlp.config import PairMLPConfig


def test_pairwise_runner_dry_run(tmp_path):
    result = run_pairwise_mlp_experiment(
        Namespace(
            period="period_10",
            output=str(tmp_path),
            data_dir="/tmp/pairmlp_data",
            precompute_dir="/tmp/pairmlp_precompute",
            precompute_remote_dir=None,
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
            upload_backend="yadisk",
            remote_dir=None,
            token_env="YADISK_TOKEN",
            dry_run=True,
            no_amp=False,
        )
    )

    assert result["mode"] == "dry_run"
    assert result["n_negatives"] == 30
    assert result["selected_features"] == ["cn_uu", "aa_uu"]
    assert result["precompute_remote_dir"] is None


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
        model={"features": ["cn_uu"], "loss": "bce", "precompute_remote_dir": "remote/precompute"},
    )

    payload = PairwiseMLPAdapter().build_runner_kwargs(spec)

    assert payload["period"] == "period_10"
    assert payload["n_negatives"] == 50
    assert payload["features"] == ["cn_uu"]
    assert payload["loss"] == "bce"
    assert payload["precompute_remote_dir"] == "remote/precompute"
    assert payload["upload"] is True


def test_pairwise_final_checkpoint_load_uses_weights_only(monkeypatch, tmp_path):
    cfg = PairMLPConfig(
        local_output_dir=str(tmp_path / "out"),
        local_precompute_dir=str(tmp_path / "pre"),
        local_data_dir=str(tmp_path / "data"),
        n_epochs=1,
        patience=1,
        eval_every=1,
    )

    (tmp_path / "pre" / "pairmlp_precompute_10").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    meta_path = Path(cfg.precompute_artifact_dir) / "meta.json"
    meta_path.write_text('{"correctness_checks": {"ok": true}, "n_train": 1}')

    class DummyModel:
        def to(self, device):
            return self

        def parameters(self):
            return []

        def load_state_dict(self, state):
            self.state = state

    monkeypatch.setattr(pairwise_run, "ensure_precomputed", lambda *args, **kwargs: None)
    monkeypatch.setattr(pairwise_run, "load_stream_graph", lambda *args, **kwargs: object())
    monkeypatch.setattr(pairwise_run, "split_stream_graph", lambda *args, **kwargs: ([1], [2], [3]))
    monkeypatch.setattr(pairwise_run, "load_adjacency", lambda *args, **kwargs: ([0], "adj_dir", "adj_undir"))
    monkeypatch.setattr(pairwise_run, "load_node_features_sparse", lambda *args, **kwargs: ([0], object()))
    monkeypatch.setattr(pairwise_run, "build_train_neighbor_sets", lambda *args, **kwargs: {})
    monkeypatch.setattr(pairwise_run, "precompute_degrees", lambda *args, **kwargs: ("deg_u", "deg_d"))
    monkeypatch.setattr(pairwise_run, "precompute_aa_weights", lambda *args, **kwargs: ("w_u", "w_d"))
    monkeypatch.setattr(pairwise_run, "load_dataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(pairwise_run, "build_model", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(pairwise_run, "build_eval_cache", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        pairwise_run,
        "evaluate_split",
        lambda *args, **kwargs: {"mrr": 0.5, "hits@1": 0.4, "hits@10": 0.8},
    )
    monkeypatch.setattr(
        pairwise_run,
        "train",
        lambda **kwargs: {
            "ckpt_path": str(tmp_path / "best.pt"),
            "best_val_mrr": 0.5,
            "train_loss": [0.1],
            "val_mrr_epoch": [1],
            "val_mrr": [0.5],
        },
    )

    captured = {}

    def fake_torch_load(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return {"model_state": {"w": 1}, "epoch": 1, "val_mrr": 0.5}

    monkeypatch.setattr(pairwise_run.torch, "load", fake_torch_load)

    pairwise_run.run_experiment(cfg, token="")

    assert captured["path"] == str(tmp_path / "best.pt")
    assert captured["weights_only"] is True
