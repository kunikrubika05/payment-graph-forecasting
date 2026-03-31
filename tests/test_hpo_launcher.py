from __future__ import annotations

import pytest

from payment_graph_forecasting.experiments import hpo


def test_get_hpo_entrypoint_rejects_unknown_model():
    with pytest.raises(KeyError):
        hpo.get_hpo_entrypoint("unknown")


def test_run_hpo_dispatches_to_registered_entrypoint(monkeypatch):
    observed = {}

    def _fake_main():
        import sys

        observed["argv"] = sys.argv[1:]

    monkeypatch.setitem(hpo.HPO_REGISTRY, "graphmixer", _fake_main)

    assert hpo.run_hpo("graphmixer", ["--parquet-path", "/tmp/stream.parquet"]) == 0
    assert observed["argv"] == ["--parquet-path", "/tmp/stream.parquet"]


def test_main_dispatches_model_and_passthrough_args(monkeypatch):
    observed = {}

    def _fake_run_hpo(model_name: str, argv: list[str] | None = None) -> int:
        observed["model"] = model_name
        observed["argv"] = argv
        return 0

    monkeypatch.setattr(hpo, "run_hpo", _fake_run_hpo)

    assert hpo.main(["eagle", "--parquet-path", "/tmp/stream.parquet", "--n-trials", "2"]) == 0
    assert observed == {
        "model": "eagle",
        "argv": ["--parquet-path", "/tmp/stream.parquet", "--n-trials", "2"],
    }
