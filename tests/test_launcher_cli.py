from payment_graph_forecasting.experiments import launcher


def test_launcher_main_accepts_config_flag(monkeypatch, tmp_path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: smoke",
                "  model: graphmixer",
                "data:",
                "  source: sliding_window",
                "training:",
                "  epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    observed = {}

    def _fake_launch(spec):
        observed["name"] = spec.experiment.name
        observed["model"] = spec.model_name
        observed["dry_run"] = spec.runtime.dry_run
        return {"ok": True}

    monkeypatch.setattr(launcher, "launch_experiment", _fake_launch)

    exit_code = launcher.main(["--config", str(spec_path), "--dry-run"])

    assert exit_code == 0
    assert observed == {"name": "smoke", "model": "graphmixer", "dry_run": True}
