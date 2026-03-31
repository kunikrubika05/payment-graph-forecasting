from __future__ import annotations

import importlib

from payment_graph_forecasting.infra import extensions


def test_selected_build_specs_defaults_to_temporal_cpp():
    parser = extensions.build_arg_parser()
    args = parser.parse_args([])

    selected = extensions.selected_build_specs(args)

    assert [spec.name for spec in selected] == ["temporal_sampling_cpp"]


def test_selected_build_specs_all_and_graph_metrics():
    parser = extensions.build_arg_parser()
    args = parser.parse_args(["--all", "--graph-metrics", "--graph-metrics-cuda"])

    selected = extensions.selected_build_specs(args)

    assert [spec.name for spec in selected] == [
        "temporal_sampling_cpp",
        "temporal_sampling_cuda",
        "graph_metrics_cpp",
        "graph_metrics_cuda",
    ]


def test_run_selected_builds_dispatches_each_requested_spec(monkeypatch, tmp_path):
    parser = extensions.build_arg_parser()
    args = parser.parse_args(["--cuda", "--graph-metrics"])
    observed: list[str] = []

    def _fake_build_extension(spec, *, models_dir):
        observed.append(f"{spec.name}@{models_dir}")
        return spec.name

    monkeypatch.setattr(extensions, "build_extension", _fake_build_extension)

    results = extensions.run_selected_builds(args, models_dir=tmp_path)

    assert results == ["temporal_sampling_cuda", "graph_metrics_cpp"]
    assert observed == [
        f"temporal_sampling_cuda@{tmp_path}",
        f"graph_metrics_cpp@{tmp_path}",
    ]


def test_build_extension_requires_ninja(monkeypatch, tmp_path):
    monkeypatch.setattr(extensions.shutil, "which", lambda name: None)

    try:
        extensions.build_extension(extensions.GRAPH_METRICS_CPP, models_dir=tmp_path)
    except RuntimeError as exc:
        assert "ninja" in str(exc)
        assert "graph_metrics_cpp" in str(exc)
    else:
        raise AssertionError("Expected missing-ninja prerequisite error")


def test_extension_main_returns_error_code_for_missing_prerequisites(monkeypatch):
    monkeypatch.setattr(extensions.shutil, "which", lambda name: None)

    result = extensions.main(["--graph-metrics"])

    assert result == 1


def test_legacy_build_ext_shim_exports_package_builders():
    shim = importlib.import_module("src.models.build_ext")

    assert shim.build_cpp is extensions.build_temporal_sampling_cpp
    assert shim.build_cuda is extensions.build_temporal_sampling_cuda
    assert shim.build_graph_metrics_cpp is extensions.build_graph_metrics_cpp
    assert shim.build_graph_metrics_cuda is extensions.build_graph_metrics_cuda
