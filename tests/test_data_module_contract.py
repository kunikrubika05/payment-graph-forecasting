from __future__ import annotations

import importlib
from pathlib import Path


def test_glformer_cuda_eval_uses_neutral_stream_graph_helpers():
    module = importlib.import_module("src.models.GLFormer_cuda.glformer_evaluate")
    neutral = importlib.import_module("src.models.stream_graph_data")

    assert module.build_temporal_csr is neutral.build_temporal_csr
    assert module.generate_negatives_for_eval is neutral.generate_negatives_for_eval


def test_graphmixer_legacy_sidecars_use_neutral_stream_graph_loader():
    neutral = importlib.import_module("src.models.stream_graph_data")
    launcher = importlib.import_module("src.models.cuda_exp_graphmixer_a10.launcher")
    assert launcher.load_stream_graph_data is neutral.load_stream_graph_data

    hpo_source = Path("src/models/GraphMixer/graphmixer_hpo.py").read_text(encoding="utf-8")
    assert "from src.models.stream_graph_data import load_stream_graph_data" in hpo_source
