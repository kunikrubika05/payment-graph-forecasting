"""Compatibility tests for old import paths kept during the refactor."""

from __future__ import annotations

from payment_graph_forecasting.experiments.runners.glformer import (
    build_glformer_arg_parser as build_glformer_package_parser,
)
from payment_graph_forecasting.experiments.runners.hyperevent import (
    build_hyperevent_arg_parser as build_hyperevent_package_parser,
)
from payment_graph_forecasting.experiments.runners.sg_graphmixer import (
    build_sg_graphmixer_arg_parser as build_sg_graphmixer_package_parser,
)
from src.models.eagle import EAGLETime
from src.models.eagle_train import prepare_eagle_batch
from src.models.GLFormer_cuda import glformer_launcher as legacy_glformer_cuda_launcher
from src.models.HyperEvent import hyperevent_launcher as legacy_hyperevent_launcher
from src.models.sg_graphmixer import launcher as legacy_sg_graphmixer_launcher
from src.models.tppr import TPPR, get_forward_edge_mask


def test_legacy_eagle_import_path_resolves():
    assert EAGLETime is not None


def test_legacy_tppr_import_path_resolves():
    assert TPPR is not None
    assert callable(get_forward_edge_mask)


def test_legacy_eagle_train_import_path_resolves():
    assert callable(prepare_eagle_batch)


def test_legacy_hyperevent_launcher_is_package_wrapper():
    assert legacy_hyperevent_launcher.build_hyperevent_arg_parser is build_hyperevent_package_parser
    assert callable(legacy_hyperevent_launcher.main)
    assert callable(legacy_hyperevent_launcher.run_experiment)


def test_legacy_sg_graphmixer_launcher_is_package_wrapper():
    assert legacy_sg_graphmixer_launcher.build_sg_graphmixer_arg_parser is build_sg_graphmixer_package_parser
    assert callable(legacy_sg_graphmixer_launcher.main)
    assert callable(legacy_sg_graphmixer_launcher.run_experiment)


def test_legacy_glformer_cuda_launcher_is_package_wrapper():
    assert legacy_glformer_cuda_launcher.build_glformer_arg_parser is build_glformer_package_parser
    assert callable(legacy_glformer_cuda_launcher.main)
    assert callable(legacy_glformer_cuda_launcher.run_experiment)
