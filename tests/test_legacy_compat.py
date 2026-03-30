"""Compatibility tests for old import paths kept during the refactor."""

from __future__ import annotations

from src.models.eagle import EAGLETime
from src.models.eagle_train import prepare_eagle_batch
from src.models.tppr import TPPR, get_forward_edge_mask


def test_legacy_eagle_import_path_resolves():
    assert EAGLETime is not None


def test_legacy_tppr_import_path_resolves():
    assert TPPR is not None
    assert callable(get_forward_edge_mask)


def test_legacy_eagle_train_import_path_resolves():
    assert callable(prepare_eagle_batch)
