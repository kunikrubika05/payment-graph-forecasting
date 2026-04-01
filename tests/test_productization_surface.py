from __future__ import annotations

import importlib

import numpy as np
from scipy import sparse

import payment_graph_forecasting as pgf
from payment_graph_forecasting.cuda import (
    CudaCapabilities,
    describe_cuda_capabilities,
    has_temporal_sampling_cpp,
    has_temporal_sampling_cuda,
)
from payment_graph_forecasting.graph_metrics import CommonNeighbors, has_cpp, has_cuda
from payment_graph_forecasting.sampling import TemporalGraphSampler, resolve_backend
from payment_graph_forecasting.sampling import has_cpp as has_temporal_sampling_cpp_backend
from payment_graph_forecasting.sampling import has_cuda as has_temporal_sampling_cuda_backend


def test_top_level_package_exports_productized_cuda_surfaces():
    assert pgf.TemporalGraphSampler is TemporalGraphSampler
    assert pgf.CommonNeighbors is CommonNeighbors
    assert pgf.describe_cuda_capabilities is describe_cuda_capabilities


def test_sampling_package_exports_temporal_runtime_helpers():
    assert resolve_backend("python").value == "python"
    assert resolve_backend("auto").value in {"python", "cpp", "cuda"}
    assert isinstance(has_temporal_sampling_cpp_backend(), bool)
    assert isinstance(has_temporal_sampling_cuda_backend(), bool)


def test_temporal_graph_sampler_package_wrapper_supports_python_backend():
    sampler = TemporalGraphSampler(
        num_nodes=4,
        src=np.array([0, 0, 1], dtype=np.int32),
        dst=np.array([1, 2, 2], dtype=np.int32),
        timestamps=np.array([1.0, 3.0, 2.0], dtype=np.float64),
        edge_ids=np.array([0, 1, 2], dtype=np.int64),
        backend="python",
    )

    batch = sampler.sample_neighbors(
        np.array([0, 1], dtype=np.int32),
        np.array([4.0, 3.0], dtype=np.float64),
        num_neighbors=2,
    )

    assert batch.on_gpu is False
    assert batch.lengths.tolist() == [2, 1]
    assert batch.neighbor_ids.tolist() == [[1, 2], [2, -1]]


def test_common_neighbors_package_wrapper_supports_python_backend():
    adj = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    cn = CommonNeighbors(adj, backend="python")

    counts = cn.compute(np.array([0, 0], dtype=np.int64), np.array([1, 3], dtype=np.int64))

    assert counts.tolist() == [1, 1]
    assert cn.backend == "python"


def test_productized_cuda_availability_helpers_return_bools():
    assert isinstance(has_cpp(), bool)
    assert isinstance(has_cuda(), bool)
    assert isinstance(has_temporal_sampling_cpp(), bool)
    assert isinstance(has_temporal_sampling_cuda(), bool)


def test_describe_cuda_capabilities_returns_package_contract():
    capabilities = describe_cuda_capabilities()

    assert isinstance(capabilities, CudaCapabilities)
    assert isinstance(capabilities.cuda_runtime_available, bool)
    assert isinstance(capabilities.temporal_sampling_cpp_available, bool)
    assert isinstance(capabilities.temporal_sampling_cuda_available, bool)
    assert isinstance(capabilities.graph_metrics_cpp_available, bool)
    assert isinstance(capabilities.graph_metrics_cuda_available, bool)


def test_model_specific_data_utils_point_to_neutral_stream_graph_bridge():
    neutral = importlib.import_module("src.models.stream_graph_data")
    for module_name in (
        "src.models.GLFormer.data_utils",
        "src.models.GraphMixer.data_utils",
        "src.models.DyGFormer.data_utils",
        "src.models.HyperEvent.data_utils",
    ):
        module = importlib.import_module(module_name)
        assert module.load_stream_graph_data is neutral.load_stream_graph_data
        assert module.build_temporal_csr is neutral.build_temporal_csr
