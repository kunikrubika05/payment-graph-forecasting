"""Unified temporal graph sampler with three backends: Python, C++, CUDA.

Provides a single API for temporal neighbor sampling and feature gathering,
automatically selecting the best available backend or allowing explicit control.

Usage:
    sampler = TemporalGraphSampler(num_nodes, src, dst, timestamps, edge_ids,
                                    node_feats=nf, edge_feats=ef,
                                    backend='auto')

    # Temporal neighbor lookup
    nbr = sampler.sample_neighbors(query_nodes, query_times, num_neighbors=20)

    # Feature gathering
    feat = sampler.featurize(nbr)

    # Negative sampling (for evaluation)
    neg = sampler.sample_negatives(src_batch, true_dst_batch, query_times,
                                    n_negatives=100, strategy='mixed')

Backends:
    'python' — pure NumPy, always available, slowest
    'cpp'    — C++ pybind11 extension (~3-5x vs Python)
    'cuda'   — CUDA extension (~10-50x vs C++ for large batches)
    'auto'   — CUDA if available, else C++, else Python
"""

import logging
import importlib.util
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Backend(Enum):
    PYTHON = "python"
    CPP = "cpp"
    CUDA = "cuda"
    AUTO = "auto"


@dataclass
class NeighborBatch:
    """Result of temporal neighbor sampling.

    Attributes:
        neighbor_ids: [B, K] int32, padded with -1.
        timestamps: [B, K] float64, padded with 0.
        edge_ids: [B, K] int64, padded with -1.
        lengths: [B] int32, actual neighbor count per query.
        on_gpu: Whether data resides on GPU (torch tensors) or CPU (numpy).
    """
    neighbor_ids: object
    timestamps: object
    edge_ids: object
    lengths: object
    on_gpu: bool = False


@dataclass
class FeatureBatch:
    """Result of neighbor featurization.

    Attributes:
        node_features: [B, K, node_feat_dim] float32.
        edge_features: [B, K, edge_feat_dim] float32.
        rel_timestamps: [B, K] float64, relative (query_t - neighbor_t).
        on_gpu: Whether data resides on GPU (torch tensors) or CPU (numpy).
    """
    node_features: object
    edge_features: object
    rel_timestamps: object
    on_gpu: bool = False


def _try_load_cpp():
    """Try to load the C++ temporal sampling extension."""
    ext_name = "temporal_sampling_cpp"
    try:
        from torch.utils.cpp_extension import load as _load_ext
        cpp_path = str(Path(__file__).parent / "csrc" / "temporal_sampling.cpp")
        build_dir = str(Path(__file__).parent / "csrc" / "build")
        if Path(cpp_path).exists():
            ext = _load_ext(
                name=ext_name,
                sources=[cpp_path],
                build_directory=build_dir,
                extra_cflags=["-O3"],
                verbose=False,
            )
            return ext
    except Exception as e:
        logger.debug("C++ extension unavailable: %s", e)
    return _load_prebuilt_extension(ext_name, Path(__file__).parent / "csrc" / "build")


def _load_prebuilt_extension(module_name: str, build_dir: Path):
    """Load an already-built extension from the local build directory."""
    if not build_dir.exists():
        return None
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    candidates = sorted(
        path for path in build_dir.iterdir()
        if path.name.startswith(module_name) and path.name.endswith(suffixes)
    )
    for candidate in reversed(candidates):
        try:
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.debug("Prebuilt extension load failed for %s: %s", candidate, e)
    return None


def _try_load_cuda():
    """Try to load the CUDA temporal sampling extension."""
    ext_name = "temporal_sampling_cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        from torch.utils.cpp_extension import load as _load_ext
        cu_path = str(Path(__file__).parent / "csrc" / "temporal_sampling.cu")
        build_dir = str(Path(__file__).parent / "csrc" / "build_cuda")
        if Path(cu_path).exists():
            ext = _load_ext(
                name=ext_name,
                sources=[cu_path],
                build_directory=build_dir,
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
            return ext
    except Exception as e:
        logger.debug("CUDA extension unavailable: %s", e)
    return _load_prebuilt_extension(ext_name, Path(__file__).parent / "csrc" / "build_cuda")


_ext_cache = {}


def _get_extension(backend: Backend):
    """Get the requested extension, with caching."""
    if backend.value in _ext_cache:
        return _ext_cache[backend.value]

    ext = None
    if backend == Backend.CUDA:
        ext = _try_load_cuda()
    elif backend == Backend.CPP:
        ext = _try_load_cpp()
    elif backend == Backend.AUTO:
        ext = _try_load_cuda()
        if ext is not None:
            _ext_cache[Backend.CUDA.value] = ext
            return ext
        ext = _try_load_cpp()
        if ext is not None:
            _ext_cache[Backend.CPP.value] = ext
            return ext

    _ext_cache[backend.value] = ext
    return ext


def resolve_backend(requested: str) -> Backend:
    """Resolve 'auto' to the best available backend."""
    backend = Backend(requested)
    if backend != Backend.AUTO:
        return backend

    if _try_load_cuda() is not None:
        return Backend.CUDA
    if _try_load_cpp() is not None:
        return Backend.CPP
    return Backend.PYTHON


def has_cpp() -> bool:
    """Return whether the temporal sampling C++ backend is available."""

    return _try_load_cpp() is not None


def has_cuda() -> bool:
    """Return whether the temporal sampling CUDA backend is available."""

    return _try_load_cuda() is not None


class TemporalGraphSampler:
    """Unified temporal graph sampler supporting Python, C++, and CUDA backends.

    The CSR structure is built once at construction time. For CUDA backend,
    the entire CSR is transferred to GPU memory and stays there.

    Args:
        num_nodes: Total number of nodes in graph.
        src: Source node indices, int32 or int64.
        dst: Destination node indices, int32 or int64.
        timestamps: Edge timestamps, float64.
        edge_ids: Edge indices, int64.
        node_feats: Optional [num_nodes, feat_dim] float32 array.
        edge_feats: Optional [num_edges, feat_dim] float32 array.
        backend: 'auto', 'python', 'cpp', or 'cuda'.
    """

    def __init__(
        self,
        num_nodes: int,
        src: np.ndarray,
        dst: np.ndarray,
        timestamps: np.ndarray,
        edge_ids: np.ndarray,
        node_feats: Optional[np.ndarray] = None,
        edge_feats: Optional[np.ndarray] = None,
        backend: str = "auto",
    ):
        self.num_nodes = num_nodes
        self.num_edges = len(src)
        self._backend = resolve_backend(backend)
        self._node_feats_np = node_feats
        self._edge_feats_np = edge_feats

        logger.info(
            "TemporalGraphSampler: %d nodes, %d edges, backend=%s",
            num_nodes, self.num_edges, self._backend.value,
        )

        if self._backend == Backend.CUDA:
            self._init_cuda(num_nodes, src, dst, timestamps, edge_ids,
                            node_feats, edge_feats)
        elif self._backend == Backend.CPP:
            self._init_cpp(num_nodes, src, dst, timestamps, edge_ids)
        else:
            self._init_python(num_nodes, src, dst, timestamps, edge_ids)

    def _init_python(self, num_nodes, src, dst, timestamps, edge_ids):
        """Build CSR in pure Python/NumPy."""
        sort_idx = np.lexsort((timestamps, src))
        self._py_neighbors = dst[sort_idx].astype(np.int32)
        self._py_timestamps = timestamps[sort_idx].astype(np.float64)
        self._py_edge_ids = edge_ids[sort_idx].astype(np.int64)

        self._py_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        src_sorted = src[sort_idx]
        for s in src_sorted:
            self._py_indptr[s + 1] += 1
        np.cumsum(self._py_indptr, out=self._py_indptr)

    def _init_cpp(self, num_nodes, src, dst, timestamps, edge_ids):
        """Build CSR via C++ extension."""
        ext = _get_extension(Backend.CPP)
        self._cpp_csr = ext.TemporalCSR(
            num_nodes,
            np.ascontiguousarray(src, dtype=np.int32),
            np.ascontiguousarray(dst, dtype=np.int32),
            np.ascontiguousarray(timestamps, dtype=np.float64),
            np.ascontiguousarray(edge_ids, dtype=np.int64),
        )
        self._cpp_ext = ext

    def _init_cuda(self, num_nodes, src, dst, timestamps, edge_ids,
                   node_feats, edge_feats):
        """Build CSR on GPU via CUDA extension."""
        import torch

        ext = _get_extension(Backend.CUDA)
        self._cuda_csr = ext.TemporalCSR_CUDA(
            num_nodes,
            torch.from_numpy(np.ascontiguousarray(src, dtype=np.int32)),
            torch.from_numpy(np.ascontiguousarray(dst, dtype=np.int32)),
            torch.from_numpy(np.ascontiguousarray(timestamps, dtype=np.float64)),
            torch.from_numpy(np.ascontiguousarray(edge_ids, dtype=np.int64)),
        )
        self._cuda_ext = ext

        if node_feats is not None:
            self._node_feats_gpu = torch.from_numpy(
                np.ascontiguousarray(node_feats, dtype=np.float32)
            ).cuda()
        else:
            self._node_feats_gpu = None

        if edge_feats is not None:
            self._edge_feats_gpu = torch.from_numpy(
                np.ascontiguousarray(edge_feats, dtype=np.float32)
            ).cuda()
        else:
            self._edge_feats_gpu = None

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return self._backend.value

    def sample_neighbors(
        self,
        nodes: np.ndarray,
        query_timestamps: np.ndarray,
        num_neighbors: int = 20,
    ) -> NeighborBatch:
        """Sample K most recent temporal neighbors for a batch of queries.

        Args:
            nodes: [B] query node indices.
            query_timestamps: [B] query timestamps.
            num_neighbors: Max neighbors per query (K).

        Returns:
            NeighborBatch with results on GPU (CUDA) or CPU (Python/C++).
        """
        if self._backend == Backend.CUDA:
            return self._sample_neighbors_cuda(nodes, query_timestamps,
                                                num_neighbors)
        elif self._backend == Backend.CPP:
            return self._sample_neighbors_cpp(nodes, query_timestamps,
                                               num_neighbors)
        else:
            return self._sample_neighbors_python(nodes, query_timestamps,
                                                  num_neighbors)

    def _sample_neighbors_python(self, nodes, query_timestamps,
                                  num_neighbors) -> NeighborBatch:
        """Pure Python/NumPy temporal neighbor sampling."""
        batch_size = len(nodes)
        out_n = np.full((batch_size, num_neighbors), -1, dtype=np.int32)
        out_ts = np.zeros((batch_size, num_neighbors), dtype=np.float64)
        out_eid = np.full((batch_size, num_neighbors), -1, dtype=np.int64)
        out_len = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            node = int(nodes[i])
            before_time = float(query_timestamps[i])

            if node < 0 or node >= self.num_nodes:
                continue

            start = self._py_indptr[node]
            end = self._py_indptr[node + 1]
            if start == end:
                continue

            ts_slice = self._py_timestamps[start:end]
            valid_end = np.searchsorted(ts_slice, before_time, side="left")
            if valid_end == 0:
                continue

            actual_start = max(0, valid_end - num_neighbors)
            length = valid_end - actual_start
            out_len[i] = length

            abs_start = start + actual_start
            abs_end = start + valid_end
            out_n[i, :length] = self._py_neighbors[abs_start:abs_end]
            out_ts[i, :length] = self._py_timestamps[abs_start:abs_end]
            out_eid[i, :length] = self._py_edge_ids[abs_start:abs_end]

        return NeighborBatch(out_n, out_ts, out_eid, out_len, on_gpu=False)

    def _sample_neighbors_cpp(self, nodes, query_timestamps,
                               num_neighbors) -> NeighborBatch:
        """C++ backend temporal neighbor sampling."""
        result = self._cpp_ext.sample_neighbors_batch(
            self._cpp_csr,
            np.ascontiguousarray(nodes, dtype=np.int32),
            np.ascontiguousarray(query_timestamps, dtype=np.float64),
            num_neighbors,
        )
        nn, nts, neids, lengths = result
        return NeighborBatch(nn, nts, neids, lengths, on_gpu=False)

    def _sample_neighbors_cuda(self, nodes, query_timestamps,
                                num_neighbors) -> NeighborBatch:
        """CUDA backend temporal neighbor sampling."""
        import torch

        q_nodes = torch.from_numpy(
            np.ascontiguousarray(nodes, dtype=np.int32)
        )
        q_times = torch.from_numpy(
            np.ascontiguousarray(query_timestamps, dtype=np.float64)
        )

        results = self._cuda_ext.sample_neighbors_batch_cuda(
            self._cuda_csr, q_nodes, q_times, num_neighbors
        )
        return NeighborBatch(
            results[0], results[1], results[2], results[3], on_gpu=True
        )

    def featurize(
        self,
        nbr: NeighborBatch,
        query_timestamps: Optional[np.ndarray] = None,
        node_feats: Optional[np.ndarray] = None,
        edge_feats: Optional[np.ndarray] = None,
    ) -> FeatureBatch:
        """Gather node/edge features for sampled neighbors.

        Args:
            nbr: NeighborBatch from sample_neighbors().
            query_timestamps: [B] query timestamps (required for CPU backends).
            node_feats: Override node features (if not set at construction).
            edge_feats: Override edge features (if not set at construction).

        Returns:
            FeatureBatch with features on same device as input.
        """
        if self._backend == Backend.CUDA and nbr.on_gpu:
            return self._featurize_cuda(nbr, query_timestamps)

        nf = node_feats if node_feats is not None else self._node_feats_np
        ef = edge_feats if edge_feats is not None else self._edge_feats_np
        if nf is None or ef is None:
            raise ValueError("node_feats and edge_feats required for featurize")

        if self._backend == Backend.CPP:
            return self._featurize_cpp(nbr, query_timestamps, nf, ef)
        return self._featurize_python(nbr, query_timestamps, nf, ef)

    def _featurize_python(self, nbr, query_ts, node_feats,
                           edge_feats) -> FeatureBatch:
        """Pure Python/NumPy feature gathering."""
        nn = nbr.neighbor_ids
        neids = nbr.edge_ids
        lengths = nbr.lengths
        nts = nbr.timestamps

        batch_size, K = nn.shape
        nf_dim = node_feats.shape[1]
        ef_dim = edge_feats.shape[1]
        num_total_edges = len(edge_feats)

        out_nf = np.zeros((batch_size, K, nf_dim), dtype=np.float32)
        out_ef = np.zeros((batch_size, K, ef_dim), dtype=np.float32)
        out_rt = np.zeros((batch_size, K), dtype=np.float64)

        for i in range(batch_size):
            length = int(lengths[i])
            if length == 0:
                continue
            valid_nids = nn[i, :length]
            out_nf[i, :length] = node_feats[valid_nids]

            for j in range(length):
                eid = int(neids[i, j])
                if 0 <= eid < num_total_edges:
                    out_ef[i, j] = edge_feats[eid]

            out_rt[i, :length] = float(query_ts[i]) - nts[i, :length]

        return FeatureBatch(out_nf, out_ef, out_rt, on_gpu=False)

    def _featurize_cpp(self, nbr, query_ts, node_feats,
                        edge_feats) -> FeatureBatch:
        """C++ backend feature gathering."""
        result = self._cpp_ext.featurize_neighbors(
            np.ascontiguousarray(nbr.neighbor_ids, dtype=np.int32),
            np.ascontiguousarray(nbr.edge_ids, dtype=np.int64),
            np.ascontiguousarray(nbr.lengths, dtype=np.int32),
            np.ascontiguousarray(nbr.timestamps, dtype=np.float64),
            np.ascontiguousarray(query_ts, dtype=np.float64),
            np.ascontiguousarray(node_feats, dtype=np.float32),
            np.ascontiguousarray(edge_feats, dtype=np.float32),
        )
        return FeatureBatch(result[0], result[1], result[2], on_gpu=False)

    def _featurize_cuda(self, nbr, query_ts) -> FeatureBatch:
        """CUDA backend feature gathering."""
        import torch

        if self._node_feats_gpu is None or self._edge_feats_gpu is None:
            raise ValueError("node_feats and edge_feats required on GPU")

        if query_ts is not None and not isinstance(query_ts, torch.Tensor):
            q_ts = torch.from_numpy(
                np.ascontiguousarray(query_ts, dtype=np.float64)
            ).cuda()
        elif query_ts is not None:
            q_ts = query_ts.cuda() if not query_ts.is_cuda else query_ts
        else:
            raise ValueError("query_timestamps required for featurize")

        results = self._cuda_ext.featurize_neighbors_cuda(
            nbr.neighbor_ids,
            nbr.edge_ids,
            nbr.lengths,
            nbr.timestamps,
            q_ts,
            self._node_feats_gpu,
            self._edge_feats_gpu,
        )
        return FeatureBatch(results[0], results[1], results[2], on_gpu=True)

    def sample_negatives(
        self,
        src_nodes: np.ndarray,
        true_dst: np.ndarray,
        query_timestamps: np.ndarray,
        n_negatives: int = 100,
        strategy: str = "mixed",
        hist_ratio: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Generate negative destination nodes for evaluation.

        Supports multiple strategies for flexible experimentation.

        Args:
            src_nodes: [B] source node indices.
            true_dst: [B] true destination indices (excluded from negatives).
            query_timestamps: [B] query timestamps.
            n_negatives: Total negatives per query.
            strategy: 'random', 'historical', 'mixed', or 'active'.
            hist_ratio: Fraction of historical negatives (for 'mixed').
            rng: Random number generator.

        Returns:
            [B, n_negatives] int32 array of negative node indices.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        batch_size = len(src_nodes)
        negatives = np.zeros((batch_size, n_negatives), dtype=np.int32)

        n_hist = int(n_negatives * hist_ratio) if strategy == "mixed" else 0
        if strategy == "historical":
            n_hist = n_negatives
        n_rand = n_negatives - n_hist

        for i in range(batch_size):
            src = int(src_nodes[i])
            dst = int(true_dst[i])
            ts = float(query_timestamps[i])

            neg_set = set()
            exclude = {src, dst}
            offset = 0

            if n_hist > 0:
                nbr = self.sample_neighbors(
                    np.array([src], dtype=np.int32),
                    np.array([ts], dtype=np.float64),
                    num_neighbors=500,
                )
                if nbr.on_gpu:
                    hist_ids = nbr.neighbor_ids[0].cpu().numpy()
                    hist_len = int(nbr.lengths[0].cpu().item())
                else:
                    hist_ids = nbr.neighbor_ids[0]
                    hist_len = int(nbr.lengths[0])

                hist_pool = []
                for j in range(hist_len):
                    nid = int(hist_ids[j])
                    if nid not in exclude:
                        hist_pool.append(nid)

                if len(hist_pool) >= n_hist:
                    chosen = rng.choice(hist_pool, size=n_hist, replace=False)
                elif len(hist_pool) > 0:
                    chosen = rng.choice(hist_pool, size=n_hist, replace=True)
                else:
                    chosen = rng.integers(0, self.num_nodes, size=n_hist)

                for val in chosen:
                    negatives[i, offset] = val
                    neg_set.add(int(val))
                    offset += 1

            exclude = exclude | neg_set
            filled = 0
            while filled < n_rand:
                candidates = rng.integers(0, self.num_nodes, size=n_rand * 2)
                for c in candidates:
                    c_int = int(c)
                    if c_int not in exclude:
                        negatives[i, offset + filled] = c_int
                        exclude.add(c_int)
                        filled += 1
                        if filled >= n_rand:
                            break

        return negatives

    def to_numpy(self, nbr: NeighborBatch) -> NeighborBatch:
        """Convert GPU NeighborBatch to CPU numpy arrays."""
        if not nbr.on_gpu:
            return nbr
        return NeighborBatch(
            neighbor_ids=nbr.neighbor_ids.cpu().numpy(),
            timestamps=nbr.timestamps.cpu().numpy(),
            edge_ids=nbr.edge_ids.cpu().numpy(),
            lengths=nbr.lengths.cpu().numpy(),
            on_gpu=False,
        )

    def to_numpy_features(self, feat: FeatureBatch) -> FeatureBatch:
        """Convert GPU FeatureBatch to CPU numpy arrays."""
        if not feat.on_gpu:
            return feat
        return FeatureBatch(
            node_features=feat.node_features.cpu().numpy(),
            edge_features=feat.edge_features.cpu().numpy(),
            rel_timestamps=feat.rel_timestamps.cpu().numpy(),
            on_gpu=False,
        )
