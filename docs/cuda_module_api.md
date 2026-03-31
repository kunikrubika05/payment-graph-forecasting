# GPU Primitives for Temporal Graph Learning

This module provides CUDA-accelerated primitives for training and inference
on temporal transaction graphs. Each primitive supports three backends
(Python, C++, CUDA) with a unified API and automatic backend selection.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/kunikrubika05/payment-graph-forecasting.git
cd payment-graph-forecasting
pip install -e .

# Build C++ and CUDA extensions (requires NVCC)
# Also requires `ninja` in the active environment for torch extension builds.
TORCH_CUDA_ARCH_LIST="7.0"  # V100
# TORCH_CUDA_ARCH_LIST="8.6"  # A10/A100
python -m payment_graph_forecasting.infra.extensions --all --graph-metrics --graph-metrics-cuda
```

Verified on: Python 3.12, PyTorch 2.5.1+cu121, CUDA 12.8, V100 / A10.

`src/models/build_ext.py` still works as a compatibility shim, but the
package-facing build CLI above is the canonical entrypoint.

For package-facing experiments, the CUDA backend is enabled by config, not by
importing a separate runner. The stable public surface is:

```yaml
experiment:
  model: dygformer

sampling:
  backend: cuda
```

and launched through:

```bash
python -m payment_graph_forecasting.experiments.launcher --config spec.yaml
```

---

## Primitive 1: Temporal Neighbor Sampling

Retrieves the K most recent neighbors of a node at a given timestamp.
Required for training GraphMixer, DyGFormer, TGN, and similar models.

### Quick start

```python
from src.models.temporal_graph_sampler import TemporalGraphSampler, Backend
import numpy as np

# Build sampler from event stream (sorted by timestamp)
# src, dst: [E] int64 — edge endpoints
# ts:       [E] float64 — edge timestamps (ascending)
# node_feat:[N, F_n] float32 — static node features
# edge_feat:[E, F_e] float32 — edge features
sampler = TemporalGraphSampler(
    src=src, dst=dst, timestamps=ts,
    node_feat=node_feat, edge_feat=edge_feat,
    backend=Backend.AUTO,  # cuda > cpp > python
)

# Sample K=20 most recent neighbors for a batch of queries
neighbors = sampler.sample_neighbors(
    nodes=np.array([0, 1, 2]),      # [B] query nodes
    times=np.array([1000, 2000, 3000]),  # [B] query timestamps
    K=20,
)
# neighbors.node_ids:   [B, K] int64 — neighbor node indices (-1 = padding)
# neighbors.edge_ids:   [B, K] int64 — edge indices
# neighbors.timestamps: [B, K] float64 — edge timestamps
# neighbors.mask:       [B, K] bool — True where valid

# Gather features for sampled neighbors
features = sampler.featurize(neighbors)
# features.node_feat: [B, K, F_n] float32
# features.edge_feat: [B, K, F_e] float32
# features.rel_time:  [B, K] float32 — time elapsed since query

# Sample negative destinations
negatives = sampler.sample_negatives(
    nodes=np.array([0, 1, 2]),
    times=np.array([1000, 2000, 3000]),
    n_neg=100,
    strategy="mixed",  # "random" | "historical" | "mixed"
)
# negatives: [B, n_neg] int64 — negative destination node indices
```

### Backend selection

```python
from src.models.temporal_graph_sampler import Backend

sampler = TemporalGraphSampler(..., backend=Backend.AUTO)   # recommended
sampler = TemporalGraphSampler(..., backend=Backend.CUDA)   # GPU, fastest for K>=100
sampler = TemporalGraphSampler(..., backend=Backend.CPP)    # CPU C++, sparse graphs
sampler = TemporalGraphSampler(..., backend=Backend.PYTHON) # always available
```

### Performance (V100, synthetic graph)

| Setting | Python | C++ | CUDA | CUDA speedup |
|---------|--------|-----|------|--------------|
| batch=512, K=20 (GraphMixer) | 7.55ms | 0.21ms | 0.15ms | **50×** |
| batch=512, K=512 (DyGFormer small) | 11.49ms | 4.04ms | 0.18ms | **64×** |
| batch=2048, K=512 (DyGFormer) | 62.33ms | 34.22ms | 0.34ms | **184×** |
| batch=512, K=512, 5M nodes | 8.64ms | 2.61ms | 0.18ms | **49×** |

The C++ backend uses binary search on a timestamp-sorted CSR structure.
The CUDA backend parallelises search and featurization across the entire
batch, achieving near-linear scaling with batch size and K.

**When to use CUDA:** K ≥ 100 or batch_size ≥ 1024. For K=20 (GraphMixer),
C++ is already fast enough — CUDA provides a modest additional gain.

### DyGFormer integration in the framework

DyGFormer now uses the same package-facing API as the other integrated models.

Relevant files:

- [payment_graph_forecasting/models/dygformer.py](/Users/kunikrubika/Desktop/payment-graph-forecasting/payment_graph_forecasting/models/dygformer.py)
- [payment_graph_forecasting/training/api.py](/Users/kunikrubika/Desktop/payment-graph-forecasting/payment_graph_forecasting/training/api.py)
- [payment_graph_forecasting/experiments/runners/dygformer.py](/Users/kunikrubika/Desktop/payment-graph-forecasting/payment_graph_forecasting/experiments/runners/dygformer.py)
- [src/models/DyGFormer/dygformer_cuda_train.py](/Users/kunikrubika/Desktop/payment-graph-forecasting/src/models/DyGFormer/dygformer_cuda_train.py)

Dispatch rule:

- `sampling.backend: auto` -> legacy CPU/C++-compatible path
- `sampling.backend: cuda` -> `train_dygformer_cuda(...)`

The model itself remains standard PyTorch DyGFormer. The optimisation is in
the training pipeline around temporal sampling and batch preparation.

### Real V100 results for the integrated DyGFormer path

Measured on ORBITAAL stream-graph, summer 2020, 10% sample:

| Config | Epoch estimate |
|---|---|
| old package path, `batch_size=1536`, `num_neighbors=32` | `~47.9 min` |
| CUDA path, `batch_size=1536`, `num_neighbors=32` | `~27.1 min` |
| CUDA path, `batch_size=3072`, `num_neighbors=32` | `~26.6 min` |
| CUDA path, `batch_size=3072`, `num_neighbors=24` | `~19.7 min` |

Recommended default for full training:

```yaml
sampling:
  backend: cuda
  num_neighbors: 32

training:
  batch_size: 3072
```

`num_neighbors: 24` is the speed-oriented alternative. It is faster, but it is
not the safest default for quality-sensitive long runs.

---

## Primitive 2: Common Neighbors

Computes the exact intersection size |N(u) ∩ N(v)| for a batch of node
pairs. Strongest structural heuristic for link prediction. Can be used
as a real-time feature inside neural network scoring functions.

### Quick start

```python
from src.models.graph_metrics import CommonNeighbors
from scipy import sparse
import numpy as np

# Load a static undirected binary adjacency (scipy CSR)
adj = sparse.load_npz("adj_undirected.npz")   # symmetric, sorted indices

cn = CommonNeighbors(adj, backend="auto")  # cuda > cpp > python

# Compute CN for a batch of pairs
src = np.array([0, 1, 100], dtype=np.int64)
dst = np.array([3, 4, 200], dtype=np.int64)
counts = cn.compute(src, dst)
# counts: [3] int32 — exact |N(src[i]) ∩ N(dst[i])|
```

### Using CN as a neural network feature

```python
# Example: BUDDY-style scoring with exact CN instead of MinHash approximation
cn_vals = cn.compute(src_batch, dst_batch)        # [B] int32, exact
cn_feat = np.log1p(cn_vals).astype(np.float32)   # log-normalise
# Concatenate with learned node embeddings and pass to MLP scorer
score = mlp(np.stack([h_src, h_dst, cn_feat], axis=-1))
```

### Performance (V100, synthetic graph)

| Graph regime | Python | C++ | CUDA | Best backend |
|---|---|---|---|---|
| Sparse, avg_deg=6 (Bitcoin) | 0.42ms | **0.07ms** | 1.18ms | **C++** |
| Transition, avg_deg=50 | 1.27ms | **0.57ms** | 0.73ms | **C++** |
| Dense, avg_deg=200 | 2.00ms | 1.02ms | **0.42ms** | **CUDA** |
| Dense, avg_deg=500 | 8.51ms | 5.05ms | **0.71ms** | **CUDA** |
| Dense, avg_deg=1000 | 35.79ms | 20.85ms | **1.66ms** | **CUDA** |

Batch size = 512–2048 pairs. N = 50K–500K nodes.

**When to use CUDA:** avg_deg ≥ 100–150. On sparse real-world graphs
(social, citation, e-commerce at density < 50), the C++ backend is faster
due to GPU kernel launch overhead exceeding the actual computation.

**Why not approximation?** BUDDY (Chamberlain et al., ICLR 2023) approximates
CN via MinHash to avoid the computational cost of exact computation. Our
CUDA implementation makes exact CN practical for dense graphs, eliminating
the approximation error entirely.

---

## Running benchmarks

```bash
# Temporal sampling (5 scenarios)
PYTHONPATH=. python scripts/bench_sampling.py --cuda 2>&1 | tee /tmp/bench_sampling.log

# Common neighbors (5 scenarios)
PYTHONPATH=. python scripts/bench_cn.py --cuda 2>&1 | tee /tmp/bench_cn.log
```

---

## Running tests

```bash
# 30 tests: Python / C++ / CUDA backend equivalence for temporal sampling
PYTHONPATH=. python -m pytest tests/test_temporal_sampler.py -v

# 16 tests: Python / C++ / CUDA correctness for common neighbors
PYTHONPATH=. python -m pytest tests/test_graph_metrics.py -v
```

All 46 tests pass on V100 with the compiled CUDA extensions.
