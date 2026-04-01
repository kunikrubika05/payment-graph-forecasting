# CUDA Module Overview

This repository exposes two maintained CUDA-capable runtime primitives through
the package surface:

- `payment_graph_forecasting.TemporalGraphSampler`
- `payment_graph_forecasting.CommonNeighbors`

The capability probe is:

- `payment_graph_forecasting.describe_cuda_capabilities()`

Control surfaces:

- YAML runs: `sampling.backend`
- direct Python runtime use: package imports above

Reference entrypoints:

- [cuda_module_api.md](cuda_module_api.md)
- [README.md](../README.md)

Backend selection summary:

| Primitive | Python | C++ | CUDA |
| --- | --- | --- | --- |
| `TemporalGraphSampler` | fallback | strong CPU default | best for large batched temporal sampling |
| `CommonNeighbors` | fallback | often best on sparse graphs | best on denser graphs |
