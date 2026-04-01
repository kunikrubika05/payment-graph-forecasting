# GPU Primitives for Temporal Graph Learning

Maintained package-facing CUDA/runtime surface:

- experiment control: YAML `sampling.backend`
- library/runtime control:
  - `payment_graph_forecasting.TemporalGraphSampler`
  - `payment_graph_forecasting.CommonNeighbors`
  - `payment_graph_forecasting.describe_cuda_capabilities()`

## Verified Entry Points

```bash
./venv/bin/python -m payment_graph_forecasting.infra.extensions --help
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --help
./venv/bin/python -m payment_graph_forecasting.experiments.hpo --help
```

## Experiment Contract

Use CUDA through YAML:

```yaml
experiment:
  model: dygformer

sampling:
  backend: cuda
```

Run the spec through the package launcher.

## Library Contract

```python
import payment_graph_forecasting as pgf

caps = pgf.describe_cuda_capabilities()
sampler_cls = pgf.TemporalGraphSampler
graph_metric_cls = pgf.CommonNeighbors
```

## Primitive Summary

### Temporal Neighbor Sampling

`TemporalGraphSampler` provides a unified interface over Python, C++, and CUDA
backends for temporal neighbor lookup and feature gathering.

Backend rule of thumb:

- `python`: fallback, always available
- `cpp`: strong default for CPU runs and sparse graphs
- `cuda`: best fit for large batches and large neighbor counts

### Common Neighbors

`CommonNeighbors` provides exact common-neighbor counts with the same backend
selection pattern.

Backend rule of thumb:

- sparse graphs: C++ is usually the best backend
- denser graphs: CUDA can win clearly

## Related Docs

- [cuda_module_overview.md](cuda_module_overview.md)
- [cuda_temporal_sampling.md](cuda_temporal_sampling.md)
- [cuda_common_neighbors.md](cuda_common_neighbors.md)
