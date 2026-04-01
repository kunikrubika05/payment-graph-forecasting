# Temporal Sampling Notes

`payment_graph_forecasting.TemporalGraphSampler` is the maintained runtime API
for temporal neighbor sampling.

What it covers:

- batched lookup of recent temporal neighbors before a query timestamp
- optional feature gathering for sampled neighbors
- backend selection across Python, C++, and CUDA

Preferred control surfaces:

- YAML experiments: `sampling.backend`
- direct runtime use: `payment_graph_forecasting.TemporalGraphSampler`

When CUDA is most useful:

- large `num_neighbors`
- large batch size
- training paths such as `dygformer` where temporal sampling dominates runtime

For maintained commands and entrypoints, see
[cuda_module_api.md](cuda_module_api.md).
