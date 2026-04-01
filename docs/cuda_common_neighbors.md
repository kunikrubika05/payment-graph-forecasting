# Common Neighbors Notes

`payment_graph_forecasting.CommonNeighbors` is the maintained runtime API for
exact common-neighbor counts.

What it covers:

- exact `|N(u) ∩ N(v)|` computation for batched node pairs
- unified backend selection across Python, C++, and CUDA

Practical rule of thumb:

- sparse graphs: C++ is usually the best backend
- denser graphs: CUDA becomes more attractive

For maintained commands and entrypoints, see
[cuda_module_api.md](cuda_module_api.md).
