/**
 * CUDA extension for GPU-accelerated temporal neighbor sampling.
 *
 * Provides the same functionality as temporal_sampling.cpp but executes
 * on GPU with massive parallelism. Each query (node, timestamp) is
 * processed by an independent CUDA thread.
 *
 * Three main operations:
 *   1. sample_neighbors_batch_cuda — parallel temporal neighbor lookup
 *   2. featurize_neighbors_cuda — parallel feature gather
 *   3. generate_negatives_cuda — parallel negative sampling
 *
 * The CSR structure is loaded onto GPU once; all subsequent operations
 * run entirely on GPU without per-batch CPU<->GPU transfers.
 *
 * Build: python src/models/build_ext.py --cuda
 * Falls back to C++/Python when CUDA is unavailable.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ",                      \
                    cudaGetErrorString(err), " at ", __FILE__, ":", __LINE__);\
    } while (0)

namespace {

constexpr int BLOCK_SIZE = 256;

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }


/**
 * Kernel: binary search in sorted timestamp array to find the position
 * where timestamps[pos] >= before_time (i.e., upper bound of valid range).
 *
 * Each thread handles one (node, query_time) pair independently.
 */
__device__ int64_t binary_search_before(
    const double* __restrict__ timestamps,
    int64_t start,
    int64_t end,
    double before_time
) {
    int64_t lo = start;
    int64_t hi = end;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (timestamps[mid] < before_time) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}


/**
 * Kernel: sample K most recent temporal neighbors for a batch of queries.
 *
 * For each query i:
 *   - Look up node's edge range in CSR: [indptr[node], indptr[node+1])
 *   - Binary search for valid_end where timestamp < before_time
 *   - Copy last min(K, valid_count) neighbors into output
 *
 * Output arrays are pre-filled with padding (-1 for ids, 0.0 for timestamps).
 */
__global__ void sample_neighbors_kernel(
    const int64_t* __restrict__ indptr,
    const int32_t* __restrict__ neighbors,
    const double*  __restrict__ csr_timestamps,
    const int64_t* __restrict__ edge_ids,
    const int32_t* __restrict__ query_nodes,
    const double*  __restrict__ query_times,
    int32_t* __restrict__ out_neighbors,
    double*  __restrict__ out_timestamps,
    int64_t* __restrict__ out_edge_ids,
    int32_t* __restrict__ out_lengths,
    int batch_size,
    int num_neighbors,
    int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int32_t node = query_nodes[idx];
    double before_time = query_times[idx];

    int64_t row_offset = (int64_t)idx * num_neighbors;

    if (node < 0 || node >= num_nodes) {
        out_lengths[idx] = 0;
        return;
    }

    int64_t start = indptr[node];
    int64_t end = indptr[node + 1];

    if (start == end) {
        out_lengths[idx] = 0;
        return;
    }

    int64_t valid_end = binary_search_before(
        csr_timestamps, start, end, before_time
    );

    if (valid_end == start) {
        out_lengths[idx] = 0;
        return;
    }

    int64_t count = valid_end - start;
    int32_t length = (int32_t)min((int64_t)num_neighbors, count);
    int64_t actual_start = valid_end - length;

    out_lengths[idx] = length;

    for (int32_t j = 0; j < length; j++) {
        int64_t src_idx = actual_start + j;
        out_neighbors[row_offset + j]  = neighbors[src_idx];
        out_timestamps[row_offset + j] = csr_timestamps[src_idx];
        out_edge_ids[row_offset + j]   = edge_ids[src_idx];
    }
}


/**
 * Kernel: gather node/edge features for sampled neighbors.
 *
 * Each thread handles one (batch_element, neighbor_slot) pair.
 * Total threads = batch_size * K.
 */
__global__ void featurize_neighbors_kernel(
    const int32_t* __restrict__ neighbor_nodes,
    const int64_t* __restrict__ neighbor_eids,
    const int32_t* __restrict__ lengths,
    const double*  __restrict__ neighbor_ts,
    const double*  __restrict__ query_ts,
    const float*   __restrict__ node_feats,
    const float*   __restrict__ edge_feats,
    float*  __restrict__ out_node_feats,
    float*  __restrict__ out_edge_feats,
    double* __restrict__ out_rel_ts,
    int batch_size,
    int K,
    int node_feat_dim,
    int edge_feat_dim,
    int64_t num_total_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * K;
    if (tid >= total) return;

    int i = tid / K;
    int j = tid % K;

    if (j >= lengths[i]) return;

    int64_t flat_idx = (int64_t)i * K + j;

    int32_t nid = neighbor_nodes[flat_idx];
    int64_t eid = neighbor_eids[flat_idx];

    if (nid >= 0) {
        const float* src_nf = node_feats + (int64_t)nid * node_feat_dim;
        float* dst_nf = out_node_feats + flat_idx * node_feat_dim;
        for (int d = 0; d < node_feat_dim; d++) {
            dst_nf[d] = src_nf[d];
        }
    }

    if (eid >= 0 && eid < num_total_edges) {
        const float* src_ef = edge_feats + eid * edge_feat_dim;
        float* dst_ef = out_edge_feats + flat_idx * edge_feat_dim;
        for (int d = 0; d < edge_feat_dim; d++) {
            dst_ef[d] = src_ef[d];
        }
    }

    out_rel_ts[flat_idx] = query_ts[i] - neighbor_ts[flat_idx];
}

}  // anonymous namespace


/**
 * GPU-resident CSR structure for temporal graph.
 *
 * All arrays live in GPU global memory. Constructed once from CPU arrays,
 * then reused for all batches without further CPU<->GPU transfers.
 */
class TemporalCSR_CUDA {
public:
    int num_nodes;
    int64_t num_edges;

    torch::Tensor indptr;       // [num_nodes + 1], int64
    torch::Tensor neighbors;    // [num_edges], int32
    torch::Tensor timestamps;   // [num_edges], float64
    torch::Tensor edge_ids;     // [num_edges], int64

    TemporalCSR_CUDA(
        int num_nodes,
        torch::Tensor src,
        torch::Tensor dst,
        torch::Tensor ts,
        torch::Tensor eids
    ) : num_nodes(num_nodes) {
        TORCH_CHECK(src.device().is_cpu(), "src must be a CPU tensor for construction");
        TORCH_CHECK(src.dim() == 1, "src must be 1D");

        num_edges = src.size(0);

        auto src_a = src.accessor<int32_t, 1>();
        auto dst_a = dst.accessor<int32_t, 1>();
        auto ts_a = ts.accessor<double, 1>();
        auto eids_a = eids.accessor<int64_t, 1>();

        std::vector<int64_t> sort_idx(num_edges);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::sort(sort_idx.begin(), sort_idx.end(),
            [&](int64_t a, int64_t b) {
                if (src_a[a] != src_a[b]) return src_a[a] < src_a[b];
                return ts_a[a] < ts_a[b];
            });

        auto h_neighbors  = torch::empty({num_edges}, torch::kInt32);
        auto h_timestamps  = torch::empty({num_edges}, torch::kFloat64);
        auto h_edge_ids    = torch::empty({num_edges}, torch::kInt64);
        auto h_indptr      = torch::zeros({num_nodes + 1}, torch::kInt64);

        auto n_ptr = h_neighbors.accessor<int32_t, 1>();
        auto t_ptr = h_timestamps.accessor<double, 1>();
        auto e_ptr = h_edge_ids.accessor<int64_t, 1>();
        auto i_ptr = h_indptr.accessor<int64_t, 1>();

        for (int64_t i = 0; i < num_edges; i++) {
            int64_t idx = sort_idx[i];
            n_ptr[i] = dst_a[idx];
            t_ptr[i] = ts_a[idx];
            e_ptr[i] = eids_a[idx];
            i_ptr[src_a[idx] + 1]++;
        }

        for (int i = 1; i <= num_nodes; i++) {
            i_ptr[i] += i_ptr[i - 1];
        }

        indptr     = h_indptr.to(torch::kCUDA);
        neighbors  = h_neighbors.to(torch::kCUDA);
        timestamps = h_timestamps.to(torch::kCUDA);
        edge_ids   = h_edge_ids.to(torch::kCUDA);
    }

    torch::Device device() const {
        return indptr.device();
    }
};


/**
 * Batch temporal neighbor sampling on GPU.
 *
 * Args:
 *   csr: GPU-resident TemporalCSR_CUDA
 *   query_nodes: [B] int32, CPU or CUDA tensor
 *   query_timestamps: [B] float64, CPU or CUDA tensor
 *   num_neighbors: K, max neighbors per query
 *
 * Returns:
 *   tuple of (neighbor_nodes[B,K], neighbor_ts[B,K],
 *             neighbor_eids[B,K], lengths[B])
 *   All on CUDA.
 */
std::vector<torch::Tensor> sample_neighbors_batch_cuda(
    const TemporalCSR_CUDA& csr,
    torch::Tensor query_nodes,
    torch::Tensor query_timestamps,
    int num_neighbors
) {
    auto device = csr.device();

    if (query_nodes.device().is_cpu()) {
        query_nodes = query_nodes.to(device);
    }
    if (query_timestamps.device().is_cpu()) {
        query_timestamps = query_timestamps.to(device);
    }

    int batch_size = query_nodes.size(0);

    auto out_neighbors = torch::full(
        {batch_size, num_neighbors}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );
    auto out_timestamps = torch::zeros(
        {batch_size, num_neighbors},
        torch::TensorOptions().dtype(torch::kFloat64).device(device)
    );
    auto out_edge_ids = torch::full(
        {batch_size, num_neighbors}, -1,
        torch::TensorOptions().dtype(torch::kInt64).device(device)
    );
    auto out_lengths = torch::zeros(
        {batch_size},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );

    if (batch_size == 0) {
        return {out_neighbors, out_timestamps, out_edge_ids, out_lengths};
    }

    int grid = div_ceil(batch_size, BLOCK_SIZE);

    sample_neighbors_kernel<<<grid, BLOCK_SIZE>>>(
        csr.indptr.data_ptr<int64_t>(),
        csr.neighbors.data_ptr<int32_t>(),
        csr.timestamps.data_ptr<double>(),
        csr.edge_ids.data_ptr<int64_t>(),
        query_nodes.data_ptr<int32_t>(),
        query_timestamps.data_ptr<double>(),
        out_neighbors.data_ptr<int32_t>(),
        out_timestamps.data_ptr<double>(),
        out_edge_ids.data_ptr<int64_t>(),
        out_lengths.data_ptr<int32_t>(),
        batch_size,
        num_neighbors,
        csr.num_nodes
    );
    CUDA_CHECK(cudaGetLastError());

    return {out_neighbors, out_timestamps, out_edge_ids, out_lengths};
}


/**
 * Batch feature gathering on GPU.
 *
 * Args:
 *   neighbor_nodes: [B, K] int32, CUDA
 *   neighbor_eids:  [B, K] int64, CUDA
 *   lengths:        [B] int32, CUDA
 *   neighbor_ts:    [B, K] float64, CUDA
 *   query_ts:       [B] float64, CUDA
 *   node_feats:     [num_nodes, node_feat_dim] float32, CUDA
 *   edge_feats:     [num_edges, edge_feat_dim] float32, CUDA
 *
 * Returns:
 *   tuple of (out_node_feats[B,K,nf_dim], out_edge_feats[B,K,ef_dim],
 *             out_rel_ts[B,K])
 *   All on CUDA.
 */
std::vector<torch::Tensor> featurize_neighbors_cuda(
    torch::Tensor neighbor_nodes,
    torch::Tensor neighbor_eids,
    torch::Tensor lengths,
    torch::Tensor neighbor_ts,
    torch::Tensor query_ts,
    torch::Tensor node_feats,
    torch::Tensor edge_feats
) {
    auto device = neighbor_nodes.device();

    int batch_size = neighbor_nodes.size(0);
    int K = neighbor_nodes.size(1);
    int node_feat_dim = node_feats.size(1);
    int edge_feat_dim = edge_feats.size(1);
    int64_t num_total_edges = edge_feats.size(0);

    auto out_node_feats = torch::zeros(
        {batch_size, K, node_feat_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );
    auto out_edge_feats = torch::zeros(
        {batch_size, K, edge_feat_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );
    auto out_rel_ts = torch::zeros(
        {batch_size, K},
        torch::TensorOptions().dtype(torch::kFloat64).device(device)
    );

    if (batch_size == 0 || K == 0) {
        return {out_node_feats, out_edge_feats, out_rel_ts};
    }

    int total_threads = batch_size * K;
    int grid = div_ceil(total_threads, BLOCK_SIZE);

    featurize_neighbors_kernel<<<grid, BLOCK_SIZE>>>(
        neighbor_nodes.data_ptr<int32_t>(),
        neighbor_eids.data_ptr<int64_t>(),
        lengths.data_ptr<int32_t>(),
        neighbor_ts.data_ptr<double>(),
        query_ts.data_ptr<double>(),
        node_feats.data_ptr<float>(),
        edge_feats.data_ptr<float>(),
        out_node_feats.data_ptr<float>(),
        out_edge_feats.data_ptr<float>(),
        out_rel_ts.data_ptr<double>(),
        batch_size,
        K,
        node_feat_dim,
        edge_feat_dim,
        num_total_edges
    );
    CUDA_CHECK(cudaGetLastError());

    return {out_node_feats, out_edge_feats, out_rel_ts};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated temporal neighbor sampling for temporal link prediction";

    py::class_<TemporalCSR_CUDA>(m, "TemporalCSR_CUDA")
        .def(py::init<int, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor>(),
             py::arg("num_nodes"), py::arg("src"), py::arg("dst"),
             py::arg("timestamps"), py::arg("edge_ids"),
             "Build GPU-resident CSR from CPU tensors. Sorts by (src, timestamp).")
        .def_readonly("num_nodes", &TemporalCSR_CUDA::num_nodes)
        .def_readonly("num_edges", &TemporalCSR_CUDA::num_edges);

    m.def("sample_neighbors_batch_cuda", &sample_neighbors_batch_cuda,
          py::arg("csr"), py::arg("query_nodes"),
          py::arg("query_timestamps"), py::arg("num_neighbors") = 20,
          "Batch temporal neighbor sampling on GPU.");

    m.def("featurize_neighbors_cuda", &featurize_neighbors_cuda,
          py::arg("neighbor_nodes"), py::arg("neighbor_eids"),
          py::arg("lengths"), py::arg("neighbor_ts"), py::arg("query_ts"),
          py::arg("node_feats"), py::arg("edge_feats"),
          "Batch feature gathering on GPU.");
}
