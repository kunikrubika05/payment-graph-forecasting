/**
 * CUDA extension for GPU-accelerated common neighbors via bitset intersection.
 *
 * Algorithm:
 *   For each node u, represent its neighbor set as a bitset of N bits
 *   packed into ceil(N/32) uint32 words. Then:
 *     CN(u, v) = popcount(bitset_u AND bitset_v)
 *
 *   For a batch of B pairs:
 *     1. build_bitsets_kernel  — one block per unique node; threads set bits.
 *     2. compute_cn_kernel     — one block per pair; threads AND + popcount.
 *
 * Complexity:
 *   Memory : O(M * W * 4) bytes, W = ceil(N/32), M = unique nodes in batch.
 *   Compute: O(M * W / BLOCK_SIZE + B * W / BLOCK_SIZE) kernel rounds.
 *
 * Efficiency:
 *   Dense graphs (avg degree >= 100): 30–200x over C++ sorted-merge.
 *   Sparse graphs (avg degree < 20) : C++ is faster due to kernel overhead.
 *
 * Build: python src/models/build_ext.py --graph-metrics-cuda
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        TORCH_CHECK(_e == cudaSuccess, "CUDA error: ",                         \
                    cudaGetErrorString(_e), " at ", __FILE__, ":", __LINE__);  \
    } while (0)

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_WARPS  = BLOCK_SIZE / 32;

/**
 * Build bitsets for M nodes in parallel.
 * Grid: (M,) blocks × BLOCK_SIZE threads.
 * Each block owns one node; threads cooperate to clear and then set bits.
 */
__global__ void build_bitsets_kernel(
    const int32_t* __restrict__ row_ptr,   // [N+1]
    const int32_t* __restrict__ col_idx,   // [E]
    const int32_t* __restrict__ node_ids,  // [M] sorted unique node IDs
    uint32_t*      __restrict__ bitsets,   // [M, W] output
    int M,
    int W
) {
    int m = blockIdx.x;
    if (m >= M) return;

    int       node = node_ids[m];
    uint32_t* bs   = bitsets + (int64_t)m * W;

    for (int w = threadIdx.x; w < W; w += blockDim.x)
        bs[w] = 0u;
    __syncthreads();

    int start = row_ptr[node];
    int end   = row_ptr[node + 1];
    for (int e = start + (int)threadIdx.x; e < end; e += blockDim.x) {
        int nb = col_idx[e];
        atomicOr(&bs[nb >> 5], 1u << (nb & 31));
    }
}

/**
 * Compute CN for B pairs using prebuilt bitsets.
 * Grid: (B,) blocks × BLOCK_SIZE threads.
 * Each block owns one pair; threads AND bitsets and accumulate popcount.
 */
__global__ void compute_cn_kernel(
    const uint32_t* __restrict__ bitsets,   // [M, W]
    const int32_t*  __restrict__ src_local, // [B] index of src in bitsets
    const int32_t*  __restrict__ dst_local, // [B] index of dst in bitsets
    int32_t*        __restrict__ cn_out,    // [B]
    int B,
    int W
) {
    int b = blockIdx.x;
    if (b >= B) return;

    const uint32_t* bu = bitsets + (int64_t)src_local[b] * W;
    const uint32_t* bv = bitsets + (int64_t)dst_local[b] * W;

    int cnt = 0;
    for (int w = threadIdx.x; w < W; w += blockDim.x)
        cnt += __popc(bu[w] & bv[w]);

    // Warp-level reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        cnt += __shfl_down_sync(0xffffffffu, cnt, mask);

    __shared__ int sdata[MAX_WARPS];
    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) sdata[warpId] = cnt;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total  = 0;
        int nwarps = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nwarps; ++i) total += sdata[i];
        cn_out[b] = total;
    }
}

} // namespace

/**
 * Compute common neighbors for a batch of pairs via GPU bitset intersection.
 *
 * The Python wrapper (graph_metrics.py) handles the global→local node
 * index mapping before calling this function.
 *
 * Args:
 *   row_ptr:      [N+1] int32 CSR row pointers (CPU or CUDA).
 *   col_idx:      [E]   int32 CSR column indices (CPU or CUDA).
 *   unique_nodes: [M]   int32 sorted unique node IDs in batch (CUDA).
 *   src_local:    [B]   int32 local index of src into unique_nodes (CUDA).
 *   dst_local:    [B]   int32 local index of dst into unique_nodes (CUDA).
 *   N:            total number of nodes (sets W = ceil(N/32)).
 *
 * Returns:
 *   [B] int32 tensor on CUDA with common neighbor counts.
 */
torch::Tensor common_neighbors_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor unique_nodes,
    torch::Tensor src_local,
    torch::Tensor dst_local,
    int N
) {
    auto rp = row_ptr.contiguous().to(torch::kCUDA).to(torch::kInt32);
    auto ci = col_idx.contiguous().to(torch::kCUDA).to(torch::kInt32);
    auto un = unique_nodes.contiguous().to(torch::kInt32);
    auto sl = src_local.contiguous().to(torch::kInt32);
    auto dl = dst_local.contiguous().to(torch::kInt32);

    int M = (int)un.size(0);
    int B = (int)sl.size(0);
    int W = (N + 31) / 32;

    int64_t bitset_bytes = (int64_t)M * W * 4;
    TORCH_CHECK(bitset_bytes <= 4LL * 1024 * 1024 * 1024,
                "Bitset buffer too large (", bitset_bytes / (1 << 20), " MB). "
                "Reduce batch size or use the C++ backend for N > 10M.");

    auto bitsets = torch::zeros(
        {(int64_t)M, W},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)
    );

    build_bitsets_kernel<<<M, BLOCK_SIZE>>>(
        rp.data_ptr<int32_t>(),
        ci.data_ptr<int32_t>(),
        un.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(bitsets.data_ptr<int32_t>()),
        M, W
    );
    CUDA_CHECK(cudaGetLastError());

    auto cn_out = torch::zeros(
        {B},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)
    );

    compute_cn_kernel<<<B, BLOCK_SIZE>>>(
        reinterpret_cast<const uint32_t*>(bitsets.data_ptr<int32_t>()),
        sl.data_ptr<int32_t>(),
        dl.data_ptr<int32_t>(),
        cn_out.data_ptr<int32_t>(),
        B, W
    );
    CUDA_CHECK(cudaGetLastError());

    return cn_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("common_neighbors_cuda", &common_neighbors_cuda,
          "Batch common neighbors via GPU bitset intersection (popcount)");
}
