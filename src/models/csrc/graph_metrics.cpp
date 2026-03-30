/**
 * C++ extension for common neighbors computation (CPU backend).
 *
 * Uses sorted-merge on CSR adjacency lists: O(d_u + d_v) per pair.
 * Neighbor lists within each row must be sorted (standard CSR).
 *
 * Build: python src/models/build_ext.py --graph-metrics
 */

#include <torch/extension.h>
#include <cstdint>

namespace {

inline int32_t cn_merge(
    const int32_t* col,
    int32_t su, int32_t eu,
    int32_t sv, int32_t ev
) {
    int32_t cnt = 0;
    int32_t i = su, j = sv;
    while (i < eu && j < ev) {
        if      (col[i] == col[j]) { ++cnt; ++i; ++j; }
        else if (col[i] <  col[j]) { ++i; }
        else                       { ++j; }
    }
    return cnt;
}

} // namespace

/**
 * Compute common neighbors for a batch of node pairs via C++ sorted merge.
 *
 * Args:
 *   row_ptr: [N+1] int32 CSR row pointers.
 *   col_idx: [E]   int32 CSR column indices (sorted within each row).
 *   src:     [B]   int64 source node indices.
 *   dst:     [B]   int64 destination node indices.
 *
 * Returns:
 *   [B] int32 common neighbor counts.
 */
torch::Tensor common_neighbors_cpp(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor src,
    torch::Tensor dst
) {
    TORCH_CHECK(row_ptr.dtype() == torch::kInt32, "row_ptr must be int32");
    TORCH_CHECK(col_idx.dtype() == torch::kInt32, "col_idx must be int32");
    TORCH_CHECK(src.dtype() == torch::kInt64,     "src must be int64");
    TORCH_CHECK(dst.dtype() == torch::kInt64,     "dst must be int64");

    const auto* rp = row_ptr.contiguous().data_ptr<int32_t>();
    const auto* ci = col_idx.contiguous().data_ptr<int32_t>();
    const auto* sb = src.contiguous().data_ptr<int64_t>();
    const auto* db = dst.contiguous().data_ptr<int64_t>();

    int B = (int)src.size(0);
    auto result = torch::zeros({B}, torch::kInt32);
    auto* res   = result.data_ptr<int32_t>();

    for (int i = 0; i < B; ++i) {
        int32_t u = (int32_t)sb[i];
        int32_t v = (int32_t)db[i];
        res[i] = cn_merge(ci, rp[u], rp[u + 1], rp[v], rp[v + 1]);
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("common_neighbors_cpp", &common_neighbors_cpp,
          "Batch common neighbors via C++ sorted list merge");
}
