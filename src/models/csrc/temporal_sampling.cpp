/**
 * C++ extension for accelerated temporal neighbor sampling.
 *
 * Replaces Python loops in TemporalCSR, sample_neighbors_batch, and
 * featurize_neighbors with optimized C++ implementations.
 * Provides ~3-5x speedup over pure Python/NumPy.
 *
 * Build: python src/models/build_ext.py
 * Used automatically by data_utils.py when compiled.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace py = pybind11;

class TemporalCSR {
   public:
    int num_nodes;

    TemporalCSR(int num_nodes, py::array_t<int32_t> src,
                py::array_t<int32_t> dst, py::array_t<double> ts,
                py::array_t<int64_t> eids)
        : num_nodes(num_nodes) {
        auto src_buf = src.request();
        auto dst_buf = dst.request();
        auto ts_buf = ts.request();
        auto eids_buf = eids.request();

        int64_t n_edges = src_buf.shape[0];

        const int32_t* src_ptr = static_cast<const int32_t*>(src_buf.ptr);
        const int32_t* dst_ptr = static_cast<const int32_t*>(dst_buf.ptr);
        const double* ts_ptr = static_cast<const double*>(ts_buf.ptr);
        const int64_t* eids_ptr = static_cast<const int64_t*>(eids_buf.ptr);

        std::vector<int64_t> sort_idx(n_edges);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::sort(sort_idx.begin(), sort_idx.end(),
                  [&](int64_t a, int64_t b) {
                      if (src_ptr[a] != src_ptr[b])
                          return src_ptr[a] < src_ptr[b];
                      return ts_ptr[a] < ts_ptr[b];
                  });

        neighbors_.resize(n_edges);
        timestamps_.resize(n_edges);
        edge_ids_.resize(n_edges);
        indptr_.resize(num_nodes + 1, 0);

        for (int64_t i = 0; i < n_edges; i++) {
            int64_t idx = sort_idx[i];
            neighbors_[i] = dst_ptr[idx];
            timestamps_[i] = ts_ptr[idx];
            edge_ids_[i] = eids_ptr[idx];
            indptr_[src_ptr[idx] + 1]++;
        }

        for (int i = 1; i <= num_nodes; i++) {
            indptr_[i] += indptr_[i - 1];
        }
    }

    py::tuple get_temporal_neighbors(int node, double before_time,
                                     int k) const {
        if (node < 0 || node >= num_nodes) {
            return py::make_tuple(py::array_t<int32_t>(0),
                                  py::array_t<double>(0),
                                  py::array_t<int64_t>(0));
        }

        int64_t start = indptr_[node];
        int64_t end = indptr_[node + 1];

        if (start == end) {
            return py::make_tuple(py::array_t<int32_t>(0),
                                  py::array_t<double>(0),
                                  py::array_t<int64_t>(0));
        }

        auto valid_end_it = std::lower_bound(timestamps_.begin() + start,
                                             timestamps_.begin() + end,
                                             before_time);
        int64_t valid_end = start + (valid_end_it - (timestamps_.begin() + start));

        if (valid_end == start) {
            return py::make_tuple(py::array_t<int32_t>(0),
                                  py::array_t<double>(0),
                                  py::array_t<int64_t>(0));
        }

        int64_t actual_start =
            std::max(start, valid_end - static_cast<int64_t>(k));
        int64_t length = valid_end - actual_start;

        auto nids = py::array_t<int32_t>(length);
        auto nts = py::array_t<double>(length);
        auto neids = py::array_t<int64_t>(length);

        std::memcpy(nids.mutable_data(), neighbors_.data() + actual_start,
                     length * sizeof(int32_t));
        std::memcpy(nts.mutable_data(), timestamps_.data() + actual_start,
                     length * sizeof(double));
        std::memcpy(neids.mutable_data(), edge_ids_.data() + actual_start,
                     length * sizeof(int64_t));

        return py::make_tuple(nids, nts, neids);
    }

    const std::vector<int64_t>& indptr() const { return indptr_; }
    const std::vector<int32_t>& neighbors() const { return neighbors_; }
    const std::vector<double>& timestamps() const { return timestamps_; }
    const std::vector<int64_t>& edge_ids() const { return edge_ids_; }

   private:
    std::vector<int64_t> indptr_;
    std::vector<int32_t> neighbors_;
    std::vector<double> timestamps_;
    std::vector<int64_t> edge_ids_;
};

py::tuple sample_neighbors_batch(const TemporalCSR& csr,
                                 py::array_t<int32_t> nodes,
                                 py::array_t<double> query_timestamps,
                                 int num_neighbors) {
    auto nodes_buf = nodes.request();
    auto ts_buf = query_timestamps.request();
    int64_t batch_size = nodes_buf.shape[0];

    const int32_t* nodes_ptr = static_cast<const int32_t*>(nodes_buf.ptr);
    const double* ts_ptr = static_cast<const double*>(ts_buf.ptr);

    auto nn =
        py::array_t<int32_t>({batch_size, static_cast<int64_t>(num_neighbors)});
    auto nts =
        py::array_t<double>({batch_size, static_cast<int64_t>(num_neighbors)});
    auto neids =
        py::array_t<int64_t>({batch_size, static_cast<int64_t>(num_neighbors)});
    auto lengths = py::array_t<int32_t>(batch_size);

    int32_t* nn_ptr = nn.mutable_data();
    double* nts_ptr = nts.mutable_data();
    int64_t* neids_ptr = neids.mutable_data();
    int32_t* len_ptr = lengths.mutable_data();

    std::memset(nn_ptr, 0xFF,
                batch_size * num_neighbors * sizeof(int32_t));
    std::memset(nts_ptr, 0,
                batch_size * num_neighbors * sizeof(double));
    std::memset(neids_ptr, 0xFF,
                batch_size * num_neighbors * sizeof(int64_t));
    std::memset(len_ptr, 0, batch_size * sizeof(int32_t));

    const auto& indptr = csr.indptr();
    const auto& neighbors = csr.neighbors();
    const auto& timestamps = csr.timestamps();
    const auto& edge_ids = csr.edge_ids();

    for (int64_t i = 0; i < batch_size; i++) {
        int node = nodes_ptr[i];
        double before_time = ts_ptr[i];

        if (node < 0 || node >= csr.num_nodes) continue;

        int64_t start = indptr[node];
        int64_t end = indptr[node + 1];
        if (start == end) continue;

        auto valid_end_it = std::lower_bound(
            timestamps.begin() + start, timestamps.begin() + end, before_time);
        int64_t valid_end =
            start + (valid_end_it - (timestamps.begin() + start));
        if (valid_end == start) continue;

        int64_t actual_start =
            std::max(start, valid_end - static_cast<int64_t>(num_neighbors));
        int32_t length = static_cast<int32_t>(valid_end - actual_start);
        len_ptr[i] = length;

        int64_t row_offset = i * num_neighbors;
        std::memcpy(nn_ptr + row_offset, neighbors.data() + actual_start,
                     length * sizeof(int32_t));
        std::memcpy(nts_ptr + row_offset, timestamps.data() + actual_start,
                     length * sizeof(double));
        std::memcpy(neids_ptr + row_offset, edge_ids.data() + actual_start,
                     length * sizeof(int64_t));
    }

    return py::make_tuple(nn, nts, neids, lengths);
}

py::tuple featurize_neighbors(py::array_t<int32_t> neighbor_nodes,
                              py::array_t<int64_t> neighbor_eids,
                              py::array_t<int32_t> lengths,
                              py::array_t<double> neighbor_ts,
                              py::array_t<double> query_ts,
                              py::array_t<float> node_feats,
                              py::array_t<float> edge_feats) {
    auto nn_r = neighbor_nodes.unchecked<2>();
    auto ne_r = neighbor_eids.unchecked<2>();
    auto len_r = lengths.unchecked<1>();
    auto nts_r = neighbor_ts.unchecked<2>();
    auto qts_r = query_ts.unchecked<1>();
    auto nf_r = node_feats.unchecked<2>();
    auto ef_r = edge_feats.unchecked<2>();

    int64_t batch_size = nn_r.shape(0);
    int64_t K = nn_r.shape(1);
    int64_t node_feat_dim = nf_r.shape(1);
    int64_t edge_feat_dim = ef_r.shape(1);

    auto out_nnf = py::array_t<float>({batch_size, K, node_feat_dim});
    auto out_nef = py::array_t<float>({batch_size, K, edge_feat_dim});
    auto out_nrt = py::array_t<double>({batch_size, K});

    float* nnf_ptr = out_nnf.mutable_data();
    float* nef_ptr = out_nef.mutable_data();
    double* nrt_ptr = out_nrt.mutable_data();

    std::memset(nnf_ptr, 0, batch_size * K * node_feat_dim * sizeof(float));
    std::memset(nef_ptr, 0, batch_size * K * edge_feat_dim * sizeof(float));
    std::memset(nrt_ptr, 0, batch_size * K * sizeof(double));

    const float* nf_data = node_feats.data();
    const float* ef_data = edge_feats.data();
    int64_t num_total_edges = edge_feats.shape()[0];

    for (int64_t i = 0; i < batch_size; i++) {
        int32_t length = len_r(i);
        if (length == 0) continue;

        double qt = qts_r(i);
        int64_t nnf_row = i * K * node_feat_dim;
        int64_t nef_row = i * K * edge_feat_dim;
        int64_t nrt_row = i * K;

        for (int32_t j = 0; j < length; j++) {
            int32_t nid = nn_r(i, j);
            int64_t eid = ne_r(i, j);

            std::memcpy(nnf_ptr + nnf_row + j * node_feat_dim,
                         nf_data + static_cast<int64_t>(nid) * node_feat_dim,
                         node_feat_dim * sizeof(float));

            if (eid >= 0 && eid < num_total_edges) {
                std::memcpy(nef_ptr + nef_row + j * edge_feat_dim,
                             ef_data + eid * edge_feat_dim,
                             edge_feat_dim * sizeof(float));
            }

            nrt_ptr[nrt_row + j] = qt - nts_r(i, j);
        }
    }

    return py::make_tuple(out_nnf, out_nef, out_nrt);
}

PYBIND11_MODULE(temporal_sampling_cpp, m) {
    m.doc() = "C++ accelerated temporal neighbor sampling for GraphMixer";

    py::class_<TemporalCSR>(m, "TemporalCSR")
        .def(py::init<int, py::array_t<int32_t>, py::array_t<int32_t>,
                       py::array_t<double>, py::array_t<int64_t>>(),
             py::arg("num_nodes"), py::arg("src"), py::arg("dst"),
             py::arg("timestamps"), py::arg("edge_ids"))
        .def("get_temporal_neighbors", &TemporalCSR::get_temporal_neighbors,
             py::arg("node"), py::arg("before_time"), py::arg("k") = 20)
        .def_readonly("num_nodes", &TemporalCSR::num_nodes);

    m.def("sample_neighbors_batch", &sample_neighbors_batch, py::arg("csr"),
          py::arg("nodes"), py::arg("timestamps"),
          py::arg("num_neighbors") = 20);

    m.def("featurize_neighbors", &featurize_neighbors,
          py::arg("neighbor_nodes"), py::arg("neighbor_eids"),
          py::arg("lengths"), py::arg("neighbor_ts"), py::arg("query_ts"),
          py::arg("node_feats"), py::arg("edge_feats"));
}
