from __future__ import annotations

import numpy as np
import torch

from src.models.stream_graph_data import temporal_data_to_edge_data


class _TemporalDataStub:
    def __init__(self, src, dst, t, msg):
        self.src = torch.tensor(src, dtype=torch.long)
        self.dst = torch.tensor(dst, dtype=torch.long)
        self.t = torch.tensor(t, dtype=torch.long)
        self.msg = torch.tensor(msg, dtype=torch.float32)


def test_temporal_data_to_edge_data_remaps_sparse_node_ids_to_dense():
    temporal = _TemporalDataStub(
        src=[100, 300],
        dst=[200, 100],
        t=[1, 2],
        msg=[[1.0, 2.0], [3.0, 4.0]],
    )

    data = temporal_data_to_edge_data(temporal, undirected=False)

    assert data.num_nodes == 3
    assert data.src.tolist() == [0, 2]
    assert data.dst.tolist() == [1, 0]
    assert data.reverse_node_map.tolist() == [100, 200, 300]


def test_temporal_data_to_edge_data_reindexes_external_node_features():
    temporal = _TemporalDataStub(
        src=[10, 30],
        dst=[20, 10],
        t=[1, 2],
        msg=[[1.0, 2.0], [3.0, 4.0]],
    )
    external = np.zeros((31, 2), dtype=np.float32)
    external[10] = [1.0, 1.5]
    external[20] = [2.0, 2.5]
    external[30] = [3.0, 3.5]

    data = temporal_data_to_edge_data(
        temporal,
        undirected=False,
        external_node_feats=external,
    )

    assert data.node_feats.tolist() == [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]
