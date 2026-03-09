"""Legacy prototype for building payment graphs from CSV samples.

Provides the PaymentGraph class that converts raw ORBITAAL DataFrames
into a PyTorch Geometric-compatible format (edge_index, edge_attr).
Graphs are serialized as pickle files.

Note:
    For the full pipeline with global node indexing, use build_pipeline.py instead.

Usage:
    python src/build_graphs.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/samples"
GRAPH_DIR = ROOT / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)


class PaymentGraph:
    """A payment transaction graph in PyTorch Geometric-compatible format.

    Attributes:
        name: Human-readable identifier for this graph.
        edge_index: np.ndarray of shape (2, num_edges), int64.
        edge_attr: np.ndarray of shape (num_edges, 2) with [btc, usd] per edge.
        node_mapping: Dict mapping original entity_id to dense node index.
        reverse_mapping: Dict mapping dense node index back to entity_id.
        num_nodes: Total number of unique entities.
        num_edges: Total number of edges.
        timestamps: Optional np.ndarray of UNIX timestamps per edge.
    """

    def __init__(self, name):
        self.name = name
        self.edge_index = None
        self.edge_attr = None
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.num_nodes = 0
        self.num_edges = 0
        self.timestamps = None

    def from_dataframe(self, df, remove_self_loops=True, remove_entity_zero=True):
        """Build graph from a raw ORBITAAL DataFrame.

        Args:
            df: DataFrame with columns SRC_ID, DST_ID, VALUE_SATOSHI, VALUE_USD,
                and optionally TIMESTAMP.
            remove_self_loops: If True, remove edges where SRC_ID == DST_ID.
            remove_entity_zero: If True, remove edges involving entity 0 (coinbase).

        Returns:
            self, for method chaining.
        """
        df = df.copy()
        if remove_entity_zero:
            df = df[(df['SRC_ID'] != 0) & (df['DST_ID'] != 0)]
        if remove_self_loops:
            df = df[df['SRC_ID'] != df['DST_ID']]

        all_nodes = pd.concat([df['SRC_ID'], df['DST_ID']]).unique()
        self.node_mapping = {old: new for new, old in enumerate(all_nodes)}
        self.reverse_mapping = {new: old for old, new in self.node_mapping.items()}
        self.num_nodes = len(all_nodes)

        src = df['SRC_ID'].map(self.node_mapping).values
        dst = df['DST_ID'].map(self.node_mapping).values
        self.edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        self.num_edges = self.edge_index.shape[1]

        btc = (df['VALUE_SATOSHI'].values / 1e8).astype(np.float32)
        usd = df['VALUE_USD'].values.astype(np.float32)
        self.edge_attr = np.stack([btc, usd], axis=1)

        if 'TIMESTAMP' in df.columns:
            self.timestamps = df['TIMESTAMP'].values.astype(np.int64)

        return self

    def to_pyg_format(self):
        """Convert to PyTorch Geometric tensors.

        Returns:
            Dict with 'edge_index', 'edge_attr', 'num_nodes', and optionally 'timestamps'.
            Falls back to numpy arrays if torch is not installed.
        """
        try:
            import torch
            data = {
                'edge_index': torch.tensor(self.edge_index, dtype=torch.long),
                'edge_attr': torch.tensor(self.edge_attr, dtype=torch.float),
                'num_nodes': self.num_nodes
            }
            if self.timestamps is not None:
                data['timestamps'] = torch.tensor(self.timestamps, dtype=torch.long)
            return data
        except ImportError:
            print("PyTorch not installed, returning numpy arrays")
            return self.to_dict()

    def to_dict(self):
        """Serialize graph data to a plain dict of numpy arrays."""
        return {
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'timestamps': self.timestamps,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'node_mapping': self.node_mapping
        }

    def save(self, path):
        """Save graph as a pickle file.

        Args:
            path: Output file path.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)
        print(f"Saved: {path}")

    def stats(self):
        """Print summary statistics for this graph."""
        print(f"\n{'='*50}")
        print(f"Graph: {self.name}")
        print(f"{'='*50}")
        print(f"Nodes: {self.num_nodes:,}")
        print(f"Edges: {self.num_edges:,}")
        if self.num_nodes > 1:
            print(f"Density: {self.num_edges / (self.num_nodes * (self.num_nodes - 1)):.2e}")
        else:
            print(f"Density: N/A (fewer than 2 nodes)")
        print(f"Edge attr shape: {self.edge_attr.shape}")
        print(f"BTC range: [{self.edge_attr[:,0].min():.6f}, {self.edge_attr[:,0].max():.2f}]")
        if self.timestamps is not None:
            print(f"Has timestamps: Yes ({len(self.timestamps):,})")
        else:
            print(f"Has timestamps: No (snapshot)")


files = {
    'stream_08': DATA_DIR / "orbitaal-stream_graph-2016_07_08.csv",
    'stream_09': DATA_DIR / "orbitaal-stream_graph-2016_07_09.csv",
    'snapshot_08': DATA_DIR / "orbitaal-snapshot-2016_07_08.csv",
    'snapshot_09': DATA_DIR / "orbitaal-snapshot-2016_07_09.csv"
}

graphs = {}
for name, path in files.items():
    print(f"\nBuilding {name}...")
    df = pd.read_csv(path)
    g = PaymentGraph(name).from_dataframe(df)
    g.stats()
    g.save(GRAPH_DIR / f"{name}.pkl")
    graphs[name] = g

print(f"\n{'='*50}")
print("All graphs saved to graphs/")
print(f"{'='*50}")
