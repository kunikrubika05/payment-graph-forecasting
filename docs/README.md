# Documentation

## Modules

### `src/build_pipeline.py`

Main data pipeline for the ORBITAAL dataset. Provides 5 idempotent steps:

| Step | Function | Description |
|------|----------|-------------|
| `download` | `step_download()` | Download tar.gz archives from Zenodo via wget |
| `extract` | `step_extract()` | Extract tar.gz archives |
| `mapping` | `step_mapping()` | Build global entity_id -> node_index mapping (dense 0..N-1) |
| `snapshots` | `step_snapshots()` | Convert raw snapshots to processed parquet with global indices |
| `upload` | `step_upload()` | Upload processed data to Yandex.Disk via REST API |

Key exported functions (used in tests):
- `_collect_entity_ids_from_csv(filepath)` — extract unique entity IDs from a CSV
- `_extract_date(filename)` — parse date from ORBITAAL filename patterns
- `_process_single_snapshot(filepath, entity_to_idx, output_dir, fmt)` — process one day
- `step_mapping(input_dir, output_dir, fmt, n_workers)` — build global mapping
- `step_snapshots(input_dir, output_dir, fmt, n_workers)` — build all daily snapshots

### `src/compute_features.py`

Computes graph-level and node-level features from daily snapshot parquets.

**Key functions:**
- `build_adjacency(src, dst, num_nodes)` — build compressed sparse adjacency
- `compute_pagerank(adj)` — PageRank via scipy power iteration
- `compute_clustering(adj)` — undirected clustering coefficient via sparse ops
- `compute_k_core(adj)` — Batagelj-Zaversnik peeling algorithm
- `compute_triangle_counts(adj)` — per-node triangle count via sparse A²·A
- `compute_node_features(df, adj, ...)` — all 26 node-level features for one day
- `compute_graph_features(df, date, adj, ..., node_features_df)` — all ~40 graph-level features
- `process_single_day(filepath)` — end-to-end for one snapshot
- `run_pipeline(input_dir, output_dir, upload, batch_size)` — main loop with progress bar
- `upload_batch_and_cleanup(file_paths, remote_dir, token)` — upload to Yandex.Disk + delete local

### `src/build_graphs.py`

Legacy prototype for building graphs from CSV samples. Provides `PaymentGraph` class
that converts raw DataFrames to PyTorch Geometric format. Uses per-graph local node
indexing (not global). Saved as pickle.

Superseded by `build_pipeline.py` for production use.

### `src/analyze.py`

EDA script that prints statistics for all CSV files in `data/samples/`:
entity counts, value distributions, degree distributions, self-loops, timestamps.

### `src/visualize.py`

Generates 6 plots from CSV sample data:
1. Degree distribution (log-log)
2. Hourly activity pattern
3. Transaction value distribution
4. Temporal transaction rate (30-min bins)
5. Entity overlap between days
6. Network subgraph (top-30 nodes)

## Data formats

See the main [README.md](../README.md) for detailed column descriptions of:
- `node_mapping.parquet`
- Daily snapshot parquet files
- `daily_stats.csv`
