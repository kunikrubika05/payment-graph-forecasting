# Documentation

## Current package-facing surface

The active library-facing API is `payment_graph_forecasting.*`.

Key package areas:

| Package | Purpose |
| --- | --- |
| `payment_graph_forecasting.config` | Typed experiment specs and YAML loading |
| `payment_graph_forecasting.models` | Library model exports and adapters |
| `payment_graph_forecasting.training` | Stable training wrappers |
| `payment_graph_forecasting.evaluation` | Stable evaluation wrappers |
| `payment_graph_forecasting.experiments` | Unified launcher and library runners |
| `payment_graph_forecasting.infra` | Runtime / device / extension-build / upload infrastructure |

Currently supported library model variants:

- `graphmixer`
- `sg_graphmixer`
- `eagle`
- `glformer`
- `hyperevent`
- `pairwise_mlp`

Reference YAML specs live in `exps/examples/`.

Current package-facing example specs:

- `graphmixer_library.yaml`
- `sg_graphmixer_library.yaml`
- `eagle_library.yaml`
- `glformer_library.yaml`
- `hyperevent_library.yaml`
- `pairwise_mlp_library.yaml`

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

### `src/baselines/` — Baseline experiments pipeline

Pipeline for training and evaluating link prediction, graph-level forecasting,
and heuristic baselines. Uses TGB-style per-source ranking evaluation.

| Module | Description |
|--------|-------------|
| `config.py` | `ExperimentConfig` dataclass, period definitions, HP grids |
| `data_loader.py` | Download/load node features and daily snapshots from Yandex.Disk |
| `feature_engineering.py` | Mean/time-weighted feature aggregation, pair feature construction (float32) |
| `evaluation.py` | Ranking metrics (MRR, Hits@K), time series metrics (MAE, RMSE, MAPE, sMAPE) |
| `experiment_logger.py` | Logging: config, metrics, models, predictions, upload to Yandex.Disk |
| `link_prediction.py` | Link prediction pipeline: Mode A (single train) + Mode B (live-update retrain) |
| `graph_forecasting.py` | ARIMA, SARIMAX, Holt-Winters, Prophet on graph-level time series |
| `heuristic_baselines.py` | Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment |
| `runner.py` | Queue-based experiment runner with resume and error handling |
| `launcher.py` | Generates 35 experiment configs, distributes across tmux sessions |

**Evaluation protocol** (see also `evaluation_protocols_temporal_lp_ru.md`):
- Per-source ranking: for each positive edge (s, d), fix s, build candidate set
  {d_true} ∪ {neg_1, ..., neg_q}, rank candidates by model score.
- n_negatives=100, 50/50 historical+random negative mix.
- Metrics: filtered MRR (primary), Hits@1, Hits@3, Hits@10.
- HP search uses PR-AUC on validation set (binary classification metric).

**Key design decisions:**
- Pair features use float32 to reduce memory (50% vs float64).
- Evaluation batched (EVAL_BATCH_SIZE=1000) to avoid OOM on heavy periods.
- Per-source batch negative sampling in training (not per-edge) for efficiency.
- Mode B limited to lightweight periods (mid_2015q3) due to 24h compute budget.

## Data formats

See the main [README.md](../README.md) for detailed column descriptions of:
- `node_mapping.parquet`
- Daily snapshot parquet files
- `daily_stats.csv`

## Refactoring notes

Useful notes for the current refactor:

- [design/sg_graphmixer_vs_graphmixer.md](design/sg_graphmixer_vs_graphmixer.md)
- [design/tooling_migration_status.md](design/tooling_migration_status.md)
- [design/refactoring_completion_plan.md](design/refactoring_completion_plan.md)
- [design/legacy_surface_matrix.md](design/legacy_surface_matrix.md)
- [experiments/cuda_exp_graphmixer_a10.md](experiments/cuda_exp_graphmixer_a10.md)

Team-specific operational context is kept separately in
[TEAM.md](../TEAM.md) so the library-facing docs stay abstract.

Useful package-facing CLIs:

- `python -m payment_graph_forecasting.experiments.launcher`
- `python -m payment_graph_forecasting.experiments.hpo`
- `python -m payment_graph_forecasting.infra.extensions`
- `docs/design/dev_machine_validation_protocol.md`
