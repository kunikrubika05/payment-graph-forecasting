# Team Operations Note

## Purpose

This file stores team-specific operational context that should not be treated as
part of the abstract library API.

Anything here may be real and important for the team, but it is intentionally
separated from the canonical package-facing documentation so that
`payment_graph_forecasting` remains a reusable library surface.

## Current Team-Specific Storage Conventions

These are the paths and conventions currently used by the team infrastructure.

### Yandex.Disk roots

- shared processed-data link: `https://disk.yandex.ru/d/uJavr5EtMWj4Jg`
- experiments browser link: `https://disk.yandex.ru/client/disk/orbitaal_processed/experiments`
- main processed-data root: `orbitaal_processed`
- experiments root: `orbitaal_processed/experiments`
- stream-graph root: `orbitaal_processed/stream_graph`
- node-features root: `orbitaal_processed/node_features`

### Typical artifacts

- processed snapshots
- node features
- stream graph parquet files
- experiment outputs
- HPO outputs
- pairwise-MLP precompute artifacts

## Environment Conventions

### Upload token

- env var: `YADISK_TOKEN`

### Typical local scratch locations

These are conventions, not library requirements:

- `/tmp/graphmixer_results`
- `/tmp/eagle_results`
- `/tmp/glformer_results`
- `/tmp/hyperevent_results`
- `/tmp/sg_graphmixer_results`
- `/tmp/pairmlp_results`

## Library Boundary

The package-facing library should not hardcode team-specific remote roots into
its canonical execution paths.

Instead, team-specific storage should flow through explicit configuration:

- `upload.enabled`
- `upload.backend`
- `upload.remote_dir`
- `upload.token_env`

That keeps the functionality intact while preventing the library API from being
implicitly tied to one team's storage layout.

## Validation Reminder

When running the full functional validation protocol later, include at least one
real upload/download check against the team's storage setup:

1. create a temporary output artifact locally;
2. upload it through the package-facing API / configured upload path;
3. verify the file appears in the expected remote location;
4. verify download or listing access for the same location if needed.

This check belongs to the validation phase, not to the core refactor itself.

## Scope Note

This file is operational context, not a stable public API guarantee.
