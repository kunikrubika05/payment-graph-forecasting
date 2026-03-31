# `sg_graphmixer` vs current `graphmixer`

## Short Answer

They belong to the same GraphMixer family, but they are not equivalent
execution paths today.

`sg_graphmixer` is a stream-graph / `sg_baselines`-aligned variant with its own
data regime, negative-sampling regime, node-universe assumptions, and exact
evaluation semantics. The current library-backed `graphmixer` is a different
GraphMixer path built around the newer unified temporal model contract.

## Shared Core

Both variants:

- implement GraphMixer-style temporal link prediction;
- rely on temporal neighborhood sampling;
- use the same broad encoder/classifier design family;
- target stream-graph link prediction.

## Actual Differences

### Model API

- `sg_graphmixer` uses `src/models/graphmixer.py::GraphMixer`
- current library `graphmixer` uses `src/models/GraphMixer/graphmixer.py::GraphMixerTime`

The newer `GraphMixerTime` surface is aligned with the unified
`encode_nodes(...)` / `edge_predictor(...)` contract used by other migrated
models.

### Data Regime

`sg_graphmixer` is tied to the `sg_baselines` data world:

- period-based truncation via `sg_baselines.config.PERIODS`;
- `sg_baselines.data` loaders and cached Yandex.Disk artifacts;
- train-node mapping and sparse feature/adjaсency artifacts;
- explicit dense remapping of active train nodes only.

The current library `graphmixer` uses the newer package-facing stream-graph
loaders and does not assume the same `sg_baselines` artifact contract.

### Training Semantics

`sg_graphmixer` training:

- uses exactly 1 negative per positive edge;
- mixes hard negatives with roughly 50% historical and 50% random;
- samples negatives from train nodes only;
- optimizes toward parity with the `sg_baselines` regime.

The current library `graphmixer` training path uses a different training setup
and a different model wrapper.

### Evaluation Semantics

`sg_graphmixer` evaluation is designed to match `sg_baselines` exactly:

- only evaluates edges whose endpoints are in the train node universe;
- uses `sg_baselines.sampling.sample_negatives_for_eval`;
- uses conservative ranking;
- uses exact seed offsets and query subsampling rules from that workflow.

The current library `graphmixer` follows the newer unified TGB-style evaluation
path and does not preserve all of those semantics.

## Refactoring Implication

At refactor time, `sg_graphmixer` should be treated as a distinct GraphMixer
variant, not as a cosmetic wrapper around the current library `graphmixer`.

That is why it is being introduced as a separate library-facing variant first.
If it remains the production/default GraphMixer path, a later migration step
can invert the naming:

- canonical `graphmixer` -> current `sg_graphmixer` semantics
- old library `graphmixer` -> legacy variant / shim

## Recommended Near-Term Direction

1. Keep both paths explicit.
2. Treat `sg_graphmixer` as the production-oriented GraphMixer variant.
3. Avoid silently changing the meaning of current `graphmixer` until parity and
   migration impact are understood.
