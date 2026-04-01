# Evaluation Protocols for Temporal Link Prediction

This note summarizes the ranking-oriented evaluation protocol used across the
repository for temporal link prediction on dynamic transaction graphs.

## Recommended Evaluation Shape

For a positive interaction `(src, dst_true, t)`:

- keep `src` and `t` fixed
- construct a candidate set `{dst_true} ∪ negatives`
- rank all candidates by model score
- compute the rank of `dst_true`

Primary metrics:

- `MRR`
- `Hits@1`
- `Hits@3`
- `Hits@10`

These metrics should be aggregated per query, then averaged.

## Negative Sampling

The maintained protocol uses a mixed negative set:

- historical negatives
- random negatives

The usual evaluation contract in this repository is:

- `n_negatives = 100`
- approximately `50/50` historical and random when possible

This follows the TGB-style large-graph ranking setup more closely than the old
single-random-negative binary classification protocol.

## Chronological Causality

All evaluation must preserve temporal causality:

- no future edges may appear in the information available for scoring an event
- train, validation, and test partitions must remain chronological

## Practical Interpretation

For this repository, the important rule is simple:

- if a model is compared against other supported temporal link prediction paths,
  it should be evaluated with the same per-query ranking protocol and the same
  candidate-generation policy

## Related Surfaces

- package launchers under `payment_graph_forecasting.experiments`
- evaluation wrappers under `payment_graph_forecasting.evaluation`
- negative sampling helpers under `payment_graph_forecasting.sampling`
