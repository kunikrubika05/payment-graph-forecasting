"""Pairwise MLP for temporal link prediction on stream graphs.

A small MLP trained with BPR loss over structural pairwise features
(CN, AA, Jaccard, PA — all from train adjacency only). Targets beating
the CN heuristic (test MRR ~ 0.64) with a minimal learned ranker.

Pipeline:
  1. precompute.py  — CPU script: download adj + stream graph, compute 7
                      pairwise features for train pos/neg edges, upload.
  2. run.py         — GPU script: download precomputed features, train
                      PairMLP with BPR loss, evaluate TGB-style, upload.
"""
