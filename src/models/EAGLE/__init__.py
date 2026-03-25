"""EAGLE-Time model for temporal link prediction on stream graphs.

Modules:
    eagle          — EAGLETime model architecture (MLP-Mixer on temporal delta times)
    eagle_train    — Training loop and validation
    eagle_evaluate — TGB-style evaluation
    eagle_launcher — CLI entry point for running experiments
    eagle_hpo      — Optuna hyperparameter search
    tppr           — Temporal Personalized PageRank (structure scoring)
    data_utils     — Stream graph loading and TemporalEdgeData construction
"""

from src.models.EAGLE.eagle import EAGLETime

__all__ = ["EAGLETime"]
