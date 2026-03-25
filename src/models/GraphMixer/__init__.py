"""GraphMixer model for temporal link prediction on stream graphs.

Adapted from the original GraphMixer (Cong et al., ICLR 2023) to work
with the stream graph parquet format used by EAGLE and GLFormer.
"""

from src.models.GraphMixer.graphmixer import GraphMixerTime

__all__ = ["GraphMixerTime"]
