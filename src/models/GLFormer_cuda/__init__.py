"""GLFormer_cuda: GLFormer pipeline using CUDA-accelerated temporal sampling.

Same GLFormerTime model architecture as GLFormer/, but training and evaluation
use TemporalGraphSampler with the CUDA backend for neighbor sampling and
feature gathering. This eliminates the CPU bottleneck present in the
standard C++/Python pipeline.

For architecture details, see src/models/GLFormer/glformer.py.
For CUDA sampling details, see src/models/temporal_graph_sampler.py.
"""

from src.models.GLFormer.glformer import GLFormerTime

__all__ = ["GLFormerTime"]
