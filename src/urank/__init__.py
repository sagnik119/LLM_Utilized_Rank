"""
Utilized Rank LLM - Core library for utilized rank analysis and compression.

This package provides modular utilities for:
- Layer-wise utilized-rank estimation from data-driven subspaces
- Mixed-rank compression with energy preservation
- Binary search for optimal ranks with validation constraints
- Weight transformation and factorization
"""

from .instrumentation import iter_linear_layers, replace_linear
from .activation_cache import ActivationStatsCollector
from .subspace import topk_from_xtx, projector_from_vecs
from .transform import transform_weight_ps_pt, factorize_rank
from .rank_search import RankSearcher

__version__ = "0.1.0"

__all__ = [
    "iter_linear_layers",
    "replace_linear",
    "ActivationStatsCollector",
    "topk_from_xtx",
    "projector_from_vecs",
    "transform_weight_ps_pt",
    "factorize_rank",
    "RankSearcher",
]