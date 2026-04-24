"""Selection methods for CSCS-AL."""

from .cscs import select_cscs, select_cscs_from_df
from .gamma_scheduler import compute_dcr, compute_gamma, compute_alpha_eff
from .scoring import compute_score, rank_normalize
from .clustering import cluster_embeddings
from .registry import REGISTRY, get_selector, list_methods

__all__ = [
    "select_cscs",
    "select_cscs_from_df",
    "compute_dcr",
    "compute_gamma",
    "compute_alpha_eff",
    "compute_score",
    "rank_normalize",
    "cluster_embeddings",
    "REGISTRY",
    "get_selector",
    "list_methods",
]
