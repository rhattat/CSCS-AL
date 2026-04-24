"""Feature computation and I/O for CSCS-AL."""

from .typicality import compute_typicality, compute_typicality_faiss
from .normalization import minmax_normalize, zscore_normalize
from .io import load_features, load_embeddings_dir, save_features, normalize_volume_id

__all__ = [
    "compute_typicality",
    "compute_typicality_faiss",
    "minmax_normalize",
    "zscore_normalize",
    "load_features",
    "load_embeddings_dir",
    "save_features",
    "normalize_volume_id",
]
