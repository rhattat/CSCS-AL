"""
Method registry: maps method names to selector functions.

All selectors have the unified signature:
    selector(volume_ids, U, T, budget, embeddings=None, seed=42, **kwargs)
    → (selection_df, metadata)
"""

from __future__ import annotations

from .cscs import select_cscs_from_df
from .baselines import (
    select_random,
    select_fps,
    select_typiclust,
    select_probcover,
    select_csal3d,
)


def _cscs_wrapper(volume_ids, U, T, budget, embeddings=None, seed=42, **kwargs):
    if embeddings is None:
        from .clustering import fallback_embedding
        embeddings = fallback_embedding(len(volume_ids), U, T)
    return select_cscs_from_df(volume_ids, U, T, embeddings, budget, seed=seed, **kwargs)


REGISTRY: dict[str, callable] = {
    "cscs":       _cscs_wrapper,
    "random":     select_random,
    "fps":        select_fps,
    "typiclust":  select_typiclust,
    "probcover":  select_probcover,
    "csal3d":     select_csal3d,
}


def list_methods() -> list[str]:
    """Return sorted list of registered method names."""
    return sorted(REGISTRY.keys())


def get_selector(method: str):
    """Return the selector function for a given method name."""
    if method not in REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available: {list_methods()}"
        )
    return REGISTRY[method]
