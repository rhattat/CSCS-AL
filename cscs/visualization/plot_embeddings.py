"""
2-D embedding space visualization (UMAP or t-SNE fallback).

Shows uncertainty and typicality overlaid on the SSL embedding space,
useful for understanding dataset structure and DCR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_embedding_space(
    embeddings: np.ndarray,
    volume_ids: list[str],
    uncertainty_norm: np.ndarray,
    typicality_norm: np.ndarray,
    dcr: float,
    output_dir: str | Path,
    dataset_name: str = "",
    annotate_top_n: int = 5,
) -> Path:
    """
    Project embeddings to 2-D (UMAP or t-SNE) and color by U and T.

    Args:
        embeddings:       (N, d) feature matrix.
        volume_ids:       List of N volume IDs.
        uncertainty_norm: Normalized uncertainty scores (N,).
        typicality_norm:  Normalized typicality scores (N,).
        dcr:              DCR value for annotation.
        output_dir:       Output directory for the figure.
        dataset_name:     Dataset name for figure title.
        annotate_top_n:   Number of top-scoring samples to annotate.

    Returns:
        Path to saved PNG.
    """
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=min(15, len(embeddings) - 1))
        emb_2d = reducer.fit_transform(embeddings)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42,
                       perplexity=min(30, len(embeddings) - 1))
        emb_2d = reducer.fit_transform(embeddings)
        method = "t-SNE"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title = f"{dataset_name} SSL Embedding Space ({method})" if dataset_name else f"SSL Embedding Space ({method})"
    fig.suptitle(f"{title}\nDCR = {dcr:+.3f}", fontsize=13)

    for ax, scores, label, cmap in zip(
        axes,
        [uncertainty_norm, typicality_norm],
        ["Uncertainty U(x)", "Typicality T(x)"],
        ["Reds", "Blues"],
    ):
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=scores, cmap=cmap,
                        s=20, alpha=0.8, edgecolors="k", linewidths=0.2)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        top_idx = np.argsort(scores)[-annotate_top_n:]
        for idx in top_idx:
            ax.annotate(volume_ids[idx], (emb_2d[idx, 0], emb_2d[idx, 1]),
                        fontsize=5, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embedding_space.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
