"""
Visualize the adaptive gamma as a function of DCR and alpha_eff.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_gamma_surface(
    output_dir: str | Path,
    dcr_range: tuple[float, float] = (-1.0, 1.0),
    alpha_range: tuple[float, float] = (0.0, 1.5),
    gamma_lo: float = 0.3,
    gamma_hi: float = 0.7,
    n_points: int = 100,
) -> Path:
    """
    Plot gamma as a function of DCR (x-axis) and alpha_eff (color-coded lines).

    Args:
        output_dir:  Output directory.
        dcr_range:   Range of DCR values to display.
        alpha_range: Range of alpha_eff values to display (3 representative curves).
        gamma_lo:    Lower clip bound.
        gamma_hi:    Upper clip bound.
        n_points:    Number of points per curve.

    Returns:
        Path to saved PNG.
    """
    dcrs = np.linspace(*dcr_range, n_points)
    alpha_values = [0.1, 0.3, 0.6, 1.0, 1.5]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_values)))

    for alpha, color in zip(alpha_values, colors):
        gammas = np.clip(0.5 + (dcrs / 4.0) * (alpha / (1.0 + alpha)), gamma_lo, gamma_hi)
        ax.plot(dcrs, gammas, color=color, linewidth=1.8, label=f"α_eff={alpha:.1f}")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.fill_between(dcrs, gamma_lo, gamma_hi, alpha=0.05, color="blue")
    ax.set_xlabel("DCR (Spearman U vs T)", fontsize=11)
    ax.set_ylabel("γ (balance parameter)", fontsize=11)
    ax.set_title("Adaptive gamma: γ = clip(0.5 + DCR/4 · α_eff/(1+α_eff), 0.3, 0.7)", fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9, title="Effective budget", title_fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gamma_surface.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
