"""
Adaptive gamma scheduler for CSCS.

Formula (from paper):
    gamma = clip(0.5 + (DCR / 4) * (alpha_eff / (1 + alpha_eff)), gamma_lo, gamma_hi)
    alpha_eff = B / sqrt(N)
    DCR = Spearman(U, T)

Interpretation:
    - DCR > 0  : uncertain samples tend to be atypical → increase uncertainty weight (gamma > 0.5)
    - DCR < 0  : uncertain samples tend to be typical  → keep typicality weight (gamma < 0.5)
    - gamma is clipped to [0.3, 0.7] to avoid extreme strategies
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_dcr(U: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """
    Compute DCR (Dataset Characterization Ratio).

    DCR = Spearman rank correlation between uncertainty U and typicality T.
    A positive DCR means difficult (uncertain) samples tend to be atypical.

    Args:
        U: Uncertainty scores, shape (N,)
        T: Typicality scores, shape (N,)

    Returns:
        dcr:  Spearman correlation in [-1, 1]
        pval: Two-sided p-value
    """
    U = np.asarray(U, dtype=float)
    T = np.asarray(T, dtype=float)
    if len(U) < 3:
        return 0.0, 1.0
    dcr, pval = spearmanr(U, T)
    return float(dcr), float(pval)


def compute_alpha_eff(budget: int, n_pool: int) -> float:
    """
    Effective budget ratio: alpha_eff = B / sqrt(N).

    Normalises the raw budget ratio B/N by sqrt(N) so that datasets of
    different sizes produce comparable gamma values.

    Args:
        budget: Number of samples to select (B)
        n_pool: Pool size (N)

    Returns:
        alpha_eff >= 0
    """
    if n_pool <= 0:
        raise ValueError(f"n_pool must be positive, got {n_pool}")
    return budget / np.sqrt(n_pool)


def compute_gamma(
    budget: int,
    n_pool: int,
    dcr: float,
    gamma_lo: float = 0.3,
    gamma_hi: float = 0.7,
) -> tuple[float, float]:
    """
    Compute adaptive gamma and alpha_eff.

    gamma = clip(0.5 + (DCR / 4) * (alpha_eff / (1 + alpha_eff)), gamma_lo, gamma_hi)

    Args:
        budget:   Number of samples to select (B)
        n_pool:   Unlabeled pool size (N)
        dcr:      DCR value (Spearman correlation U vs T)
        gamma_lo: Lower clip bound (default 0.3)
        gamma_hi: Upper clip bound (default 0.7)

    Returns:
        gamma:     Clipped gamma in [gamma_lo, gamma_hi]
        alpha_eff: Effective budget ratio B / sqrt(N)
    """
    alpha_eff = compute_alpha_eff(budget, n_pool)
    gamma_raw = 0.5 + (dcr / 4.0) * (alpha_eff / (1.0 + alpha_eff))
    gamma = float(np.clip(gamma_raw, gamma_lo, gamma_hi))
    return gamma, alpha_eff


def gamma_summary(budget: int, n_pool: int, dcr: float,
                  gamma_lo: float = 0.3, gamma_hi: float = 0.7) -> dict:
    """Return a dict with all gamma-related quantities (useful for logging/output)."""
    gamma, alpha_eff = compute_gamma(budget, n_pool, dcr, gamma_lo, gamma_hi)
    return {
        "dcr":       dcr,
        "alpha_eff": alpha_eff,
        "gamma":     gamma,
        "gamma_lo":  gamma_lo,
        "gamma_hi":  gamma_hi,
        "budget":    budget,
        "n_pool":    n_pool,
    }
