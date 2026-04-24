"""
Statistical tests for comparing active learning methods.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Appropriate when comparing two methods across the same seeds.

    Returns:
        (statistic, p_value)
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    result = stats.wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
    return float(result.statistic), float(result.pvalue)


def significance_label(pval: float) -> str:
    """Return *** / ** / * / n.s. based on p-value."""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "n.s."
