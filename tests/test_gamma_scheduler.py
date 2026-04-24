"""Unit tests for gamma_scheduler."""

import numpy as np
import pytest
from cscs.selection.gamma_scheduler import compute_dcr, compute_alpha_eff, compute_gamma


class TestComputeAlphaEff:
    def test_basic(self):
        # B=10, N=100 → alpha_eff = 10/sqrt(100) = 1.0
        assert compute_alpha_eff(10, 100) == pytest.approx(1.0)

    def test_small_budget(self):
        # B=5, N=25 → alpha_eff = 5/sqrt(25) = 1.0
        assert compute_alpha_eff(5, 25) == pytest.approx(1.0)

    def test_larger_n(self):
        # B=10, N=400 → alpha_eff = 10/20 = 0.5
        assert compute_alpha_eff(10, 400) == pytest.approx(0.5)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            compute_alpha_eff(10, 0)


class TestComputeDcr:
    def test_positive_dcr(self):
        # Perfectly correlated: U == T → DCR = 1.0
        U = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dcr, pval = compute_dcr(U, T)
        assert dcr == pytest.approx(1.0, abs=1e-6)

    def test_negative_dcr(self):
        # Anti-correlated: U inversely proportional to T → DCR = -1.0
        U = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        T = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        dcr, pval = compute_dcr(U, T)
        assert dcr == pytest.approx(-1.0, abs=1e-6)

    def test_small_input(self):
        # Less than 3 samples → returns 0.0
        dcr, pval = compute_dcr(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert dcr == 0.0

    def test_returns_float(self):
        U = np.random.rand(20)
        T = np.random.rand(20)
        dcr, pval = compute_dcr(U, T)
        assert isinstance(dcr, float)
        assert isinstance(pval, float)


class TestComputeGamma:
    def test_zero_dcr_gives_half(self):
        # DCR=0 → gamma_raw = 0.5 → gamma = 0.5
        gamma, alpha_eff = compute_gamma(10, 100, dcr=0.0)
        assert gamma == pytest.approx(0.5)

    def test_positive_dcr_increases_gamma(self):
        gamma_pos, _ = compute_gamma(10, 100, dcr=0.5)
        gamma_zero, _ = compute_gamma(10, 100, dcr=0.0)
        assert gamma_pos > gamma_zero

    def test_negative_dcr_decreases_gamma(self):
        gamma_neg, _ = compute_gamma(10, 100, dcr=-0.5)
        gamma_zero, _ = compute_gamma(10, 100, dcr=0.0)
        assert gamma_neg < gamma_zero

    def test_clipping_upper(self):
        # Large positive DCR + large budget → clipped to gamma_hi=0.7
        gamma, _ = compute_gamma(100, 100, dcr=1.0, gamma_hi=0.7)
        assert gamma <= 0.7

    def test_clipping_lower(self):
        # Large negative DCR → clipped to gamma_lo=0.3
        gamma, _ = compute_gamma(100, 100, dcr=-1.0, gamma_lo=0.3)
        assert gamma >= 0.3

    def test_alpha_eff_returned(self):
        gamma, alpha_eff = compute_gamma(10, 100, dcr=0.0)
        assert alpha_eff == pytest.approx(1.0)

    def test_paper_formula(self):
        # Manual check: B=77, N=387, DCR=0.2
        # alpha_eff = 77/sqrt(387) = 77/19.673 ≈ 3.914
        # gamma_raw = 0.5 + (0.2/4) * (3.914/4.914) = 0.5 + 0.05*0.797 = 0.5398
        # clipped to [0.3, 0.7] → 0.5398
        gamma, alpha_eff = compute_gamma(77, 387, dcr=0.2)
        expected_alpha = 77 / np.sqrt(387)
        expected_gamma_raw = 0.5 + (0.2 / 4.0) * (expected_alpha / (1.0 + expected_alpha))
        expected_gamma = float(np.clip(expected_gamma_raw, 0.3, 0.7))
        assert gamma == pytest.approx(expected_gamma, rel=1e-6)
        assert alpha_eff == pytest.approx(expected_alpha, rel=1e-6)
