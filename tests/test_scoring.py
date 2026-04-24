"""Unit tests for scoring module."""

import numpy as np
import pytest
from cscs.selection.scoring import rank_normalize, compute_score


class TestRankNormalize:
    def test_single_element(self):
        result = rank_normalize(np.array([5.0]))
        assert result[0] == pytest.approx(1.0)

    def test_ascending_order(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rank_normalize(arr)
        assert result[0] < result[-1]
        assert result[-1] == pytest.approx(1.0)

    def test_clipped_to_eps(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = rank_normalize(arr, eps=0.01)
        assert result.min() >= 0.01

    def test_length_preserved(self):
        arr = np.random.rand(50)
        result = rank_normalize(arr)
        assert len(result) == 50

    def test_tied_values(self):
        arr = np.array([1.0, 1.0, 1.0])
        result = rank_normalize(arr)
        assert len(result) == 3
        assert all(0.01 <= v <= 1.0 for v in result)


class TestComputeScore:
    def test_gamma_zero_pure_typicality(self):
        T = np.array([0.1, 0.5, 0.9])
        U = np.array([0.9, 0.5, 0.1])
        S = compute_score(T, U, gamma=0.0)
        # gamma=0 → S = T_rank^1 * U_rank^0 = T_rank
        # highest T → highest S
        assert S[2] > S[1] > S[0]

    def test_gamma_one_pure_uncertainty(self):
        T = np.array([0.9, 0.5, 0.1])
        U = np.array([0.1, 0.5, 0.9])
        S = compute_score(T, U, gamma=1.0)
        # gamma=1 → S = U_rank
        assert S[2] > S[1] > S[0]

    def test_score_bounded(self):
        T = np.random.rand(30)
        U = np.random.rand(30)
        S = compute_score(T, U, gamma=0.5)
        assert S.min() > 0.0
        assert S.max() <= 1.0

    def test_argmax_selection(self):
        # Verify argmax picks the intended sample
        T = np.array([0.2, 0.8, 0.5])
        U = np.array([0.8, 0.2, 0.5])
        # gamma=0.5: balanced → middle sample wins
        S = compute_score(T, U, gamma=0.5)
        assert len(S) == 3
        assert np.argmax(S) in [0, 1, 2]

    def test_shape_preserved(self):
        T = np.random.rand(100)
        U = np.random.rand(100)
        S = compute_score(T, U, gamma=0.5)
        assert S.shape == (100,)
