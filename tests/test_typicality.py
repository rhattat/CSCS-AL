"""Unit tests for typicality computation."""

import numpy as np
import pytest
from cscs.features.typicality import compute_typicality


class TestComputeTypicality:
    def test_shape(self):
        emb = np.random.rand(20, 64)
        typ = compute_typicality(emb, k=5)
        assert typ.shape == (20,)

    def test_positive_values(self):
        emb = np.random.rand(20, 64)
        typ = compute_typicality(emb, k=5)
        assert (typ > 0).all()

    def test_denser_cluster_higher_typicality(self):
        # Two clusters: tight cluster vs sparse outlier
        rng = np.random.RandomState(42)
        tight = rng.randn(19, 8) * 0.1        # tight cluster → high typicality
        outlier = rng.randn(1, 8) * 10.0      # outlier → low typicality
        emb = np.vstack([tight, outlier])
        typ = compute_typicality(emb, k=5)
        # The outlier (index 19) should be less typical than the tight cluster
        assert typ[19] < np.median(typ[:19])

    def test_single_sample(self):
        emb = np.array([[1.0, 2.0, 3.0]])
        typ = compute_typicality(emb, k=20)
        assert len(typ) == 1
        assert typ[0] > 0

    def test_k_capped_to_n_minus_1(self):
        emb = np.random.rand(5, 16)
        # k=100 should not crash; capped to 4
        typ = compute_typicality(emb, k=100)
        assert typ.shape == (5,)
