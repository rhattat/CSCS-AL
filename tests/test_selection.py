"""Integration tests for the selection pipeline."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from cscs.selection.cscs import select_cscs_from_df
from cscs.selection.baselines import select_random, select_fps, select_typiclust, select_csal3d
from cscs.selection import select_cscs


# ============================================================================
# Fixtures
# ============================================================================

def make_pool(N: int = 30, d: int = 16, seed: int = 0):
    rng = np.random.RandomState(seed)
    embeddings = rng.randn(N, d)
    U = rng.rand(N)
    T = rng.rand(N)
    volume_ids = [f"vol_{i:03d}" for i in range(N)]
    return volume_ids, U, T, embeddings


def make_features_csv(N: int = 30, d: int = 16, seed: int = 0) -> tuple[Path, Path]:
    rng = np.random.RandomState(seed)
    volume_ids = [f"vol_{i:03d}" for i in range(N)]
    U = rng.rand(N)
    T = rng.rand(N)
    embeddings = rng.randn(N, d)

    tmp = Path(tempfile.mkdtemp())
    df = pd.DataFrame({
        "volume_id": volume_ids,
        "uncertainty": U,
        "typicality": T,
        "split": ["train"] * N,
    })
    csv_path = tmp / "features.csv"
    df.to_csv(csv_path, index=False)

    emb_dir = tmp / "embeddings"
    emb_dir.mkdir()
    for vid, emb in zip(volume_ids, embeddings):
        np.save(emb_dir / f"{vid}.npy", emb)

    return csv_path, emb_dir


# ============================================================================
# CSCS core tests
# ============================================================================

class TestCSCSSelection:
    def test_budget_equals_n_selected(self):
        vids, U, T, emb = make_pool()
        budget = 10
        df, meta = select_cscs_from_df(vids, U, T, emb, budget)
        assert df["selected"].sum() == budget

    def test_no_duplicates(self):
        vids, U, T, emb = make_pool()
        df, _ = select_cscs_from_df(vids, U, T, emb, budget=10)
        selected = df[df["selected"]]["volume_id"].tolist()
        assert len(set(selected)) == len(selected)

    def test_output_columns(self):
        vids, U, T, emb = make_pool()
        df, _ = select_cscs_from_df(vids, U, T, emb, budget=5)
        for col in ["volume_id", "uncertainty", "typicality", "cluster_id", "score", "selected", "rank"]:
            assert col in df.columns

    def test_metadata_keys(self):
        vids, U, T, emb = make_pool()
        _, meta = select_cscs_from_df(vids, U, T, emb, budget=5)
        for key in ["dcr", "alpha_eff", "gamma", "budget", "n_pool"]:
            assert key in meta

    def test_gamma_in_bounds(self):
        vids, U, T, emb = make_pool()
        _, meta = select_cscs_from_df(vids, U, T, emb, budget=5)
        assert 0.3 <= meta["gamma"] <= 0.7

    def test_budget_1(self):
        vids, U, T, emb = make_pool(N=20)
        df, meta = select_cscs_from_df(vids, U, T, emb, budget=1)
        assert df["selected"].sum() == 1

    def test_from_csv(self):
        csv_path, emb_dir = make_features_csv()
        with tempfile.TemporaryDirectory() as outdir:
            df, meta = select_cscs(
                features_csv=csv_path,
                budget=8,
                output_dir=outdir,
                embeddings_dir=emb_dir,
                verbose=False,
            )
            assert df["selected"].sum() == 8
            assert (Path(outdir) / "selected_ids.csv").exists()
            assert (Path(outdir) / "metadata.json").exists()

    def test_different_seeds_different_selections(self):
        vids, U, T, emb = make_pool(N=50)
        df1, _ = select_cscs_from_df(vids, U, T, emb, budget=10, seed=42)
        df2, _ = select_cscs_from_df(vids, U, T, emb, budget=10, seed=123)
        sel1 = set(df1[df1["selected"]]["volume_id"])
        sel2 = set(df2[df2["selected"]]["volume_id"])
        assert sel1 != sel2  # Different seeds → different selections (usually)

    def test_budget_exceeds_pool_raises(self):
        vids, U, T, emb = make_pool(N=5)
        with pytest.raises(ValueError):
            select_cscs_from_df(vids, U, T, emb, budget=10)


# ============================================================================
# Baseline tests
# ============================================================================

class TestBaselines:
    def test_random_budget(self):
        vids, U, T, emb = make_pool()
        df, _ = select_random(vids, U, T, budget=10)
        assert df["selected"].sum() == 10

    def test_random_no_duplicates(self):
        vids, U, T, emb = make_pool()
        df, _ = select_random(vids, U, T, budget=10, seed=42)
        sel = df[df["selected"]]["volume_id"].tolist()
        assert len(set(sel)) == len(sel)

    def test_fps_budget(self):
        vids, U, T, emb = make_pool()
        df, _ = select_fps(vids, U, T, budget=10, embeddings=emb)
        assert df["selected"].sum() == 10

    def test_typiclust_budget(self):
        vids, U, T, emb = make_pool(N=40)
        df, _ = select_typiclust(vids, U, T, budget=8, embeddings=emb)
        assert df["selected"].sum() == 8

    def test_csal3d_budget(self):
        vids, U, T, emb = make_pool()
        df, _ = select_csal3d(vids, U, T, budget=10, embeddings=emb)
        assert df["selected"].sum() == 10

    def test_all_methods_same_interface(self):
        from cscs.selection.registry import REGISTRY, list_methods
        vids, U, T, emb = make_pool(N=30)
        for method_name in list_methods():
            selector = REGISTRY[method_name]
            df, meta = selector(volume_ids=vids, U=U, T=T, budget=8, embeddings=emb)
            assert df["selected"].sum() == 8, f"{method_name} selected wrong number"
            selected = df[df["selected"]]["volume_id"].tolist()
            assert len(set(selected)) == len(selected), f"{method_name} has duplicates"
