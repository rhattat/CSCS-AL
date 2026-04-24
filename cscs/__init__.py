"""
CSCS-AL: Cold-Start Curriculum Selection for Active Learning
in 3D Medical Image Segmentation.

Paper: "CSCS: A Dataset-Aware Curriculum for Cold-Start Active Learning
        in 3D Medical Image Segmentation"

Quick start:
    from cscs.selection import select_cscs

    df, meta = select_cscs(
        features_csv="path/to/features.csv",
        budget=10,
        output_dir="path/to/output/",
        embeddings_dir="path/to/embeddings/",
    )
    print(df[df["selected"]]["volume_id"].tolist())
"""

__version__ = "1.0.0"
__author__  = "Rémi HATTAT"

from .selection import select_cscs, select_cscs_from_df

__all__ = ["select_cscs", "select_cscs_from_df"]
