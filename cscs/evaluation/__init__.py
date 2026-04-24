"""Evaluation utilities for CSCS-AL."""

from .metrics import discover_result_folders, find_eval_csv, parse_result_row
from .summarize import aggregate_seeds, format_cell
from .statistics import wilcoxon_test, significance_label

__all__ = [
    "discover_result_folders",
    "find_eval_csv",
    "parse_result_row",
    "aggregate_seeds",
    "format_cell",
    "wilcoxon_test",
    "significance_label",
]
