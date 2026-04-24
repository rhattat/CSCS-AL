"""
Budget progression figure: Dice / HD95 vs annotation budget for all datasets.

Layout (3×3 grid):
    (0,0) BraTS  Dice  | (0,1) Spleen Dice | (0,2) FeTA  Dice
    (1,0) BraTS  HD95  | (1,1) Spleen HD95 | (1,2) FeTA  HD95
    (2,0) DIANE  Dice  | (2,1) DIANE  HD95 | (2,2) Legend

Usage:
    from cscs.visualization.plot_budget_curves import plot_budget_figure
    plot_budget_figure(agg_df, dataset_configs, output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


METHOD_STYLE: dict[str, dict] = {
    "Random":    {"color": "#9467bd", "marker": "o", "ls": "--", "lw": 1.3, "ms": 5},
    "TypiClust": {"color": "#2ca02c", "marker": "s", "ls": "--", "lw": 1.3, "ms": 5},
    "FPS":       {"color": "#1f77b4", "marker": "^", "ls": "--", "lw": 1.3, "ms": 5},
    "ProbCover": {"color": "#ff7f0e", "marker": "D", "ls": "--", "lw": 1.3, "ms": 5},
    "CSAL-3D":   {"color": "#d62728", "marker": "v", "ls": "--", "lw": 1.3, "ms": 5},
    "CSCS":      {"color": "#8c564b", "marker": "o", "ls": "-",  "lw": 2.0, "ms": 6},
}

METHOD_ORDER = ["Random", "TypiClust", "FPS", "ProbCover", "CSAL-3D", "CSCS"]


def _draw_panel(
    ax: plt.Axes,
    agg_df: pd.DataFrame,
    budgets: list[int],
    metric_mean_col: str,
    metric_std_col: str,
    upper_bound: Optional[float] = None,
    ylabel: str = "",
) -> None:
    """Draw a single Dice or HD95 panel."""
    for method in METHOD_ORDER:
        sub = agg_df[agg_df["method"] == method] if not agg_df.empty else pd.DataFrame()
        sty = METHOD_STYLE.get(method, {"color": "gray", "marker": "o", "ls": "-", "lw": 1, "ms": 4})
        xs, ys, errs = [], [], []
        for K in budgets:
            row = sub[sub["K"] == K] if not sub.empty else pd.DataFrame()
            if row.empty:
                continue
            mu  = row[metric_mean_col].values[0] if metric_mean_col in row.columns else np.nan
            std = row[metric_std_col].values[0]  if metric_std_col  in row.columns else 0.0
            if np.isnan(mu):
                continue
            xs.append(K); ys.append(mu); errs.append(std if not np.isnan(std) else 0.0)

        if not xs:
            continue
        ax.plot(xs, ys, color=sty["color"], marker=sty["marker"],
                linestyle=sty["ls"], linewidth=sty["lw"], markersize=sty["ms"],
                zorder=4, label=method)
        ax.errorbar(xs, ys, yerr=errs, fmt="none",
                    ecolor=sty["color"], elinewidth=0.7,
                    capsize=2.5, capthick=0.7, alpha=0.55, zorder=3)

    if upper_bound is not None and not np.isnan(upper_bound):
        ax.axhline(upper_bound, color="black", linestyle="--", linewidth=1.1, zorder=5)

    ax.set_xticks(budgets)
    ax.set_xticklabels([str(k) for k in budgets], fontsize=7)
    margin = (budgets[-1] - budgets[0]) * 0.08 if len(budgets) > 1 else 1
    ax.set_xlim(budgets[0] - margin, budgets[-1] + margin)
    ax.set_xlabel("Annotation Budget", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _draw_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    handles = []
    for method in METHOD_ORDER:
        sty = METHOD_STYLE.get(method, {})
        handles.append(mlines.Line2D(
            [], [], color=sty.get("color", "gray"),
            marker=sty.get("marker", "o"), linestyle=sty.get("ls", "-"),
            linewidth=sty.get("lw", 1), markersize=sty.get("ms", 5), label=method
        ))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--",
                                  linewidth=1.2, markersize=0, label="Full supervision"))
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5),
              fontsize=10, frameon=True, title="Methods", title_fontsize=11)


def plot_budget_figure(
    agg_df: pd.DataFrame,
    dataset_configs: dict,
    output_dir: str | Path,
    save_pdf: bool = True,
    filename: str = "fig_budget_all_datasets",
) -> Path:
    """
    Generate the 3×3 budget progression figure.

    Args:
        agg_df:          Aggregated results DataFrame with columns:
                         method, dataset, K, {metric}_mean_fg_mean, {metric}_mean_fg_std
        dataset_configs: Dict mapping dataset key → config dict with keys:
                         label, budgets, ub_dice, ub_hd95, tag
        output_dir:      Directory to save the figure.
        save_pdf:        Also save a PDF version.
        filename:        Output filename (without extension).

    Returns:
        Path to the saved PNG.
    """
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_list = list(dataset_configs.keys())

    # Build grid: up to 4 datasets × 2 metrics, plus legend
    grid_entries = []
    row, col = 0, 0
    for ds in ds_list[:3]:
        grid_entries.append((0, col, ds, "dice")); col += 1
    col = 0
    for ds in ds_list[:3]:
        grid_entries.append((1, col, ds, "hd95")); col += 1
    if len(ds_list) > 3:
        ds4 = ds_list[3]
        grid_entries.append((2, 0, ds4, "dice"))
        grid_entries.append((2, 1, ds4, "hd95"))

    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    plt.subplots_adjust(hspace=0.32, wspace=0.18)

    for (row, col, ds, metric) in grid_entries:
        ax = axes[row, col]
        cfg = dataset_configs.get(ds, {})
        ds_data = agg_df[agg_df["dataset"] == ds] if not agg_df.empty else pd.DataFrame()
        budgets = cfg.get("budgets", [])
        ub = cfg.get("ub_dice") if metric == "dice" else cfg.get("ub_hd95")
        ylabel = "Dice (%)" if metric == "dice" else "HD95 (mm)"
        m_col = "dice_mean_fg_mean" if metric == "dice" else "hd95_mean_fg_mean"
        s_col = "dice_mean_fg_std"  if metric == "dice" else "hd95_mean_fg_std"
        _draw_panel(ax, ds_data, budgets, m_col, s_col, upper_bound=ub, ylabel=ylabel)

    # Hide unused subplots
    used = {(r, c) for r, c, _, _ in grid_entries}
    for r in range(3):
        for c in range(3):
            if (r, c) not in used and (r, c) != (2, 2):
                axes[r, c].set_visible(False)

    _draw_legend(axes[2, 2])

    # Column titles
    fig.canvas.draw()
    for col_i, ds in enumerate(ds_list[:3]):
        cfg = dataset_configs.get(ds, {})
        tag = cfg.get("tag", ds)
        pos = axes[0, col_i].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.012,
                 tag, ha="center", va="bottom", fontsize=12, fontweight="bold")

    png_path = output_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {png_path}")
    return png_path
