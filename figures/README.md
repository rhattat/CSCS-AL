# Figures

This directory is for generated figures. It is intentionally empty in the repository.

To reproduce the paper figures:
```bash
python scripts/summarize_results.py --root outputs/Article/ --outdir outputs/tables/
python -c "
from cscs.visualization.plot_budget_curves import plot_budget_figure
# See docs/reproduce_experiments.md for full instructions
"
```
