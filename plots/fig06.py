"""Figure 06 -- accepted-run-length distributions, as small multiples.

One histogram per condition, faceted -- never overlaid. Overlaying these would be the
classic mistake: the distributions have very different shapes, and stacking translucent
bars on shared axes turns "n-gram is bimodal, the draft model is not" into visual mush.
Small multiples keep every distribution readable at the cost of a little space, and the
comparison is what the figure is for.

This is the one figure where seaborn earns its place: ``FacetGrid`` handles the grid,
the shared axes and the per-facet titles correctly, and hand-rolling that is where
subplot bugs live. Its *styling* still comes from ``plots.style`` -- the grid is
constructed with our rcParams rather than a seaborn theme, so the figure does not change
appearance when seaborn does.

The distribution matters more than the mean it implies. Two conditions with the same
mean acceptance length behave differently if one accepts 3 tokens steadily and the other
alternates between 0 and 6 -- and only the shape shows that.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pandas as pd

from analysis.derive import acceptance_table, condition_label
from analysis.load import require_measured
from plots import style


def render(df: pd.DataFrame, outdir: Path, *, col_wrap: int = 3) -> Path:
    require_measured(df)
    import seaborn as sns

    table = acceptance_table(df)
    long = _to_long(table)

    style.apply_style()
    with mpl.rc_context(style.seaborn_rc()):
        grid = sns.FacetGrid(
            long,
            col="condition",
            col_wrap=min(col_wrap, long["condition"].nunique()),
            sharey=False,
            height=2.1,
            aspect=1.25,
            despine=True,
        )
        grid.map_dataframe(
            sns.barplot,
            x="run_length",
            y="fraction",
            color=style.SERIES[0],
            edgecolor=style.SURFACE,
            linewidth=1.2,
            width=0.86,
        )
        grid.set_axis_labels("accepted run length (tokens per step)", "fraction of steps")
        grid.set_titles(col_template="{col_name}", size=8)
        grid.figure.set_facecolor(style.SURFACE)

        for ax, (_, sub) in zip(grid.axes.flat, long.groupby("condition", sort=False)):
            ax.set_facecolor(style.SURFACE)
            ax.grid(axis="y", color=style.GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            mean_len = float((sub["run_length"] * sub["fraction"]).sum())
            ax.axvline(mean_len, color=style.REFERENCE, linewidth=1.0,
                       linestyle=(0, (3, 2)), zorder=3)
            ax.annotate(
                f"mean {mean_len:.2f}\n{int(sub['n_steps'].iloc[0]):,} steps",
                xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=7, color=style.INK_SECONDARY,
            )

        grid.figure.tight_layout()
        return style.save(grid.figure, outdir, "fig06_acceptance_distribution")


def _to_long(table: pd.DataFrame) -> pd.DataFrame:
    """Explode summed histograms into one row per (condition, run length).

    Normalized to a fraction of steps within each condition, because conditions ran for
    different numbers of verification steps -- raw counts would make a slow condition
    look like a common one.
    """
    rows = []
    for _, row in table.iterrows():
        hist = row["accept_length_histogram"]
        total = sum(hist)
        if total <= 0:
            raise ValueError(f"condition {row['condition_id']}: histogram sums to zero")
        label = condition_label(row, include_model=True)
        for k, count in enumerate(hist):
            rows.append({
                "condition": label,
                "condition_id": row["condition_id"],
                "run_length": k,
                "count": int(count),
                "fraction": count / total,
                "n_steps": total,
            })
    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("fig06: no acceptance histograms to plot")
    return long
