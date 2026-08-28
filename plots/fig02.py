"""Figure 02 -- composition: does speculative speedup survive lower precision?

The headline test. Grouped bars: target precision on x, geometric-mean speculative
speedup on y, 95% paired intervals.

The hypothesis: if speculation and quantization spend the same memory-bound slack, the
bars fall from left to right. A flat row falsifies it. That prediction used to be drawn
on the figure as an arrow and a two-line caption, so a reader met it before the bars.
It now lives in the report caption instead -- with seven gammas across three precisions
there are twenty-one bars, and the annotation sat on top of the legend. The reference
line at 1.0 carries the only in-figure claim that still needs making, which is where
speculation stops paying for itself at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.derive import select_model, speedup_table
from analysis.load import assert_single_stack, require_measured
from plots import style


def render(df: pd.DataFrame, outdir: Path, *, target_model: str | None = None) -> Path:
    require_measured(df)
    assert_single_stack(df, "fig02")

    # Precision is this figure's x axis, and each precision is its own checkpoint, so the
    # frame legitimately spans three target_models. One bar is one (precision, gamma)
    # cell; as long as each of those resolves to a single checkpoint, nothing is averaged.
    table = select_model(speedup_table(df), target_model,
                         cell_keys=("target_dtype", "num_speculative_tokens"))
    table = table[table["batch_size"] == table["batch_size"].min()]
    if table.empty:
        raise ValueError("fig02: no batch-1 speculative conditions to plot")

    gammas = sorted(table["num_speculative_tokens"].dropna().unique())
    dtypes = [d for d in style.DTYPE_ORDER if d in set(table["target_dtype"])]
    if not dtypes:
        raise ValueError("fig02: no recognized target_dtype values present")

    # Wider than the other figures: this one carries len(gammas) bars per precision, and
    # at seven they need the room.
    fig, ax = style.new_figure(figsize=(7.6, 4.2))

    x = np.arange(len(dtypes), dtype=float)
    n_groups = len(gammas)
    # gamma is an ordered magnitude, not an identity, so it gets a light-to-dark ramp.
    # Cycling the categorical slots would give two different gammas the same hue.
    ramp = style.ordinal_ramp(n_groups)
    # A 2px surface gap between adjacent fills, expressed in axis units.
    width = min(0.72 / max(n_groups, 1), 0.28)
    gap = width * 0.06

    for gi, gamma in enumerate(gammas):
        offsets = x + (gi - (n_groups - 1) / 2) * (width + gap)
        sub = table[table["num_speculative_tokens"] == gamma].set_index("target_dtype")
        heights, los, his = [], [], []
        for d in dtypes:
            if d not in sub.index:
                # A missing cell is left missing. Interpolating it would be an invented
                # measurement, which is the one thing this study never does.
                heights.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            row = sub.loc[d]
            heights.append(float(row["speedup"]))
            los.append(float(row["speedup_lo95"]))
            his.append(float(row["speedup_hi95"]))

        color = ramp[gi]
        ax.bar(
            offsets, heights, width=width, color=color, edgecolor=style.SURFACE,
            linewidth=1.2, zorder=3, label=f"γ = {int(gamma)}",
        )
        h = np.asarray(heights, dtype=float)
        yerr = np.vstack([h - np.asarray(los, float), np.asarray(his, float) - h])
        ax.errorbar(
            offsets, h, yerr=np.clip(yerr, 0, None), linestyle="none",
            color=style.INK_SECONDARY, elinewidth=1.1, capsize=2.5, zorder=4,
        )
        # Per-bar value labels are set vertically. Horizontally they overlapped their
        # neighbours from about five gammas onward -- at seven they were unreadable, and
        # an unreadable number on a chart is worse than no number, because it still costs
        # the reader the attempt. Exact values live in the primary table.
        for xpos, height in zip(offsets, h):
            if np.isnan(height):
                continue
            ax.annotate(
                f"{height:.2f}×", xy=(xpos, height), xytext=(0, 7),
                textcoords="offset points", ha="center", va="bottom", rotation=90,
                fontsize=6.5, color=style.INK_SECONDARY,
            )

    style.reference_line(ax, 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([style.DTYPE_LABEL[d] for d in dtypes])
    ax.set_xlabel("target precision")
    ax.set_ylabel("speculative speedup (geometric mean)")
    ax.set_ylim(bottom=0)
    # Headroom for the rotated labels, so the tallest one does not run into the legend.
    top = float(np.nanmax(table["speedup_hi95"].to_numpy(dtype=float)))
    ax.set_ylim(top=top * 1.42)
    legend = ax.legend(loc="upper right", ncol=min(len(gammas), 4), title="drafted tokens",
                       framealpha=0.95)
    legend.get_title().set_fontsize(7.5)
    fig.tight_layout()
    return style.save(fig, outdir, "fig02_composition")
