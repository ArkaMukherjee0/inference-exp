"""Figure 03 -- batch-size collapse.

Speculative speedup against batch size (log x), one line per precision, with a 1.0x
reference line and the predicted crossover marked.

Batching and speculation both spend the same idle compute, so the prediction is that
speedup decays toward 1.0 as the batch grows -- and crosses below it sooner at lower
precision, where less slack was left to begin with. The crossover is where speculative
decoding stops paying for itself, which is the number a serving team actually needs.
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
    assert_single_stack(df, "fig03")

    table = select_model(speedup_table(df), target_model)
    if table["batch_size"].nunique() < 2:
        raise ValueError(
            "fig03 needs at least two batch sizes; this log has "
            f"{table['batch_size'].nunique()}."
        )

    # One gamma only -- mixing gammas on this axis would confound the two effects.
    gamma = _dominant_gamma(table)
    table = table[table["num_speculative_tokens"] == gamma]

    fig, ax = style.new_figure(figsize=(6.6, 4.2))
    dtypes = [d for d in style.DTYPE_ORDER if d in set(table["target_dtype"])]

    for dtype in dtypes:
        sub = table[table["target_dtype"] == dtype].sort_values("batch_size")
        if sub.empty:
            continue
        color = style.dtype_color(dtype)
        x = sub["batch_size"].to_numpy(dtype=float)
        y = sub["speedup"].to_numpy(dtype=float)
        ax.plot(
            x, y, color=color, marker=style.dtype_marker(dtype),
            linestyle=style.dtype_linestyle(dtype), zorder=4,
            markeredgecolor=style.SURFACE, markeredgewidth=0.8,
        )
        style.error_band(ax, x, sub["speedup_lo95"], sub["speedup_hi95"], color)
        style.direct_label(ax, x[-1], y[-1], style.DTYPE_LABEL[dtype], color)
        _mark_crossover(ax, x, y, color)

    style.reference_line(ax, 1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel(f"speculative speedup at γ = {int(gamma)} (geometric mean)")
    ax.set_xticks(sorted(table["batch_size"].unique()))
    ax.get_xaxis().set_major_formatter(_plain_int_formatter())
    ax.margins(x=0.12)
    style.annotate_n(ax, int(table["n_prompts"].min()), loc="lower left")
    fig.tight_layout()
    return style.save(fig, outdir, "fig03_batch_collapse")


def _dominant_gamma(table: pd.DataFrame) -> int:
    """The gamma covering the most batch sizes -- the one the sweep actually swept."""
    counts = table.groupby("num_speculative_tokens")["batch_size"].nunique()
    return int(counts.idxmax())


def _mark_crossover(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    """Mark where the measured curve crosses 1.0, by interpolation between neighbours.

    This annotates a crossing that the *measured* points bracket -- it is a reading of
    the data, not an extrapolation beyond it. If the curve never crosses, nothing is
    drawn rather than an estimate of where it might have.
    """
    below = np.where(y < 1.0)[0]
    if below.size == 0 or below[0] == 0:
        return
    i = below[0]
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    if y0 == y1:
        return
    # Interpolate in log-batch space, which is the axis the reader sees.
    lx = np.log2(x0) + (1.0 - y0) * (np.log2(x1) - np.log2(x0)) / (y1 - y0)
    xc = float(2 ** lx)
    ax.plot([xc], [1.0], marker="v", color=color, markersize=7,
            markeredgecolor=style.SURFACE, markeredgewidth=0.8, zorder=5, clip_on=False)
    ax.annotate(
        f"crosses 1.0× at batch ≈ {xc:.1f}",
        xy=(xc, 1.0), xytext=(0, -14), textcoords="offset points",
        ha="center", fontsize=7, color=style.INK_SECONDARY,
    )


def _plain_int_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: f"{int(v)}")
