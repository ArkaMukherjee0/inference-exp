"""Figure 01 -- quality/throughput Pareto.

One point per condition: decode throughput on x, GSM8K exact match on y, with 95%
intervals on both axes. The Pareto frontier is traced so the reader can see which
conditions are actually on it and which are simply dominated -- a point that is slower
*and* less accurate than another is not a trade-off, it is a mistake.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.derive import condition_label, throughput_table
from analysis.load import assert_single_stack, require_measured
from analysis.stats import Interval, median_over_repeats
from plots import style


def render(df: pd.DataFrame, outdir: Path, *, scored: pd.DataFrame | None = None) -> Path:
    require_measured(df)
    if scored is None:
        from evals.gsm8k import score_frame

        scored = score_frame(df)

    tput = throughput_table(df)
    assert_single_stack(df[df["condition_id"].isin(tput["condition_id"])], "fig01")

    points = _assemble(tput, scored, df)
    fig, ax = style.new_figure(figsize=(6.6, 4.4))

    for _, row in points.iterrows():
        color = style.dtype_color(row["target_dtype"])
        marker = "o" if row["spec_method"] == "none" else style.dtype_marker(row["target_dtype"])
        filled = row["spec_method"] != "none"
        ax.errorbar(
            row["throughput_tok_s"], row["em"],
            xerr=[[row["throughput_tok_s"] - row["tput_lo"]], [row["tput_hi"] - row["throughput_tok_s"]]],
            yerr=[[row["em"] - row["em_lo"]], [row["em_hi"] - row["em"]]],
            color=color, marker=marker, markersize=7,
            markerfacecolor=color if filled else style.SURFACE,
            markeredgecolor=color, markeredgewidth=1.4,
            elinewidth=1.0, linestyle="none", zorder=4,
        )

    _trace_frontier(ax, points)
    _label_points(ax, points)

    ax.set_xlabel("decode throughput (tokens/s, higher is better)")
    ax.set_ylabel("GSM8K exact match")
    ax.grid(axis="x", color=style.GRID, linewidth=0.6)
    style.annotate_n(ax, int(points["n_prompts"].min()), loc="lower left")

    handles = _legend_handles(points)
    ax.legend(handles=handles, loc="best", ncol=2)
    fig.tight_layout()
    return style.save(fig, outdir, "fig01_quality_throughput_pareto")


def _assemble(tput: pd.DataFrame, scored: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Join throughput and quality per condition, with an interval on each."""
    collapsed = median_over_repeats(df)
    rows = []
    for _, t in tput.iterrows():
        cid = t["condition_id"]
        em_vec = scored.loc[scored["condition_id"] == cid, "em"].to_numpy(dtype=float)
        if em_vec.size == 0:
            raise ValueError(f"condition {cid} has throughput but no quality score")
        em_ci = _bootstrap_mean(em_vec)

        tpot = collapsed.loc[collapsed["condition_id"] == cid, "tpot_ms"].to_numpy(dtype=float)
        tput_ci = _bootstrap_throughput(tpot)

        row = t.to_dict()
        row.update({
            "em": em_ci.point, "em_lo": em_ci.lo95, "em_hi": em_ci.hi95,
            "throughput_tok_s": tput_ci.point,
            "tput_lo": tput_ci.lo95, "tput_hi": tput_ci.hi95,
            "n_prompts": min(int(em_ci.n), int(tput_ci.n)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean(values: np.ndarray, *, n_boot: int = 10_000, seed: int = 0) -> Interval:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Interval(float(values.mean()), float(lo), float(hi), int(values.size))


def _bootstrap_throughput(tpot_ms: np.ndarray, *, n_boot: int = 10_000, seed: int = 0) -> Interval:
    """Throughput is 1000/tpot, a ratio -- so it is summarized geometrically."""
    if np.any(tpot_ms <= 0):
        raise ValueError("non-positive tpot_ms in throughput bootstrap")
    log_tput = np.log(1000.0) - np.log(tpot_ms)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, log_tput.size, size=(n_boot, log_tput.size))
    boot = np.exp(log_tput[idx].mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Interval(float(np.exp(log_tput.mean())), float(lo), float(hi), int(log_tput.size))


def _trace_frontier(ax, points: pd.DataFrame) -> None:
    """The Pareto-optimal set: no other point is both faster and more accurate."""
    ordered = points.sort_values("throughput_tok_s", ascending=False)
    frontier = []
    best_em = -np.inf
    for _, row in ordered.iterrows():
        if row["em"] > best_em:
            frontier.append(row)
            best_em = row["em"]
    if len(frontier) < 2:
        return
    fx = [r["throughput_tok_s"] for r in frontier]
    fy = [r["em"] for r in frontier]
    ax.plot(fx, fy, color=style.NEUTRAL, linewidth=1.2, linestyle=(0, (5, 3)),
            zorder=2, label="Pareto frontier")


def _label_points(ax, points: pd.DataFrame) -> None:
    """Direct labels on the frontier only.

    A label on every point would be unreadable; the frontier is where the decisions
    get made, so that is what gets named.
    """
    ordered = points.sort_values("throughput_tok_s", ascending=False)
    best_em = -np.inf
    for _, row in ordered.iterrows():
        if row["em"] > best_em:
            best_em = row["em"]
            ax.annotate(
                condition_label(row),
                xy=(row["throughput_tok_s"], row["em"]),
                xytext=(0, 9), textcoords="offset points",
                fontsize=7.5, color=style.INK, ha="center",
            )


def _legend_handles(points: pd.DataFrame) -> list:
    from matplotlib.lines import Line2D

    handles = []
    for dtype in style.DTYPE_ORDER:
        if dtype in set(points["target_dtype"]):
            handles.append(Line2D(
                [], [], marker=style.dtype_marker(dtype), linestyle="none",
                color=style.dtype_color(dtype), markersize=7,
                label=style.DTYPE_LABEL[dtype],
            ))
    handles += [
        Line2D([], [], marker="o", linestyle="none", color=style.INK_SECONDARY,
               markerfacecolor=style.SURFACE, markersize=7, label="no speculation"),
        Line2D([], [], marker="o", linestyle="none", color=style.INK_SECONDARY,
               markerfacecolor=style.INK_SECONDARY, markersize=7, label="speculative"),
    ]
    return handles
