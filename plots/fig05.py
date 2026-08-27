"""Figure 05 -- speculative speedup against measured ridge point.

Three platform points: CPU, H100 TP1, H100 TP2. X is the machine's *measured* ridge
point (achieved TFLOP/s over achieved GB/s) on a log axis; y is speculative speedup.

**Measured and extrapolated points are drawn differently, and this is not a stylistic
choice.** An extrapolated point rendered identically to a measured one is a
misrepresentation -- a reader has no way to tell which numbers came from a machine.
Measured points are filled markers on a solid connector; extrapolated points are open
markers on a dashed one, and the legend says so.

The speedups on this axis are all *within-stack ratios*. A llama.cpp number and a vLLM
number are never compared directly -- their overheads differ entirely -- but the ratio of
speculative to non-speculative within each stack is comparable, because whatever the
stack costs cancels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.derive import speedup_table
from analysis.load import require_measured
from plots import style


def render(
    df: pd.DataFrame,
    outdir: Path,
    *,
    ridge_points: dict[str, float],
    extrapolated: set[str] | None = None,
) -> Path:
    """``ridge_points`` maps a platform key to its measured FLOP/byte, from bench/micro.py."""
    require_measured(df)
    extrapolated = extrapolated or set()

    table = speedup_table(df)
    table = table[table["batch_size"] == table["batch_size"].min()]
    points = _platform_points(table, ridge_points)
    if points.empty:
        raise ValueError("fig05: no platform points could be assembled")

    fig, ax = style.new_figure(figsize=(6.6, 4.3))

    measured = points[~points["key"].isin(extrapolated)].sort_values("ridge_point")
    extrap = points[points["key"].isin(extrapolated)].sort_values("ridge_point")

    if len(measured) >= 2:
        ax.plot(measured["ridge_point"], measured["speedup"], color=style.NEUTRAL,
                linewidth=1.2, zorder=2)
    if not extrap.empty:
        joined = points.sort_values("ridge_point")
        ax.plot(joined["ridge_point"], joined["speedup"], color=style.NEUTRAL,
                linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    for _, row in points.iterrows():
        is_extrap = row["key"] in extrapolated
        color = row["color"]
        ax.errorbar(
            [row["ridge_point"]], [row["speedup"]],
            yerr=[[row["speedup"] - row["speedup_lo95"]], [row["speedup_hi95"] - row["speedup"]]],
            color=color, marker=row["marker"], markersize=9,
            markerfacecolor=style.SURFACE if is_extrap else color,
            markeredgecolor=color, markeredgewidth=1.6,
            elinewidth=1.2, linestyle="none", zorder=4,
        )
        ax.annotate(
            row["label"] + (" (extrapolated)" if is_extrap else ""),
            xy=(row["ridge_point"], row["speedup"]),
            xytext=(0, 12), textcoords="offset points", ha="center",
            fontsize=7.5, color=style.INK,
        )

    style.reference_line(ax, 1.0)
    ax.set_xscale("log")
    ax.set_xlabel("measured ridge point (FLOP/byte, achieved compute ÷ achieved bandwidth)")
    ax.set_ylabel("speculative speedup (geometric mean)")
    ax.grid(axis="x", color=style.GRID, linewidth=0.6)
    ax.margins(x=0.18, y=0.22)
    ax.legend(handles=style.measured_vs_extrapolated_handles(), loc="lower right")
    style.annotate_n(ax, int(points["n_prompts"].min()), loc="upper left")
    fig.tight_layout()
    return style.save(fig, outdir, "fig05_platform_curve")


def _platform_points(table: pd.DataFrame, ridge_points: dict[str, float]) -> pd.DataFrame:
    """Collapse the sweep to one speedup per platform key.

    The key is ``platform`` plus tensor-parallel size, because TP changes the effective
    bandwidth per parameter and is therefore a different point on this axis, not a
    variation of the same one.
    """
    rows = []
    for (plat, tp), group in table.groupby(["platform", "tensor_parallel_size"]):
        key = f"{plat}_tp{int(tp)}" if plat != "cpu" else "cpu"
        if key not in ridge_points:
            raise ValueError(
                f"no measured ridge point for {key!r}. Run bench/micro.py on that "
                "platform -- figure 05's x axis must not carry a spec-sheet number."
            )
        # Within a platform, take the best-performing gamma: the question this figure
        # asks is how much speculation *can* buy on a machine, not what an arbitrary
        # gamma happened to give.
        best = group.loc[group["speedup"].idxmax()]
        idx = len(rows) % len(style.SERIES)
        rows.append({
            "key": key,
            "label": _label(plat, int(tp), int(best["num_speculative_tokens"])),
            "ridge_point": float(ridge_points[key]),
            "speedup": float(best["speedup"]),
            "speedup_lo95": float(best["speedup_lo95"]),
            "speedup_hi95": float(best["speedup_hi95"]),
            "n_prompts": int(best["n_prompts"]),
            "color": style.SERIES[idx],
            "marker": style.MARKERS[idx],
        })
    return pd.DataFrame(rows)


def _label(platform: str, tp: int, gamma: int) -> str:
    base = "CPU" if platform == "cpu" else f"{platform.upper()} TP{tp}"
    return f"{base}\nγ={gamma}"


def ridge_points_from_micro(paths: dict[str, str | Path]) -> dict[str, float]:
    """Read measured ridge points out of ``logs/*_micro.json``.

    Deliberately the only supported source. There is no parameter anywhere in this
    module for supplying a ridge point by hand.
    """
    import json

    out: dict[str, float] = {}
    for key, path in paths.items():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        value = data.get("ridge_point_flop_per_byte")
        if not value or value <= 0:
            raise ValueError(f"{path}: no usable ridge_point_flop_per_byte")
        out[key] = float(value)
    return out
