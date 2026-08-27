"""Figure 08 -- gamma-slope, dense versus MoE.

Speedup against gamma, normalized to gamma=1 within each architecture, with the fitted
slope annotated.

**The slope is the measurement, not the level.** A dense model and an MoE model have
different absolute speedups for reasons that have nothing to do with the question here
(different sizes, different drafters, different acceptance rates). Normalizing each to
its own gamma=1 point removes that offset and leaves the thing actually being compared:
how much each additional drafted token buys on each architecture. Plotting raw speedups
would let a difference in level masquerade as a difference in slope.

The fit is ordinary least squares on the normalized points, and the fitted line is drawn
only across the gamma range that was measured -- never extended past it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.derive import speedup_table
from analysis.load import assert_single_stack, require_measured
from plots import style


def render(
    df: pd.DataFrame,
    outdir: Path,
    *,
    architecture: dict[str, str],
) -> Path:
    """``architecture`` maps target_model -> 'dense' or 'moe'.

    Passed in rather than inferred: guessing an architecture from a model name would put
    a model in the wrong group on a naming convention, and the grouping *is* the figure.
    """
    require_measured(df)
    assert_single_stack(df, "fig08")

    table = speedup_table(df)
    table = table[table["batch_size"] == table["batch_size"].min()]

    unknown = sorted(set(table["target_model"]) - set(architecture))
    if unknown:
        raise ValueError(
            f"fig08: no architecture given for model(s) {unknown}. Supply dense/moe "
            "explicitly; it must not be inferred from the model name."
        )
    table["architecture"] = table["target_model"].map(architecture)

    bad = sorted(set(table["architecture"]) - {"dense", "moe"})
    if bad:
        raise ValueError(f"fig08: architecture must be 'dense' or 'moe', got {bad}")

    fig, ax = style.new_figure(figsize=(6.4, 4.2))
    labels = {"dense": "dense", "moe": "MoE"}
    fits: dict[str, float] = {}

    for i, arch in enumerate(["dense", "moe"]):
        sub = table[table["architecture"] == arch].sort_values("num_speculative_tokens")
        if sub.empty:
            continue
        st = style.series_style(i)
        gammas = sub["num_speculative_tokens"].to_numpy(dtype=float)
        speedup = sub["speedup"].to_numpy(dtype=float)

        anchor = _anchor_at_gamma_one(gammas, speedup, arch)
        normalized = speedup / anchor
        lo = sub["speedup_lo95"].to_numpy(dtype=float) / anchor
        hi = sub["speedup_hi95"].to_numpy(dtype=float) / anchor

        style.ci_errorbar(ax, gammas, normalized, lo, hi,
                          color=st["color"], marker=st["marker"])

        slope, intercept = np.polyfit(gammas, normalized, 1)
        fits[arch] = float(slope)
        xs = np.linspace(gammas.min(), gammas.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=st["color"], linewidth=1.4,
                linestyle=st["linestyle"], alpha=0.8, zorder=3)
        style.direct_label(ax, gammas[-1], normalized[-1], labels[arch], st["color"])
        ax.annotate(
            f"{labels[arch]}: slope {slope:+.3f} per γ",
            xy=(0.02, 0.96 - 0.07 * i), xycoords="axes fraction",
            fontsize=7.5, color=style.INK_SECONDARY, va="top",
        )

    ax.axhline(1.0, color=style.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlabel("γ (drafted tokens per step)")
    ax.set_ylabel("speedup normalized to γ = 1")
    ax.set_xticks(sorted(table["num_speculative_tokens"].dropna().unique()))
    ax.margins(x=0.12)

    if len(fits) == 2:
        ax.annotate(
            f"slope difference (MoE − dense): {fits['moe'] - fits['dense']:+.3f} per γ",
            xy=(0.99, 0.02), xycoords="axes fraction", ha="right",
            fontsize=7.5, color=style.INK,
        )
    style.annotate_n(ax, int(table["n_prompts"].min()), loc="lower left")
    fig.tight_layout()
    return style.save(fig, outdir, "fig08_gamma_slope_dense_vs_moe")


def _anchor_at_gamma_one(gammas: np.ndarray, speedup: np.ndarray, arch: str) -> float:
    """The gamma=1 speedup each curve is normalized against.

    Raises when gamma=1 was not measured. Normalizing to the smallest gamma present
    instead would silently change what "normalized" means between the two curves, and
    the whole comparison rests on both being anchored the same way.
    """
    match = np.where(gammas == 1)[0]
    if match.size == 0:
        raise ValueError(
            f"fig08: architecture {arch!r} has no γ=1 measurement to normalize against "
            f"(measured γ: {sorted(int(g) for g in gammas)}). Run γ=1 rather than "
            "anchoring the two curves differently."
        )
    return float(speedup[match[0]])
