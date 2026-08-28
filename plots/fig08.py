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

**One curve per model, never per architecture.** Two MoE checkpoints in the same frame
are two different models, not two samples of "MoE": they have different expert counts,
different sizes and different acceptance behaviour. Pooling them would sort their points
together by gamma, fit a single line through both, and -- because the anchor search takes
the first gamma=1 row it finds -- normalize one model against the *other* model's anchor.
Every one of those is silent. So the grouping key is the model, and architecture only
decides ordering and labelling.
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

    groups = _series_groups(table)
    if len(groups) > len(style.SERIES):
        raise ValueError(
            f"fig08: {len(groups)} model curves requested but the validated palette has "
            f"{len(style.SERIES)} slots. Split this into small multiples rather than "
            "generating a fourth hue, or drop a model from the frame."
        )

    fig, ax = style.new_figure(figsize=(6.4, 4.2))
    labels = {"dense": "dense", "moe": "MoE"}
    fits: dict[str, float] = {}
    arch_slopes: dict[str, list[float]] = {"dense": [], "moe": []}

    for i, (arch, model) in enumerate(groups):
        sub = table[table["target_model"] == model].sort_values("num_speculative_tokens")
        if sub.empty:
            continue
        st = style.series_style(i)
        gammas = sub["num_speculative_tokens"].to_numpy(dtype=float)
        speedup = sub["speedup"].to_numpy(dtype=float)

        anchor = _anchor_at_gamma_one(gammas, speedup, f"{arch} / {model}")
        normalized = speedup / anchor
        lo = sub["speedup_lo95"].to_numpy(dtype=float) / anchor
        hi = sub["speedup_hi95"].to_numpy(dtype=float) / anchor

        style.ci_errorbar(ax, gammas, normalized, lo, hi,
                          color=st["color"], marker=st["marker"])

        slope, intercept = np.polyfit(gammas, normalized, 1)
        fits[model] = float(slope)
        arch_slopes[arch].append(float(slope))
        xs = np.linspace(gammas.min(), gammas.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=st["color"], linewidth=1.4,
                linestyle=st["linestyle"], alpha=0.8, zorder=3)
        label = labels[arch] if len(groups) <= 2 else f"{_short(model)} ({labels[arch]})"
        style.direct_label(ax, gammas[-1], normalized[-1], label, st["color"])
        ax.annotate(
            f"{label}: slope {slope:+.3f} per γ",
            xy=(0.02, 0.96 - 0.07 * i), xycoords="axes fraction",
            fontsize=7.5, color=st["color"], va="top",
            # Three series push this block down into the plotted lines. The box keeps the
            # text readable where it overlaps a fit without moving the annotation off its
            # anchor; colouring it to match the curve also removes the need to re-read
            # the label to know which series a slope belongs to.
            bbox=dict(boxstyle="round,pad=0.25", facecolor=style.SURFACE,
                      edgecolor="none", alpha=0.85),
        )

    ax.axhline(1.0, color=style.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlabel("γ (drafted tokens per step)")
    ax.set_ylabel("speedup normalized to γ = 1")
    ax.set_xticks(sorted(table["num_speculative_tokens"].dropna().unique()))
    ax.margins(x=0.12)

    # Only meaningful as a single number when exactly one model stands on each side.
    # With two MoE curves there is no one "the MoE slope" to subtract, and printing the
    # difference against an average of two unrelated models would invent a quantity.
    if len(arch_slopes["dense"]) == 1 and len(arch_slopes["moe"]) == 1:
        ax.annotate(
            f"slope difference (MoE − dense): "
            f"{arch_slopes['moe'][0] - arch_slopes['dense'][0]:+.3f} per γ",
            xy=(0.99, 0.02), xycoords="axes fraction", ha="right",
            fontsize=7.5, color=style.INK,
        )
    style.annotate_n(ax, int(table["n_prompts"].min()), loc="lower left")
    fig.tight_layout()
    return style.save(fig, outdir, "fig08_gamma_slope_dense_vs_moe")


def _series_groups(table: pd.DataFrame) -> list[tuple[str, str]]:
    """(architecture, model) pairs, dense first, models sorted inside each architecture.

    Deterministic so colour assignment does not depend on log ordering: re-rendering the
    same frame must reproduce the same figure.
    """
    groups: list[tuple[str, str]] = []
    for arch in ("dense", "moe"):
        models = sorted(set(table.loc[table["architecture"] == arch, "target_model"]))
        groups.extend((arch, m) for m in models)
    return groups


def _short(model: str) -> str:
    """Checkpoint name without the org prefix; long enough to stay unambiguous."""
    return model.rsplit("/", 1)[-1]


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
