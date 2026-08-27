"""Figure 04 -- analytical model against measurement.

Measured speedup per gamma, with the Leviathan curve overlaid and a residual panel
beneath sharing the x axis.

The residual panel is the real content. A curve drawn through points always looks
convincing at figure scale; the residuals show whether the model is wrong in a
*systematic* direction, which is what would tell us it is missing a real cost. Both
model inputs -- alpha from acceptance counts, c from isolated batch-1 step timings --
are measured independently of the speedup being predicted, so agreement here is
evidence rather than a fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.derive import acceptance_table, select_model, speedup_table
from analysis.load import assert_single_stack, require_measured
from analysis.model import expected_speedup, np_expected_speedup, optimal_gamma
from plots import style


def render(
    df: pd.DataFrame,
    outdir: Path,
    *,
    c: float | None = None,
    c_by_pair: dict[tuple[str, str], float] | None = None,
    target_model: str | None = None,
) -> Path:
    require_measured(df)
    assert_single_stack(df, "fig04")

    table = select_model(speedup_table(df), target_model)
    table = table[table["batch_size"] == table["batch_size"].min()]
    accept = acceptance_table(df).set_index("condition_id")
    table = table.join(accept[["acceptance_rate", "mean_accept_length",
                               "accept_length_histogram"]], on="condition_id")

    c_value = _resolve_c(table, c, c_by_pair)

    fig, (ax, ax_res) = _panels()
    dtypes = [d for d in style.DTYPE_ORDER if d in set(table["target_dtype"])]

    all_residuals = []
    summaries: list[tuple[str, str]] = []
    optima: dict[int, list[str]] = {}

    for dtype in dtypes:
        sub = table[table["target_dtype"] == dtype].sort_values("num_speculative_tokens")
        if sub.empty:
            continue
        color = style.dtype_color(dtype)
        gammas = sub["num_speculative_tokens"].to_numpy(dtype=float)
        measured = sub["speedup"].to_numpy(dtype=float)

        style.ci_errorbar(
            ax, gammas, measured, sub["speedup_lo95"], sub["speedup_hi95"],
            color=color, marker=style.dtype_marker(dtype),
        )

        alpha = float(np.average(sub["acceptance_rate"], weights=sub["n_prompts"]))
        # Continuous gamma, so the curve is smooth. Rounding gamma to an integer inside
        # this call would draw a staircase and hide exactly the curvature -- the
        # saturating numerator against the linear denominator -- that the figure exists
        # to show.
        smooth = np.linspace(gammas.min(), gammas.max(), 200)
        predicted_smooth = np_expected_speedup(
            np.full_like(smooth, alpha), smooth, np.full_like(smooth, c_value)
        )
        ax.plot(smooth, predicted_smooth, color=color, linewidth=1.3,
                linestyle=style.dtype_linestyle(dtype), alpha=0.75, zorder=3)

        # Residuals are evaluated at the integer gammas that were actually measured.
        predicted = np.array([expected_speedup(alpha, int(g), c_value) for g in gammas])
        residual = measured - predicted
        all_residuals.append(residual)
        ax_res.plot(gammas, residual, color=color, marker=style.dtype_marker(dtype),
                    linestyle="none", markersize=5,
                    markeredgecolor=style.SURFACE, markeredgewidth=0.8, zorder=4)

        g_opt = optimal_gamma(alpha, c_value, gamma_max=int(gammas.max()))
        optima.setdefault(g_opt, []).append(style.DTYPE_LABEL[dtype])
        summaries.append((
            f"{style.DTYPE_LABEL[dtype]}:  α = {alpha:.3f},  predicted γ* = {g_opt}",
            color,
        ))
        style.direct_label(ax, gammas[-1], measured[-1], style.DTYPE_LABEL[dtype], color)

    # One vline per distinct optimum, labelled once. Drawing one per precision stacked
    # three identical lines and three overlapping labels on the same x.
    for g_opt, labels in optima.items():
        ax.axvline(g_opt, color=style.NEUTRAL, linewidth=0.9, linestyle=(0, (2, 3)),
                   alpha=0.7, zorder=1)
        ax.annotate(
            f"predicted γ* = {g_opt}\n({', '.join(labels)})",
            xy=(g_opt, 0.58), xycoords=("data", "axes fraction"),
            xytext=(5, 0), textcoords="offset points",
            fontsize=7, color=style.INK_SECONDARY, va="center",
        )

    style.reference_line(ax, 1.0)
    ax_res.axhline(0.0, color=style.REFERENCE, linewidth=1.0, zorder=2)

    ax.set_ylabel("speedup (geometric mean)")
    ax_res.set_ylabel("residual\n(measured − predicted)")
    ax_res.set_xlabel("γ (drafted tokens per step)")
    ax.tick_params(labelbottom=False)

    gammas_all = sorted(table["num_speculative_tokens"].dropna().unique())
    ax_res.set_xticks(gammas_all)
    ax.set_xticks(gammas_all)

    # Headroom for the text block, so it never lands on a data point.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)
    ax.annotate(
        f"model:  S = (1 − α^(γ+1)) / ((1 − α)(γc + 1)),   measured c = {c_value:.4f}",
        xy=(0.01, 0.98), xycoords="axes fraction", va="top", fontsize=7.5,
        color=style.INK_SECONDARY,
    )
    # Stacked, one line each: three annotations anchored to the same point overlapped
    # into an unreadable pile.
    for i, (text, color) in enumerate(summaries):
        ax.annotate(
            text, xy=(0.01, 0.92 - 0.055 * i), xycoords="axes fraction", va="top",
            fontsize=7, color=style.INK,
        )
        ax.plot([0.005], [0.905 - 0.055 * i], transform=ax.transAxes, marker="s",
                markersize=4, color=color, clip_on=False)

    if all_residuals:
        pooled = np.concatenate(all_residuals)
        ax_res.annotate(
            f"mean residual {pooled.mean():+.3f}, RMS {np.sqrt((pooled ** 2).mean()):.3f}",
            xy=(0.99, 0.08), xycoords="axes fraction", ha="right", fontsize=7,
            color=style.INK_MUTED,
        )
    style.annotate_n(ax_res, int(table["n_prompts"].min()), outside=True)
    # No tight_layout here: it would override the gridspec's deliberate panel spacing.
    # savefig(bbox="tight") in style.save already trims the margins.
    return style.save(fig, outdir, "fig04_model_vs_measured")


def _panels():
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(6.6, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    return fig, axes


def _resolve_c(table: pd.DataFrame, c: float | None,
               c_by_pair: dict[tuple[str, str], float] | None) -> float:
    """The measured draft/target cost ratio. Never defaulted."""
    if c is not None:
        return float(c)
    if c_by_pair:
        pairs = {(r["target_model"], r["draft_model"]) for _, r in table.iterrows()}
        values = {c_by_pair[p] for p in pairs if p in c_by_pair}
        if len(values) == 1:
            return float(next(iter(values)))
        if not values:
            raise ValueError(f"no measured c for any of the draft/target pairs {pairs}")
        raise ValueError(
            f"multiple distinct c values apply to this figure ({sorted(values)}); render "
            "one draft/target pair per figure rather than averaging cost ratios."
        )
    raise ValueError(
        "fig04 needs a measured c. Run analysis.model.measure_c on isolated batch-1 step "
        "timings for this draft/target pair and pass it in -- the model is only a "
        "prediction if its inputs were measured independently."
    )
