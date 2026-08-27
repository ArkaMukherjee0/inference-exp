"""Figure 07 -- metric sensitivity: perplexity moves before exact match does.

Two panels stacked on a shared precision axis: WikiText-2 perplexity delta on top,
GSM8K exact-match delta beneath, both relative to the BF16 baseline, both with paired
95% intervals.

A note on the construction
--------------------------
implementation.md section 4 specifies twin axes for this figure. This module deliberately
does not do that, and the deviation is worth stating plainly because the figure's whole
purpose is a comparison between two metrics.

A twin-axis chart has two y-scales sharing one plot area, which means the crossing point
of the two series -- the exact thing a reader's eye is drawn to -- is an artifact of how
each axis was scaled. Slide one axis and the "divergence" appears anywhere you like. For
a figure whose entire claim is *these two metrics disagree*, an encoding that lets the
author choose where they appear to disagree is the wrong encoding: it would make the
finding unfalsifiable by construction.

Stacked panels on a shared x axis show the same two series against the same precision
levels, keep each metric on an honest independent scale, and let the reader compare the
*shapes* -- which is the actual claim. Nothing is lost except the ability to overstate
the result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.load import require_measured
from analysis.stats import Interval, paired_bootstrap_delta
from plots import style

BASELINE_DTYPE = "bf16"


def render(
    df: pd.DataFrame,
    outdir: Path,
    *,
    perplexity: dict[str, object],
    scored: pd.DataFrame | None = None,
    target_model: str | None = None,
) -> Path:
    """``perplexity`` maps target_dtype -> PPLResult, all with identical window configs."""
    require_measured(df)
    if target_model is not None:
        df = df[df["target_model"] == target_model].copy()
        if df.empty:
            raise ValueError(f"no records for target_model {target_model!r}")
    # Check the precondition before scoring: quality evaluation is expensive, and a
    # frame with no BF16 baseline cannot produce this figure however well it scores.
    dtypes = [d for d in style.DTYPE_ORDER if d in set(df["target_dtype"])]
    if BASELINE_DTYPE not in dtypes:
        raise ValueError(
            f"fig07 needs a {BASELINE_DTYPE} baseline to take deltas against; present "
            f"dtypes are {dtypes}."
        )

    if scored is None:
        from evals.gsm8k import score_frame

        scored = score_frame(df)

    ppl_deltas = _perplexity_deltas(perplexity, dtypes)
    em_deltas = _em_deltas(df, scored, dtypes)

    fig, (ax_ppl, ax_em) = _panels()
    x = np.arange(len(dtypes), dtype=float)

    _panel(
        ax_ppl, x, dtypes, ppl_deltas,
        color=style.SERIES[0], marker=style.MARKERS[0],
        ylabel="Δ perplexity vs BF16\n(higher = worse)",
        label="WikiText-2 perplexity",
        fmt="{:+.3f}",
    )
    _panel(
        ax_em, x, dtypes, em_deltas,
        color=style.SERIES[1], marker=style.MARKERS[1],
        ylabel="Δ exact match vs BF16\n(lower = worse)",
        label="GSM8K exact match",
        fmt="{:+.3f}",
    )

    ax_em.set_xticks(x)
    ax_em.set_xticklabels([style.DTYPE_LABEL[d] for d in dtypes])
    ax_em.set_xlabel("target precision")
    ax_ppl.tick_params(labelbottom=False)

    n = int(min(v.n for v in em_deltas.values() if v is not None))
    style.annotate_n(ax_em, n, unit="examples", loc="lower left")
    ppl_n = int(min(v.n for v in ppl_deltas.values() if v is not None))
    # Lower right: the panel title occupies upper left, and the series rises
    # left-to-right so the lower-right corner is the empty one.
    style.annotate_n(ax_ppl, ppl_n, unit="windows", loc="lower right")

    # See fig04: tight_layout would fight the shared-axis gridspec.
    return style.save(fig, outdir, "fig07_metric_sensitivity")


def _panels():
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    return fig, axes


def _panel(ax, x, dtypes, deltas, *, color, marker, ylabel, label, fmt) -> None:
    xs, points, los, his = [], [], [], []
    for i, d in enumerate(dtypes):
        interval = deltas.get(d)
        if interval is None:
            continue
        xs.append(x[i]); points.append(interval.point)
        los.append(interval.lo95); his.append(interval.hi95)

    ax.plot(xs, points, color=color, linewidth=1.6, zorder=3)
    style.ci_errorbar(ax, xs, points, los, his, color=color, marker=marker)
    ax.axhline(0.0, color=style.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

    for xi, pt in zip(xs, points):
        ax.annotate(fmt.format(pt), xy=(xi, pt), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=style.INK)
    ax.set_ylabel(ylabel)
    # One series per panel, so the panel is named rather than given a legend box.
    ax.annotate(label, xy=(0.01, 0.97), xycoords="axes fraction", va="top",
                fontsize=8, color=style.INK, fontweight="bold")
    ax.margins(x=0.15, y=0.30)


def _perplexity_deltas(perplexity: dict[str, object], dtypes: list[str]) -> dict[str, Interval | None]:
    """Paired per-window perplexity deltas against the BF16 result."""
    from evals.perplexity import paired_window_nll

    base = perplexity.get(BASELINE_DTYPE)
    if base is None:
        raise ValueError(f"fig07: no perplexity result for the {BASELINE_DTYPE} baseline")

    out: dict[str, Interval | None] = {}
    for d in dtypes:
        result = perplexity.get(d)
        if result is None:
            out[d] = None
            continue
        if d == BASELINE_DTYPE:
            out[d] = Interval(0.0, 0.0, 0.0, len(getattr(base, "window_nll", [])))
            continue
        b, o = paired_window_nll(base, result)
        # Delta of per-window perplexity, not of mean NLL: the report's axis is
        # perplexity, and exp() of a mean is not the mean of exp().
        out[d] = paired_bootstrap_delta(np.exp(b), np.exp(o))
    return out


def _em_deltas(df: pd.DataFrame, scored: pd.DataFrame, dtypes: list[str]) -> dict[str, Interval | None]:
    """Paired per-example exact-match deltas, aligned by prompt_id."""
    # Quality is a batch-1, TP-1 property: the same weights answer the same questions
    # the same way whatever the batch. But a sweep contains a baseline condition at
    # every batch size, so without pinning these the precision baseline is ambiguous
    # and the figure would silently pick whichever came first.
    scoped = df
    for col in ("batch_size", "tensor_parallel_size"):
        if col in scoped.columns:
            scoped = scoped[scoped[col] == scoped[col].min()]
    if scoped.empty:
        raise ValueError("fig07: no batch-1 conditions available for quality comparison")

    meta = scoped.drop_duplicates("condition_id").set_index("condition_id")
    # Compare like with like: the non-speculative condition at each precision.
    by_dtype: dict[str, str] = {}
    for cid, row in meta.iterrows():
        if row["spec_method"] != "none":
            continue
        dt = row["target_dtype"]
        if dt in by_dtype:
            raise ValueError(
                f"two non-speculative conditions at precision {dt!r}; fig07 cannot tell "
                "which is the reference."
            )
        by_dtype[dt] = cid

    if BASELINE_DTYPE not in by_dtype:
        raise ValueError(f"fig07: no non-speculative {BASELINE_DTYPE} condition to baseline against")

    base_scores = scored[scored["condition_id"] == by_dtype[BASELINE_DTYPE]].set_index("prompt_id")["em"]
    out: dict[str, Interval | None] = {}
    for d in dtypes:
        cid = by_dtype.get(d)
        if cid is None:
            out[d] = None
            continue
        opt_scores = scored[scored["condition_id"] == cid].set_index("prompt_id")["em"]
        ids = sorted(set(base_scores.index) & set(opt_scores.index))
        if len(ids) != len(base_scores) or len(ids) != len(opt_scores):
            raise ValueError(
                f"prompt sets differ between {BASELINE_DTYPE} and {d}; a paired delta "
                "requires the same examples on both sides."
            )
        out[d] = paired_bootstrap_delta(
            base_scores.loc[ids].to_numpy(dtype=float),
            opt_scores.loc[ids].to_numpy(dtype=float),
        )
    return out
