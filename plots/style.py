"""Shared figure style: rcParams, palette, CI rendering, saving.

Every figure module imports from here and nothing else sets a style. Two rules govern
this file.

**No network at plot time.** No style downloads, no font fetching, no CDN. The font is
DejaVu Sans, which ships with matplotlib, so a figure renders identically on a machine
with no network and no system fonts installed.

**Seaborn is used for plot types, never for styling.** Its faceting is genuinely better
than a hand-rolled subplot grid (figure 06), so it is worth having -- but its default
aesthetics have changed across minor versions, and a figure whose appearance depends on
the installed version cannot be regenerated identically next month. So ``apply_style()``
owns every rcParam, seaborn is called with ``rc`` we supply, and ``sns.set_theme`` is
never invoked.

Palette
-------
Three categorical slots, taken in fixed order from a validated palette and verified with
the palette validator (all-pairs mode, light surface): worst CVD Delta E 9.2, worst
normal-vision Delta E 24.0. Three is the cap -- the fourth slot would put yellow beside
orange, which fails the all-pairs floor. Every figure here needs at most three series,
which is why the study's figures were specified that way.

The aqua slot sits at 2.74:1 against the light surface, below the 3:1 bar, so the
**relief rule** applies: every series carries a visible direct label and a distinct
marker shape. Identity is never carried by colour alone -- which also happens to be what
a printed, possibly greyscale, report needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # no display, and no attempt to find one

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# -- palette ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7a7975"
GRID = "#e4e3df"

# Fixed order. Assigned by entity, never by rank: a filter that drops a series must not
# repaint the survivors.
SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
NEUTRAL = "#8a8985"        # model curves, reference lines, non-entity marks
REFERENCE = "#52514e"      # the 1.0x line and other "no effect" annotations

# Secondary encoding, so identity survives greyscale printing and CVD.
MARKERS: tuple[str, ...] = ("o", "s", "^")
LINESTYLES: tuple[str, ...] = ("-", "--", "-.")

# Stable role assignments, so the same precision is the same colour in every figure.
DTYPE_ORDER: tuple[str, ...] = ("bf16", "fp8", "w4a16")
DTYPE_LABEL = {"bf16": "BF16", "fp8": "FP8", "w4a16": "W4A16"}


def dtype_color(dtype: str) -> str:
    return SERIES[DTYPE_ORDER.index(dtype) % len(SERIES)]


def dtype_marker(dtype: str) -> str:
    return MARKERS[DTYPE_ORDER.index(dtype) % len(MARKERS)]


def dtype_linestyle(dtype: str) -> str:
    return LINESTYLES[DTYPE_ORDER.index(dtype) % len(LINESTYLES)]


# Ordinal ramp: one hue, light -> dark, for an *ordered* quantity such as gamma.
# Categorical slots encode identity and must never be cycled; gamma is not an identity,
# it is a magnitude, so it gets a ramp instead. The lightest step used is 250, the
# lightest that still clears 2:1 against the light surface.
_ORDINAL_STEPS: tuple[str, ...] = (
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)


def ordinal_ramp(n: int) -> list[str]:
    """``n`` evenly spread steps of the sequential blue ramp, light to dark.

    Use for ordered quantities (gamma, batch size). Unlike the categorical slots this
    has no series cap: a ramp with more steps stays readable because the *order* carries
    the meaning, not the hue.
    """
    if n < 1:
        raise ValueError(f"ordinal_ramp needs n >= 1 (got {n})")
    if n == 1:
        return [_ORDINAL_STEPS[len(_ORDINAL_STEPS) // 2]]
    last = len(_ORDINAL_STEPS) - 1
    return [_ORDINAL_STEPS[round(i * last / (n - 1))] for i in range(n)]


def series_style(index: int) -> dict[str, Any]:
    """Colour + marker + linestyle for the nth entity, in fixed slot order."""
    if index >= len(SERIES):
        raise ValueError(
            f"series index {index} exceeds the validated three-slot palette. A fourth "
            "hue would fail the all-pairs colour-separation floor; fold the extra "
            "series into small multiples instead of generating a colour."
        )
    return {
        "color": SERIES[index],
        "marker": MARKERS[index],
        "linestyle": LINESTYLES[index],
    }


# -- rcParams --------------------------------------------------------------------------

RC: dict[str, Any] = {
    "figure.figsize": (6.4, 4.0),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    # DejaVu ships with matplotlib: no font is ever fetched.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    # Recessive chrome: the data is the figure, the frame is not.
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "legend.frameon": False,
    "errorbar.capsize": 2.5,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,   # real text in the PDF, not outlines -- searchable and selectable
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Install the study's rcParams. Idempotent; call at the top of every render()."""
    matplotlib.rcParams.update(RC)


def seaborn_rc() -> dict[str, Any]:
    """rcParams to hand seaborn explicitly.

    Seaborn functions take ``rc`` so they inherit our style rather than their own
    version-dependent theme.
    """
    return dict(RC)


# -- rendering helpers -----------------------------------------------------------------


def annotate_n(ax: Any, n: int, *, unit: str = "prompts", loc: str = "lower right",
               outside: bool = False) -> None:
    """Sample size, in every figure. A CI without an n is not interpretable.

    ``outside`` places it below the axes, which is what a bar chart needs: bars are
    anchored to the baseline, so anything drawn in a lower corner sits on top of data.
    """
    if outside:
        ax.annotate(
            f"n = {n} {unit}",
            xy=(1.0, -0.16), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5, color=INK_MUTED,
            annotation_clip=False,
        )
        return
    xy = {"lower right": (0.99, 0.01), "lower left": (0.01, 0.01),
          "upper right": (0.99, 0.99), "upper left": (0.01, 0.99)}[loc]
    ax.annotate(
        f"n = {n} {unit}",
        xy=xy, xycoords="axes fraction",
        ha="right" if "right" in loc else "left",
        va="bottom" if "lower" in loc else "top",
        fontsize=7.5, color=INK_MUTED,
    )


def reference_line(ax: Any, y: float = 1.0, *, label: str = "no speedup (1.0x)") -> None:
    """The line a speedup must clear to mean anything."""
    ax.axhline(y, color=REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        label, xy=(0.01, y), xycoords=("axes fraction", "data"),
        xytext=(0, 3), textcoords="offset points",
        fontsize=7.5, color=INK_SECONDARY, va="bottom",
    )


def direct_label(ax: Any, x: float, y: float, text: str, color: str, **kwargs: Any) -> None:
    """Label a series at its end, in ink -- never in the series colour.

    Text wears text tokens; the coloured mark beside it carries identity. This is also
    the relief the aqua slot's sub-3:1 contrast requires.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
        fontsize=8, color=INK, va="center", ha="left",
        **kwargs,
    )
    ax.plot([x], [y], marker="o", markersize=4, color=color, zorder=5, clip_on=False)


def error_band(ax: Any, x: Sequence[float], lo: Sequence[float], hi: Sequence[float],
               color: str, *, alpha: float = 0.15) -> None:
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=2)


def ci_errorbar(ax: Any, x: Any, point: Any, lo: Any, hi: Any, *, color: str,
                marker: str = "o", label: str | None = None, **kwargs: Any) -> Any:
    """Point estimate with an asymmetric 95% interval.

    Bootstrap intervals are not symmetric about the point estimate, so they are drawn
    from the actual percentiles rather than as +/- one number.
    """
    import numpy as np

    point_arr = np.atleast_1d(np.asarray(point, dtype=float))
    lo_arr = np.atleast_1d(np.asarray(lo, dtype=float))
    hi_arr = np.atleast_1d(np.asarray(hi, dtype=float))
    yerr = np.vstack([point_arr - lo_arr, hi_arr - point_arr])
    if (yerr < -1e-9).any():
        raise ValueError(
            "confidence interval does not bracket its point estimate; the numbers are "
            "inconsistent and must not be drawn."
        )
    return ax.errorbar(
        x, point_arr, yerr=np.clip(yerr, 0, None), color=color, marker=marker,
        label=label, linestyle="none", elinewidth=1.2, markeredgecolor=SURFACE,
        markeredgewidth=0.8, zorder=4, **kwargs,
    )


def measured_vs_extrapolated_handles() -> list[Any]:
    """Legend handles distinguishing measured points from extrapolated ones.

    Figure 05 requires this distinction visually. An extrapolated point drawn
    identically to a measured one is a misrepresentation, so extrapolations are open
    markers on dashed lines and the legend says so explicitly.
    """
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], marker="o", color=INK_SECONDARY, linestyle="-",
               markerfacecolor=INK_SECONDARY, markersize=6, label="measured"),
        Line2D([], [], marker="o", color=INK_SECONDARY, linestyle=(0, (4, 3)),
               markerfacecolor="none", markersize=6, label="extrapolated"),
    ]


def save(fig: Figure, outdir: str | Path, name: str) -> Path:
    """Write PDF plus 200-dpi PNG. Returns the PDF path.

    Vector for the document, raster for review and for pasting into a message. No title
    is drawn inside the figure -- captions live in the report, where they can be edited
    without re-running anything.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{name}.pdf"
    png = out / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf


def new_figure(*, figsize: tuple[float, float] | None = None, **kwargs: Any) -> tuple[Figure, Any]:
    apply_style()
    return plt.subplots(figsize=figsize or RC["figure.figsize"], **kwargs)
