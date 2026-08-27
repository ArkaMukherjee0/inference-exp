"""Paired bootstraps and aggregation.

Three decisions are baked into the shape of this module, not merely documented in it:

1. **Ratios are geometric.** ``paired_bootstrap_speedup`` is the only public path to a
   speedup number and it is geometric internally. There is no flag to change that. An
   arithmetic mean of ratios reports a gain when there is none: 2x on one prompt and
   0.5x on another averages to 1.25x arithmetically and 1.0x geometrically, and 1.0x is
   the truth.

2. **Comparisons are paired by prompt_id.** Prompts differ from each other far more than
   conditions do; pairing cancels that variation instead of letting it drown the effect.
   Misalignment raises -- it never truncates to the shorter vector, which would silently
   compare unrelated prompts.

3. **Repeats collapse before the bootstrap.** Five repeats of one prompt are five looks
   at one thing, not five independent observations. Bootstrapping over them would
   inflate n fivefold and produce intervals roughly sqrt(5) too tight. The public API
   accepts only per-prompt vectors, so the mistake is unreachable rather than merely
   discouraged.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

# Columns that identify one measurement cell.
_CELL = ["condition_id", "prompt_id"]


class Interval(NamedTuple):
    """A point estimate and its 95% bootstrap interval."""

    point: float
    lo95: float
    hi95: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.4g} [{self.lo95:.4g}, {self.hi95:.4g}] (n={self.n})"


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def median_over_repeats(
    df: pd.DataFrame,
    *,
    value_cols: tuple[str, ...] = ("tpot_ms", "total_ms", "ttft_ms", "acceptance_rate",
                                   "mean_accept_length"),
) -> pd.DataFrame:
    """Collapse repeats within (condition_id, prompt_id) by median.

    Median rather than mean: repeat-level noise is dominated by occasional slow outliers
    -- a scheduler hiccup, a background process waking up -- which is exactly the
    distribution shape where the median is robust and the mean is not.

    Warmup rows must already have been dropped by ``analysis.load.load_runs``; this
    raises if any survived, because averaging a cold first iteration into a steady-state
    measurement is precisely the bug the warmup discipline exists to prevent.
    """
    if "is_warmup" in df.columns and df["is_warmup"].astype(bool).any():
        raise ValueError(
            "median_over_repeats received warmup records. Load with drop_warmup=True; "
            "cold iterations must not enter a steady-state measurement."
        )
    for col in _CELL:
        if col not in df.columns:
            raise ValueError(f"median_over_repeats: missing column {col!r}")

    present = [c for c in value_cols if c in df.columns]
    if not present:
        raise ValueError(f"median_over_repeats: none of {value_cols} present in frame")

    grouped = df.groupby(_CELL, dropna=False)
    out = grouped[present].median()
    out["n_repeats"] = grouped.size()
    # Carry the condition-identifying fields through, so downstream code does not have
    # to re-join to get them.
    carry = [
        c for c in ("stack", "platform", "target_dtype", "spec_method",
                    "num_speculative_tokens", "batch_size", "tensor_parallel_size",
                    "draft_tensor_parallel_size", "target_model", "draft_model",
                    "provenance", "latency_valid", "hostname")
        if c in df.columns
    ]
    if carry:
        out = out.join(grouped[carry].first())
    return out.reset_index()


def align_pair(
    base: pd.DataFrame,
    opt: pd.DataFrame,
    *,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align two per-prompt frames on prompt_id. Raises on any mismatch.

    Interleaved execution means log order is not condition order, so positional
    alignment would compare unrelated prompts and produce a tight, meaningless interval.
    """
    for name, frame in (("base", base), ("opt", opt)):
        if "prompt_id" not in frame.columns:
            raise ValueError(f"align_pair: {name} frame has no prompt_id column")
        if frame["prompt_id"].duplicated().any():
            dupes = frame.loc[frame["prompt_id"].duplicated(), "prompt_id"].unique()[:5]
            raise ValueError(
                f"align_pair: {name} frame has repeated prompt_ids ({list(dupes)}...). "
                "Collapse repeats with median_over_repeats first."
            )
        if value_col not in frame.columns:
            raise ValueError(f"align_pair: {name} frame has no column {value_col!r}")

    b = base.set_index("prompt_id")[value_col]
    o = opt.set_index("prompt_id")[value_col]
    only_base = sorted(set(b.index) - set(o.index))
    only_opt = sorted(set(o.index) - set(b.index))
    if only_base or only_opt:
        raise ValueError(
            "align_pair: prompt sets differ. "
            f"{len(only_base)} only in base (e.g. {only_base[:3]}), "
            f"{len(only_opt)} only in opt (e.g. {only_opt[:3]}). Refusing to truncate to "
            "the intersection -- a condition missing prompts is a gap to explain, not to "
            "silently drop."
        )
    ids = sorted(b.index)
    bv = b.loc[ids].to_numpy(dtype=float)
    ov = o.loc[ids].to_numpy(dtype=float)
    if np.isnan(bv).any() or np.isnan(ov).any():
        raise ValueError(f"align_pair: {value_col!r} contains NaN after alignment")
    return bv, ov, ids


# --------------------------------------------------------------------------------------
# Bootstraps
# --------------------------------------------------------------------------------------


def _resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """One index matrix, applied to *both* vectors -- that is what makes it paired."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def paired_bootstrap_speedup(
    t_base: np.ndarray | list[float],
    t_opt: np.ndarray | list[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Geometric mean of per-prompt speedups, with a paired 95% interval.

    ``t_base`` and ``t_opt`` are per-prompt times (lower is faster), already aligned by
    ``align_pair``. Speedup is ``t_base / t_opt``, so >1 means the optimized condition
    is faster.

    The geometric mean is computed in log space -- ``exp(mean(log(x)))`` -- rather than
    as an n-th root of a product, which underflows to zero for a few hundred ratios
    below 1.0 and returns a confident, silently wrong answer.
    """
    b = np.asarray(t_base, dtype=float)
    o = np.asarray(t_opt, dtype=float)
    if b.shape != o.shape:
        raise ValueError(f"paired_bootstrap_speedup: shapes differ {b.shape} vs {o.shape}")
    if b.ndim != 1 or b.size == 0:
        raise ValueError("paired_bootstrap_speedup: expected non-empty 1-D vectors")
    if np.any(b <= 0) or np.any(o <= 0):
        raise ValueError(
            "paired_bootstrap_speedup: non-positive timings present. A zero or negative "
            "duration means something upstream is broken; it is not a data point."
        )

    log_ratio = np.log(b) - np.log(o)
    point = float(np.exp(log_ratio.mean()))

    idx = _resample_indices(log_ratio.size, n_boot, seed)
    boot = np.exp(log_ratio[idx].mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Interval(point=point, lo95=float(lo), hi95=float(hi), n=int(log_ratio.size))


def paired_bootstrap_delta(
    score_base: np.ndarray | list[float],
    score_opt: np.ndarray | list[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Mean of per-example deltas (opt - base), with a paired 95% interval.

    Used for quality: exact-match is a per-example binary vector and perplexity is a
    per-window value. Differences average arithmetically -- only *ratios* need the
    geometric treatment -- so this is a plain mean, deliberately.
    """
    b = np.asarray(score_base, dtype=float)
    o = np.asarray(score_opt, dtype=float)
    if b.shape != o.shape:
        raise ValueError(f"paired_bootstrap_delta: shapes differ {b.shape} vs {o.shape}")
    if b.ndim != 1 or b.size == 0:
        raise ValueError("paired_bootstrap_delta: expected non-empty 1-D vectors")
    if np.isnan(b).any() or np.isnan(o).any():
        raise ValueError("paired_bootstrap_delta: NaN in input")

    delta = o - b
    point = float(delta.mean())
    idx = _resample_indices(delta.size, n_boot, seed)
    boot = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Interval(point=point, lo95=float(lo), hi95=float(hi), n=int(delta.size))


def speedup_between(
    collapsed: pd.DataFrame,
    *,
    base_condition: str,
    opt_condition: str,
    value_col: str = "tpot_ms",
    n_boot: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Convenience wrapper: two condition ids in a collapsed frame -> one Interval.

    Takes the *collapsed* frame only. There is deliberately no variant of this that
    accepts raw repeat-level rows.
    """
    if "n_repeats" not in collapsed.columns:
        raise ValueError(
            "speedup_between expects the output of median_over_repeats. Passing raw "
            "repeat-level rows would treat repeats as independent observations."
        )
    base = collapsed[collapsed["condition_id"] == base_condition]
    opt = collapsed[collapsed["condition_id"] == opt_condition]
    if base.empty:
        raise ValueError(f"no rows for base condition {base_condition!r}")
    if opt.empty:
        raise ValueError(f"no rows for opt condition {opt_condition!r}")
    bv, ov, _ = align_pair(base, opt, value_col=value_col)
    return paired_bootstrap_speedup(bv, ov, n_boot=n_boot, seed=seed)


def geometric_mean(values: np.ndarray | list[float]) -> float:
    """Geometric mean in log space. Raises on non-positive input."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("geometric_mean of an empty sequence")
    if np.any(arr <= 0):
        raise ValueError("geometric_mean requires strictly positive values")
    return float(np.exp(np.log(arr).mean()))
