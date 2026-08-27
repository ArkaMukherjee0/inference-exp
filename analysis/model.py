"""The analytical model: Leviathan speculative speedup, optimal gamma, roofline.

Used as a *prediction*, not a curve fit. Both inputs are measured independently and
beforehand -- alpha from acceptance counts, c from isolated batch-1 step timings -- so a
match between predicted and measured is evidence rather than a tautology. That is only
true because ``measure_c`` refuses to look at parameter counts; see its docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_speedup(alpha: float, gamma: int, c: float) -> float:
    """Leviathan et al. expected speedup from speculative decoding.

            1 - alpha**(gamma+1)
        S = --------------------
            (1 - alpha)(gamma*c + 1)

    The numerator is the expected tokens produced per verification round; the
    denominator is what the round costs. Raising gamma saturates the numerator (each
    extra guess is less likely to survive) while the denominator keeps climbing
    linearly, which is why an optimum exists.

    ``alpha`` is the per-token acceptance probability, ``gamma`` the number of tokens
    drafted per round, ``c`` the cost of one draft step relative to one target step.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1] (got {alpha})")
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0 (got {gamma})")
    if c < 0:
        raise ValueError(f"c must be >= 0 (got {c})")

    denom_cost = gamma * c + 1.0
    if alpha == 1.0:
        # Both halves of the fraction go to zero here. The limit is exact and reachable
        # in practice: an n-gram drafter on repetitive text genuinely accepts every
        # token over a window, and an acceptance rate computed from integer counts lands
        # exactly on 1.0 whenever nothing was rejected. Returning inf/nan would poison a
        # whole aggregation, and nudging alpha to 0.999999 would be wrong by an unbounded
        # amount at large gamma.
        return (gamma + 1.0) / denom_cost
    return (1.0 - alpha ** (gamma + 1)) / ((1.0 - alpha) * denom_cost)


def optimal_gamma(alpha: float, c: float, gamma_max: int = 16) -> int:
    """The integer gamma maximizing ``expected_speedup``.

    Enumeration, not calculus. The derivative has no closed-form root for integer gamma,
    and relaxing to a continuous variable then rounding is not guaranteed to land on the
    integer maximum. The search space is sixteen evaluations of a closed-form expression:
    exact, obviously correct on inspection, and immeasurably fast. Optimizing it would
    trade a correctness guarantee for nothing.
    """
    if gamma_max < 1:
        raise ValueError(f"gamma_max must be >= 1 (got {gamma_max})")
    candidates = range(1, gamma_max + 1)
    return max(candidates, key=lambda g: expected_speedup(alpha, g, c))


def measure_c(draft_step_ms: float, target_step_ms: float) -> float:
    """Relative cost of a draft step, from two measured batch-1 step times.

    This function takes timings and nothing else. It has no code path that reads a model
    config, and that is deliberate: estimating ``c`` from a parameter-count ratio is
    wrong in the exact regime this study operates in. A 0.5B draft beside a 7B target is
    nowhere near 14x cheaper per step at batch 1, because fixed per-step overheads --
    kernel launches, sampling, scheduler work, Python dispatch -- do not shrink with the
    model and come to dominate for small ones. Using the parameter ratio would inflate
    predicted speedup precisely where we are trying to test the prediction.
    """
    if draft_step_ms <= 0 or target_step_ms <= 0:
        raise ValueError(
            f"step times must be > 0 (draft={draft_step_ms}, target={target_step_ms})"
        )
    if draft_step_ms >= target_step_ms:
        raise ValueError(
            f"draft step ({draft_step_ms} ms) is not faster than the target step "
            f"({target_step_ms} ms), so c >= 1 and speculation cannot pay. Check that "
            "the draft and target models were not swapped."
        )
    return draft_step_ms / target_step_ms


def ridge_point(tflops: float, bandwidth_gbs: float) -> float:
    """Arithmetic intensity at which a machine stops being memory-bound, in FLOP/byte.

    Both arguments must come from ``bench/micro.py`` -- achieved numbers from a
    STREAM-style triad and a large GEMM sweep. Never a spec-sheet constant: peak-to-
    achieved ratios differ substantially between an H100 and a desktop CPU, and figure
    05 places platforms along this very axis, so quoted figures would distort the axis
    in a platform-dependent direction.
    """
    if tflops <= 0 or bandwidth_gbs <= 0:
        raise ValueError(f"tflops and bandwidth must be > 0 (got {tflops}, {bandwidth_gbs})")
    # TFLOP/s -> FLOP/s is 1e12; GB/s -> byte/s is 1e9.
    return (tflops * 1e12) / (bandwidth_gbs * 1e9)


def predicted_vs_measured(
    df: pd.DataFrame,
    *,
    c_by_pair: dict[tuple[str, str], float] | None = None,
    c_col: str = "c",
    alpha_col: str = "acceptance_rate",
    gamma_col: str = "num_speculative_tokens",
    measured_col: str = "speedup",
) -> pd.DataFrame:
    """Add ``predicted`` and ``residual`` columns to a per-condition frame.

    ``c`` comes either from a ``c`` column already on the frame or from
    ``c_by_pair[(target_model, draft_model)]``. It is never defaulted: a missing ``c``
    raises, because a prediction made with a guessed cost ratio is not a prediction.
    """
    out = df.copy()
    for col in (alpha_col, gamma_col):
        if col not in out.columns:
            raise ValueError(f"predicted_vs_measured: missing column {col!r}")

    if c_col not in out.columns:
        if c_by_pair is None:
            raise ValueError(
                "predicted_vs_measured: no 'c' column and no c_by_pair mapping. Measure c "
                "for every draft/target pair with measure_c before predicting."
            )
        for col in ("target_model", "draft_model"):
            if col not in out.columns:
                raise ValueError(f"predicted_vs_measured: missing column {col!r} for c lookup")

        def _lookup(row: pd.Series) -> float:
            key = (row["target_model"], row["draft_model"])
            if key not in c_by_pair:
                raise ValueError(f"no measured c for draft/target pair {key}")
            return c_by_pair[key]

        out[c_col] = out.apply(_lookup, axis=1)

    if out[[alpha_col, gamma_col, c_col]].isna().any().any():
        raise ValueError(
            "predicted_vs_measured: NaN in alpha, gamma or c. Every prediction input must "
            "be a real measurement."
        )

    out["predicted"] = [
        expected_speedup(float(a), int(g), float(c))
        for a, g, c in zip(out[alpha_col], out[gamma_col], out[c_col])
    ]
    if measured_col in out.columns:
        out["residual"] = out[measured_col] - out["predicted"]
    return out


def speedup_curve(alpha: float, c: float, gammas: range | list[int]) -> pd.DataFrame:
    """The predicted curve, for overlaying on measured points in figure 04."""
    gs = list(gammas)
    return pd.DataFrame({
        "num_speculative_tokens": gs,
        "predicted": [expected_speedup(alpha, g, c) for g in gs],
    })


def alpha_from_histogram(hist: list[int], gamma: int) -> float:
    """Per-token acceptance probability implied by an accepted-run-length histogram.

    Under the geometric model behind ``expected_speedup``, each drafted token is
    accepted with probability alpha independently, so the expected accepted run length
    determines alpha. We invert the relation numerically rather than using
    ``accepted / proposed``, which is the *observed* rate and equals alpha only when
    every position is drafted independently of the last -- true for a draft model, not
    for a prefix-matching n-gram drafter.
    """
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1 (got {gamma})")
    if len(hist) != gamma + 1:
        raise ValueError(f"histogram length {len(hist)} != gamma + 1 ({gamma + 1})")
    steps = sum(hist)
    if steps <= 0:
        raise ValueError("histogram sums to zero steps")

    mean_run = sum(k * n for k, n in enumerate(hist)) / steps
    if mean_run >= gamma:
        return 1.0

    # E[run] = sum_{k=1..gamma} alpha**k, monotone in alpha -> bisection is exact enough
    # and cannot diverge, unlike a Newton step near alpha = 1.
    def expected_run(a: float) -> float:
        return sum(a ** k for k in range(1, gamma + 1))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if expected_run(mid) < mean_run:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def roofline_bound(intensity: float, tflops: float, bandwidth_gbs: float) -> float:
    """Achievable FLOP/s at a given arithmetic intensity, in TFLOP/s.

    The roofline itself: ``min(peak_compute, intensity * bandwidth)``, with both
    "peaks" being measured achieved values.
    """
    if intensity <= 0:
        raise ValueError(f"intensity must be > 0 (got {intensity})")
    memory_bound = intensity * bandwidth_gbs * 1e9
    return float(min(tflops * 1e12, memory_bound) / 1e12)


def np_expected_speedup(alpha: np.ndarray, gamma: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorized ``expected_speedup`` for plotting smooth curves.

    Kept separate from the scalar version so the scalar one stays readable and stays the
    thing unit tests check against hand-computed values.
    """
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    c = np.asarray(c, dtype=float)
    denom_cost = gamma * c + 1.0
    limit = (gamma + 1.0) / denom_cost
    with np.errstate(divide="ignore", invalid="ignore"):
        general = (1.0 - alpha ** (gamma + 1)) / ((1.0 - alpha) * denom_cost)
    return np.where(alpha >= 1.0, limit, general)
