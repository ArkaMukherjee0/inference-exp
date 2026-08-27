"""Statistics tests.

The acceptance criterion from the spec is the coverage test: on synthetic data with a
known 2.0x ratio and known noise, the interval must cover 2.0 at roughly the nominal
rate over repeated trials. An interval that is too tight is the specific failure that
makes a benchmark study wrong while looking rigorous, so it is tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.stats import (
    align_pair,
    geometric_mean,
    median_over_repeats,
    paired_bootstrap_delta,
    paired_bootstrap_speedup,
    speedup_between,
)


# -- the acceptance criterion ----------------------------------------------------------


def test_ci_covers_known_ratio_at_nominal_rate():
    """95% intervals should cover the truth about 95% of the time. Not 60%, not 100%."""
    rng = np.random.default_rng(12345)
    true_ratio = 2.0
    n_prompts, n_trials = 60, 300
    covered = 0

    for trial in range(n_trials):
        base = rng.lognormal(mean=np.log(20.0), sigma=0.35, size=n_prompts)
        opt = base / true_ratio * np.exp(rng.normal(0, 0.08, size=n_prompts))
        ci = paired_bootstrap_speedup(base, opt, n_boot=800, seed=trial)
        if ci.lo95 <= true_ratio <= ci.hi95:
            covered += 1

    rate = covered / n_trials
    assert 0.88 <= rate <= 0.99, f"coverage {rate:.3f} is not near the nominal 95%"


def test_point_estimate_recovers_known_ratio():
    rng = np.random.default_rng(7)
    base = rng.lognormal(np.log(20.0), 0.4, size=400)
    opt = base / 2.0
    ci = paired_bootstrap_speedup(base, opt, n_boot=500)
    assert ci.point == pytest.approx(2.0, rel=1e-9)


# -- geometric, not arithmetic ---------------------------------------------------------


def test_geometric_mean_of_symmetric_ratios_is_one():
    """2x and 0.5x is no change. The arithmetic mean would claim 1.25x."""
    base = np.array([10.0, 10.0])
    opt = np.array([5.0, 20.0])       # ratios 2.0 and 0.5
    ci = paired_bootstrap_speedup(base, opt, n_boot=200)
    assert ci.point == pytest.approx(1.0, rel=1e-12)
    arithmetic = np.mean(base / opt)
    assert arithmetic == pytest.approx(1.25)


def test_geometric_mean_survives_many_small_ratios():
    """The naive product form underflows to 0.0 here and returns a confident wrong answer.

    A double underflows below about 0.5**1075, so this is the sample size at which
    prod()**(1/n) silently starts reporting a geometric mean of exactly zero. The log-space
    form is unaffected.
    """
    values = np.full(1200, 0.5)
    assert geometric_mean(values) == pytest.approx(0.5, rel=1e-9)
    assert np.prod(values) == 0.0
    assert np.prod(values) ** (1 / values.size) == 0.0


def test_geometric_mean_rejects_non_positive():
    with pytest.raises(ValueError):
        geometric_mean([1.0, 0.0, 2.0])


def test_non_positive_timings_raise():
    with pytest.raises(ValueError, match="non-positive"):
        paired_bootstrap_speedup([1.0, 2.0], [1.0, 0.0])


# -- pairing ---------------------------------------------------------------------------


def test_align_pair_raises_on_differing_prompt_sets():
    base = pd.DataFrame({"prompt_id": ["a", "b", "c"], "tpot_ms": [1.0, 2.0, 3.0]})
    opt = pd.DataFrame({"prompt_id": ["a", "b"], "tpot_ms": [1.0, 2.0]})
    with pytest.raises(ValueError, match="prompt sets differ"):
        align_pair(base, opt, value_col="tpot_ms")


def test_align_pair_reorders_rather_than_zipping():
    """Interleaved logs are not in condition order; positional pairing would be wrong."""
    base = pd.DataFrame({"prompt_id": ["a", "b", "c"], "tpot_ms": [10.0, 20.0, 30.0]})
    opt = pd.DataFrame({"prompt_id": ["c", "a", "b"], "tpot_ms": [15.0, 5.0, 10.0]})
    bv, ov, ids = align_pair(base, opt, value_col="tpot_ms")
    assert ids == ["a", "b", "c"]
    assert list(bv) == [10.0, 20.0, 30.0]
    assert list(ov) == [5.0, 10.0, 15.0]


def test_align_pair_rejects_uncollapsed_repeats():
    base = pd.DataFrame({"prompt_id": ["a", "a"], "tpot_ms": [1.0, 1.1]})
    opt = pd.DataFrame({"prompt_id": ["a", "a"], "tpot_ms": [0.5, 0.6]})
    with pytest.raises(ValueError, match="repeated prompt_ids"):
        align_pair(base, opt, value_col="tpot_ms")


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shapes differ"):
        paired_bootstrap_speedup([1.0, 2.0, 3.0], [1.0, 2.0])


# -- collapse before bootstrap ---------------------------------------------------------


def _repeat_frame(n_prompts=20, n_repeats=5, ratio=2.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prompts):
        base_time = 20.0 * (1 + 0.3 * rng.standard_normal())
        for r in range(n_repeats):
            rows.append({"condition_id": "base", "prompt_id": f"p{p}", "repeat_idx": r,
                         "tpot_ms": abs(base_time) * (1 + 0.02 * rng.standard_normal()),
                         "is_warmup": False})
            rows.append({"condition_id": "opt", "prompt_id": f"p{p}", "repeat_idx": r,
                         "tpot_ms": abs(base_time) / ratio * (1 + 0.02 * rng.standard_normal()),
                         "is_warmup": False})
    return pd.DataFrame(rows)


def test_median_over_repeats_collapses_to_one_row_per_cell():
    df = _repeat_frame()
    out = median_over_repeats(df)
    assert len(out) == df["condition_id"].nunique() * df["prompt_id"].nunique()
    assert (out["n_repeats"] == 5).all()


def test_median_over_repeats_rejects_warmups():
    """Averaging a cold iteration into a steady-state measurement is the bug."""
    df = _repeat_frame()
    df.loc[0, "is_warmup"] = True
    with pytest.raises(ValueError, match="warmup"):
        median_over_repeats(df)


def test_speedup_between_refuses_uncollapsed_frame():
    """The API shape is the guard: there is no public path from repeats to a CI."""
    df = _repeat_frame()
    with pytest.raises(ValueError, match="median_over_repeats"):
        speedup_between(df, base_condition="base", opt_condition="opt")


def test_treating_repeats_as_independent_would_tighten_the_ci():
    """Demonstrates the failure the API prevents, so the guard has a reason on record."""
    df = _repeat_frame(n_prompts=25, n_repeats=5, ratio=2.0, seed=3)
    collapsed = median_over_repeats(df)
    correct = speedup_between(collapsed, base_condition="base", opt_condition="opt")

    # The wrong way: bootstrap over all repeat-level rows as if independent.
    base = df[df["condition_id"] == "base"].sort_values(["prompt_id", "repeat_idx"])
    opt = df[df["condition_id"] == "opt"].sort_values(["prompt_id", "repeat_idx"])
    wrong = paired_bootstrap_speedup(base["tpot_ms"].to_numpy(), opt["tpot_ms"].to_numpy())

    correct_width = correct.hi95 - correct.lo95
    wrong_width = wrong.hi95 - wrong.lo95
    assert wrong_width < correct_width, (
        "inflating n by treating repeats as independent should narrow the interval; "
        "if it does not, this test no longer demonstrates the hazard"
    )
    assert correct.n == 25 and wrong.n == 125


def test_speedup_between_recovers_designed_ratio():
    df = _repeat_frame(n_prompts=40, ratio=2.0, seed=11)
    collapsed = median_over_repeats(df)
    ci = speedup_between(collapsed, base_condition="base", opt_condition="opt")
    assert ci.lo95 <= 2.0 <= ci.hi95
    assert ci.point == pytest.approx(2.0, rel=0.02)


# -- deltas ----------------------------------------------------------------------------


def test_paired_delta_on_binary_scores():
    base = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=float)
    opt = base.copy()
    opt[2] = 1  # one example improved
    ci = paired_bootstrap_delta(base, opt)
    assert ci.point == pytest.approx(0.1)
    assert ci.lo95 >= 0.0


def test_paired_delta_zero_when_identical():
    v = np.array([1.0, 0.0, 1.0, 1.0])
    ci = paired_bootstrap_delta(v, v)
    assert ci.point == 0.0 and ci.lo95 == 0.0 and ci.hi95 == 0.0


def test_bootstrap_is_reproducible_under_seed():
    rng = np.random.default_rng(1)
    base = rng.lognormal(np.log(20), 0.3, 50)
    opt = base / 1.5
    a = paired_bootstrap_speedup(base, opt, seed=99, n_boot=500)
    b = paired_bootstrap_speedup(base, opt, seed=99, n_boot=500)
    assert a == b
