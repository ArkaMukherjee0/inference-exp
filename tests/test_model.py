"""Analytical-model tests: hand-computed values, limits, and monotonicity."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.model import (
    alpha_from_histogram,
    expected_speedup,
    measure_c,
    np_expected_speedup,
    optimal_gamma,
    ridge_point,
    roofline_bound,
)


def _hand(alpha: float, gamma: int, c: float) -> float:
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))


@pytest.mark.parametrize("alpha", [0.5, 0.8])
@pytest.mark.parametrize("gamma", [1, 4])
@pytest.mark.parametrize("c", [0.05, 0.2])
def test_against_hand_computed_values(alpha, gamma, c):
    assert expected_speedup(alpha, gamma, c) == pytest.approx(_hand(alpha, gamma, c), rel=1e-12)


def test_specific_known_value():
    # alpha=0.8, gamma=4, c=0.05: numerator 1-0.8^5 = 0.67232,
    # denominator (0.2)(1.2) = 0.24 -> 2.80133...
    assert expected_speedup(0.8, 4, 0.05) == pytest.approx(0.67232 / 0.24, rel=1e-9)


def test_alpha_one_uses_the_limit_not_a_division_by_zero():
    """Reachable in practice: an n-gram drafter on repetitive text accepts everything."""
    value = expected_speedup(1.0, 4, 0.05)
    assert np.isfinite(value)
    assert value == pytest.approx(5.0 / 1.2, rel=1e-12)


def test_alpha_approaching_one_converges_to_the_limit():
    limit = expected_speedup(1.0, 4, 0.05)
    near = expected_speedup(0.999999, 4, 0.05)
    assert near == pytest.approx(limit, rel=1e-4)


def test_alpha_zero_gives_no_speedup_beyond_the_bonus_token():
    # Nothing accepted: one token per round, at a cost of gamma*c + 1.
    assert expected_speedup(0.0, 4, 0.05) == pytest.approx(1.0 / 1.2, rel=1e-12)


def test_speedup_increases_in_alpha():
    values = [expected_speedup(a, 4, 0.1) for a in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_speedup_decreases_in_c():
    values = [expected_speedup(0.8, 4, c) for c in (0.01, 0.05, 0.1, 0.3, 0.6)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_speedup_turns_over_in_gamma():
    """The optimum exists: the numerator saturates while the denominator keeps climbing."""
    values = [expected_speedup(0.6, g, 0.25) for g in range(1, 17)]
    peak = int(np.argmax(values))
    assert 0 < peak < len(values) - 1, "expected an interior maximum in gamma"


def test_optimal_gamma_matches_brute_force():
    for alpha in (0.3, 0.5, 0.7, 0.85, 0.95):
        for c in (0.02, 0.1, 0.3):
            best = optimal_gamma(alpha, c, gamma_max=16)
            brute = max(range(1, 17), key=lambda g: expected_speedup(alpha, g, c))
            assert best == brute, (alpha, c)


def test_optimal_gamma_grows_with_alpha():
    lows = optimal_gamma(0.4, 0.05)
    highs = optimal_gamma(0.9, 0.05)
    assert highs >= lows


def test_optimal_gamma_shrinks_as_drafting_gets_expensive():
    assert optimal_gamma(0.8, 0.5) <= optimal_gamma(0.8, 0.02)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        expected_speedup(1.2, 4, 0.1)
    with pytest.raises(ValueError):
        expected_speedup(0.8, 4, -0.1)
    with pytest.raises(ValueError):
        optimal_gamma(0.8, 0.1, gamma_max=0)


# -- measure_c -------------------------------------------------------------------------


def test_measure_c_is_a_ratio_of_timings():
    assert measure_c(1.2, 24.0) == pytest.approx(0.05)


def test_measure_c_rejects_a_draft_slower_than_the_target():
    with pytest.raises(ValueError, match="not faster"):
        measure_c(30.0, 24.0)


def test_measure_c_rejects_non_positive_times():
    with pytest.raises(ValueError):
        measure_c(0.0, 24.0)


def test_measure_c_takes_no_model_argument():
    """Guards the rule that c is measured, never derived from parameter counts."""
    import inspect

    params = set(inspect.signature(measure_c).parameters)
    assert params == {"draft_step_ms", "target_step_ms"}
    assert "param" not in inspect.getsource(measure_c).split('"""')[0]


# -- roofline --------------------------------------------------------------------------


def test_ridge_point_units():
    # 1000 TFLOP/s over 2000 GB/s = 1e15 / 2e12 = 500 FLOP/byte
    assert ridge_point(1000.0, 2000.0) == pytest.approx(500.0)


def test_ridge_point_rejects_non_positive():
    with pytest.raises(ValueError):
        ridge_point(0.0, 2000.0)


def test_roofline_is_memory_bound_below_the_ridge():
    tflops, bw = 900.0, 3000.0
    ridge = ridge_point(tflops, bw)
    below = roofline_bound(ridge * 0.5, tflops, bw)
    above = roofline_bound(ridge * 2.0, tflops, bw)
    assert below < tflops
    assert above == pytest.approx(tflops)


# -- alpha from histogram --------------------------------------------------------------


def test_alpha_from_histogram_recovers_a_designed_alpha():
    rng = np.random.default_rng(0)
    gamma, alpha = 5, 0.7
    hist = [0] * (gamma + 1)
    for _ in range(200_000):
        k = 0
        while k < gamma and rng.random() < alpha:
            k += 1
        hist[k] += 1
    assert alpha_from_histogram(hist, gamma) == pytest.approx(alpha, abs=0.01)


def test_alpha_from_histogram_saturates_at_one():
    gamma = 4
    hist = [0, 0, 0, 0, 100]      # every step accepted all four
    assert alpha_from_histogram(hist, gamma) == 1.0


def test_alpha_from_histogram_rejects_wrong_length():
    with pytest.raises(ValueError, match="histogram length"):
        alpha_from_histogram([1, 2, 3], 4)


# -- vectorized form -------------------------------------------------------------------


def test_vectorized_matches_scalar():
    alphas = np.array([0.2, 0.5, 0.8, 1.0])
    gammas = np.array([1, 3, 4, 6])
    cs = np.array([0.05, 0.1, 0.2, 0.05])
    vec = np_expected_speedup(alphas, gammas, cs)
    scalar = [expected_speedup(float(a), int(g), float(c))
              for a, g, c in zip(alphas, gammas, cs)]
    assert vec == pytest.approx(scalar, rel=1e-12)
