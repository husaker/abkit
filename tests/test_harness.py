"""Acceptance tests for chapter 0: the simulation harness (abkit.harness).

Written for you: make them pass. They use synthetic data, so the whole file runs in seconds.
SciPy's Welch t-test stands in as the test method until you build your own in chapter 1.
"""

import dataclasses

import numpy as np
import pytest
from scipy import stats

from abkit.harness import (
    SimulationResult,
    add_conversions,
    multiply,
    run_aa,
    run_ab,
    split_users,
)

ALPHA = 0.05


def welch(control, treatment):
    return stats.ttest_ind(control, treatment, equal_var=False).pvalue


def monte_carlo_band(rate, n_simulations, width=3):
    """The range a correct harness lands in almost always: rate ± 3 Monte Carlo standard errors."""
    se = np.sqrt(rate * (1 - rate) / n_simulations)
    return rate - width * se, rate + width * se


@pytest.fixture
def revenue_like():
    """400 users with a continuous metric (mean 10, sd 2)."""
    return np.random.default_rng(20261017).normal(loc=10.0, scale=2.0, size=400)


@pytest.fixture
def conversions():
    """4,000 users who converted (1) or not (0), at a 10% rate."""
    return (np.random.default_rng(20261018).random(4000) < 0.10).astype(float)


# --- split_users -------------------------------------------------------------

@pytest.mark.parametrize("n_users, n_treated", [(10, 5), (11, 5), (2, 1)])
def test_split_users_treats_exactly_half(n_users, n_treated):
    treated = split_users(n_users, np.random.default_rng(1))

    assert treated.dtype == bool
    assert len(treated) == n_users
    assert treated.sum() == n_treated


def test_split_users_is_reproducible_from_a_seed():
    first = split_users(1000, np.random.default_rng(7))
    again = split_users(1000, np.random.default_rng(7))
    other = split_users(1000, np.random.default_rng(8))

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


# --- SimulationResult --------------------------------------------------------

def test_simulation_result_summarises_p_values():
    result = SimulationResult(p_values=np.array([0.01, 0.20, 0.04, 0.50]), alpha=ALPHA)

    assert result.n_simulations == 4
    assert result.rejection_rate == pytest.approx(0.5)
    assert result.mc_standard_error == pytest.approx(np.sqrt(0.5 * 0.5 / 4))


def test_simulation_result_cannot_be_changed():
    result = SimulationResult(p_values=np.array([0.3]), alpha=ALPHA)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.alpha = 0.10


# --- A/A tests -----------------------------------------------------------------

def test_run_aa_is_reproducible_from_a_seed(revenue_like):
    first = run_aa(revenue_like, welch, n_simulations=50, seed=3)
    again = run_aa(revenue_like, welch, n_simulations=50, seed=3)
    other = run_aa(revenue_like, welch, n_simulations=50, seed=4)

    np.testing.assert_array_equal(first.p_values, again.p_values)
    assert not np.array_equal(first.p_values, other.p_values)


def test_aa_false_positive_rate_is_close_to_alpha(revenue_like):
    result = run_aa(revenue_like, welch, n_simulations=2000, alpha=ALPHA, seed=11)

    low, high = monte_carlo_band(ALPHA, 2000)
    assert low <= result.rejection_rate <= high


def test_aa_p_values_are_uniform(revenue_like):
    # Kohavi recommends a goodness-of-fit test such as Kolmogorov–Smirnov for this (p. 207).
    result = run_aa(revenue_like, welch, n_simulations=2000, seed=12)

    assert stats.kstest(result.p_values, "uniform").pvalue > 0.001


# --- effect injection ----------------------------------------------------------

def test_multiply_applies_a_relative_lift():
    lifted = multiply(np.array([10.0, 20.0]), 0.10, np.random.default_rng(0))

    assert lifted == pytest.approx([11.0, 22.0])


def test_add_conversions_raises_the_rate_by_a_relative_lift():
    rng = np.random.default_rng(5)
    before = (rng.random(200_000) < 0.10).astype(float)

    after = add_conversions(before, 0.20, rng)

    assert set(np.unique(after)) <= {0.0, 1.0}  # still a 0/1 metric
    assert np.all(after >= before)  # converters are never undone
    # The noise in this rate is about 0.0003, so 0.0012 is about 4 standard errors.
    assert after.mean() == pytest.approx(before.mean() * 1.20, abs=0.0012)


def test_add_conversions_rejects_an_impossible_lift():
    sixty_percent = np.array([1.0] * 6 + [0.0] * 4)

    with pytest.raises(ValueError):
        add_conversions(sixty_percent, 1.0, np.random.default_rng(0))  # 60% → 120%


# --- A/B tests -----------------------------------------------------------------

def test_ab_with_no_lift_behaves_like_aa(conversions):
    result = run_ab(conversions, welch, inject=add_conversions, lift=0.0, n_simulations=1000, seed=21)

    low, high = monte_carlo_band(ALPHA, 1000)
    assert low <= result.rejection_rate <= high


def test_ab_detects_a_large_lift(revenue_like):
    result = run_ab(revenue_like, welch, inject=multiply, lift=0.10, n_simulations=200, seed=22)

    assert result.rejection_rate > 0.95


def test_bigger_lifts_are_detected_more_often(conversions):
    small = run_ab(conversions, welch, inject=add_conversions, lift=0.10, n_simulations=300, seed=23)
    large = run_ab(conversions, welch, inject=add_conversions, lift=0.30, n_simulations=300, seed=23)

    assert large.rejection_rate > small.rejection_rate + 0.3
