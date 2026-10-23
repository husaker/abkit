"""Acceptance tests for chapter 1: the z-test for two conversion rates (abkit.ztest).

Written for you: make them pass. Three kinds of check, each with its own tolerance:
- formulas and statsmodels: the same maths in different code, so they agree to ~9 digits;
- textbook examples: printed with 2-3 digits, so they agree to the last printed digit;
- the harness: random, so it agrees within a few Monte Carlo standard errors.
"""

import math

import numpy as np
import pytest
# Import the module, not the functions: pytest would collect a function named
# test_proportions_2indep as one of your tests.
import statsmodels.stats.proportion as smp

from abkit.harness import add_conversions, run_aa, run_ab
from abkit.ztest import ComparisonResult, z_test_proportions

ALPHA = 0.05


def users(converted, n):
    """n users, the first `converted` of them converted (1), the rest not (0)."""
    return np.r_[np.ones(converted), np.zeros(n - converted)]


def z_p_value(control, treatment):
    return z_test_proportions(control, treatment).p_value


@pytest.fixture
def conversions():
    """4,000 users who converted (1) or not (0), at a 10% rate."""
    return (np.random.default_rng(20261018).random(4000) < 0.10).astype(float)


# --- the formulas, by hand ------------------------------------------------------

def test_estimate_is_treatment_minus_control():
    result = z_test_proportions(users(10, 100), users(15, 100))

    assert isinstance(result, ComparisonResult)
    assert result.estimate == pytest.approx(0.05)


def test_unpooled_standard_error_statistic_and_p_value():
    result = z_test_proportions(users(10, 100), users(15, 100))

    se = math.sqrt(0.15 * 0.85 / 100 + 0.10 * 0.90 / 100)
    assert result.standard_error == pytest.approx(se)
    assert result.statistic == pytest.approx(0.05 / se)
    assert result.p_value == pytest.approx(0.2837, abs=1e-4)  # two tails beyond |z| = 1.072


def test_pooled_changes_the_statistic_but_not_the_interval():
    unpooled = z_test_proportions(users(10, 100), users(15, 100))
    pooled = z_test_proportions(users(10, 100), users(15, 100), pooled=True)

    pooled_se = math.sqrt(0.125 * 0.875 * (1 / 100 + 1 / 100))  # 25 of 200 converted overall
    assert pooled.statistic == pytest.approx(0.05 / pooled_se)
    assert pooled.standard_error == pytest.approx(unpooled.standard_error)
    assert (pooled.ci_low, pooled.ci_high) == pytest.approx((unpooled.ci_low, unpooled.ci_high))


def test_interval_is_estimate_plus_minus_z_star_standard_errors():
    result = z_test_proportions(users(10, 100), users(15, 100), alpha=0.01)

    z_star = 2.5758  # 99% of N(0, 1) lies within ±2.5758
    assert result.alpha == 0.01
    assert result.ci_low == pytest.approx(result.estimate - z_star * result.standard_error, rel=1e-4)
    assert result.ci_high == pytest.approx(result.estimate + z_star * result.standard_error, rel=1e-4)


# --- worked examples from OpenIntro Statistics (rounded in the book) --------------

def test_matches_openintro_cpr_interval():
    # Example 6.12, p. 218: survival 14/40 with a blood thinner vs 11/50 without, 90% interval.
    result = z_test_proportions(users(11, 50), users(14, 40), alpha=0.10)

    assert result.estimate == pytest.approx(0.13, abs=0.005)
    assert result.standard_error == pytest.approx(0.095, abs=0.001)
    assert (result.ci_low, result.ci_high) == pytest.approx((-0.026, 0.286), abs=0.002)


def test_matches_openintro_mammogram_test():
    # Examples 6.16-6.18, pp. 220-221: deaths 500/44,925 with mammograms vs 505/44,910 without.
    # The book rounds the difference to -0.00012 before dividing, so its z (-0.17) and
    # p (0.8650) are a little off the exact values; the tolerance allows for that.
    result = z_test_proportions(users(505, 44_910), users(500, 44_925), pooled=True)

    assert result.statistic == pytest.approx(-0.17, abs=0.01)
    assert result.p_value == pytest.approx(0.8650, abs=0.01)


# --- statsmodels as an oracle -----------------------------------------------------

DATASETS = [
    pytest.param(10, 100, 15, 100, id="small"),
    pytest.param(11, 50, 14, 40, id="unequal-groups"),
    pytest.param(3, 1000, 9, 1000, id="rare-event"),
    pytest.param(690, 10_000, 760, 10_000, id="january-like"),
    pytest.param(90, 900, 30, 100, id="90-10-split"),
    pytest.param(505, 44_910, 500, 44_925, id="mammogram"),
]


@pytest.mark.parametrize("k_c, n_c, k_t, n_t", DATASETS)
def test_pooled_matches_statsmodels(k_c, n_c, k_t, n_t):
    result = z_test_proportions(users(k_c, n_c), users(k_t, n_t), pooled=True)

    z, p = smp.proportions_ztest([k_t, k_c], [n_t, n_c])  # treatment first: z has the sign of t − c
    assert result.statistic == pytest.approx(z, rel=1e-9)
    assert result.p_value == pytest.approx(p, rel=1e-9)


@pytest.mark.parametrize("k_c, n_c, k_t, n_t", DATASETS)
def test_unpooled_matches_statsmodels(k_c, n_c, k_t, n_t):
    result = z_test_proportions(users(k_c, n_c), users(k_t, n_t))

    oracle = smp.test_proportions_2indep(k_t, n_t, k_c, n_c, method="wald")
    low, high = smp.confint_proportions_2indep(k_t, n_t, k_c, n_c, method="wald")
    assert result.statistic == pytest.approx(oracle.statistic, rel=1e-9)
    assert result.p_value == pytest.approx(oracle.pvalue, rel=1e-9)
    assert (result.ci_low, result.ci_high) == pytest.approx((low, high), rel=1e-9)


def test_tiny_p_values_are_not_rounded_to_zero():
    # z is about 16.5 here. 1 − Φ(16.5) is far below the spacing of floats near 1, so computing
    # it as a subtraction gives exactly 0. abs=0 switches off approx's default absolute
    # tolerance (1e-12), which would otherwise accept 0 as "close enough" to 3.7e-61.
    result = z_test_proportions(users(100, 1000), users(400, 1000))

    oracle = smp.test_proportions_2indep(400, 1000, 100, 1000, method="wald")
    assert result.p_value > 0
    assert result.p_value == pytest.approx(oracle.pvalue, rel=1e-6, abs=0)


# --- properties every correct z-test has -------------------------------------------

def test_swapping_the_groups_flips_the_sign_but_not_the_p_value():
    forward = z_test_proportions(users(10, 100), users(15, 120))
    backward = z_test_proportions(users(15, 120), users(10, 100))

    assert backward.estimate == pytest.approx(-forward.estimate)
    assert backward.statistic == pytest.approx(-forward.statistic)
    assert backward.p_value == pytest.approx(forward.p_value)
    assert (backward.ci_low, backward.ci_high) == pytest.approx((-forward.ci_high, -forward.ci_low))


def test_interval_excludes_zero_exactly_when_p_is_below_alpha():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n_c, n_t = rng.integers(50, 2000, size=2)
        rate = rng.uniform(0.05, 0.5)
        control = (rng.random(n_c) < rate).astype(float)
        treatment = (rng.random(n_t) < rate * rng.uniform(0.8, 1.3)).astype(float)

        result = z_test_proportions(control, treatment)

        excludes_zero = result.ci_low > 0 or result.ci_high < 0
        assert excludes_zero == (result.p_value < ALPHA)


@pytest.mark.parametrize("bad", [np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 1.0])], ids=["two", "half"])
def test_rejects_values_that_are_not_zero_or_one(bad):
    with pytest.raises(ValueError):
        z_test_proportions(users(10, 100), bad)


def test_rejects_an_empty_group():
    with pytest.raises(ValueError):
        z_test_proportions(users(10, 100), np.array([]))


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("value", [0, 1], ids=["nobody-converted", "everybody-converted"])
def test_rejects_groups_with_no_variation(value, pooled):
    # Both groups identical and constant: the standard error is 0 and z would be 0 / 0.
    same = np.full(100, float(value))

    with pytest.raises(ValueError):
        z_test_proportions(same, same.copy(), pooled=pooled)


# --- the harness ---------------------------------------------------------------------

@pytest.mark.parametrize("pooled", [False, True])
def test_aa_false_positive_rate_is_close_to_alpha(conversions, pooled):
    def test(control, treatment):
        return z_test_proportions(control, treatment, pooled=pooled).p_value

    result = run_aa(conversions, test, n_simulations=2000, alpha=ALPHA, seed=31)

    se = math.sqrt(ALPHA * (1 - ALPHA) / 2000)
    assert ALPHA - 3 * se <= result.rejection_rate <= ALPHA + 3 * se


def test_simulated_power_matches_the_formula(conversions):
    # The formula is the oracle here: z under the alternative is roughly N(δ / SE, 1), so
    # power ≈ Φ(δ / SE − z*) + Φ(−δ / SE − z*). A +20% lift puts power near 50%, where the
    # harness and the formula agree closely (the page explains why they drift apart near 85%).
    lift, n_simulations = 0.20, 2000
    result = run_ab(conversions, z_p_value, inject=add_conversions, lift=lift, n_simulations=n_simulations, seed=32)

    r = conversions.mean()
    n = len(conversions) // 2
    se = math.sqrt(r * (1 - r) / n + r * (1 + lift) * (1 - r * (1 + lift)) / n)
    shift, z_star = r * lift / se, 1.959964
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
    formula = phi(shift - z_star) + phi(-shift - z_star)
    mc_se = math.sqrt(formula * (1 - formula) / n_simulations)
    assert abs(result.rejection_rate - formula) <= 4 * mc_se
