"""Acceptance tests for chapter 2: the Welch t-test (abkit.ttest).

Written for you: make them pass. SciPy is the oracle here, the way statsmodels was in chapter 1.
"""

import math

import numpy as np
import pytest
from scipy import stats

from abkit.harness import run_aa
from abkit.ttest import welch_t_test
from abkit.ztest import ComparisonResult, z_test_proportions

ALPHA = 0.05


def sample_with(mean, sd, n):
    """n numbers whose sample mean and sample standard deviation (ddof=1) are exactly mean and sd."""
    v = np.linspace(-1.0, 1.0, n)
    v = (v - v.mean()) / v.std(ddof=1)
    return mean + sd * v


def scipy_welch(control, treatment):
    # SciPy tests mean(first) − mean(second), so treatment goes first to match abkit's sign.
    return stats.ttest_ind(treatment, control, equal_var=False)


# --- the formulas, by hand ------------------------------------------------------

def test_returns_a_comparison_result_with_finite_df():
    result = welch_t_test(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0]))

    assert isinstance(result, ComparisonResult)
    assert math.isfinite(result.df)


def test_estimate_standard_error_and_df_by_hand():
    control, treatment = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0])

    result = welch_t_test(control, treatment)

    v_c, v_t = control.var(ddof=1) / 4, treatment.var(ddof=1) / 3   # 1.6667/4 and 4/3
    assert result.estimate == pytest.approx(4.0 - 2.5)
    assert result.standard_error == pytest.approx(math.sqrt(v_c + v_t))
    assert result.statistic == pytest.approx(1.5 / math.sqrt(v_c + v_t))
    assert result.df == pytest.approx((v_c + v_t) ** 2 / (v_c ** 2 / 3 + v_t ** 2 / 2))


def test_a_z_test_is_a_t_test_with_infinite_df():
    # The normal distribution is the t distribution with infinitely many degrees of freedom (2.6),
    # so reading the z-test's statistic off a t distribution with its own df changes nothing.
    result = z_test_proportions(np.r_[np.ones(10), np.zeros(90)], np.r_[np.ones(15), np.zeros(85)])

    assert result.df == math.inf
    assert 2 * stats.t.sf(abs(result.statistic), result.df) == pytest.approx(result.p_value, rel=1e-9)
    t_star = stats.t.ppf(1 - result.alpha / 2, result.df)
    assert result.ci_high - result.estimate == pytest.approx(t_star * result.standard_error, rel=1e-9)


# --- worked examples from OpenIntro Statistics ---------------------------------------

def test_matches_openintro_stem_cell_example():
    # Examples 7.21-7.22, pp. 268-269: ESC sheep (n 9, mean 3.50, sd 5.17) vs control (n 9, mean -4.33, sd 2.76).
    # The book uses the conservative df = min(n1, n2) - 1 = 8, so its interval (3.32, 12.34) is wider
    # than Welch's. Yours must sit inside it.
    result = welch_t_test(sample_with(-4.33, 2.76, 9), sample_with(3.50, 5.17, 9))

    assert result.estimate == pytest.approx(7.83, abs=0.005)
    assert result.standard_error == pytest.approx(1.95, abs=0.005)
    assert result.df > 8
    assert 3.32 < result.ci_low and result.ci_high < 12.34


def test_matches_openintro_birth_weight_example():
    # Example 7.25, p. 271: non-smokers (n 100, mean 7.18, sd 1.60) vs smokers (n 50, mean 6.78, sd 1.43).
    # The book rounds SE to 0.26 before dividing (T = 1.54) and uses df = 49 (p = 0.135).
    result = welch_t_test(sample_with(6.78, 1.43, 50), sample_with(7.18, 1.60, 100))

    assert result.standard_error == pytest.approx(0.26, abs=0.005)
    assert result.statistic == pytest.approx(1.54, abs=0.02)
    assert result.p_value < 0.135   # more df, thinner tails, smaller p for the same T


# --- SciPy as an oracle ------------------------------------------------------------

def _datasets():
    rng = np.random.default_rng(20261023)
    revenue_like = lambda n: (rng.random(n) < 0.07) * rng.lognormal(3.4, 1.0, n)
    return {
        "small-normal": (rng.normal(10, 2, 8), rng.normal(11, 4, 12)),
        "skewed": (rng.exponential(3.0, 200), rng.exponential(3.5, 50)),
        "revenue-like-90-10": (revenue_like(900), revenue_like(100)),
        "large-equal": (rng.normal(0, 1, 5000), rng.normal(0.05, 1, 5000)),
    }


DATASETS = _datasets()


@pytest.mark.parametrize("name", list(DATASETS))
def test_matches_scipy(name):
    control, treatment = DATASETS[name]

    result = welch_t_test(control, treatment)

    oracle = scipy_welch(control, treatment)
    low, high = oracle.confidence_interval(confidence_level=1 - ALPHA)
    assert result.statistic == pytest.approx(oracle.statistic, rel=1e-9)
    assert result.p_value == pytest.approx(oracle.pvalue, rel=1e-9)
    assert result.df == pytest.approx(oracle.df, rel=1e-9)
    assert (result.ci_low, result.ci_high) == pytest.approx((low, high), rel=1e-9)


def test_tiny_p_values_are_not_rounded_to_zero():
    rng = np.random.default_rng(4)
    control, treatment = rng.normal(0, 1, 2000), rng.normal(0.5, 1, 2000)   # t is about 16

    result = welch_t_test(control, treatment)

    assert result.p_value > 0
    assert result.p_value == pytest.approx(scipy_welch(control, treatment).pvalue, rel=1e-6, abs=0)


# --- properties --------------------------------------------------------------------

def test_df_lies_between_the_conservative_and_the_pooled_choice():
    # Welch's df is never below min(n_c, n_t) - 1 (OpenIntro's shortcut) or above n_c + n_t - 2.
    rng = np.random.default_rng(6)
    for _ in range(300):
        n_c, n_t = rng.integers(2, 60, size=2)
        control = rng.normal(0, rng.uniform(0.1, 5), n_c)
        treatment = rng.normal(0, rng.uniform(0.1, 5), n_t)

        df = welch_t_test(control, treatment).df

        assert min(n_c, n_t) - 1 - 1e-9 <= df <= n_c + n_t - 2 + 1e-9


def test_swapping_the_groups_flips_the_sign_but_not_the_p_value():
    control, treatment = DATASETS["skewed"]

    forward, backward = welch_t_test(control, treatment), welch_t_test(treatment, control)

    assert backward.estimate == pytest.approx(-forward.estimate)
    assert backward.statistic == pytest.approx(-forward.statistic)
    assert backward.p_value == pytest.approx(forward.p_value)
    assert backward.df == pytest.approx(forward.df)


def test_interval_excludes_zero_exactly_when_p_is_below_alpha():
    rng = np.random.default_rng(7)
    for _ in range(300):
        n_c, n_t = rng.integers(5, 300, size=2)
        control = rng.exponential(3.0, n_c)
        treatment = rng.exponential(3.0 * rng.uniform(0.8, 1.3), n_t)

        result = welch_t_test(control, treatment)

        assert (result.ci_low > 0 or result.ci_high < 0) == (result.p_value < ALPHA)


def test_large_samples_agree_with_the_z_test():
    # With thousands of users per group the t and normal distributions are practically the same.
    # Not to 9 digits, though: Welch divides by n − 1 where the z-test divides by n, and far out
    # in the tail that small change moves p by about 0.1%.
    rng = np.random.default_rng(9)
    control = (rng.random(20_000) < 0.069).astype(float)
    treatment = (rng.random(20_000) < 0.074).astype(float)

    t, z = welch_t_test(control, treatment), z_test_proportions(control, treatment)

    assert t.estimate == pytest.approx(z.estimate)
    assert t.p_value == pytest.approx(z.p_value, rel=1e-2)
    assert t.standard_error == pytest.approx(z.standard_error, rel=1e-4)


@pytest.mark.parametrize("control, treatment", [
    pytest.param(np.array([1.0]), np.array([1.0, 2.0, 3.0]), id="one-control-user"),
    pytest.param(np.array([1.0, 2.0]), np.array([5.0]), id="one-treated-user"),
])
def test_rejects_groups_too_small_for_a_variance(control, treatment):
    with pytest.raises(ValueError):
        welch_t_test(control, treatment)


def test_rejects_groups_with_no_variation():
    with pytest.raises(ValueError):
        welch_t_test(np.full(10, 3.0), np.full(12, 3.0))


# --- why t and not z: small samples ----------------------------------------------------

def test_welch_holds_alpha_where_the_normal_cutoff_does_not():
    # Five users per group from a normal distribution: the statistic is the same either way,
    # but comparing it with ±1.96 (the normal cutoff) raises too many false alarms.
    rng = np.random.default_rng(9)
    welch_alarms = normal_alarms = 0
    for _ in range(4000):
        result = welch_t_test(rng.normal(0, 1, 5), rng.normal(0, 1, 5))
        welch_alarms += result.p_value < ALPHA
        normal_alarms += math.erfc(abs(result.statistic) / math.sqrt(2)) < ALPHA

    assert welch_alarms / 4000 < 0.06
    assert normal_alarms / 4000 > 0.07


# --- heavy tails: one huge value ---------------------------------------------------------

def test_one_huge_value_piles_p_values_up_near_0_32():
    # Kohavi p. 207: the outlier lands in one group, Δ ≈ o/n and its SE ≈ o/n, so t ≈ ±1 and p ≈ 0.32.
    values = np.r_[np.random.default_rng(3).normal(10, 2, 999), 1e6]

    def test(control, treatment):
        return welch_t_test(control, treatment).p_value

    result = run_aa(values, test, n_simulations=500, seed=2)

    near_032 = np.mean((result.p_values > 0.30) & (result.p_values < 0.34))
    assert near_032 > 0.9
    assert result.rejection_rate < 0.01
