"""Chapter 5: abkit.delta, the delta method for ratio metrics and relative lifts, and linearisation.

Synthetic data only, so these run in a few seconds. The real-data checks are in
test_delta_data.py.
"""

import dataclasses
import inspect
import math
from statistics import NormalDist

import numpy as np
import pytest

from abkit.bootstrap import bootstrap, bootstrap_ratio
from abkit.delta import RatioMetric, delta_ratio_test, relative_lift_test
from abkit.ztest import ComparisonResult

# The chapter's five-user shops. Each user has some sessions (the denominator) and some sessions
# with a purchase (the numerator). Shops A and B have the same totals, 4 purchase sessions in 20,
# but in B it is the heavy user who never buys.
SESSIONS = [1, 2, 3, 4, 10]
SHOP_A = ([0, 1, 1, 1, 1], SESSIONS)
SHOP_B = ([1, 1, 1, 1, 0], SESSIONS)
SHOP_C = ([1, 1, 1, 1, 1], SESSIONS)

# The chapter's five-user revenue groups from part 0 of chapter 4.
CONTROL = np.array([2.0, 4.0, 4.0, 6.0, 9.0])
TREATMENT = np.array([3.0, 5.0, 7.0, 8.0, 12.0])


def mc_tolerance_for_an_sd(n_resamples):
    """Four Monte Carlo standard errors of an SD estimated from n_resamples draws, relative."""
    return 4 / math.sqrt(2 * (n_resamples - 1))


def clustered_group(n_users, rng, rate=0.2, icc=0.3, lift=1.0):
    """Users with a geometric number of sessions and their own purchase rate per session.

    The rates are Beta-distributed around `rate` with intra-user correlation `icc`, so sessions of
    one user are correlated, as in Deng, Knoblich & Lu (2018) section 3. Returns the number of
    sessions with a purchase and the number of sessions, one of each per user.
    """
    sessions = rng.geometric(1 / 2.36, n_users)  # mean 2.36 sessions per user, like January
    a, b = rate * (1 - icc) / icc, (1 - rate) * (1 - icc) / icc
    rates = np.clip(rng.beta(a, b, n_users) * lift, 0, 1)
    return rng.binomial(sessions, rates).astype(float), sessions.astype(float)


def session_level_z(control, treatment):
    """The naive test: every session an independent 0/1 observation, as in chapter 1."""
    variance = 0.0
    for metric in (control, treatment):
        rate = metric.ratio
        variance += rate * (1 - rate) / metric.denominator.sum()
    z = (treatment.ratio - control.ratio) / math.sqrt(variance)
    return 2 * (1 - NormalDist().cdf(abs(z)))


# --- RatioMetric: the ratio ----------------------------------------------------------------

def test_ratio_is_the_sum_over_the_sum_not_the_mean_of_per_user_ratios():
    metric = RatioMetric(*SHOP_A)

    assert metric.ratio == pytest.approx(4 / 20)
    assert metric.ratio != pytest.approx(np.mean([0 / 1, 1 / 2, 1 / 3, 1 / 4, 1 / 10]))


def test_n_users_counts_the_users():
    assert RatioMetric(*SHOP_A).n_users == 5


def test_inputs_can_be_lists_and_are_stored_as_float_arrays():
    metric = RatioMetric([1, 2], [3, 4])

    assert metric.numerator.dtype == float and metric.denominator.dtype == float
    assert metric.numerator.shape == (2,)


# --- RatioMetric: the delta-method standard error --------------------------------------------

def test_standard_error_matches_the_hand_computation_for_shop_a():
    # Section 2.3: x̄ = 0.8, ȳ = 4, s_x² = 0.2, s_y² = 12.5, s_xy = 0.75, so var = 1/200.
    metric = RatioMetric(*SHOP_A)

    assert metric.standard_error == pytest.approx(math.sqrt(1 / 200), rel=1e-9)


def test_same_totals_but_a_different_heavy_user_give_a_different_standard_error():
    # The naive formula sqrt(r(1 - r) / sessions) cannot tell A and B apart; the delta method can,
    # because in B the heavy user's sessions pull the ratio down (s_xy = -1.5), so var = 13/800.
    a, b = RatioMetric(*SHOP_A), RatioMetric(*SHOP_B)

    assert b.ratio == a.ratio
    assert b.standard_error == pytest.approx(math.sqrt(13 / 800), rel=1e-9)
    assert b.standard_error > a.standard_error


def test_standard_error_uses_sample_covariances_with_ddof_1():
    # Shop C: every user buys once, so s_x² = s_xy = 0 and var = x̄² s_y² / (n ȳ⁴) = 5/512 with ddof=1.
    metric = RatioMetric(*SHOP_C)

    assert metric.standard_error == pytest.approx(math.sqrt(5 / 512), rel=1e-9)


def test_standard_error_reduces_to_s_over_root_n_when_every_user_has_one_session():
    # With all denominators equal to 1 the ratio is a plain mean, and the delta method must give
    # chapter 1's standard error.
    values = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    metric = RatioMetric(values, np.ones_like(values))

    assert metric.standard_error == pytest.approx(values.std(ddof=1) / math.sqrt(values.size), rel=1e-12)


def test_standard_error_is_zero_when_every_user_has_exactly_the_same_rate():
    # If x_i = 0.3 y_i for every user, every resample gives the ratio 0.3 and there is no uncertainty.
    # The covariance term is what makes the three terms cancel; drop it and the SE comes out positive.
    sessions = np.array([1.0, 2.0, 3.0, 4.0, 10.0])

    metric = RatioMetric(0.3 * sessions, sessions)

    assert metric.standard_error == pytest.approx(0.0, abs=1e-12)


def test_standard_error_agrees_with_the_user_level_bootstrap():
    rng = np.random.default_rng(4)
    purchases, sessions = clustered_group(3000, rng)
    n_resamples = 2000

    metric = RatioMetric(purchases, sessions)

    positions = rng.integers(0, metric.n_users, size=(n_resamples, metric.n_users))
    resampled = purchases[positions].sum(axis=1) / sessions[positions].sum(axis=1)
    assert metric.standard_error == pytest.approx(resampled.std(ddof=1), rel=mc_tolerance_for_an_sd(n_resamples))


# --- RatioMetric: linearisation --------------------------------------------------------------

def test_linearize_subtracts_k_times_the_denominator():
    metric = RatioMetric(*SHOP_A)

    linearized = metric.linearize(0.5)

    assert np.allclose(linearized, metric.numerator - 0.5 * metric.denominator)


def test_linearized_values_average_to_zero_at_the_metrics_own_ratio():
    # Section 3.1: L_i = x_i - R y_i with R = Σx/Σy, so ΣL_i = Σx - R Σy = 0 exactly.
    metric = RatioMetric(*SHOP_A)

    linearized = metric.linearize()

    assert np.allclose(linearized, [-0.2, 0.6, 0.4, 0.2, -1.0])
    assert linearized.mean() == pytest.approx(0.0, abs=1e-12)


def test_standard_error_of_the_linearized_mean_over_y_bar_is_the_delta_se():
    # Derivation D3: s_L / (sqrt(n) ȳ) equals the delta-method SE, exactly, for the metric's own ratio.
    rng = np.random.default_rng(5)
    metric = RatioMetric(*clustered_group(500, rng))

    linearized = metric.linearize()

    linearized_se = linearized.std(ddof=1) / math.sqrt(metric.n_users) / metric.denominator.mean()
    assert linearized_se == pytest.approx(metric.standard_error, rel=1e-9)


def test_linearize_returns_a_new_array_and_leaves_the_metric_unchanged():
    metric = RatioMetric(*SHOP_A)

    linearized = metric.linearize()
    linearized[0] = 99.0

    assert metric.numerator[0] == 0.0
    assert metric.ratio == pytest.approx(0.2)


# --- RatioMetric: validation and immutability ------------------------------------------------

@pytest.mark.parametrize("numerator, denominator, problem", [
    ([1, 2, 3], [1, 2], "different lengths"),
    ([1], [2], "one user"),
    ([1, 2], [3, -1], "a negative denominator"),
    ([1, 2], [0, 0], "all denominators zero"),
    ([1, float("nan")], [1, 2], "a NaN"),
    ([[1, 2]], [[3, 4]], "two-dimensional input"),
])
def test_impossible_metrics_are_rejected(numerator, denominator, problem):
    with pytest.raises(ValueError):
        RatioMetric(numerator, denominator)


def test_a_user_with_zero_sessions_is_allowed():
    # Someone who appears in the user table without a session in the period contributes 0/0 to the
    # sums; the ratio of sums is still defined.
    metric = RatioMetric([0, 1, 1], [0, 2, 3])

    assert metric.ratio == pytest.approx(2 / 5)


def test_metric_is_frozen_and_its_arrays_are_read_only():
    metric = RatioMetric(*SHOP_A)

    with pytest.raises(dataclasses.FrozenInstanceError):
        metric.numerator = np.zeros(5)
    with pytest.raises(ValueError):
        metric.numerator[0] = 7.0


def test_metric_copies_its_inputs_so_later_edits_do_not_leak_in():
    numerator = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

    metric = RatioMetric(numerator, SESSIONS)
    numerator[0] = 100.0

    assert metric.ratio == pytest.approx(0.2)


# --- delta_ratio_test ------------------------------------------------------------------------

def test_estimate_is_the_difference_of_ratios_treatment_minus_control():
    result = delta_ratio_test(RatioMetric(*SHOP_A), RatioMetric(*SHOP_C))

    assert result.estimate == pytest.approx(0.25 - 0.2, rel=1e-12)


def test_standard_error_adds_the_two_groups_variances():
    # Independent groups: var(R_T - R_C) = var(R_T) + var(R_C), chapter 4 part 0.6.
    result = delta_ratio_test(RatioMetric(*SHOP_A), RatioMetric(*SHOP_C))

    assert result.standard_error == pytest.approx(math.sqrt(1 / 200 + 5 / 512), rel=1e-9)


def test_statistic_p_value_and_interval_follow_the_normal_curve():
    result = delta_ratio_test(RatioMetric(*SHOP_A), RatioMetric(*SHOP_C))

    z = result.estimate / result.standard_error
    assert result.statistic == pytest.approx(z, rel=1e-12)
    assert result.p_value == pytest.approx(2 * (1 - NormalDist().cdf(abs(z))), rel=1e-9)
    assert result.ci_low == pytest.approx(result.estimate - 1.959964 * result.standard_error, rel=1e-6)
    assert result.ci_high == pytest.approx(result.estimate + 1.959964 * result.standard_error, rel=1e-6)
    assert result.df == math.inf
    assert result.alpha == 0.05


def test_alpha_changes_the_interval_but_not_the_p_value():
    control, treatment = RatioMetric(*SHOP_A), RatioMetric(*SHOP_C)

    wide, narrow = delta_ratio_test(control, treatment, alpha=0.01), delta_ratio_test(control, treatment, alpha=0.1)

    assert wide.p_value == narrow.p_value
    assert wide.alpha == 0.01 and narrow.alpha == 0.1
    assert narrow.ci_high == pytest.approx(narrow.estimate + 1.644854 * narrow.standard_error, rel=1e-6)
    assert wide.ci_high - wide.ci_low > narrow.ci_high - narrow.ci_low


def test_swapping_the_groups_flips_the_sign_and_keeps_the_p_value():
    control, treatment = RatioMetric(*SHOP_A), RatioMetric(*SHOP_C)

    forward, backward = delta_ratio_test(control, treatment), delta_ratio_test(treatment, control)

    assert backward.estimate == pytest.approx(-forward.estimate)
    assert backward.statistic == pytest.approx(-forward.statistic)
    assert backward.p_value == pytest.approx(forward.p_value)
    assert backward.standard_error == pytest.approx(forward.standard_error)


def test_identical_groups_give_zero_estimate_and_p_of_one():
    metric = RatioMetric(*SHOP_A)

    result = delta_ratio_test(metric, metric)

    assert result.estimate == 0.0
    assert result.p_value == pytest.approx(1.0)


def test_the_result_is_a_comparison_result():
    assert isinstance(delta_ratio_test(RatioMetric(*SHOP_A), RatioMetric(*SHOP_C)), ComparisonResult)


@pytest.mark.parametrize("alpha", [0, 1, 1.5, -0.1])
def test_alpha_outside_0_1_is_rejected(alpha):
    with pytest.raises(ValueError):
        delta_ratio_test(RatioMetric(*SHOP_A), RatioMetric(*SHOP_C), alpha=alpha)


def test_groups_with_no_variation_are_rejected_rather_than_divided_by_zero():
    sessions = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    control, treatment = RatioMetric(0.3 * sessions, sessions), RatioMetric(0.3 * sessions, sessions)

    with pytest.raises(ValueError):
        delta_ratio_test(control, treatment)


def test_aa_false_alarms_hold_5_percent_where_the_session_level_z_test_does_not():
    # Section 2.5: with correlated sessions, treating each session as an independent observation
    # understates the variance (Kohavi p. 195; Deng et al. 2018 Table 2). The delta method does not.
    rng = np.random.default_rng(7)
    n_simulations, users_per_group = 1000, 2000
    rejected_delta = rejected_naive = 0
    for _ in range(n_simulations):
        control = RatioMetric(*clustered_group(users_per_group, rng))
        treatment = RatioMetric(*clustered_group(users_per_group, rng))
        rejected_delta += delta_ratio_test(control, treatment).p_value < 0.05
        rejected_naive += session_level_z(control, treatment) < 0.05

    # 1,000 simulations measure a 5% rate to about ±0.7 points.
    assert 0.03 <= rejected_delta / n_simulations <= 0.08
    assert rejected_naive / n_simulations > 0.10


def test_a_real_lift_is_detected():
    rng = np.random.default_rng(8)
    n_simulations, users_per_group = 300, 2000
    rejected = 0
    for _ in range(n_simulations):
        control = RatioMetric(*clustered_group(users_per_group, rng))
        treatment = RatioMetric(*clustered_group(users_per_group, rng, lift=1.25))
        rejected += delta_ratio_test(control, treatment).p_value < 0.05

    assert rejected / n_simulations > 0.9


def test_agrees_with_the_user_level_bootstrap_of_chapter_4():
    # The formula and the resampling estimate the same quantity (Kohavi p. 195).
    rng = np.random.default_rng(9)
    num_c, den_c = clustered_group(400, rng)
    num_t, den_t = clustered_group(500, rng, lift=1.1)
    n_resamples = 2000

    formula = delta_ratio_test(RatioMetric(num_c, den_c), RatioMetric(num_t, den_t))
    resampled = bootstrap_ratio(num_c, den_c, num_t, den_t, n_resamples=n_resamples, seed=1)

    assert formula.estimate == pytest.approx(resampled.estimate, rel=1e-12)
    assert formula.standard_error == pytest.approx(resampled.standard_error, rel=mc_tolerance_for_an_sd(n_resamples))


# --- relative_lift_test ----------------------------------------------------------------------

def test_relative_estimate_is_the_ratio_of_means_minus_one():
    result = relative_lift_test(CONTROL, TREATMENT)

    assert result.estimate == pytest.approx(7 / 5 - 1, rel=1e-12)


def test_relative_standard_error_follows_kohavi_eq_18_6():
    # x̄_C = 5, x̄_T = 7, v_C = 1.4, v_T = 2.3: var = 2.3/25 + 49 × 1.4/625 = 0.20176.
    result = relative_lift_test(CONTROL, TREATMENT)

    assert result.standard_error == pytest.approx(math.sqrt(2.3 / 25 + 49 * 1.4 / 625), rel=1e-9)
    assert result.standard_error > math.sqrt(2.3 + 1.4) / 5  # the "common mistake", var(Δ)/x̄_C²


def test_with_equal_means_the_relative_se_is_the_absolute_se_over_the_control_mean():
    # Chapter 4, derivation D5: the two formulas agree when x̄_T = x̄_C, whatever the spreads.
    treatment = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # mean 5, like CONTROL, but s² = 10

    result = relative_lift_test(CONTROL, treatment)

    assert result.estimate == pytest.approx(0.0, abs=1e-12)
    assert result.standard_error == pytest.approx(math.sqrt(1.4 + 2.0) / 5, rel=1e-9)


def test_relative_statistic_p_value_interval_and_df():
    result = relative_lift_test(CONTROL, TREATMENT, alpha=0.1)

    z = result.estimate / result.standard_error
    assert result.statistic == pytest.approx(z, rel=1e-12)
    assert result.p_value == pytest.approx(2 * (1 - NormalDist().cdf(abs(z))), rel=1e-9)
    assert result.ci_low == pytest.approx(result.estimate - 1.644854 * result.standard_error, rel=1e-6)
    assert result.df == math.inf
    assert result.alpha == 0.1
    assert isinstance(result, ComparisonResult)


def test_relative_aa_holds_5_percent_and_a_true_lift_is_covered():
    rng = np.random.default_rng(10)
    n_simulations, n_users = 1000, 500
    rejected = covered = 0
    for _ in range(n_simulations):
        control, treatment = rng.lognormal(0, 1, n_users), rng.lognormal(0, 1, n_users)
        rejected += relative_lift_test(control, treatment).p_value < 0.05
        with_lift = relative_lift_test(control, treatment * 1.5)
        covered += with_lift.ci_low <= 0.5 <= with_lift.ci_high

    assert 0.03 <= rejected / n_simulations <= 0.08
    assert 0.92 <= covered / n_simulations <= 0.98


def test_relative_se_agrees_with_the_relative_bootstrap_of_chapter_4():
    rng = np.random.default_rng(11)
    control, treatment = rng.normal(10, 2, 1000), rng.normal(15, 3, 1000)
    n_resamples = 2000

    formula = relative_lift_test(control, treatment)
    resampled = bootstrap(control, treatment, n_resamples=n_resamples, relative=True, seed=1)

    assert formula.estimate == pytest.approx(resampled.estimate, rel=1e-12)
    assert formula.standard_error == pytest.approx(resampled.standard_error, rel=mc_tolerance_for_an_sd(n_resamples))


def test_relative_lift_inputs_can_be_lists():
    assert relative_lift_test([2, 4, 4, 6, 9], [3, 5, 7, 8, 12]).estimate == pytest.approx(0.4)


@pytest.mark.parametrize("control, treatment", [
    ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),  # control mean zero: no relative lift
    ([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]),  # control mean zero with spread: still no relative lift
    ([1.0], [1.0, 2.0]),  # one user
    ([1.0, 2.0], [1.0, float("inf")]),  # not finite
])
def test_impossible_relative_lifts_are_rejected(control, treatment):
    with pytest.raises(ValueError):
        relative_lift_test(control, treatment)


@pytest.mark.parametrize("alpha", [0, 1, 2])
def test_relative_alpha_outside_0_1_is_rejected(alpha):
    with pytest.raises(ValueError):
        relative_lift_test(CONTROL, TREATMENT, alpha=alpha)


# --- the API: options are keyword-only (Effective Python item 25) ----------------------------

@pytest.mark.parametrize("function", [delta_ratio_test, relative_lift_test])
def test_alpha_can_only_be_passed_by_keyword(function):
    parameter = inspect.signature(function).parameters["alpha"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 0.05
