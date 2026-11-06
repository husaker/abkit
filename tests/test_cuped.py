"""Chapter 6: CUPED, variance reduction with a pre-experiment covariate.

The five-user example is the one of section 2.5: y is a user's January revenue and x the same
user's revenue over the three months before. In the balanced version both groups have the same
mean covariate, so the estimate is unchanged and only the standard error shrinks; in the skewed
version the treatment group happened to draw the heavier past spenders, and CUPED takes that out.
"""

import dataclasses
import inspect
import math

import numpy as np
import pytest

from abkit.cuped import UserMetric, cuped_test, theta
from abkit.regression import ols
from abkit.ttest import welch_t_test
from abkit.ztest import ComparisonResult

CONTROL_Y, TREATMENT_Y = [2.0, 4.0, 4.0, 6.0, 9.0], [3.0, 5.0, 7.0, 8.0, 12.0]
CONTROL_X, TREATMENT_X = [4.0, 1.0, 3.0, 4.0, 8.0], [2.0, 1.0, 5.0, 5.0, 7.0]  # balanced: both means are 4
SKEWED_CONTROL_X, SKEWED_TREATMENT_X = [3.0, 5.0, 0.0, 4.0, 5.0], [1.0, 3.0, 5.0, 7.0, 7.0]  # means 3.4 and 4.6


@pytest.fixture
def balanced():
    return UserMetric(CONTROL_Y, CONTROL_X), UserMetric(TREATMENT_Y, TREATMENT_X)


@pytest.fixture
def skewed():
    return UserMetric(CONTROL_Y, SKEWED_CONTROL_X), UserMetric(TREATMENT_Y, SKEWED_TREATMENT_X)


def correlated_pairs(rng, n, rho, mean=10.0, sd=3.0):
    """n users whose metric y and covariate x are normal with correlation rho."""
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    y = mean + sd * z1
    x = mean + sd * (rho * z1 + math.sqrt(1 - rho**2) * z2)
    return y, x


# ---------------------------------------------------------------- theta

def test_theta_is_the_pooled_covariance_over_the_covariate_variance():
    # by hand: sum of (x - 4)(y - 6) over the ten users is 50, sum of (x - 4)^2 is 50
    y, x = CONTROL_Y + TREATMENT_Y, CONTROL_X + TREATMENT_X
    assert theta(np.array(y), np.array(x)) == pytest.approx(1.0)
    assert theta(np.array(CONTROL_Y), np.array(CONTROL_X)) == pytest.approx(20 / 26)
    assert theta(np.array(TREATMENT_Y), np.array(TREATMENT_X)) == pytest.approx(30 / 24)


def test_theta_is_the_least_squares_slope():
    rng = np.random.default_rng(1)
    y, x = correlated_pairs(rng, 300, 0.6)
    design = np.column_stack([np.ones(300), x])
    assert theta(y, x) == pytest.approx(ols(y, design).coef[1], rel=1e-10)


def test_theta_does_not_depend_on_ddof():
    rng = np.random.default_rng(2)
    y, x = correlated_pairs(rng, 50, 0.4)
    with_n_minus_1 = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    with_n = np.cov(y, x, ddof=0)[0, 1] / x.var(ddof=0)
    assert theta(y, x) == pytest.approx(with_n_minus_1) == pytest.approx(with_n)


@pytest.mark.parametrize("values, covariate", [
    ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),          # constant covariate: variance 0
    ([1.0, 2.0, 3.0], [1.0, 2.0]),               # different lengths
    ([1.0], [2.0]),                              # fewer than two users
    ([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]),       # not finite
    ([1.0, 2.0, 3.0], [1.0, 2.0, np.inf]),
    ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),  # not one value per user
])
def test_theta_rejects_bad_input(values, covariate):
    with pytest.raises(ValueError):
        theta(np.array(values), np.array(covariate))


# ---------------------------------------------------------------- UserMetric

def test_user_metric_accepts_lists_and_stores_float_copies():
    metric = UserMetric([1, 2, 3], [0, 1, 0])
    assert metric.values.dtype == float and metric.covariate.dtype == float
    assert metric.n_users == 3


def test_user_metric_keeps_its_own_copy_of_the_arrays():
    values, covariate = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0])
    metric = UserMetric(values, covariate)
    values[0] = 100.0
    assert metric.values[0] == 1.0


def test_user_metric_is_read_only():
    metric = UserMetric([1.0, 2.0, 3.0], [0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        metric.values[0] = 100.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        metric.values = np.zeros(3)  # type: ignore[misc]


@pytest.mark.parametrize("values, covariate", [
    ([1.0, 2.0, 3.0], [1.0, 2.0]),
    ([1.0], [2.0]),
    ([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]),
    ([1.0, 2.0, 3.0], [np.inf, 2.0, 3.0]),
    ([[1.0, 2.0]], [[1.0, 2.0]]),
])
def test_user_metric_rejects_bad_input(values, covariate):
    with pytest.raises(ValueError):
        UserMetric(values, covariate)


def test_a_covariate_of_zeros_for_most_users_is_allowed():
    # new users have no pre-period data; their covariate is 0
    metric = UserMetric([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 5.0])
    assert metric.theta == pytest.approx(theta(metric.values, metric.covariate))


def test_the_own_theta_and_correlation_properties(balanced):
    control, treatment = balanced
    assert control.theta == pytest.approx(20 / 26)
    assert control.correlation == pytest.approx(np.corrcoef(CONTROL_Y, CONTROL_X)[0, 1])
    assert treatment.correlation == pytest.approx(np.corrcoef(TREATMENT_Y, TREATMENT_X)[0, 1])


def test_correlation_is_undefined_for_a_constant_array():
    with pytest.raises(ValueError):
        UserMetric([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]).correlation


# ---------------------------------------------------------------- adjust

def test_adjust_with_the_pooled_theta_matches_the_hand_computation(balanced):
    control, treatment = balanced
    # y - 1 * (x - 4) for each user
    np.testing.assert_allclose(control.adjust(theta=1.0, covariate_mean=4.0), [2.0, 7.0, 5.0, 6.0, 5.0])
    np.testing.assert_allclose(treatment.adjust(theta=1.0, covariate_mean=4.0), [5.0, 8.0, 6.0, 7.0, 9.0])


def test_adjust_returns_a_new_float_array_and_leaves_the_metric_alone(balanced):
    control, _ = balanced
    adjusted = control.adjust(theta=1.0, covariate_mean=4.0)
    assert adjusted is not control.values and adjusted.dtype == float
    np.testing.assert_array_equal(control.values, CONTROL_Y)


def test_adjust_defaults_to_the_groups_own_theta_and_mean(balanced):
    control, _ = balanced
    np.testing.assert_allclose(control.adjust(), control.adjust(theta=control.theta, covariate_mean=control.covariate.mean()))


def test_adjusting_with_the_own_theta_keeps_the_mean_and_removes_rho_squared_of_the_variance():
    rng = np.random.default_rng(3)
    y, x = correlated_pairs(rng, 2000, 0.7)
    metric = UserMetric(y, x)
    adjusted = metric.adjust()
    assert adjusted.mean() == pytest.approx(y.mean())
    rho = metric.correlation
    assert adjusted.var(ddof=1) == pytest.approx(y.var(ddof=1) * (1 - rho**2), rel=1e-10)


# ---------------------------------------------------------------- cuped_test

def test_cuped_test_on_balanced_covariates_keeps_the_estimate_and_shrinks_the_standard_error(balanced):
    control, treatment = balanced
    raw = welch_t_test(control.values, treatment.values)
    result = cuped_test(control, treatment)
    assert isinstance(result, ComparisonResult)
    assert raw.estimate == pytest.approx(2.0) and result.estimate == pytest.approx(2.0)
    assert raw.standard_error == pytest.approx(math.sqrt(7 / 5 + 11.5 / 5))
    assert result.standard_error == pytest.approx(math.sqrt(3.5 / 5 + 2.5 / 5))
    assert result.p_value < raw.p_value


def test_cuped_test_is_welch_on_the_adjusted_values(balanced):
    control, treatment = balanced
    expected = welch_t_test(control.adjust(theta=1.0, covariate_mean=4.0), treatment.adjust(theta=1.0, covariate_mean=4.0))
    result = cuped_test(control, treatment)
    assert result == expected


def test_cuped_test_uses_one_theta_from_the_pooled_users(skewed):
    control, treatment = skewed
    result = cuped_test(control, treatment)
    # pooled theta is 1; the groups' own slopes are 0.581 and 1.176 and would give 0.945
    assert result.estimate == pytest.approx(0.8)


def test_the_estimate_is_the_raw_difference_corrected_by_theta_times_the_covariate_difference(skewed):
    control, treatment = skewed
    raw = welch_t_test(control.values, treatment.values).estimate
    pooled_theta = theta(np.concatenate([control.values, treatment.values]), np.concatenate([control.covariate, treatment.covariate]))
    covariate_difference = treatment.covariate.mean() - control.covariate.mean()
    assert cuped_test(control, treatment).estimate == pytest.approx(raw - pooled_theta * covariate_difference)


def test_cuped_is_the_regression_coefficient_on_the_treatment_dummy_when_covariate_means_are_equal(balanced):
    control, treatment = balanced
    y = np.concatenate([control.values, treatment.values])
    x = np.concatenate([control.covariate, treatment.covariate])
    z = np.concatenate([np.zeros(5), np.ones(5)])
    fit = ols(y, np.column_stack([np.ones(10), z, x]))
    assert fit.coef[1] == pytest.approx(cuped_test(control, treatment).estimate, rel=1e-10)


def test_swapping_the_groups_flips_the_sign(balanced):
    control, treatment = balanced
    assert cuped_test(treatment, control).estimate == pytest.approx(-cuped_test(control, treatment).estimate)


def test_the_result_carries_welch_degrees_of_freedom(balanced):
    control, treatment = balanced
    result = cuped_test(control, treatment)
    v_c, v_t = 3.5 / 5, 2.5 / 5
    assert result.df == pytest.approx((v_c + v_t) ** 2 / (v_c**2 / 4 + v_t**2 / 4))
    assert math.isfinite(result.df)


def test_alpha_is_keyword_only_and_only_changes_the_interval(balanced):
    control, treatment = balanced
    parameter = inspect.signature(cuped_test).parameters["alpha"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default == 0.05
    narrow, wide = cuped_test(control, treatment, alpha=0.05), cuped_test(control, treatment, alpha=0.20)
    assert narrow.estimate == wide.estimate and narrow.p_value == wide.p_value
    assert narrow.ci_high - narrow.ci_low > wide.ci_high - wide.ci_low


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_cuped_test_rejects_an_impossible_alpha(balanced, alpha):
    control, treatment = balanced
    with pytest.raises(ValueError):
        cuped_test(control, treatment, alpha=alpha)


def test_cuped_test_rejects_a_constant_pooled_covariate():
    control = UserMetric([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    treatment = UserMetric([2.0, 3.0, 4.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        cuped_test(control, treatment)


def test_a_binary_covariate_is_post_stratification():
    # Deng et al. 2013, appendix A: with a 0/1 covariate and the group's own theta, the adjusted mean
    # centred on the pooled share w is exactly the stratified mean w * mean(returning) + (1 - w) * mean(new)
    rng = np.random.default_rng(4)
    returning = rng.random(600) < 0.3
    y = np.where(returning, 8.0, 2.0) + rng.standard_normal(600)
    treated = np.zeros(600, dtype=bool)
    treated[rng.permutation(600)[300:]] = True
    weight = returning.mean()
    for group in [~treated, treated]:
        metric = UserMetric(y[group], returning[group])
        stratified = weight * y[group & returning].mean() + (1 - weight) * y[group & ~returning].mean()
        assert metric.theta == pytest.approx(y[group & returning].mean() - y[group & ~returning].mean())
        assert metric.adjust(covariate_mean=weight).mean() == pytest.approx(stratified, rel=1e-10)


# ---------------------------------------------------------------- in the harness's terms

@pytest.fixture(scope="module")
def population():
    rng = np.random.default_rng(5)
    y, x = correlated_pairs(rng, 2000, 0.8)
    return y, x


def halvings(n_users, n_splits, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_splits):
        treated = np.zeros(n_users, dtype=bool)
        treated[rng.permutation(n_users)[n_users // 2:]] = True
        yield treated


def test_cuped_keeps_the_false_alarm_rate_at_alpha(population):
    y, x = population
    p_values = np.array([cuped_test(UserMetric(y[~t], x[~t]), UserMetric(y[t], x[t])).p_value for t in halvings(2000, 1000, 6)])
    # 1,000 A/A tests measure 5% to about +-0.7 points
    assert 0.03 <= np.mean(p_values < 0.05) <= 0.075


def test_the_standard_error_shrinks_by_the_square_root_of_one_minus_rho_squared(population):
    y, x = population
    rho = UserMetric(y, x).correlation
    ratios = []
    for t in halvings(2000, 200, 7):
        raw = welch_t_test(y[~t], y[t]).standard_error
        ratios.append(cuped_test(UserMetric(y[~t], x[~t]), UserMetric(y[t], x[t])).standard_error / raw)
    assert np.mean(ratios) == pytest.approx(math.sqrt(1 - rho**2), abs=0.02)


def test_cuped_is_unbiased_and_more_powerful_with_a_real_effect(population):
    y, x = population
    lift = 0.3  # about a tenth of a standard deviation: raw power well below CUPED's
    estimates, raw_rejections, cuped_rejections = [], 0, 0
    for t in halvings(2000, 400, 8):
        treatment_y = y[t] + lift
        raw_rejections += welch_t_test(y[~t], treatment_y).p_value < 0.05
        result = cuped_test(UserMetric(y[~t], x[~t]), UserMetric(treatment_y, x[t]))
        cuped_rejections += result.p_value < 0.05
        estimates.append(result.estimate)
    assert np.mean(estimates) == pytest.approx(lift, abs=0.02)
    assert cuped_rejections / 400 > raw_rejections / 400 + 0.2


def test_a_covariate_moved_by_the_treatment_biases_the_estimate(population):
    # Deng et al. 2013, section 5.3: the covariate must not be affected by the treatment
    y, x = population
    lift = 0.5
    estimates = []
    for t in halvings(2000, 200, 9):
        moved = x[t] + lift  # the treatment also raises the "covariate" by the same amount
        estimates.append(cuped_test(UserMetric(y[~t], x[~t]), UserMetric(y[t] + lift, moved)).estimate)
    theta_pooled = theta(y, x)
    assert np.mean(estimates) == pytest.approx(lift - theta_pooled * lift, abs=0.03)
    assert np.mean(estimates) < lift - 0.2
