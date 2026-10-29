"""Chapter 4: abkit.bootstrap, the bootstrap for a difference between two groups.

Synthetic data only, so these run in a few seconds. The real-data checks are in
test_bootstrap_data.py.
"""

import dataclasses
import math
import tracemalloc
from functools import partial
from statistics import NormalDist

import numpy as np
import pytest

from abkit.bootstrap import BootstrapResult, bootstrap, bootstrap_ratio


def mc_tolerance_for_an_sd(n_resamples):
    """Four Monte Carlo standard errors of an SD estimated from n_resamples draws, relative.

    The SD of an SD estimated from B roughly normal values is about SD / sqrt(2(B - 1)).
    """
    return 4 / math.sqrt(2 * (n_resamples - 1))


@pytest.fixture
def groups():
    rng = np.random.default_rng(1)
    return rng.normal(10, 2, size=200), rng.normal(11, 3, size=300)


def sessions_data():
    """Per-user numerators and denominators for two groups of different sizes."""
    rng = np.random.default_rng(6)
    den_c, den_t = rng.integers(1, 20, 300), rng.integers(1, 20, 400)
    num_c, num_t = rng.binomial(den_c, 0.1), rng.binomial(den_t, 0.12)
    return num_c.astype(float), den_c.astype(float), num_t.astype(float), den_t.astype(float)


# --- what the result contains ---------------------------------------------------------

def test_estimate_is_the_statistic_on_the_original_data(groups):
    control, treatment = groups

    result = bootstrap(control, treatment, n_resamples=200)

    assert result.estimate == pytest.approx(treatment.mean() - control.mean(), rel=1e-12)


def test_relative_estimate_is_the_ratio_minus_one(groups):
    control, treatment = groups

    result = bootstrap(control, treatment, n_resamples=200, relative=True)

    assert result.estimate == pytest.approx(treatment.mean() / control.mean() - 1, rel=1e-12)


def test_distribution_has_one_value_per_resample(groups):
    result = bootstrap(*groups, n_resamples=321)

    assert result.distribution.shape == (321,)


def test_defaults_are_2000_resamples_and_seed_0(groups):
    default = bootstrap(*groups)

    assert default.distribution.shape == (2000,)
    assert np.array_equal(default.distribution, bootstrap(*groups, seed=0).distribution)


def test_standard_error_is_the_sd_of_the_distribution(groups):
    # ISLP eq. 5.8: the bootstrap SE is the standard deviation of the B bootstrap estimates, with B - 1.
    result = bootstrap(*groups, n_resamples=500)

    assert result.standard_error == pytest.approx(result.distribution.std(ddof=1), rel=1e-12)


def test_percentile_interval_cuts_alpha_over_two_from_each_end(groups):
    # The spec says np.quantile with its default (linear) interpolation.
    result = bootstrap(*groups, n_resamples=1000, alpha=0.1, method="percentile")

    low, high = np.quantile(result.distribution, [0.05, 0.95])
    assert (result.ci_low, result.ci_high) == pytest.approx((low, high), rel=1e-12)


def test_normal_interval_is_estimate_plus_minus_z_times_se(groups):
    # z = Φ⁻¹(1 - alpha/2), here 1.645 for alpha = 0.1, not a hard-coded 1.96.
    result = bootstrap(*groups, n_resamples=1000, alpha=0.1, method="normal")

    half = NormalDist().inv_cdf(0.95) * result.standard_error
    assert (result.ci_low, result.ci_high) == pytest.approx((result.estimate - half, result.estimate + half), rel=1e-12)
    assert result.method == "normal"
    assert result.alpha == 0.1


def test_result_is_a_frozen_bootstrap_result(groups):
    result = bootstrap(*groups, n_resamples=100)

    assert isinstance(result, BootstrapResult)
    assert result.method == "percentile"
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.estimate = 0.0


def test_distribution_is_read_only(groups):
    # A frozen data class only stops you replacing the array; this stops you changing its contents.
    result = bootstrap(*groups, n_resamples=100)

    with pytest.raises(ValueError):
        result.distribution[0] = 1e9


# --- how the resampling works ---------------------------------------------------------

def test_bootstrap_se_of_a_mean_matches_the_plug_in_formula():
    # Each bootstrap draw picks one of the n values at random, so its variance is the ddof=0 variance.
    # Ten values per group make the gap from the ddof=1 formula about 5%, big enough to see.
    rng = np.random.default_rng(2)
    control, treatment = rng.exponential(1.0, size=10), rng.exponential(2.0, size=10)
    n_resamples = 20_000

    result = bootstrap(control, treatment, n_resamples=n_resamples)

    plug_in = math.sqrt(control.var(ddof=0) / 10 + treatment.var(ddof=0) / 10)
    assert result.standard_error == pytest.approx(plug_in, rel=mc_tolerance_for_an_sd(n_resamples))


def test_the_first_and_last_values_can_be_drawn():
    # Drawing positions from 0 to n - 2 (or 1 to n - 1) by mistake would make this SE 21% smaller.
    control = np.r_[1.0, np.zeros(8), 1.0]
    n_resamples = 20_000

    result = bootstrap(control, np.zeros(10), n_resamples=n_resamples)

    assert result.standard_error == pytest.approx(math.sqrt(control.var() / 10), rel=mc_tolerance_for_an_sd(n_resamples))


def test_groups_are_resampled_separately():
    # Every resample of a constant group is the same constant, so the difference never moves.
    # Resampling the pooled data instead would mix 0s and 1s and give a positive SE.
    control, treatment = np.zeros(50), np.ones(50)

    result = bootstrap(control, treatment, n_resamples=500)

    assert result.standard_error == 0
    assert (result.ci_low, result.ci_high) == (1, 1)


def test_the_two_groups_get_independent_resamples():
    # Two identical groups: if both were drawn at the same positions, every difference would be exactly 0.
    values = np.arange(50.0)
    plug_in = math.sqrt(2 * values.var() / values.size)

    result = bootstrap(values, values.copy(), n_resamples=2000)
    ratio = bootstrap_ratio(values, np.ones(50), values.copy(), np.ones(50), n_resamples=2000)

    assert result.standard_error == pytest.approx(plug_in, rel=mc_tolerance_for_an_sd(2000))
    assert ratio.standard_error == pytest.approx(plug_in, rel=mc_tolerance_for_an_sd(2000))


def group_size(values, axis=-1):
    """A 'statistic' that reports how many values each resample has."""
    return np.full(np.shape(values)[:-1], np.shape(values)[axis], dtype=float)


def test_each_resample_has_the_size_of_its_group():
    control, treatment = np.arange(30.0), np.arange(70.0)

    result = bootstrap(control, treatment, statistic=group_size, n_resamples=200)

    assert result.estimate == 40
    assert np.all(result.distribution == 40)


def test_works_with_other_statistics(groups):
    control, treatment = groups
    q90 = partial(np.quantile, q=0.9)

    for statistic in (np.median, q90):
        result = bootstrap(control, treatment, statistic=statistic, n_resamples=500)
        expected = float(statistic(treatment, axis=-1) - statistic(control, axis=-1))
        assert result.estimate == pytest.approx(expected, rel=1e-12)
        assert result.standard_error > 0
        assert result.ci_low < result.estimate < result.ci_high


def test_same_seed_same_result_and_different_seed_different_result(groups):
    first = bootstrap(*groups, n_resamples=300, seed=7)
    again = bootstrap(*groups, n_resamples=300, seed=7)
    other = bootstrap(*groups, n_resamples=300, seed=8)

    assert np.array_equal(first.distribution, again.distribution)
    assert not np.array_equal(first.distribution, other.distribution)


def test_resamples_are_drawn_in_batches_of_batch_size():
    # batch_size is the number of resamples drawn in one vectorised call, so the statistic never
    # receives more rows than that, and it sees every resample of both groups exactly once.
    rows = []

    def recording_mean(values, axis=-1):
        if np.ndim(values) == 2:
            rows.append(np.shape(values)[0])
        return np.mean(values, axis=axis)

    bootstrap(np.arange(10.0), np.arange(12.0), statistic=recording_mean, n_resamples=20, batch_size=7)

    assert max(rows) <= 7
    assert sum(rows) == 2 * 20


def test_batch_size_changes_nothing_but_the_noise(groups):
    n_resamples = 4000

    one_go = bootstrap(*groups, n_resamples=n_resamples, batch_size=n_resamples)
    small = bootstrap(*groups, n_resamples=n_resamples, batch_size=7)

    # Two noisy SEs: the SD of their difference is sqrt(2) times one SE's Monte Carlo error.
    assert small.distribution.shape == (n_resamples,)
    assert small.standard_error == pytest.approx(one_go.standard_error, rel=math.sqrt(2) * mc_tolerance_for_an_sd(n_resamples))


def test_memory_stays_bounded_for_large_groups():
    # 100,000 values per group and 500 resamples in one go would hold 500 x 100,000 indices and
    # values per group: about 800 MB. By default the work must be split into batches.
    rng = np.random.default_rng(3)
    control, treatment = rng.exponential(size=100_000), rng.exponential(size=100_000)

    tracemalloc.start()
    bootstrap(control, treatment, n_resamples=500)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 250e6, f"peak memory {peak / 1e6:.0f} MB"


# --- relative lift ---------------------------------------------------------------------

def test_relative_se_includes_the_control_means_noise():
    # Kohavi eq. 18.6 at +50%. Dividing by the original control mean, or var(delta) / mean_c**2,
    # would make this SE about 21% too small. Resample both groups and recompute the ratio each time.
    rng = np.random.default_rng(10)
    control, treatment = rng.normal(10, 5, 2000), rng.normal(15, 5, 2000)
    n_resamples = 4000

    result = bootstrap(control, treatment, n_resamples=n_resamples, relative=True)

    mc, mt = control.mean(), treatment.mean()
    vc, vt = control.var() / control.size, treatment.var() / treatment.size
    eq_18_6 = math.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
    assert result.standard_error == pytest.approx(eq_18_6, rel=mc_tolerance_for_an_sd(n_resamples))


def test_relative_lift_refuses_a_resampled_control_statistic_of_zero():
    # One 5 among nine 0s: about a third of the resamples miss the 5, so their mean is 0 and the
    # lift would be infinite. The spec says to raise ValueError rather than return inf or nan.
    control = np.r_[np.zeros(9), 5.0]

    with pytest.raises(ValueError):
        bootstrap(control, np.arange(10.0), n_resamples=200, relative=True)


# --- does it work as an interval? -----------------------------------------------------

def coverage(make_groups, true_difference, repetitions, n_resamples, method, seed):
    rng = np.random.default_rng(seed)
    covered = 0
    for repetition in range(repetitions):
        control, treatment = make_groups(rng)
        result = bootstrap(control, treatment, n_resamples=n_resamples, method=method, seed=repetition)
        covered += result.ci_low <= true_difference <= result.ci_high
    return covered / repetitions


def test_intervals_cover_the_truth_about_95_percent_of_the_time():
    # 300 repetitions measure coverage to about ±1.3 points. The band is wide enough for any correct
    # implementation, whatever order it draws random numbers in, and still catches a broken one.
    rate = coverage(lambda rng: (rng.normal(0, 1, 300), rng.normal(0.5, 1, 300)), 0.5,
                    repetitions=300, n_resamples=500, method="percentile", seed=4)

    assert 0.90 <= rate <= 0.99


def test_percentile_intervals_undercover_on_small_skewed_samples():
    # Section 3.5: with ten lognormal values per group, the percentile interval follows the sample's
    # chance skew and covers the truth about 89-91% of the time instead of 95%.
    rate = coverage(lambda rng: (rng.lognormal(size=10), rng.lognormal(size=10)), 0.0,
                    repetitions=1000, n_resamples=1000, method="percentile", seed=5)

    assert 0.85 <= rate <= 0.93


# --- ratio metrics: resample users, not sessions --------------------------------------

def users_with_sessions(n_users, rng):
    """Users who differ in how often they buy, so one user's sessions are correlated."""
    sessions = rng.geometric(0.3, size=n_users)
    buys_often = rng.random(n_users) < 0.1
    purchases = rng.binomial(sessions, np.where(buys_often, 0.6, 0.02))
    return purchases.astype(float), sessions.astype(float)


def delta_method_se(numerator, denominator):
    """Kohavi eq. 18.5 for the variance of a ratio of two user-level means."""
    n = numerator.size
    mx, my = numerator.mean(), denominator.mean()
    vx, vy = numerator.var(ddof=1) / n, denominator.var(ddof=1) / n
    cov = np.cov(numerator, denominator, ddof=1)[0, 1] / n
    return math.sqrt(vx / my**2 + mx**2 * vy / my**4 - 2 * mx * cov / my**3)


def test_ratio_estimate_is_a_ratio_of_sums():
    num_c, den_c = np.array([0.0, 1, 2]), np.array([1.0, 2, 5])
    num_t, den_t = np.array([1.0, 1]), np.array([2.0, 2])

    result = bootstrap_ratio(num_c, den_c, num_t, den_t, n_resamples=100)

    assert result.estimate == pytest.approx(2 / 4 - 3 / 8, rel=1e-12)


def test_ratio_passes_alpha_and_method_through():
    data = sessions_data()

    percentile = bootstrap_ratio(*data, n_resamples=500, alpha=0.1, batch_size=50)
    normal = bootstrap_ratio(*data, n_resamples=500, alpha=0.1, method="normal")

    assert percentile.distribution.shape == (500,)
    low, high = np.quantile(percentile.distribution, [0.05, 0.95])
    assert (percentile.ci_low, percentile.ci_high) == pytest.approx((low, high), rel=1e-12)
    half = NormalDist().inv_cdf(0.95) * normal.standard_error
    assert (normal.ci_low, normal.ci_high) == pytest.approx((normal.estimate - half, normal.estimate + half), rel=1e-12)
    assert (normal.method, normal.alpha) == ("normal", 0.1)


def test_a_user_is_resampled_with_all_of_their_sessions():
    # Every user's numerator equals their denominator, so every resample has a ratio of exactly 1.
    # Drawing numerators and denominators independently would break that.
    rng = np.random.default_rng(6)
    sessions_c, sessions_t = rng.integers(1, 20, 100).astype(float), rng.integers(1, 20, 100).astype(float)

    result = bootstrap_ratio(sessions_c, sessions_c, sessions_t, sessions_t, n_resamples=300)

    assert result.standard_error == pytest.approx(0, abs=1e-12)


def test_ratio_se_matches_the_delta_method():
    rng = np.random.default_rng(7)
    (num_c, den_c), (num_t, den_t) = users_with_sessions(4000, rng), users_with_sessions(4000, rng)
    n_resamples = 4000

    result = bootstrap_ratio(num_c, den_c, num_t, den_t, n_resamples=n_resamples)

    delta = math.hypot(delta_method_se(num_c, den_c), delta_method_se(num_t, den_t))
    assert result.standard_error == pytest.approx(delta, rel=mc_tolerance_for_an_sd(n_resamples))


def test_treating_sessions_as_independent_understates_the_se():
    rng = np.random.default_rng(8)
    (num_c, den_c), (num_t, den_t) = users_with_sessions(4000, rng), users_with_sessions(4000, rng)

    result = bootstrap_ratio(num_c, den_c, num_t, den_t, n_resamples=2000)

    def session_level_variance(num, den):
        rate = num.sum() / den.sum()
        return rate * (1 - rate) / den.sum()

    naive = math.sqrt(session_level_variance(num_c, den_c) + session_level_variance(num_t, den_t))
    assert result.standard_error > 1.5 * naive


def test_ratio_with_one_session_per_user_is_a_bootstrap_of_means():
    rng = np.random.default_rng(9)
    conv_c, conv_t = (rng.random(3000) < 0.1).astype(float), (rng.random(3000) < 0.12).astype(float)
    n_resamples = 4000

    result = bootstrap_ratio(conv_c, np.ones(3000), conv_t, np.ones(3000), n_resamples=n_resamples)

    plug_in = math.sqrt(conv_c.var(ddof=0) / 3000 + conv_t.var(ddof=0) / 3000)
    assert result.estimate == pytest.approx(conv_t.mean() - conv_c.mean(), rel=1e-12)
    assert result.standard_error == pytest.approx(plug_in, rel=mc_tolerance_for_an_sd(n_resamples))


# --- impossible inputs ---------------------------------------------------------------

@pytest.mark.parametrize("call", [
    pytest.param(lambda: bootstrap(np.array([1.0]), np.arange(5.0)), id="one-value-control"),
    pytest.param(lambda: bootstrap(np.arange(5.0), np.array([1.0])), id="one-value-treatment"),
    pytest.param(lambda: bootstrap(np.arange(5.0), np.arange(5.0), alpha=0.0), id="alpha-zero"),
    pytest.param(lambda: bootstrap(np.arange(5.0), np.arange(5.0), alpha=1.0), id="alpha-one"),
    pytest.param(lambda: bootstrap(np.arange(5.0), np.arange(5.0), alpha=1.5), id="alpha-above-one"),
    pytest.param(lambda: bootstrap(np.arange(5.0), np.arange(5.0), n_resamples=1), id="one-resample"),
    pytest.param(lambda: bootstrap(np.arange(5.0), np.arange(5.0), method="bca"), id="unknown-method"),
    pytest.param(lambda: bootstrap(np.arange(5.0), np.arange(5.0), batch_size=0), id="batch-size-zero"),
    pytest.param(lambda: bootstrap(np.zeros(5), np.arange(5.0), relative=True), id="relative-to-zero"),
    pytest.param(lambda: bootstrap_ratio(np.ones(3), np.ones(4), np.ones(3), np.ones(3)), id="ratio-length-mismatch"),
    pytest.param(lambda: bootstrap_ratio(np.ones(3), np.zeros(3), np.ones(3), np.ones(3)), id="ratio-zero-control-denominator"),
    pytest.param(lambda: bootstrap_ratio(np.ones(3), np.ones(3), np.ones(3), np.zeros(3)), id="ratio-zero-treatment-denominator"),
    pytest.param(lambda: bootstrap_ratio(np.ones(1), np.ones(1), np.ones(3), np.ones(3)), id="ratio-one-user"),
    pytest.param(lambda: bootstrap_ratio(np.ones(3), np.ones(3), np.ones(3), np.ones(3), alpha=0.0), id="ratio-alpha-zero"),
    pytest.param(lambda: bootstrap_ratio(np.ones(3), np.ones(3), np.ones(3), np.ones(3), n_resamples=1), id="ratio-one-resample"),
    pytest.param(lambda: bootstrap_ratio(np.ones(3), np.ones(3), np.ones(3), np.ones(3), method="bca"), id="ratio-unknown-method"),
    pytest.param(lambda: bootstrap_ratio(np.r_[np.zeros(9), 1.0], np.r_[np.zeros(9), 1.0], np.ones(10), np.ones(10),
                                         n_resamples=200), id="ratio-resampled-denominator-zero"),
])
def test_rejects_impossible_inputs(call):
    with pytest.raises(ValueError):
        call()
