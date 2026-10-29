"""Chapter 4: abkit.mannwhitney, and what the Mann-Whitney test actually detects.

SciPy's mannwhitneyu is the oracle. Note the argument order: SciPy's statistic is the U of its first
argument, so the treatment goes first there.
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from abkit.harness import run_aa
from abkit.mannwhitney import MannWhitneyResult, mann_whitney


def scipy_mwu(control, treatment):
    return stats.mannwhitneyu(treatment, control, method="asymptotic", use_continuity=True)


# --- the statistic ------------------------------------------------------------------

def test_u_counts_the_pairs_where_treatment_is_higher():
    # The chapter's tiny example: 17 of the 25 (control, treatment) pairs have the treatment value higher.
    control, treatment = np.array([2.0, 4, 4, 6, 9]), np.array([3.0, 5, 7, 8, 12])

    result = mann_whitney(control, treatment)

    assert result.u == 17
    assert result.prob_superiority == pytest.approx(0.68)
    # Section 4.1: z = (17 - 12.5 - 0.5) / sqrt(25/12 * (11 - 6/90)) = 0.838, two-sided p = 0.402.
    assert result.statistic == pytest.approx(4 / math.sqrt(25 / 12 * (11 - 6 / 90)), rel=1e-12)
    assert result.p_value == pytest.approx(0.40197, abs=1e-5)


def test_a_tie_counts_as_half_a_pair():
    # Pairs with t > c: 1 + 2 + 3 = 6. Tied pairs: (2, 2) and (3, 3), worth 1/2 each. So U = 7.
    result = mann_whitney(np.array([1.0, 2, 3]), np.array([2.0, 3, 4]))

    assert result.u == 7
    assert result.prob_superiority == pytest.approx(7 / 9)


def test_u_equals_a_brute_force_pair_count():
    rng = np.random.default_rng(1)
    control, treatment = rng.integers(0, 6, 40).astype(float), rng.integers(0, 7, 55).astype(float)

    result = mann_whitney(control, treatment)

    diff = treatment[:, None] - control[None, :]
    assert result.u == (diff > 0).sum() + 0.5 * (diff == 0).sum()


@pytest.mark.parametrize("seed", range(6))
def test_matches_scipy_with_and_without_ties(seed):
    rng = np.random.default_rng(seed)
    if seed % 2:
        control, treatment = rng.integers(0, 5, 80).astype(float), rng.integers(0, 6, 120).astype(float)
    else:
        control, treatment = rng.lognormal(size=80), rng.lognormal(0.3, 1, size=120)

    ours, theirs = mann_whitney(control, treatment), scipy_mwu(control, treatment)

    # SciPy doesn't report z, but its p-value determines |z|, and the sign follows U - n_c n_t / 2.
    z = math.copysign(stats.norm.isf(theirs.pvalue / 2), theirs.statistic - control.size * treatment.size / 2)
    assert ours.u == theirs.statistic
    assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-6, abs=1e-12)
    assert ours.statistic == pytest.approx(z, rel=1e-6, abs=1e-9)


def test_matches_scipy_on_zero_inflated_revenue():
    # Most users spend nothing, so most values are tied at 0: the tie correction matters here.
    rng = np.random.default_rng(10)
    control = np.where(rng.random(3000) < 0.07, rng.lognormal(3, 1, 3000), 0.0)
    treatment = np.where(rng.random(3000) < 0.08, rng.lognormal(3, 1, 3000), 0.0)

    ours, theirs = mann_whitney(control, treatment), scipy_mwu(control, treatment)

    assert ours.u == theirs.statistic
    assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-6, abs=1e-12)


def test_a_tie_group_of_millions_does_not_overflow():
    # (counts**3 - counts) on integer counts overflows int64 silently once a tie group passes
    # 2,097,152 values, which a month of users with 93% zeros easily does. Convert to float first.
    control, treatment = np.zeros(1_100_000), np.r_[np.zeros(1_100_000), np.ones(2000)]

    ours, theirs = mann_whitney(control, treatment), scipy_mwu(control, treatment)

    assert ours.p_value == pytest.approx(theirs.pvalue, rel=1e-6, abs=1e-12)


def test_swapping_the_groups_mirrors_the_result():
    rng = np.random.default_rng(2)
    control, treatment = rng.normal(0, 1, 30), rng.normal(0.4, 1, 50)

    forward, backward = mann_whitney(control, treatment), mann_whitney(treatment, control)

    assert backward.u == 30 * 50 - forward.u
    assert backward.prob_superiority == pytest.approx(1 - forward.prob_superiority)
    assert backward.statistic == pytest.approx(-forward.statistic)
    assert backward.p_value == pytest.approx(forward.p_value)


def test_identical_groups_give_a_p_value_of_one():
    values = np.array([1.0, 3, 3, 8, 20])

    result = mann_whitney(values, values.copy())

    assert result.u == 12.5
    assert result.statistic == 0
    assert result.p_value == 1


def test_only_the_order_of_values_matters():
    # Ranks don't change under any increasing transformation, so neither does the test.
    rng = np.random.default_rng(3)
    control, treatment = rng.lognormal(size=60), rng.lognormal(0.2, 1, size=60)

    raw, logged = mann_whitney(control, treatment), mann_whitney(np.log(control), np.log(treatment))

    assert logged.u == raw.u
    assert logged.p_value == pytest.approx(raw.p_value, rel=1e-12)


def test_result_is_a_frozen_mann_whitney_result():
    result = mann_whitney(np.arange(5.0), np.arange(5.0) + 1)

    assert isinstance(result, MannWhitneyResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.u = 0.0


# --- what it detects ----------------------------------------------------------------

def revenue(n, buy_rate, rng, spend_scale=1.0):
    """Zero-inflated revenue: a share of buyers, each spending a lognormal amount."""
    return np.where(rng.random(n) < buy_rate, rng.lognormal(3, 1, n) * spend_scale, 0.0)


def mw_p(control, treatment):
    return mann_whitney(control, treatment).p_value


def welch_p(control, treatment):
    return stats.ttest_ind(treatment, control, equal_var=False).pvalue


def rejection_rates(make_treatment, n_simulations, n, seed):
    rng = np.random.default_rng(seed)
    rejected_mw = rejected_welch = 0
    for _ in range(n_simulations):
        control, treatment = revenue(n, 0.1, rng), make_treatment(rng)
        rejected_mw += mw_p(control, treatment) < 0.05
        rejected_welch += welch_p(control, treatment) < 0.05
    return rejected_mw / n_simulations, rejected_welch / n_simulations


def test_approximately_holds_alpha_on_heavy_tailed_data_even_with_a_10_90_split():
    # In an A/A test both groups come from one distribution, so every arrangement of ranks is equally
    # likely, for any split and any tail. The normal approximation makes the rate only approximately
    # 5% (about 5.7% on this data set), and the band leaves room for how your harness draws splits.
    rng = np.random.default_rng(4)
    values = revenue(1000, 0.07, rng)

    result = run_aa(values, mw_p, n_simulations=1000, seed=5, split=0.1)

    assert 0.03 <= result.rejection_rate <= 0.08


def test_rejects_when_more_people_buy_even_if_the_mean_is_flat():
    # 12% buyers spending 10/12 as much: the mean is unchanged, but P(T > C) rises above 1/2.
    mw, welch = rejection_rates(lambda rng: revenue(5000, 0.12, rng, spend_scale=0.10 / 0.12),
                                n_simulations=200, n=5000, seed=6)

    assert mw > 0.65
    assert welch < 0.10


def test_barely_reacts_when_only_the_biggest_spenders_spend_more():
    # Tripling the top 1% moves the mean a lot, but changes the order of very few pairs.
    threshold = np.quantile(revenue(1_000_000, 0.1, np.random.default_rng(7)), 0.99)

    def whales_spend_three_times_as_much(rng):
        treatment = revenue(5000, 0.1, rng)
        return np.where(treatment > threshold, treatment * 3, treatment)

    mw, welch = rejection_rates(whales_spend_three_times_as_much, n_simulations=200, n=5000, seed=8)

    assert mw < 0.10
    assert welch > mw + 0.3


# --- impossible inputs --------------------------------------------------------------

@pytest.mark.parametrize("control, treatment", [
    pytest.param(np.array([]), np.arange(5.0), id="empty-control"),
    pytest.param(np.arange(5.0), np.array([]), id="empty-treatment"),
    pytest.param(np.ones(4), np.ones(6), id="all-values-identical"),
])
def test_rejects_impossible_inputs(control, treatment):
    with pytest.raises(ValueError):
        mann_whitney(control, treatment)
