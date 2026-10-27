"""Acceptance tests for chapter 3: power, sample size, MDE and design analysis (abkit.power).

Written for you: make them pass. statsmodels and OpenIntro's worked examples are the oracles for
the formulas; plain simulation is the oracle for Gelman & Carlin's design quantities.
"""

import dataclasses
import math
from statistics import NormalDist

import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower

from abkit.power import DesignResult, design, mde, power, sample_size

ALPHA = 0.05
NORMAL = NormalDist()


def z_star(alpha):
    return NORMAL.inv_cdf(1 - alpha / 2)


# --- power ------------------------------------------------------------------------

def test_power_matches_openintro_blood_pressure_example():
    # OpenIntro pp. 280-281: a 3 mmHg effect, sd 12, about 42% power with 100 per group
    # and 97.7% with 500 per group.
    assert power(3.0, 12.0, 100, 100) == pytest.approx(0.42, abs=0.005)
    assert power(3.0, 12.0, 500, 500) == pytest.approx(0.977, abs=0.001)


@pytest.mark.parametrize("effect, sd, n_control, n_treatment, alpha", [
    pytest.param(3.0, 12.0, 100, 100, 0.05, id="openintro"),
    pytest.param(0.0035, 0.2535, 85_000, 85_000, 0.05, id="conversion-like"),
    pytest.param(0.16, 18.76, 205_036, 205_037, 0.05, id="january-revenue"),
    pytest.param(0.05, 1.0, 9_000, 1_000, 0.05, id="90-10-split"),
    pytest.param(0.1, 1.0, 1_500, 2_500, 0.01, id="alpha-0.01"),
])
def test_power_matches_statsmodels(effect, sd, n_control, n_treatment, alpha):
    oracle = NormalIndPower().power(effect_size=abs(effect) / sd, nobs1=n_control, alpha=alpha,
                                    ratio=n_treatment / n_control, alternative="two-sided")

    assert power(effect, sd, n_control, n_treatment, alpha=alpha) == pytest.approx(oracle, rel=1e-9)


def test_power_with_different_spreads_in_the_two_groups():
    # SE² = sd_C²/n_C + sd_T²/n_T, as in chapter 2's Welch test. A multiplicative lift, for
    # example, stretches the treated group's spread by (1 + lift).
    effect, sd, sd_t, n_c, n_t = 0.1, 1.0, 1.5, 2_000, 1_000
    shift = effect / math.sqrt(sd ** 2 / n_c + sd_t ** 2 / n_t)
    by_hand = NORMAL.cdf(shift - z_star(ALPHA)) + NORMAL.cdf(-shift - z_star(ALPHA))

    assert power(effect, sd, n_c, n_t, sd_treatment=sd_t) == pytest.approx(by_hand, rel=1e-9)
    assert power(effect, sd, n_c, n_t, sd_treatment=sd) == pytest.approx(power(effect, sd, n_c, n_t))


def test_power_with_no_effect_is_alpha():
    # With no effect, "power" is the chance of a false alarm.
    assert power(0.0, 1.0, 100, 100) == pytest.approx(ALPHA)
    assert power(0.0, 1.0, 100, 100, alpha=0.01) == pytest.approx(0.01)


def test_power_grows_with_users_and_with_the_effect():
    assert power(0.1, 1.0, 100, 100) < power(0.1, 1.0, 400, 400) < power(0.1, 1.0, 1600, 1600)
    assert power(0.05, 1.0, 400, 400) < power(0.10, 1.0, 400, 400) < power(0.20, 1.0, 400, 400)


def test_power_ignores_the_sign_of_the_effect():
    assert power(-0.1, 1.0, 500, 700) == pytest.approx(power(0.1, 1.0, 500, 700))


# --- sample size ----------------------------------------------------------------

def test_sample_size_matches_openintro():
    # OpenIntro p. 282 rounds the multiplier to 2.8 and gets 250.88, so 251 per group.
    # With the exact quantiles it's 251.16, so 252 after rounding up.
    n_control, n_treatment = sample_size(3.0, 12.0)

    assert n_control == n_treatment
    assert 251 <= n_control <= 252


def test_sample_size_returns_whole_numbers_of_users():
    n_control, n_treatment = sample_size(0.1, 1.0, split=0.3)

    assert isinstance(n_control, int) and isinstance(n_treatment, int)


@pytest.mark.parametrize("effect, sd, split, target_power, alpha", [
    (3.0, 12.0, 0.5, 0.8, 0.05),
    (0.0035, 0.2535, 0.5, 0.8, 0.05),
    (0.1, 1.0, 0.1, 0.8, 0.05),
    (0.2, 2.0, 0.25, 0.8, 0.05),
    (0.1, 1.0, 0.5, 0.9, 0.01),
    (0.05, 1.0, 0.25, 0.9, 0.01),
])
def test_sample_size_matches_statsmodels(effect, sd, split, target_power, alpha):
    n_control, n_treatment = sample_size(effect, sd, target_power=target_power, alpha=alpha, split=split)

    oracle = NormalIndPower().solve_power(effect_size=effect / sd, alpha=alpha, power=target_power, ratio=split / (1 - split))
    assert n_control == pytest.approx(oracle, abs=1)
    assert n_treatment == pytest.approx(oracle * split / (1 - split), abs=1)


@pytest.mark.parametrize("target_power, alpha", [(0.8, 0.05), (0.9, 0.01)])
def test_sample_size_is_the_smallest_that_reaches_the_target(target_power, alpha):
    rng = np.random.default_rng(1)
    for _ in range(200):
        sd, split = rng.uniform(0.5, 5.0), rng.uniform(0.1, 0.9)
        effect = sd * rng.uniform(0.1, 1.0)

        n_control, n_treatment = sample_size(effect, sd, target_power=target_power, alpha=alpha, split=split)

        assert power(effect, sd, n_control, n_treatment, alpha=alpha) >= target_power - 1e-9
        assert power(effect, sd, n_control - 1, n_treatment - 1, alpha=alpha) < target_power


def test_kohavis_16_is_per_group():
    # Kohavi's rule of thumb (p. 189): n ≈ 16σ²/δ². The exact multiplier is 2(1.96 + 0.84)² ≈ 15.7,
    # and it gives the users in EACH group.
    n_control, n_treatment = sample_size(0.1, 1.0)

    assert n_control == n_treatment
    assert n_control == pytest.approx(16 * 1.0 ** 2 / 0.1 ** 2, rel=0.02)


def test_simulation_agrees_with_your_power_per_group_not_in_total():
    # Fresh normal samples, σ = 1, δ = 0.1. Simulated power at 1,600 users per group, and at 1,600
    # users in total (800 per group), must match your power() for those sizes.
    rng = np.random.default_rng(2)

    def simulated_power(n_per_group, n_simulations=2000):
        diff = rng.normal(0.1, 1.0, (n_simulations, n_per_group)).mean(axis=1) - rng.normal(0.0, 1.0, (n_simulations, n_per_group)).mean(axis=1)
        return np.mean(np.abs(diff / math.sqrt(2 / n_per_group)) > z_star(ALPHA))

    for n_per_group in (1600, 800):
        expected = power(0.1, 1.0, n_per_group, n_per_group)
        tolerance = 4 * math.sqrt(expected * (1 - expected) / 2000)
        assert simulated_power(n_per_group) == pytest.approx(expected, abs=tolerance)
    assert power(0.1, 1.0, 800, 800) < 0.6


def test_an_uneven_split_needs_more_users():
    # The SE grows with 1/n_C + 1/n_T: at 10/90 that's 11.1/N against 4/N at 50/50.
    even = sum(sample_size(0.1, 1.0, split=0.5))
    uneven = sum(sample_size(0.1, 1.0, split=0.1))

    assert uneven / even == pytest.approx((1 / 0.1 + 1 / 0.9) / 4, rel=0.01)


# --- MDE ---------------------------------------------------------------------------

def test_mde_is_the_effect_that_reaches_the_target_power():
    effect = mde(18.76, 205_036, 205_037)

    assert power(effect, 18.76, 205_036, 205_037) == pytest.approx(0.8, rel=1e-4)


def test_mde_is_about_2_8_standard_errors():
    # OpenIntro p. 282: at 80% power and α = 0.05 the alternative sits 0.84 + 1.96 = 2.8 SE from 0.
    se = 2.0 * math.sqrt(1 / 1000 + 1 / 1000)

    assert mde(2.0, 1000, 1000) / se == pytest.approx(z_star(0.05) + NORMAL.inv_cdf(0.8), rel=1e-6)


def test_mde_at_90_percent_power_and_alpha_001():
    # OpenIntro p. 283: 1.28 + 2.58 = 3.86 SE.
    se = 2.0 * math.sqrt(1 / 1000 + 1 / 1000)

    result = mde(2.0, 1000, 1000, target_power=0.9, alpha=0.01) / se

    assert result == pytest.approx(z_star(0.01) + NORMAL.inv_cdf(0.9), rel=1e-6)
    assert result == pytest.approx(3.86, abs=0.01)


def test_mde_with_an_uneven_split():
    uneven = mde(1.0, 9_000, 1_000)

    assert power(uneven, 1.0, 9_000, 1_000) == pytest.approx(0.8, rel=1e-4)
    assert uneven / mde(1.0, 5_000, 5_000) == pytest.approx(math.sqrt((1 / 9_000 + 1 / 1_000) / (2 / 5_000)), rel=1e-9)


def test_four_times_the_users_halves_the_mde():
    assert mde(1.0, 4000, 4000) == pytest.approx(mde(1.0, 1000, 1000) / 2)


# --- design analysis: Type S and Type M --------------------------------------------

def test_design_returns_a_frozen_design_result():
    result = design(2.8, 1.0)

    assert isinstance(result, DesignResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.power = 0.5


def test_design_matches_gelman_and_carlin_at_80_percent_power():
    # Gelman & Carlin (2014): an effect 2.8 SE from zero gives power 0.80, a Type S rate of
    # 1.2e-6 and an exaggeration ratio of 1.12.
    result = design(2.8, 1.0)

    assert result.power == pytest.approx(0.80, abs=0.002)
    assert result.type_s == pytest.approx(1.2e-6, rel=0.05)
    assert result.exaggeration == pytest.approx(1.12, abs=0.01)


@pytest.mark.parametrize("effect_in_se, expected_power", [(2.0, 0.50), (1.0, 0.17), (0.65, 0.10)])
def test_design_matches_gelman_and_carlins_reference_points(effect_in_se, expected_power):
    # Gelman & Carlin (2014) give these reference points: about 50% power at 2 standard errors,
    # 17% at 1 and 10% at 0.65.
    assert design(effect_in_se, 1.0).power == pytest.approx(expected_power, abs=0.02)


@pytest.mark.parametrize("effect, se, alpha", [(0.15, 0.5, 0.05), (0.5, 0.5, 0.05), (1.0, 0.5, 0.05), (0.75, 0.5, 0.01)],
                         ids=["power-6pc", "power-17pc", "power-52pc", "alpha-0.01"])
def test_design_matches_simulation(effect, se, alpha):
    # Simulation is the oracle: a million estimates from N(effect, SE²), keep the significant ones.
    # At very low power many significant estimates have the wrong sign, so every branch matters.
    estimates = np.random.default_rng(3).normal(effect, se, 1_000_000)
    significant = estimates[np.abs(estimates / se) > z_star(alpha)]

    result = design(effect, se, alpha=alpha)

    # Each tolerance is 4 Monte Carlo standard errors of the simulated quantity (chapter 0).
    n, k = 1_000_000, len(significant)
    sim_power, sim_type_s = k / n, np.mean(significant < 0)
    sim_exaggeration = np.mean(np.abs(significant)) / effect
    assert result.power == pytest.approx(sim_power, abs=4 * math.sqrt(sim_power * (1 - sim_power) / n))
    assert result.type_s == pytest.approx(sim_type_s, abs=4 * math.sqrt(sim_type_s * (1 - sim_type_s) / k))
    assert result.exaggeration == pytest.approx(sim_exaggeration, abs=4 * np.std(np.abs(significant)) / math.sqrt(k) / effect)


def test_exaggeration_shrinks_as_power_grows():
    ratios = [design(effect, 1.0).exaggeration for effect in (0.5, 1.0, 2.0, 3.0)]

    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < 1.1


# --- input checks -------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    pytest.param(lambda: power(0.1, 0.0, 100, 100), id="power-zero-sd"),
    pytest.param(lambda: power(0.1, 1.0, 100, 100, sd_treatment=0.0), id="power-zero-treated-sd"),
    pytest.param(lambda: power(0.1, 1.0, 0, 100), id="power-empty-group"),
    pytest.param(lambda: power(0.1, 1.0, 100, 100, alpha=0.0), id="power-alpha-zero"),
    pytest.param(lambda: power(0.1, 1.0, 100, 100, alpha=1.5), id="power-alpha-above-one"),
    pytest.param(lambda: sample_size(0.0, 1.0), id="sample-size-zero-effect"),
    pytest.param(lambda: sample_size(0.1, 1.0, split=1.0), id="sample-size-bad-split"),
    pytest.param(lambda: sample_size(0.1, 1.0, target_power=1.0), id="sample-size-bad-power"),
    pytest.param(lambda: sample_size(0.1, 1.0, alpha=1.5), id="sample-size-bad-alpha"),
    pytest.param(lambda: mde(-1.0, 100, 100), id="mde-negative-sd"),
    pytest.param(lambda: mde(1.0, 100, 100, alpha=1.5), id="mde-bad-alpha"),
    pytest.param(lambda: design(0.0, 1.0), id="design-zero-effect"),
    pytest.param(lambda: design(0.1, 0.0), id="design-zero-se"),
    pytest.param(lambda: design(0.1, 1.0, alpha=1.5), id="design-bad-alpha"),
])
def test_rejects_impossible_inputs(call):
    with pytest.raises(ValueError):
        call()
