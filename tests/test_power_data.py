"""Chapter 3 on real data: the power formula against your harness, on January 2020 users.

Skipped when the processed files don't exist. Takes about 10 seconds.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit.harness import add_conversions, multiply, run_ab
from abkit.power import mde, power
from abkit.ttest import welch_t_test
from abkit.ztest import z_test_proportions

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)


@pytest.fixture(scope="module")
def january():
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["month", "converted", "revenue"])
    return months.loc[months["month"].eq("2020-01")]


def groups(n):
    """The two group sizes split_users makes from n users at 50/50."""
    return n - n // 2, n // 2


def within_4_mc_se(simulated, formula, n_simulations):
    return abs(simulated - formula) <= 4 * math.sqrt(formula * (1 - formula) / n_simulations)


def test_simulated_conversion_power_matches_the_formula(january):
    conversions = january["converted"].to_numpy(dtype=float)
    rate, lift, n_simulations = conversions.mean(), 0.03, 500

    def z_p_value(control, treatment):
        return z_test_proportions(control, treatment).p_value

    simulated = run_ab(conversions, z_p_value, inject=add_conversions, lift=lift, n_simulations=n_simulations, seed=11)

    # Equal spreads are a good model here: at 10,000 simulations the harness measured 74.5% ± 0.4
    # against the formula's 74.3%.
    formula = power(rate * lift, math.sqrt(rate * (1 - rate)), *groups(len(conversions)))
    assert within_4_mc_se(simulated.rejection_rate, formula, n_simulations)


def test_simulated_revenue_power_matches_the_formula(january):
    revenue = january["revenue"].to_numpy(dtype=float)
    lift, n_simulations = 0.05, 500

    def welch_p_value(control, treatment):
        return welch_t_test(control, treatment).p_value

    simulated = run_ab(revenue, welch_p_value, inject=multiply, lift=lift, n_simulations=n_simulations, seed=12)

    # multiply() also stretches the treated group's spread by (1 + lift), and Welch's SE sees it.
    # The equal-spread formula says 78.5%; with the treated SD it's 76.5%, and 10,000 harness
    # simulations measured 75.9% ± 0.4.
    sd = revenue.std(ddof=1)
    formula = power(revenue.mean() * lift, sd, *groups(len(revenue)), sd_treatment=(1 + lift) * sd)
    assert within_4_mc_se(simulated.rejection_rate, formula, n_simulations)


def test_januarys_relative_mdes(january):
    # With all 410,073 January users split 50/50, the smallest lift you'd catch 80% of the time.
    conversions = january["converted"].to_numpy(dtype=float)
    revenue = january["revenue"].to_numpy(dtype=float)
    n_c, n_t = groups(len(january))
    rate = conversions.mean()

    conversion_mde = mde(math.sqrt(rate * (1 - rate)), n_c, n_t) / rate
    revenue_mde = mde(revenue.std(ddof=1), n_c, n_t) / revenue.mean()

    assert conversion_mde == pytest.approx(0.032, abs=0.001)
    assert revenue_mde == pytest.approx(0.051, abs=0.001)
