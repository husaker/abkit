"""Chapter 2 on real data: January 2020 revenue per user, a heavy-tailed metric.

Skipped when the processed files don't exist. Takes about half a minute.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit.capping import cap_upper
from abkit.harness import run_aa
from abkit.ttest import welch_t_test

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)


@pytest.fixture(scope="module")
def january_revenue():
    """One value per user active in January 2020: what they spent, 0 for non-buyers."""
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["month", "revenue"])
    return months.loc[months["month"].eq("2020-01"), "revenue"].to_numpy(dtype=float)


def welch(control, treatment):
    return welch_t_test(control, treatment).p_value


def capped_welch(control, treatment):
    capped_c, capped_t, _ = cap_upper(control, treatment, quantile=0.99)
    return welch_t_test(capped_c, capped_t).p_value


def test_kohavis_rule_of_thumb_on_january_revenue(january_revenue):
    # Kohavi p. 187: each variant needs about 355·s² users, s = skewness. Capping cuts it tenfold.
    raw = 355 * stats.skew(january_revenue) ** 2
    capped_c, capped_t, _ = cap_upper(january_revenue, january_revenue[:0], quantile=0.99)
    capped = 355 * stats.skew(capped_c) ** 2

    assert raw > 80_000
    assert capped < 12_000


def test_all_january_users_split_in_half_hold_alpha(january_revenue):
    n_simulations = 300

    result = run_aa(january_revenue, welch, n_simulations=n_simulations, seed=2022)

    se = math.sqrt(0.05 * 0.95 / n_simulations)
    assert 0.05 - 3 * se <= result.rejection_rate <= 0.05 + 3 * se


@pytest.fixture(scope="module")
def thousand_users(january_revenue):
    return january_revenue[np.random.default_rng(0).choice(len(january_revenue), 1000, replace=False)]


def test_heavy_tails_and_a_10_90_split_break_the_test(thousand_users):
    result = run_aa(thousand_users, welch, n_simulations=2000, seed=1, split=0.1)

    assert result.rejection_rate > 0.08


def test_capping_reduces_the_damage(thousand_users):
    raw = run_aa(thousand_users, welch, n_simulations=2000, seed=1, split=0.1)
    capped = run_aa(thousand_users, capped_welch, n_simulations=2000, seed=1, split=0.1)

    assert capped.rejection_rate < raw.rejection_rate


def test_one_big_spender_makes_the_test_too_cautious(january_revenue):
    # This sample of 1,000 users contains one person who spent 1,102.
    sample = january_revenue[np.random.default_rng(4).choice(len(january_revenue), 1000, replace=False)]

    result = run_aa(sample, welch, n_simulations=1000, seed=1)

    assert result.rejection_rate < 0.02
