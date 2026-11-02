"""Chapter 4 on real data: Mann-Whitney against Welch on January 2020 revenue.

Skipped when the processed files don't exist. Takes about half a minute.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit.mannwhitney import mann_whitney

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)


@pytest.fixture(scope="module")
def january_revenue():
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["month", "revenue"])
    return months.loc[months["month"].eq("2020-01"), "revenue"].to_numpy(dtype=float)


def test_small_uneven_aa_tests_hold_alpha_where_welch_does_not(january_revenue):
    # Chapter 2: Welch on 1,000 users split 10/90 raised false alarms about 13% of the time.
    rng = np.random.default_rng(7)
    n_simulations = 1000
    rejected_mw = rejected_welch = 0
    for _ in range(n_simulations):
        users = january_revenue[rng.integers(0, january_revenue.size, 1000)]
        order = rng.permutation(1000)
        treatment, control = users[order[:100]], users[order[100:]]
        rejected_mw += mann_whitney(control, treatment).p_value < 0.05
        rejected_welch += stats.ttest_ind(treatment, control, equal_var=False).pvalue < 0.05

    se = math.sqrt(0.05 * 0.95 / n_simulations)
    assert 0.05 - 3 * se <= rejected_mw / n_simulations <= 0.05 + 3 * se
    assert rejected_welch / n_simulations > 0.09


def test_spending_more_per_buyer_is_almost_invisible_to_mann_whitney(january_revenue):
    # Every buyer spends 5% more: revenue per user is up 5%, yet 93% of users are tied at 0.
    rng = np.random.default_rng(8)
    n_simulations = 100
    half = january_revenue.size // 2
    rejected_mw = rejected_welch = 0
    for _ in range(n_simulations):
        order = rng.permutation(january_revenue.size)
        control, treatment = january_revenue[order[:half]], january_revenue[order[half:]] * 1.05
        rejected_mw += mann_whitney(control, treatment).p_value < 0.05
        rejected_welch += stats.ttest_ind(treatment, control, equal_var=False).pvalue < 0.05

    assert rejected_mw / n_simulations < 0.15
    assert rejected_welch / n_simulations > 0.55
