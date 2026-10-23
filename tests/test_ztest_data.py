"""Chapter 1 on real data: the z-test on January 2020 conversion.

Skipped when the processed files don't exist. Takes a few seconds.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.stats.proportion as smp

from abkit.harness import run_aa, split_users
from abkit.ztest import z_test_proportions

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)


@pytest.fixture(scope="module")
def january_conversion():
    """One value per user active in January 2020: 1 if they bought anything, else 0."""
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["month", "converted"])
    return months.loc[months["month"].eq("2020-01"), "converted"].to_numpy(dtype=float)


def test_real_aa_with_the_z_test_holds_alpha(january_conversion):
    n_simulations, alpha = 300, 0.05

    def test(control, treatment):
        return z_test_proportions(control, treatment).p_value

    result = run_aa(january_conversion, test, n_simulations=n_simulations, alpha=alpha, seed=2021)

    se = math.sqrt(alpha * (1 - alpha) / n_simulations)
    assert alpha - 3 * se <= result.rejection_rate <= alpha + 3 * se


def test_real_split_matches_statsmodels(january_conversion):
    treated = split_users(len(january_conversion), np.random.default_rng(2020))
    control, treatment = january_conversion[~treated], january_conversion[treated]
    k_c, k_t = int(control.sum()), int(treatment.sum())

    unpooled = z_test_proportions(control, treatment)
    pooled = z_test_proportions(control, treatment, pooled=True)

    oracle = smp.test_proportions_2indep(k_t, len(treatment), k_c, len(control), method="wald")
    z, p = smp.proportions_ztest([k_t, k_c], [len(treatment), len(control)])
    assert unpooled.p_value == pytest.approx(oracle.pvalue, rel=1e-9)
    assert pooled.p_value == pytest.approx(p, rel=1e-9)
