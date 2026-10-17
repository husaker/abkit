"""Chapter 0 on real data: an A/A check of January 2020 conversion.

Skipped when the processed files don't exist. Takes a few seconds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit.harness import run_aa

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)


def welch(control, treatment):
    return stats.ttest_ind(control, treatment, equal_var=False).pvalue


@pytest.fixture(scope="module")
def january_conversion():
    """One value per user active in January 2020: 1 if they bought anything, else 0."""
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["month", "converted"])
    return months.loc[months["month"].eq("2020-01"), "converted"].to_numpy(dtype=float)


def test_real_aa_on_january_conversion_holds_alpha(january_conversion):
    n_simulations, alpha = 300, 0.05
    result = run_aa(january_conversion, welch, n_simulations=n_simulations, alpha=alpha, seed=2020)

    se = np.sqrt(alpha * (1 - alpha) / n_simulations)
    assert alpha - 3 * se <= result.rejection_rate <= alpha + 3 * se
