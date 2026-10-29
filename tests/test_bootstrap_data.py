"""Chapter 4 on real data: a user-level bootstrap on January 2020 sessions and revenue.

Skipped when the processed files don't exist. Takes about half a minute.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit.bootstrap import bootstrap, bootstrap_ratio

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "sessions.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)


@pytest.fixture(scope="module")
def january_users():
    """One row per user with a session starting in January 2020: sessions, and sessions with a purchase.

    That's 408,345 users, fewer than the 410,073 of user_months, which counts every user with an
    event in January: 55 users had no session id, and some sessions began in December.
    """
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet", columns=["user_id", "session_start", "n_purchases"])
    january = sessions[(sessions["session_start"] >= "2020-01-01") & (sessions["session_start"] < "2020-02-01")]
    users = (january.assign(bought=january["n_purchases"].gt(0))
             .groupby("user_id")
             .agg(sessions=("bought", "size"), purchase_sessions=("bought", "sum")))
    return users["purchase_sessions"].to_numpy(dtype=float), users["sessions"].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def january_revenue():
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["month", "revenue"])
    return months.loc[months["month"].eq("2020-01"), "revenue"].to_numpy(dtype=float)


def test_january_has_more_sessions_than_users(january_users):
    purchases, sessions = january_users

    assert sessions.size == 408_345
    assert sessions.sum() == 963_271
    assert purchases.sum() / sessions.sum() == pytest.approx(0.0333, abs=1e-4)


def aa_coverage(purchases, sessions, n_simulations, users_per_test, seed):
    """Split random January users in half many times and count how often each interval covers 0."""
    rng = np.random.default_rng(seed)
    user_level = session_level = 0
    for simulation in range(n_simulations):
        picked = rng.permutation(sessions.size)[:users_per_test]
        half = users_per_test // 2
        c, t = picked[:half], picked[half:]

        result = bootstrap_ratio(purchases[c], sessions[c], purchases[t], sessions[t],
                                 n_resamples=400, seed=simulation)
        user_level += result.ci_low <= 0 <= result.ci_high

        variance = 0.0
        for group in (c, t):
            rate = purchases[group].sum() / sessions[group].sum()
            variance += rate * (1 - rate) / sessions[group].sum()
        session_level += abs(result.estimate) <= 1.959964 * math.sqrt(variance)
    return user_level / n_simulations, session_level / n_simulations


def test_user_level_bootstrap_covers_the_truth_in_aa_tests(january_users):
    # The chapter's done-when: the true difference in an A/A split is 0, and a correct 95% interval
    # covers it 95% of the time. Treating sessions as independent covers it far less often.
    n_simulations = 300

    user_level, session_level = aa_coverage(*january_users, n_simulations=n_simulations,
                                            users_per_test=40_000, seed=2020)

    # 300 splits measure coverage to about ±1.3 points; 400 resamples keep the interval ends steady.
    assert 0.90 <= user_level <= 0.99
    assert session_level < 0.88


def test_relative_lift_se_is_not_the_absolute_se_over_the_control_mean(january_revenue):
    # Kohavi pp. 194-195: dividing var(delta) by the control mean squared is wrong, because the control
    # mean is random too. Eq. 18.6 is the right formula, and the bootstrap should agree with it.
    rng = np.random.default_rng(31)
    users = january_revenue[rng.permutation(january_revenue.size)[:100_000]]
    control, treatment = users[:50_000], users[50_000:] * 1.5
    n_resamples = 2000

    result = bootstrap(control, treatment, n_resamples=n_resamples, relative=True, seed=1)

    mc, mt = control.mean(), treatment.mean()
    vc, vt = control.var(ddof=1) / control.size, treatment.var(ddof=1) / treatment.size
    wrong = math.sqrt(vc + vt) / mc
    eq_18_6 = math.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
    assert result.standard_error == pytest.approx(eq_18_6, rel=4 / math.sqrt(2 * (n_resamples - 1)))
    assert result.standard_error > 1.1 * wrong
