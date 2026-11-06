"""Chapter 5 on real data: ratio metrics on January 2020 sessions, with the delta method.

Skipped when the processed files don't exist. Takes about half a minute.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit.delta import RatioMetric, delta_ratio_test, relative_lift_test
from abkit.harness import run_aa
from abkit.ttest import welch_t_test
from abkit.ztest import z_test_proportions

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "sessions.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)

N_SPLITS = 400


@pytest.fixture(scope="module")
def january():
    """January 2020 sessions, and the same data summed per user.

    Returns a dict: `users` has the per-user purchase sessions and sessions (sorted by user id);
    `session_user` is each session's position in that user table; `session_bought` is each
    session's 0/1 purchase flag. 408,345 users and 963,271 sessions, as in chapter 4.
    """
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet", columns=["user_id", "session_start", "n_purchases"])
    in_january = (sessions["session_start"] >= "2020-01-01") & (sessions["session_start"] < "2020-02-01")
    sessions = sessions[in_january]
    bought = sessions["n_purchases"].gt(0).to_numpy(dtype=float)
    users = (sessions.assign(bought=bought).groupby("user_id")
             .agg(sessions=("bought", "size"), purchase_sessions=("bought", "sum")))
    return {
        "purchases": users["purchase_sessions"].to_numpy(dtype=float),
        "sessions": users["sessions"].to_numpy(dtype=float),
        "session_user": np.searchsorted(users.index.to_numpy(), sessions["user_id"].to_numpy()),
        "session_bought": bought,
    }


@pytest.fixture(scope="module")
def aa_p_values(january):
    """P-values from N_SPLITS random halvings of the January users, an A/A test, by three methods."""
    purchases, sessions = january["purchases"], january["sessions"]
    session_user, session_bought = january["session_user"], january["session_bought"]
    rng = np.random.default_rng(2020)
    n_users, half = purchases.size, purchases.size // 2
    naive, delta, linearized = (np.empty(N_SPLITS) for _ in range(3))
    for i in range(N_SPLITS):
        treated = np.zeros(n_users, dtype=bool)
        treated[rng.permutation(n_users)[half:]] = True
        control = RatioMetric(purchases[~treated], sessions[~treated])
        treatment = RatioMetric(purchases[treated], sessions[treated])

        treated_session = treated[session_user]
        naive[i] = z_test_proportions(session_bought[~treated_session], session_bought[treated_session]).p_value
        delta[i] = delta_ratio_test(control, treatment).p_value
        linearized[i] = welch_t_test(control.linearize(), treatment.linearize(control.ratio)).p_value
    return {"naive": naive, "delta": delta, "linearized": linearized}


def test_january_session_conversion_and_how_far_the_naive_se_falls_short(january):
    # Section 2.4: 32,098 purchase sessions in 963,271; the delta-method SE is 1.62 times the
    # session-level one, so the naive formula acts as if there were 2.6 times as many observations.
    metric = RatioMetric(january["purchases"], january["sessions"])

    naive_se = math.sqrt(metric.ratio * (1 - metric.ratio) / metric.denominator.sum())

    assert metric.n_users == 408_345
    assert metric.ratio == pytest.approx(0.03332, abs=5e-5)
    assert metric.standard_error == pytest.approx(0.000296, rel=0.02)
    assert metric.standard_error / naive_se == pytest.approx(1.62, abs=0.03)


def test_the_session_level_test_raises_false_alarms_in_aa_splits(aa_p_values):
    # Chapter 4 found the naive interval covering 0 about 78% of the time; the rejection rate is
    # the other side of the same coin. 400 splits measure a 22% rate to about ±2 points.
    assert np.mean(aa_p_values["naive"] < 0.05) > 0.13


def test_the_delta_method_holds_5_percent_in_aa_splits(aa_p_values):
    # 400 splits: ±3 Monte Carlo SEs around 5% is 1.7% to 8.3%.
    assert 0.017 <= np.mean(aa_p_values["delta"] < 0.05) <= 0.083


def test_welch_on_linearized_values_agrees_with_the_delta_method(aa_p_values):
    # Section 3: with k = the control ratio, Welch's test on L_i = x_i - k y_i is the delta-method
    # test up to the difference between the two groups' ratios, which in an A/A test is noise.
    gap = np.abs(aa_p_values["linearized"] - aa_p_values["delta"])

    assert np.median(gap) < 0.005
    assert gap.max() < 0.05
    assert 0.017 <= np.mean(aa_p_values["linearized"] < 0.05) <= 0.083


def test_a_linearized_metric_runs_through_the_harness(january):
    # Section 3.3: once the ratio is a per-user value, chapter 0's harness needs no changes. Here
    # k is the ratio over all users, fixed before the splits, which is fine under the null.
    metric = RatioMetric(january["purchases"], january["sessions"])
    n_simulations = 200

    result = run_aa(metric.linearize(), lambda c, t: welch_t_test(c, t).p_value,
                    n_simulations=n_simulations, seed=5)

    assert 0.01 <= result.rejection_rate <= 0.09  # ±3 MC SEs at 200 simulations


def test_relative_lift_se_on_january_revenue_exceeds_the_common_mistake_by_8_percent():
    # Section 2.6, on revenue per user: at +20% the correct SE (eq. 18.6) is about 8% larger than
    # var(Δ)/x̄_C² would give; the gap is the control mean's own noise.
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["month", "revenue"])
    revenue = months.loc[months["month"].eq("2020-01"), "revenue"].to_numpy(dtype=float)
    rng = np.random.default_rng(1)
    order = rng.permutation(revenue.size)
    control, treatment = revenue[order[: revenue.size // 2]], revenue[order[revenue.size // 2:]] * 1.2

    result = relative_lift_test(control, treatment)

    mistaken = math.sqrt(control.var(ddof=1) / control.size + treatment.var(ddof=1) / treatment.size) / control.mean()
    assert result.estimate == pytest.approx(treatment.mean() / control.mean() - 1, rel=1e-12)
    assert 1.06 <= result.standard_error / mistaken <= 1.10
