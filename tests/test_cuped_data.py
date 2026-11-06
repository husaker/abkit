"""Chapter 6 on real data: CUPED on January 2020 users with October to December 2019 as the pre-period.

Skipped when the processed files don't exist. Takes about a minute.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit.cuped import UserMetric, cuped_test, theta
from abkit.harness import run_aa
from abkit.regression import ols
from abkit.ttest import welch_t_test

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)

N_SPLITS = 400


@pytest.fixture(scope="module")
def january():
    """January 2020 users with their pre-period covariates, one row per user, sorted by user id.

    Columns: revenue, sessions, converted (January); revenue_dec, revenue_novdec, revenue_pre,
    sessions_pre, converted_pre (0/1) and active_pre (0/1, any session in October to December).
    A user absent from a pre-period month has 0 for that month. 410,073 users, as in chapter 0.
    """
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["user_id", "month", "n_sessions", "revenue", "converted"])
    users = months[months["month"] == "2020-01"].set_index("user_id").sort_index()
    table = pd.DataFrame(index=users.index)
    table["revenue"] = users["revenue"]
    table["sessions"] = users["n_sessions"].astype(float)
    table["converted"] = users["converted"].astype(float)
    for month, tag in [("2019-10", "oct"), ("2019-11", "nov"), ("2019-12", "dec")]:
        earlier = months[months["month"] == month].set_index("user_id").reindex(table.index)
        table[f"revenue_{tag}"] = earlier["revenue"].fillna(0.0)
        table[f"sessions_{tag}"] = earlier["n_sessions"].fillna(0).astype(float)
        table[f"converted_{tag}"] = earlier["converted"].fillna(False).astype(float)
        table[f"active_{tag}"] = earlier["n_sessions"].notna().astype(float)
    table["revenue_novdec"] = table["revenue_nov"] + table["revenue_dec"]
    table["revenue_pre"] = table["revenue_oct"] + table["revenue_novdec"]
    table["sessions_pre"] = table["sessions_oct"] + table["sessions_nov"] + table["sessions_dec"]
    table["converted_pre"] = (table["converted_oct"] + table["converted_nov"] + table["converted_dec"]).gt(0).astype(float)
    table["active_pre"] = (table["active_oct"] + table["active_nov"] + table["active_dec"]).gt(0).astype(float)
    return table


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_one_january_user_in_five_has_pre_period_data(january):
    assert len(january) == 410_073
    assert january["active_pre"].mean() == pytest.approx(0.198, abs=0.005)


def test_pre_period_revenue_removes_a_tenth_of_the_variance_of_january_revenue(january):
    metric = UserMetric(january["revenue"].to_numpy(), january["revenue_pre"].to_numpy())
    assert metric.correlation == pytest.approx(0.331, abs=0.01)
    adjusted = metric.adjust()
    assert adjusted.var(ddof=1) / metric.values.var(ddof=1) == pytest.approx(0.89, abs=0.01)
    assert adjusted.mean() == pytest.approx(metric.values.mean())


def test_pre_period_sessions_remove_half_the_variance_of_sessions_per_user(january):
    metric = UserMetric(january["sessions"].to_numpy(), january["sessions_pre"].to_numpy())
    assert metric.correlation == pytest.approx(0.714, abs=0.01)
    assert metric.adjust().var(ddof=1) / metric.values.var(ddof=1) == pytest.approx(0.49, abs=0.01)


def test_a_longer_pre_period_gives_a_higher_correlation_and_more_coverage(january):
    revenue = january["revenue"].to_numpy()
    rho = [correlation(revenue, january[c].to_numpy()) for c in ["revenue_dec", "revenue_novdec", "revenue_pre"]]
    assert rho[0] < rho[1] < rho[2]
    assert rho[0] == pytest.approx(0.21, abs=0.01) and rho[2] == pytest.approx(0.33, abs=0.01)
    coverage = [january[c].mean() for c in ["active_dec", "active_pre"]]
    assert coverage[0] < coverage[1]


@pytest.fixture(scope="module")
def aa_splits(january):
    """N_SPLITS random halvings of the January users: raw and CUPED p-values and SE ratios for revenue and sessions."""
    rng = np.random.default_rng(2020)
    n_users, half = len(january), len(january) // 2
    out = {k: np.empty(N_SPLITS) for k in ["revenue_raw", "revenue_cuped", "revenue_ratio", "sessions_raw", "sessions_cuped", "sessions_ratio"]}
    columns = {"revenue": ("revenue", "revenue_pre"), "sessions": ("sessions", "sessions_pre")}
    arrays = {name: (january[y].to_numpy(), january[x].to_numpy()) for name, (y, x) in columns.items()}
    for i in range(N_SPLITS):
        treated = np.zeros(n_users, dtype=bool)
        treated[rng.permutation(n_users)[half:]] = True
        for name, (y, x) in arrays.items():
            raw = welch_t_test(y[~treated], y[treated])
            adjusted = cuped_test(UserMetric(y[~treated], x[~treated]), UserMetric(y[treated], x[treated]))
            out[f"{name}_raw"][i], out[f"{name}_cuped"][i] = raw.p_value, adjusted.p_value
            out[f"{name}_ratio"][i] = adjusted.standard_error / raw.standard_error
    return out


def test_cuped_keeps_the_false_alarm_rate_at_five_percent(aa_splits):
    # 400 splits measure a 5% rate to about +-1.1 points; 3% to 8% is the agreed band
    assert 0.03 <= np.mean(aa_splits["revenue_cuped"] < 0.05) <= 0.08
    assert 0.03 <= np.mean(aa_splits["sessions_cuped"] < 0.05) <= 0.08


def test_the_standard_error_shrinks_by_the_square_root_of_one_minus_rho_squared(aa_splits):
    assert aa_splits["revenue_ratio"].mean() == pytest.approx(math.sqrt(0.8905), abs=0.005)
    assert aa_splits["sessions_ratio"].mean() == pytest.approx(math.sqrt(0.491), abs=0.005)


def test_the_harness_runs_welch_on_adjusted_values_without_a_wrapper(january):
    adjusted = UserMetric(january["sessions"].to_numpy(), january["sessions_pre"].to_numpy()).adjust()
    result = run_aa(adjusted, welch_t_test, n_simulations=200, seed=9)
    assert 0.02 <= result.rejection_rate <= 0.09


@pytest.fixture(scope="module")
def one_split(january):
    rng = np.random.default_rng(2020)
    treated = np.zeros(len(january), dtype=bool)
    treated[rng.permutation(len(january))[len(january) // 2:]] = True
    return treated


def test_cuped_is_the_regression_coefficient_on_the_treatment_dummy(january, one_split):
    y, x = january["revenue"].to_numpy(), january["revenue_pre"].to_numpy()
    treated = one_split
    result = cuped_test(UserMetric(y[~treated], x[~treated]), UserMetric(y[treated], x[treated]))
    design = np.column_stack([np.ones(len(y)), treated.astype(float), x])
    fit = ols(y, design)
    assert fit.coef[1] == pytest.approx(result.estimate, abs=1e-5)
    assert fit.robust_se[1] == pytest.approx(result.standard_error, rel=1e-3)
    assert fit.coef[2] == pytest.approx(theta(y, x), rel=1e-3)


def test_a_binary_covariate_is_post_stratification(january, one_split):
    """Deng et al. 2013, appendix A, on a real split: for each group, the adjusted mean with the group's
    own theta, centred on the pooled share of returning users, is the post-stratified mean."""
    y, active = january["revenue"].to_numpy(), january["active_pre"].to_numpy()
    weight = active.mean()
    for group in [~one_split, one_split]:
        metric = UserMetric(y[group], active[group])
        stratified = weight * y[group & (active == 1)].mean() + (1 - weight) * y[group & (active == 0)].mean()
        assert metric.adjust(covariate_mean=weight).mean() == pytest.approx(stratified, rel=1e-10)
    # with the one pooled theta that cuped_test uses, the two-group delta matches the stratified delta to 1e-6
    result = cuped_test(UserMetric(y[~one_split], active[~one_split]), UserMetric(y[one_split], active[one_split]))
    returning = y[one_split & (active == 1)].mean() - y[~one_split & (active == 1)].mean()
    new = y[one_split & (active == 0)].mean() - y[~one_split & (active == 0)].mean()
    assert result.estimate == pytest.approx(weight * returning + (1 - weight) * new, abs=1e-6)


def test_a_covariate_moved_by_the_treatment_biases_the_estimate(january):
    """Deng et al. 2013, section 5.3: +10% revenue and +10% sessions in treatment; January sessions as the covariate."""
    y, sessions, pre_sessions = (january[c].to_numpy() for c in ["revenue", "sessions", "sessions_pre"])
    rng = np.random.default_rng(11)
    n_users, half, n_sims = len(y), len(y) // 2, 100
    truth = 0.10 * y.mean()
    valid, invalid = np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        treated = np.zeros(n_users, dtype=bool)
        treated[rng.permutation(n_users)[half:]] = True
        control_y, treatment_y = y[~treated], y[treated] * 1.10
        valid[i] = cuped_test(UserMetric(control_y, pre_sessions[~treated]), UserMetric(treatment_y, pre_sessions[treated])).estimate
        invalid[i] = cuped_test(UserMetric(control_y, sessions[~treated]), UserMetric(treatment_y, sessions[treated] * 1.10)).estimate
    assert valid.mean() == pytest.approx(truth, abs=0.015)
    assert invalid.mean() < truth - 0.03  # about a fifth of the effect is "adjusted away"
