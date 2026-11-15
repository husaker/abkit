"""Chapter 7 on real data: CUPAC on January 2020 users with features from October to December 2019.

Skipped when the processed files don't exist. Takes about a minute, most of it in the boosted models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from abkit.cupac import RidgeRegressor, cross_fit, cupac_test
from abkit.cuped import UserMetric
from abkit.harness import run_aa
from abkit.ttest import welch_t_test

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)

COLUMNS = ["n_sessions", "n_events", "n_views", "n_carts", "n_purchases", "revenue"]


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def january():
    """January 2020 users, sorted by user id, with 21 pre-period features.

    For each of October, November and December 2019: sessions, events, views, carts, purchases and
    revenue, with 0 for a user who was absent that month, and a 0/1 flag for being present. The
    dictionary holds the feature matrix ``X`` and the January metrics ``revenue``, ``sessions`` and
    ``converted``, each with one entry per user (410,073 users, as in chapter 0).
    """
    months = pd.read_parquet(PROCESSED / "user_months.parquet", columns=["user_id", "month", "converted", *COLUMNS])
    users = months[months["month"] == "2020-01"].set_index("user_id").sort_index()
    blocks = []
    for month in ["2019-10", "2019-11", "2019-12"]:
        earlier = months[months["month"] == month].set_index("user_id").reindex(users.index)
        blocks.append(earlier[COLUMNS].fillna(0.0).to_numpy(dtype=float))
        blocks.append(earlier["n_sessions"].notna().to_numpy(dtype=float)[:, None])
    return {
        "X": np.column_stack(blocks),
        "revenue": users["revenue"].to_numpy(dtype=float),
        "sessions": users["n_sessions"].to_numpy(dtype=float),
        "converted": users["converted"].to_numpy(dtype=float),
    }


@pytest.fixture(scope="module")
def ridge_prediction(january):
    """The cross-fitted ridge prediction of January revenue, one per user."""
    return cross_fit(RidgeRegressor(), january["X"], january["revenue"])


@pytest.fixture(scope="module")
def one_split(january):
    n_users = len(january["revenue"])
    rng = np.random.default_rng(2020)
    treated = np.zeros(n_users, dtype=bool)
    treated[rng.permutation(n_users)[n_users // 2:]] = True
    return treated


def test_the_feature_table_has_one_row_per_january_user(january):
    assert january["X"].shape == (410_073, 21)
    assert np.isfinite(january["X"]).all()


def test_the_ridge_prediction_beats_the_single_covariate_by_a_little(january, ridge_prediction):
    revenue = january["revenue"]
    pre_revenue = january["X"][:, [5, 12, 19]].sum(axis=1)             # chapter 6's covariate
    assert correlation(revenue, pre_revenue) == pytest.approx(0.331, abs=0.005)
    assert correlation(revenue, ridge_prediction) == pytest.approx(0.342, abs=0.005)
    assert 1 - correlation(revenue, ridge_prediction) ** 2 == pytest.approx(0.883, abs=0.005)


def test_a_small_model_on_many_users_barely_overfits(january, ridge_prediction):
    # 21 slopes fitted on 410,073 users: predicting the users you trained on changes rho in the third decimal
    revenue = january["revenue"]
    in_sample = RidgeRegressor().fit(january["X"], revenue).predict(january["X"])
    assert 0 < correlation(revenue, in_sample) - correlation(revenue, ridge_prediction) < 0.01


def test_a_deep_tree_looks_excellent_in_sample_and_poor_out_of_fold(january):
    revenue, X = january["revenue"], january["X"]
    in_sample = DecisionTreeRegressor(random_state=0).fit(X, revenue).predict(X)
    out_of_fold = cross_fit(DecisionTreeRegressor(random_state=0), X, revenue)
    assert correlation(revenue, in_sample) > 0.6                       # 0.70 on the reference run
    assert correlation(revenue, out_of_fold) < 0.3                     # 0.18: worse than one covariate


def test_boosting_helps_conversion_and_not_sessions(january):
    X = january["X"]
    boosted = {name: cross_fit(HistGradientBoostingRegressor(random_state=0), X, january[name])
               for name in ["converted", "sessions"]}
    ridge = {name: cross_fit(RidgeRegressor(), X, january[name]) for name in ["converted", "sessions"]}
    # conversion: 0.33 against 0.28, the trees find what a straight line misses
    assert correlation(january["converted"], boosted["converted"]) > correlation(january["converted"], ridge["converted"]) + 0.03
    # sessions: 0.36 against 0.71, the trees cannot follow the heaviest users
    assert correlation(january["sessions"], ridge["sessions"]) == pytest.approx(0.709, abs=0.01)
    assert correlation(january["sessions"], boosted["sessions"]) < 0.5


def test_the_standard_error_ratio_is_root_one_minus_rho_squared(january, ridge_prediction, one_split):
    revenue, treated = january["revenue"], one_split
    raw = welch_t_test(revenue[~treated], revenue[treated])
    result = cupac_test(revenue, january["X"], treated, RidgeRegressor())
    rho = correlation(revenue, ridge_prediction)
    assert result.standard_error / raw.standard_error == pytest.approx(np.sqrt(1 - rho ** 2), abs=0.003)
    assert result.standard_error / raw.standard_error == pytest.approx(0.940, abs=0.005)


def test_the_adjusted_metric_passes_the_aa_check(january, ridge_prediction):
    adjusted = UserMetric(january["revenue"], ridge_prediction).adjust()
    result = run_aa(adjusted, welch_t_test, n_simulations=400, seed=3)
    assert 0.03 <= result.rejection_rate <= 0.08


def test_cupac_recovers_an_injected_effect(january):
    revenue, X = january["revenue"], january["X"]
    truth = 0.10 * revenue.mean()                                      # 0.322
    rng = np.random.default_rng(5)
    estimates = []
    for _ in range(30):
        treated = np.zeros(len(revenue), dtype=bool)
        treated[rng.permutation(len(revenue))[len(revenue) // 2:]] = True
        values = revenue.copy()
        values[treated] *= 1.10
        estimates.append(cupac_test(values, X, treated, RidgeRegressor()).estimate)
    monte_carlo_se = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
    assert np.mean(estimates) == pytest.approx(truth, abs=3.5 * monte_carlo_se)


def test_an_in_sample_deep_tree_loses_a_large_part_of_the_effect(january, one_split):
    # The mistake cross-fitting exists to prevent: fit on everyone, predict everyone, use it as the covariate.
    revenue, X, treated = january["revenue"], january["X"], one_split
    truth = 0.10 * revenue.mean()
    values = revenue.copy()
    values[treated] *= 1.10
    in_sample = DecisionTreeRegressor(random_state=0).fit(X, values).predict(X)
    adjusted = UserMetric(values, in_sample).adjust()
    estimate = adjusted[treated].mean() - adjusted[~treated].mean()
    honest = cupac_test(values, X, treated, DecisionTreeRegressor(random_state=0)).estimate
    assert estimate < 0.8 * truth                                      # about 0.18 of 0.32 on the reference runs
    assert abs(honest - truth) < 0.2                                   # cross-fitted: noisy, but centred on the truth
