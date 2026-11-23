"""Chapter 9 data checks: seven January metrics in one scorecard, corrected as a family.

Each fixture splits the January users 200 times and runs chapter 8's scorecard on seven metrics,
so the file takes about ten seconds. Skipped until data/rees46/processed/user_months.parquet exists.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit.harness import split_users
from abkit.multiple import benjamini_hochberg, bonferroni, holm
from abkit.srm import scorecard
from abkit.ttest import welch_t_test
from abkit.ztest import z_test_proportions

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"
needs_data = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)

COUNTS = ["n_sessions", "n_events", "n_views", "n_carts", "n_purchases", "revenue"]
METRICS = COUNTS + ["converted"]
VIEWS, EVENTS = METRICS.index("n_views"), METRICS.index("n_events")
UNTOUCHED = [i for i in range(len(METRICS)) if i not in (VIEWS, EVENTS)]
N_SPLITS = 200
LIFT = 0.03  # the planted effect: each treated user's views grow by 3%, and their events by the same views


@pytest.fixture(scope="module")
def january():
    months = pd.read_parquet(PROCESSED / "user_months.parquet")
    users = months[months["month"] == "2020-01"].sort_values("user_id").reset_index(drop=True)
    return {name: users[name].to_numpy(dtype=float) for name in METRICS}


def scorecard_p_values(values, treated):
    tests = {name: (z_test_proportions if name == "converted" else welch_t_test) for name in METRICS}
    card = scorecard(treated, {name: (values[name], tests[name]) for name in METRICS})
    return [float(card.results[name].p_value) for name in METRICS]


@pytest.fixture(scope="module")
def aa(january):
    """p-values of the seven metrics in 200 A/A splits: every null hypothesis is true."""
    n_users = len(january["converted"])
    return np.array([scorecard_p_values(january, split_users(n_users, np.random.default_rng(seed)))
                     for seed in range(N_SPLITS)])


@pytest.fixture(scope="module")
def planted(january):
    """p-values in 200 splits where treatment adds 3% to views: two nulls are false, five are true."""
    n_users = len(january["converted"])
    rows = []
    for seed in range(1000, 1000 + N_SPLITS):
        treated = split_users(n_users, np.random.default_rng(seed))
        extra = np.where(treated, LIFT * january["n_views"], 0.0)
        values = january | {"n_views": january["n_views"] + extra, "n_events": january["n_events"] + extra}
        rows.append(scorecard_p_values(values, treated))
    return np.array(rows)


def rejections(p_values, correction=None):
    """A splits-by-metrics table of decisions; with no correction, each p-value is compared with 0.05."""
    if correction is None:
        return p_values <= 0.05
    return np.array([correction(row).reject for row in p_values])


@needs_data
def test_holm_cuts_the_aa_false_alarms_from_37_splits_to_4(aa):
    assert rejections(aa).any(axis=1).sum() == 37
    # fewer than the 1 - 0.95 ** 7 = 30% that seven independent metrics would give: these are correlated
    assert rejections(aa).any(axis=1).mean() < 1 - 0.95 ** 7
    assert rejections(aa, holm).any(axis=1).sum() == 4
    assert rejections(aa, bonferroni).any(axis=1).sum() == 4


@needs_data
def test_benjamini_hochberg_keeps_aa_false_alarms_just_as_rare(aa):
    # when every null is true, every rejection is false, so V/R is 1 or 0 and the FDR is the FWER
    assert rejections(aa, benjamini_hochberg).any(axis=1).sum() == 4


@needs_data
def test_correction_costs_power_on_the_planted_lift(planted):
    found_views = [rejections(planted, how)[:, VIEWS].sum() for how in (None, holm, benjamini_hochberg)]
    found_events = [rejections(planted, how)[:, EVENTS].sum() for how in (None, holm, benjamini_hochberg)]
    assert found_views == [120, 74, 75]
    assert found_events == [41, 9, 15]


@needs_data
def test_correction_removes_most_false_alarms_on_the_five_untouched_metrics(planted):
    false_alarms = [rejections(planted, how)[:, UNTOUCHED].sum() for how in (None, holm, benjamini_hochberg)]
    assert false_alarms == [35, 3, 4]
