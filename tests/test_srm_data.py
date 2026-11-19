"""Chapter 8 data checks: the sample ratio check on your January users, with planted bugs.

Skipped until data/rees46/processed/user_months.parquet exists.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit.errors import SampleRatioMismatchError
from abkit.harness import split_users
from abkit.srm import check_srm, scorecard, srm_test
from abkit.ztest import z_test_proportions

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"
needs_data = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)


@pytest.fixture(scope="module")
def january():
    months = pd.read_parquet(PROCESSED / "user_months.parquet")
    users = months[months["month"] == "2020-01"].sort_values("user_id").reset_index(drop=True)
    seen_before = set(months.loc[months["month"] < "2020-01", "user_id"])
    users["returning"] = users["user_id"].isin(seen_before)
    return users


@needs_data
def test_january_has_410073_users(january):
    assert len(january) == 410_073


@needs_data
def test_split_users_never_shows_a_mismatch(january):
    treated = split_users(len(january), np.random.default_rng(0))
    result = check_srm([int((~treated).sum()), int(treated.sum())])
    assert result.statistic < 1e-5


@needs_data
def test_a_fair_coin_per_user_wanders_by_chance(january):
    treated = np.random.default_rng(8).random(len(january)) < 0.5
    result = check_srm([int((~treated).sum()), int(treated.sum())])
    assert result.counts == (205_173, 204_900)
    assert result.p_value == pytest.approx(0.670, abs=0.001)


@needs_data
def test_a_lossy_logger_is_caught_before_any_metric_is_read(january):
    rng = np.random.default_rng(11)
    treated = rng.random(len(january)) < 0.5
    single = (january["n_sessions"] == 1).to_numpy()
    lost = treated & single & (rng.random(len(january)) < 0.05)
    kept = january[~lost]
    converted = kept["converted"].to_numpy().astype(float)
    with pytest.raises(SampleRatioMismatchError):
        scorecard(treated[~lost], {"converted": (converted, z_test_proportions)})


@needs_data
def test_an_eager_bot_filter_slips_past_the_final_count_but_not_the_removed_count(january):
    rng = np.random.default_rng(12)
    treated = rng.random(len(january)) < 0.5
    views = january["n_views"].to_numpy() * np.where(treated, 1.25, 1.0)
    bot = views > 64
    kept_counts = [int((~bot & ~treated).sum()), int((~bot & treated).sum())]
    removed_counts = [int((bot & ~treated).sum()), int((bot & treated).sum())]
    assert check_srm(kept_counts).p_value > 0.001
    with pytest.raises(SampleRatioMismatchError):
        check_srm(removed_counts)


@needs_data
def test_a_bug_among_returning_users_shows_in_their_segment_only(january):
    rng = np.random.default_rng(55)
    treated = rng.random(len(january)) < 0.5
    returning = january["returning"].to_numpy()
    kept = ~(treated & returning & (rng.random(len(january)) < 0.15))

    def counts(mask):
        return [int((mask & kept & ~treated).sum()), int((mask & kept & treated).sum())]

    assert srm_test(counts(np.ones(len(january), dtype=bool))).p_value < 1e-10
    assert srm_test(counts(~returning)).p_value > 0.01
    assert srm_test(counts(returning)).p_value < 1e-10
