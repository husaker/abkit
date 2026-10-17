"""Data checks on the real processed REES46 tables.

They run on the files written by scripts/build_rees46.py and are skipped when
those files don't exist (for example, on a fresh clone of the repo).

They encode the data decisions in README.md:
- all 20,692,840 raw events are kept, including duplicates and negative prices;
- rows without a session stay in events and user_months, but not in sessions.
"""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).parents[2] / "data" / "rees46" / "processed"
RAW_EVENT_COUNT = 20_692_840  # the five raw CSVs together, header lines excluded

# A module-level mark applies to every test in this file.
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "user_months.parquet").exists(),
    reason="run scripts/build_rees46.py first",
)

EVENT_COLUMNS = ["event_time", "event_type", "price", "user_id", "user_session", "source_file"]


# scope="module": each table is read once and shared by all tests below.
@pytest.fixture(scope="module")
def events():
    return pd.read_parquet(PROCESSED / "events.parquet", columns=EVENT_COLUMNS)


@pytest.fixture(scope="module")
def sessions():
    return pd.read_parquet(PROCESSED / "sessions.parquet")


@pytest.fixture(scope="module")
def user_months():
    return pd.read_parquet(PROCESSED / "user_months.parquet")


@pytest.fixture(scope="module")
def purchase_revenue(events):
    return events.loc[events["event_type"].eq("purchase"), "price"].sum()


# C1: nothing is dropped on the way from the raw CSVs.
def test_every_raw_event_is_kept(events):
    assert len(events) == RAW_EVENT_COUNT


# C2: one row per (user, session) pair, so the 272 shared session IDs become separate rows.
def test_one_row_per_user_and_session(sessions):
    assert not sessions.duplicated(["user_id", "user_session"]).any()


# C3: revenue survives aggregation, in both tables.
def test_session_revenue_matches_purchases(sessions, purchase_revenue):
    assert sessions["revenue"].sum() == pytest.approx(purchase_revenue)


def test_user_month_revenue_matches_purchases(user_months, purchase_revenue):
    assert user_months["revenue"].sum() == pytest.approx(purchase_revenue)


# C4: every event falls in its file's month, in UTC (catches timezone bugs).
def test_every_event_falls_in_its_files_utc_month(events):
    t = events["event_time"]
    event_month = t.dt.year * 100 + t.dt.month  # e.g. 201910
    month_of_file = {
        f: int(pd.to_datetime(f, format="%Y-%b").strftime("%Y%m")) for f in events["source_file"].unique()
    }
    file_month = events["source_file"].map(month_of_file)

    outside = events.loc[event_month != file_month, ["event_time", "source_file"]]
    assert outside.empty, f"{len(outside)} events fall outside their file's month:\n{outside.head()}"


# C5: each user's event counts agree across the three tables.
def test_each_users_event_counts_agree_across_tables(events, sessions, user_months):
    all_events = events.groupby("user_id").size()
    events_with_session = events[events["user_session"].notna()].groupby("user_id").size()

    # user_months keeps every event, including rows without a session
    pd.testing.assert_series_equal(
        user_months.groupby("user_id")["n_events"].sum().sort_index(),
        all_events.sort_index(),
        check_names=False,
        check_dtype=False,
    )
    # sessions keeps only events that have a session
    pd.testing.assert_series_equal(
        sessions.groupby("user_id")["n_events"].sum().sort_index(),
        events_with_session.sort_index(),
        check_names=False,
        check_dtype=False,
    )
