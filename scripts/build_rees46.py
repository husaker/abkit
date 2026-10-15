import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # scripts → abkit → L4_ds_path
RAW = ROOT / "data" / "rees46" / "raw"
PROCESSED = ROOT / "data" / "rees46" / "processed"

files = sorted(RAW.glob('*.csv'))   # all 5 monthly files

def read_events(paths):
    """
    Function takes a list of paths of data files in csv and merges them into one dataframe
    and adds additional column with the name of source file
    Event_time column str in format of 2019-12-01 00:00:00 UTC
    """

    raw_data = []

    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = str(path).rsplit('/', 1)[-1].rsplit('.', 1)[0]  # adding source file name
        raw_data.append(df)

    df = pd.concat(raw_data, ignore_index=True)  # merging all data files
    df["event_time"] = pd.to_datetime(df["event_time"])  # changing str column to dt type

    return df

def build_sessions(events):
    """
    function takes a df with raw events by user and aggregates them into user x session dataframe
    """
    # data have 272 sessions ID shared by two user_id, to deal with that problem and not
    # to lose data we will considel a session a pair user_session + user_id
    events["user_session"] = events["user_session"] + events["user_id"].str()

    sessions = (
        events
        .assign(
            is_view=events["event_type"].eq("view"),
            is_cart=events["event_type"].eq("cart"),
            is_purchase=events["event_type"].eq("purchase"),
            purchase_price=events["price"].where(events["event_type"].eq("purchase"))
        )
        .groupby(by=["user_id", "user_session"], as_index=False)
        .agg(
            session_start=("event_time", "min"),
            session_end=("event_time", "max"),
            n_events=("event_type", "count"),
            n_views=("is_view", "sum"),
            n_carts=("is_cart", "sum"),
            n_purchases=("is_purchase", "sum"),
            revenue=("purchase_price", "sum")
        )
    )

    return sessions


def build_user_months(events):
    """
    function takes events table and builds aggreagate by user and month
    """

    user_agg = (
        events.assign(
            month=events["event_time"].dt.strftime("%Y-%m"),
            is_view=events["event_type"].eq("view"),
            is_cart=events["event_type"].eq("cart"),
            is_purchase=events["event_type"].eq("purchase"),
            purchase_price=events["price"].where(events["event_type"].eq("purchase"))
        )
        .groupby(by=["user_id", "month"], as_index=False)
        .agg(
            n_sessions=("user_session", "nunique"),
            n_events=("event_type", "count"),
            n_views=("is_view", "sum"),
            n_carts=("is_cart", "sum"),
            n_purchases=("is_purchase", "sum"),
            revenue=("purchase_price", "sum"),
            converted=("is_purchase", "any")

        )
    )

    return user_agg


events = read_events(files)
sessions = build_sessions(events)
user_months = build_user_months(events)

PROCESSED.mkdir(parents=True, exist_ok=True)

events.to_parquet(path=PROCESSED / "events.parquet")
sessions.to_parquet(path=PROCESSED / "sessions.parquet")
user_months.to_parquet(path=PROCESSED / "user_months.parquet")