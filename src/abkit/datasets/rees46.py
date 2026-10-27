import pandas as pd

def read_events(paths: list) -> pd.DataFrame:
    """Read REES46 event CSV files into one table.

    Parameters
    ----------
    paths : list
        Paths to the CSV files; the dataset has one per month, e.g. 2019-Dec.csv.

    Returns
    -------
    pd.DataFrame
        All rows from all files with a fresh index, event_time parsed from text such
        as "2019-12-01 00:00:00 UTC" to a UTC datetime, and a source_file column
        holding each row's file name without the extension.
    """

    raw_data = []

    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = str(path).rsplit('/', 1)[-1].rsplit('.', 1)[0]  # adding source file name
        raw_data.append(df)

    df = pd.concat(raw_data, ignore_index=True)  # merging all data files
    df["event_time"] = pd.to_datetime(df["event_time"])  # changing str column to dt type

    return df

def build_sessions(events:pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw events into one row per user session.

    Parameters
    ----------
    events : pd.DataFrame
        Raw events from read_events.

    Returns
    -------
    pd.DataFrame
        One row per (user_id, user_session) with session_start, session_end,
        n_events, n_views, n_carts, n_purchases and revenue (sum of purchase prices,
        0 for a session without purchases). Events without a user_session are left
        out.
    """

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


def build_user_months(events:pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw events into one row per user and calendar month.

    Parameters
    ----------
    events : pd.DataFrame
        Raw events from read_events.

    Returns
    -------
    pd.DataFrame
        One row per (user_id, month), with month as "YYYY-MM" in UTC, and columns
        n_sessions (distinct non-missing user_session values), n_events, n_views,
        n_carts, n_purchases, revenue (sum of purchase prices) and converted (True
        if the user made a purchase that month).
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