import pandas as pd

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