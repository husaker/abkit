from abkit.datasets.rees46 import read_events, build_sessions, build_user_months
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # scripts → abkit → L4_ds_path
RAW = ROOT / "data" / "rees46" / "raw"
PROCESSED = ROOT / "data" / "rees46" / "processed"


def main() -> None:
    events = read_events(sorted(RAW.glob('*.csv')))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_parquet(path=PROCESSED / "events.parquet")
    build_sessions(events).to_parquet(path=PROCESSED / "sessions.parquet")
    build_user_months(events).to_parquet(path=PROCESSED / "user_months.parquet")


if __name__ == "__main__":
    main()