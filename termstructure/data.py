"""Download and cache US Treasury data from FRED (no API key needed)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# H.15 constant-maturity Treasury par yields, keyed by FRED id -> maturity in years.
CMT_SERIES = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}
SHORT_RATE_SERIES = "DTB3"  # 3-month T-bill, secondary market

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CMT_FILE = "treasury_cmt.csv"
SHORT_RATE_FILE = "dtb3.csv"


def fetch_fred(series_ids, start="1990-01-01") -> pd.DataFrame:
    """Daily FRED series as decimals (percent / 100), indexed by date."""
    frames = []
    for sid in series_ids:
        df = pd.read_csv(FRED_CSV.format(sid), na_values=".")
        date_col = df.columns[0]  # "observation_date" (older exports used "DATE")
        s = df.set_index(pd.to_datetime(df[date_col]))[sid].astype(float) / 100.0
        frames.append(s)
    out = pd.concat(frames, axis=1)
    out.index.name = "date"
    return out.loc[start:]


def refresh(start="1990-01-01", data_dir: Path = DATA_DIR) -> None:
    """Re-download the CMT curve and short rate into the data/ cache."""
    data_dir.mkdir(parents=True, exist_ok=True)
    fetch_fred(CMT_SERIES, start).to_csv(data_dir / CMT_FILE)
    fetch_fred([SHORT_RATE_SERIES], start).dropna().to_csv(data_dir / SHORT_RATE_FILE)


def load_cmt(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Cached CMT par yields; columns renamed to maturities in years."""
    df = pd.read_csv(data_dir / CMT_FILE, index_col="date", parse_dates=True)
    return df.rename(columns=CMT_SERIES)


def load_short_rate(data_dir: Path = DATA_DIR) -> pd.Series:
    df = pd.read_csv(data_dir / SHORT_RATE_FILE, index_col="date", parse_dates=True)
    return df[SHORT_RATE_SERIES]


def curve_on(date, cmt: pd.DataFrame | None = None) -> pd.Series:
    """Par curve (maturity -> yield) on a date, dropping maturities not quoted that day."""
    cmt = load_cmt() if cmt is None else cmt
    return cmt.loc[pd.Timestamp(date)].dropna()
