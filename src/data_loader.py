"""
data_loader.py

Handles fetching historical price data from Yahoo Finance via yfinance.
"""

import pandas as pd
import yfinance as yf


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV price data for a given ticker.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "USDBRL=X").
        start: Start date in "YYYY-MM-DD" format.
        end: End date in "YYYY-MM-DD" format.

    Returns:
        A DataFrame with columns [Open, High, Low, Close, Volume],
        indexed by date.
    """
    raw_data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    # Newer yfinance versions return a MultiIndex on columns
    # (e.g. level 0 = "Close"/"High"/..., level 1 = ticker), even for a
    # single ticker. We flatten it here so the rest of the pipeline can
    # rely on simple column names regardless of yfinance version.
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    return raw_data


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate raw price data returned by fetch_price_data.

    Steps:
        1. Validate that the expected columns are present.
        2. Sort the index chronologically (defensive — should already be sorted).
        3. Drop duplicate dates, keeping the first occurrence.
        4. Forward-fill missing values (carries the last known price forward,
           which is a reasonable assumption for FX data with sparse gaps).
        5. Drop any rows still containing NaNs (can happen if the very first
           rows in the series are missing, since ffill has nothing to carry
           forward from).

    Args:
        df: Raw DataFrame as returned by fetch_price_data.

    Returns:
        A cleaned DataFrame, ready for indicator calculations.

    Raises:
        ValueError: If expected columns are missing from the input.
    """
    expected_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.ffill()
    df = df.dropna()

    return df


if __name__ == "__main__":
    # Quick manual check while developing — not a substitute for real tests
    # (those come in a later step).
    raw_df = fetch_price_data("USDBRL=X", "2024-01-01", "2024-12-31")
    print("Raw data:")
    print(raw_df.head())
    print(raw_df.shape)

    clean_df = clean_price_data(raw_df)
    print("\nCleaned data:")
    print(clean_df.head())
    print(clean_df.shape)
    print(f"\nNaNs remaining: {clean_df.isna().sum().sum()}")