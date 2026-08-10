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
    return raw_data


if __name__ == "__main__":
    # Quick manual check while developing — not a substitute for real tests
    # (those come in a later step).
    df = fetch_price_data("USDBRL=X", "2024-01-01", "2024-12-31")
    print(df.head())
    print(df.shape)