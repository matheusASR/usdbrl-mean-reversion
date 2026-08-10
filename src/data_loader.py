"""
data_loader.py

Handles fetching historical price data from Yahoo Finance via yfinance.
"""

import logging
import os

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when price data cannot be fetched or is invalid."""


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

    Raises:
        ValueError: If start is not before end.
        DataLoadError: If the download fails or returns no data
            (e.g. an invalid ticker symbol).
    """
    if start >= end:
        raise ValueError(f"start date ({start}) must be before end date ({end})")

    try:
        raw_data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        raise DataLoadError(f"Failed to download data for '{ticker}': {exc}") from exc

    # Newer yfinance versions return a MultiIndex on columns
    # (e.g. level 0 = "Close"/"High"/..., level 1 = ticker), even for a
    # single ticker. We flatten it here so the rest of the pipeline can
    # rely on simple column names regardless of yfinance version.
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    if raw_data.empty:
        raise DataLoadError(
            f"No data returned for ticker '{ticker}' between {start} and {end}. "
            "Check that the ticker symbol is correct."
        )

    logger.info("Fetched %d rows for %s (%s to %s)", len(raw_data), ticker, start, end)
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

    n_missing = int(df.isna().any(axis=1).sum())
    if n_missing > 0:
        logger.warning("%d rows had missing values before forward-fill", n_missing)

    df = df.ffill()
    df = df.dropna()

    logger.info("Cleaned data: %d rows remaining", len(df))
    return df


def _build_cache_path(ticker: str, start: str, end: str, cache_dir: str) -> str:
    """
    Build a deterministic cache file path for a given ticker/date range.

    The ticker is sanitized (e.g. "=" replaced) since some symbols like
    "USDBRL=X" contain characters that are best avoided in filenames.
    """
    safe_ticker = ticker.replace("=", "_").replace("/", "_")
    filename = f"{safe_ticker}_{start}_{end}.csv"
    return os.path.join(cache_dir, filename)


def load_price_data(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = "data",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load cleaned price data for a ticker, using a local cache when available.

    On the first call for a given ticker/date range, downloads and cleans
    the data via fetch_price_data + clean_price_data, then saves the result
    to a local CSV cache. Subsequent calls with the same parameters read
    directly from the cache instead of hitting the yfinance API again.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start: Start date ("YYYY-MM-DD").
        end: End date ("YYYY-MM-DD").
        cache_dir: Directory where cache files are stored.
        force_refresh: If True, ignores any existing cache and re-downloads.

    Returns:
        A cleaned DataFrame with OHLCV data, indexed by date.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _build_cache_path(ticker, start, end, cache_dir)

    if os.path.exists(cache_path) and not force_refresh:
        try:
            logger.info("Loading from cache: %s", cache_path)
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        except (pd.errors.ParserError, OSError) as exc:
            # Cache file exists but is unreadable/corrupted — fall back to
            # a fresh download instead of crashing.
            logger.warning("Cache file unreadable (%s), re-downloading", exc)

    logger.info("Downloading fresh data for %s (%s to %s)", ticker, start, end)
    raw_df = fetch_price_data(ticker, start, end)
    clean_df = clean_price_data(raw_df)
    clean_df.to_csv(cache_path)

    return clean_df


if __name__ == "__main__":
    # Library modules should not call basicConfig themselves — only the
    # entry point (this script, in this case) configures logging output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # First call: downloads and caches. Second call: reads from cache.
    # (delete the file in data/ if you want to force a fresh download)
    df1 = load_price_data("USDBRL=X", "2024-01-01", "2024-12-31")
    print(df1.head())

    df2 = load_price_data("USDBRL=X", "2024-01-01", "2024-12-31")
    print(df2.head())

    # Demonstrating the error handling: an invalid ticker should raise
    # DataLoadError with a clear message instead of failing silently.
    try:
        load_price_data("THIS_IS_NOT_A_REAL_TICKER", "2024-01-01", "2024-12-31")
    except DataLoadError as exc:
        logger.error("Handled expected failure: %s", exc)