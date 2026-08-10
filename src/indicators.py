"""
indicators.py

Technical indicator calculations used as inputs to the strategy signals:
RSI, Z-score, and ATR. Each function here is a pure calculation — it takes
a price DataFrame and returns a new Series, with no decision logic about
when to buy or sell. That logic lives in strategy.py.
"""

import pandas as pd


def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = "Close") -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) using Wilder's smoothing
    method — the original formula proposed by J. Welles Wilder, and the
    one most charting platforms (TradingView, MetaTrader) use by default.

    Args:
        df: DataFrame containing at least the price_col column.
        period: Lookback window, in periods (default 14, the standard).
        price_col: Column to compute RSI on (default "Close").

    Returns:
        A Series of RSI values (0-100), aligned to df's index. The first
        `period` values will be NaN, since there isn't enough history yet
        to compute a meaningful average.
    """
    delta = df[price_col].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Wilder's smoothing: an EMA with alpha = 1/period, adjust=False.
    # This is what distinguishes the "official" RSI from a simpler
    # simple-moving-average version some naive implementations use.
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Edge case: no losses in the window -> RS is undefined (division by
    # zero) but the correct RSI here is 100 (pure upward strength).
    rsi = rsi.where(avg_loss != 0, 100)

    # Edge case: no gains AND no losses (flat price) -> RSI is undefined;
    # by convention, treat this as neutral (50).
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50)

    rsi.name = "RSI"
    return rsi


def calculate_zscore(df: pd.DataFrame, period: int = 20, price_col: str = "Close") -> pd.Series:
    """
    Calculate the rolling Z-score of price relative to its own moving
    average — how many standard deviations the current price sits from
    its recent mean.

    Uses a 20-period window by convention, matching the standard window
    used for Bollinger Bands (this Z-score is mathematically equivalent
    to the price's position within Bollinger Bands, expressed as a single
    number instead of two lines).

    Args:
        df: DataFrame containing at least the price_col column.
        period: Rolling window size (default 20).
        price_col: Column to compute the Z-score on (default "Close").

    Returns:
        A Series of Z-score values, aligned to df's index. The first
        `period - 1` values will be NaN (not enough history yet).
    """
    price = df[price_col]
    rolling_mean = price.rolling(window=period).mean()
    rolling_std = price.rolling(window=period).std()

    zscore = (price - rolling_mean) / rolling_std

    # Edge case: flat price for the entire window -> std is 0 -> division
    # by zero. The price is exactly at its own mean, so 0 is correct.
    zscore = zscore.where(rolling_std != 0, 0)

    zscore.name = "Z_score"
    return zscore


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Average True Range (ATR) using Wilder's smoothing method
    — the same technique (and same original author) as the RSI above.

    True Range for a given day is the largest of:
        1. High - Low
        2. |High - previous Close|
        3. |Low - previous Close|
    This captures both intraday movement and any gap versus the prior
    close, which a simple High - Low would miss.

    Args:
        df: DataFrame containing High, Low, and Close columns.
        period: Lookback window, in periods (default 14).

    Returns:
        A Series of ATR values, in price units (not %), aligned to df's
        index. The first `period` values will be NaN.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    range_1 = high - low
    range_2 = (high - prev_close).abs()
    range_3 = (low - prev_close).abs()

    true_range = pd.concat([range_1, range_2, range_3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr.name = "ATR"
    return atr


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_price_data

    prices = load_price_data("USDBRL=X", "2024-01-01", "2024-12-31")
    prices["RSI"] = calculate_rsi(prices)
    prices["Z_score"] = calculate_zscore(prices)
    prices["ATR"] = calculate_atr(prices)
    print(prices[["Close", "RSI", "Z_score", "ATR"]].tail(20))