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


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_price_data

    prices = load_price_data("USDBRL=X", "2024-01-01", "2024-12-31")
    prices["RSI"] = calculate_rsi(prices)
    print(prices[["Close", "RSI"]].tail(20))
