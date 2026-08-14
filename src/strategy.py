"""
strategy.py

Signal generation logic: given a price DataFrame already containing the
indicators (RSI, Z_score, ATR — see indicators.py), decide which rows
qualify as entry or exit signals.

Important design note: this module is stateless. It doesn't know whether
a position is currently open — it only evaluates, for each row
independently, whether that row's indicator values satisfy the entry or
exit conditions. Tracking open positions over time and actually acting on
these signals is the responsibility of backtester.py.
"""

import numpy as np
import pandas as pd


def generate_entry_signal(
    df: pd.DataFrame,
    zscore_entry: float = 2.0,
    rsi_lower: float = 30,
    rsi_upper: float = 70,
) -> pd.Series:
    """
    Generate entry signals based on Z-score + RSI confirmation.

    Long entry:  Z_score < -zscore_entry  AND  RSI < rsi_lower
    Short entry: Z_score > +zscore_entry  AND  RSI > rsi_upper

    Args:
        df: DataFrame containing Z_score and RSI columns
            (see indicators.add_indicators).
        zscore_entry: Absolute Z-score threshold to trigger an entry.
        rsi_lower: RSI threshold below which the long condition confirms.
        rsi_upper: RSI threshold above which the short condition confirms.

    Returns:
        A Series of integers aligned to df's index:
            1  -> long entry signal
           -1  -> short entry signal
            0  -> no entry signal
    """
    long_condition = (df["Z_score"] < -zscore_entry) & (df["RSI"] < rsi_lower)
    short_condition = (df["Z_score"] > zscore_entry) & (df["RSI"] > rsi_upper)

    signal = np.select(
        condlist=[long_condition, short_condition],
        choicelist=[1, -1],
        default=0,
    )

    return pd.Series(signal, index=df.index, name="entry_signal")


def generate_exit_signal(df: pd.DataFrame, zscore_exit_band: float = 0.5) -> pd.Series:
    """
    Generate mean-reversion exit signals: True when the Z-score has
    returned close enough to zero that the reversion thesis is considered
    fulfilled.

    This condition is direction-agnostic — it applies the same way to
    both long and short positions, since "the price reverted to its
    mean" is true regardless of which side the position is on. Rows
    with a NaN Z_score (e.g. warm-up period) naturally evaluate to False,
    since NaN comparisons are always False in pandas.

    Args:
        df: DataFrame containing a Z_score column.
        zscore_exit_band: Absolute Z-score threshold below which the
            reversion is considered complete (default 0.5, i.e. the
            band [-0.5, +0.5]).

    Returns:
        A boolean Series aligned to df's index.
    """
    exit_signal = df["Z_score"].abs() <= zscore_exit_band
    exit_signal.name = "exit_signal"
    return exit_signal


def generate_signals(
    df: pd.DataFrame,
    zscore_entry: float = 2.0,
    rsi_lower: float = 30,
    rsi_upper: float = 70,
    zscore_exit_band: float = 0.5,
) -> pd.DataFrame:
    """
    Add entry_signal and exit_signal columns to a price DataFrame that
    already contains the indicators (see indicators.add_indicators).

    This is the single entry point the rest of the pipeline
    (backtester.py) should use — same design pattern as
    indicators.add_indicators: works on a copy, doesn't mutate the input.

    Args:
        df: DataFrame containing RSI and Z_score columns.
        zscore_entry: Absolute Z-score threshold to trigger an entry.
        rsi_lower: RSI threshold below which the long condition confirms.
        rsi_upper: RSI threshold above which the short condition confirms.
        zscore_exit_band: Absolute Z-score threshold for exit.

    Returns:
        A new DataFrame (input not mutated) with entry_signal (-1/0/1)
        and exit_signal (bool) columns added.
    """
    result = df.copy()
    result["entry_signal"] = generate_entry_signal(
        result, zscore_entry=zscore_entry, rsi_lower=rsi_lower, rsi_upper=rsi_upper
    )
    result["exit_signal"] = generate_exit_signal(result, zscore_exit_band=zscore_exit_band)
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_price_data
    from src.indicators import add_indicators

    prices = load_price_data("USDBRL=X", "2015-01-01", "2024-12-31")
    prices = add_indicators(prices)
    prices = generate_signals(prices)

    print(prices["entry_signal"].value_counts())
    print(f"\nExit signal True on {prices['exit_signal'].sum()} of {len(prices)} days")
    print("\nSample long entries:")
    print(prices[prices["entry_signal"] == 1][["Close", "RSI", "Z_score"]].head())