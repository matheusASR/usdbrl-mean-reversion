"""
risk.py

Position sizing and stop-loss logic. Given an entry signal and the ATR at
that point in time, this module decides two things: where the stop-loss
sits, and how many units to trade so that a fixed % of capital is at risk
if that stop is hit (volatility-based position sizing).
"""

import pandas as pd


def calculate_stop_price(
    entry_price,
    atr,
    direction,
    atr_multiplier: float = 2.0,
):
    """
    Calculate the stop-loss price for a position, given its entry price,
    the ATR at entry, and its direction.

    Uses a single unified formula for both long and short positions by
    treating direction as a mathematical sign (+1 or -1):

        stop_price = entry_price - direction * atr_multiplier * ATR

    For a long (direction=1), this places the stop below the entry price.
    For a short (direction=-1), the double negative flips it, placing the
    stop above the entry price.

    Args:
        entry_price: Price at which the position was opened. Can be a
            scalar or a pandas Series (vectorized).
        atr: ATR value at entry. Same shape as entry_price.
        direction: 1 for long, -1 for short. Same shape as entry_price,
            or a scalar applied to all rows.
        atr_multiplier: How many ATRs away the stop sits (default 2.0).

    Returns:
        The stop-loss price, same shape as the inputs.
    """
    return entry_price - direction * atr_multiplier * atr


if __name__ == "__main__":
    # Scalar example: a single long trade
    stop = calculate_stop_price(entry_price=5.00, atr=0.05, direction=1)
    print(f"Long entry at 5.00, ATR=0.05 -> stop at {stop:.4f}")

    stop_short = calculate_stop_price(entry_price=5.00, atr=0.05, direction=-1)
    print(f"Short entry at 5.00, ATR=0.05 -> stop at {stop_short:.4f}")

    # Vectorized example: several hypothetical entries at once
    df = pd.DataFrame({
        "Close": [5.00, 4.80, 5.20],
        "ATR": [0.05, 0.06, 0.04],
        "direction": [1, -1, 1],
    })
    df["stop_price"] = calculate_stop_price(df["Close"], df["ATR"], df["direction"])
    print()
    print(df)
