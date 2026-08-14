"""
risk.py

Position sizing and stop-loss logic. Given an entry signal and the ATR at
that point in time, this module decides two things: where the stop-loss
sits, and how many units to trade so that a fixed % of capital is at risk
if that stop is hit (volatility-based position sizing).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price,
    stop_price,
):
    """
    Calculate position size such that, if the stop-loss is hit, the loss
    equals exactly `risk_per_trade` fraction of capital — no more, no
    less. This is what makes sizing "volatility-based": trades with a
    wider stop (more volatile conditions) get a smaller position, and
    vice versa, keeping the dollar risk per trade constant.

        risk_amount   = capital * risk_per_trade
        stop_distance = |entry_price - stop_price|
        position_size = risk_amount / stop_distance

    Args:
        capital: Total capital available.
        risk_per_trade: Fraction of capital to risk per trade (e.g. 0.01
            for 1%).
        entry_price: Price at which the position was opened. Scalar or
            pandas Series (vectorized).
        stop_price: Stop-loss price (see calculate_stop_price). Same
            shape as entry_price.

    Returns:
        Position size in units of the underlying asset, same shape as
        the inputs. Returns 0 for any row where the stop distance is 0
        (an undefined risk band — safer to size no position at all).
    """
    risk_amount = capital * risk_per_trade
    stop_distance = abs(entry_price - stop_price)

    if np.any(stop_distance == 0):
        logger.warning("Zero stop distance encountered — sizing position to 0 for those rows")

    # Replace a zero distance with infinity before dividing: risk / inf
    # naturally evaluates to 0, avoiding a division-by-zero warning while
    # still producing the correct "don't trade" result.
    safe_distance = np.where(stop_distance == 0, np.inf, stop_distance)
    position_size = risk_amount / safe_distance

    return position_size


def apply_risk_management(
    df: pd.DataFrame,
    capital: float,
    risk_per_trade: float = 0.01,
    atr_multiplier: float = 2.0,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Add stop_price and position_size columns for every row with an entry
    signal (see strategy.generate_signals).

    Only rows where entry_signal != 0 get risk parameters computed — rows
    without a signal are left as NaN. This is deliberate: computing a
    "stop distance" for direction=0 would degenerate to stop_price ==
    entry_price (distance zero), which is meaningless and would trigger
    the zero-distance warning from calculate_position_size on nearly
    every row.

    Args:
        df: DataFrame containing entry_signal, ATR, and price_col columns
            (see strategy.generate_signals + indicators.add_indicators).
        capital: Total capital available.
        risk_per_trade: Fraction of capital to risk per trade (e.g. 0.01).
        atr_multiplier: How many ATRs away the stop sits (default 2.0).
        price_col: Column used as the entry price (default "Close").

    Returns:
        A new DataFrame (input not mutated) with stop_price and
        position_size columns added.
    """
    result = df.copy()
    result["stop_price"] = np.nan
    result["position_size"] = np.nan

    has_signal = result["entry_signal"] != 0
    if has_signal.any():
        entries = result.loc[has_signal]
        stop_price = calculate_stop_price(
            entries[price_col], entries["ATR"], entries["entry_signal"], atr_multiplier
        )
        position_size = calculate_position_size(
            capital, risk_per_trade, entries[price_col], stop_price
        )
        result.loc[has_signal, "stop_price"] = stop_price
        result.loc[has_signal, "position_size"] = position_size

    return result


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
    df["position_size"] = calculate_position_size(
        capital=100_000, risk_per_trade=0.01, entry_price=df["Close"], stop_price=df["stop_price"]
    )
    print()
    print(df)

    # Orchestration example: a DataFrame mixing signal and no-signal rows,
    # like what strategy.generate_signals would produce.
    signals_df = pd.DataFrame({
        "Close": [5.00, 5.02, 4.80, 5.01],
        "ATR": [0.05, 0.05, 0.06, 0.05],
        "entry_signal": [1, 0, -1, 0],
    })
    result = apply_risk_management(signals_df, capital=100_000, risk_per_trade=0.01)
    print()
    print(result)