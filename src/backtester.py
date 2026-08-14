"""
backtester.py

Trade simulation engine. Walks the price history day by day, opening and
closing positions according to the signals from strategy.py and the risk
parameters from risk.py, applying transaction costs, and producing a
trade log plus a daily equity curve.

This is the first module in the pipeline that carries state across rows.
Everything before this (indicators, signals, risk parameters) was a pure
per-row calculation with no memory of prior days — this module is where
that changes.
"""

from dataclasses import dataclass

import pandas as pd

DEFAULT_COST_PCT = 0.0005  # 0.05% per leg, matching the project spec


@dataclass
class Trade:
    """A single closed trade, with both theoretical and cost-adjusted prices."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int  # 1 = long, -1 = short
    entry_price: float  # theoretical (signal) price, before transaction cost
    exit_price: float  # theoretical (signal) price, before transaction cost
    execution_entry_price: float  # after transaction cost
    execution_exit_price: float  # after transaction cost
    position_size: float
    exit_reason: str  # "stop_loss" or "mean_reversion"

    @property
    def gross_pnl(self) -> float:
        """P&L using theoretical prices, ignoring transaction costs."""
        return self.direction * (self.exit_price - self.entry_price) * self.position_size

    @property
    def net_pnl(self) -> float:
        """P&L using cost-adjusted execution prices — the realistic result."""
        return (
            self.direction
            * (self.execution_exit_price - self.execution_entry_price)
            * self.position_size
        )

    @property
    def costs(self) -> float:
        """Total transaction cost paid on this trade (gross_pnl - net_pnl)."""
        return self.gross_pnl - self.net_pnl


def apply_execution_cost(
    price: float,
    direction: int,
    is_entry: bool,
    cost_pct: float = DEFAULT_COST_PCT,
) -> float:
    """
    Adjust a theoretical price for transaction cost (spread + slippage),
    simulating a slightly worse fill than the "clean" signal price.

    Uses the same unified-sign trick as calculate_stop_price in risk.py:
    whether this leg is effectively a "buy" or a "sell" depends on both
    the position's direction and whether it's an entry or an exit, and
    that combination collapses into a single sign:

        buy_sign = direction * (1 if is_entry else -1)

    A "buy" (buy_sign=+1) always fills at a slightly worse (higher)
    price; a "sell" (buy_sign=-1) always fills at a slightly worse
    (lower) price. See the module docstring's table for the four cases.

    Args:
        price: The theoretical (signal) price.
        direction: 1 for long, -1 for short.
        is_entry: True if this is the entry leg, False if the exit leg.
        cost_pct: Cost per leg, as a fraction (default 0.0005 = 0.05%).

    Returns:
        The cost-adjusted execution price.
    """
    entry_sign = 1 if is_entry else -1
    buy_sign = direction * entry_sign
    return price * (1 + buy_sign * cost_pct)


if __name__ == "__main__":
    # Sanity check: all four direction/leg combinations
    for direction, label in [(1, "long"), (-1, "short")]:
        for is_entry, leg in [(True, "entry"), (False, "exit")]:
            adjusted = apply_execution_cost(5.00, direction, is_entry)
            print(f"{label:5s} {leg:5s}: 5.00 -> {adjusted:.4f}")

    # Example trade, to sanity-check the Trade dataclass end-to-end
    trade = Trade(
        entry_date=pd.Timestamp("2024-01-01"),
        exit_date=pd.Timestamp("2024-01-10"),
        direction=1,
        entry_price=5.00,
        exit_price=5.10,
        execution_entry_price=apply_execution_cost(5.00, 1, is_entry=True),
        execution_exit_price=apply_execution_cost(5.10, 1, is_entry=False),
        position_size=10_000,
        exit_reason="mean_reversion",
    )
    print(f"\nGross PnL: {trade.gross_pnl:.2f}")
    print(f"Net PnL:   {trade.net_pnl:.2f}")
    print(f"Costs:     {trade.costs:.2f}")
