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

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

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


@dataclass
class OpenPosition:
    """A currently open (not yet closed) position, tracked while the
    simulation walks forward in time."""

    entry_date: pd.Timestamp
    direction: int
    entry_price: float
    execution_entry_price: float
    stop_price: float
    position_size: float


def run_simulation(
    df: pd.DataFrame,
    cost_pct: float = DEFAULT_COST_PCT,
) -> tuple[list[Trade], Optional[OpenPosition]]:
    """
    Walk the price/signal history day by day, opening and closing
    positions, and return the list of closed trades.

    Expects df to contain: Close, exit_signal, entry_signal, stop_price,
    and position_size columns (see strategy.generate_signals and
    risk.apply_risk_management).

    Design decisions (see module docstring for the full reasoning):
        - Exit conditions are evaluated using the state as of the start
          of the day. If a position closes today, a new one is NOT
          opened on the same day.
        - If both the stop-loss and the mean-reversion exit trigger on
          the same day, the stop-loss takes priority.
        - The stop-loss is checked against the closing price only (not
          intraday High/Low) — consistent with the project's "next-bar
          execution at close" simplification.

    Args:
        df: Price DataFrame with signals and risk parameters already
            computed.
        cost_pct: Transaction cost per leg (default 0.05%).

    Returns:
        A tuple of (trades, open_position):
            - trades: list of closed Trade objects.
            - open_position: the still-open OpenPosition at the end of
              the data, or None if flat. A non-None value here means the
              backtest ended mid-trade — the equity curve step needs
              this to mark the final days to market.
    """
    trades: list[Trade] = []
    open_position: Optional[OpenPosition] = None

    for date, row in df.iterrows():
        if open_position is not None:
            stop_hit = (
                row["Close"] <= open_position.stop_price
                if open_position.direction == 1
                else row["Close"] >= open_position.stop_price
            )
            mean_reversion_exit = bool(row["exit_signal"])

            if stop_hit or mean_reversion_exit:
                exit_reason = "stop_loss" if stop_hit else "mean_reversion"
                exit_price = open_position.stop_price if stop_hit else row["Close"]
                execution_exit_price = apply_execution_cost(
                    exit_price, open_position.direction, is_entry=False, cost_pct=cost_pct
                )
                trades.append(
                    Trade(
                        entry_date=open_position.entry_date,
                        exit_date=date,
                        direction=open_position.direction,
                        entry_price=open_position.entry_price,
                        exit_price=exit_price,
                        execution_entry_price=open_position.execution_entry_price,
                        execution_exit_price=execution_exit_price,
                        position_size=open_position.position_size,
                        exit_reason=exit_reason,
                    )
                )
                open_position = None
                continue  # no same-day re-entry

        if open_position is None and row["entry_signal"] != 0:
            direction = int(row["entry_signal"])
            entry_price = row["Close"]
            open_position = OpenPosition(
                entry_date=date,
                direction=direction,
                entry_price=entry_price,
                execution_entry_price=apply_execution_cost(
                    entry_price, direction, is_entry=True, cost_pct=cost_pct
                ),
                stop_price=row["stop_price"],
                position_size=row["position_size"],
            )

    if open_position is not None:
        logger.warning(
            "Backtest ended with an open %s position from %s, still unresolved",
            "long" if open_position.direction == 1 else "short",
            open_position.entry_date.date(),
        )

    return trades, open_position


def build_equity_curve(
    df: pd.DataFrame,
    trades: list[Trade],
    open_position: Optional[OpenPosition],
    initial_capital: float,
) -> pd.Series:
    """
    Build a daily mark-to-market equity curve, starting from
    initial_capital.

    Each day's equity is initial_capital plus:
        - realized P&L: the sum of net_pnl from every trade whose
          exit_date is on or before that day (already cost-adjusted).
        - unrealized P&L: for any position open on that day (either a
          now-closed trade, for the days before it exited, or the
          dangling open_position at the end), the mark-to-market gain/
          loss using that day's Close price against the cost-adjusted
          entry price.

    Using the cost-adjusted entry price for unrealized P&L (rather than
    the raw theoretical entry price) keeps the transition from
    "unrealized" to "realized" smooth on the exit day — the only jump
    left is the exit cost itself, which is real.

    Args:
        df: Price DataFrame (must contain Close, indexed by date, same
            index used during the simulation).
        trades: Closed trades from run_simulation.
        open_position: The still-open position from run_simulation, or
            None.
        initial_capital: Starting capital.

    Returns:
        A pd.Series indexed like df, representing total capital at the
        close of each day.
    """
    dates = df.index
    realized_pnl = pd.Series(0.0, index=dates)
    unrealized_pnl = pd.Series(0.0, index=dates)

    for trade in trades:
        # Realized P&L kicks in from the exit date onward.
        realized_pnl.loc[trade.exit_date:] += trade.net_pnl

        # Mark-to-market for the days the trade was open: from entry up
        # to (but not including) the exit day, which is realized instead.
        open_days = dates[(dates >= trade.entry_date) & (dates < trade.exit_date)]
        unrealized_pnl.loc[open_days] += (
            trade.direction
            * (df.loc[open_days, "Close"] - trade.execution_entry_price)
            * trade.position_size
        )

    if open_position is not None:
        open_days = dates[dates >= open_position.entry_date]
        unrealized_pnl.loc[open_days] += (
            open_position.direction
            * (df.loc[open_days, "Close"] - open_position.execution_entry_price)
            * open_position.position_size
        )

    equity = initial_capital + realized_pnl + unrealized_pnl
    equity.name = "equity"
    return equity


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """
    Convert a list of Trade objects into a DataFrame, materializing the
    computed properties (gross_pnl, net_pnl, costs) as real columns.
    """
    columns = [
        "entry_date", "exit_date", "direction", "entry_price", "exit_price",
        "execution_entry_price", "execution_exit_price", "position_size",
        "exit_reason", "gross_pnl", "net_pnl", "costs",
    ]
    if not trades:
        return pd.DataFrame(columns=columns)

    records = [
        {
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "execution_entry_price": t.execution_entry_price,
            "execution_exit_price": t.execution_exit_price,
            "position_size": t.position_size,
            "exit_reason": t.exit_reason,
            "gross_pnl": t.gross_pnl,
            "net_pnl": t.net_pnl,
            "costs": t.costs,
        }
        for t in trades
    ]
    return pd.DataFrame(records, columns=columns)


@dataclass
class BacktestResult:
    """Everything metrics.py needs to evaluate a backtest run."""

    trades: pd.DataFrame
    equity_curve: pd.Series
    open_position: Optional[OpenPosition]


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float,
    cost_pct: float = DEFAULT_COST_PCT,
) -> BacktestResult:
    """
    Single entry point for the backtest engine: runs the day-by-day
    simulation and builds the equity curve, returning everything
    metrics.py needs to evaluate performance.

    Args:
        df: Price DataFrame with signals and risk parameters already
            computed (see strategy.generate_signals and
            risk.apply_risk_management).
        initial_capital: Starting capital.
        cost_pct: Transaction cost per leg (default 0.05%).

    Returns:
        A BacktestResult with the trade log (as a DataFrame), the daily
        equity curve, and any position still open at the end.
    """
    trade_list, open_position = run_simulation(df, cost_pct=cost_pct)
    equity_curve = build_equity_curve(df, trade_list, open_position, initial_capital)
    trades_df = trades_to_dataframe(trade_list)

    final_equity = equity_curve.iloc[-1]
    total_return_pct = (final_equity / initial_capital - 1) * 100
    logger.info(
        "Backtest complete: %d trades, final equity %.2f (%.2f%% return)",
        len(trade_list), final_equity, total_return_pct,
    )

    return BacktestResult(trades=trades_df, equity_curve=equity_curve, open_position=open_position)


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

    # Simulation example: a synthetic 8-day series with one long entry
    # (triggered manually here, since we're not running the full
    # indicators/strategy pipeline in this quick check) that reverts
    # and exits via mean reversion — plus a second entry that gets
    # stopped out.
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    sim_df = pd.DataFrame(
        {
            "Close":         [5.00, 5.02, 5.05, 5.10, 4.90, 4.85, 4.80, 4.95],
            "entry_signal":  [1,    0,    0,    0,    -1,   0,    0,    0],
            "exit_signal":   [False, False, False, True, False, False, False, True],
            "stop_price":    [4.90, None, None, None, 4.95, None, None, None],
            "position_size": [10_000, None, None, None, 8_000, None, None, None],
        },
        index=dates,
    )
    print("\nSimulation input:")
    print(sim_df)

    trades, dangling_position = run_simulation(sim_df)
    print(f"\nClosed trades: {len(trades)}")
    for t in trades:
        print(
            f"  {t.entry_date.date()} -> {t.exit_date.date()} | "
            f"dir={t.direction:+d} | reason={t.exit_reason} | net_pnl={t.net_pnl:.2f}"
        )
    print(f"Still open at end: {dangling_position}")

    equity = build_equity_curve(sim_df, trades, dangling_position, initial_capital=100_000)
    print("\nEquity curve:")
    print(equity)

    # Single entry-point example — what strategy.py/risk.py hand off to,
    # and what metrics.py will consume next.
    result = run_backtest(sim_df, initial_capital=100_000)
    print("\n--- run_backtest result ---")
    print("\nTrades:")
    print(result.trades)
    print(f"\nFinal equity: {result.equity_curve.iloc[-1]:.2f}")