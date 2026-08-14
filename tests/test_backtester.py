"""
tests/test_backtester.py

Unit tests for backtester.py: execution cost adjustment, the Trade
dataclass, the simulation state machine, the equity curve, and the
run_backtest orchestration function.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    Trade,
    apply_execution_cost,
    build_equity_curve,
    run_backtest,
    run_simulation,
    trades_to_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sim_df() -> pd.DataFrame:
    """
    The same 8-day synthetic scenario used during development: a long
    trade that exits via mean reversion, followed by a short trade that
    exits via stop-loss (with both the stop and the mean-reversion exit
    triggering on the very same day, to test priority).
    """
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "Close":         [5.00, 5.02, 5.05, 5.10, 4.90, 4.85, 4.80, 4.95],
            "entry_signal":  [1,    0,    0,    0,    -1,   0,    0,    0],
            "exit_signal":   [False, False, False, True, False, False, False, True],
            "stop_price":    [4.90, None, None, None, 4.95, None, None, None],
            "position_size": [10_000, None, None, None, 8_000, None, None, None],
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# apply_execution_cost
# ---------------------------------------------------------------------------

class TestApplyExecutionCost:
    def test_long_entry_pays_more(self):
        assert apply_execution_cost(5.00, direction=1, is_entry=True) > 5.00

    def test_long_exit_receives_less(self):
        assert apply_execution_cost(5.00, direction=1, is_entry=False) < 5.00

    def test_short_entry_receives_less(self):
        assert apply_execution_cost(5.00, direction=-1, is_entry=True) < 5.00

    def test_short_exit_pays_more(self):
        assert apply_execution_cost(5.00, direction=-1, is_entry=False) > 5.00

    def test_default_cost_is_five_bps(self):
        adjusted = apply_execution_cost(100.0, direction=1, is_entry=True)
        assert adjusted == pytest.approx(100.05)


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

class TestTrade:
    def _make_trade(self, direction=1, entry=5.00, exit_=5.10, size=10_000):
        return Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-10"),
            direction=direction,
            entry_price=entry,
            exit_price=exit_,
            execution_entry_price=apply_execution_cost(entry, direction, is_entry=True),
            execution_exit_price=apply_execution_cost(exit_, direction, is_entry=False),
            position_size=size,
            exit_reason="mean_reversion",
        )

    def test_gross_pnl_long(self):
        trade = self._make_trade(direction=1, entry=5.00, exit_=5.10, size=10_000)
        assert trade.gross_pnl == pytest.approx(1000.0)

    def test_gross_pnl_short(self):
        # short profits when price falls
        trade = self._make_trade(direction=-1, entry=5.00, exit_=4.90, size=10_000)
        assert trade.gross_pnl == pytest.approx(1000.0)

    def test_net_pnl_is_lower_than_gross_due_to_costs(self):
        trade = self._make_trade()
        assert trade.net_pnl < trade.gross_pnl

    def test_costs_equal_gross_minus_net(self):
        trade = self._make_trade()
        assert trade.costs == pytest.approx(trade.gross_pnl - trade.net_pnl)


# ---------------------------------------------------------------------------
# run_simulation
# ---------------------------------------------------------------------------

class TestRunSimulation:
    def test_produces_two_trades(self):
        trades, open_position = run_simulation(make_sim_df())
        assert len(trades) == 2
        assert open_position is None

    def test_first_trade_exits_by_mean_reversion(self):
        trades, _ = run_simulation(make_sim_df())
        assert trades[0].exit_reason == "mean_reversion"
        assert trades[0].exit_date == pd.Timestamp("2024-01-04")
        assert trades[0].exit_price == pytest.approx(5.10)  # Close, not stop

    def test_second_trade_exits_by_stop_loss_with_priority(self):
        # Day 8 has BOTH exit_signal=True AND the stop price hit —
        # stop-loss must take priority, and the fill must be at the stop
        # price, not the closing price (even though they're equal here
        # by construction, this documents which one the code chose).
        trades, _ = run_simulation(make_sim_df())
        assert trades[1].exit_reason == "stop_loss"
        assert trades[1].exit_price == pytest.approx(4.95)

    def test_no_same_day_reentry(self):
        # Day 4 closes trade 1; entry_signal is 0 on day 4 in the fixture
        # anyway, but this test makes the "no same-day re-entry" rule
        # explicit with a scenario where it would matter.
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Close":         [5.00, 5.10, 5.10],
                "entry_signal":  [1,    1,    0],
                "exit_signal":   [False, True, False],
                "stop_price":    [4.90, None, None],
                "position_size": [10_000, None, None],
            },
            index=dates,
        )
        trades, open_position = run_simulation(df)
        # trade closes on day 2 (mean reversion); a new entry_signal=1
        # ALSO fires on day 2, but must NOT open a new position that
        # same day.
        assert len(trades) == 1
        assert open_position is None

    def test_no_signals_produces_no_trades(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Close": [5.00, 5.01, 5.02],
                "entry_signal": [0, 0, 0],
                "exit_signal": [False, False, False],
                "stop_price": [None, None, None],
                "position_size": [None, None, None],
            },
            index=dates,
        )
        trades, open_position = run_simulation(df)
        assert trades == []
        assert open_position is None

    def test_dangling_position_returned_when_still_open(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Close": [5.00, 5.01, 5.02],
                "entry_signal": [1, 0, 0],
                "exit_signal": [False, False, False],
                "stop_price": [4.90, None, None],
                "position_size": [10_000, None, None],
            },
            index=dates,
        )
        trades, open_position = run_simulation(df)
        assert trades == []
        assert open_position is not None
        assert open_position.direction == 1

    def test_dangling_position_logs_warning(self, caplog):
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        df = pd.DataFrame(
            {
                "Close": [5.00, 5.01],
                "entry_signal": [1, 0],
                "exit_signal": [False, False],
                "stop_price": [4.90, None],
                "position_size": [10_000, None],
            },
            index=dates,
        )
        with caplog.at_level(logging.WARNING):
            run_simulation(df)
        assert "open" in caplog.text.lower()


# ---------------------------------------------------------------------------
# build_equity_curve
# ---------------------------------------------------------------------------

class TestBuildEquityCurve:
    def test_matches_hand_calculated_values(self):
        df = make_sim_df()
        trades, open_position = run_simulation(df)
        equity = build_equity_curve(df, trades, open_position, initial_capital=100_000)

        # Values hand-verified during development (see conversation).
        expected = {
            "2024-01-01": 99975.0,
            "2024-01-04": 100949.5,
            "2024-01-08": 100510.1,
        }
        for date_str, exp_value in expected.items():
            assert equity.loc[date_str] == pytest.approx(exp_value, abs=0.01)

    def test_length_matches_input(self):
        df = make_sim_df()
        trades, open_position = run_simulation(df)
        equity = build_equity_curve(df, trades, open_position, initial_capital=100_000)
        assert len(equity) == len(df)

    def test_flat_equity_when_no_trades(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"Close": [5.00, 5.01, 5.02]}, index=dates)
        equity = build_equity_curve(df, trades=[], open_position=None, initial_capital=100_000)
        assert (equity == 100_000).all()


# ---------------------------------------------------------------------------
# trades_to_dataframe
# ---------------------------------------------------------------------------

class TestTradesToDataframe:
    def test_empty_list_returns_empty_df_with_columns(self):
        result = trades_to_dataframe([])
        assert len(result) == 0
        assert "net_pnl" in result.columns

    def test_nonempty_list_has_expected_columns_and_values(self):
        trades, _ = run_simulation(make_sim_df())
        result = trades_to_dataframe(trades)
        assert len(result) == 2
        assert result.loc[0, "exit_reason"] == "mean_reversion"
        assert result.loc[0, "net_pnl"] == pytest.approx(949.5, abs=0.01)


# ---------------------------------------------------------------------------
# run_backtest (orchestration)
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_returns_expected_trade_count(self):
        result = run_backtest(make_sim_df(), initial_capital=100_000)
        assert len(result.trades) == 2

    def test_final_equity_matches_sum_of_net_pnl(self):
        result = run_backtest(make_sim_df(), initial_capital=100_000)
        expected_final = 100_000 + result.trades["net_pnl"].sum()
        assert result.equity_curve.iloc[-1] == pytest.approx(expected_final, abs=0.01)

    def test_no_open_position_when_backtest_ends_flat(self):
        result = run_backtest(make_sim_df(), initial_capital=100_000)
        assert result.open_position is None
