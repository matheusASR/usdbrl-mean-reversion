"""
tests/test_risk.py

Unit tests for risk.py: stop-loss pricing, volatility-based position
sizing, and the orchestration function that ties them together.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.risk import apply_risk_management, calculate_position_size, calculate_stop_price


# ---------------------------------------------------------------------------
# calculate_stop_price
# ---------------------------------------------------------------------------

class TestCalculateStopPrice:
    def test_long_stop_is_below_entry(self):
        stop = calculate_stop_price(entry_price=5.00, atr=0.05, direction=1, atr_multiplier=2.0)
        assert stop == pytest.approx(4.90)

    def test_short_stop_is_above_entry(self):
        stop = calculate_stop_price(entry_price=5.00, atr=0.05, direction=-1, atr_multiplier=2.0)
        assert stop == pytest.approx(5.10)

    def test_custom_multiplier(self):
        stop = calculate_stop_price(entry_price=5.00, atr=0.05, direction=1, atr_multiplier=3.0)
        assert stop == pytest.approx(4.85)

    def test_vectorized_matches_scalar(self):
        df = pd.DataFrame({
            "entry": [5.00, 4.80, 5.20],
            "atr": [0.05, 0.06, 0.04],
            "direction": [1, -1, 1],
        })
        result = calculate_stop_price(df["entry"], df["atr"], df["direction"])
        expected = [4.90, 4.92, 5.12]
        for actual, exp in zip(result, expected):
            assert actual == pytest.approx(exp)


# ---------------------------------------------------------------------------
# calculate_position_size
# ---------------------------------------------------------------------------

class TestCalculatePositionSize:
    def test_known_values(self):
        # risk_amount = 100,000 * 0.01 = 1,000
        # stop_distance = |5.00 - 4.90| = 0.10
        # position_size = 1,000 / 0.10 = 10,000
        size = calculate_position_size(
            capital=100_000, risk_per_trade=0.01, entry_price=5.00, stop_price=4.90
        )
        assert size == pytest.approx(10_000)

    def test_wider_stop_gives_smaller_position(self):
        # same risk budget, but a wider stop -> smaller position (this is
        # the core idea behind volatility-based sizing)
        tight = calculate_position_size(100_000, 0.01, entry_price=5.00, stop_price=4.90)
        wide = calculate_position_size(100_000, 0.01, entry_price=5.00, stop_price=4.70)
        assert wide < tight

    def test_zero_stop_distance_returns_zero(self):
        size = calculate_position_size(
            capital=100_000, risk_per_trade=0.01, entry_price=5.00, stop_price=5.00
        )
        assert size == 0

    def test_zero_stop_distance_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            calculate_position_size(capital=100_000, risk_per_trade=0.01, entry_price=5.00, stop_price=5.00)
        assert "Zero stop distance" in caplog.text

    def test_vectorized_matches_scalar(self):
        entry = pd.Series([5.00, 4.80])
        stop = pd.Series([4.90, 4.92])
        result = calculate_position_size(100_000, 0.01, entry, stop)
        assert result[0] == pytest.approx(10_000)
        assert result[1] == pytest.approx(1_000 / 0.12)


# ---------------------------------------------------------------------------
# apply_risk_management (orchestration)
# ---------------------------------------------------------------------------

class TestApplyRiskManagement:
    def _make_df(self):
        return pd.DataFrame({
            "Close": [5.00, 5.02, 4.80, 5.01],
            "ATR": [0.05, 0.05, 0.06, 0.05],
            "entry_signal": [1, 0, -1, 0],
        })

    def test_adds_expected_columns(self):
        result = apply_risk_management(self._make_df(), capital=100_000)
        assert "stop_price" in result.columns
        assert "position_size" in result.columns

    def test_signal_rows_are_populated(self):
        result = apply_risk_management(self._make_df(), capital=100_000)
        signal_rows = result[result["entry_signal"] != 0]
        assert signal_rows["stop_price"].notna().all()
        assert signal_rows["position_size"].notna().all()

    def test_no_signal_rows_are_nan(self):
        result = apply_risk_management(self._make_df(), capital=100_000)
        no_signal_rows = result[result["entry_signal"] == 0]
        assert no_signal_rows["stop_price"].isna().all()
        assert no_signal_rows["position_size"].isna().all()

    def test_does_not_mutate_input(self):
        df = self._make_df()
        original_columns = list(df.columns)
        apply_risk_management(df, capital=100_000)
        assert list(df.columns) == original_columns

    def test_no_spurious_zero_distance_warning(self, caplog):
        # Regression test: rows with entry_signal == 0 must NOT be run
        # through the stop/size calculation (that would degenerate to
        # stop_price == entry_price and trigger a false "zero distance"
        # warning on every single no-signal row).
        with caplog.at_level(logging.WARNING):
            apply_risk_management(self._make_df(), capital=100_000)
        assert "Zero stop distance" not in caplog.text

    def test_correct_values_for_a_known_row(self):
        result = apply_risk_management(self._make_df(), capital=100_000, risk_per_trade=0.01)
        # row 0: Close=5.00, ATR=0.05, long -> stop=4.90, size=10,000
        assert result.loc[0, "stop_price"] == pytest.approx(4.90)
        assert result.loc[0, "position_size"] == pytest.approx(10_000)
