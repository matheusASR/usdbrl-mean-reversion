"""
tests/test_strategy.py

Unit tests for strategy.py. Since the signal logic is direct comparisons
(no recursive formulas), test series are built with values whose expected
outcome can be reasoned about directly, including exact boundary values.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy import generate_entry_signal, generate_exit_signal, generate_signals


# ---------------------------------------------------------------------------
# generate_entry_signal
# ---------------------------------------------------------------------------

class TestGenerateEntrySignal:
    def test_long_signal_when_both_conditions_met(self):
        df = pd.DataFrame({"Z_score": [-2.5], "RSI": [25]})
        signal = generate_entry_signal(df)
        assert signal.iloc[0] == 1

    def test_short_signal_when_both_conditions_met(self):
        df = pd.DataFrame({"Z_score": [2.5], "RSI": [75]})
        signal = generate_entry_signal(df)
        assert signal.iloc[0] == -1

    def test_no_signal_when_only_zscore_condition_met(self):
        # Z-score is extreme, but RSI doesn't confirm -> no entry
        df = pd.DataFrame({"Z_score": [-2.5], "RSI": [50]})
        signal = generate_entry_signal(df)
        assert signal.iloc[0] == 0

    def test_no_signal_when_only_rsi_condition_met(self):
        # RSI is extreme, but Z-score doesn't confirm -> no entry
        df = pd.DataFrame({"Z_score": [-1.0], "RSI": [25]})
        signal = generate_entry_signal(df)
        assert signal.iloc[0] == 0

    def test_no_signal_in_neutral_zone(self):
        df = pd.DataFrame({"Z_score": [0.0], "RSI": [50]})
        signal = generate_entry_signal(df)
        assert signal.iloc[0] == 0

    def test_boundary_exactly_at_threshold_does_not_trigger(self):
        # Condition is strict "<", so a value exactly AT the threshold
        # should NOT count as a signal.
        df = pd.DataFrame({"Z_score": [-2.0], "RSI": [29]})
        signal = generate_entry_signal(df, zscore_entry=2.0, rsi_lower=30)
        assert signal.iloc[0] == 0

    def test_custom_thresholds_are_respected(self):
        df = pd.DataFrame({"Z_score": [-1.6], "RSI": [35]})
        # doesn't qualify under default thresholds...
        assert generate_entry_signal(df).iloc[0] == 0
        # ...but does under looser custom ones
        loose = generate_entry_signal(df, zscore_entry=1.5, rsi_lower=40)
        assert loose.iloc[0] == 1

    def test_mixed_series(self):
        df = pd.DataFrame({
            "Z_score": [-2.5, -1.0, 0.0, 2.5, 3.0],
            "RSI":     [25,   20,   50,  75,  60],
        })
        signal = generate_entry_signal(df)
        assert list(signal) == [1, 0, 0, -1, 0]


# ---------------------------------------------------------------------------
# generate_exit_signal
# ---------------------------------------------------------------------------

class TestGenerateExitSignal:
    def test_true_inside_band(self):
        df = pd.DataFrame({"Z_score": [0.0, 0.3, -0.3]})
        signal = generate_exit_signal(df, zscore_exit_band=0.5)
        assert signal.all()

    def test_false_outside_band(self):
        df = pd.DataFrame({"Z_score": [0.6, -0.6, 2.0]})
        signal = generate_exit_signal(df, zscore_exit_band=0.5)
        assert not signal.any()

    def test_boundary_exactly_at_band_triggers(self):
        # Condition is "<=", so a value exactly at the band edge counts.
        df = pd.DataFrame({"Z_score": [0.5, -0.5]})
        signal = generate_exit_signal(df, zscore_exit_band=0.5)
        assert signal.all()

    def test_nan_zscore_is_false_not_an_error(self):
        df = pd.DataFrame({"Z_score": [np.nan]})
        signal = generate_exit_signal(df)
        assert signal.iloc[0] == False  # noqa: E712 (explicit bool check)


# ---------------------------------------------------------------------------
# generate_signals (orchestration)
# ---------------------------------------------------------------------------

class TestGenerateSignals:
    def _make_df(self):
        return pd.DataFrame({
            "Z_score": [-2.5, -1.0, 0.0, 2.5],
            "RSI":     [25,   20,   50,  75],
        })

    def test_adds_expected_columns(self):
        result = generate_signals(self._make_df())
        assert "entry_signal" in result.columns
        assert "exit_signal" in result.columns

    def test_does_not_mutate_input(self):
        df = self._make_df()
        original_columns = list(df.columns)
        generate_signals(df)
        assert list(df.columns) == original_columns

    def test_custom_parameters_propagate(self):
        df = pd.DataFrame({"Z_score": [-1.6], "RSI": [35]})
        result = generate_signals(df, zscore_entry=1.5, rsi_lower=40)
        assert result["entry_signal"].iloc[0] == 1
