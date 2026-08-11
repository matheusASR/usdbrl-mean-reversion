"""
tests/test_indicators.py

Unit tests for indicators.py. Two complementary strategies are used:
  1. Extreme cases with an obvious expected result (e.g. price rising
     every single day -> RSI should hit the ceiling, 100).
  2. "Golden value" tests — the exact expected output, computed once with
     the same formula on a small controlled series, hardcoded here as a
     fixed reference.
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import add_indicators, calculate_atr, calculate_rsi, calculate_zscore


# ---------------------------------------------------------------------------
# calculate_rsi
# ---------------------------------------------------------------------------

class TestCalculateRSI:
    def test_golden_values(self):
        df = pd.DataFrame({"Close": [10, 11, 12, 11, 13, 12, 14]})
        rsi = calculate_rsi(df, period=3)

        expected = [np.nan, np.nan, np.nan, 66.666667, 83.333333, 60.606061, 78.333333]
        for actual, exp in zip(rsi, expected):
            if np.isnan(exp):
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(exp, rel=1e-4)

    def test_monotonically_rising_price_hits_100(self):
        # every day is a gain, no losses at all -> RSI must be 100
        df = pd.DataFrame({"Close": range(1, 21)})  # 1, 2, 3, ..., 20
        rsi = calculate_rsi(df, period=14)
        assert (rsi.dropna() == 100).all()

    def test_monotonically_falling_price_hits_0(self):
        # every day is a loss, no gains at all -> RSI must be 0
        df = pd.DataFrame({"Close": range(20, 0, -1)})  # 20, 19, 18, ..., 1
        rsi = calculate_rsi(df, period=14)
        assert (rsi.dropna() == 0).all()

    def test_flat_price_is_neutral_50(self):
        # no movement at all -> RSI is undefined by formula; we define it as 50
        df = pd.DataFrame({"Close": [10.0] * 20})
        rsi = calculate_rsi(df, period=14)
        assert (rsi.dropna() == 50).all()

    def test_warmup_period_is_nan(self):
        df = pd.DataFrame({"Close": range(1, 21)})
        rsi = calculate_rsi(df, period=14)
        assert rsi.iloc[:14].isna().all()
        assert rsi.iloc[14:].notna().all()

    def test_stays_within_bounds(self):
        rng = np.random.default_rng(seed=42)
        prices = 100 + np.cumsum(rng.normal(0, 1, size=200))
        df = pd.DataFrame({"Close": prices})
        rsi = calculate_rsi(df, period=14).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()


# ---------------------------------------------------------------------------
# calculate_atr
# ---------------------------------------------------------------------------

class TestCalculateATR:
    def test_golden_values(self):
        df = pd.DataFrame({
            "High":  [10.5, 11.2, 12.3, 11.8, 13.5],
            "Low":   [9.8, 10.6, 11.5, 10.9, 12.4],
            "Close": [10.2, 11.0, 12.0, 11.2, 13.1],
        })
        atr = calculate_atr(df, period=3)

        expected = [np.nan, np.nan, np.nan, 1.1, 1.5]
        for actual, exp in zip(atr, expected):
            if np.isnan(exp):
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(exp, rel=1e-4)

    def test_never_negative(self):
        # True Range is built from absolute differences, so ATR can never
        # be negative regardless of price direction.
        rng = np.random.default_rng(seed=7)
        n = 100
        close = 100 + np.cumsum(rng.normal(0, 1, size=n))
        high = close + rng.uniform(0, 1, size=n)
        low = close - rng.uniform(0, 1, size=n)
        df = pd.DataFrame({"High": high, "Low": low, "Close": close})

        atr = calculate_atr(df, period=14).dropna()
        assert (atr >= 0).all()

    def test_warmup_period_is_nan(self):
        df = pd.DataFrame({
            "High": range(1, 21), "Low": range(0, 20), "Close": range(1, 21),
        })
        atr = calculate_atr(df, period=14)
        assert atr.iloc[:14].isna().all()
        assert atr.iloc[14:].notna().all()


# ---------------------------------------------------------------------------
# calculate_zscore
# ---------------------------------------------------------------------------

class TestCalculateZScore:
    def test_golden_values(self):
        df = pd.DataFrame({"Close": [10, 12, 11, 15, 9]})
        zscore = calculate_zscore(df, period=3)

        expected = [np.nan, np.nan, 0.0, 1.120897, -0.872872]
        for actual, exp in zip(zscore, expected):
            if np.isnan(exp):
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(exp, rel=1e-4)

    def test_flat_price_gives_zero_not_error(self):
        # std == 0 for the whole window -> would normally divide by zero;
        # we define this edge case as Z-score = 0.
        df = pd.DataFrame({"Close": [10.0] * 20})
        zscore = calculate_zscore(df, period=10)
        assert (zscore.dropna() == 0).all()

    def test_warmup_period_is_nan(self):
        df = pd.DataFrame({"Close": range(1, 21)})
        zscore = calculate_zscore(df, period=20)
        assert zscore.iloc[:19].isna().all()
        assert zscore.iloc[19:].notna().all()


# ---------------------------------------------------------------------------
# add_indicators (orchestration)
# ---------------------------------------------------------------------------

class TestAddIndicators:
    def _make_df(self, n=40):
        rng = np.random.default_rng(seed=1)
        close = 5 + np.cumsum(rng.normal(0, 0.02, size=n))
        high = close + rng.uniform(0, 0.05, size=n)
        low = close - rng.uniform(0, 0.05, size=n)
        return pd.DataFrame({"High": high, "Low": low, "Close": close})

    def test_adds_expected_columns(self):
        df = self._make_df()
        result = add_indicators(df)
        for col in ["RSI", "Z_score", "ATR"]:
            assert col in result.columns

    def test_does_not_mutate_input(self):
        df = self._make_df()
        original_columns = list(df.columns)
        add_indicators(df)
        # the original DataFrame passed in should be untouched
        assert list(df.columns) == original_columns

    def test_dropna_removes_warmup_rows(self):
        df = self._make_df(n=40)
        # RSI(14), Z-score(20), ATR(14) -> the longest warm-up is 20
        # periods (Z-score), so the first 19 rows should be dropped.
        result = add_indicators(df, rsi_period=14, zscore_period=20, atr_period=14)
        assert len(result) == len(df) - 19
        assert result[["RSI", "Z_score", "ATR"]].isna().sum().sum() == 0

    def test_dropna_false_keeps_all_rows(self):
        df = self._make_df(n=40)
        result = add_indicators(df, dropna=False)
        assert len(result) == len(df)
