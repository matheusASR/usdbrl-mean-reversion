"""
tests/test_data_loader.py

Unit tests for data_loader.py. All calls to yfinance are mocked — these
tests must run fast and deterministically, without touching the network.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data_loader import (
    DataLoadError,
    _build_cache_path,
    clean_price_data,
    fetch_price_data,
    load_price_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw_frame(with_multiindex: bool = False) -> pd.DataFrame:
    """
    Build a small synthetic OHLCV DataFrame for use across tests.

    Deliberately unsorted, with one duplicate date and one row containing
    missing values, so the same fixture can exercise every cleaning rule.
    """
    dates = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-04"])
    df = pd.DataFrame(
        {
            "Open": [5.0, 4.9, 4.9, None],
            "High": [5.1, 5.0, 5.0, 5.2],
            "Low": [4.9, 4.8, 4.8, 5.0],
            "Close": [5.05, 4.95, 4.95, None],
            "Volume": [0, 0, 0, 0],
        },
        index=dates,
    )
    df.index.name = "Date"

    if with_multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["USDBRL=X"]])

    return df


# ---------------------------------------------------------------------------
# clean_price_data
# ---------------------------------------------------------------------------

class TestCleanPriceData:
    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"Close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing expected columns"):
            clean_price_data(df)

    def test_sorts_index_chronologically(self):
        cleaned = clean_price_data(make_raw_frame())
        assert cleaned.index.is_monotonic_increasing

    def test_removes_duplicate_dates(self):
        cleaned = clean_price_data(make_raw_frame())
        assert not cleaned.index.duplicated().any()

    def test_forward_fills_missing_values(self):
        cleaned = clean_price_data(make_raw_frame())
        assert cleaned.isna().sum().sum() == 0
        # the last row's Open/Close were None -> should carry the previous row's values
        assert cleaned.iloc[-1]["Close"] == cleaned.iloc[-2]["Close"]


# ---------------------------------------------------------------------------
# _build_cache_path
# ---------------------------------------------------------------------------

def test_build_cache_path_sanitizes_ticker(tmp_path):
    path = _build_cache_path("USDBRL=X", "2024-01-01", "2024-12-31", str(tmp_path))
    assert "USDBRL_X" in path
    assert "=" not in path
    assert path.endswith("2024-01-01_2024-12-31.csv")


# ---------------------------------------------------------------------------
# fetch_price_data (yfinance mocked — no real network calls)
# ---------------------------------------------------------------------------

class TestFetchPriceData:
    def test_raises_on_invalid_date_range(self):
        with pytest.raises(ValueError, match="must be before"):
            fetch_price_data("USDBRL=X", "2024-12-31", "2024-01-01")

    @patch("src.data_loader.yf.download")
    def test_flattens_multiindex_columns(self, mock_download):
        mock_download.return_value = make_raw_frame(with_multiindex=True)
        result = fetch_price_data("USDBRL=X", "2024-01-01", "2024-12-31")
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @patch("src.data_loader.yf.download")
    def test_raises_data_load_error_on_empty_result(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        with pytest.raises(DataLoadError, match="No data returned"):
            fetch_price_data("FAKE_TICKER", "2024-01-01", "2024-12-31")

    @patch("src.data_loader.yf.download")
    def test_wraps_download_exceptions(self, mock_download):
        mock_download.side_effect = RuntimeError("network unreachable")
        with pytest.raises(DataLoadError, match="Failed to download"):
            fetch_price_data("USDBRL=X", "2024-01-01", "2024-12-31")


# ---------------------------------------------------------------------------
# load_price_data (cache behavior)
# ---------------------------------------------------------------------------

class TestLoadPriceData:
    @patch("src.data_loader.fetch_price_data")
    def test_downloads_and_caches_on_first_call(self, mock_fetch, tmp_path):
        mock_fetch.return_value = make_raw_frame()
        load_price_data("USDBRL=X", "2024-01-01", "2024-12-31", cache_dir=str(tmp_path))

        mock_fetch.assert_called_once()
        assert len(list(tmp_path.glob("*.csv"))) == 1

    @patch("src.data_loader.fetch_price_data")
    def test_second_call_uses_cache_not_network(self, mock_fetch, tmp_path):
        mock_fetch.return_value = make_raw_frame()
        load_price_data("USDBRL=X", "2024-01-01", "2024-12-31", cache_dir=str(tmp_path))
        load_price_data("USDBRL=X", "2024-01-01", "2024-12-31", cache_dir=str(tmp_path))

        # fetch_price_data must only be called once — the second load
        # should be served entirely from the cache file.
        mock_fetch.assert_called_once()

    @patch("src.data_loader.fetch_price_data")
    def test_force_refresh_ignores_cache(self, mock_fetch, tmp_path):
        mock_fetch.return_value = make_raw_frame()
        load_price_data("USDBRL=X", "2024-01-01", "2024-12-31", cache_dir=str(tmp_path))
        load_price_data(
            "USDBRL=X", "2024-01-01", "2024-12-31",
            cache_dir=str(tmp_path), force_refresh=True,
        )

        assert mock_fetch.call_count == 2
