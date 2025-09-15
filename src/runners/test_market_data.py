import types
import builtins
import importlib

from unittest.mock import patch, MagicMock

import importlib.util
from pathlib import Path

# import module directly to avoid package-level side-effects (config requiring env vars)
spec = importlib.util.spec_from_file_location("market_data", str(Path(__file__).resolve().parents[1] / "src" / "core" / "market_data.py"))
market_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(market_data)


class DummyDF:
    def __init__(self, empty=False):
        self._empty = empty
    @property
    def empty(self):
        return self._empty


def test_fetch_price_history_returns_df_when_available(monkeypatch):
    # Mock yfinance.Ticker and pandas DataFrame behavior
    dummy = DummyDF(empty=False)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = dummy

    fake_yf = types.SimpleNamespace(Ticker=lambda t: mock_ticker)

    with patch.dict('sys.modules', {'yfinance': fake_yf}):
        df = market_data.fetch_price_history('AEM.TO', '2020-01-01', '2020-12-31')
        assert df is dummy


def test_fetch_price_history_returns_none_when_empty(monkeypatch):
    dummy = DummyDF(empty=True)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = dummy

    fake_yf = types.SimpleNamespace(Ticker=lambda t: mock_ticker)

    with patch.dict('sys.modules', {'yfinance': fake_yf}):
        df = market_data.fetch_price_history('AEM.TO', '2020-01-01', '2020-12-31')
        assert df is None
