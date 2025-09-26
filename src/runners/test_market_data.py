import types
import builtins
import importlib

from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Ensure project src is on path, then import module normally
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from core import stock_utils as market_data  # type: ignore


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
