import os
import sys
import pathlib
import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"


@pytest.mark.integration
def test_yfinance_single_and_indicators():
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")
    sys.path.insert(0, str(PROJECT_SRC))
    stock_app = __import__("apps.stock_analysis_app")
    df = stock_app.get_stock_data("AAPL", period="6mo")
    assert isinstance(df, pd.DataFrame) and not df.empty
    out = stock_app.add_technical_indicators(df)
    assert all(col in out.columns for col in ["SMA_20", "RSI", "MACD"]) 
