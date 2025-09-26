import os
import sys
import pathlib
import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"


@pytest.mark.integration
def test_fred_series_csv_or_api():
    sys.path.insert(0, str(PROJECT_SRC))
    mod = __import__("apps.macro_sector_app", fromlist=["load_fred_series"])  # reuse robust loader
    load_fred_series = getattr(mod, "load_fred_series")
    fred_key = os.getenv("FRED_API_KEY")
    df = load_fred_series("CPIAUCSL", fred_key=fred_key, start="2015-01-01")
    assert isinstance(df, pd.DataFrame)
    # tolère vide si réseau bloqué; sinon on veut des lignes
    if os.getenv("AF_ALLOW_INTERNET"):
        assert not df.empty


@pytest.mark.integration
def test_gscpi_and_gpr_fetchers():
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")
    sys.path.insert(0, str(PROJECT_SRC))
    macro = __import__("apps.macro_sector_app")
    gscpi = macro.fetch_gscpi()
    assert gscpi is None or isinstance(gscpi, pd.DataFrame)
    gpr = macro.fetch_gpr()
    assert gpr is None or isinstance(gpr, pd.DataFrame)


@pytest.mark.integration
def test_yfinance_multi_prices():
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")
    sys.path.insert(0, str(PROJECT_SRC))
    macro = __import__("apps.macro_sector_app")
    prices = macro.get_multi_yf(["SPY", "QQQ"], start="2020-01-01")
    assert isinstance(prices, pd.DataFrame)
    assert prices.shape[1] >= 1
