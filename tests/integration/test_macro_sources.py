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


@pytest.mark.integration
def test_fred_api_real_json(monkeypatch):
    """
    Real call to FRED JSON API (requires AF_ALLOW_INTERNET=1 and FRED_API_KEY).
    Validates inputs/outputs for fetch_fred_series from the real module.
    """
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")

    key = os.getenv("FRED_API_KEY")
    if not key:
        pytest.skip("FRED_API_KEY not set; export it to run this test")

    sys.path.insert(0, str(PROJECT_SRC))
    macro = __import__("analytics.phase3_macro", fromlist=["fetch_fred_series"])  # real module

    df = macro.fetch_fred_series(["CPIAUCSL", "UNRATE"], start="2015-01-01")
    # Inputs respected (columns present)
    assert set(["CPIAUCSL", "UNRATE"]).issubset(df.columns)
    # Outputs validated (non-empty, datetime index, recent range)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df.dropna(how="all")) > 0
    # Most recent observation within the last 2 years (loose check)
    assert df.index.max().year >= pd.Timestamp.now().year - 2


@pytest.mark.integration
def test_fred_api_real_csv_fallback(monkeypatch):
    """
    Real call to FRED CSV endpoint (no API key) to validate fallback path.
    """
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")

    # Ensure JSON path is disabled
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    sys.path.insert(0, str(PROJECT_SRC))
    macro = __import__("analytics.phase3_macro", fromlist=["fetch_fred_series"])  # real module

    df = macro.fetch_fred_series(["CPIAUCSL"], start="2018-01-01")
    assert "CPIAUCSL" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df["CPIAUCSL"].dropna()) > 0


@pytest.mark.integration
def test_get_macro_features_real():
    """
    End-to-end: build macro features with real network (JSON or CSV depending on env).
    Validates structure and presence of core keys.
    """
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")

    sys.path.insert(0, str(PROJECT_SRC))
    macro = __import__("analytics.phase3_macro", fromlist=["get_macro_features"])  # real module

    feats = macro.get_macro_features()
    assert isinstance(feats, dict)
    # Either a top-level error (fatal) or a valid structure
    if "error" in feats:
        # When key is invalid or network blocked, surface error
        assert isinstance(feats["error"], str) and len(feats["error"]) > 0
    else:
        assert "macro_nowcast" in feats and isinstance(feats["macro_nowcast"], dict)
        sc = feats["macro_nowcast"].get("scores", {})
        assert set(["Growth", "Inflation", "Policy"]).issubset(set(sc.keys()))
        # Timestamp is a string date (YYYY-MM-DD) per implementation
        ts = feats.get("timestamp")
        assert isinstance(ts, str) and len(ts) >= 10


@pytest.mark.integration
def test_fred_api_key_from_config_real_json(monkeypatch):
    """
    Real JSON call without passing the key via env/CLI: the key is sourced
    from the local config (secrets_local.get_key). Ensures JSON path works
    even when FRED_API_KEY env var is absent.
    """
    if not os.getenv("AF_ALLOW_INTERNET"):
        pytest.skip("Set AF_ALLOW_INTERNET=1 to run network integration tests")

    # Ensure env var is NOT used
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    # Require that secrets_local provides a key
    try:
        import src.secrets_local as s  # prefer project-local
    except Exception:
        import secrets_local as s
    key = getattr(s, "FRED_API_KEY", None) or s.get_key("FRED_API_KEY")
    if not key:
        pytest.skip("secrets_local has no FRED_API_KEY; add it to run this test")

    # Disable CSV fallback so we truly exercise JSON path via config key
    import urllib.request as _url
    monkeypatch.setattr(_url, "urlopen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CSV disabled for test")))

    sys.path.insert(0, str(PROJECT_SRC))
    macro = __import__("analytics.phase3_macro", fromlist=["fetch_fred_series"])  # real module

    df = macro.fetch_fred_series(["CPIAUCSL"], start="2018-01-01")
    assert "CPIAUCSL" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df["CPIAUCSL"].dropna()) > 0

def test_fred_api_series_missing_is_nonfatal(monkeypatch):
    """
    With FRED_API_KEY set, a 400 'series does not exist' for one series should
    NOT crash fetch_fred_series; other series must load via JSON API.
    """
    import types
    import json
    import requests

    sys.path.insert(0, str(PROJECT_SRC))
    mod = __import__("analytics.phase3_macro", fromlist=["fetch_fred_series"])  # real module

    # Force a key so JSON path is used
    monkeypatch.setenv("FRED_API_KEY", "test_key_123")

    def fake_get(url, **kwargs):
        params = kwargs.get("params", {}) or {}
        sid = params.get("series_id", "")
        resp = requests.Response()
        if "api.stlouisfed.org" in url:
            if sid == "NAPM":
                resp.status_code = 400
                resp._content = b'{"error_code":400,"error_message":"Bad Request. The series does not exist."}'
            else:
                payload = {
                    "observations": [
                        {"date": "2019-01-01", "value": "100.0"},
                        {"date": "2020-01-01", "value": "101.0"},
                    ]
                }
                resp.status_code = 200
                resp._content = json.dumps(payload).encode("utf-8")
        else:
            resp.status_code = 404
            resp._content = b""
        resp.headers["Content-Type"] = "application/json"
        return resp

    monkeypatch.setattr("requests.get", fake_get)

    df = mod.fetch_fred_series(["NAPM", "CPIAUCSL"], start="2015-01-01")
    # The valid series should be present
    assert "CPIAUCSL" in df.columns
    assert len(df["CPIAUCSL"]) >= 1
    # The missing series can be absent or empty but must not crash
    assert ("NAPM" not in df.columns) or df["NAPM"].dropna().empty


def test_fred_api_invalid_key_is_fatal(monkeypatch):
    """
    With FRED_API_KEY set, an auth-like error should raise a RuntimeError.
    """
    import requests

    sys.path.insert(0, str(PROJECT_SRC))
    mod = __import__("analytics.phase3_macro", fromlist=["fetch_fred_series"])  # real module

    monkeypatch.setenv("FRED_API_KEY", "bad_key")

    def fake_get(url, **kwargs):
        resp = requests.Response()
        if "api.stlouisfed.org" in url:
            resp.status_code = 401
            resp._content = b'{"error_code":401,"error_message":"Invalid api key."}'
            resp.headers["Content-Type"] = "application/json"
            return resp
        resp.status_code = 404
        resp._content = b""
        return resp

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(RuntimeError):
        mod.fetch_fred_series(["CPIAUCSL"], start="2015-01-01")
