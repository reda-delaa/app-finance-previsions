import sys
import types
import pathlib
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"


@pytest.fixture(autouse=True)
def ensure_src_on_path():
    p = str(PROJECT_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
    yield


def _df(series_id: str):
    idx = pd.date_range("2020-01-01", periods=3, freq="M")
    return pd.DataFrame({series_id: [1.0, 2.0, 3.0]}, index=idx)


def test_render_macro_runs_with_stubs(monkeypatch):
    import apps.macro_sector_app as macro

    # Stubs: éviter le réseau
    monkeypatch.setattr(macro, "load_fred_series", lambda s, fred_key=None, start=None: _df(s))
    monkeypatch.setattr(macro, "get_multi_yf", lambda symbols, start="2010-01-01": pd.DataFrame(
        {sym: np.linspace(100, 110, 10) for sym in symbols}, index=pd.date_range("2021-01-01", periods=10)
    ))
    monkeypatch.setattr(macro, "fetch_gscpi", lambda: pd.DataFrame({"GSCPI": [0.0, 0.1]}, index=pd.date_range("2021-01-01", periods=2)))
    monkeypatch.setattr(macro, "fetch_gpr", lambda: pd.DataFrame({"GPR": [100, 90]}, index=pd.date_range("2021-01-01", periods=2)))
    monkeypatch.setattr(macro, "fetch_vix_history", lambda: pd.Series([15.0, 16.0], index=pd.date_range("2021-01-01", periods=2), name="VIX"))

    # Appel — avec st_compat, st est un no-op; on vérifie juste l'absence d'exception
    macro.render_macro()

