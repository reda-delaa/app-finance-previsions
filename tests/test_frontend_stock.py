import sys
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"


@pytest.fixture(autouse=True)
def ensure_src_on_path():
    p = str(PROJECT_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
    yield


def test_render_stock_exists_and_runs():
    import apps.stock_analysis_app as stock
    assert hasattr(stock, "render_stock")
    stock.render_stock(default_ticker="AAPL")

