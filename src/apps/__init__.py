"""Web applications package for financial analysis visualization.

Re-exports select helpers so tests can import via `__import__('apps.macro_sector_app')`
without using fromlist.
"""

try:
    from .macro_sector_app import fetch_gscpi, fetch_gpr, get_multi_yf, load_fred_series  # noqa: F401
except Exception:
    # optional in bare environments
    pass

try:
    from .stock_analysis_app import get_stock_data, add_technical_indicators  # noqa: F401
except Exception:
    pass
