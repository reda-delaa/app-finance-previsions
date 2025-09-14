"""Core financial market data utilities."""

from typing import Optional


def fetch_price_history(ticker: str, start_date: str, end_date: str) -> Optional["pd.DataFrame"]:
    """
    Fetch historical price data for a given ticker.
    Returns OHLCV DataFrame or None if not found.
    """
    try:
        import pandas as pd
        import yfinance as yf

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df if not df.empty else None
    except Exception:
        return None
