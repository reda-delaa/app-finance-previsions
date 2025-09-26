"""
GPR (Geopolitical Risk Index) data source with multiple mirror URLs.
Robust fetching with network tolerance and fallbacks.
"""
import io
from ..warn_log import warn_once

# Mirror URLs for GPR data
URLS = [
    "https://www2.bc.edu/matteo-iacoviello/gpr_files/GPRD.csv",
    "https://raw.githubusercontent.com/pmorissette/gpr-data/main/GPRD.csv",
]

def fetch_gpr():
    """
    Fetch GPR data from multiple sources with network tolerance and fallbacks.

    Returns:
        pd.DataFrame: GPR data with date index, or None if all sources fail
    """
    import pandas as pd
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (GPRFetcher/1.0)"

    for url in URLS:
        try:
            r = s.get(url, timeout=8)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))

            # Normalization - handle different date column names
            date_columns = ("DATE", "Date", "date")
            for c in date_columns:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c])
                    df = df.set_index(c).sort_index()
                    return df
        except Exception:
            continue
    warn_once("GPR_FAIL", "GPR unavailable — skipping series.")
    return None

# Export for import compatibility
__all__ = ['fetch_gpr']
