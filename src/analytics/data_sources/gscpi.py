"""
GSCPI (Global Supply Chain Pressure Index) data source with multiple mirror URLs.
Robust fetching with fallback mechanisms.
"""
import pandas as pd
from utils import warn_once

# Mirror URLs for GSCPI data
URLS = [
    # Official NY Fed mirror (frequently updated)
    "https://www.newyorkfed.org/medialibrary/research/gscpi/files/gscpi_data.csv",
    # Community mirrors
    "https://raw.githubusercontent.com/QUANTAXIS/QAData/main/gscpi_data.csv",
]

def fetch_gscpi():
    """
    Fetch GSCPI data from multiple sources with fallback.

    Returns:
        pd.DataFrame: GSCPI data with date index, or None if all sources fail
    """
    import pandas as pd
    import requests

    last_err = None
    for url in URLS:
        try:
            df = pd.read_csv(url)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            return df
        except Exception as e:
            last_err = e
            continue
    warn_once("GSCPI_404", f"GSCPI failed on all URLs, returning None. Last error: {last_err}")
    return None

# Export for import compatibility
__all__ = ['fetch_gscpi']
