#!/usr/bin/env python3
"""
Debug script to test FRED API calls and identify data retrieval issues.
"""

import requests
import pandas as pd
from src.analytics.phase3_macro import _fred_csv

def test_single_fred_series(series_id):
    """Test fetching a single FRED series"""
    print(f"\n=== Testing {series_id} ===")

    try:
        # Test direct URL approach
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        print(f"URL: {url}")

        response = requests.get(url, timeout=20)
        print(f"Status Code: {response.status_code}")
        print(f"Content Type: {response.headers.get('content-type', 'N/A')}")

        if response.status_code == 200:
            # Check if content looks like CSV
            content_preview = response.text[:500]
            print(f"Content Preview:\n{content_preview[:200]}...")

            # Try to parse as CSV
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                print(f"Successfully parsed CSV with columns: {list(df.columns)}")
                print(f"Data shape: {df.shape}")
                if not df.empty:
                    print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
            except Exception as e:
                print(f"Failed to parse CSV: {e}")
        else:
            print(f"Request failed with status code: {response.status_code}")

        # Test the actual function
        print("\n--- Testing _fred_csv function ---")
        series = _fred_csv(series_id)
        print(f"Series length: {len(series)}")
        print(f"Series is empty: {series.empty}")
        if not series.empty:
            print(f"Date range: {series.index.min()} to {series.index.max()}")
            print(f"Sample values: {series.head(3).to_dict()}")

    except Exception as e:
        print(f"Error testing {series_id}: {e}")

if __name__ == "__main__":
    # Test the key series from get_us_macro_bundle
    test_series = [
        "INDPRO",
        "PAYEMS",
        "RSAFS",
        "NAPM",
        "CPIAUCSL",
        "CPILFESL",
        "T10YIE",
        "FEDFUNDS",
        "DGS10",
        "DGS2",
        "DTWEXBGS",
        "NFCI"
    ]

    for series_id in test_series:
        test_single_fred_series(series_id)
