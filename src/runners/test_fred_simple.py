#!/usr/bin/env python3
"""
Simple test script to verify FRED API functionality
"""

import requests
import pandas as pd
import os
from io import StringIO

def test_fred_csv_url(series_id):
    """Test fetching FRED data using the CSV endpoint"""
    print(f"\n=== Testing FRED CSV endpoint for {series_id} ===")

    # Build URL
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"URL: {url}")

    try:
        # Make request
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Content Type: {response.headers.get('content-type', 'N/A')}")

        if response.status_code == 200:
            print("✅ Request successful")

            # Check if content looks like CSV
            content_preview = response.text[:500]
            if response.text.strip().startswith("DATE") or response.text.strip().startswith("observation_date"):
                print("✅ Response appears to be CSV formatted")

                # Try to parse as CSV
                try:
                    df = pd.read_csv(StringIO(response.text))
                    if not df.empty:
                        print(f"✅ Successfully parsed CSV with shape: {df.shape}")
                        print(f"✅ Date range: {df['observation_date' if 'observation_date' in df.columns else 'DATE'].min()} to {df['observation_date' if 'observation_date' in df.columns else 'DATE'].max()}")
                        return True
                    else:
                        print("❌ Parsed dataframe is empty")
                        return False
                except Exception as e:
                    print(f"❌ Failed to parse CSV: {e}")
                    return False
            else:
                print("❌ Response does not appear to be CSV formatted")
                print(f"Content preview: {content_preview[:200]}...")
                return False
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Error during request: {e}")
        return False

def test_fred_api_url(series_id, api_key):
    """Test fetching FRED data using the JSON API endpoint"""
    print(f"\n=== Testing FRED API endpoint for {series_id} ===")

    # Build API URL
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json"
    }
    print(f"API URL: {base_url}")
    print(f"API Key: {api_key[:8]}...")  # Show only first 8 chars for security

    try:
        # Make request
        response = requests.get(base_url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Content Type: {response.headers.get('content-type', 'N/A')}")

        if response.status_code == 200:
            print("✅ API request successful")
            try:
                data = response.json()
                if 'observations' in data and data['observations']:
                    observations = data['observations']
                    valid_obs = [o for o in observations if o.get('value')]
                    if valid_obs:
                        print(f"✅ Found {len(valid_obs)} valid observations")
                        # Show recent value
                        recent_obs = valid_obs[-1]
                        print(f"✅ Most recent value: {recent_obs.get('value')} on {recent_obs.get('date')}")
                        return True
                    else:
                        print("❌ No valid observations found")
                        return False
                else:
                    print("❌ Invalid response format - missing observations")
                    return False
            except Exception as e:
                print(f"❌ Failed to parse JSON response: {e}")
                return False
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Error during API request: {e}")
        return False

if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("❌ FRED_API_KEY not found in environment variables")
        exit(1)

    print("🚀 Starting FRED API Tests...")
    print(f"Using API key: {api_key[:8]}...")

    # Test series
    test_series = ["CPIAUCSL", "UNRATE", "INDPRO"]

    # Test CSV endpoint (no API key needed)
    print("\n" + "="*50)
    print("TESTING CSV ENDPOINT (No API key required)")
    print("="*50)
    csv_results = []
    for series in test_series:
        result = test_fred_csv_url(series)
        csv_results.append(result)

    # Test API endpoint (API key required)
    print("\n" + "="*50)
    print("TESTING API ENDPOINT (API key required)")
    print("="*50)
    api_results = []
    for series in test_series:
        result = test_fred_api_url(series, api_key)
        api_results.append(result)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"CSV Tests: {sum(csv_results)}/{len(csv_results)} successful")
    print(f"API Tests: {sum(api_results)}/{len(api_results)} successful")

    if all(csv_results):
        print("✅ CSV endpoint is working correctly")
    else:
        print("❌ CSV endpoint has issues")

    if all(api_results):
        print("✅ API endpoint is working correctly")
    else:
        print("❌ API endpoint has issues")
