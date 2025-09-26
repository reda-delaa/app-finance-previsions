#!/usr/bin/env python3
"""
Integration test to verify FRED functions work in the project context
"""

import os
import sys

# Ensure we can import the project modules
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that we can import all FRED-related modules"""
    print("=== Testing Imports ===")

    try:
        # Test core FRED functionality
        from src.ingestion.macro_derivatives_client import fred_series
        print("✅ macro_derivatives_client.fred_series imported")

        # Test apps that use FRED
        from src.apps.macro_sector_app import load_fred_series
        print("✅ macro_sector_app.load_fred_series imported")

        # Test phase3_macro FRED functions
        from src.analytics.phase3_macro import _fred_csv, fetch_fred_series
        print("✅ phase3_macro FRED functions imported")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_fred_function():
    """Test a simple FRED function call"""
    print("\n=== Testing FRED Function Call ===")

    try:
        from src.secrets_local import get_key
        from src.ingestion.macro_derivatives_client import fred_series

        # Get API key
        api_key = get_key("FRED_API_KEY")
        if not api_key:
            print("❌ No FRED API key available")
            return False

        print(f"✅ Using FRED API key: {api_key[:8]}...")

        # Test a simple series request
        print("Testing fred_series function with CPIAUCSL...")
        result = fred_series(["CPIAUCSL"])

        if "CPIAUCSL" not in result:
            print("❌ No CPIAUCSL data returned")
            return False

        data = result["CPIAUCSL"]
        if not data:
            print("❌ CPIAUCSL data is empty")
            return False

        # Check that we have recent data
        if len(data) < 10:
            print("⚠️ Warning: Limited data returned")
        else:
            print(f"✅ Successfully retrieved {len(data)} observations")

        # Check most recent date
        if data and data[-1].get('date'):
            print(f"✅ Most recent data: {data[-1]['value']} on {data[-1]['date']}")

        return True

    except Exception as e:
        print(f"❌ Function test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting FRED Integration Tests...\n")

    results = []

    # Test imports
    results.append(test_imports())

    # Test FRED function
    results.append(test_fred_function())

    # Report results
    print("\n=== Test Results ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("🎉 All FRED integration tests passed!")
        return 0
    else:
        print("❌ Some FRED integration tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
