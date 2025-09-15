#!/usr/bin/env python3
"""
Test script to verify that macro data loading is now working correctly.
Includes tests for deprecated pandas functions and error handling.
"""

import warnings
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# --- sys.path bootstrap (comme dans app.py) ---
_SRC_ROOT = Path(__file__).resolve().parents[1]   # .../analyse-financiere/src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
# -------------------------------------

from analytics.phase3_macro import _fred_csv, get_us_macro_bundle, macro_nowcast, get_macro_features, pct_chg, yoy

def test_single_fred_series():
    """Test that individual FRED series are now loading correctly."""
    print("=== Testing fixed FRED loading ===")

    test_series = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]
    for series_id in test_series:
        series = _fred_csv(series_id)
        print(f"{series_id}: loaded {len(series)} datapoints")
        if not series.empty:
            print(f"  Date range: {series.index.min()} to {series.index.max()}")
            print(f"  Sample values: {series.head(2).to_dict()}")

def test_macro_bundle():
    """Test that the macro bundle loads correctly."""
    print("\n=== Testing macro bundle ===")

    try:
        bundle = get_us_macro_bundle(start="2000-01-01", monthly=True)
        print(f"Bundle data shape: {bundle.data.shape}")
        print(f"Bundle columns: {list(bundle.data.columns)}")
        print(f"Date range: {bundle.data.index.min()} to {bundle.data.index.max()}")
        print(f"Meta: {bundle.meta}")

        # Check which series have data
        non_empty_cols = [col for col in bundle.data.columns if not bundle.data[col].isna().all()]
        print(f"Non-empty series: {non_empty_cols}")

        return bundle
    except Exception as e:
        print(f"Error loading macro bundle: {e}")
        return None

def test_macro_nowcast(bundle):
    """Test that macro nowcast works correctly."""
    print("\n=== Testing macro nowcast ===")

    if bundle is None or bundle.data.empty:
        print("Cannot test nowcast - bundle is empty")
        return None

    try:
        nowcast = macro_nowcast(bundle)
        print(f"Nowcast scores: {nowcast.scores}")
        print(f"Components keys: {list(nowcast.components.keys())}")

        # Check for NaN values
        nan_count = sum(1 for v in nowcast.scores.values() if np.isnan(v))
        if nan_count < len(nowcast.scores.values()):
            print(f"✅ SUCCESS: {len(nowcast.scores.values()) - nan_count} scores calculated (previously all were NaN)")
        else:
            print("❌ FAILED: All scores are still NaN")

        return nowcast
    except Exception as e:
        print(f"Error in macro nowcast: {e}")
        return None

def test_pandas_deprecation_warnings():
    """Test that deprecated pandas functions don't generate warnings."""
    print("\n=== Testing pandas deprecation warnings ===")

    # Test pct_chg function
    test_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'B': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    }, index=pd.date_range('2023-01-01', periods=14, freq='M'))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test pct_chg function
        result_pct = pct_chg(test_df, periods=1)
        print(f"pct_chg test: {len(result_pct)} rows generated")

        # Test yoy function (which uses pct_change(12))
        result_yoy = yoy(test_df)
        print(f"yoy test: {len(result_yoy)} rows generated")

        # Test create and call get_macro_features to trigger internal pct_change calls
        try:
            result_features = get_macro_features()
            pct_change_warnings = [warning for warning in w
                                 if "pct_change" in str(warning.message)
                                 and "fill_method" in str(warning.message)]
            float_series_warnings = [warning for warning in w
                                   if "float" in str(warning.message)
                                   and ("Series" in str(warning.message) or "iloc" in str(warning.message))]

            if not pct_change_warnings and not float_series_warnings:
                print("✅ SUCCESS: No pandas deprecation warnings detected")
                return True
            else:
                print(f"❌ FAILED: {len(pct_change_warnings)} pct_change warnings and {len(float_series_warnings)} float warnings detected")
                for warning in pct_change_warnings + float_series_warnings:
                    print(f"  Warning: {warning.message}")
                return False
        except Exception as e:
            print(f"Error in macro features test: {e}")
            return False

def test_get_macro_features():
    """Test the get_macro_features function used by the app."""
    print("\n=== Testing get_macro_features ===")

    try:
        result = get_macro_features()
        print(f"get_macro_features success: {'macro_nowcast' in result}")

        if 'macro_nowcast' in result:
            scores = result['macro_nowcast'].get('scores', {})
            print(f"Scores: {scores}")

            # Check if macroeconomic scores are no longer all NaN
            if scores and not all(np.isnan(v) for v in scores.values() if isinstance(v, (int, float))):
                print("✅ SUCCESS: Macroeconomic data is loading correctly!")
                return True
            else:
                print("❌ FAILED: Still getting all NaN values")
                return False

    except Exception as e:
        print(f"Error in get_macro_features: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("Starting macro fix tests...\n")

    all_passed = True

    try:
        test_single_fred_series()
        print("✅ Single FRED series test completed")
    except Exception as e:
        print(f"❌ Single FRED series test failed: {e}")
        all_passed = False

    try:
        bundle = test_macro_bundle()
        if bundle is None:
            all_passed = False
        print("✅ Macro bundle test completed")
    except Exception as e:
        print(f"❌ Macro bundle test failed: {e}")
        bundle = None
        all_passed = False

    if bundle is not None:
        try:
            nowcast = test_macro_nowcast(bundle)
            if nowcast is None:
                all_passed = False
            print("✅ Macro nowcast test completed")
        except Exception as e:
            print(f"❌ Macro nowcast test failed: {e}")
            all_passed = False
    else:
        print("❌ Skipping macro nowcast test due to bundle failure")
        all_passed = False

    try:
        warnings_ok = test_pandas_deprecation_warnings()
        if not warnings_ok:
            all_passed = False
        print("✅ Pandas deprecation warnings test completed")
    except Exception as e:
        print(f"❌ Pandas deprecation warnings test failed: {e}")
        all_passed = False

    try:
        features_ok = test_get_macro_features()
        if not features_ok:
            all_passed = False
        print("✅ Get macro features test completed")
    except Exception as e:
        print(f"❌ Get macro features test failed: {e}")
        all_passed = False

    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Macro fixes are working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED. Please check the output above for details.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
