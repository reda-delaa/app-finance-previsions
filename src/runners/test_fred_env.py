#!/usr/bin/env python3
"""
Quick test for FRED API key availability
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from secrets_local import get_key
    print("✅ secrets_local imported successfully")

    # Test direct environment variable
    env_key = os.getenv("FRED_API_KEY")
    print(f"Environment variable FRED_API_KEY: {'✅ Found' if env_key else '❌ Missing'}")
    if env_key:
        print(f"Key length: {len(env_key)} (32 chars expected for FRED)")

    # Test via get_key function
    secret_key = get_key("FRED_API_KEY")
    print(f"secrets_local.get_key('FRED_API_KEY'): {'✅ Found' if secret_key else '❌ Missing'}")
    if secret_key:
        print(f"Key length: {len(secret_key)} (32 chars expected for FRED)")

    # Check if both match
    if env_key and secret_key:
        if env_key == secret_key:
            print("✅ Environment variable and secrets_local match")
        else:
            print("⚠️ Environment variable and secrets_local differ")

    if secret_key or env_key:
        print("✅ FRED API key is available to the project")
        sys.exit(0)
    else:
        print("❌ FRED API key is not available")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Failed to import secrets_local: {e}")
    sys.exit(1)

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
