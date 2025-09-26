#!/usr/bin/env python3
"""FRED API key availability — usable as PyTest and as a script."""

import os
import sys
import pytest


def _get_key(name: str):
    try:
        from secrets_local import get_key as _gk  # type: ignore
        return _gk(name)
    except Exception:
        return os.getenv(name)


@pytest.mark.integration
def test_fred_api_key_available():
    env_key = os.getenv("FRED_API_KEY")
    secret_key = _get_key("FRED_API_KEY")
    assert env_key or secret_key, "FRED_API_KEY is not available"


if __name__ == "__main__":
    key = _get_key("FRED_API_KEY")
    print("✅ FRED_API_KEY available" if key else "❌ FRED_API_KEY missing")
    sys.exit(0 if key else 1)
