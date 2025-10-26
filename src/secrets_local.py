# secrets_local.py
# Define local API keys here for development only.
# This file is read by some modules directly (import src.secrets_local)
# and will also set environment variables consumed by `src.core.config.Config`.
# Do NOT commit this file to version control.

# Example placeholders (replace with your real keys):
# ALPHA_VANTAGE_KEY = "AV_XXXXXXXX"
# YAHOO_API_KEY = "YF_XXXXXXXX"
# FINNHUB_API_KEY = "finnhub_xxx"
# FIRECRAWL_API_KEY = "fc_xxx"

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file BEFORE using them
load_dotenv()

# --- Local keys (read from environment only; do not hardcode real secrets here) ---
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Optional keys
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
YAHOO_API_KEY = os.getenv("YAHOO_API_KEY")
OPERATIVE_API_KEY = os.getenv("OPERATIVE_API_KEY")

# --- Export to environment for modules that read os.environ / dotenv ---
# Only export when a key is actually set to avoid leaking placeholders.
if FIRECRAWL_API_KEY:
    os.environ.setdefault("FIRECRAWL_API_KEY", str(FIRECRAWL_API_KEY))
if SERPER_API_KEY:
    os.environ.setdefault("SERPER_API_KEY", str(SERPER_API_KEY))
if TAVILY_API_KEY:
    os.environ.setdefault("TAVILY_API_KEY", str(TAVILY_API_KEY))
if FINNHUB_API_KEY:
    os.environ.setdefault("FINNHUB_API_KEY", str(FINNHUB_API_KEY))
    os.environ.setdefault("FINNHUB_KEY", str(FINNHUB_API_KEY))
if ALPHA_VANTAGE_KEY:
    os.environ.setdefault("ALPHA_VANTAGE_KEY", str(ALPHA_VANTAGE_KEY))
if YAHOO_API_KEY:
    os.environ.setdefault("YAHOO_API_KEY", str(YAHOO_API_KEY))
if OPERATIVE_API_KEY:
    os.environ.setdefault("OPERATIVE_API_KEY", str(OPERATIVE_API_KEY))


def get_key(name: str) -> Optional[str]:
	"""Return an API key by checking environment variables first,
	then this module's attributes as a fallback.
	"""
	val = os.getenv(name)
	if val:
		return val
	# common alias mappings
	aliases = {
		"FINNHUB_KEY": "FINNHUB_API_KEY",
		"ALPHA_VANTAGE_KEY": "ALPHA_VANTAGE_KEY",
		"YAHOO_API_KEY": "YAHOO_API_KEY",
		"FIRECRAWL_API_KEY": "FIRECRAWL_API_KEY",
		"FRED_API_KEY": "FRED_API_KEY",
	}
	attr = aliases.get(name, name)
	return getattr(__import__(__name__), attr, None)

# End of secrets_local.py
