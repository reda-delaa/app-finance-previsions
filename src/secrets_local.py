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

# --- Local keys (edit these) ---
FIRECRAWL_API_KEY = "fc-48a370c2f5874d4ab418adb2257d0cf5"
SERPER_API_KEY = "133fc32fa88cd9413a2f8286009eb40a5d7e93b2"
TAVILY_API_KEY = "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"
FINNHUB_API_KEY = "d31j44pr01qsprr18im0d31j44pr01qsprr18img"
FRED_API_KEY = "63bcdd7052a9d5f2339d2a631b4f1f5a"



# Optional keys you can add when available
ALPHA_VANTAGE_KEY = None
YAHOO_API_KEY = None

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
	# keep compatibility with older env var name
	os.environ.setdefault("FINNHUB_KEY", str(FINNHUB_API_KEY))
if ALPHA_VANTAGE_KEY:
	os.environ.setdefault("ALPHA_VANTAGE_KEY", str(ALPHA_VANTAGE_KEY))
if YAHOO_API_KEY:
	os.environ.setdefault("YAHOO_API_KEY", str(YAHOO_API_KEY))


def get_key(name: str) -> str | None:
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