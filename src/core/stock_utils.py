"""
Simple stock utilities for ticker mapping and validation.
"""

from typing import Optional
import pandas as pd

def guess_ticker(company_name: str) -> str:
    """
    Simple heuristic to guess ticker from company name.
    This is a basic implementation - in production you'd want a proper mapping database.
    """
    if not company_name:
        return ""

    # Remove common words and clean up
    name = company_name.lower().strip()
    name = name.replace("corporation", "").replace("corp", "").replace("inc", "").replace("ltd", "").replace("limited", "").strip()

    # Simple mappings for common companies
    mappings = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "google": "GOOGL",
        "facebook": "META",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "netflix": "NFLX",
        "alphabet": "GOOGL",
        "meta": "META",
        "twitter": "TWTR",
        "uber": "UBER",
        "lyft": "LYFT",
        "airbnb": "ABNB",
        "spotify": "SPOT",
        "zoom": "ZM",
        "shopify": "SHOP",
        "square": "SQ",
        "paypal": "PYPL",
        "datadog": "DDOG",
        "snowflake": "SNOW",
        "mongodb": "MDB",
        "atlassian": "TEAM",
        "crowdstrike": "CRWD",
        "okta": "OKTA",
        "twilio": "TWLO",
        "cisco": "CSCO",
        "salesforce": "CRM",
        "oracle": "ORCL",
        "adobe": "ADBE",
        "intuit": "INTU",
        "workday": "WDAY",
        "etsy": "ETSY",
        "ebay": "EBAY",
        "walmart": "WMT",
        "target": "TGT",
        "costco": "COST",
        "home depot": "HD",
        "mcdonalds": "MCD",
        "starbucks": "SBUX",
        "chipotle": "CMG",
        "dominos": "DPZ",
        "cocacola": "KO",
        "pepsi": "PEP",
        "pfizer": "PFE",
        "johnson johnson": "JNJ",
        "merck": "MRK",
        "exxon mobil": "XOM",
        "chevron": "CVX",
        "conocophillips": "COP",
        "occidental": "OXY",
        "marathon oil": "MRO",
        "eog resources": "EOG",
        "pioneer natural": "PXD",
        "devon energy": "DVN",
        "cabot oil": "CTRA",
        "diamondback": "FANG",
        "coterra": "CTRA",
        "antero": "AR",
        "eQT": "EQT",
        "cabot": "CTRA",
        "range resources": "RRC",
        "southwestern energy": "SWN",
        "cheniere": "LNG",
        "williams": "WMB",
        "kinder morgan": "KMI",
        "enterprise products": "EPD",
        "magna": "MGA",
        "magellan midstream": "MMP",
        "equitrans": "ETRN",
        "targa resources": "TRGP",
        "oneok": "OKE",
        "antero midstream": "AM",
        "semgroup": "SEMG",
        "enable midstream": "ENBL",
        "crestwood equity": "CEQP",
        "nuestar energy": "NS",
        "altus midstream": "ALTM",
        "pembina pipeline": "PBA",
        "inter pipeline": "IPPLF",
        "tc energy": "TRP",
        "enbridge": "ENB",
    }

    # Return ticker if found, otherwise return empty string
    return mappings.get(name, "")


def fetch_price_history(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data for a given ticker via yfinance.
    Returns OHLCV DataFrame or None if not found.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        return df if not df.empty else None
    except Exception:
        return None
