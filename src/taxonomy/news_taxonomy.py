"""
news_taxonomy.py
-----------------
Common taxonomies + lightweight taggers used by the news pipeline.

Design goals
- **Pure-Python, zero external deps** (regex + simple rules)
- Fast, language-agnostic best-effort (works on EN/FR/DE text)
- Safe to call on every article (title/summary/body)

Provides
- SECTOR_KEYWORDS: industry lexicons (banking, energy, defense, etc.)
- EVENT_PATTERNS: regexes to detect events (M&A, earnings, guidance, sanctions...)
- GEO_KEYWORDS: geopolitics taxonomy (Ukraine, Gaza, BRICS, NATO, tariffs, etc.)
- COMMODITY_KEYWORDS: crude, gas, gold, copper, wheat, etc.
- RISK_KEYWORDS: strikes, cyberattack, recall, antitrust, export ban...

- tag_sectors(text)
- classify_event(text)
- tag_geopolitics(text)
- tag_commodities(text)
- tag_risks(text)

Return format: sorted unique lowercase tags (list[str]).
"""
from __future__ import annotations
import re
from typing import Iterable, List, Dict, Tuple, Set

# --------- Utilities ---------

def _norm(t: str) -> str:
    return (t or "").lower()

_def_sep = r"[\s\-/_,.:;()\[\]""'’]+"


def _compile_words(words: Iterable[str]) -> re.Pattern:
    # word-boundary-ish for unicode languages
    safe = [re.escape(w.lower()) for w in words if w.strip()]
    if not safe:
        return re.compile(r"^$")  # never matches
    pattern = r"(?<!\w)(" + "|".join(safe) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


# --------- Taxonomies ---------

SECTOR_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    # Finance
    "banking": (
        "bank", "banque", "banking", "lender", "fintech", "mortgage", "credit union",
        "retail banking", "investment bank", "brokerage", "securities", "asset manager",
        "private equity", "hedge fund",
    ),
    "insurance": (
        "insurer", "assureur", "underwriting", "reinsurance", "premium", "claims",
        "life insurance", "p&c", "property and casualty",
    ),
    "payments": (
        "payments", "paiements", "wallet", "card network", "issuer", "acquirer",
        "pos", "interchange", "remittance", "wire transfer",
    ),

    # Energy & Materials
    "energy": (
        "oil", "crude", "brent", "wti", "natural gas", "lng", "refinery", "opec",
        "pipeline", "offshore", "renewables", "wind farm", "solar", "pv", "battery",
        "storage", "grid", "nuclear", "uranium", "hydrogen",
    ),
    "metals_mining": (
        "copper", "iron ore", "nickel", "lithium", "cobalt", "gold", "silver",
        "aluminum", "bauxite", "smelter", "mine",
    ),
    "chemicals": (
        "petrochemical", "chemical", "polymer", "fertilizer", "ammonia", "methanol",
    ),

    # Industrials & Transport
    "aerospace_defense": (
        "aerospace", "defense", "defence", "rafale", "fighter jet", "missile", "nato",
        "shipyard", "frigate", "drone", "airbus", "boeing", "dassault",
    ),
    "autos": (
        "automaker", "ev", "electric vehicle", "battery plant", "dealership", "zev",
        "autonomous driving", "charging", "gigafactory",
    ),
    "logistics": (
        "port", "terminal", "freight", "shipping", "container", "rail", "air cargo",
        "3pl", "warehouse", "last mile",
    ),
    "construction": (
        "construction", "builder", "cement", "infrastructure", "contractor", "epc",
    ),

    # TMT
    "semiconductors": (
        "foundry", "node", "fab", "wafer", "chip", "gpu", "asic", "eda", "lithography",
        "euV", "arm soc", "risc-v",
    ),
    "software_cloud": (
        "saas", "cloud", "sovereign cloud", "openstack", "kubernetes", "hyperscaler",
        "data center", "cdn", "mlops", "microservices",
    ),
    "media_internet": (
        "streaming", "social network", "ads", "adtech", "subscription", "newsroom",
    ),
    "telecom": (
        "telecom", "5g", "fiber", "ftth", "spectrum", "carrier", "mvno", "satellite",
    ),

    # Healthcare & Consumer
    "healthcare": (
        "pharma", "drug", "vaccine", "clinical trial", "fda", "ema", "insulin",
        "hospital", "clinic", "medtech", "device", "cashless insurance",
    ),
    "consumer": (
        "retail", "e-commerce", "supermarket", "apparel", "luxury", "beverage",
    ),
    "agri_food": (
        "agri", "farmer", "crop", "wheat", "corn", "soy", "fertilizer", "gst on bakery",
    ),

    # Real Assets
    "real_estate": (
        "reit", "invits", "leasing", "occupancy", "cap rate", "development",
    ),
}

# Pre-compile sector regexes
_SECTOR_PATTERNS: Dict[str, re.Pattern] = {
    sector: _compile_words(words) for sector, words in SECTOR_KEYWORDS.items()
}

EVENT_PATTERNS: Dict[str, re.Pattern] = {
    # Corporate
    "earnings": re.compile(r"\b(q\d|quarter|earnings|eps|revenue|guidance|outlook)\b", re.I),
    "mna": re.compile(r"\b(m&a|merger|acquisition|acquires?|takeover|buyout|stake)\b", re.I),
    "divestiture": re.compile(r"\b(divest|spinoff|spin-off|carve[- ]?out|asset sale)\b", re.I),
    "buyback": re.compile(r"\b(buyback|repurchase)\b", re.I),
    "capital_raise": re.compile(r"\b(secondary offering|rights issue|convertible|bond issue)\b", re.I),
    "guidance": re.compile(r"\b(guidance|outlook|forecast|raises?|cuts?|revis(es|ion))\b", re.I),
    "downgrade": re.compile(r"\b(downgraded?|underperform|sell rating)\b", re.I),
    "upgrade": re.compile(r"\b(upgraded?|overweight|buy rating)\b", re.I),

    # Policy / legal / macro
    "sanctions": re.compile(r"\b(sanction|embargo|tariff|export ban|price cap)\b", re.I),
    "regulation": re.compile(r"\b(regulation|antitrust|anticoncurrentiel|competition authority)\b", re.I),
    "litigation": re.compile(r"\b(lawsuit|injunction|class action|probe|investigation)\b", re.I),
    "labor": re.compile(r"\b(strike|walkout|union|collective bargaining|layoffs?)\b", re.I),
    "cyber": re.compile(r"\b(cyberattack|ransomware|data breach)\b", re.I),
}

GEO_KEYWORDS: Tuple[str, ...] = (
    # Conflicts / blocs / alliances
    "ukraine", "russia", "poland", "belarus", "nato", "eu",
    "gaza", "israel", "qatar", "iran", "saudi", "yemen",
    "brics", "tariff", "sanctions", "trade war", "export control",
    "south china sea", "taiwan",
)
_GEO_PATTERN = _compile_words(GEO_KEYWORDS)

COMMODITY_KEYWORDS: Tuple[str, ...] = (
    "brent", "wti", "oil", "crude", "diesel", "gasoline", "natural gas", "lng",
    "coal", "uranium", "gold", "silver", "copper", "nickel", "lithium", "cobalt",
    "wheat", "corn", "soy", "sugar",
)
_COMMO_PATTERN = _compile_words(COMMODITY_KEYWORDS)

RISK_KEYWORDS: Tuple[str, ...] = (
    "strike", "cyberattack", "data breach", "recall", "explosion", "shutdown",
    "fire", "accident", "boycott", "ban", "earthquake", "flood",
)
_RISK_PATTERN = _compile_words(RISK_KEYWORDS)


# --------- Public API ---------

def tag_sectors(text: str) -> List[str]:
    t = _norm(text)
    hits: Set[str] = set()
    for sector, pat in _SECTOR_PATTERNS.items():
        if pat.search(t):
            hits.add(sector)
    return sorted(hits)


def classify_event(text: str) -> List[str]:
    t = _norm(text)
    tags = [name for name, pat in EVENT_PATTERNS.items() if pat.search(t)]
    # Prioritization: earnings/guidance often overlap
    if "guidance" in tags and "earnings" not in tags and re.search(r"\b(q\d|quarter)\b", t):
        tags.append("earnings")
    return sorted(set(tags))


def tag_geopolitics(text: str) -> List[str]:
    t = _norm(text)
    return sorted(set(m.group(0).lower() for m in _GEO_PATTERN.finditer(t)))


def tag_commodities(text: str) -> List[str]:
    t = _norm(text)
    return sorted(set(m.group(0).lower() for m in _COMMO_PATTERN.finditer(t)))


def tag_risks(text: str) -> List[str]:
    t = _norm(text)
    return sorted(set(m.group(0).lower() for m in _RISK_PATTERN.finditer(t)))


# --------- Simple smoke test ---------
if __name__ == "__main__":
    sample = (
        "Poland invokes NATO Article 4 after drone strikes; Brent jumps;"
        " Oracle ups guidance while EU mulls export controls on chips."
    )
    print("sectors:", tag_sectors(sample))
    print("events:", classify_event(sample))
    print("geo:", tag_geopolitics(sample))
    print("commodities:", tag_commodities(sample))
    print("risks:", tag_risks(sample))
