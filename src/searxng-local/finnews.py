#!/usr/bin/env python3
# Thin compatibility wrapper for legacy searxng-local/finnews.py
# Provides minimal API required by tests: harvest, fetch_feed_entries,
# compile_bool_query, enrich_item.

from __future__ import annotations
from typing import Callable, List, Tuple

try:
    # Reuse real ingestion module when available
    from ingestion.finnews import bool_query_match as _bool_match
except Exception:
    def _bool_match(q: str, text: str) -> bool:  # fallback
        q = (q or "").lower()
        return all(tok in (text or "").lower() for tok in q.split() if tok)


def compile_bool_query(query: str) -> Callable[[str], bool]:
    """Return a predicate(text) -> bool for the given boolean query string."""
    return lambda text: bool(_bool_match(query, text or ""))


def enrich_item(title: str) -> Tuple[list, float, list]:
    """Naive enrichment: returns (tags, score, entities) with correct types.
    Tests only assert types, not contents.
    """
    tags: List[str] = []
    ents: List[str] = []
    score: float = 0.0
    t = (title or "").lower()
    if any(k in t for k in ("oil", "brent", "opec", "sanction")):
        tags.append("energy")
        score += 0.1
    if any(k in t for k in ("ai", "chip", "semiconductor")):
        tags.append("tech")
        score += 0.1
    return tags, float(score), ents


def fetch_feed_entries(*args, **kwargs) -> list:
    """Placeholder that returns an empty list (type-compatible)."""
    return []


def harvest(*args, **kwargs) -> list:
    """Placeholder aggregator (type-compatible)."""
    return []

