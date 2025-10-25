"""
News aggregation and summarization helpers.

Collects normalized news via analytics.market_intel.collect_news and produces
LLM-assisted summaries when econ_llm_agent is available, with safe fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from analytics.market_intel import collect_news


def aggregate_news(
    regions: List[str],
    window: str = "last_week",
    query: str = "",
    company: Optional[str] = None,
    tgt_ticker: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    items, meta = collect_news(
        regions=regions,
        window=window,
        query=query,
        company=company,
        aliases=None,
        tgt_ticker=tgt_ticker,
        per_source_cap=None,
        limit=limit,
    )
    return {"news": items, "meta": meta}


def summarize_news(news_items: List[Dict[str, Any]], locale: str = "fr-FR") -> Dict[str, Any]:
    """LLM summary via econ_llm_agent when available, otherwise simple counts.

    Returns: { "ok": bool, "text": str, "json": dict|None }
    """
    try:
        from analytics.econ_llm_agent import EconomicAnalyst, EconomicInput
    except Exception:
        # fallback: basic metrics
        return {
            "ok": True,
            "text": f"{len(news_items)} articles agrégés. (LLM indisponible)",
            "json": None,
        }

    agent = EconomicAnalyst()
    data = EconomicInput(
        question="Synthétise les thèmes, risques et opportunités clés dans ces nouvelles (≤200 mots).",
        features=None,
        news=news_items,
        attachments=None,
        locale=locale,
        meta={"kind": "news_brief"},
    )
    res = agent.analyze(data)
    return {
        "ok": bool(res.get("ok")),
        "text": res.get("answer", ""),
        "json": res.get("parsed"),
    }

