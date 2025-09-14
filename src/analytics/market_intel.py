# src/analytics/market_intel.py
# -*- coding: utf-8 -*-
"""
Market Intelligence Orchestrator
- Agrège Nouvelles (finnews), Ownership & Insiders (financials_ownership_client),
  et optionnellement Finviz (company snapshot / options / technicals) + Macro/Futures (macro_derivatives_client).
- Produit un JSON unifié et des features agrégées compatibles avec econ_llm_agent.

Sorties:
- "news": liste normalisée d'items (JSONL-friendly)
- "features": dict de métriques (globales ou par ticker s'il est fourni)
- "meta": contexte & sources utilisées

Usage CLI:
  python -m src.analytics.market_intel run \
      --regions US,CA,INTL --window last_week --limit 200 \
      --outdir data/real --prefix auto
  python -m src.analytics.market_intel run \
      --ticker NGD --query "gold OR mine" --regions US,INTL \
      --limit 150 --outdir artifacts

Auteur: toi
"""
from __future__ import annotations

import os, re, sys, json, argparse, datetime as dt
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict

# ========= Imports internes (best effort) =========
try:
    from ..ingestion.finnews import run_pipeline as finnews_run, build_news_features as finnews_build_features
except Exception as e:
    finnews_run = None
    finnews_build_features = None

try:
    from ..ingestion.financials_ownership_client import build_ownership_snapshot
except Exception as e:
    build_ownership_snapshot = None

# Optionnels (tu les ajouteras ensuite) — tout est déjà protégé par best-effort
try:
    from ..ingestion.finviz_client import finviz_company_snapshot, finviz_options_strikes, finviz_futures_snapshot
except Exception:
    finviz_company_snapshot = None
    finviz_options_strikes = None
    finviz_futures_snapshot = None

try:
    from ..ingestion.macro_derivatives_client import macro_term_structures, rates_fx_commod_summary
except Exception:
    macro_term_structures = None
    rates_fx_commod_summary = None

# ============== Utils & Config ==============

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def iso(dtobj: Optional[dt.datetime] = None) -> str:
    d = dtobj or now_utc()
    return d.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x,(int,float)): return float(x)
        s = str(x).replace(",","").replace("$","").replace("%","").strip()
        if s in ("","-","N/A","NaN","None","null"): return None
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMBT])\s*$", s, flags=re.I)
        if m:
            v = float(m.group(1)); mult = {"K":1e3,"M":1e6,"B":1e9,"T":1e12}[m.group(2).upper()]
            return v*mult
        return float(s)
    except Exception:
        return None

def _mean(a: List[float]) -> float:
    a2 = [float(x) for x in a if x is not None]
    return float(sum(a2)/max(1,len(a2))) if a2 else 0.0

def _safe_get(d: Dict[str,Any], *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(p)
    return cur if cur is not None else default

CACHE_DIR = Path(os.getenv("MARKET_INTEL_CACHE","cache/market_intel"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============== Normalisation News (min) ==============

def _news_to_minimal(item: Dict[str,Any]) -> Dict[str,Any]:
    """Mappe un NewsItem (finnews) vers un dict minimal commun (stable pour JSONL)."""
    return {
        "ts": item.get("published") or item.get("ts") or iso(),
        "source": item.get("source") or item.get("meta",{}).get("domain"),
        "title": item.get("title"),
        "link": item.get("link"),
        "sent": float(item.get("sentiment") or 0.0),
        "tickers": item.get("tickers") or [],
        "summary": item.get("summary") or item.get("raw_text","")[:500],
        "tags": {
            "earnings": "earnings" in (item.get("event_types") or []),
            "sanctions": "sanctions" in (item.get("event_types") or []),
            "geopolitics": "geopolitics" in (item.get("event_types") or []),
            "energy": ("energy" in (item.get("sectors") or [])) or ("energy_shock" in (item.get("event_types") or [])),
            "banks": "banks" in (item.get("sectors") or []),
            "defense": "defense" in (item.get("sectors") or []),
            "tech": "tech" in (item.get("sectors") or []),
        }
    }

# ============== Features agrégées ==============

def _aggregate_features(news_items: List[Dict[str,Any]], target_ticker: Optional[str]) -> Dict[str,Any]:
    """
    Si finnews_build_features est dispo, on s'en sert. Sinon, on calcule des metrics simples.
    """
    if finnews_build_features:
        feats_all = finnews_build_features(news_items, target_ticker=target_ticker)
        if target_ticker:
            return feats_all.get(target_ticker.upper(), {})
        # sinon, on compacte sur clé 'ALL'
        flat = defaultdict(float)
        for dd in feats_all.values():
            for k,v in dd.items():
                flat[k] += float(v or 0.0)
        # normalise quelques ratios au besoin
        if flat.get("news_count",0)>0:
            c = flat["news_count"]
            for k in ("pos_ratio","neg_ratio","mean_sentiment"):
                flat[k] = float(flat.get(k,0.0))/max(1.0,c) if k!="mean_sentiment" else float(flat.get(k,0.0))/max(1.0,len(feats_all))
        return dict(flat)

    # ---- fallback minimal ----
    sents = [float(x.get("sent",0.0)) for x in news_items]
    pos = sum(1 for s in sents if s>0.15)
    neg = sum(1 for s in sents if s<-0.15)
    tags_list = [x.get("tags",{}) for x in news_items]
    sectors = Counter()
    events = Counter()
    for t in tags_list:
        if t.get("energy"): sectors["energy"] += 1
        if t.get("banks"): sectors["banks"] += 1
        if t.get("defense"): sectors["defense"] += 1
        if t.get("tech"): sectors["tech"] += 1
        if t.get("earnings"): events["earnings"] += 1
        if t.get("geopolitics"): events["geopolitics"] += 1
        if t.get("sanctions"): events["sanctions"] += 1

    return {
        "news_count": float(len(news_items)),
        "mean_sentiment": round(_mean(sents),4),
        "pos_ratio": round(pos/max(1,len(news_items)),4),
        "neg_ratio": round(neg/max(1,len(news_items)),4),
        "unique_sources": float(len(set(x.get("source") for x in news_items if x.get("source")))),
        "flag_earnings": float(events.get("earnings",0)>0),
        "flag_mna": 0.0,  # non calculé en fallback
        "flag_sanctions": float(events.get("sanctions",0)>0),
        "flag_geopolitics": float(events.get("geopolitics",0)>0),
        "flag_energy_shock": float(sectors.get("energy",0)>0),
        "sector_energy": float(sectors.get("energy",0)),
        "sector_banks": float(sectors.get("banks",0)),
        "sector_defense": float(sectors.get("defense",0)),
        "sector_tech": float(sectors.get("tech",0)),
    }

# ============== Collectors ==============

def collect_news(regions: List[str], window: str, query: str = "", company: Optional[str] = None,
                 aliases: Optional[List[str]] = None, tgt_ticker: Optional[str] = None,
                 per_source_cap: Optional[int] = None, limit: int = 150) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
    meta = {"source_ok": False, "regions": regions, "window": window, "limit": limit}
    if not finnews_run:
        return [], {**meta, "error": "finnews module unavailable"}
    items = finnews_run(
        regions=regions,
        window=window,
        query=query,
        company=company,
        aliases=aliases or [],
        tgt_ticker=tgt_ticker,
        per_source_cap=per_source_cap,
        limit=limit
    )
    norm = [_news_to_minimal(asdict(x) if hasattr(x, "__dict__") else x) for x in items]
    meta["source_ok"] = True
    return norm, meta

def collect_ownership(ticker: str) -> Dict[str,Any]:
    if not build_ownership_snapshot:
        return {"ok": False, "error": "financials_ownership_client unavailable"}
    return build_ownership_snapshot(ticker)

def collect_finviz(ticker: Optional[str], want_futures: bool) -> Dict[str,Any]:
    out = {"ok": True, "ticker": ticker, "futures": None, "company": None, "options": None, "source_ok": False}
    try:
        if want_futures and finviz_futures_snapshot:
            out["futures"] = finviz_futures_snapshot()  # agrégat global
            out["source_ok"] = True
        if ticker and finviz_company_snapshot:
            out["company"] = finviz_company_snapshot(ticker)
            out["source_ok"] = True
        if ticker and finviz_options_strikes:
            out["options"] = finviz_options_strikes(ticker)
            out["source_ok"] = True
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", **out}
    return out

def collect_macro_derivs() -> Dict[str,Any]:
    out = {"ok": True, "term_structures": None, "rates_fx_commod": None, "source_ok": False}
    try:
        if macro_term_structures:
            out["term_structures"] = macro_term_structures()
            out["source_ok"] = True
        if rates_fx_commod_summary:
            out["rates_fx_commod"] = rates_fx_commod_summary()
            out["source_ok"] = True
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", **out}
    return out

# ============== Fusion / Scoring additionnel ==============

def _ownership_signals(own: Dict[str,Any]) -> Dict[str,Any]:
    sig = {}
    # insider 30/90j net trades
    net30 = _safe_get(own, "insiders","aggregates","window_30d","net_trades", default=None)
    net90 = _safe_get(own, "insiders","aggregates","window_90d","net_trades", default=None)
    if net30 is not None: sig["insider_net_trades_30d"] = net30
    if net90 is not None: sig["insider_net_trades_90d"] = net90
    # options overview
    calls = _safe_get(own, "options_overview","calls_count", default=None)
    puts  = _safe_get(own, "options_overview","puts_count", default=None)
    if calls is not None and puts is not None and (calls+puts)>0:
        sig["opt_activity_ratio_calls"] = round(calls/max(1,calls+puts),4)
        sig["opt_activity_ratio_puts"]  = round(puts /max(1,calls+puts),4)
    # short % float (Yahoo best-effort)
    spf = _safe_get(own, "yahoo","short_percent_float", default=None)
    if spf is not None: sig["short_percent_float"] = float(spf)
    return sig

def _finviz_company_signals(fv: Dict[str,Any]) -> Dict[str,Any]:
    if not fv or not fv.get("company"): return {}
    c = fv["company"]
    out = {}
    for k in ("InsiderOwn","InsiderTrans","InstOwn","InstTrans","PerfW","PerfYTD","RSI","SMA50","SMA200"):
        v = c.get(k)
        if v is None: continue
        out[f"finviz_{k.lower()}"] = _to_float(v)
    # options skew si fourni par finviz_options_strikes
    if fv.get("options") and isinstance(fv["options"], dict):
        skew = fv["options"].get("skew_25d")  # hypothétique champ si ton client le calcule
        if skew is not None:
            out["options_skew_25d"] = _to_float(skew)
    return out

def _macro_signals(m: Dict[str,Any]) -> Dict[str,Any]:
    if not m or not m.get("source_ok"): 
        # si rates_fx_commod summary renvoie dict simple, on copie
        out = {}
        rfc = m.get("rates_fx_commod")
        if isinstance(rfc, dict):
            # attend ex: {"DXY_wow":-0.04, "Brent_wow":2.09, "UST10Y_bp": -0.5, ...}
            for k,v in rfc.items():
                out[f"macro_{k}"] = _to_float(v)
        return out
    out = {}
    if m.get("rates_fx_commod"):
        for k,v in m["rates_fx_commod"].items():
            out[f"macro_{k}"] = _to_float(v)
    return out

def build_unified_features(news_items: List[Dict[str,Any]],
                           target_ticker: Optional[str],
                           ownership: Optional[Dict[str,Any]],
                           finviz_blob: Optional[Dict[str,Any]],
                           macro_blob: Optional[Dict[str,Any]]) -> Dict[str,Any]:
    base = _aggregate_features(news_items, target_ticker=target_ticker)
    # enrich ownership
    if ownership and ownership.get("ok"):
        base.update(_ownership_signals(ownership))
        # fondamentaux (prix/mcap/pe/beta…)
        y = ownership.get("yahoo") or {}
        for k in ("price","market_cap","pe","beta","dividend_yield"):
            if y.get(k) is not None:
                base[f"y_{k}"] = _to_float(y.get(k))
    # finviz enhancement
    base.update(_finviz_company_signals(finviz_blob))
    # macro / futures
    base.update(_macro_signals(macro_blob or {}))
    return base

# ============== Snapshot Orchestrator ==============

def build_snapshot(regions: List[str], window: str, query: str = "",
                   ticker: Optional[str] = None, company: Optional[str] = None,
                   aliases: Optional[List[str]] = None, limit: int = 150,
                   per_source_cap: Optional[int] = None,
                   include_finviz: bool = True, include_futures: bool = True,
                   include_macro_derivs: bool = False) -> Dict[str,Any]:
    """
    Retourne un JSON unifié: {news:[...], features:{...}, meta:{...}, ownership?:{...}, finviz?:{...}, macro?:{...}}
    """
    # 1) News
    news, news_meta = collect_news(
        regions=regions, window=window, query=query, company=company,
        aliases=aliases or [], tgt_ticker=ticker, per_source_cap=per_source_cap, limit=limit
    )
    # 2) Ownership / Insiders
    ownership = collect_ownership(ticker) if ticker else None
    # 3) Finviz (optionnel)
    finviz_blob = collect_finviz(ticker, want_futures=include_futures) if include_finviz else None
    # 4) Macro / dérivés (optionnel)
    macro_blob = collect_macro_derivs() if include_macro_derivs else None

    # 5) Features unifiées
    feats = build_unified_features(news, target_ticker=ticker, ownership=ownership,
                                   finviz_blob=finviz_blob, macro_blob=macro_blob)

    # 6) meta pack
    meta = {
        "asof_utc": iso(),
        "regions": regions,
        "window": window,
        "query": query,
        "ticker": ticker,
        "company": company,
        "aliases": aliases or [],
        "limits": {"per_source_cap": per_source_cap, "limit": limit},
        "sources_ok": {
            "finnews": news_meta.get("source_ok", False),
            "ownership": bool(ownership and ownership.get("ok")),
            "finviz": bool(finviz_blob and finviz_blob.get("source_ok")),
            "macro_derivs": bool(macro_blob and macro_blob.get("source_ok")),
        },
        "notes": "Best-effort merge. Modules finviz_client/macro_derivatives_client sont optionnels.",
    }

    return {
        "news": news,
        "features": feats,
        "meta": meta,
        "ownership": ownership,
        "finviz": finviz_blob,
        "macro": macro_blob,
    }

# ============== IO helpers ==============

def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _write_jsonl(path: Path, rows: List[Dict[str,Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _prefix_or_now(prefix: Optional[str]) -> str:
    if prefix and prefix.lower() != "auto":
        return prefix
    # auto
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M")

# ============== CLI ==============

def _cli_run(args):
    regions = [r.strip() for r in (args.regions or "US,INTL").split(",") if r.strip()]
    aliases = [a.strip() for a in (args.aliases.split(",") if args.aliases else []) if a.strip()]
    snap = build_snapshot(
        regions=regions,
        window=args.window,
        query=args.query,
        ticker=args.ticker,
        company=args.company,
        aliases=aliases,
        limit=args.limit,
        per_source_cap=args.per_source_cap,
        include_finviz=not args.no_finviz,
        include_futures=not args.no_futures,
        include_macro_derivs=args.macro_derivs
    )
    # impression écran
    if args.stdout:
        print(json.dumps(snap, ensure_ascii=False, indent=2))

    # fichiers
    if args.outdir:
        outdir = Path(args.outdir)
        prefix = _prefix_or_now(args.prefix)
        # fichiers compatibles avec build_real_dataset.py
        _write_json(outdir / f"{prefix}_features.json", {"features": snap["features"]})
        _write_jsonl(outdir / f"{prefix}_news.jsonl", snap["news"])
        # meta courte
        _write_json(outdir / f"{prefix}_meta.json", {
            "attachments": [
                f"FX/Commod (si dispo): {snap.get('features',{}).get('macro_DXY_wow','?')} DXY WoW; "
                f"Brent {snap.get('features',{}).get('macro_Brent_wow','?')} WoW; "
                f"UST10Y {snap.get('features',{}).get('macro_UST10Y_bp','?')} bp."
            ],
            "notes": "Market intel snapshot (finnews + ownership + finviz + macro_derivs best-effort).",
            "sources_used": [
                "finnews","YahooFinance","SEC EDGAR"
            ] + (["Finviz"] if not args.no_finviz else []) + (["MacroDerivatives"] if args.macro_derivs else [])
        })

def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Market Intelligence Orchestrator")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="Build unified snapshot + features")
    pr.add_argument("--regions", type=str, default="US,CA,INTL,GEO")
    pr.add_argument("--window", type=str, default="last_week",
                    help="1h,6h,12h,24h,48h,last_day,last_week,last_month,all")
    pr.add_argument("--query", type=str, default="")
    pr.add_argument("--ticker", type=str, default=None)
    pr.add_argument("--company", type=str, default=None)
    pr.add_argument("--aliases", type=str, default=None)
    pr.add_argument("--limit", type=int, default=150)
    pr.add_argument("--per_source_cap", type=int, default=None)
    pr.add_argument("--no_finviz", action="store_true")
    pr.add_argument("--no_futures", action="store_true")
    pr.add_argument("--macro_derivs", action="store_true",
                    help="Active la collecte macro_derivatives_client si dispo")
    pr.add_argument("--stdout", action="store_true", help="Print full JSON snapshot to stdout")
    pr.add_argument("--outdir", type=str, default=None, help="Si défini, écrit 3 fichiers (features/news/meta)")
    pr.add_argument("--prefix", type=str, default="auto", help="Préfixe de fichiers (auto => YYYYMMDD_HHMM)")

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help(); return 0
    if args.cmd == "run":
        _cli_run(args)
        return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
