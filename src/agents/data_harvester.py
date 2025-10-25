"""
Data Harvester Agent — 24/7 ingestion and investigations.

Capabilities
- Periodic tasks (news, macro, prices/fundamentals) with best‑effort retries
- Backfill historical news (up to 1y) when API keys allow (Tavily)
- Investigation reports: macro anomalies → web research → IA summary (econ_llm_agent)

Usage examples
- One cycle:      python -m src.agents.data_harvester --once
- Daemon (loop):  python -m src.agents.data_harvester --daemon --interval 1800
- Backfill news:  python -m src.agents.data_harvester --backfill-news-days 365 --query "gold OR mining"

Notes
- Writes Parquet under data/* using core.data_store
- Stores last run state in data/state/harvester_state.json
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import datetime as _dt

# Local imports (ensure src on path if needed when run as script)
try:
    from analytics.market_intel import collect_news
    from analytics.econ_llm_agent import EconomicAnalyst, EconomicInput
    from core.data_store import write_parquet
    from core.market_data import get_fred_series, get_price_history, get_fundamentals
except Exception:
    import sys as _sys
    _SRC = Path(__file__).resolve().parents[1]
    if str(_SRC) not in _sys.path:
        _sys.path.insert(0, str(_SRC))
    from analytics.market_intel import collect_news
    from analytics.econ_llm_agent import EconomicAnalyst, EconomicInput
    from core.data_store import write_parquet
    from core.market_data import get_fred_series, get_price_history, get_fundamentals


STATE_PATH = Path("data/state/harvester_state.json")


def _load_state() -> Dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"last_runs": {}}


def _save_state(st: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def _utcnow() -> datetime:
    return datetime.utcnow()


def _iso(dtobj: Optional[datetime] = None) -> str:
    d = dtobj or _utcnow()
    return d.isoformat() + "Z"


def _is_nan(val: Any) -> bool:
    try:
        import math
        return isinstance(val, float) and math.isnan(val)
    except Exception:
        return False


def _clean_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON safety."""
    import math
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ------------------------ NEWS ---------------------------------
def _persist_news_rows(rows: List[Dict[str, Any]], asof: datetime) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    # enforce minimal columns
    for c in ["ts","source","title","link","sent","tickers","summary"]:
        if c not in df.columns:
            df[c] = None
    # basic enrichment: event flags & sector hints by keywords (best-effort)
    try:
        def _kw(s):
            s = str(s or '').lower()
            return s
        def _flag(row, words):
            txt = _kw(row.get('title')) + ' ' + _kw(row.get('summary'))
            return any(w in txt for w in words)
        df['flag_earnings'] = df.apply(lambda r: _flag(r, ['earnings','results','guidance','profit','revenue']), axis=1)
        df['flag_mna'] = df.apply(lambda r: _flag(r, ['merger','acquisition','m&a','buyout','takeover']), axis=1)
        df['flag_geopolitics'] = df.apply(lambda r: _flag(r, ['geopolitic','sanction','war','conflict','election']), axis=1)
        df['flag_macro'] = df.apply(lambda r: _flag(r, ['inflation','cpi','jobs','payrolls','fomc','rate hike','rate cut','gdp']), axis=1)
        def _sector(txt):
            t = _kw(txt)
            if any(w in t for w in ['gold','mine','miner','gdx']): return 'gold'
            if any(w in t for w in ['bank','loan','credit']): return 'financials'
            if any(w in t for w in ['oil','energy','gas','brent']): return 'energy'
            if any(w in t for w in ['chip','semiconductor','ai','software','tech']): return 'technology'
            return None
        df['sector_hint'] = df.apply(lambda r: _sector(r.get('title')) or _sector(r.get('summary')), axis=1)
    except Exception:
        pass
    p = Path("data/news") / f"dt={asof.strftime('%Y-%m-%d')}" / f"news_{asof.strftime('%H%M%S')}.parquet"
    write_parquet(df, p)


def harvest_news_recent(regions: List[str], watchlist: List[str], query: str = "", window: str = "24h", limit: int = 300) -> int:
    rows: List[Dict[str, Any]] = []
    # generic query
    items, meta = collect_news(regions=regions, window=window, query=query, company=None, aliases=None, tgt_ticker=None, per_source_cap=None, limit=limit)
    rows.extend(items)
    # per-ticker/topical
    for t in watchlist:
        it, m = collect_news(regions=regions, window=window, query=t, company=None, aliases=None, tgt_ticker=t, per_source_cap=None, limit=int(limit/len(watchlist)) if watchlist else 100)
        rows.extend(it)
    _persist_news_rows(rows, _utcnow())
    return len(rows)


def _tavily_search_raw(q: str, time_range: str = "year", api_key: Optional[str] = None) -> Dict[str, Any]:
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        return {"ok": False, "error": "missing TAVILY_API_KEY", "results": []}
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": q, "time_range": time_range, "max_results": 25},
            timeout=30,
        )
        r.raise_for_status()
        js = r.json()
        return {"ok": True, "results": js.get("results", [])}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}


def _serper_search_raw(q: str, api_key: Optional[str] = None, time_range: str = "y1") -> Dict[str, Any]:
    """Serper.dev Google search API wrapper (news-like results).

    time_range values (Serper docs): h24, d7, m1, y1, etc. Default y1.
    """
    key = api_key or os.getenv("SERPER_API_KEY")
    if not key:
        return {"ok": False, "error": "missing SERPER_API_KEY", "results": []}
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": key, "Content-Type": "application/json"}
        payload = {
            "q": q,
            "tbs": f"qdr:{time_range}",  # time based restriction
            "num": 20,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        js = r.json()
        # unify top-level items (organic/news) into a common list
        results = []
        for sec in ("news", "organic", "topStories"):
            for it in js.get(sec, []) or []:
                results.append({
                    "title": it.get("title"),
                    "link": it.get("link") or it.get("url"),
                    "snippet": it.get("snippet") or it.get("content") or "",
                    "source": it.get("source") or it.get("displayedLink"),
                    "date": it.get("date") or it.get("datePublished"),
                })
        return {"ok": True, "results": results}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}


def backfill_news(years: float = 1.0, topic_queries: Optional[List[str]] = None) -> int:
    """Backfill older news using Tavily if available; fallback to finnews all-window queries.

    Writes parquet batches under data/news/dt=YYYY-MM-DD.
    """
    total = 0
    topic_queries = topic_queries or ["economy OR inflation", "gold OR mining", "geopolitics energy banks"]
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_serper = bool(os.getenv("SERPER_API_KEY"))
    today = _utcnow().date()
    cutoff = today - timedelta(days=int(years * 365))
    for q in topic_queries:
        rows: List[Dict[str, Any]] = []
        if has_tavily:
            out = _tavily_search_raw(q, time_range="year")
            for r in out.get("results", []):
                rows.append({
                    "ts": r.get("published_date") or _iso(),
                    "source": r.get("source") or r.get("url"),
                    "title": r.get("title"),
                    "link": r.get("url"),
                    "sent": 0.0,
                    "tickers": [],
                    "summary": r.get("content") or r.get("snippet") or "",
                })
        if has_serper:
            out2 = _serper_search_raw(q, time_range="y1")
            for r in out2.get("results", []):
                rows.append({
                    "ts": r.get("date") or _iso(),
                    "source": r.get("source") or r.get("link"),
                    "title": r.get("title"),
                    "link": r.get("link"),
                    "sent": 0.0,
                    "tickers": [],
                    "summary": r.get("snippet") or "",
                })
        if not rows:
            # fallback: broad finnews window
            items, _ = collect_news(regions=["US","INTL"], window="all", query=q, company=None, aliases=None, tgt_ticker=None, per_source_cap=None, limit=400)
            rows.extend(items)
        if rows:
            _persist_news_rows(rows, _utcnow())
            total += len(rows)
        time.sleep(0.5)
    return total


# ------------------------ MACRO / PRICES ----------------------
FRED_SERIES_DEFAULT = [
    "CPIAUCSL","T10YIE","INDPRO","GDPC1","UNRATE","PAYEMS","DGS10","DGS2","DTWEXBGS","NFCI","BAMLC0A0CM","BAMLH0A0HYM2","USREC"
]


def update_macro(series_ids: Optional[List[str]] = None) -> int:
    series_ids = series_ids or FRED_SERIES_DEFAULT
    cnt = 0
    for sid in series_ids:
        try:
            df = get_fred_series(sid)
            if df is None or df.empty:
                continue
            p = Path("data/macro") / f"series_id={sid}" / "series.parquet"
            # overwrite with latest snapshot (series are small)
            write_parquet(df.reset_index().rename(columns={df.columns[0]: sid, "date": "date"}), p)
            cnt += 1
        except Exception:
            pass
    return cnt


def update_prices_and_fundamentals(tickers: List[str]) -> int:
    cnt = 0
    for t in tickers:
        try:
            hist = get_price_history(t, start=(datetime.utcnow() - timedelta(days=365*5)).strftime("%Y-%m-%d"))
            if hist is not None and not hist.empty:
                p = Path("data/prices") / f"ticker={t}" / "prices.parquet"
                write_parquet(hist.reset_index().rename(columns={"Date": "date"}), p)
            fundamentals = get_fundamentals(t)
            pf = Path("data/fundamentals") / f"ticker={t}" / f"fundamentals_{datetime.utcnow().strftime('%Y%m%d')}.json"
            pf.parent.mkdir(parents=True, exist_ok=True)
            pf.write_text(json.dumps({"asof": _iso(), **fundamentals}, ensure_ascii=False, indent=2), encoding="utf-8")
            cnt += 1
            time.sleep(0.2)
        except Exception:
            pass
    return cnt


# ------------------------ INVESTIGATIONS ----------------------
def _next_weekday(d: _dt.date, target_weekday: int) -> _dt.date:
    # Monday=0 ... Sunday=6
    delta = (target_weekday - d.weekday()) % 7
    if delta == 0:
        delta = 7
    return d + _dt.timedelta(days=delta)

def _midmonth(d: _dt.date) -> _dt.date:
    # approximate CPI mid‑month: 15th or next business day
    target = d.replace(day=15)
    if target < d:
        # move to next month
        m = 1 if target.month == 12 else target.month + 1
        y = target.year + 1 if target.month == 12 else target.year
        target = target.replace(year=y, month=m, day=15)
    # if weekend, push to Monday
    while target.weekday() >= 5:
        target += _dt.timedelta(days=1)
    return target

def generate_upcoming_events(days_ahead: int = 14) -> List[Dict[str, Any]]:
    today = _dt.date.today()
    events: List[Dict[str, Any]] = []
    # CPI (approx mid‑month)
    cpi_date = _midmonth(today)
    if 0 <= (cpi_date - today).days <= days_ahead:
        events.append({"date": cpi_date.isoformat(), "name": "Inflation (CPI) — États‑Unis", "impact": "Prix à la consommation, peut influer sur les taux et le dollar"})
    # NFP (approx first Friday next month or next Friday if beginning)
    next_friday = _next_weekday(today, 4)
    if 0 <= (next_friday - today).days <= days_ahead:
        events.append({"date": next_friday.isoformat(), "name": "Emploi (NFP) — États‑Unis", "impact": "Marché du travail, peut influer sur la politique monétaire"})
    # FOMC (approx: next Wednesday + 2 weeks)
    next_wed = _next_weekday(today, 2)
    fomc_date = next_wed + _dt.timedelta(days=14)
    if 0 <= (fomc_date - today).days <= days_ahead:
        events.append({"date": fomc_date.isoformat(), "name": "Décision de la Fed (FOMC)", "impact": "Taux directeurs et communication sur la politique monétaire"})
    return events

def investigate_macro(theme: str = "gold miners & macro context") -> Dict[str, Any]:
    """Produce a concise investigation report combining macro deltas and a news brief."""
    # Macro snapshot (best-effort)
    macro: Dict[str, Any] = {}
    try:
        dxy = get_fred_series("DTWEXBGS"); dgs10 = get_fred_series("DGS10")
        s = dxy.iloc[:, 0].dropna() if dxy is not None and not dxy.empty else pd.Series([], dtype=float)
        s10 = dgs10.iloc[:, 0].dropna() if dgs10 is not None and not dgs10.empty else pd.Series([], dtype=float)
        if len(s) > 5:
            macro["DXY_wow"] = float((s.iloc[-1] / s.iloc[-5]) - 1.0)
        else:
            macro["DXY_wow"] = None
        if len(s10) > 5:
            macro["UST10Y_bp_wow"] = float((s10.iloc[-1] - s10.iloc[-5]) * 100.0)
        else:
            macro["UST10Y_bp_wow"] = None
    except Exception:
        pass
    # Gather recent news
    news_items, _ = collect_news(regions=["US","INTL"], window="last_week", query="gold OR mining", company=None, aliases=None, tgt_ticker=None, per_source_cap=None, limit=200)
    agent = EconomicAnalyst()
    # Load role briefs to guide LLM output
    try:
        from core.prompt_context import load_role_briefs
        role_briefs = load_role_briefs()
    except Exception:
        role_briefs = ""
    data = EconomicInput(
        question=(
            "Analyse le contexte macro (taux/dollar) et l'actualité liée à l'or et aux miners."
            " Donne: thèmes, risques, drivers, et implications pour le secteur dans 1w/1m."
        ),
        features={"macro": macro},
        news=news_items,
        attachments=[{"role_briefs": role_briefs}] if role_briefs else None,
        locale="fr-FR",
        meta={"kind": "investigation", "theme": theme},
    )
    # Prefer ensemble with adjudication; fallback to single
    try:
        ens = agent.analyze_ensemble(data, top_n=3, force_power=True, adjudicate=True)
        res = None
        # choose best text to show: judge decision if available else first ok result's answer
        answer_txt = (ens.get('adjudication') or {}).get('decision') or ""
        if not answer_txt:
            for r in (ens.get('results') or []):
                if r.get('ok') and r.get('answer'):
                    answer_txt = r['answer']; break
        parsed_obj = None
        # try to pick a parsed JSON from any ok result
        for r in (ens.get('results') or []):
            pj = r.get('parsed')
            if pj:
                parsed_obj = pj; break
        res = {"answer": answer_txt, "parsed": parsed_obj, "ensemble": ens}
    except Exception:
        res = agent.analyze(data)
    report = {
        "asof": _iso(),
        "macro": macro,
        "answer": res.get("answer"),
        "parsed": res.get("parsed"),
        "ensemble": res.get("ensemble") if isinstance(res, dict) else None,
    }
    outdir = Path("data/reports") / f"dt={datetime.utcnow().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)
    safe_report = _clean_json(report)
    (outdir / "investigation_gold.json").write_text(json.dumps(safe_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ------------------------ LOOP / CLI --------------------------
def discover_topics_via_llm(watchlist: List[str]) -> List[str]:
    """Use econ_llm_agent (g4f-backed) to propose search queries given macro + watchlist.

    Returns a list of short query strings (<= 10).
    """
    macro: Dict[str, Any] = {}
    try:
        dxy = get_fred_series("DTWEXBGS"); dgs10 = get_fred_series("DGS10")
        s = dxy.iloc[:, 0].dropna() if dxy is not None and not dxy.empty else pd.Series([], dtype=float)
        s10 = dgs10.iloc[:, 0].dropna() if dgs10 is not None and not dgs10.empty else pd.Series([], dtype=float)
        macro["DXY_wow"] = float((s.iloc[-1] / s.iloc[-5]) - 1.0) if len(s) > 5 else None
        macro["UST10Y_bp_wow"] = float((s10.iloc[-1] - s10.iloc[-5]) * 100.0) if len(s10) > 5 else None
    except Exception:
        pass

    agent = EconomicAnalyst()
    try:
        from core.prompt_context import load_role_briefs
        role_briefs = load_role_briefs()
    except Exception:
        role_briefs = ""
    prompt = (
        "Tu es un analyste macro-financier. Donne une liste de 6–10 requêtes concises pour rechercher des nouvelles"
        " pertinentes pour la compréhension du contexte économique et du secteur de l'or/mines,"
        " en tenant compte des deltas macro ci‑dessous et du watchlist."
        " Format: une puce par requête, 5–7 mots max; pas d'explication."
    )
    data = EconomicInput(
        question=prompt,
        features={"macro": macro, "watchlist": watchlist},
        news=None,
        attachments=[{"role_briefs": role_briefs}] if role_briefs else None,
        locale="fr-FR",
        meta={"kind": "topic_discovery"},
    )
    # Prefer ensemble + adjudication for robust prompts
    try:
        res = agent.analyze_ensemble(data, top_n=3, force_power=True, adjudicate=True)
        text = ((res or {}).get('adjudication') or {}).get('decision') or (res or {}).get('answer','')
    except Exception:
        res = agent.analyze(data)
        text = (res or {}).get("answer", "")
    queries: List[str] = []
    if text:
        for line in text.splitlines():
            s = line.strip("- •\t ")
            if len(s) >= 4 and len(queries) < 10 and not s.lower().startswith("#"):
                queries.append(s)
    # persist
    out = {"asof": _iso(), "macro": macro, "watchlist": watchlist, "queries": queries, "ensemble": res if isinstance(res, dict) else None}
    outdir = Path("data/reports") / f"dt={datetime.utcnow().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "topics.json").write_text(json.dumps(_clean_json(out), ensure_ascii=False, indent=2), encoding="utf-8")
    return queries

def run_once() -> Dict[str, Any]:
    st = _load_state()
    # WATCHLIST from env; fallback to data/watchlist.json if present
    wl_env = os.getenv("WATCHLIST") or "NGD.TO,AEM.TO,ABX.TO,K.TO,GDX"
    watchlist = [x.strip().upper() for x in wl_env.split(",") if x.strip()]
    try:
        p = Path('data/watchlist.json')
        if p.exists():
            obj = json.loads(p.read_text(encoding='utf-8'))
            lst = [x.strip().upper() for x in (obj.get('watchlist') or []) if isinstance(x, str) and x.strip()]
            if lst:
                watchlist = lst
    except Exception:
        pass
    out: Dict[str, Any] = {"asof": _iso(), "actions": []}
    try:
        n = harvest_news_recent(["US","INTL"], watchlist, query=os.getenv("NEWS_QUERY",""), window=os.getenv("NEWS_WINDOW","24h"), limit=300)
        out["actions"].append({"news_recent": n})
    except Exception as e:
        out["actions"].append({"news_recent_error": str(e)})
    try:
        m = update_macro()
        out["actions"].append({"macro_updated": m})
    except Exception as e:
        out["actions"].append({"macro_error": str(e)})
    try:
        # Refresh prices/fundamentals at most every PRICE_REFRESH_HOURS
        refresh_hours = int(os.getenv("PRICE_REFRESH_HOURS", "24"))
        last = st.get("last_runs", {}).get("last_prices_refresh_iso")
        do_refresh = True
        if last:
            try:
                from datetime import datetime, timezone
                last_dt = datetime.fromisoformat(last.replace("Z","+00:00"))
                age_h = (datetime.now(timezone.utc) - last_dt).total_seconds()/3600.0
                do_refresh = age_h >= max(1, refresh_hours)
            except Exception:
                do_refresh = True
        if do_refresh:
            p = update_prices_and_fundamentals(watchlist)
            out["actions"].append({"prices_funda": p})
            # store timestamp
            from datetime import datetime, timezone
            st.setdefault("last_runs", {})["last_prices_refresh_iso"] = datetime.now(timezone.utc).isoformat()
        else:
            out["actions"].append({"prices_funda": "skipped_recently_refreshed"})
    except Exception as e:
        out["actions"].append({"prices_error": str(e)})
    try:
        if os.getenv("DO_INVESTIGATE","1") == "1":
            rep = investigate_macro()
            out["actions"].append({"investigation": bool(rep.get("answer"))})
    except Exception as e:
        out["actions"].append({"investigation_error": str(e)})
    # Write upcoming macro events (heuristic, user-friendly labels)
    try:
        days_ahead = int(os.getenv("EVENTS_DAYS_AHEAD","14"))
        evts = generate_upcoming_events(days_ahead=days_ahead)
        if evts:
            from datetime import datetime as _dt2
            evdir = Path("data/events") / f"dt={_dt2.utcnow().strftime('%Y%m%d')}"
            evdir.mkdir(parents=True, exist_ok=True)
            payload = {"asof": _iso(), "days_ahead": days_ahead, "events": evts}
            (evdir / "events.json").write_text(json.dumps(_clean_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        out["actions"].append({"events": len(evts) if evts else 0})
    except Exception as e:
        out["actions"].append({"events_error": str(e)})
    try:
        if os.getenv("DO_DISCOVER_TOPICS","1") == "1":
            qs = discover_topics_via_llm(watchlist)
            out["actions"].append({"topics": len(qs)})
            # light backfill on top 3 queries
            if qs:
                _ = backfill_news(years=1.0, topic_queries=qs[:3])
    except Exception as e:
        out["actions"].append({"topics_error": str(e)})
    # Optionally refresh g4f working models on schedule
    try:
        refresh_hours = int(os.getenv("G4F_REFRESH_HOURS", "6"))
        last = st.get("last_runs", {}).get("last_g4f_refresh_iso")
        do_refresh = True
        if last:
            from datetime import datetime, timezone
            last_dt = datetime.fromisoformat(last.replace("Z","+00:00"))
            age_h = (datetime.now(timezone.utc) - last_dt).total_seconds()/3600.0
            do_refresh = age_h >= max(1, refresh_hours)
        if do_refresh and os.getenv("G4F_AUTO_REFRESH","1") == "1":
            try:
                from agents.g4f_model_watcher import refresh as _g4f_refresh
                p = _g4f_refresh(limit=int(os.getenv("G4F_TEST_LIMIT","8")), refresh_verified=True)
                out["actions"].append({"g4f_models_refresh": str(p)})
                from datetime import datetime, timezone
                st.setdefault("last_runs", {})["last_g4f_refresh_iso"] = datetime.now(timezone.utc).isoformat()
            except Exception as ie:
                out["actions"].append({"g4f_models_refresh_error": str(ie)})
        else:
            out["actions"].append({"g4f_models_refresh": "skipped"})
    except Exception as e:
        out["actions"].append({"g4f_refresh_error": str(e)})
    st["last_runs"]["once"] = out
    _save_state(st)
    return out


def daemon_loop(interval_seconds: int = 1800) -> None:
    while True:
        try:
            run_once()
        except Exception:
            pass
        time.sleep(max(60, int(interval_seconds)))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Data Harvester Agent")
    p.add_argument("--once", action="store_true", help="Run a single harvesting cycle")
    p.add_argument("--daemon", action="store_true", help="Run forever with an interval")
    p.add_argument("--interval", type=int, default=int(os.getenv("HARVEST_INTERVAL","1800")), help="Interval seconds for daemon mode")
    p.add_argument("--backfill-news-days", type=int, default=0, help="Backfill news this many days via Tavily/finnews")
    p.add_argument("--query", type=str, default="", help="Backfill query/topic")
    args = p.parse_args()

    if args.backfill_news_days and args.backfill_news_days > 0:
        q = args.query or "economy OR inflation OR gold"
        print(json.dumps({"ok": True, "backfilled": backfill_news(years=args.backfill_news_days/365.0, topic_queries=[q])}, ensure_ascii=False))
    elif args.daemon:
        print(json.dumps({"ok": True, "mode": "daemon", "interval": args.interval}, ensure_ascii=False))
        daemon_loop(args.interval)
    else:
        print(json.dumps(run_once(), ensure_ascii=False))
