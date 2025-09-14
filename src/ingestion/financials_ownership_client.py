# src/ingestion/financials_ownership_client.py
# -*- coding: utf-8 -*-
"""
Financials & Ownership client (Yahoo + SEC EDGAR: submissions / 10-K/10-Q/8-K / Form 4 / 13F)
- Télécharge, met en cache, normalise et agrège données 'fundamentals', 'insiders' et 'institutionnels'.
- Conçu pour cohabiter avec finnews / macro_derivatives_client; schémas stables et JSONL-friendly.

Fonctions principales:
- yahoo_snapshot(ticker) -> dict (price, mcap, pe, beta, sector, industry, calendar, dividends)
- yahoo_options_chain(ticker, expiry=None) -> dict(calls=[...], puts=[...], expiries=[...])
- sec_submissions(cik_or_ticker) -> dict (company profile + derniers filings)
- sec_filings_index(cik_or_ticker, forms=['10-K','10-Q','8-K'], limit=100) -> list[...]
- sec_form4_insiders(cik_or_ticker, limit=200) -> dict(transactions=[...], aggregates={...})
- sec_13f_holdings(cik_or_ticker, limit_filings=1) -> dict(filings=[...], holdings=[...])
- build_ownership_snapshot(ticker) -> dict unifié (yahoo, filings, insiders, 13F, options meta)

Notes:
- Aucune clé obligatoire. Respecte les bonnes pratiques SEC (User-Agent).
- Cache local dans cache/ownership/ + retries/backoff.

Auteur: toi
"""
from __future__ import annotations

import os, re, io, csv, sys, json, math, time, enum, random, hashlib, datetime as dt
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlencode

# --------- HTTP / Parsing ---------
try:
    import requests
except Exception as e:
    raise RuntimeError("financials_ownership_client requires `requests` (pip install requests)") from e

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw): 
        return it

# =========================
# Config, cache & utilities
# =========================

CACHE_DIR = Path("cache") / "ownership"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMEOUT = float(os.getenv("OWN_TIMEOUT", "25"))
RETRIES = int(os.getenv("OWN_RETRIES", "2"))
BACKOFF = float(os.getenv("OWN_BACKOFF", "1.6"))

UA_ROT = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
]
SEC_UA = os.getenv("SEC_USER_AGENT", "yourname-contact@example.com (edu/test)")  # mets un contact réel si possible

def _ua(): 
    return random.choice(UA_ROT)

def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _iso(d: dt.datetime) -> str: 
    return d.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{_sha1(key)}.cache"

def _get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
         use_cache=True, as_json=False, as_text=True, raw=False) -> Any:
    params = params or {}
    key = url if not params else f"{url}?{urlencode(sorted(params.items()))}"
    cpath = _cache_path(key)
    if use_cache and cpath.exists():
        try:
            b = cpath.read_bytes()
            if as_json:
                return json.loads(b.decode("utf-8", errors="ignore"))
            if as_text:
                return b.decode("utf-8", errors="ignore")
            return b
        except Exception:
            pass
    last_err = None
    wait = 0.3
    for att in range(1, RETRIES + 2):
        try:
            r = requests.get(url, params=params, headers=headers or {"User-Agent": _ua()}, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                if raw:
                    out = r.content
                    try: cpath.write_bytes(out)
                    except Exception: pass
                    return out
                if as_json:
                    out = r.json()
                    try: cpath.write_text(json.dumps(out), encoding="utf-8")
                    except Exception: pass
                    return out
                if as_text:
                    out = r.text
                    try: cpath.write_text(out, encoding="utf-8")
                    except Exception: pass
                    return out
                return r.content
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(wait); wait *= BACKOFF
    raise RuntimeError(f"GET failed {url} | last_err={last_err}")

def _to_float(x: Any) -> Optional[float]:
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(",","").replace("$","").replace("%","").strip()
    if s in ("","-","N/A","NaN","null","None"): return None
    # "1.23B", "456.7M"
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMBT])\s*$", s, flags=re.I)
    if m:
        val = float(m.group(1))
        mult = {"K":1e3,"M":1e6,"B":1e9,"T":1e12}[m.group(2).upper()]
        return val * mult
    try: return float(s)
    except Exception: return None

def _strip_html(txt: str) -> str:
    if not txt: return ""
    if BeautifulSoup:
        return BeautifulSoup(txt, "html.parser").get_text(" ", strip=True)
    return re.sub("<[^>]*>", " ", str(txt or ""))


# small helpers to safely unwrap callables and convert pandas objects
def _val(x: Any):
    """Return x() if x is callable else x. Swallows exceptions and returns the original on error."""
    try:
        return x() if callable(x) else x
    except Exception:
        return x

def _df_to_dict(df: Any):
    """Safely call pandas.DataFrame/Series .to_dict(), return None on failure or if df is None."""
    try:
        if df is None:
            return None
        if hasattr(df, "to_dict"):
            return df.to_dict()
        return df
    except Exception:
        return None

# ====================
# Yahoo Finance client
# ====================

def yahoo_snapshot(ticker: str, use_cache=True) -> Dict[str, Any]:
    """
    Essaye yfinance d'abord; sinon fallback à parsing HTML light de la 'quoteSummary' page.
    Retour: dict avec price, market_cap, pe, beta, sector, industry, dividend_yield, earnings_dates, short_interest (best-effort)
    """
    t = ticker.strip().upper()
    out: Dict[str, Any] = {"ticker": t, "source": "YahooFinance", "ok": True, "asof_utc": _iso(_now())}
    used = "yfinance" if yf else "html"
    try:
        if yf:
            y = yf.Ticker(t)
            info = {}
            try:
                info = y.info or {}
            except Exception:
                # yfinance peut lever si endpoint bloque; on passe en html
                info = {}
            price = None
            try:
                px = y.history(period="1d", interval="1d")
                if px is not None and not px.empty:
                    price = float(px["Close"].iloc[-1])
            except Exception:
                price = None
            cal = {}
            try:
                cal = y.calendar or {}
            except Exception:
                cal = {}
            div = None
            try:
                div = float(info.get("dividendYield")) if info.get("dividendYield") is not None else None
            except Exception:
                div = None
            out.update({
                "price": price,
                "market_cap": _to_float(info.get("marketCap")),
                "pe": _to_float(info.get("trailingPE") or info.get("forwardPE")),
                "beta": _to_float(info.get("beta")),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "dividend_yield": div,
                # yfinance may expose earnings_dates as a property or a callable; use _val to be safe
                # avoid truth-testing dataframes (which raises ValueError)
                "earnings_dates": [],
                "short_name": info.get("shortName") or info.get("longName"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
                "source_used": used,
            })
            # short interest approximatif — yfinance n’expose pas uniformément.
            # On essaie un endpoint non-officiel (best effort).
            try:
                txt = _get(f"https://finance.yahoo.com/quote/{t}", use_cache=use_cache, as_text=True)
                m = re.search(r'"shortPercentOfFloat"\s*:\s*([0-9\.]+)', txt)
                out["short_percent_float"] = _to_float(m.group(1)) if m else None
            except Exception:
                out["short_percent_float"] = None
            return out

        # ---- fallback HTML léger ----
        txt = _get(f"https://finance.yahoo.com/quote/{t}", use_cache=use_cache, as_text=True)
        # price
        m_px = re.search(r'"regularMarketPrice"\s*:\s*\{"raw":\s*([0-9\.]+)', txt)
        # market cap
        m_mcap = re.search(r'"marketCap"\s*:\s*\{"raw":\s*([0-9eE\.\-]+)', txt)
        m_pe = re.search(r'"trailingPE"\s*:\s*\{"raw":\s*([0-9eE\.\-]+)', txt)
        m_beta = re.search(r'"beta"\s*:\s*\{"raw":\s*([0-9eE\.\-]+)', txt)
        m_sector = re.search(r'"sector"\s*:\s*"([^"]+)"', txt)
        m_industry = re.search(r'"industry"\s*:\s*"([^"]+)"', txt)
        m_name = re.search(r'"shortName"\s*:\s*"([^"]+)"', txt)
        m_curr = re.search(r'"currency"\s*:\s*"([^"]+)"', txt)
        out.update({
            "price": _to_float(m_px.group(1)) if m_px else None,
            "market_cap": _to_float(m_mcap.group(1)) if m_mcap else None,
            "pe": _to_float(m_pe.group(1)) if m_pe else None,
            "beta": _to_float(m_beta.group(1)) if m_beta else None,
            "sector": m_sector.group(1) if m_sector else None,
            "industry": m_industry.group(1) if m_industry else None,
            "short_name": m_name.group(1) if m_name else None,
            "currency": m_curr.group(1) if m_curr else None,
            "source_used": "html",
            "dividend_yield": None,
            "earnings_dates": [],
            "short_percent_float": None,
            "exchange": None
        })
        return out
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "source": "YahooFinance", "ticker": t, "asof_utc": _iso(_now())}

# -------- Yahoo options chain --------

def yahoo_options_chain(ticker: str, expiry: Optional[str] = None, use_cache=True) -> Dict[str, Any]:
    """
    Retourne {calls:[...], puts:[...], expiries:[...]} avec champs: strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility
    - Avec yfinance: simple.
    - Fallback HTML (best effort) si yfinance absent (retourne expiries uniquement).
    expiry: 'YYYY-MM-DD' optionnelle; si None -> prochaine échéance.
    """
    t = ticker.strip().upper()
    result = {"ticker": t, "ok": True, "source": "YahooOptions", "asof_utc": _iso(_now())}
    try:
        if yf:
            y = yf.Ticker(t)
            exps = list(getattr(y, "options", []))
            result["expiries"] = exps
            if not exps:
                result.update({"calls": [], "puts": []})
                return result
            exp = expiry or exps[0]
            opt = _val(getattr(y, "option_chain", lambda e: None))(exp) if hasattr(y, "option_chain") else None
            # normalize rows
            def _norm(df):
                if df is None or (hasattr(df, "empty") and getattr(df, "empty")):
                    return []
                rows = []
                # iterate either pandas DataFrame or list/dict
                if hasattr(df, "iterrows"):
                    iterable = df.iterrows()
                    for _, r in iterable:
                        rows.append({
                            "contractSymbol": r.get("contractSymbol"),
                            "strike": _to_float(r.get("strike")),
                            "lastPrice": _to_float(r.get("lastPrice")),
                            "bid": _to_float(r.get("bid")),
                            "ask": _to_float(r.get("ask")),
                            "volume": _to_float(r.get("volume")),
                            "openInterest": _to_float(r.get("openInterest")),
                            "impliedVolatility": _to_float(r.get("impliedVolatility")),
                            "inTheMoney": bool(r.get("inTheMoney")),
                        })
                else:
                    for r in (df or []):
                        rows.append({
                            "contractSymbol": r.get("contractSymbol") if isinstance(r, dict) else None,
                            "strike": _to_float(r.get("strike") if isinstance(r, dict) else None),
                            "lastPrice": _to_float(r.get("lastPrice") if isinstance(r, dict) else None),
                            "bid": _to_float(r.get("bid") if isinstance(r, dict) else None),
                            "ask": _to_float(r.get("ask") if isinstance(r, dict) else None),
                            "volume": _to_float(r.get("volume") if isinstance(r, dict) else None),
                            "openInterest": _to_float(r.get("openInterest") if isinstance(r, dict) else None),
                            "impliedVolatility": _to_float(r.get("impliedVolatility") if isinstance(r, dict) else None),
                            "inTheMoney": bool(r.get("inTheMoney")) if isinstance(r, dict) else False,
                        })
                return rows
            result["chosen_expiry"] = exp
            # opt may be a object with .calls/.puts or a tuple
            calls_src = None
            puts_src = None
            if opt is None:
                calls_src = puts_src = []
            elif isinstance(opt, tuple) and len(opt) >= 2:
                calls_src, puts_src = opt[0], opt[1]
            else:
                calls_src = getattr(opt, "calls", None) or getattr(opt, 0, None)
                puts_src = getattr(opt, "puts", None) or getattr(opt, 1, None)
            result["calls"] = _norm(calls_src)
            result["puts"]  = _norm(puts_src)
            return result

        # fallback: extraire expiries depuis HTML
        txt = _get(f"https://finance.yahoo.com/quote/{t}/options", use_cache=use_cache, as_text=True)
        exps = re.findall(r'\/options\?date\=([0-9]{10})', txt)
        exps = sorted(set(exps))
        result["expiries"] = exps
        result["calls"], result["puts"] = [], []
        return result
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "source": "YahooOptions", "ticker": t, "asof_utc": _iso(_now())}

# ==============
# SEC - Submissions
# ==============
SEC_API = "https://data.sec.gov"

def _sec_headers():
    return {"User-Agent": SEC_UA, "Accept": "application/json"}

def _slug_cik(x: str) -> str:
    s = re.sub(r"\D", "", x or "")  # garder digits
    return s.zfill(10) if s else ""

def _cik_from_submissions_json(js: Dict[str, Any]) -> Optional[str]:
    try:
        cik = js.get("cik")
        if cik:
            return _slug_cik(str(cik))
    except Exception:
        pass
    return None

def sec_submissions(cik_or_ticker: str, use_cache=True) -> Dict[str, Any]:
    """
    data.sec.gov/submissions/CIK##########.json
    Si un ticker est donné: le même endpoint accepte `CIK##########` uniquement → on détecte via map 'tickers' retournée par SEC si dispo.
    """
    key = cik_or_ticker.strip().upper()
    cik = _slug_cik(key)
    j: Optional[Dict[str, Any]] = None

    # 1) si on a déjà un CIK
    if cik:
        try:
            j = _get(f"{SEC_API}/submissions/CIK{cik}.json", headers=_sec_headers(), use_cache=use_cache, as_json=True)
        except Exception:
            j = None

    # 2) sinon on tente tickers mapping via 'companyfacts' échec → fallback à /submissions pour guess (lourd)
    if j is None:
        # petit hack: la page /submissions pour ticker renvoie 404; on parcourt l'index 'tickers' si dispo
        # On tente une simple page 'company-concepts' pour ticker (échoue 404 sauf CIK) -> skip
        return {"ok": False, "error": "CIK required or ticker->CIK mapping unavailable without external db", "query": key}

    # normalize
    try:
        prof = {
            "cik": _cik_from_submissions_json(j),
            "ticker": j.get("tickers", [None])[0] if j.get("tickers") else None,
            "name": j.get("name"),
            "sic": j.get("sic"),
            "sicDescription": j.get("sicDescription"),
            "stateOfIncorporation": j.get("stateOfIncorporation"),
            "filings_count": j.get("filings",{}).get("recent",{}).get("accessionNumber",[]).__len__(),
            "tickers": j.get("tickers") or [],
        }
        return {"ok": True, "profile": prof, "raw": j}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# =====================
# SEC - Filings index
# =====================

def sec_filings_index(cik_or_ticker: str, forms: List[str] = ["10-K","10-Q","8-K"], 
                      limit: int = 100, use_cache=True) -> List[Dict[str, Any]]:
    """
    A partir de submissions.json: reconstruit tableau des filings récents et filtre par 'forms'.
    """
    sub = sec_submissions(cik_or_ticker, use_cache=use_cache)
    if not sub.get("ok"):
        return []
    raw = sub.get("raw", {})
    recent = raw.get("filings",{}).get("recent",{})
    acc = recent.get("accessionNumber", [])
    form = recent.get("form", [])
    filings: List[Dict[str, Any]] = []
    for i in range(min(len(acc), len(form))):
        if forms and form[i] not in forms:
            continue
        accn = acc[i].replace("-", "")
        cik = _slug_cik(str(raw.get("cik")))
        primary = recent.get("primaryDocument", [""]*len(acc))[i]
        filingDate = recent.get("filingDate", [""]*len(acc))[i]
        periodOfReport = recent.get("reportDate", [""]*len(acc))[i] or None
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/{primary}"
        filings.append({
            "cik": cik, "form": form[i], "filingDate": filingDate, "periodOfReport": periodOfReport,
            "accession": acc[i], "primaryDoc": primary, "url": url
        })
        if len(filings) >= limit:
            break
    return filings

# =====================
# SEC - Form 4 (insiders)
# =====================

def _sec_owner_feed(cik: str, limit: int = 200, use_cache=True) -> str:
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "CIK": cik,
        "type": "4",
        "owner": "only",
        "count": str(limit),
        "output": "atom",
    }
    return _get(url, params=params, headers={"User-Agent": SEC_UA, "Accept": "application/atom+xml"},
                use_cache=use_cache, as_text=True)

def _parse_atom_owner(atom_xml: str) -> List[Dict[str, Any]]:
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(atom_xml, "xml")
    entries = soup.find_all("entry")
    out = []
    for e in entries:
        title = _strip_html(e.title.text if e.title else "")
        link = e.link.get("href") if e.link else None
        updated = e.updated.text if e.updated else None
        summary = _strip_html(e.summary.text if e.summary else "")
        # deviner sens achat/vente depuis le titre/summary (best effort)
        # e.g., "Statement of changes in beneficial ownership of securities"
        # On cherchera "A" (acquired) vs "D" (disposed) dans des tables plus tard si on télécharge le XML détaillé.
        out.append({
            "title": title, "link": link, "updated": updated, "summary": summary
        })
    return out

def sec_form4_insiders(cik_or_ticker: str, limit: int = 200, use_cache=True) -> Dict[str, Any]:
    """
    Retourne {transactions:[...], aggregates:{window_30d:{buys, sells, net_shares, net_value}, window_90d:...}}
    NB: Sans suivre chaque 'primaryDoc.xml', on reste best-effort; pour la plupart des cas, l'Atom suffit pour signaux grossiers.
    """
    sub = sec_submissions(cik_or_ticker, use_cache=use_cache)
    if not sub.get("ok"):
        return {"ok": False, "error": "submissions not found"}
    cik = sub["profile"]["cik"]

    try:
        feed = _sec_owner_feed(cik, limit=limit, use_cache=use_cache)
        ents = _parse_atom_owner(feed)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Heuristiques légères: si 'sale'/'sell' -> vente; 'purchase'/'buy' -> achat
    txs = []
    for x in ents:
        kind = None
        low = (x.get("summary","") + " " + x.get("title","")).lower()
        if re.search(r"\bbuy|purchase|acquir", low):
            kind = "BUY"
        elif re.search(r"\bsell|dispos", low):
            kind = "SELL"
        txs.append({
            "title": x["title"], "link": x["link"], "updated": x["updated"],
            "detected_kind": kind, "summary": x["summary"]
        })

    # Aggrégation simple par fenêtre (sans montants si XML non parsé)
    def _window(days: int):
        cutoff = _now() - dt.timedelta(days=days)
        buys = sum(1 for t in txs if t.get("detected_kind") == "BUY" and _ts_ok(t.get("updated"), cutoff))
        sells = sum(1 for t in txs if t.get("detected_kind") == "SELL" and _ts_ok(t.get("updated"), cutoff))
        return {"buys": buys, "sells": sells, "net_trades": buys - sells}
    aggr = {"window_30d": _window(30), "window_90d": _window(90)}

    return {"ok": True, "cik": cik, "transactions": txs, "aggregates": aggr, "asof_utc": _iso(_now())}

def _ts_ok(ts_iso: Optional[str], cutoff: dt.datetime) -> bool:
    if not ts_iso: 
        return False
    try:
        t = dt.datetime.fromisoformat(ts_iso.replace("Z","+00:00"))
        return t >= cutoff
    except Exception:
        return False

# =====================
# SEC - 13F holdings
# =====================

def _download_text(url: str, use_cache=True) -> str:
    return _get(url, headers={"User-Agent": SEC_UA}, use_cache=use_cache, as_text=True)

def _download_raw(url: str, use_cache=True) -> bytes:
    return _get(url, headers={"User-Agent": SEC_UA}, use_cache=use_cache, raw=True)

def _guess_13f_table_links(filing_url: str, html: Optional[str] = None) -> List[str]:
    """
    A partir de l'URL du primaryDoc (ex: .../primary_doc.html), retrouver liens 'information table' XML/HTML.
    """
    try:
        page = html or _download_text(filing_url)
    except Exception:
        return []
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(page, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"(informationtable|infotable|form13fInfoTable)\.(xml|html|htm)$", href, re.I):
            # résoudre lien relatif
            base = filing_url.rsplit("/", 1)[0]
            if href.startswith("http"):
                links.append(href)
            else:
                links.append(f"{base}/{href}")
    # fallback: essayer list.json (si dispo) pour repérer un xml
    if not links:
        try:
            base = filing_url.rsplit("/", 1)[0]
            j = _get(f"{base}/index.json", headers={"User-Agent": SEC_UA}, as_json=True)
            for it in j.get("directory", {}).get("item", []):
                n = it.get("name", "")
                if re.search(r"(informationtable|infotable|form13fInfoTable)\.(xml|html|htm)$", n, re.I):
                    links.append(f"{base}/{n}")
        except Exception:
            pass
    return links

def _parse_13f_xml(xml_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Parsing simplifié du XML 13F 'informationTable' (sélection des champs clés).
    """
    # Eviter dépendances lourdes -> parse via BeautifulSoup xml si dispo
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(xml_bytes, "xml")
    rows = []
    for inf in soup.find_all(["infoTable","informationTable"]):
        name = _strip_html((inf.find("nameOfIssuer") or {}).get_text() if inf.find("nameOfIssuer") else "")
        cusip = _strip_html((inf.find("cusip") or {}).get_text() if inf.find("cusip") else "")
        val  = _to_float((inf.find("value") or {}).get_text() if inf.find("value") else None)
        shrs = _to_float((inf.find("sshPrnamt") or {}).get_text() if inf.find("sshPrnamt") else None)
        shType = _strip_html((inf.find("sshPrnamtType") or {}).get_text() if inf.find("sshPrnamtType") else "")
        putCall = _strip_html((inf.find("putCall") or {}).get_text() if inf.find("putCall") else "")
        rows.append({
            "issuer": name, "cusip": cusip, "value_usd_thousands": val,
            "shares": shrs, "share_type": shType, "put_call": putCall
        })
    return rows

def sec_13f_holdings(cik_or_ticker: str, limit_filings: int = 1, use_cache=True) -> Dict[str, Any]:
    """
    Liste les dernières 13F-HR et agrège le(s) infoTable en un tableau 'holdings'.
    """
    subs = sec_filings_index(cik_or_ticker, forms=["13F-HR","13F-HR/A"], limit=limit_filings, use_cache=use_cache)
    if not subs:
        return {"ok": False, "error": "No 13F filings found"}
    all_holdings = []
    filing_meta = []
    for f in subs:
        url = f["url"]
        try:
            html = _download_text(url, use_cache=use_cache)
            links = _guess_13f_table_links(url, html=html)
            got = False
            for lk in links:
                try:
                    raw = _download_raw(lk, use_cache=use_cache)
                    rows = _parse_13f_xml(raw)
                    all_holdings.extend(rows)
                    got = True
                except Exception:
                    continue
            filing_meta.append({"form": f["form"], "filingDate": f["filingDate"], "url": url, "tables_found": len(links), "parsed": got})
        except Exception:
            filing_meta.append({"form": f["form"], "filingDate": f["filingDate"], "url": url, "tables_found": 0, "parsed": False})

    # petit tri et uniques par CUSIP/issuer
    keyset = set(); uniq = []
    for r in all_holdings:
        k = (r.get("cusip"), r.get("issuer"))
        if k not in keyset:
            keyset.add(k); uniq.append(r)

    return {"ok": True, "filings": filing_meta, "holdings": uniq, "count": len(uniq), "asof_utc": _iso(_now())}

# ==========================
# Ownership/Insider snapshot
# ==========================

def build_ownership_snapshot(ticker: str, use_cache=True) -> Dict[str, Any]:
    """
    assemble: yahoo (price/funda), options expiries, sec filings head, insiders (Form4 agg), 13F last
    """
    t = ticker.strip().upper()
    y = yahoo_snapshot(t, use_cache=use_cache)
    # essayer SEC submissions avec ticker ou CIK
    sec_prof = sec_submissions(t, use_cache=use_cache)
    filings = sec_filings_index(t, forms=["10-K","10-Q","8-K"], limit=50, use_cache=use_cache) if sec_prof.get("ok") else []
    insiders = sec_form4_insiders(t, limit=200, use_cache=use_cache) if sec_prof.get("ok") else {"ok": False}
    hf = sec_13f_holdings(t, limit_filings=1, use_cache=use_cache)

    # options (seulement expiries si pas yfinance)
    opts = yahoo_options_chain(t, expiry=None, use_cache=use_cache)

    # petit score d’alerte insiders: (buys - sells) 30j
    alert = None
    try:
        a30 = insiders.get("aggregates",{}).get("window_30d",{})
        if a30:
            b, s = a30.get("buys", 0), a30.get("sells", 0)
            alert = {"insider_30d_net_trades": (b - s), "note": "positive if net insider buying"}
    except Exception:
        pass

    return {
        "ok": True,
        "ticker": t,
        "asof_utc": _iso(_now()),
        "yahoo": y,
        "sec_profile": sec_prof.get("profile") if sec_prof.get("ok") else None,
        "filings_head": filings,
        "insiders": insiders,
        "thirteenF": hf,
        "options_overview": {
            "expiries": opts.get("expiries"),
            "chosen_expiry": opts.get("chosen_expiry"),
            "calls_count": len(opts.get("calls",[])),
            "puts_count": len(opts.get("puts",[])),
        },
        "signals": {"insider_alert": alert},
        "sources_used": ["YahooFinance", "SEC EDGAR"],
        "notes": "Best-effort; pour Form 4 détaillé et montants, parser les XML ownership filing individuels."
    }

# =========
# CLI tools
# =========

def _print(obj: Any):
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Financials & Ownership client (Yahoo + SEC filings/Form4/13F)")
    sub = ap.add_subparsers(dest="cmd")

    # yahoo
    ap_y = sub.add_parser("yahoo", help="Yahoo snapshot")
    ap_y.add_argument("--ticker", required=True)
    ap_y.add_argument("--no_cache", action="store_true")

    # options
    ap_o = sub.add_parser("options", help="Yahoo options chain")
    ap_o.add_argument("--ticker", required=True)
    ap_o.add_argument("--expiry", default=None, help="YYYY-MM-DD (optional)")
    ap_o.add_argument("--no_cache", action="store_true")

    # submissions (profile)
    ap_s = sub.add_parser("submissions", help="SEC submissions profile")
    ap_s.add_argument("--cik_or_ticker", required=True)
    ap_s.add_argument("--no_cache", action="store_true")

    # filings
    ap_f = sub.add_parser("filings", help="SEC filings index (10-K/Q, 8-K)")
    ap_f.add_argument("--cik_or_ticker", required=True)
    ap_f.add_argument("--forms", default="10-K,10-Q,8-K")
    ap_f.add_argument("--limit", type=int, default=100)
    ap_f.add_argument("--no_cache", action="store_true")

    # form4
    ap_form4 = sub.add_parser("form4", help="SEC Form 4 (insiders)")
    ap_form4.add_argument("--cik_or_ticker", required=True)
    ap_form4.add_argument("--limit", type=int, default=200)
    ap_form4.add_argument("--no_cache", action="store_true")

    # 13F
    ap_13f = sub.add_parser("13f", help="SEC 13F holdings")
    ap_13f.add_argument("--cik_or_ticker", required=True)
    ap_13f.add_argument("--limit_filings", type=int, default=1)
    ap_13f.add_argument("--no_cache", action="store_true")

    # snapshot
    ap_snap = sub.add_parser("snapshot", help="Ownership/Insider snapshot")
    ap_snap.add_argument("--ticker", required=True)
    ap_snap.add_argument("--no_cache", action="store_true")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help(); sys.exit(0)

    if args.cmd == "yahoo":
        _print(yahoo_snapshot(args.ticker, use_cache=not args.no_cache)); return

    if args.cmd == "options":
        _print(yahoo_options_chain(args.ticker, expiry=args.expiry, use_cache=not args.no_cache)); return

    if args.cmd == "submissions":
        _print(sec_submissions(args.cik_or_ticker, use_cache=not args.no_cache)); return

    if args.cmd == "filings":
        forms = [x.strip() for x in args.forms.split(",") if x.strip()]
        _print(sec_filings_index(args.cik_or_ticker, forms=forms, limit=args.limit, use_cache=not args.no_cache)); return

    if args.cmd == "form4":
        _print(sec_form4_insiders(args.cik_or_ticker, limit=args.limit, use_cache=not args.no_cache)); return

    if args.cmd == "13f":
        _print(sec_13f_holdings(args.cik_or_ticker, limit_filings=args.limit_filings, use_cache=not args.no_cache)); return

    if args.cmd == "snapshot":
        _print(build_ownership_snapshot(args.ticker, use_cache=not args.no_cache)); return

# helpers

if __name__ == "__main__":
    main()