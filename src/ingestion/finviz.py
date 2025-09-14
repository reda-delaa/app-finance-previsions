# src/ingestion/providers/finviz.py
# -*- coding: utf-8 -*-
"""
Finviz provider (HTML parsing, no auth)
Couvre:
- Global news:        https://finviz.com/news.ashx?v=2
- Company news:       https://finviz.com/quote.ashx?t=<TICKER>
- Insider (global):   https://finviz.com/insidertrading.ashx (et variantes)
- Insider (company):  https://finviz.com/quote.ashx?t=<TICKER> (bloc News/Insider)
- Analyst ratings:    https://finviz.com/quote.ashx?t=<TICKER> (bloc Upgrades/Downgrades)
- Snapshot métriques: https://finviz.com/quote.ashx?t=<TICKER> (tableau “snapshot”)
- Détention instits:  https://finviz.com/quote.ashx?t=<TICKER> (table holders, si présent)

Sorties : listes de dict “bruts” compatibles avec la couche de normalisation finnews
 (title, link, published ISO Z, summary, raw_text, source, _id) + champs spécifiques “kind”.
"""

from __future__ import annotations
import re, time, hashlib, datetime as dt
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup

FINVIZ_ROOT = "https://finviz.com/"
FINVIZ_NEWS_V2 = urljoin(FINVIZ_ROOT, "news.ashx?v=2")
FINVIZ_QUOTE = urljoin(FINVIZ_ROOT, "quote.ashx")
FINVIZ_INSIDER = urljoin(FINVIZ_ROOT, "insidertrading.ashx")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://finviz.com/",
}

_REL_TIME_RE = re.compile(r"(?:(\d+)\s*(?:min|mins|minutes?)|(\d+)\s*h(?:ours?)?|(\d+)\s*d(?:ays?)?)\s*ago", re.I)
_ABS_TIME_RE = re.compile(r"[A-Za-z]{3}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?:AM|PM)", re.I)

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _to_iso_z(t: dt.datetime) -> str:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _parse_relative_or_abs_time(txt: str, now: Optional[dt.datetime] = None) -> Optional[dt.datetime]:
    if not txt:
        return None
    now = now or _now_utc()
    m = _REL_TIME_RE.search(txt)
    if m:
        mins = m.group(1)
        hours = m.group(2)
        days = m.group(3)
        delta = dt.timedelta(
            minutes=int(mins) if mins else 0,
            hours=int(hours) if hours else 0,
            days=int(days) if days else 0
        )
        return now - delta
    m2 = _ABS_TIME_RE.search(txt)
    if m2:
        raw = m2.group(0)
        for pat in ("%b-%d-%y %I:%M%p", "%b-%d-%Y %I:%M%p"):
            try:
                dt_naive = dt.datetime.strptime(raw, pat)
                # NB : Finviz affiche souvent en ET; on pose UTC par prudence (ou adapter ici).
                return dt_naive.replace(tzinfo=dt.timezone.utc)
            except Exception:
                pass
    return None

def _compact_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

class FinvizClient:
    def __init__(self, timeout: int = 20, sleep_between: float = 0.6, headers: Optional[Dict[str, str]] = None):
        self.sess = requests.Session()
        self.sess.headers.update(headers or _DEFAULT_HEADERS)
        self.timeout = timeout
        self.sleep = sleep_between

    def get(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        r = self.sess.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        time.sleep(self.sleep)  # politeness
        return r

    # ------------- NEWS (global + company) -------------

    def fetch_global_news(self, limit: int = 200) -> List[Dict]:
        """
        Parse https://finviz.com/news.ashx?v=2
        Retourne : [{title, link, published, summary, raw_text, source, _id, kind='news'}]
        """
        resp = self.get(FINVIZ_NEWS_V2)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Heuristique : attraper tous les liens avec contexte de temps autour
        rows = []
        for a in soup.find_all("a", href=True):
            title = _compact_spaces(a.get_text(" ", strip=True))
            href = a["href"]
            if not title or not href:
                continue
            if href.startswith("#") or "javascript:void" in href.lower():
                continue
            # contexte de temps dans siblings / parent
            around = " | ".join(
                _compact_spaces(x.get_text(" ", strip=True))
                for x in [a.parent, getattr(a.parent, "previous_sibling", None), getattr(a.parent, "next_sibling", None)]
                if hasattr(x, "get_text")
            )
            time_hint = ""
            mrel = _REL_TIME_RE.search(around or "")
            mabs = _ABS_TIME_RE.search(around or "")
            if mrel: time_hint = mrel.group(0)
            elif mabs: time_hint = mabs.group(0)
            rows.append({"title": title, "href": href, "time_hint": time_hint})

        now = _now_utc()
        out = []
        for r in rows[:limit]:
            link = r["href"]
            if link.startswith("/"):
                link = urljoin(FINVIZ_ROOT, link)
            t = _parse_relative_or_abs_time(r.get("time_hint", ""), now=now) or now
            iso = _to_iso_z(t)
            item = {
                "title": r["title"],
                "link": link,
                "published": iso,
                "summary": "",
                "raw_text": "",
                "source": "finviz/news",
                "kind": "news",
            }
            item["_id"] = _sha1(f"{item['source']}|{item['title']}|{item['published']}|{item['link']}")
            out.append(item)
        out.sort(key=lambda x: x["published"], reverse=True)
        return out

    def fetch_company_news(self, ticker: str, limit: int = 120) -> List[Dict]:
        """
        Parse https://finviz.com/quote.ashx?t=<TICKER> (bloc News)
        """
        ticker = (ticker or "").upper().strip()
        if not ticker:
            return []
        resp = self.get(FINVIZ_QUOTE, params={"t": ticker})
        soup = BeautifulSoup(resp.text, "html.parser")

        # Cherche des sections contenant "News"
        candidates = []
        for tag in soup.find_all(["table", "div", "section", "td", "span"]):
            txt = _compact_spaces(tag.get_text(" ", strip=True)).lower()
            if "news" in txt[:20]:  # titre ou en-tête proche
                candidates.append(tag)
        if not candidates:
            candidates = [soup]

        rows = []
        for c in candidates:
            for a in c.find_all("a", href=True):
                title = _compact_spaces(a.get_text(" ", strip=True))
                href = a["href"]
                if not title or not href:
                    continue
                if href.startswith("#") or "javascript:void" in href.lower():
                    continue
                around = " | ".join(
                    _compact_spaces(x.get_text(" ", strip=True))
                    for x in [a.parent, getattr(a.parent, "previous_sibling", None), getattr(a.parent, "next_sibling", None)]
                    if hasattr(x, "get_text")
                )
                time_hint = ""
                mrel = _REL_TIME_RE.search(around or "")
                mabs = _ABS_TIME_RE.search(around or "")
                if mrel: time_hint = mrel.group(0)
                elif mabs: time_hint = mabs.group(0)
                rows.append({"title": title, "href": href, "time_hint": time_hint})

        now = _now_utc()
        out = []
        for r in rows[:limit]:
            link = r["href"]
            if link.startswith("/"):
                link = urljoin(FINVIZ_ROOT, link)
            t = _parse_relative_or_abs_time(r.get("time_hint", ""), now=now) or now
            iso = _to_iso_z(t)
            item = {
                "title": r["title"],
                "link": link,
                "published": iso,
                "summary": "",
                "raw_text": "",
                "source": f"finviz/quote?t={ticker}",
                "kind": "news",
                "ticker": ticker,
            }
            item["_id"] = _sha1(f"{item['source']}|{item['title']}|{item['published']}|{item['link']}")
            out.append(item)
        out.sort(key=lambda x: x["published"], reverse=True)
        return out

    # ------------- INSIDER TRADING (global + company) -------------

    def fetch_insider_recent(self, limit: int = 200) -> List[Dict]:
        """
        Parse des tables globales d'insiders.
        Pages typiques: https://finviz.com/insidertrading.ashx (ou avec params)
        Retour: [{kind='insider', action, owner, relationship, ticker, date, shares, price, value, link, source, _id}]
        """
        resp = self.get(FINVIZ_INSIDER)
        soup = BeautifulSoup(resp.text, "html.parser")

        out = []
        for table in soup.find_all("table"):
            headers = [ _compact_spaces(th.get_text(" ", strip=True)).lower() for th in table.find_all("th") ]
            # Heuristique: une table insider a des colonnes "ticker", "owner", "relationship", "transaction", "date", "shares", "price", "value", ...
            if not headers:
                continue
            score = sum(int(any(k in h for k in ["ticker","owner","relationship","price","value","shares","date","transaction"])) for h in headers)
            if score < 4:
                continue
            # Parse rows
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 5:
                    continue
                cells = [ _compact_spaces(td.get_text(" ", strip=True)) for td in tds ]
                href = ""
                a0 = tds[0].find("a", href=True) or tds[1].find("a", href=True)
                if a0:
                    href = a0["href"]
                    if href.startswith("/"): href = urljoin(FINVIZ_ROOT, href)

                rec = {
                    "kind": "insider",
                    "source": "finviz/insider",
                    "link": href,
                    "title": f"Insider: {' | '.join(cells[:4])}",
                    "summary": " | ".join(cells),
                    "raw_text": " | ".join(cells),
                    # mappages best-effort
                    "ticker": next((c for c in cells if re.fullmatch(r"[A-Z]{1,5}", c)), None),
                    "owner": next((c for c in cells if "director" in c.lower() or "officer" in c.lower() or "beneficial" in c.lower()), None),
                    "relationship": next((c for c in cells if "director" in c.lower() or "officer" in c.lower()), None),
                }
                # Date (YYYY-MM-DD si possible)
                date_str = next((c for c in cells if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", c)), "")
                dt_parsed = None
                for pat in ("%m/%d/%y", "%m/%d/%Y"):
                    try:
                        dt_parsed = dt.datetime.strptime(date_str, pat).replace(tzinfo=dt.timezone.utc)
                        break
                    except Exception:
                        pass
                rec["published"] = _to_iso_z(dt_parsed) if dt_parsed else _to_iso_z(_now_utc())

                # Shares / Price / Value
                sh = next((c for c in cells if re.fullmatch(r"[0-9,]+", c)), "")
                pr = next((c for c in cells if re.match(r"\$?\d[\d,]*\.?\d*", c)), "")
                val = None
                mval = re.search(r"\$[\d,]+", " ".join(cells))
                if mval: val = mval.group(0)
                rec["shares"] = sh.replace(",", "") if sh else None
                rec["price"]  = pr.replace("$", "").replace(",", "") if pr else None
                rec["value"]  = (val or "").replace("$", "").replace(",", "") or None

                rec["_id"] = _sha1(f"{rec['source']}|{rec.get('ticker')}|{rec['summary']}|{rec['published']}")
                out.append(rec)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out

    def fetch_company_insiders(self, ticker: str, limit: int = 80) -> List[Dict]:
        """
        Gratte la page quote et tente d'extraire un tableau 'Insider' pour ce ticker (quand présent).
        """
        ticker = (ticker or "").upper().strip()
        if not ticker:
            return []
        resp = self.get(FINVIZ_QUOTE, params={"t": ticker})
        soup = BeautifulSoup(resp.text, "html.parser")

        out = []
        for table in soup.find_all("table"):
            headers = [ _compact_spaces(th.get_text(" ", strip=True)).lower() for th in table.find_all("th") ]
            if not headers:
                continue
            score = sum(int(any(k in h for k in ["insider","owner","relationship","transaction","date","shares","price","value"])) for h in headers)
            if score < 2:
                continue
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 4:
                    continue
                cells = [ _compact_spaces(td.get_text(" ", strip=True)) for td in tds ]
                item = {
                    "kind": "insider",
                    "source": f"finviz/quote?t={ticker}",
                    "link": resp.url,
                    "ticker": ticker,
                    "title": f"Insider {ticker}: {' | '.join(cells[:4])}",
                    "summary": " | ".join(cells),
                    "raw_text": " | ".join(cells),
                    "published": _to_iso_z(_now_utc()),
                }
                item["_id"] = _sha1(f"{item['source']}|{item['summary']}|{item['published']}")
                out.append(item)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out

    # ------------- ANALYST RATINGS (company) -------------

    def fetch_company_ratings(self, ticker: str, limit: int = 60) -> List[Dict]:
        """
        Repère un bloc 'Upgrades/Downgrades' ou colonnes avec 'Analyst', 'Rating', 'Price Target' sur la page quote.
        Retour: [{kind='rating', ticker, analyst, action, rating_from, rating_to, pt_from, pt_to, published, source, link, _id}]
        """
        ticker = (ticker or "").upper().strip()
        if not ticker:
            return []
        resp = self.get(FINVIZ_QUOTE, params={"t": ticker})
        soup = BeautifulSoup(resp.text, "html.parser")

        out = []
        # Cherche des lignes évoquant upgrades/downgrades
        for table in soup.find_all("table"):
            headers = [ _compact_spaces(th.get_text(" ", strip=True)).lower() for th in table.find_all("th") ]
            header_txt = " ".join(headers)
            if any(k in header_txt for k in ["upgrade","downgrade","analyst","rating","price target","pt"]):
                for tr in table.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 3:
                        continue
                    cells = [ _compact_spaces(td.get_text(" ", strip=True)) for td in tds ]
                    row = " | ".join(cells)
                    # Date (si visible, ex: 09/12/25)
                    published = _to_iso_z(_now_utc())
                    mdate = re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", row)
                    if mdate:
                        parsed = None
                        for pat in ("%m/%d/%y", "%m/%d/%Y"):
                            try:
                                parsed = dt.datetime.strptime(mdate.group(0), pat).replace(tzinfo=dt.timezone.utc)
                                break
                            except Exception:
                                pass
                        if parsed:
                            published = _to_iso_z(parsed)

                    item = {
                        "kind": "rating",
                        "ticker": ticker,
                        "analyst": next((c for c in cells if re.search(r"(JP|J\.P\.)? Morgan|Goldman|UBS|Citi|BofA|Barclays|Jefferies|Wells|RBC|HSBC|Deutsche|Credit Suisse|BNP|SocGen|Oddo|Morningstar", c, re.I)), None),
                        "action": next((c for c in cells if re.search(r"upgrade|downgrade|initiates|reiterates|maintains", c, re.I)), None),
                        "rating_from": None,
                        "rating_to": None,
                        "pt_from": None,
                        "pt_to": None,
                        "title": f"Rating {ticker}: {row}",
                        "summary": row,
                        "raw_text": row,
                        "published": published,
                        "source": f"finviz/quote?t={ticker}",
                        "link": resp.url,
                    }
                    # Essai extraction PT “from->to” et rating “from->to”
                    mpt = re.search(r"(\$?\d[\d,]*\.?\d*)\s*->\s*(\$?\d[\d,]*\.?\d*)", row)
                    if mpt:
                        item["pt_from"] = mpt.group(1).replace("$", "").replace(",", "")
                        item["pt_to"]   = mpt.group(2).replace("$", "").replace(",", "")
                    mrat = re.search(r"(\bBuy|Hold|Sell|Neutral|Overweight|Underweight|Outperform|Underperform)\b.*->.*(\bBuy|Hold|Sell|Neutral|Overweight|Underweight|Outperform|Underperform)\b", row, re.I)
                    if mrat:
                        item["rating_from"] = mrat.group(1)
                        item["rating_to"]   = mrat.group(2)

                    item["_id"] = _sha1(f"{item['source']}|{item['summary']}|{item['published']}")
                    out.append(item)
                    if len(out) >= limit:
                        return out
        return out

    # ------------- SNAPSHOT METRICS (company) -------------

    def fetch_company_snapshot(self, ticker: str) -> Dict:
        """
        Récupère le tableau “snapshot” (pairs label: value) sur la quote.
        Retour: dict {kind='snapshot', ticker, metrics:{...}, source, link, _id, published}
        """
        ticker = (ticker or "").upper().strip()
        if not ticker:
            return {}
        resp = self.get(FINVIZ_QUOTE, params={"t": ticker})
        soup = BeautifulSoup(resp.text, "html.parser")

        metrics: Dict[str, str] = {}
        # Heuristique: tables avec beaucoup de cellules Label/Value
        for table in soup.find_all("table"):
            tds = table.find_all("td")
            if len(tds) < 12:
                continue
            # Parcours par paires
            for i in range(0, len(tds)-1, 2):
                key = _compact_spaces(tds[i].get_text(" ", strip=True))
                val = _compact_spaces(tds[i+1].get_text(" ", strip=True))
                if key and val and len(key) <= 32:
                    # Exemples d'intérêts : P/E, P/S, PEG, Debt/Eq, ROE, Margin, EPS next Y, Sales Q/Q, EPS Q/Q,
                    # Beta, ATR, SMA20/50/200, Perf W/M/Q/Y, Short Float, Float, Insider Own/Trans, Inst Own/Trans...
                    metrics[key] = val

        item = {
            "kind": "snapshot",
            "ticker": ticker,
            "metrics": metrics,
            "source": f"finviz/quote?t={ticker}",
            "link": resp.url,
            "published": _to_iso_z(_now_utc()),
            "title": f"Snapshot {ticker}",
            "summary": "",
            "raw_text": "",
        }
        item["_id"] = _sha1(f"{item['source']}|{ticker}|{len(metrics)}")
        return item

    # ------------- OWNERSHIP (institutionnel) -------------

    def fetch_company_institutions(self, ticker: str, limit: int = 120) -> List[Dict]:
        """
        Tente de trouver un tableau “Holders / Institutional” dans la page quote.
        Retour: [{kind='institution', ticker, holder, position, pct, change, date, source, link, _id, published}]
        """
        ticker = (ticker or "").upper().strip()
        if not ticker:
            return []
        resp = self.get(FINVIZ_QUOTE, params={"t": ticker})
        soup = BeautifulSoup(resp.text, "html.parser")

        out = []
        for table in soup.find_all("table"):
            headers = [ _compact_spaces(th.get_text(" ", strip=True)).lower() for th in table.find_all("th") ]
            if not headers:
                continue
            # Cherche “institution” / “holder” / “shares” / “%”
            header_txt = " ".join(headers)
            score = sum(int(k in header_txt for k in ["institution","holder","shares","%","position","change"]))
            if score < 2:
                continue
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 3:
                    continue
                cells = [ _compact_spaces(td.get_text(" ", strip=True)) for td in tds ]
                holder = cells[0]
                pos = next((c for c in cells if re.search(r"[\d,]+\s*(?:sh|shares)?", c, re.I)), None)
                pct = next((c for c in cells if re.search(r"\d+(\.\d+)?\s*%", c)), None)
                chg = next((c for c in cells if re.search(r"[+\-]?\d+(\.\d+)?\s*%", c)), None)

                item = {
                    "kind": "institution",
                    "ticker": ticker,
                    "holder": holder,
                    "position": (pos or "").replace("shares", "").replace("sh", "").strip(),
                    "pct": (pct or "").replace("%", "").strip() or None,
                    "change": (chg or "").replace("%", "").strip() or None,
                    "source": f"finviz/quote?t={ticker}",
                    "link": resp.url,
                    "published": _to_iso_z(_now_utc()),
                    "title": f"Institution {ticker}: {holder}",
                    "summary": " | ".join(cells),
                    "raw_text": " | ".join(cells),
                }
                item["_id"] = _sha1(f"{item['source']}|{item['holder']}|{item['summary']}")
                out.append(item)
                if len(out) >= limit:
                    return out
        return out


# --------- Helpers de haut niveau (APIs simples) ---------

def fetch_finviz_global_news(limit: int = 200, timeout: int = 20, sleep_between: float = 0.6) -> List[Dict]:
    cli = FinvizClient(timeout=timeout, sleep_between=sleep_between)
    return cli.fetch_global_news(limit=limit)

def fetch_finviz_company_news(ticker: str, limit: int = 120, timeout: int = 20, sleep_between: float = 0.6) -> List[Dict]:
    cli = FinvizClient(timeout=timeout, sleep_between=sleep_between)
    return cli.fetch_company_news(ticker=ticker, limit=limit)

def fetch_finviz_insider_recent(limit: int = 200, timeout: int = 20, sleep_between: float = 0.6) -> List[Dict]:
    cli = FinvizClient(timeout=timeout, sleep_between=sleep_between)
    return cli.fetch_insider_recent(limit=limit)

def fetch_finviz_company_insiders(ticker: str, limit: int = 80, timeout: int = 20, sleep_between: float = 0.6) -> List[Dict]:
    cli = FinvizClient(timeout=timeout, sleep_between=sleep_between)
    return cli.fetch_company_insiders(ticker=ticker, limit=limit)

def fetch_finviz_company_ratings(ticker: str, limit: int = 60, timeout: int = 20, sleep_between: float = 0.6) -> List[Dict]:
    cli = FinvizClient(timeout=timeout, sleep_between=sleep_between)
    return cli.fetch_company_ratings(ticker=ticker, limit=limit)

def fetch_finviz_company_snapshot(ticker: str, timeout: int = 20, sleep_between: float = 0.6) -> Dict:
    cli = FinvizClient(timeout=timeout, sleep_between=sleep_between)
    return cli.fetch_company_snapshot(ticker=ticker)

def fetch_finviz_company_institutions(ticker: str, limit: int = 120, timeout: int = 20, sleep_between: float = 0.6) -> List[Dict]:
    cli = FinvizClient(timeout=timeout, sleep_between=sleep_between)
    return cli.fetch_company_institutions(ticker=ticker, limit=limit)