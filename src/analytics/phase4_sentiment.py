# phase4_sentiment.py
# -*- coding: utf-8 -*-
"""
Phase 4 — Sentiment & News / NLP pour actions US/CA

Sources "sans clé":
  - yfinance.Ticker.news  (titres + liens + éditeur + datetime)
  - (Optionnel) RSS SEC / SEDAR+ si vous fournissez l'URL RSS
  - (Optionnel) Texte libre passé par l'API (ex: dépêches que vous scrapez)

Pipeline:
  fetch -> clean -> dedupe -> score_sentiment (VADER + HF si dispo) ->
  summarize (TextRank light) -> extract_events -> aggregate (daily/weekly) ->
  produce signals (news shock, drift, risk flags)

Auteur: toi + IA (2025) — Licence MIT (à adapter)
"""

from __future__ import annotations
import re
import math
import time
import json
import hashlib
import html
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# -------- Optional NLP backends -------- #
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    HAS_VADER = True
except Exception:
    HAS_VADER = False

try:
    from transformers import pipeline  # type: ignore
    HF_PIPE = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    HAS_HF = True
except Exception:
    HAS_HF = False

# ---------------------- Dataclasses ---------------------- #

@dataclass
class NewsItem:
    ticker: str
    title: str
    summary: str
    source: str
    url: str
    published: pd.Timestamp
    raw_text: Optional[str] = None
    tags: Optional[List[str]] = None

    def key(self) -> str:
        base = f"{self.ticker}|{self.title}|{self.source}|{self.url}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()


@dataclass
class SentimentDetail:
    vader: Optional[float]
    hf: Optional[float]
    ensemble: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vader": None if self.vader is None else float(self.vader),
            "hf": None if self.hf is None else float(self.hf),
            "ensemble": float(self.ensemble),
        }


@dataclass
class EventSignal:
    """Événements structurés extraits d’un article."""
    type: str           # 'earnings', 'guidance_up', 'mna', 'litigation', 'downgrade', ...
    strength: float     # 0..1
    polarity: int       # -1, 0, +1
    evidence: str       # court extrait

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "strength": float(self.strength), "polarity": int(self.polarity), "evidence": self.evidence}


@dataclass
class ScoredNews:
    item: NewsItem
    sentiment: SentimentDetail
    events: List[EventSignal]
    importance: float   # poids (source, récence, longueur, tags)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": asdict(self.item),
            "sentiment": self.sentiment.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "importance": float(self.importance),
        }


@dataclass
class AggregateSentiment:
    """
    Agrégations par jour/semaine.
    """
    by_day: pd.DataFrame
    by_week: pd.DataFrame
    shock_score: float
    drift_score: float
    risk_flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "shock_score": float(self.shock_score),
            "drift_score": float(self.drift_score),
            "risk_flags": list(self.risk_flags),
        }
        if not self.by_day.empty:
            out["by_day_cols"] = list(self.by_day.columns)
            out["by_day_rows"] = int(len(self.by_day))
        if not self.by_week.empty:
            out["by_week_cols"] = list(self.by_week.columns)
            out["by_week_rows"] = int(len(self.by_week))
        return out


# ---------------------- Fetchers ---------------------- #

def fetch_yf_news(ticker: str, max_items: int = 40) -> List[NewsItem]:
    """
    Récupère des news via yfinance (titres + lien). Texte intégral non garanti.
    """
    out: List[NewsItem] = []
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        news = []
    for n in news[:max_items]:
        title = n.get("title") or ""
        link = n.get("link") or n.get("url") or ""
        pub_ms = n.get("providerPublishTime")
        src = (n.get("publisher") or n.get("provider") or "Unknown").strip()
        try:
            ts = pd.to_datetime(pub_ms, unit="s", utc=True).tz_convert(None) if pub_ms else pd.Timestamp.utcnow()
        except Exception:
            ts = pd.Timestamp.utcnow()
        out.append(NewsItem(
            ticker=ticker.upper(), title=_clean_text(title),
            summary="", source=src, url=link, published=ts
        ))
    return out


def fetch_rss(url: str, ticker_hint: Optional[str] = None, max_items: int = 50) -> List[NewsItem]:
    """
    Parser RSS minimaliste (sans dépendances) — fonctionne pour flux ATOM/RSS simples.
    Utilisez pour SEC/SEDAR si vous avez l’URL, sinon ignorez.
    """
    items: List[NewsItem] = []
    try:
        import urllib.request
        data = urllib.request.urlopen(url, timeout=15).read().decode("utf-8", errors="ignore")
    except Exception:
        return items

    # naïf: split par <item> ou <entry>
    raw_items = re.split(r"<item>|<entry>", data, flags=re.IGNORECASE)[1:]
    for r in raw_items[:max_items]:
        title = _extract_tag(r, "title")
        link = _extract_tag(r, "link") or _extract_attr(r, "link", "href")
        pub = _extract_tag(r, "pubDate") or _extract_tag(r, "updated") or _extract_tag(r, "published")
        src = _extract_tag(r, "source") or _extract_tag(r, "author") or "RSS"
        try:
            ts = pd.to_datetime(pub).tz_convert(None) if pub else pd.Timestamp.utcnow()
        except Exception:
            ts = pd.Timestamp.utcnow()
        items.append(NewsItem(
            ticker=(ticker_hint or "").upper(),
            title=_clean_text(html.unescape(title or "")),
            summary="",
            source=_clean_text(html.unescape(src or "RSS")),
            url=(link or "").strip(),
            published=ts
        ))
    return items


def _extract_tag(xml: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_attr(xml: str, tag: str, attr: str) -> Optional[str]:
    m = re.search(rf"<{tag}[^>]*{attr}=['\"]([^'\"]+)['\"][^>]*>", xml, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------- NLP utils ---------------------- #

_PUCTBL = str.maketrans("", "", "\t\r")

def _clean_text(s: str) -> str:
    s = (s or "").replace("\xa0", " ").translate(_PUCTBL)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sent_vader(text: str) -> Optional[float]:
    if not HAS_VADER:
        return None
    if not text:
        return None
    try:
        vs = SentimentIntensityAnalyzer()
        score = vs.polarity_scores(text)["compound"]
        # map [-1,1] → [-1,1] (déjà)
        return float(score)
    except Exception:
        return None


def _sent_hf(text: str) -> Optional[float]:
    if not HAS_HF:
        return None
    if not text:
        return None
    try:
        out = HF_PIPE(text[:512])  # tronque pour vitesse
        if not out:
            return None
        lab = out[0]["label"].lower()
        sc = float(out[0]["score"])
        # map HF (POS/NEG) sur [-1,1]
        return float(sc if "pos" in lab else -sc)
    except Exception:
        return None


def score_sentiment(title: str, abstract: str = "", weight_title: float = 0.65) -> SentimentDetail:
    """
    Ensemble: VADER et/ou HF ; priorité au titre.
    """
    t = _clean_text(title)
    a = _clean_text(abstract)
    text_title = t
    text_body = a if len(a) >= 20 else ""

    v_title = _sent_vader(text_title)
    v_body = _sent_vader(text_body) if text_body else None
    h_title = _sent_hf(text_title)
    h_body = _sent_hf(text_body) if text_body else None

    # agrégation par modalité
    def _agg(x: List[Optional[float]]) -> Optional[float]:
        xs = [z for z in x if z is not None]
        return float(np.mean(xs)) if xs else None

    vader = _agg([v_title, v_body])
    hf = _agg([h_title, h_body])

    # ensemble final
    parts = []
    if vader is not None:
        parts.append(vader)
    if hf is not None:
        parts.append(hf)
    if not parts:
        # fallback heuristique: neutre sauf mots clés
        key_plus = bool(re.search(r"\b(beat|surpass|record|approval|upgrade|partnership|wins?)\b", t, flags=re.I))
        key_minus = bool(re.search(r"\b(miss|downgrade|lawsuit|probe|recall|fraud|bankruptcy)\b", t, flags=re.I))
        est = 0.25 if key_plus and not key_minus else (-0.25 if key_minus and not key_plus else 0.0)
        return SentimentDetail(vader=None, hf=None, ensemble=float(est))

    # pondérer le titre un peu plus fort (via duplication)
    ens_list = []
    for sc in parts:
        ens_list.extend([sc] * 2)  # simulate weight on headline
    ensemble = float(np.mean(ens_list))
    return SentimentDetail(vader=vader, hf=hf, ensemble=ensemble)


# ---------------------- Résumé & Événements ---------------------- #

def summarize_textrank(sentences: List[str], k: int = 3) -> str:
    """
    TextRank très light basé sur similarité TF (cosinus) — sans lib externe.
    Pour titres courts, retourne le texte original.
    """
    sents = [s.strip() for s in sentences if s and len(s.strip()) > 0]
    if len(sents) <= k:
        return " ".join(sents)

    # vocabulaire
    def tok(x: str) -> List[str]:
        return re.findall(r"[A-Za-z]{3,}", x.lower())

    bags = [pd.value_counts(tok(s)) for s in sents]
    cols = list(set().union(*[set(b.index) for b in bags]))
    M = np.zeros((len(sents), len(cols)))
    for i, b in enumerate(bags):
        for j, w in enumerate(cols):
            M[i, j] = b.get(w, 0.0)
    # normalisation L2
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    # matrice de similarité
    S = M @ M.T
    np.fill_diagonal(S, 0.0)
    # score de centralité
    central = S.sum(axis=1)
    idx = np.argsort(-central)[:k]
    ordered = sorted(idx.tolist())
    return " ".join([sents[i] for i in ordered])


_EVENT_PATTERNS = [
    ("earnings_beat", +1, r"\b(earnings|eps)\s+(beat|above|topped)\b"),
    ("earnings_miss", -1, r"\b(earnings|eps)\s+(miss|below|under)\b"),
    ("guidance_up", +1, r"\b(raises|increase[d]?|upgrades?)\s+(guidance|outlook)\b"),
    ("guidance_down", -1, r"\b(cuts?|reduce[d]?|lowers?)\s+(guidance|outlook)\b"),
    ("mna", +1, r"\b(acquires?|merger|merges|takeover|deal)\b"),
    ("litigation", -1, r"\b(lawsuit|probe|investigation|class action|settlement)\b"),
    ("downgrade", -1, r"\b(downgrade[sd]?|cuts? rating)\b"),
    ("upgrade", +1, r"\b(upgrade[sd]?|raises? rating)\b"),
    ("dividend_up", +1, r"\b(raises?|increase[s]?)\s+dividend\b"),
    ("dividend_cut", -1, r"\b(cuts?|suspend[s]?)\s+dividend\b"),
]

def extract_events(text: str, max_events: int = 4) -> List[EventSignal]:
    evts: List[EventSignal] = []
    if not text:
        return evts
    t = text.lower()
    for typ, pol, pat in _EVENT_PATTERNS:
        m = re.search(pat, t, flags=re.I)
        if m:
            strength = min(1.0, 0.5 + 0.1 * len(re.findall(pat, t, flags=re.I)))
            span = m.group(0)
            evts.append(EventSignal(type=typ, strength=float(strength), polarity=int(np.sign(pol)), evidence=span))
            if len(evts) >= max_events:
                break
    return evts


# ---------------------- Importance / Dédoublonnage ---------------------- #

_SOURCE_WEIGHTS = {
    "Bloomberg": 1.0, "Reuters": 1.0, "The Wall Street Journal": 0.9,
    "CNBC": 0.8, "MarketWatch": 0.7, "Yahoo": 0.6, "Unknown": 0.5, "RSS": 0.5
}

def _importance(item: NewsItem, sentiment: SentimentDetail) -> float:
    src_w = _SOURCE_WEIGHTS.get(item.source, 0.6)
    recency_w = 1.0 / (1.0 + max(0.0, (pd.Timestamp.utcnow() - item.published).days) * 0.2)
    sent_w = 0.6 + 0.4 * abs(sentiment.ensemble)  # polarité forte = plus “important”
    return float(src_w * recency_w * sent_w)


def dedupe_news(items: List[NewsItem]) -> List[NewsItem]:
    seen = set()
    out: List[NewsItem] = []
    for it in items:
        k = it.key()
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    # également dédoublage sur titre proche
    titles = {}
    final: List[NewsItem] = []
    for it in out:
        h = hashlib.md5(it.title.lower().encode()).hexdigest()[:10]
        if h in titles:
            # garder le plus récent
            if it.published > titles[h].published:
                titles[h] = it
        else:
            titles[h] = it
    final.extend(list(titles.values()))
    final.sort(key=lambda x: x.published, reverse=True)
    return final


# ---------------------- Pipeline principal ---------------------- #

def score_news_items(items: List[NewsItem]) -> List[ScoredNews]:
    scored: List[ScoredNews] = []
    for it in items:
        sent = score_sentiment(it.title, it.summary or (it.raw_text or ""))
        text_for_events = " ".join([it.title, it.summary or "", (it.raw_text or "")])
        evs = extract_events(text_for_events)
        imp = _importance(it, sent)
        scored.append(ScoredNews(item=it, sentiment=sent, events=evs, importance=imp))
    return scored


def aggregate_sentiment(scored: List[ScoredNews]) -> AggregateSentiment:
    if not scored:
        empty = pd.DataFrame(columns=["mean_sent", "n", "pos_share", "imp_weighted", "event_plus", "event_minus"])
        return AggregateSentiment(by_day=empty, by_week=empty, shock_score=0.0, drift_score=0.0, risk_flags=[])

    rows = []
    for s in scored:
        d = {
            "date": s.item.published.normalize(),
            "sent": s.sentiment.ensemble,
            "imp": s.importance,
            "pos": 1 if s.sentiment.ensemble > 0.15 else 0,
            "neg": 1 if s.sentiment.ensemble < -0.15 else 0,
            "evt_plus": sum(1 for e in s.events if e.polarity > 0),
            "evt_minus": sum(1 for e in s.events if e.polarity < 0),
        }
        rows.append(d)
    df = pd.DataFrame(rows)

    by_day = df.groupby("date").agg(
        mean_sent=("sent", "mean"),
        n=("sent", "size"),
        pos_share=("pos", "mean"),
        imp_weighted=("imp", "sum"),
        event_plus=("evt_plus", "sum"),
        event_minus=("evt_minus", "sum"),
    )
    by_week = df.groupby(pd.Grouper(key="date", freq="W-MON")).agg(
        mean_sent=("sent", "mean"),
        n=("sent", "size"),
        pos_share=("pos", "mean"),
        imp_weighted=("imp", "sum"),
        event_plus=("evt_plus", "sum"),
        event_minus=("evt_minus", "sum"),
    )

    # Shock score: pic soudain de volume/importance * polarité absolue
    shock = 0.0
    if len(by_day) >= 5:
        imp_z = (by_day["imp_weighted"] - by_day["imp_weighted"].rolling(20, min_periods=5).mean()) / (
            by_day["imp_weighted"].rolling(20, min_periods=5).std().replace(0, np.nan)
        )
        sent_abs = by_day["mean_sent"].abs()
        shock_series = (imp_z.fillna(0) * sent_abs.fillna(0)).replace([np.inf, -np.inf], 0.0)
        shock = float(shock_series.tail(3).max())  # pic récent

    # Drift post-earnings: moyenne 5 jours après un evt earnings_beat/miss (proxy simple)
    drift = 0.0
    if not by_day.empty:
        # Si beaucoup d'événements +/- sur quelques jours récents, drift = moyenne signée
        last5 = by_day.tail(5)
        if last5["event_plus"].sum() + last5["event_minus"].sum() >= 3:
            drift = float(last5["mean_sent"].mean())

    # Risk flags
    flags = []
    if (by_week["event_minus"].tail(4).sum() if not by_week.empty else 0) >= 3:
        flags.append("Événements négatifs répétés (4 sem.)")
    if (by_week["mean_sent"].tail(4).mean() if not by_week.empty else 0) < -0.1:
        flags.append("Sentiment moyen négatif (4 sem.)")
    if (by_day["imp_weighted"].tail(3).mean() if not by_day.empty else 0) > (by_day["imp_weighted"].rolling(60).mean().iloc[-1] if len(by_day) > 60 else 0) * 2.5:
        flags.append("Pic d’attention médiatique")

    return AggregateSentiment(by_day=by_day, by_week=by_week, shock_score=shock, drift_score=drift, risk_flags=flags)


# ---------------------- API haut niveau ---------------------- #

def build_sentiment_view(ticker: str,
                         rss_urls: Optional[List[str]] = None,
                         extra_texts: Optional[List[Tuple[str, str]]] = None,
                         max_items: int = 60) -> Dict[str, Any]:
    """
    Récupère, note, agrège, et retourne un snapshot prêt pour l'UI.

    Args:
      ticker: symbole
      rss_urls: liste d’URLs RSS supplémentaires (facultatif)
      extra_texts: liste de (title, text) pour intégrer vos propres sources
      max_items: limite news par source

    Returns:
      dict avec: items_scored (n<50), aggregates, top_stories, signals
    """
    items: List[NewsItem] = []

    # yfinance
    items.extend(fetch_yf_news(ticker, max_items=max_items))

    # RSS optionnels
    if rss_urls:
        for u in rss_urls:
            try:
                items.extend(fetch_rss(u, ticker_hint=ticker, max_items=max_items))
            except Exception:
                pass
            time.sleep(0.1)

    # Textes fournis
    if extra_texts:
        now = pd.Timestamp.utcnow()
        for title, txt in extra_texts:
            items.append(NewsItem(
                ticker=ticker.upper(), title=_clean_text(title),
                summary="", source="USER", url="", published=now, raw_text=_clean_text(txt)
            ))

    # Dédoublonnage & tri
    items = dedupe_news(items)[:max_items]

    # Scoring
    scored = score_news_items(items)

    # Résumés (à partir de titres + fallback raw_text)
    for s in scored:
        if (not s.item.summary) and s.item.raw_text:
            s.item.summary = summarize_textrank(re.split(r"[.!?]\s+", s.item.raw_text), k=2)

    # Agrégation
    aggr = aggregate_sentiment(scored)

    # Top stories (importance * |sentiment|)
    top = sorted(scored, key=lambda z: z.importance * (abs(z.sentiment.ensemble) + 0.2), reverse=True)[:8]
    top_pack = [{
        "title": z.item.title,
        "source": z.item.source,
        "when": z.item.published.isoformat(),
        "url": z.item.url,
        "sent": round(z.sentiment.ensemble, 3),
        "events": [e.type for e in z.events]
    } for z in top]

    # Signals lisibles
    signals = []
    if aggr.shock_score > 1.5:
        signals.append(f"News shock récent (score={aggr.shock_score:.1f})")
    if aggr.drift_score > 0.1:
        signals.append("Drift post-annonce positif")
    elif aggr.drift_score < -0.1:
        signals.append("Drift post-annonce négatif")
    signals.extend(aggr.risk_flags)

    # Sortie compacte
    view = {
        "ticker": ticker.upper(),
        "summary": {
            "mean_sent_7d": float(np.nanmean([x.sentiment.ensemble for x in scored if (pd.Timestamp.utcnow() - x.item.published).days <= 7]) if scored else np.nan),
            "mean_sent_30d": float(np.nanmean([x.sentiment.ensemble for x in scored if (pd.Timestamp.utcnow() - x.item.published).days <= 30]) if scored else np.nan),
            "shock_score": aggr.shock_score,
            "drift_score": aggr.drift_score
        },
        "signals": signals,
        "top_stories": top_pack,
        "aggregates": aggr.to_dict()
    }

    # Pour UI: DataFrames exportables (optionnel)
    # (laisser à None si vous ne voulez pas renvoyer des DF volumineux)
    view["_df_by_day"] = aggr.by_day
    view["_df_by_week"] = aggr.by_week

    return view


# ---------------------- Exemple d'exécution ---------------------- #

if __name__ == "__main__":
    # Démo rapide
    tk = "NGD.TO"
    out = build_sentiment_view(tk)
    print(json.dumps({
        "ticker": out["ticker"],
        "summary": out["summary"],
        "signals": out["signals"],
        "top_stories": out["top_stories"][:5],
        "aggregates": out["aggregates"]
    }, indent=2, ensure_ascii=False))