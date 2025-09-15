"""
Lightweight NLP enrichment utilities with zero external dependencies.

Features:
- Multi-level summarization: headline (<= 120 chars), bullets (3-6 pts), narrative (short paragraph)
- Rule-based sentiment (domain-aware, finance + geopolitics lexicons)
- Simple entity extraction: companies (heuristics), tickers (regex), ISIN/CUSIP, countries/commodities
- Language hints: cheap heuristics + pluggable translator callback
- Clean dataclass API + pure functions; safe to run per-article

Intended usage:
from nlp_enrich import enrich_article
item = enrich_article(title, summary, body, source="ft", region="INTL")

Optional translator: pass translator=str->str callable (e.g., wrapper to DeepL/OpenAI/etc.)

No external imports beyond stdlib.
"""
from __future__ import annotations

# Imports optionnels pour éviter les erreurs d'import relatifs
try:
    from taxonomy.news_taxonomy import tag_sectors, classify_event, tag_geopolitics
    IMPORT_SUCCESS = True
except ImportError:
    # Fallback si import relatif échoue (ex: exécution directe)
    tag_sectors, classify_event, tag_geopolitics = lambda x: [], lambda x, y: "unknown", lambda x: []
    IMPORT_SUCCESS = False

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Callable
import re
import math

# -----------------------------
# Utilities
# -----------------------------

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _split_sentences(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    # naive split on punctuation; keep abbreviations minor handling
    # split on . ! ? followed by space and uppercase or end
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[]|$)", text)
    # fallback if giant block
    if len(parts) == 1 and ". " in text:
        parts = text.split(". ")
    return [p.strip().strip("-—• ") for p in parts if p.strip()]


# -----------------------------
# Summarization (extractive heuristic)
# -----------------------------

def summarize(text: str, level: str = "headline", max_bullets: int = 5) -> str | List[str]:
    """Very small extractive summarizer.
    level: 'headline' | 'bullets' | 'narrative'
    - headline: best short sentence/fragment <= 120 chars
    - bullets: 3..max_bullets bullet points (key sentences)
    - narrative: 3-5 sentences stitched
    """
    sents = _split_sentences(text)
    if not sents:
        return "" if level != "bullets" else []

    # scoring: position + keyword boosts + length penalty
    keywords = {
        "acquire", "merger", "guidance", "forecast", "profit", "loss", "tariff", "sanction",
        "inflation", "rate", "policy", "earnings", "IPO", "dividend", "strike", "war",
        "attack", "drone", "oil", "gas", "OPEC", "BRICS", "Ukraine", "Gaza", "Ceasefire",
    }
    scores: List[Tuple[float, str]] = []
    n = len(sents)
    for i, sent in enumerate(sents):
        l = len(sent)
        if l < 12:
            continue
        score = 0.0
        # position: first sentences matter, but de-emphasize very first if headline-ish already
        score += max(0.0, (n - i) / n)  # earlier higher
        # keyword boosts
        tokens = set(re.findall(r"[A-Za-z][A-Za-z\-]+", sent.lower()))
        score += 0.6 * sum(1 for k in keywords if k.lower() in tokens)
        # length sweet spot ~ 60-160 chars
        score -= abs(l - 110) / 220.0
        scores.append((score, sent))

    if not scores:
        return "" if level != "bullets" else []

    ranked = [s for _, s in sorted(scores, key=lambda x: x[0], reverse=True)]

    if level == "headline":
        # choose first that fits <= 120 chars; fallback to best truncated
        for s in ranked:
            if len(s) <= 120:
                return s
        return (ranked[0][:117] + "...") if ranked else ""

    elif level == "bullets":
        k = min(max(3, max_bullets), len(ranked))
        uniq: List[str] = []
        seen = set()
        for s in ranked:
            key = re.sub(r"\W+", " ", s.lower())[:80]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
            if len(uniq) >= k:
                break
        return [f"• {u}" for u in uniq]

    else:  # narrative
        k = min(5, max(3, math.ceil(len(ranked) * 0.3)))
        picked = ranked[:k]
        return " ".join(picked)


# -----------------------------
# Sentiment (rule-based, finance+geo aware)
# -----------------------------

NEG = {
    "loss", "miss", "downgrade", "probe", "fraud", "bankrupt", "default", "sanction", "tariff",
    "strike", "war", "clash", "attack", "explode", "drone", "raid", "boycott", "ban", "cut",
    "layoff", "lawsuit", "recall", "volatility", "selloff", "glut", "oversupply", "deficit",
    "inflation", "recession", "slowdown", "surge cases", "shortage", "crisis", "tensions"
}
POS = {
    "beat", "surge", "rally", "upgrade", "profit", "record", "growth", "guidance raise",
    "deal", "acquire", "merger", "approval", "win", "strong", "robust", "dividend", "buyback",
    "contract award", "ceasefire", "peace", "tariff cut", "stimulus"
}
NEG_PHRASES = {"miss estimates", "cuts outlook", "halts production", "suspends", "criminal charges"}
POS_PHRASES = {"tops estimates", "raises outlook", "resumes", "secured funding", "all-time high"}


def sentiment_score(text: str) -> float:
    """Return sentiment in [-5, 5] (closer to finance-style magnitude).
    Zero means neutral. Sign captures polarity; magnitude scales with density.
    """
    t = _clean_text(text).lower()
    if not t:
        return 0.0
    score = 0.0
    # phrase-level first (higher weight)
    for p in POS_PHRASES:
        if p in t:
            score += 2.0
    for p in NEG_PHRASES:
        if p in t:
            score -= 2.3
    # unigram-ish tokens
    tokens = re.findall(r"[a-z][a-z\-]+", t)
    for w in tokens:
        if w in POS:
            score += 0.6
        elif w in NEG:
            score -= 0.7
    # clamp and scale to [-5,5]
    score = max(-7.5, min(7.5, score))
    return round(5.0 * (score / 7.5), 2)


# -----------------------------
# Entity extraction
# -----------------------------

ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")
# very loose CUSIP (not validating check digit): 9 alphanum
CUSIP_RE = re.compile(r"\b([0-9A-Z]{9})\b")
# tickers like AAPL, MSFT, BMW.DE, 7203.T, ^GSPC etc. Keep reasonably strict for single word caps (2-6)
TICKER_RE = re.compile(r"\b([A-Z]{2,6}(?:\.[A-Z]{1,3})?)\b")

COUNTRIES = {
    "US", "United States", "Canada", "Germany", "France", "UK", "United Kingdom", "China", "India",
    "Russia", "Ukraine", "Poland", "Brazil", "Japan", "South Korea", "Qatar", "Israel", "Saudi Arabia",
}
COMMODITIES = {"oil", "gas", "gold", "silver", "copper", "wheat", "corn", "soy", "lithium", "nickel"}

COMPANY_SUFFIXES = (
    "Inc", "Inc.", "Corp", "Corp.", "Corporation", "Ltd", "Ltd.", "PLC", "LLC", "S.A.", "SE", "AG",
)


def extract_entities(title: str, summary: str = "", body: str = "") -> Dict[str, List[str]]:
    text = " ".join(filter(None, [title, summary, body]))
    t = _clean_text(text)
    out: Dict[str, List[str]] = {"companies": [], "tickers": [], "isin": [], "cusip": [],
                                 "countries": [], "commodities": []}

    # ISIN / CUSIP / Tickers
    out["isin"] = list(dict.fromkeys(ISIN_RE.findall(t)))
    out["cusip"] = [c for c in CUSIP_RE.findall(t) if not ISIN_RE.match(c)]
    # Tickers: avoid common words; filter if adjacent to finance cues or uppercase run
    raw_tk = TICKER_RE.findall(t)
    blacklist = {"THE", "AND", "FOR", "WITH", "WEEK", "BANK", "NEWS"}
    tk = []
    for token in raw_tk:
        if token in blacklist:
            continue
        if len(token) < 2 or len(token) > 8:
            continue
        tk.append(token)
    out["tickers"] = list(dict.fromkeys(tk))

    # Countries
    countries_found = []
    for c in COUNTRIES:
        # word boundary match case-insensitive
        if re.search(rf"\b{re.escape(c)}\b", text, flags=re.I):
            countries_found.append(c)
    out["countries"] = sorted(list(dict.fromkeys(countries_found)))

    # Commodities
    com = []
    tl = t.lower()
    for c in COMMODITIES:
        if re.search(rf"\b{re.escape(c)}\b", tl):
            com.append(c)
    out["commodities"] = sorted(list(dict.fromkeys(com)))

    # Companies (heuristic: Proper Noun sequences + suffix detection)
    companies = set()
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9&'\-]+(?:\s+[A-Z][A-Za-z0-9&'\-]+){0,5})\b", text):
        cand = m.group(1).strip()
        if len(cand) < 3:
            continue
        # must either contain a known suffix OR appear near finance verbs
        if cand.endswith(COMPANY_SUFFIXES) or re.search(r"\b(announced|acquire|merger|IPO|earnings|guidance|dividend|revenue|profit)\b", cand, flags=re.I):
            # filter obvious non-company phrases
            if len(cand.split()) == 1 and cand.upper() == cand:
                continue
            companies.add(cand)
    out["companies"] = sorted(companies)

    return out


# -----------------------------
# Language hints + translation hook
# -----------------------------

LATIN_FREQ_WORDS = {
    "le", "la", "les", "de", "des", "et",  # fr
    "der", "die", "das", "und", "mit",  # de
    "el", "los", "las", "y",  # es
    "il", "lo", "gli", "e",  # it
    "the", "and", "of",  # en
}


def guess_language(text: str) -> str:
    t = _clean_text(text).lower()
    if not t:
        return "unknown"
    hits = {"fr": 0, "de": 0, "es": 0, "it": 0, "en": 0}
    for w in LATIN_FREQ_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            if w in {"le", "la", "les", "de", "des", "et"}:
                hits["fr"] += 1
            elif w in {"der", "die", "das", "und", "mit"}:
                hits["de"] += 1
            elif w in {"el", "los", "las", "y"}:
                hits["es"] += 1
            elif w in {"il", "lo", "gli", "e"}:
                hits["it"] += 1
            elif w in {"the", "and", "of"}:
                hits["en"] += 1
    lang = max(hits.items(), key=lambda x: x[1])[0]
    return lang if hits[lang] > 0 else "unknown"


# -----------------------------
# Public API
# -----------------------------

@dataclass
class EnrichedArticle:
    headline: str
    bullets: List[str]
    narrative: str
    sentiment: float
    language: str
    translated: bool
    text_used: str
    entities: Dict[str, List[str]]

    def asdict(self) -> Dict:
        return asdict(self)


def enrich_article(
    title: str,
    summary: str = "",
    body: str = "",
    translator: Optional[Callable[[str, str, str], str]] = None,
    target_lang: str = "en",
) -> EnrichedArticle:
    """Compose all enrichments in one call.

    translator: optional callable(text, src_lang, target_lang) -> translated_text
    """
    title = _clean_text(title)
    summary = _clean_text(summary)
    body = _clean_text(body)

    # pick text for summarization preference: body > summary > title
    base_text = body or summary or title

    # language + optional translation
    lang = guess_language(" ".join([title, summary, body]))
    translated = False
    text_for_nlp = base_text
    if translator and (lang != "unknown" and lang != target_lang):
        try:
            text_for_nlp = translator(base_text, lang, target_lang)
            translated = True
            lang = target_lang
        except Exception:
            # fail silent, keep original
            translated = False

    # summarization
    head = summarize(text_for_nlp or title, level="headline")
    bullets = summarize(text_for_nlp or summary or title, level="bullets")  # type: ignore
    narrative = summarize(text_for_nlp or base_text, level="narrative")  # type: ignore

    # sentiment on combined (title + summary + headline)
    sent_text = " ".join([title, summary, narrative or head])
    senti = sentiment_score(sent_text)

    # entities on original (for fidelity)
    ents = extract_entities(title, summary, body)

    return EnrichedArticle(
        headline=head,
        bullets=bullets if isinstance(bullets, list) else [],
        narrative=narrative if isinstance(narrative, str) else "",
        sentiment=senti,
        language=lang,
        translated=translated,
        text_used=(text_for_nlp[:5000] if text_for_nlp else ""),
        entities=ents,
    )


# ===============================
# Hub Integration Functions
# ===============================

def ask_model(question: str, context: dict = None) -> str:
    """Fonction wrapper pour intégrer NLP_enrich avec l'app hub.

    Prend question et contexte, retourne une réponse textuelle.
    """
    if context is None:
        context = {}

    # Utilise la question comme titre principal
    enriched = enrich_article(
        title=question[:200],  # limiter la taille
        summary="",  # pas d'autre contenu pour la question
        body=question  # utiliser question complète
    )

    # Structure la réponse avec les données contextuelles
    response_parts = []

    # Réponse narrative comme base
    if enriched.narrative:
        response_parts.append(f"**Analyse :** {enriched.narrative}")

    # Ajouter les entités trouvées dans la question
    if enriched.entities.get("companies") or enriched.entities.get("tickers"):
        ents = enriched.entities.get("companies", []) + enriched.entities.get("tickers", [])
        if ents:
            response_parts.append(f"**Entités identifiées :** {', '.join(ents[:5])}")

    # Ajouter le sentiment de la question
    if enriched.sentiment != 0:
        sent_desc = "positif" if enriched.sentiment > 0 else "négatif"
        response_parts.append(f"**Ton de la question :** {sent_desc} (score: {enriched.sentiment:.2f})")

    # Informations contextuelles complémentaires
    if "scope" in context:
        scope = context["scope"]
        if scope == "stock" and "ticker" in context:
            response_parts.append(f"**Contexte :** Analyse pour l'action {context['ticker']}")
        elif scope == "macro":
            response_parts.append("**Contexte :** Analyse macroéconomique")

    # Synthèse si rien trouvé
    if not response_parts:
        response_parts.append("**Réponse :** Question traitée avec analyse NLP (pas de contexte particulier identifié)")

    return "\n\n".join(response_parts)

# -----------------------------
# Demo / self-test
# -----------------------------
if __name__ == "__main__":
    sample_title = "EU considers tariffs as oil prices whipsaw on new Russia sanctions"
    sample_sum = (
        "Brent spiked after a drone strike near Primorsk while ministers weighed new measures;"
        " analysts warned of oversupply and weak demand despite geopolitical risks."
    )
    sample_body = (
        "Oil markets swung sharply on Friday as reports of Ukrainian drones targeting the Primorsk port"
        " pushed Brent above $67 before retracing. EU finance ministers discussed broader sanctions"
        " and potential tariffs on third-country buyers of Russian crude. Meanwhile, the IEA trimmed its"
        " 2025 demand growth outlook, citing sluggish consumption in emerging economies."
    )

    enriched = enrich_article(sample_title, sample_sum, sample_body)
    import json
    print(json.dumps(enriched.asdict(), ensure_ascii=False, indent=2))

    # Test de ask_model pour l'intégration hub
    print("\n" + "="*50)
    print("TEST ask_model pour hub:")
    test_question = "Les prix du pétrole vont-ils continuer à monter ?"
    test_context = {"scope": "macro", "question": test_question}
    response = ask_model(test_question, test_context)
    print("Question:", test_question)
    print("Réponse:", response)
