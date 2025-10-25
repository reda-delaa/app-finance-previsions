# src/analytics/econ_llm_agent.py
from __future__ import annotations

import argparse
import json
import os
import sys
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from g4f.client import Client as G4FClient
except Exception as e:
    raise RuntimeError(
        "g4f n'est pas installé. Fais `pip install -U g4f` dans ton venv."
    ) from e


# ======== Modèles “power” no-auth (depuis ta working list) ====================
# IMPORTANT: on exclut gpt-4 / gpt-4.1 car ils exigent une auth et ont échoué chez toi.
# L'ordre ≈ puissance/raisonnement/contexte. Tu peux réordonner selon tes tests.
POWER_NOAUTH_MODELS: List[str] = [
    "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3-0324-Turbo",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "zai-org/GLM-4.5",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai/gpt-oss-120b",  # OSS proxy listé comme "working" no-auth
]

DEFAULT_MODEL_CANDIDATES: List[str] = POWER_NOAUTH_MODELS[:]

# Rang de “puissance” (pour choisir le juge le plus fort)
_POWER_RANK = {m: i for i, m in enumerate(POWER_NOAUTH_MODELS)}
def _power_rank(model: str) -> int:
    return _POWER_RANK.get(model, 10_000)

# Famille/fournisseur (pour diversité du comité)
def _model_family(model: str) -> str:
    m = model.lower()
    if "deepseek" in m: return "deepseek"
    if "qwen" in m: return "qwen"
    if "glm" in m: return "glm"
    if "llama" in m or "meta-llama" in m: return "llama"
    if "gpt-oss" in m or "openai/gpt-oss" in m: return "gpt-oss"
    return "other"

def _pick_top3_distinct(models: List[str]) -> List[str]:
    """Prend l’ordre fourni et retient 3 modèles de familles différentes; complète si besoin."""
    chosen, seen = [], set()
    for m in models:
        fam = _model_family(m)
        if fam not in seen:
            chosen.append(m)
            seen.add(fam)
        if len(chosen) == 3:
            return chosen
    # compléter si pas 3
    for m in models:
        if m not in chosen:
            chosen.append(m)
            if len(chosen) == 3:
                break
    return chosen


# ======== Hyperparams =========================================================
CHAR_BUDGET = int(os.getenv("ECON_AGENT_CHAR_BUDGET", "60000"))
MAX_TOKENS = int(os.getenv("ECON_AGENT_MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("ECON_AGENT_TEMPERATURE", "0.2"))
TIMEOUT = int(os.getenv("ECON_AGENT_TIMEOUT", "60"))
RETRIES_PER_MODEL = int(os.getenv("ECON_AGENT_RETRIES", "1"))

# ======== Prompts système (structurés + JSON final) ===========================
SYSTEM_PROMPT_FR = """Tu es un analyste macro-financier senior. Ne révèle pas ton raisonnement interne.
Règles :
- Utilise UNIQUEMENT le contexte fourni ; si une donnée manque, écris exactement "non fourni".
- Si un champ de features est manquant / None, ne pas le mentionner dans la réponse.
- Pas de conseil personnalisé ; reste générique et actionnable.
- ≤ 350 mots, sections strictes, puces numérotées (1., 2., 3.).
Format :
# Synthèse (1–5.)
# Contexte & Indicateurs
# Scénarios & Probabilités (table courte : nom, p en %)
# Risques clés & Signaux
# Impacts marchés (FX, taux, commodities, equity secteurs)
# Actions possibles (génériques)
# Hypothèses & Sources
À la FIN, AJOUTE UNE SEULE LIGNE JSON VALIDE (UNE ligne) :
{"summary":[...],
 "scenarios":[{"name":"base","p":0.60}],
 "risks":[...],
 "impacts":{"FX":[...],"rates":[...],"commodities":[...],"equity":[...]},
 "actions":[...],
 "confidence":0.0}
"""

SYSTEM_PROMPT_EN = """You are a senior macro analyst. Do not reveal hidden chain-of-thought.
Rules:
- ONLY use provided context; if data is missing, write exactly "not provided".
- No personalized advice; keep it generic and actionable.
- ≤ 350 words, strict sections, numbered bullets (1., 2., 3.).
Format:
# Summary (1–5.)
# Context & Indicators
# Scenarios & Probabilities (short table: name, p in %)
# Key Risks & Signals
# Market Impacts (FX, rates, commodities, equity sectors)
# Possible Actions (generic)
# Assumptions & Sources
At the END, ADD ONE valid JSON LINE (single line):
{"summary":[...],
 "scenarios":[{"name":"base","p":0.60}],
 "risks":[...],
 "impacts":{"FX":[...],"rates":[...],"commodities":[...],"equity":[...]},
 "actions":[...],
 "confidence":0.0}
"""

JSONLike = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

@dataclass
class EconomicInput:
    question: str
    features: Optional[Dict[str, Any]] = None
    news: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[JSONLike]] = None
    locale: str = "fr-FR"
    meta: Dict[str, Any] = field(default_factory=dict)


# ======== Utils de formatage ===================================================

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.75)
    tail = limit - head
    return text[:head] + "\n...\n" + text[-tail:]

def _format_features(feat: Dict[str, Any]) -> str:
    try:
        keys_order = sorted(
            feat.keys(),
            key=lambda k: (0 if str(k).startswith("flag_") else
                           1 if "sent" in str(k) or "ratio" in str(k) else
                           2 if "sector_" in str(k) else 3, str(k))
        )
        lines = [f"- {k}: {feat[k]}" for k in keys_order]
        return "## Features\n" + "\n".join(lines) + "\n"
    except Exception:
        return "## Features\n" + json.dumps(feat, ensure_ascii=False, indent=2) + "\n"

def _format_news(news: List[Dict[str, Any]], limit_items: int = 50) -> str:
    items = news[:limit_items]
    lines = []
    for i, n in enumerate(items, 1):
        ts = n.get("ts") or n.get("timestamp") or n.get("time") or ""
        src = n.get("source") or ""
        title = n.get("title") or n.get("headline") or ""
        link = n.get("link") or n.get("url") or ""
        sent = n.get("sent") or n.get("sentiment") or ""
        tickers = n.get("tickers") or n.get("symbols") or []
        summary = n.get("summary") or n.get("sum") or ""
        lines.append(
            f"{i}. [{ts}] {src} | {title}\n"
            f"    sentiment: {sent} | tickers: {tickers}\n"
            f"    {summary}\n"
            f"    {link}\n"
        )
    more = "" if len(news) <= limit_items else f"... et {len(news) - limit_items} de plus\n"
    return "## News\n" + "\n".join(lines) + more

def _format_attachments(atts: List[JSONLike], limit_chars_each: int = 8000, limit_total: int = 40000) -> str:
    chunks: List[str] = []
    total = 0
    for idx, a in enumerate(atts, 1):
        try:
            if isinstance(a, (dict, list)):
                txt = json.dumps(a, ensure_ascii=False, indent=2)
            else:
                txt = str(a)
        except Exception:
            txt = str(a)
        txt = _truncate(txt, limit_chars_each)
        if total + len(txt) > limit_total:
            break
        total += len(txt)
        chunks.append(f"### Attachment #{idx}\n{txt}\n")
    return "## Attachments\n" + ("\n".join(chunks) if chunks else "(none)\n")

def _build_context(ein: EconomicInput, char_budget: int) -> str:
    parts: List[str] = []
    parts.append(f"# Question\n{ein.question}\n")
    if ein.features:
        parts.append(_format_features(ein.features))
    if ein.news:
        news_budget = int(char_budget * 0.4)
        news_block = _format_news(ein.news, limit_items=120)
        parts.append(_truncate(news_block, news_budget))
    if ein.attachments:
        att_budget = int(char_budget * 0.4)
        att_block = _format_attachments(ein.attachments)
        parts.append(_truncate(att_block, att_budget))
    if ein.meta:
        try:
            meta_txt = json.dumps(ein.meta, ensure_ascii=False, indent=2)
        except Exception:
            meta_txt = str(ein.meta)
        parts.append("## Meta\n" + _truncate(meta_txt, int(char_budget * 0.1)))
    ctx = "\n".join(parts)
    return _truncate(ctx, char_budget)

def _pick_system_prompt(locale: str) -> str:
    return SYSTEM_PROMPT_FR if (locale or "").lower().startswith("fr") else SYSTEM_PROMPT_EN


def clean_llm_text(txt: str, max_chars: int = 3000) -> str:
    """Clean LLM responses by removing repetitions and truncating."""
    if not txt:
        return ""
    import re
    # Remove gross repetitions (same field pattern repeated 3+ times)
    txt = re.sub(r"(?:\b[\w_]+: ?[^\n]{5,}\n){3,}", lambda m: m.group(0).split("\n")[0]+"\n", txt)
    # Reduce character repetition (aaaaa → aaa)
    txt = re.sub(r"(.)\1{4,}", r"\1\1\1", txt)
    return txt[:max_chars].rstrip()

def _to_json_serializable(o):
    try:
        if o is None:
            return None
        if isinstance(o, (str, int, float, bool)):
            return o
        if isinstance(o, dict):
            return {k: _to_json_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_json_serializable(v) for v in o]
        if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
            try:
                return _to_json_serializable(o.model_dump())
            except Exception:
                pass
        elif hasattr(o, "dict") and callable(getattr(o, "dict")):
            try:
                return _to_json_serializable(o.dict())
            except Exception:
                pass
        if hasattr(o, "__dict__"):
            try:
                return _to_json_serializable(vars(o))
            except Exception:
                pass
    except Exception:
        pass
    return str(o)


# ======== Mesures d’accord & JSON parsing ====================================
_word_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9%.\-]+")

def _normalize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _word_re.findall(text)]

def _token_set_jaccard(a: str, b: str) -> float:
    sa, sb = set(_normalize(a)), set(_normalize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)

def _ngram_overlap(a: str, b: str, n: int = 3) -> float:
    def ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1)))
    ta, tb = _normalize(a), _normalize(b)
    if len(ta) < n or len(tb) < n:
        return _token_set_jaccard(a, b)
    ga, gb = ngrams(ta, n), ngrams(tb, n)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / max(1, len(ga | gb))

def _agreement_score(a: str, b: str) -> float:
    return 0.5 * _token_set_jaccard(a, b) + 0.5 * _ngram_overlap(a, b, n=3)

def _extract_tail_json_line(text: str) -> Optional[Dict[str, Any]]:
    """Récupère la DERNIÈRE ligne JSON complète à la fin de la réponse."""
    if not text:
        return None
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                # clés attendues pour nos comparaisons
                for k in ["summary","scenarios","risks","impacts","actions","confidence"]:
                    if k not in obj:
                        return None
                return obj
            except Exception:
                return None
    return None

def _list_agreement(a: List[str], b: List[str]) -> float:
    sa = set(str(x).strip().lower() for x in (a or []))
    sb = set(str(x).strip().lower() for x in (b or []))
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def _json_based_agreement(ja: Dict[str, Any], jb: Dict[str, Any]) -> float:
    if not ja or not jb: return 0.0
    scores: List[float] = []
    # summary/risks/actions
    for key in ["summary","risks","actions"]:
        va = ja.get(key) or []
        vb = jb.get(key) or []
        if isinstance(va, list) and isinstance(vb, list):
            scores.append(_list_agreement(va, vb))
    # impacts
    ia, ib = ja.get("impacts") or {}, jb.get("impacts") or {}
    for k in ["FX","rates","commodities","equity"]:
        va = ia.get(k) or []
        vb = ib.get(k) or []
        if isinstance(va, list) and isinstance(vb, list):
            scores.append(_list_agreement(va, vb))
    # scenarios: compare les noms
    sa = [s.get("name","") for s in (ja.get("scenarios") or []) if isinstance(s, dict)]
    sb = [s.get("name","") for s in (jb.get("scenarios") or []) if isinstance(s, dict)]
    scores.append(_list_agreement(sa, sb))
    return sum(scores)/len(scores) if scores else 0.0


# ======== Client LLM ==========================================================
class EconomicAnalyst:
    """
    Agent générique pour analyses économiques & Q/A multi-sources via g4f.
    - analyze(...) : essaie plusieurs modèles jusqu'à succès.
    - analyze_ensemble(..., top_n=3, force_power=False, adjudicate=False)
    """

    def __init__(
        self,
        model_candidates: Optional[List[str]] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        timeout: int = TIMEOUT,
        retries_per_model: int = RETRIES_PER_MODEL,
        char_budget: int = CHAR_BUDGET,
    ):
        env_models = self._load_models_from_env()
        base = env_models or model_candidates or DEFAULT_MODEL_CANDIDATES
        # Overlay dynamic working models if available (fresh within max age)
        try:
            if os.getenv("ECON_AGENT_DYNAMIC_MODELS", "1") == "1":
                from agents.g4f_model_watcher import load_working_models
                max_age_h = int(os.getenv("G4F_WORKING_MAX_AGE_H", "24"))
                dyn = load_working_models(max_age_hours=max_age_h)
                if dyn:
                    # Prefer dynamic ordering; keep only intersection to ensure compatibility
                    dyn_set = set(dyn)
                    base = [m for m in dyn if m in dyn_set] + [m for m in base if m not in dyn_set]
        except Exception:
            pass
        # Reorder with a light reasoning-first preference for economic/trading analysis
        try:
            pref = [
                "deepseek" ,   # R1 / V3 families first
                "qwen3-235b-a22b-thinking",
                "glm-4.5",
                "llama-3.3-70b",
                "gpt-oss-120b",
            ]
            def _key(m: str) -> tuple:
                lm = (m or "").lower()
                for i, pat in enumerate(pref):
                    if pat in lm:
                        return (0, i)  # highest bucket then by pref index
                return (1, lm)
            base = sorted(list(dict.fromkeys(base)), key=_key)
        except Exception:
            pass
        self.model_candidates = base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries_per_model = retries_per_model
        self.char_budget = char_budget
        self.client = G4FClient()

    def _load_models_from_env(self) -> Optional[List[str]]:
        raw = os.getenv("ECON_AGENT_MODELS", "").strip()
        if not raw:
            return None
        return [m.strip() for m in raw.split(",") if m.strip()]

    def _build_messages(self, data: EconomicInput) -> List[Dict[str, str]]:
        system_prompt = _pick_system_prompt(data.locale)
        context = _build_context(data, self.char_budget)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Tu recevras ci-dessous la question et le contexte. "
                    "Réponds selon le format demandé, en restant fidèle aux données.\n\n"
                    + context
                ),
            },
        ]

    def _call_model(self, model: str, messages: List[Dict[str, str]]) -> Tuple[bool, Dict[str, Any]]:
        last_err: Optional[str] = None
        for attempt in range(1, self.retries_per_model + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                text = (resp.choices[0].message.content if hasattr(resp, "choices") else str(resp))
                # Clean the LLM response to remove noise and limit length
                text = clean_llm_text(text)
                usage = getattr(resp, "usage", None)
                parsed = _extract_tail_json_line(text)
                return True, {
                    "ok": True,
                    "model": model,
                    "attempt": attempt,
                    "answer": text,
                    "parsed": parsed,
                    "usage": _to_json_serializable(usage),
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
        return False, {
            "ok": False,
            "model": model,
            "attempt": self.retries_per_model,
            "answer": "",
            "error": last_err or "Aucun provider n'a répondu",
        }

    # ---- mode simple : un seul résultat
    def analyze(self, data: EconomicInput) -> Dict[str, Any]:
        messages = self._build_messages(data)
        last_err: Optional[str] = None
        for model in self.model_candidates:
            ok, res = self._call_model(model, messages)
            if ok:
                return res
            last_err = res.get("error")
        return {
            "ok": False,
            "model": None,
            "attempt": None,
            "answer": "",
            "error": last_err or "Aucun provider n'a répondu",
        }

    # ---- ensemble : jusqu'à N réponses OK (backfill) + accord JSON-first -----
    def analyze_ensemble(self, data: EconomicInput, top_n: int = 3, force_power: bool = False, adjudicate: bool = False) -> Dict[str, Any]:
        if top_n <= 0:
            return {"ok": False, "error": "top_n doit être ≥ 1"}

        base_list = POWER_NOAUTH_MODELS if force_power else (self.model_candidates or DEFAULT_MODEL_CANDIDATES)
        first3 = _pick_top3_distinct(base_list)
        # ordre d'essai: 3 distincts d'abord, puis le reste en backfill
        backfill = [m for m in base_list if m not in first3]
        models_try_order = first3 + backfill

        messages = self._build_messages(data)
        results: List[Dict[str, Any]] = []
        tried: set = set()

        # Essayer jusqu'à obtenir top_n OK ou épuiser
        for m in models_try_order:
            if len([r for r in results if r.get("ok")]) >= top_n:
                break
            if m in tried:
                continue
            tried.add(m)
            _, r = self._call_model(m, messages)
            results.append(r)

        ok_results = [r for r in results if r.get("ok")]
        models_ok = [r["model"] for r in ok_results]

        # Accord pair-à-pair : JSON d'abord, fallback texte
        pairwise = []
        for i in range(len(ok_results)):
            for j in range(i + 1, len(ok_results)):
                ai, aj = ok_results[i], ok_results[j]
                pj_i, pj_j = ai.get("parsed"), aj.get("parsed")
                score = _json_based_agreement(pj_i, pj_j) if (pj_i and pj_j) else _agreement_score(ai.get("answer",""), aj.get("answer",""))
                pairwise.append({
                    "i": i, "j": j,
                    "model_i": ai["model"],
                    "model_j": aj["model"],
                    "agreement": round(score, 4),
                })
        avg_agreement = round(sum(p["agreement"] for p in pairwise) / max(1, len(pairwise)), 4) if pairwise else 0.0

        out: Dict[str, Any] = {
            "ok": len(ok_results) >= 1,
            "models": models_ok,
            "results": results,
            "pairwise_agreement": pairwise,
            "avg_agreement": avg_agreement,
            "consensus": [],
            "divergences": [],
        }

        # Consensus / divergences simples (sur JSON si dispo)
        if len(ok_results) >= 2:
            # collecter sets
            buckets = {"summary": [], "risks": [], "actions": []}
            for r in ok_results:
                pj = r.get("parsed") or {}
                for k in buckets:
                    vals = pj.get(k) or []
                    if isinstance(vals, list):
                        buckets[k].append(set(str(x).strip().lower() for x in vals))
            consensus = {}
            for k, sets in buckets.items():
                if sets:
                    inter = set.intersection(*sets) if len(sets) > 1 else sets[0]
                    consensus[k] = sorted(inter)
            out["consensus"] = consensus

            divergences = []
            for idx, r in enumerate(ok_results):
                pj = r.get("parsed") or {}
                uniq: List[str] = []
                for k in ["summary","risks","actions"]:
                    mine = set(str(x).strip().lower() for x in (pj.get(k) or []))
                    others = set().union(*(buckets[k][j] for j in range(len(ok_results)) if j != idx)) if buckets[k] else set()
                    uniq.extend(sorted(mine - others))
                if uniq:
                    divergences.append({"i": idx, "model": r["model"], "unique_points": uniq[:30]})
            out["divergences"] = divergences

        # Arbitrage : juge = le plus puissant no-auth NON utilisé, sinon le meilleur parmi les OK
        if adjudicate and len(ok_results) >= 2:
            judge_pool = [m for m in POWER_NOAUTH_MODELS if m not in models_ok] + POWER_NOAUTH_MODELS
            judge = judge_pool[0]
            judge_messages = self._build_messages(data)
            expert_jsons = [r.get("parsed") for r in ok_results if r.get("parsed")]
            judge_messages.append({
                "role": "user",
                "content": (
                    "Trois experts ont répondu. Compare leurs JSON et tranche.\n"
                    f"EXPERT_JSONS={json.dumps(expert_jsons, ensure_ascii=False)}\n"
                    "Donne: # Consensus, # Disagreements, # Final Decision (≤150 mots), # Confidence (0-1).\n"
                    "À la FIN, AJOUTE UNE LIGNE JSON: {\"winner_model\":\"...\",\"confidence\":0.0}"
                ),
            })
            ok, jres = self._call_model(judge, judge_messages)
            if ok:
                out["adjudication"] = {
                    "judge_model": judge,
                    "decision": jres.get("answer", ""),
                    "usage": jres.get("usage"),
                }
            else:
                out["adjudication"] = {"judge_model": judge, "error": jres.get("error")}

        return out


# ======== CLI =================================================================
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_jsonl(path: str, limit: Optional[int] = None) -> List[Any]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                out.append({"raw": line})
            if limit and i >= limit:
                break
    return out

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Economic LLM Agent (g4f)")
    p.add_argument("--question", "-q", required=True, help="Question à poser à l'agent")
    p.add_argument("--json", action="append", default=[], help="Fichier JSON (features/attachments/meta)")
    p.add_argument("--jsonl", action="append", default=[], help="Fichier JSONL (news)")
    p.add_argument("--lang", default="fr", help="fr ou en (défaut: fr)")
    p.add_argument("--max-news", type=int, default=150, help="Limiter le nombre d'items news chargés")
    p.add_argument("--models", help="Liste de modèles séparés par des virgules (override)")
    p.add_argument("--ensemble", type=int, default=0, help="Si >0, interroge N meilleurs modèles et compare les réponses")
    p.add_argument("--force_power", action="store_true",
                   help="Force la short-list POWER_NOAUTH_MODELS (3 meilleurs + backfill) et diversité de familles.")
    p.add_argument("--adjudicate", action="store_true",
                   help="Ajoute un 4ᵉ agent arbitre (le plus puissant dispo) pour trancher.")
    args = p.parse_args(argv)

    # Construire l'entrée
    features: Dict[str, Any] = {}
    news: List[Dict[str, Any]] = []
    attachments: List[JSONLike] = []
    meta: Dict[str, Any] = {}

    for path in args.json:
        obj = _load_json(path)
        if isinstance(obj, dict):
            if "features" in obj and isinstance(obj["features"], dict):
                features.update(obj["features"])
                obj = {k: v for k, v in obj.items() if k != "features"}
            meta.update(obj)
        elif isinstance(obj, list):
            attachments.append(obj)
        else:
            attachments.append(obj)

    for path in args.jsonl:
        news.extend(_load_jsonl(path))
    if args.max_news and len(news) > args.max_news:
        news = news[: args.max_news]

    if args.force_power:
        base_list = POWER_NOAUTH_MODELS[:]
    elif args.models:
        base_list = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        base_list = None

    agent = EconomicAnalyst(model_candidates=base_list)
    ein = EconomicInput(
        question=args.question,
        features=features or None,
        news=news or None,
        attachments=attachments or None,
        locale="fr" if args.lang.lower().startswith("fr") else "en",
        meta=meta or {},
    )

    if args.ensemble and args.ensemble > 0:
        res = agent.analyze_ensemble(ein, top_n=args.ensemble, force_power=args.force_power, adjudicate=args.adjudicate)
    else:
        res = agent.analyze(ein)

    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("ok") else 1


# Exports pour compatibilité avec app.py
__all__ = [
    'EconomicAnalyst',
    'EconomicInput',
    'ask_model', 
    'arbitre', 
    'analyze_economic_question',
    'POWER_NOAUTH_MODELS',
]

# Fonctions d'interface publiques
def ask_model(question: str, context: dict = None) -> str:
    """Interface simplifiée pour poser des questions économiques."""
    agent = EconomicAnalyst()
    
    ein = EconomicInput(
        question=question,
        features=context.get('features') if context else None,
        news=context.get('news') if context else None,
        attachments=context.get('attachments') if context else None,
        locale=context.get('locale', 'fr') if context else 'fr',
        meta=context or {}
    )
    
    result = agent.analyze(ein)
    return result.get('answer', 'Erreur dans la réponse LLM')

def arbitre(context: dict) -> dict:
    """Interface arbitre pour les décisions économiques."""
    agent = EconomicAnalyst()
    
    ein = EconomicInput(
        question=context.get('question', f'Analyse {context.get("scope", "macro")}'),
        features=context.get('macro_features') or context.get('tech_features'),
        news=context.get('news'),
        attachments=context.get('attachments'),
        locale=context.get('locale', 'fr'),
        meta=context
    )
    
    return agent.analyze(ein)

# Alias
analyze_economic_question = ask_model


if __name__ == "__main__":
    sys.exit(main())
