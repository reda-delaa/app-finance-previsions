# proxy.py
# OpenAI-compatible proxy au-dessus de g4f, avec:
# - Streaming SSE
# - Tools (function calling) pass-through (JSON brut)
# - Découverte dynamique des modèles "working" (GitHub) + cache local
# - Fallbacks intelligents par TIER (du plus puissant au moins puissant)
#
# Env utiles:
#   G4F_WORKING_URL   (override de l’URL GitHub)
#   G4F_WORKING_CACHE (chemin du cache local, défaut: .g4f_working.txt)
#   G4F_FETCH_TIMEOUT (seconds, défaut: 6)
#   HOST              (0.0.0.0 par défaut)
#   PORT              (4000 par défaut)
#
# Lancement:
#   pip install flask flask-cors requests g4f
#   python proxy.py
#
# Intégration:
#   Base URL: http://127.0.0.1:4000/v1
#   Models (alias): deepseek-v3, qwen3-coder-480b, deepseek-prover, deepseek-r1-distill-70b,
#                   llama-3.3-70b, phi-4-reasoning-plus, qwen3-235b, sonar-pro

from __future__ import annotations

import os
import re
import json
import uuid
import time
import logging
from typing import Dict, List, Tuple, Optional, Iterable

import requests
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import g4f
from model_tester import scan_candidates, load_last_scan

_APIKEY_HINT = re.compile(r"\b(api[_\-\s]?key|add\s+(an?\s+)?api\s*key|missing\s+key)\b", re.I)

def _looks_like_api_key_gate(s: str) -> bool:
    return bool(_APIKEY_HINT.search(s))

# -----------------------------------------------------------------------------
# Logging très verbeux
# -----------------------------------------------------------------------------
logger = logging.getLogger("proxy")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [proxy] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# 1) Fetch & parse "working models" (provider|model|modality) at startup
# -----------------------------------------------------------------------------

DEFAULT_WORKING_URL = (
    "https://raw.githubusercontent.com/maruf009sultan/g4f-working/main/working/working_results.txt"
)
WORKING_URL   = os.getenv("G4F_WORKING_URL", DEFAULT_WORKING_URL)
WORKING_CACHE = os.getenv("G4F_WORKING_CACHE", ".g4f_working.txt")
FETCH_TIMEOUT = float(os.getenv("G4F_FETCH_TIMEOUT", "6"))

def _load_cached_working() -> Optional[str]:
    try:
        if os.path.exists(WORKING_CACHE):
            with open(WORKING_CACHE, "r", encoding="utf-8") as f:
                txt = f.read()
                logger.debug(f"[WORKING] cache hit -> {WORKING_CACHE}, {len(txt)} bytes")
                return txt
    except Exception as e:
        logger.debug(f"[WORKING] cache read error: {e}")
    return None

def _save_cached_working(txt: str) -> None:
    try:
        with open(WORKING_CACHE, "w", encoding="utf-8") as f:
            f.write(txt)
        logger.debug(f"[WORKING] cache saved -> {WORKING_CACHE}, {len(txt)} bytes")
    except Exception as e:
        logger.debug(f"[WORKING] cache save error: {e}")

def _fetch_working_text() -> Optional[str]:
    try:
        logger.debug(f"[WORKING] fetching {WORKING_URL} (timeout={FETCH_TIMEOUT}s)")
        r = requests.get(WORKING_URL, timeout=FETCH_TIMEOUT)
        if r.ok and r.text:
            logger.debug(f"[WORKING] fetched {len(r.text)} bytes")
            return r.text
        logger.debug(f"[WORKING] fetch status={r.status_code}")
    except Exception as e:
        logger.debug(f"[WORKING] fetch exception: {e}")
    return None

def _parse_working(text: str) -> Dict[str, List[str]]:
    """
    Parse lignes type: Provider|Model|modality
    Retourne: registry[model] = [providers...]
    """
    registry: Dict[str, List[str]] = {}
    cnt = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        provider, model = parts[0], parts[1]
        if not provider or not model:
            continue
        registry.setdefault(model, [])
        if provider not in registry[model]:
            registry[model].append(provider)
            cnt += 1
    logger.debug(f"[WORKING] parsed providers mappings: {cnt} entries for {len(registry)} models")
    return registry

# Charge registre dynamique (avec cache & fallback)
RAW = _fetch_working_text() or _load_cached_working()
if RAW is None:
    logger.warning("[WORKING] fetch + cache failed: using minimal built-in fallback list")
    RAW = """DeepInfra|deepseek-ai/DeepSeek-Prover-V2-671B|text
DeepInfra|Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo|text
DeepInfra|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|text
DeepInfra|meta-llama/Llama-3.3-70B-Instruct|text
DeepInfra|microsoft/phi-4-reasoning-plus|text
DeepInfra|deepseek-ai/DeepSeek-V3|text
DeepInfra|Qwen/Qwen3-235B-A22B-Instruct-2507|text
PerplexityLabs|sonar-pro|text
"""
else:
    _save_cached_working(RAW)

MODEL_REGISTRY: Dict[str, List[str]] = _parse_working(RAW)

# Registre "live" (résultats du scanner)
MODEL_REGISTRY_LIVE: dict = {}

def _build_test_pool() -> list[str]:
    """
    Construit une liste de candidats à tester:
    - tous les modèles présents dans MODEL_REGISTRY (raw keys)
    - + nos alias TIERS (chaînes aplaties)
    - + fallback env G4F_FALLBACK_MODELS (si défini)
    """
    pool = set()

    # 1) raw models du registre
    for model in MODEL_REGISTRY.keys():
        pool.add(model)
        # on ajoute aussi une forme 'AnyProvider:model' utile parfois
        pool.add(f"AnyProvider:{model}")

    # 2) tous les alias -> leurs tiers
    for alias, tier_models in TIERS.items():
        for m in tier_models:
            pool.add(m)
            pool.add(f"AnyProvider:{m}")

    # 3) fallback env (CSV)
    extra = os.getenv("G4F_FALLBACK_MODELS", "").strip()
    if extra:
        for x in extra.split(","):
            x = x.strip()
            if x:
                pool.add(x)

    # On retourne dans un ordre stable
    return sorted(pool)

def _inject_live_first(candidates: list[str]) -> list[str]:
    """
    Si le scanner a trouvé des variantes OK, on les place en tête des candidates,
    en conservant l'ordre original à la suite.
    """
    if not MODEL_REGISTRY_LIVE or not MODEL_REGISTRY_LIVE.get("ok"):
        return candidates
    live_variants = [item.get("variant") for item in MODEL_REGISTRY_LIVE["ok"] if item.get("variant")]
    # déduplique en conservant l'ordre
    seen: set[str] = set()
    out: list[str] = []
    for v in live_variants + candidates:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out

def _compose_model_string(provider: str, model: str) -> str:
    # g4f accepte "Provider:Model" ou parfois juste "model" (AnyProvider)
    out = f"{provider}:{model}" if provider and provider != "AnyProvider" else model
    logger.debug(f"[CHAIN] compose -> {out}")
    return out

# -----------------------------------------------------------------------------
# 2) Alias puissants + Tiers (du + fort au - fort)
# -----------------------------------------------------------------------------

TIERS: Dict[str, List[str]] = {
    # Code pur & complexe
    "qwen3-coder-480b": [
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1-0528",
    ],
    "deepseek-prover": [
        "deepseek-ai/DeepSeek-Prover-V2-671B",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ],
    # Mix code + raisonnement
    "deepseek-r1-distill-70b": [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-V3",
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "llama-3.3-70b": [
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-V3",
    ],
    # Raisonnement logique poussé
    "phi-4-reasoning-plus": [
        "microsoft/phi-4-reasoning-plus",
        "deepseek-ai/DeepSeek-Prover-V2-671B",
        "deepseek-ai/DeepSeek-V3",
    ],
    # Généralistes solides
    "deepseek-v3": [
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1-0528",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ],
    "qwen3-235b": [
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1-0528",
    ],
    "sonar-pro": [
        "sonar-pro",
        "sonar-reasoning-pro",
        "deepseek-ai/DeepSeek-V3",
    ],
}

GENERIC_TIER = [
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
]

def _build_chain(alias: str) -> List[str]:
    """
    Retourne une liste de candidates "Provider:Model" ordonnée:
    - Pour chaque modèle du TIER, on ajoute tous les providers qui l’exposent (depuis MODEL_REGISTRY).
    - Si rien, on essaie le prochain modèle du TIER.
    - Si alias inconnu, on applique GENERIC_TIER.
    """
    alias_low = alias.strip().lower()
    tier_models = TIERS.get(alias_low, GENERIC_TIER)
    chain: List[str] = []
    seen = set()
    logger.debug(f"[CHAIN] build for alias='{alias}' -> tier={tier_models}")
    for model in tier_models:
        providers = MODEL_REGISTRY.get(model, [])
        logger.debug(f"[CHAIN] model '{model}' providers={providers}")
        if not providers:
            continue
        for prov in providers:
            key = (prov, model)
            if key in seen:
                continue
            seen.add(key)
            chain.append(_compose_model_string(prov, model))
    if not chain:
        logger.warning(f"[CHAIN] empty for alias='{alias}', using generic scan")
        chain = [
            _compose_model_string(prov, model)
            for model in GENERIC_TIER
            for prov in MODEL_REGISTRY.get(model, [])
        ] or ["AnyProvider:gpt-4o-mini"]
    logger.debug(f"[CHAIN] result ({len(chain)}): {chain[:6]}{' ...' if len(chain)>6 else ''}")
    return chain

# -----------------------------------------------------------------------------
# 3) OpenAI-compatible helpers (SSE, message shaping, tools passthrough)
# -----------------------------------------------------------------------------

def _now_unix() -> int:
    return int(time.time())

def _new_id(prefix: str = "cmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"

def _strip_md_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.DOTALL)

def _shape_messages_to_prompt(messages: List[Dict]) -> str:
    # Compacte messages en un seul prompt “user” pour g4f (certains providers préfèrent)
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                (prt.get("text") or "") if isinstance(prt, dict) else str(prt)
                for prt in content
            )
        if role == "system":
            out.append(f"[SYSTEM]\n{content}\n")
        elif role == "user":
            out.append(f"User: {content}")
        elif role == "assistant":
            out.append(f"Assistant: {content}")
        else:
            out.append(f"{role}: {content}")
    prompt = "\n".join(out)
    logger.debug(f"[PROMPT] shaped messages -> {len(prompt)} chars")
    return prompt

def _sse_event(payload: Dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

def _tool_call_from_text(txt: str) -> Optional[Dict]:
    """
    Détecte un JSON tool_call renvoyé brut par le modèle et le transforme en
    OpenAI 'tool_calls'. On tolère les ```json ... ```
    """
    s = _strip_md_fences(txt)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            tool = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": obj["name"],
                    "arguments": json.dumps(obj["arguments"], ensure_ascii=False),
                },
            }
            logger.debug(f"[TOOLS] detected tool_call -> {tool['function']['name']}")
            return tool
    except Exception:
        pass
    return None

def _candidate_variants(candidate: str, alias: str) -> list[str]:
    """Génère des variantes d'un identifiant modèle pour maximiser la compatibilité g4f."""
    # candidate peut être "DeepInfra:deepseek-ai/DeepSeek-V3"
    prov = None
    model = candidate
    if ":" in candidate:
        prov, model = candidate.split(":", 1)
    base = model.split("/")[-1] if "/" in model else model

    variants = []
    # 1) provider:model (parfois accepté)
    if prov:
        variants.append(f"{prov}:{model}")
    # 2) modèle nu complet
    variants.append(model)
    # 3) basename (ex: DeepSeek-V3)
    variants.append(base)
    # 4) alias demandé (ex: deepseek-v3)
    if alias:
        variants.append(alias)

    # dédoublonner en conservant l'ordre
    seen = set()
    uniq = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq

# -----------------------------------------------------------------------------
# 4) g4f call (streaming) avec fallback multi-providers
# -----------------------------------------------------------------------------

def _call_g4f_stream(messages: List[Dict], prompt: str, candidates: List[str],
                     temperature: float, top_p: float, alias: str) -> Iterable[str]:
    """
    Générateur textuel (chunks) avec fallback:
    - Essaye d'abord g4f en mode 'messages' (stream=True)
    - Si erreur, fallback en une requête "messages=[{'role':'user','content':prompt}]"
    - Passe au prochain provider si échec
    - Pré-lis un peu et skip les backends "API key requise"
    """
    last_error = None
    prefetch_limit_chars = int(os.getenv("PREFETCH_HEAD_CHARS", "1200"))

    for candidate in candidates:
        tried = _candidate_variants(candidate, alias)
        logger.debug(f"[G4F] candidate={candidate} -> variants={tried}")

        for variant in tried:
            # ----- messages-mode
            logger.debug(f"[G4F] trying variant: {variant} (messages-mode)")
            try:
                gen = g4f.ChatCompletion.create(
                    model=variant,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    top_p=top_p,
                )
                head = ""
                buffered = []
                for chunk in gen:
                    chunk = str(chunk)
                    buffered.append(chunk)
                    head += chunk
                    if len(head) >= prefetch_limit_chars:
                        break
                # heuristique: gate api_key ?
                if _looks_like_api_key_gate(head):
                    logger.debug(f"[G4F] variant {variant} gated by API key; skipping")
                    continue  # essaie le prochain variant

                # Sinon: on renvoie le head puis on stream le reste
                for c in buffered:
                    yield c
                for chunk in gen:
                    yield str(chunk)
                logger.debug(f"[G4F] provider (messages) succeeded: {variant}")
                return
            except Exception as e1:
                last_error = e1
                logger.debug(f"[G4F] messages-mode failed on {variant}: {e1}")

            # ----- prompt-mode (fallback)
            logger.debug(f"[G4F] trying variant: {variant} (prompt-mode)")
            try:
                gen = g4f.ChatCompletion.create(
                    model=variant,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=temperature,
                    top_p=top_p,
                )
                head = ""
                buffered = []
                for chunk in gen:
                    chunk = str(chunk)
                    buffered.append(chunk)
                    head += chunk
                    if len(head) >= prefetch_limit_chars:
                        break
                if _looks_like_api_key_gate(head):
                    logger.debug(f"[G4F] variant {variant} gated by API key (prompt); skipping")
                    continue

                for c in buffered:
                    yield c
                for chunk in gen:
                    yield str(chunk)
                logger.debug(f"[G4F] provider (prompt) succeeded: {variant}")
                return
            except Exception as e2:
                last_error = e2
                logger.debug(f"[G4F] prompt-mode failed on {variant}: {e2}")
                continue

    # Si tout a échoué:
    err = f"All providers failed. Last error: {last_error}"
    logger.error(f"[G4F] {err}")
    yield f"[ERROR]{err}"

# -----------------------------------------------------------------------------
# 5) Endpoints OpenAI-compatibles
# -----------------------------------------------------------------------------

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True, silent=True)
    if data is None:
        app.logger.error("[HTTP] invalid JSON body; refusing request")
        return jsonify({"error": {"message": "Invalid JSON body"}}), 400
    messages: List[Dict] = data.get("messages", [])
    stream: bool = bool(data.get("stream", False))
    model_alias: str = (data.get("model") or "deepseek-v3").strip()
    tools = data.get("tools") or []          # transmis au LLM via consigne
    tool_choice = data.get("tool_choice")    # "auto" | {"type":"function","function":{"name":...}} | None
    temperature = float(data.get("temperature", 0.3))
    top_p = float(data.get("top_p", 1.0))
    max_tokens = data.get("max_tokens", None)

    logger.debug(f"[HTTP] /chat/completions stream={stream} model={model_alias} tools={bool(tools)} msgs={len(messages)}")

    # 1) Construire instruction (messages -> prompt unique)
    prompt = _shape_messages_to_prompt(messages)
    max_prompt_len = int(os.getenv("MAX_PROMPT_CHARS", "0"))  # 0 = illimité
    if max_prompt_len and len(prompt) > max_prompt_len:
        logger.debug(f"[PROMPT] clipping from {len(prompt)} to {max_prompt_len} chars")
        prompt = prompt[-max_prompt_len:]
    if tools:
        prompt += "\n\n[TOOLS]\nTu peux appeler des fonctions en renvoyant STRICTEMENT un JSON {\"name\":\"...\",\"arguments\":{...}}.\n"
        prompt += "Tools:\n" + json.dumps(tools, ensure_ascii=False, indent=2)

    # 2) Construire la chaîne de providers/models dynamiquement
    candidates = _build_chain(model_alias)
    # Inject live results first if available
    candidates = _inject_live_first(candidates)

    # Pin fallback pour les aliases "noisy" (qui souvent gating ou pas fiables)
    # Ex: deepseek-v3 → pin command-r ou aria comme premier fallback si pas de live results
    if model_alias.lower() in ("deepseek-v3", "deepseek-ai/deepseek-v3") and MODEL_REGISTRY_LIVE and MODEL_REGISTRY_LIVE.get("ok"):
        best_working = next((item.get("variant") for item in MODEL_REGISTRY_LIVE["ok"]
                            if item.get("variant") in ("command-r", "aria")), None)
        if best_working and best_working not in candidates:
            logger.debug(f"[FALLBACK] pinning {best_working} first for noisy alias {model_alias}")
            candidates.insert(0, best_working)

    if stream:
        def sse_stream():
            first_delta_sent = False
            buffer_text = ""
            created_ts = _now_unix()
            msg_id = _new_id()

            # Pré-envoi role=assistant (comme OpenAI)
            def _send_role_once():
                nonlocal first_delta_sent
                if not first_delta_sent:
                    payload = {
                        "id": msg_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                    yield _sse_event(payload)
                    first_delta_sent = True

            for piece in _call_g4f_stream(messages, prompt, candidates, temperature, top_p, model_alias):
                # Si un provider renvoie entièrement la réponse d’un coup
                if piece.startswith("[ERROR]"):
                    logger.debug("[SSE] upstream error chunk received")
                    # Envoi d’un delta minimal + le message d’erreur comme 'error'
                    yield from _send_role_once()
                    payload = {
                        "id": msg_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_alias,
                        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
                    }
                    yield _sse_event(payload)
                    yield "data: [DONE]\n\n"
                    return

                # Envoyer role au premier chunk
                for ev in _send_role_once():
                    yield ev

                txt = str(piece)
                buffer_text += txt
                # streamer le delta
                payload = {
                    "id": msg_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {"content": txt}, "finish_reason": None}],
                }
                yield _sse_event(payload)

            # tentative de détection d’un tool_call en fin de flux
            tool_call = _tool_call_from_text(buffer_text)
            if tool_call:
                logger.debug("[SSE] tool_call detected at stream end")
                yield _sse_event({
                    "id": msg_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
                })
                yield _sse_event({
                    "id": msg_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}],
                })
                yield _sse_event({
                    "id": msg_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                })
            else:
                # fin normale
                yield _sse_event({
                    "id": msg_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_alias,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
            yield "data: [DONE]\n\n"

        return Response(sse_stream(), mimetype="text/event-stream")

    # Non-streaming: on consomme tout puis on renvoie un seul objet OpenAI
    full_txt = ""
    for piece in _call_g4f_stream(messages, prompt, candidates, temperature, top_p, model_alias):
        if piece.startswith("[ERROR]"):
            logger.debug("[NON-STREAM] upstream error -> 502")
            return jsonify({"error": {"message": piece[7:]}}), 502
        full_txt += str(piece)

    tool_call = _tool_call_from_text(full_txt)
    msg = {"role": "assistant", "content": None if tool_call else full_txt}
    if tool_call:
        msg["tool_calls"] = [tool_call]

    response = {
        "id": _new_id(),
        "object": "chat.completion",
        "created": _now_unix(),
        "model": model_alias,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": "tool_calls" if tool_call else "stop"
        }],
    }
    # tokens comptage approximatif (optionnel)
    if max_tokens:
        response["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    logger.debug(f"[NON-STREAM] returning {('tool_calls' if tool_call else 'text')} response, len={len(full_txt)}")
    return Response(json.dumps(response, ensure_ascii=False), mimetype="application/json")

# Sanity
@app.route("/v1/models", methods=["GET"])
def list_models():
    # Expose les alias connus + quelques modèles bruts issus du registre, avec flag "working"
    aliases = sorted(list(TIERS.keys()))
    sample_raw = sorted(list(MODEL_REGISTRY.keys()))[:60]  # éviter de renvoyer 1000+ entrées

    # Si on a fait un scan récent, flag les working models
    working_variants = set()
    if MODEL_REGISTRY_LIVE and MODEL_REGISTRY_LIVE.get("ok"):
        working_variants = {item.get("variant") for item in MODEL_REGISTRY_LIVE["ok"] if item.get("variant")}

    def build_item(name: str, object_type: str = "model") -> dict:
        return {
            "id": name,
            "object": object_type,
            "working": name in working_variants
        }

    items = [build_item(a) for a in aliases] + [build_item(m) for m in sample_raw]
    logger.debug(f"[MODELS] aliases={len(aliases)} raw={len(sample_raw)} working={len(working_variants)} -> total={len(items)}")
    return jsonify({"object": "list", "data": items})

# Health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "ts": _now_unix()}), 200

@app.route("/v1/g4f/scan", methods=["POST", "GET"])
def scan_g4f():
    """
    Lance un scan parallèle des providers/modèles disponibles via g4f.
    Query params (facultatifs):
      timeout   : float (sec)
      parallel  : int
      sample    : int (0 = tous)
      prompt    : str court (ex: "ping")
    """
    logger = logging.getLogger("proxy")
    try:
        timeout = float(request.args.get("timeout", os.getenv("G4F_TEST_TIMEOUT", "6")))
    except Exception:
        timeout = 6.0
    try:
        parallel = int(request.args.get("parallel", os.getenv("G4F_TEST_PARALLEL", "12")))
    except Exception:
        parallel = 12
    try:
        sample = int(request.args.get("sample", os.getenv("G4F_TEST_SAMPLE", "0")))
    except Exception:
        sample = 0
    prompt = request.args.get("prompt", os.getenv("G4F_TEST_PROMPT", "ping"))

    pool = _build_test_pool()
    logger.debug("[TESTER] pool size=%d timeout=%.1f parallel=%d sample=%d prompt=%r",
                 len(pool), timeout, parallel, sample, prompt)

    summary = scan_candidates(
        candidates=pool,
        prompt=prompt,
        timeout=timeout,
        max_workers=parallel,
        sample=sample,
    )

    # Mets à jour le registre live en mémoire
    MODEL_REGISTRY_LIVE.clear()
    MODEL_REGISTRY_LIVE.update(summary)

    return jsonify(summary)

@app.route("/v1/working-models", methods=["GET"])
def working_models():
    """
    Renvoie la dernière mesure live (si scan récent) sinon le cache sur disque.
    """
    data = MODEL_REGISTRY_LIVE if MODEL_REGISTRY_LIVE else load_last_scan() or {"ok": []}
    return jsonify(data)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "4000"))
    logger.debug(f"[BOOT] starting Flask debug on {host}:{port}")
    app.run(host=host, port=port, debug=True)
