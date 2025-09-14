# proxy.py
# OpenAI-compatible proxy au-dessus de g4f, avec:
# - Streaming SSE
# - Tools (function calling) basiques (passe-through / JSON)
# - Découverte dynamique des modèles "working" (GitHub) + cache local
# - Fallbacks intelligents par TIER (du plus puissant au moins puissant)
#
# Env utiles:
#   G4F_WORKING_URL   (override de l’URL GitHub)
#   G4F_WORKING_CACHE (chemin du cache local, défaut: .g4f_working.txt)
#   G4F_FETCH_TIMEOUT (seconds, défaut: 6)
#
# Lancement:
#   pip install flask flask-cors requests g4f
#   python proxy.py
#
# Intégration Cline:
#   Base URL: http://127.0.0.1:4000/v1
#   Model: p.ex. "qwen3-coder-480b", "deepseek-prover", "deepseek-r1-distill-70b",
#           "llama-3.3-70b", "phi-4-reasoning-plus", "deepseek-v3", "qwen3-235b"

import os
import re
import json
import uuid
import time
from typing import Dict, List, Tuple, Optional, Iterable

import requests
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import g4f

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# 1) Fetch & parse "working models" (provider|model|modality) at startup
# ---------------------------------------------------------------------------

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
                return f.read()
    except Exception:
        pass
    return None

def _save_cached_working(txt: str) -> None:
    try:
        with open(WORKING_CACHE, "w", encoding="utf-8") as f:
            f.write(txt)
    except Exception:
        pass

def _fetch_working_text() -> Optional[str]:
    try:
        r = requests.get(WORKING_URL, timeout=FETCH_TIMEOUT)
        if r.ok and r.text:
            return r.text
    except Exception:
        return None
    return None

def _parse_working(text: str) -> Dict[str, List[str]]:
    """
    Parse lignes type: Provider|Model|modality
    Retourne: registry[model] = [providers...]
    """
    registry: Dict[str, List[str]] = {}
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
    return registry

# Charge registre dynamique (avec cache & fallback)
RAW = _fetch_working_text() or _load_cached_working()
if RAW is None:
    # Fallback minimal si tout échoue
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

def _compose_model_string(provider: str, model: str) -> str:
    # g4f accepte "Provider:Model" ou parfois juste "model" (AnyProvider)
    return f"{provider}:{model}" if provider and provider != "AnyProvider" else model

# ---------------------------------------------------------------------------
# 2) Alias puissants + Tiers (du + fort au - fort)
#    Pour chaque alias, on essaie TOUS les providers qui exposent le model cible,
#    puis on passe au model suivant du tier.
# ---------------------------------------------------------------------------

# Tiers ordonnés pour chaque “usage”
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

# Fallback générique si l’alias n’est pas connu
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
    alias = alias.strip().lower()
    tier_models = TIERS.get(alias, GENERIC_TIER)
    chain: List[str] = []
    seen = set()
    for model in tier_models:
        providers = MODEL_REGISTRY.get(model, [])
        if not providers:
            continue
        for prov in providers:
            key = (prov, model)
            if key in seen:
                continue
            seen.add(key)
            chain.append(_compose_model_string(prov, model))
    # Si la chaîne est vide, dernier filet de sécurité
    if not chain:
        chain = [
            _compose_model_string(prov, model)
            for model in GENERIC_TIER
            for prov in MODEL_REGISTRY.get(model, [])
        ] or ["AnyProvider:gpt-4o-mini"]
    return chain

# ---------------------------------------------------------------------------
# 3) OpenAI-compatible helpers (SSE, message shaping, tools passthrough)
# ---------------------------------------------------------------------------

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
    return "\n".join(out)

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
            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": obj["name"],
                    "arguments": json.dumps(obj["arguments"], ensure_ascii=False),
                },
            }
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# 4) Chat Completions endpoint
# ---------------------------------------------------------------------------

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True, silent=True) or {}
    messages: List[Dict] = data.get("messages", [])
    stream: bool = bool(data.get("stream", False))
    model_alias: str = (data.get("model") or "deepseek-v3").strip()
    tools = data.get("tools") or []          # transmis au LLM via consigne
    tool_choice = data.get("tool_choice")    # "auto" | {"type":"function","function":{"name":...}} | None
    temperature = data.get("temperature", 0.3)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", None)

    # 1) Construire instruction (messages -> prompt unique)
    prompt = _shape_messages_to_prompt(messages)
    if tools:
        prompt += "\n\n[TOOLS]\nTu peux appeler des fonctions en renvoyant STRICTEMENT un JSON {\"name\":\"...\",\"arguments\":{...}}.\n"
        prompt += "Tools:\n" + json.dumps(tools, ensure_ascii=False, indent=2)

    # 2) Construire la chaîne de providers/models dynamiquement
    candidates = _build_chain(model_alias)

    # 3) Fonction d’appel g4f (avec fallback)
    def call_g4f_yield() -> Iterable[str]:
        last_error = None
        for candidate in candidates:
            try:
                # Certains backends g4f acceptent messages "au format OpenAI".
                # On tente la version “messages” d’abord, sinon fallback sur prompt unique.
                # Streaming: g4f renvoie déjà des chunks / strings -> on transforme en SSE.

                # a) Essai en mode messages
                try:
                    gen = g4f.ChatCompletion.create(
                        model=candidate,
                        messages=messages,
                        stream=True,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    for chunk in gen:
                        yield str(chunk)
                    return
                except Exception as e1:
                    last_error = e1

                # b) Fallback en mode prompt unique
                gen = g4f.ChatCompletion.create(
                    model=candidate,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=temperature,
                    top_p=top_p,
                )
                for chunk in gen:
                    yield str(chunk)
                return
            except Exception as e:
                last_error = e
                continue
        # Si tout a échoué:
        err = f"All providers failed for alias '{model_alias}'. Last error: {last_error}"
        yield f"[ERROR]{err}"

    # 4) Réponses
    if stream:
        def sse_stream():
            first_delta_sent = False
            buffer_text = ""
            for piece in call_g4f_yield():
                # Si un provider renvoie entièrement la réponse d’un coup
                if piece.startswith("[ERROR]"):
                    # Envoi d’un delta minimal + le message d’erreur dans un choix terminal
                    if not first_delta_sent:
                        yield _sse_event({"id": _new_id(), "object": "chat.completion.chunk",
                                          "created": _now_unix(), "model": model_alias,
                                          "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                        first_delta_sent = True
                    yield _sse_event({"id": _new_id(), "object": "chat.completion.chunk",
                                      "created": _now_unix(), "model": model_alias,
                                      "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]})
                    yield "data: [DONE]\n\n"
                    return

                # Envoyer role au premier chunk
                if not first_delta_sent:
                    yield _sse_event({"id": _new_id(), "object": "chat.completion.chunk",
                                      "created": _now_unix(), "model": model_alias,
                                      "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                    first_delta_sent = True

                txt = str(piece)
                buffer_text += txt
                # streamer le delta
                yield _sse_event({"id": _new_id(), "object": "chat.completion.chunk",
                                  "created": _now_unix(), "model": model_alias,
                                  "choices": [{"index": 0, "delta": {"content": txt}, "finish_reason": None}]})

            # tentative de détection d’un tool_call en fin de flux
            tool_call = _tool_call_from_text(buffer_text)
            if tool_call:
                # On “vide” le texte et on renvoie un tool_call final (OpenAI-style)
                yield _sse_event({"id": _new_id(), "object": "chat.completion.chunk",
                                  "created": _now_unix(), "model": model_alias,
                                  "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]})
                # encodage tool_calls pour OpenAI SSE
                yield _sse_event({"id": _new_id(), "object": "chat.completion.chunk",
                                  "created": _now_unix(), "model": model_alias,
                                  "choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}]})
                yield _sse_event({"id": _new_id(), "object": "chat.completion.chunk",
                                  "created": _now_unix(), "model": model_alias,
                                  "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
            else:
                # fin normale
                yield _sse_event({"id": _new_id(), "object": "chat.completion.chunk",
                                  "created": _now_unix(), "model": model_alias,
                                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"

        return Response(sse_stream(), mimetype="text/event-stream")

    # Non-streaming: on consomme tout puis on renvoie un seul objet OpenAI
    full_txt = ""
    for piece in call_g4f_yield():
        if piece.startswith("[ERROR]"):
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
    return Response(json.dumps(response, ensure_ascii=False), mimetype="application/json")

# Sanity
@app.route("/v1/models", methods=["GET"])
def list_models():
    # Expose les alias connus + quelques modèles bruts issus du registre
    aliases = sorted(list(TIERS.keys()))
    sample_raw = sorted(list(MODEL_REGISTRY.keys()))[:60]  # éviter de renvoyer 1000+ entrées
    items = [{"id": a, "object": "model"} for a in aliases] + \
            [{"id": m, "object": "model"} for m in sample_raw]
    return jsonify({"object": "list", "data": items})

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "4000"))
    app.run(host=host, port=port, debug=True)