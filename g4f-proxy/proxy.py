# proxy.py
import os, re, json, time, uuid, logging
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

import g4f  # pip install g4f==0.2.9.9  (ou ta version fonctionnelle)

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("g4f-proxy")

# =========================
# Modèle par défaut (puissant)
# =========================
# Choisis d’abord un modèle costaud présent dans ta liste working.
# On met une registry pour alias OpenAI -> g4f (provider/model)
MODEL_REGISTRY = {
    # alias OpenAI-ish           (provider, g4f_model)
    "gpt-4o-mini":               ("AnyProvider", "deepseek-v3"),
    "gpt-4.1":                   ("AnyProvider", "deepseek-v3"),
    "gpt-4.1-mini":              ("AnyProvider", "deepseek-v3"),
    "gpt-4":                     ("AnyProvider", "llama-3.3-70b"),
    "qwen-max":                  ("Qwen", "qwen-max"),
    "deepseek-v3":               ("AnyProvider", "deepseek-v3"),
    "sonar-pro":                 ("PerplexityLabs", "sonar-pro"),
}

DEFAULT_MODEL_ID = os.getenv("OPENAI_COMPAT_MODEL", "gpt-4o-mini")

def now_unix() -> int:
    return int(time.time())

def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"

# =========================
# Helpers tools & parsing
# =========================
FENCED_JSON = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)

def extract_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Essaie de récupérer un bloc JSON (tool call) soit en JSON brut,
    soit dans un code fence ```json ...```.
    """
    if not text:
        return None
    m = FENCED_JSON.search(text)
    candidate = m.group(1) if m else text
    try:
        obj = json.loads(candidate)
        # attendu: {"name": "...", "arguments": {...}}
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return obj
        return None
    except Exception:
        return None

def openai_message(role: str, content: Optional[str], tool_calls: Optional[List[Dict]] = None) -> Dict:
    msg = {"role": role}
    if content is not None:
        msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg

# =========================
# /health & /v1/models
# =========================
@app.get("/health")
def health():
    return {"ok": True, "time": now_unix()}, 200

@app.get("/v1/models")
def list_models():
    data = []
    created = now_unix()
    for mid in sorted(MODEL_REGISTRY.keys()):
        data.append({"id": mid, "object": "model", "created": created, "owned_by": "g4f-proxy"})
    return jsonify({"object": "list", "data": data}), 200

@app.get("/v1/models/<mid>")
def get_model(mid: str):
    if mid not in MODEL_REGISTRY:
        return jsonify({"error": {"message": f"Unknown model '{mid}'", "type": "not_found"}}), 404
    return jsonify({"id": mid, "object": "model", "created": now_unix(), "owned_by": "g4f-proxy"}), 200

# =========================
# Appel g4f (avec timeout+retry)
# =========================
def call_g4f(model_id: str, prompt: str, temperature: float, max_tokens: Optional[int]) -> Tuple[str, Dict]:
    provider, g4f_model = MODEL_REGISTRY.get(model_id, MODEL_REGISTRY[DEFAULT_MODEL_ID])
    retries = 2
    last_err = None
    for attempt in range(retries + 1):
        try:
            # g4f accepte généralement messages=[...] ; certains backends préfèrent prompt=...
            resp = g4f.ChatCompletion.create(
                model=g4f_model,
                provider=provider,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = str(resp)
            return text, {"provider": provider, "g4f_model": g4f_model, "attempt": attempt}
        except Exception as e:
            last_err = str(e)
            log.warning("g4f error (try %s/%s): %s", attempt+1, retries+1, last_err)
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"g4f failed after retries: {last_err}")

def build_prompt_from_messages(messages: List[Dict], tools: List[Dict]) -> str:
    # On garde le format ‘chatml rustique’ + instructions tools si fournis.
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "tool":
            name = m.get("name") or m.get("tool_name") or "tool"
            parts.append(f"Tool[{name}]: {content}")
        else:
            parts.append(f"User: {content}")
    if tools:
        parts.append(
            "IMPORTANT: If a function call is needed, OUTPUT A SINGLE JSON object with keys "
            "`name` and `arguments` (arguments is a JSON object). Do not add prose with the JSON."
        )
        parts.append("TOOLS=" + json.dumps(tools, ensure_ascii=False))
    return "\n".join(parts)

# =========================
# /v1/chat/completions
# =========================
@app.post("/v1/chat/completions")
def chat_completions():
    try:
        data = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": {"message": "Invalid JSON body", "type": "bad_request"}}), 400

    model_id = data.get("model") or DEFAULT_MODEL_ID
    messages = data.get("messages") or []
    tools = data.get("tools") or []
    tool_choice = (data.get("tool_choice") or "auto")
    temperature = float(data.get("temperature", 0.2))
    max_tokens = data.get("max_tokens", None)
    stream = bool(data.get("stream", False))

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": {"message": "`messages` must be a non-empty array", "type": "bad_request"}}), 400

    prompt = build_prompt_from_messages(messages, tools)

    def make_response_payload(content: Optional[str], tool_calls: Optional[List[Dict]]):
        return {
            "id": gen_id("cmpl"),
            "object": "chat.completion",
            "created": now_unix(),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": openai_message("assistant", content, tool_calls),
                "finish_reason": "stop" if not tool_calls else "tool_calls"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # Pas de streaming SSE complet ici (OpenAI delta). On renvoie du JSON final (Cline accepte).
    try:
        text, meta = call_g4f(model_id, prompt, temperature, max_tokens)
    except Exception as e:
        log.exception("g4f fatal error")
        return jsonify({"error": {"message": f"Upstream error: {e}", "type": "upstream"}}), 502

    tool_calls = None
    if tools and tool_choice != "none":
        tc = extract_tool_json(text)
        if tc:
            tool_calls = [{
                "id": gen_id("call"),
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)}
            }]
            text = None  # OpenAI renvoie souvent content=None lorsque tool_calls est présent

    return jsonify(make_response_payload(text, tool_calls)), 200

# =========================
# Entrée
# =========================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "4000"))
    log.info("Starting g4f proxy on %s:%d (default model: %s)", host, port, DEFAULT_MODEL_ID)
    app.run(host=host, port=port, debug=True)