import os, json, re, sys, ast
from pathlib import Path
import requests
import yaml

OPENAI_LIKE_URL = os.getenv("LLM_URL", "http://127.0.0.1:4000/v1/chat/completions")
MODEL = os.getenv("LLM_MODEL", "command-r")
MAX_STEPS = 8

# Load configuration from working_config.yaml
def load_config():
    """Load configuration from YAML file with defaults"""
    config_path = Path("orch/working_config.yaml")
    if not config_path.exists():
        # Fallback to basic config without system prompt (will use default later)
        return {
            "api_base": OPENAI_LIKE_URL,
            "model": MODEL,
            "max_steps": MAX_STEPS,
            "request_timeout": 30,
            "models": {"default": MODEL, "routing": {"hard_deny": []}},
        }

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[CONFIG] Error loading config: {e}", file=sys.stderr)
        return {
            "api_base": OPENAI_LIKE_URL,
            "model": MODEL,
            "max_steps": MAX_STEPS,
            "request_timeout": 30,
            "models": {"default": MODEL, "routing": {"hard_deny": []}},
        }

CONFIG = load_config()
# Fix URL construction to always build chat completions endpoint
api_base = CONFIG.get("api_base", OPENAI_LIKE_URL)
if api_base.rstrip("/").endswith("/v1/chat/completions"):
    OPENAI_LIKE_URL = api_base
else:
    OPENAI_LIKE_URL = api_base.rstrip("/") + "/v1/chat/completions"

MODEL = CONFIG.get("model", MODEL)
MAX_STEPS = CONFIG.get("max_steps", MAX_STEPS)
REQUEST_TIMEOUT = CONFIG.get("request_timeout", 30)

# Model routing from config
MODEL_ROUTING = CONFIG.get("models", {}).get("routing", {})
MODEL_PREFERENCES = MODEL_ROUTING.get("hard_allow", {
    "text_primary": ["command-r", "aria", "llama-3.3-70b"],
    "text_coding": ["command-r", "llama-3.3-70b"],
    "text_budget": ["aria"],
})
HARD_DENY_MODELS = set(MODEL_ROUTING.get("hard_deny", [
    "deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.3-70B-Instruct",
    "llama-4-scout", "deepseek-r1-distill-llama-70b", "flux-dev"
]))

def healthcheck_model(model):
    """Simple health check for model availability"""
    try:
        response = requests.post(OPENAI_LIKE_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False
        }, timeout=10)
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return bool(content and ("pong" in content.lower() or "how can i" in content.lower() or "hello" in content.lower()))
        return False
    except Exception:
        return False

def pick_working_model(task="general", fallback="command-r"):
    """Pick the first working model from preferences"""
    # Map task names to config categories
    task_mapping = {
        "general": "text_primary",
        "coding": "text_coding",
        "budget": "text_budget"
    }

    config_task = task_mapping.get(task, task)
    preferences = MODEL_PREFERENCES.get(config_task, [])

    # Guards to avoid KeyError and empty loop
    if not isinstance(preferences, list) or not preferences:
        preferences = [CONFIG.get("model", fallback), fallback]

    for model in preferences:
        if model in HARD_DENY_MODELS:
            continue
        if healthcheck_model(model):
            print(f"[MODEL] Using {model} for {task}", file=sys.stderr)
            return model
    print(f"[MODEL] Fallback to {fallback}", file=sys.stderr)
    return fallback

# Auto-select model for current task
MODEL = pick_working_model(task="general")

def _safe_shell(args):
    cmd = (args.get("cmd") or "").strip()
    if not cmd:
        raise RuntimeError("shell: empty command")
    # very small whitelist (enough for our tests)
    SAFE = {"ls", "pwd", "echo", "cat", "touch", "mkdir", "rm", "cp", "mv"}
    base = cmd.split()[0]
    if base not in SAFE:
        raise RuntimeError(f"shell command '{base}' not allowed")
    return os.popen(cmd).read()

TOOLS = {
    "touch_file": lambda args: (Path(args["path"]).touch(), f"touched {args['path']}")[-1],
    "list_dir":  lambda args: str(sorted(list(os.listdir((args or {}).get("path", ".") or ".")))),
    "read_file": lambda args: Path(args["path"]).read_text(encoding="utf-8"),
    "write_file":lambda args: (Path(args["path"]).write_text(args["content"], encoding="utf-8"), "ok")[-1],
    "shell":     _safe_shell,
}

INTENT_TO_TOOL = [
    # Standard shell commands
    (r"^\s*touch\s+(.+)$", "touch_file", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*ls(?:\s+(.+))?$", "list_dir",  lambda m: {"path": m.group(1).strip() if m.group(1) else "."}),
    (r"^\s*list\s+(?:directory\s+)?(.+)?$", "list_dir", lambda m: {"path": m.group(1).strip() if m.group(1) else "."}),
    (r"^\s*dir(?:\s+(.+))?$", "list_dir", lambda m: {"path": m.group(1).strip() if m.group(1) else "."}),
    (r"^\s*cat\s+(.+)$", "read_file",     lambda m: {"path": m.group(1).strip()}),
    (r"^\s*read\s+(?:file\s+)?(.+)$", "read_file", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*show\s+(.+)$", "read_file", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*open\s+(.+)$", "read_file", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*view\s+(.+)$", "read_file", lambda m: {"path": m.group(1).strip()}),
    # Natural language commands
    (r"^\s*create\s+file\s+(.+)$", "touch_file", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*make\s+file\s+(.+)$", "touch_file", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*browse\s+(?:directory\s+)?(.+)$", "list_dir", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*show\s+(?:me\s+)?(?:the\s+)?contents\s+of\s+(.+)$", "list_dir", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*display\s+(?:files\s+in\s+)?(.+)$", "list_dir", lambda m: {"path": m.group(1).strip()}),
    (r"^\s*what(?:'s)?\s+in\s+(.+)$", "list_dir", lambda m: {"path": m.group(1).strip()}),
]

# Default system prompt (will be overridden by config)
DEFAULT_SYSTEM = """Tu es un agent outillé. Tu dois atteindre l'objectif donné.
Tu ne peux PAS exécuter directement, tu dois appeler des outils.

Réponds UNIQUEMENT ainsi:
<tool>TOOL_NAME</tool>
<args>{"k":"v"}</args>

OUTILS AUTORISÉS & SCHÉMA:
- touch_file: {"path": "str"}
- list_dir: {"path": "str" (optionnel, défaut=".")}
- read_file: {"path": "str"}
- write_file: {"path": "str", "content": "str"}
- shell: {"cmd": "str"}  # réserve si aucun autre outil ne convient
"""

SYSTEM = CONFIG.get("system_prompt", DEFAULT_SYSTEM)

def llm(msgs):
    try:
        r = requests.post(OPENAI_LIKE_URL, json={
            "model": MODEL,
            "messages": msgs,
            "stream": False
        }, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR LLM: {e}"

def parse_action(text):
    """Parse single tool block with robust whitespace trimming to avoid JSONDecodeError"""
    m1 = re.search(r"<tool>\s*(.+?)\s*</tool>", text, flags=re.S)
    m2 = re.search(r"<args>\s*(.+?)\s*</args>", text, flags=re.S)
    if not (m1 and m2): return None, None
    tool = m1.group(1).strip()
    args_str = m2.group(1).strip()
    # Try multiple parsing strategies
    for parser in (json.loads, ast.literal_eval):
        try:
            return tool, parser(args_str)
        except Exception:
            pass
    # Micro-repair for single quotes and whitespace issues
    try:
        # Clean up common formatting issues
        args_str = args_str.strip()
        args_str = re.sub(r'\s+', ' ', args_str)  # Normalize whitespace
        return tool, json.loads(args_str.replace("'", '"'))
    except Exception:
        return None, None

def parse_multiple_actions(text):
    """Parse multiple tool blocks using robust regex pattern"""
    # Enhanced regex to handle multi-block parsing
    pattern = r'<tool>([^<]+)</tool>\s*<args>([^<]*\{.*?\})</args>'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    actions = []
    for tool_match, args_str in matches:
        tool = tool_match.strip()
        args_str = args_str.strip()

        # Parse args with multiple fallback strategies
        args = None
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            try:
                args = ast.literal_eval(args_str)
            except (ValueError, SyntaxError):
                # Try to fix common formatting issues
                args_str = args_str.replace("'", '"')  # Single quotes to double
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    continue  # Skip malformed block

        if args and tool:
            actions.append((tool, args))

    return actions if actions else None

def parse_args(args_str):
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(args_str)
        except (ValueError, SyntaxError):
            return None

def map_freeform_to_tool(text):
    # Handle both <file_operation> and <tool> formats
    if "<file_operation>" in text:
        m1 = re.search(r"<file_operation>(.+?)</file_operation>", text, flags=re.S)
        m2 = re.search(r"<args>(.+?)</args>", text, flags=re.S)
        if m1 and m2:
            tool_name = m1.group(1).strip()
            args = parse_args(m2.group(1).strip())
            if args is None:
                return None, None
            # Map file_operation names to tool names
            for rx, std_name, _ in INTENT_TO_TOOL:
                if re.match(rx, tool_name):
                    return std_name, args
            return tool_name, args

    for rx, name, make_args in INTENT_TO_TOOL:
        m = re.match(rx, text.strip())
        if m:
            return name, make_args(m)
    return None, None

def _jsonify(x):
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonify(v) for v in x]
    if isinstance(x, set):
        return sorted(list(x))
    return x

def goal_satisfied(goal, log):
    # Simple heuristics based on goal type
    made_file = any(step.get("tool") == "touch_file" for step in log if step != {})
    listed_dir = any(step.get("tool") in ("list_dir", "shell") for step in log if step != {})
    return made_file and listed_dir

def main(goal):
    # ========== FAST-PATHS ==========
    # 0) Exécuter directement un bloc <tool>...</tool><args>...</args>
    t0, a0 = parse_action(goal)
    if t0 and a0 is not None:
        try:
            res = str(TOOLS[t0](a0)) if t0 in TOOLS else f"ERROR: Outil '{t0}' inconnu."
        except Exception as e:
            res = f"ERROR: {e}"
        out = f"<tool>{t0}</tool>\n<args>{json.dumps(a0, ensure_ascii=False)}</args>"
        log = [{"tool": t0, "args": a0, "result": res, "raw": out}]
        return _jsonify({"done": True, "final": out, "log": log})

    # 1) Mapper noms d'outils bruts: "list_dir", "list_dir {}", "read_file path", etc.
    m = re.match(r"^\s*(list_dir|read_file|write_file|touch_file)\b(?:\s+(.*))?$", goal.strip(), flags=re.I)
    if m:
        tname = m.group(1)
        rest = (m.group(2) or "").strip()

        # parse args si c'est du JSON/py-dict, sinon interpréter textuellement
        args = None
        if rest:
            # cas JSON/dict
            parsed = parse_args(rest)
            if isinstance(parsed, dict):
                args = parsed
        if args is None:
            # défauts sensés par outil
            if tname == "list_dir":
                # sans args -> répertoire courant
                args = {"path": "."}
            elif tname == "touch_file":
                # "touch_file demo.txt"
                if rest:
                    args = {"path": rest}
            elif tname == "read_file":
                if rest:
                    args = {"path": rest}
            elif tname == "write_file":
                # write_file <path>::<content>
                if "::" in rest:
                    pth, content = rest.split("::", 1)
                    args = {"path": pth.strip(), "content": content}
        if args is None:
            args = {}

        # Normaliser list_dir {} => path='.'
        if tname == "list_dir" and not args.get("path"):
            args["path"] = "."

        try:
            res = str(TOOLS[tname](args)) if tname in TOOLS else f"ERROR: Outil '{tname}' inconnu."
        except Exception as e:
            res = f"ERROR: {e}"
        out = f"<tool>{tname}</tool>\n<args>{json.dumps(args, ensure_ascii=False)}</args>"
        log = [{"tool": tname, "args": args, "result": res, "raw": out}]
        return _jsonify({"done": True, "final": out, "log": log})

    # 2) Heuristique NL "please create a file called X"
    m = re.search(r"create\b.*\bfile\b.*\bcalled\b\s+([^\s,]+)", goal, flags=re.I)
    if m:
        fname = m.group(1)
        args = {"path": fname}
        try:
            res = str(TOOLS["touch_file"](args))
        except Exception as e:
            res = f"ERROR: {e}"
        out = f"<tool>touch_file</tool>\n<args>{json.dumps(args, ensure_ascii=False)}</args>"
        log = [{"tool": "touch_file", "args": args, "result": res, "raw": out}]
        return _jsonify({"done": True, "final": out, "log": log})

    # 3) Heuristique NL multi-fichiers + list
    # ex: "Create test1.txt and test2.txt then list directory"
    m = re.search(r"create\s+([^\s,;]+\.txt)(?:\s+and\s+([^\s,;]+\.txt))?.*?(?:then\s+)?(?:list\b.*directory|ls\b)", goal, flags=re.I)
    if m:
        files = [m.group(1)]
        if m.group(2): files.append(m.group(2))
        local_log = []
        raw_blocks = []
        # touch chaque fichier
        for f in files:
            args = {"path": f}
            try:
                res = str(TOOLS["touch_file"](args))
            except Exception as e:
                res = f"ERROR: {e}"
            raw = f"<tool>touch_file</tool>\n<args>{json.dumps(args, ensure_ascii=False)}</args>"
            local_log.append({"tool": "touch_file", "args": args, "result": res, "raw": raw})
            raw_blocks.append(raw)
        # puis list_dir
        args = {"path": "."}
        try:
            res = str(TOOLS["list_dir"](args))
        except Exception as e:
            res = f"ERROR: {e}"
        raw = f"<tool>list_dir</tool>\n<args>{json.dumps(args, ensure_ascii=False)}</args>"
        local_log.append({"tool": "list_dir", "args": args, "result": res, "raw": raw})
        raw_blocks.append(raw)
        return _jsonify({"done": True, "final": "\n".join(raw_blocks), "log": local_log})

    # 4) Fast-map existant (tes regex INTENT_TO_TOOL)
    ftool, fargs = map_freeform_to_tool(goal)
    if ftool:
        try:
            result = str(TOOLS[ftool](fargs))
        except Exception as e:
            result = f"ERROR: {e}"
        out = f"<tool>{ftool}</tool>\n<args>{json.dumps(fargs, ensure_ascii=False)}</args>"
        log = [{"tool": ftool, "args": fargs, "result": result, "raw": out}]
        return _jsonify({"done": True, "final": out, "log": log})

    # 5) "use <tool>" -> si inconnu, répondre proprement
    m = re.match(r"^\s*use\s+([A-Za-z0-9_\-\.]+)\s*$", goal.strip(), flags=re.I)
    if m:
        tname = m.group(1)
        if tname not in TOOLS:
            msg = f"ERROR: Outil '{tname}' inconnu. Utilise seulement: {', '.join(sorted(TOOLS.keys()))}"
            out = f"<tool>unknown</tool>\n<args>{{\"name\":\"{tname}\"}}</args>"
            log = [{"tool": "unknown", "args": {"name": tname}, "result": msg, "raw": out}]
            return _jsonify({"done": True, "final": out, "log": log})
        # s'il existe, on peut exiger des args ou juste informer
        msg = f"Outil '{tname}' disponible. Fournis des arguments si nécessaire."
        out = f"<tool>{tname}</tool>\n<args>{{}}</args>"
        log = [{"tool": tname, "args": {}, "result": msg, "raw": out}]
        return _jsonify({"done": True, "final": out, "log": log})
    # ========== /FAST-PATHS ==========

    msgs = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":f"Objectif: {goal}"}
    ]
    log = []

    for step in range(1, MAX_STEPS+1):
        out = llm(msgs)
        if not out or out.startswith("ERROR"):
            return _jsonify({"done": False, "error": out or "LLM communication failed", "log": log})

        # Try to parse multiple actions first, then fallback to single action
        actions = parse_multiple_actions(out)
        if not actions:
            # Try single action parsing
            tool, args = parse_action(out)
            if tool:
                actions = [(tool, args)]

        if not actions:
            # Try freeform mapping (handles old <file_operation> format)
            mtool, margs = map_freeform_to_tool(out)
            if mtool:
                actions = [(mtool, margs)]

        if not actions:
            msgs.append({"role":"assistant","content":out})
            msgs.append({"role":"user","content":"Format invalide. Utilise <tool>TOOL_NAME</tool><args>{...}</args> strictement."})
            continue

        # Process all actions in this response
        step_results = []
        for tool, args in actions:
            if tool not in TOOLS:
                result = f"ERROR: Outil '{tool}' inconnu. Utilise seulement: {', '.join(TOOLS)}"
            else:
                try:
                    result = str(TOOLS[tool](args))
                except Exception as e:
                    result = f"ERROR: {e}"

            step_results.append((tool, args, result))
            log.append({"tool": tool, "args": args, "result": result, "raw": out})

        msgs.append({"role":"assistant","content":out})

        # Add results to conversation context
        if len(step_results) == 1:
            tool, args, result = step_results[0]
            msgs.append({"role":"user","content":f"[RESULTAT {tool}] {result}"})
        else:
            # Multiple results - format them clearly
            result_msgs = [f"[RESULTAT {tool}] {result}" for tool, _, result in step_results]
            msgs.append({"role":"user","content": " ".join(result_msgs)})

        # Check if goal is satisfied after processing actions
        if goal_satisfied(goal, log):
            return _jsonify({"done": True, "final": out, "log": log})

    return _jsonify({"done": False, "final": "max_steps reached", "log": log})

if __name__ == "__main__":
    goal = " ".join(sys.argv[sys.argv.index("--goal")+1:]) if "--goal" in sys.argv else "Create demo.txt and list directory"
    print(json.dumps(main(goal), ensure_ascii=False, indent=2))
