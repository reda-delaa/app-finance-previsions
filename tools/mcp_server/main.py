# /Users/venom/Documents/analyse-financiere/mcp_server/main.py
import os, sys, json, asyncio, logging, subprocess, traceback
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Logs uniquement sur STDERR (stdout = JSON-RPC pour MCP)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

APP_ROOT = Path(__file__).resolve().parents[1]   # .../analyse-financiere
ORCH = APP_ROOT / "orch" / "orchestrator.py"

app = FastMCP("orch-local")

@app.tool()
def ping() -> str:
    """Petit check de santé du serveur MCP."""
    return "pong"

@app.tool()
def run_goal(goal: str) -> dict:
    """
    Lance orch/orchestrator.py avec --goal "<texte>" et renvoie son JSON.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(APP_ROOT))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("LLM_URL", env.get("LLM_URL", "http://127.0.0.1:4000/v1/chat/completions"))
    env.setdefault("LLM_MODEL", env.get("LLM_MODEL", "command-r"))

    cmd = [sys.executable, str(ORCH), "--goal", goal]
    try:
        out = subprocess.check_output(
            cmd,
            cwd=str(APP_ROOT),
            env=env,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        return json.loads(out)
    except subprocess.CalledProcessError as e:
        logging.error("orchestrator stderr:\n%s", e.stderr)
        return {"done": False, "error": f"orchestrator failed: {e}", "stderr": e.stderr}
    except Exception as e:
        logging.error("unexpected:\n%s", traceback.format_exc())
        return {"done": False, "error": str(e)}

if __name__ == "__main__":
    # IMPORTANT : ne rien imprimer sur stdout ici ; FastMCP gère l'IO stdio.
    # run_stdio_async démarre le serveur JSON-RPC sur stdin/stdout
    asyncio.run(app.run_stdio_async())
