# -*- coding: utf-8 -*-
"""
mcp_tools.py
------------
Adapteurs MCP stdio pour l'agent:
- filesystem.list_allowed_directories
- api-supermemory-ai.whoAmI
- everything.echo
- sequential-thinking.sequentialthinking

Points clés:
- Transport stdio JSON-RPC 2.0 avec framing "Content-Length" conforme MCP
  + fallback tolérant aux serveurs qui envoient du JSON par ligne.
- Logs DEBUG très verbeux (commande lancée, env, initialize, tools/list, tools/call,
  stderr serveur, timings, erreurs détaillées).
- Wrappers simples et auto-suffisants pour chaque serveur MCP visé.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
import sys
from typing import Any, Dict, Optional, List

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
import logging

logger = logging.getLogger("mcp_tools")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [mcp_tools] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)  # DEBUG verbeux par défaut


# -----------------------------------------------------------------------------
# Client JSON-RPC 2.0 sur STDIO avec framing MCP (Content-Length)
# -----------------------------------------------------------------------------

class StdioJsonRpc:
    """
    Client JSON-RPC 2.0 sur stdio pour MCP.

    - Écrit les requêtes encodées en UTF-8 précédées d'entêtes:
        Content-Length: <bytes>\r\n
        \r\n
        {json}
    - Lit les réponses en bouclant sur stdout:
        * Si "Content-Length:" -> lit exactement N caractères (fallback léger: N ~ chars)
        * Sinon, tolère un mode "ligne JSON" (rare mais vu en dev).
    - Un thread lecteur parse et pousse les messages (dict) dans une queue.
      request(...) attend un message avec le même "id".
    - Capture asynchrone de stderr vers logs DEBUG.
    """

    def __init__(
        self,
        cmd: List[str],
        env: Optional[dict] = None,
        cwd: Optional[str] = None,
        read_timeout: float = 30.0,
    ):
        self.cmd = cmd
        self.env = {**os.environ, **(env or {})}
        self.cwd = cwd
        self.read_timeout = read_timeout

        logger.debug(f"[RPC] Lancement du serveur MCP: cmd={cmd} cwd={cwd} env_add={list((env or {}).keys())}")
        t0 = time.time()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
            text=True,           # on travaille en texte pour simplicité (UTF-8)
            encoding="utf-8",
            bufsize=1,           # line-buffered
        )
        logger.debug(f"[RPC] Process démarré pid={self.proc.pid} en {time.time()-t0:.3f}s")

        # Queue de messages JSON déjà parsés par un thread lecteur
        self._msg_queue: "queue.Queue[dict]" = queue.Queue()
        self._reader_stop = threading.Event()
        self._reader = threading.Thread(target=self._reader_loop, name=f"mcp_reader_{self.proc.pid}", daemon=True)
        self._reader.start()

        # Thread STDERR
        self._stderr_stop = threading.Event()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, name=f"mcp_stderr_{self.proc.pid}", daemon=True)
        self._stderr_reader.start()

        self._lock_write = threading.Lock()

    # --------------------------
    # Framing / IO primitives
    # --------------------------

    def _send_json(self, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False)
        # Framing MCP: Content-Length + CRLFCRLF + body
        # NB: contenu en caractères (pas strictement octets) — OK en pratique (ASCII JSON).
        header = f"Content-Length: {len(data.encode('utf-8'))}\r\n\r\n"
        with self._lock_write:
            assert self.proc.stdin is not None
            logger.debug(f"[RPC->] {data}")
            self.proc.stdin.write(header)
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def _read_one_message(self) -> Optional[dict]:
        """
        Lit un message JSON encadré par "Content-Length" + \r\n\r\n, sinon tolère
        un fallback "ligne JSON".
        Retourne un dict ou None si EOF.
        """
        assert self.proc.stdout is not None

        # Lire d'abord une ligne d'en-tête ou un JSON direct
        line = self.proc.stdout.readline()
        if line == "":
            # EOF
            return None

        # Mode framing MCP
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.strip().split(":")[1].strip())
            except Exception:
                logger.debug(f"[RPC<] Entête invalide: {line.strip()}")
                return None
            # Ligne blanche
            blank = self.proc.stdout.readline()
            if blank not in ("\n", "\r\n", ""):
                # Parfois certains serveurs envoient \r seulement, tolérons.
                pass
            # Lire exactement 'length' octets -> en mode texte, on approxime avec chars
            body = self.proc.stdout.read(length)
            if body is None:
                return None
            try:
                msg = json.loads(body)
                logger.debug(f"[RPC<] {body}")
                return msg
            except Exception as e:
                logger.debug(f"[RPC<] JSON invalide (framed): {e} body={body[:200]}...")
                return None

        # Fallback "ligne JSON"
        line_stripped = line.strip()
        if not line_stripped:
            return None
        try:
            msg = json.loads(line_stripped)
            logger.debug(f"[RPC<] (line) {line_stripped}")
            return msg
        except Exception:
            # Peut-être une notification multi-lignes; continuons à lire jusqu'à JSON
            # (comportement best-effort pour dev servers)
            if line_stripped:
                logger.debug(f"[RPC<] ligne non-JSON ignorée: {line_stripped[:200]}")
            return None

    def _reader_loop(self) -> None:
        try:
            while not self._reader_stop.is_set():
                msg = self._read_one_message()
                if msg is None:
                    # EOF ou ligne vide, on attend une micro-pause pour éviter spin
                    if self.proc.poll() is not None:
                        # process terminé
                        logger.debug("[RPC] Process terminé, arrêt du reader.")
                        break
                    time.sleep(0.01)
                    continue
                # On ignore les notifications (pas de "id") mais on log
                if isinstance(msg, dict):
                    if "id" in msg:
                        self._msg_queue.put(msg)
                    else:
                        logger.debug(f"[RPC] notification: {msg}")
        except Exception as e:
            logger.debug(f"[RPC] reader exception: {e}")

    def _stderr_loop(self) -> None:
        try:
            while not self._stderr_stop.is_set():
                assert self.proc.stderr is not None
                line = self.proc.stderr.readline()
                if line == "":
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.01)
                    continue
                logger.debug(f"[RPC-STDERR] {line.rstrip()}")
        except Exception as e:
            logger.debug(f"[RPC] stderr reader exception: {e}")

    # --------------------------
    # API publique
    # --------------------------

    def request(self, method: str, params: dict | None = None, _id: Optional[str] = None) -> Any:
        """
        Envoie une requête JSON-RPC "method"/"params" et attend la réponse avec le même "id".
        """
        if _id is None:
            _id = str(uuid.uuid4())
        payload = {"jsonrpc": "2.0", "id": _id, "method": method, "params": params or {}}
        self._send_json(payload)

        t0 = time.time()
        while True:
            try:
                msg = self._msg_queue.get(timeout=self.read_timeout)
            except queue.Empty:
                raise RuntimeError(f"Timeout en attente de la réponse RPC '{method}' (id={_id})")

            if not isinstance(msg, dict):
                continue
            if msg.get("id") != _id:
                # message d'une autre requête (peu probable en séquentiel)
                logger.debug(f"[RPC] reçu id inattendu {msg.get('id')} (attendu { _id }) -> ignoré")
                continue

            # Correspondance trouvée
            if "error" in msg:
                raise RuntimeError(f"RPC Error for {method}: {msg['error']}")
            logger.debug(f"[RPC] {method} ok en {time.time()-t0:.3f}s")
            return msg.get("result")

    def close(self):
        try:
            self._reader_stop.set()
            self._stderr_stop.set()
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Appels MCP génériques
# -----------------------------------------------------------------------------

def _initialize(session: StdioJsonRpc) -> dict:
    logger.debug("[MCP] initialize()")
    return session.request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {"roots": {"listChanged": False}},
        "clientInfo": {"name": "venom-agent", "version": "0.1.0"}
    })

def _list_tools(session: StdioJsonRpc) -> list[dict]:
    logger.debug("[MCP] tools/list")
    return session.request("tools/list", {})

def _call_tool(session: StdioJsonRpc, name: str, arguments: dict) -> dict:
    logger.debug(f"[MCP] tools/call name={name} args={arguments}")
    return session.request("tools/call", {"name": name, "arguments": arguments or {}})


# -----------------------------------------------------------------------------
# Wrappers spécifiques (fichier demandé)
# -----------------------------------------------------------------------------

def _npx_cmd(pkg: str, extra: Optional[List[str]] = None) -> List[str]:
    """
    Construit la commande npx standardisée (surchargée via MCP_NODE_BIN si besoin).
    """
    node_bin = os.environ.get("MCP_NODE_BIN", "npx")
    base = [node_bin, "-y", pkg, "--stdio"]
    return base + (extra or [])


def mcp_filesystem_list_allowed_directories() -> dict:
    """
    @modelcontextprotocol/server-filesystem
    - Tool: list_allowed_directories
    - Requiert env ALLOW (liste de dossiers séparés par ':')
    """
    allow = os.environ.get("MCP_FS_ALLOW", ":".join([
        "/Users/venom/Documents/analyse-financiere",
        "/Users/venom/Documents/Cline",
        "/Users/venom/Documents"
    ]))
    env = {"ALLOW": allow}
    cmd = _npx_cmd("@modelcontextprotocol/server-filesystem")

    logger.debug(f"[FS] ALLOW={allow}")
    ses = StdioJsonRpc(cmd, env=env)
    try:
        init_res = _initialize(ses)
        logger.debug(f"[FS] initialize -> {init_res}")
        tools = _list_tools(ses)
        logger.debug(f"[FS] tools/list -> {tools}")
        res = _call_tool(ses, "list_allowed_directories", {})
        logger.debug(f"[FS] list_allowed_directories -> {res}")
        return {"ok": True, "data": res}
    except Exception as e:
        logger.exception("[FS] Erreur")
        return {"ok": False, "error": str(e)}
    finally:
        ses.close()


def mcp_supermemory_whoami() -> dict:
    """
    api-supermemory-ai
    - Tool: whoAmI
    - Deux modes:
        1) Si SUPERMEMORY_AUTH présent => on passe un header Authorization: <token>
        2) Sinon, on tente sans header (si l'outil demande login, il échouera)

    Nécessite l'outil CLI 'mcp-remote' (installé via npm) **ou** le package du serveur si accès direct.
    Ici, on utilise 'mcp-remote <url> --stdio [--header "Authorization: ..."]'
    """
    headers = []
    if os.environ.get("SUPERMEMORY_AUTH"):
        headers = ["--header", f"Authorization: {os.environ['SUPERMEMORY_AUTH']}"]
        logger.debug("[Supermemory] Authorization header activé via SUPERMEMORY_AUTH")

    # Si vous préférez appeler le package installé par install-mcp, laissez tel quel:
    cmd = _npx_cmd("mcp-remote", extra=["https://api.supermemory.ai/mcp"] + headers)

    ses = StdioJsonRpc(cmd)
    try:
        init_res = _initialize(ses)
        logger.debug(f"[Supermemory] initialize -> {init_res}")
        # Optionnel: tools list
        tools = _list_tools(ses)
        logger.debug(f"[Supermemory] tools/list -> {tools}")
        res = _call_tool(ses, "whoAmI", {})
        logger.debug(f"[Supermemory] whoAmI -> {res}")
        return {"ok": True, "data": res}
    except Exception as e:
        logger.exception("[Supermemory] Erreur")
        return {"ok": False, "error": str(e)}
    finally:
        ses.close()


def mcp_everything_echo(message: str) -> dict:
    """
    @modelcontextprotocol/server-everything
    - Tool: echo
    """
    cmd = _npx_cmd("@modelcontextprotocol/server-everything")
    ses = StdioJsonRpc(cmd)
    try:
        init_res = _initialize(ses)
        logger.debug(f"[Everything] initialize -> {init_res}")
        tools = _list_tools(ses)
        logger.debug(f"[Everything] tools/list -> {tools}")
        res = _call_tool(ses, "echo", {"message": message})
        logger.debug(f"[Everything] echo -> {res}")
        return {"ok": True, "data": res}
    except Exception as e:
        logger.exception("[Everything] Erreur")
        return {"ok": False, "error": str(e)}
    finally:
        ses.close()


def mcp_seqthink(thought: str, total_thoughts: int = 1, next_thought_needed: bool = False) -> dict:
    """
    @modelcontextprotocol/server-sequential-thinking
    - Tool: sequentialthinking
    """
    cmd = _npx_cmd("@modelcontextprotocol/server-sequential-thinking")
    ses = StdioJsonRpc(cmd)
    try:
        init_res = _initialize(ses)
        logger.debug(f"[SeqThink] initialize -> {init_res}")
        tools = _list_tools(ses)
        logger.debug(f"[SeqThink] tools/list -> {tools}")
        args = {
            "thought": thought,
            "totalThoughts": int(total_thoughts),
            "nextThoughtNeeded": bool(next_thought_needed),
            "thoughtNumber": 1
        }
        res = _call_tool(ses, "sequentialthinking", args)
        logger.debug(f"[SeqThink] sequentialthinking -> {res}")
        return {"ok": True, "data": res}
    except Exception as e:
        logger.exception("[SeqThink] Erreur")
        return {"ok": False, "error": str(e)}
    finally:
        ses.close()


# -----------------------------------------------------------------------------
# CLI de test manuel (optionnel)
# -----------------------------------------------------------------------------

def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MCP stdio adapters (filesystem, supermemory, everything, sequential-thinking)")
    ap.add_argument("--test", choices=["fs", "sm", "echo", "seq", "all"], default="all")
    ap.add_argument("--message", default="hello from mcp_tools")
    ap.add_argument("--thought", default="Testing sequential thinking connectivity")
    ap.add_argument("--total", type=int, default=1)
    ap.add_argument("--next-needed", action="store_true")
    args = ap.parse_args()

    if args.test in ("fs", "all"):
        print("\n[TEST] filesystem.list_allowed_directories")
        _print_json(mcp_filesystem_list_allowed_directories())

    if args.test in ("sm", "all"):
        print("\n[TEST] api-supermemory-ai.whoAmI")
        _print_json(mcp_supermemory_whoami())

    if args.test in ("echo", "all"):
        print("\n[TEST] everything.echo")
        _print_json(mcp_everything_echo(args.message))

    if args.test in ("seq", "all"):
        print("\n[TEST] sequential-thinking.sequentialthinking")
        _print_json(mcp_seqthink(args.thought, total_thoughts=args.total, next_thought_needed=args.next_needed))