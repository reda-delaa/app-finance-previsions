# -*- coding: utf-8 -*-
"""
Adapteurs MCP stdio pour l'agent:
- filesystem.list_allowed_directories
- api-supermemory-ai.whoAmI
- everything.echo
- sequential-thinking.sequentialthinking
"""

from __future__ import annotations
import json, os, subprocess, tempfile, threading, uuid, sys
from typing import Any, Dict, Optional

# Petit client JSON-RPC 2.0 sur STDIO (suffisant pour appeler des tools MCP)
class StdioJsonRpc:
    def __init__(self, cmd: list[str], env: Optional[dict]=None, cwd: Optional[str]=None):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, **(env or {})},
            cwd=cwd,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()

    def request(self, method: str, params: dict | None = None, _id: Optional[str]=None) -> Any:
        if _id is None:
            _id = str(uuid.uuid4())
        payload = {"jsonrpc":"2.0","id":_id,"method":method,"params":params or {}}
        line = json.dumps(payload) + "\n"
        with self._lock:
            assert self.proc.stdin is not None
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
            # lecture synchronisée (simple mais suffisant ici)
            assert self.proc.stdout is not None
            for _ in range(10000):  # garde-fou
                resp_line = self.proc.stdout.readline()
                if not resp_line:
                    break
                try:
                    resp = json.loads(resp_line)
                except Exception:
                    continue
                if isinstance(resp, dict) and resp.get("id") == _id:
                    if "error" in resp:
                        raise RuntimeError(str(resp["error"]))
                    return resp.get("result")
        raise RuntimeError("Aucune réponse JSON-RPC reçue")

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass


def _initialize(session: StdioJsonRpc) -> dict:
    # Minimal initialize MCP
    return session.request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {"roots": {"listChanged": False}},
        "clientInfo": {"name": "venom-agent", "version": "0.1.0"}
    })

def _list_tools(session: StdioJsonRpc) -> list[dict]:
    return session.request("tools/list", {})

def _call_tool(session: StdioJsonRpc, name: str, arguments: dict) -> dict:
    return session.request("tools/call", {"name": name, "arguments": arguments})


# ---------- Wrappers spécifiques ----------

def mcp_filesystem_list_allowed_directories() -> dict:
    # Besoin d'ALLOW dans l'env (tu l’as déjà configuré côté VS Code; on le remet ici au cas où)
    allow = os.environ.get("MCP_FS_ALLOW", ":".join([
        "/Users/venom/Documents/analyse-financiere",
        "/Users/venom/Documents/Cline",
        "/Users/venom/Documents"
    ]))
    cmd = ["npx","-y","@modelcontextprotocol/server-filesystem","--stdio"]
    env = {"ALLOW": allow}
    ses = StdioJsonRpc(cmd, env=env)
    try:
        _initialize(ses)
        # Optionnel: vérifier la présence du tool
        _ = _list_tools(ses)
        res = _call_tool(ses, "list_allowed_directories", {})
        return {"ok": True, "data": res}
    finally:
        ses.close()


def mcp_supermemory_whoami() -> dict:
    """
    Nécessite un runner 'mcp-remote' (installé par le setup supermemory) OU token.
    Si tu as un token d’accès OAuth, place-le dans SUPERMEMORY_AUTH (Bearer ...).
    """
    headers = []
    if os.environ.get("SUPERMEMORY_AUTH"):
        headers = ["--header", f"Authorization: {os.environ['SUPERMEMORY_AUTH']}"]

    cmd = ["npx","-y","mcp-remote","https://api.supermemory.ai/mcp","--stdio", *headers]
    ses = StdioJsonRpc(cmd)
    try:
        _initialize(ses)
        # tool name côté Supermemory: "whoAmI"
        res = _call_tool(ses, "whoAmI", {})
        return {"ok": True, "data": res}
    finally:
        ses.close()


def mcp_everything_echo(message: str) -> dict:
    cmd = ["npx","-y","@modelcontextprotocol/server-everything","--stdio"]
    ses = StdioJsonRpc(cmd)
    try:
        _initialize(ses)
        res = _call_tool(ses, "echo", {"message": message})
        return {"ok": True, "data": res}
    finally:
        ses.close()


def mcp_seqthink(thought: str, total_thoughts: int = 1, next_thought_needed: bool = False) -> dict:
    cmd = ["npx","-y","@modelcontextprotocol/server-sequential-thinking","--stdio"]
    ses = StdioJsonRpc(cmd)
    try:
        _initialize(ses)
        args = {
            "thought": thought,
            "totalThoughts": int(total_thoughts),
            "nextThoughtNeeded": bool(next_thought_needed),
            "thoughtNumber": 1
        }
        res = _call_tool(ses, "sequentialthinking", args)
        return {"ok": True, "data": res}
    finally:
        ses.close()