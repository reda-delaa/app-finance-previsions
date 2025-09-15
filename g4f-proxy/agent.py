#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent.py — Agent outillé avec mode "function calling" OpenAI-compatible.
- LLM (via proxy OpenAI-compatible) planifie et enchaîne les outils (auto tool_choice).
- Outils sandboxés: shell whitelist, lecture/écriture, listage, HTTP GET.
- Outils MCP via mcp_tools.py: filesystem, supermemory, everything, sequential-thinking.
- Contexte mémoire, journal JSONL, checkpoints, reprise.
- Fallback planner local si le LLM/proxy est indisponible.

ENV:
  AGENT_OPENAI_BASE   (ex: http://127.0.0.1:4000/v1)
  AGENT_MODEL         (ex: deepseek-v3, qwen3-coder-480b, deepseek-prover, llama-3.3-70b, phi-4-reasoning-plus)
  AGENT_SANDBOX_ROOT  (racine sandbox)
  AGENT_CHECKPOINT    (fichier checkpoint)
  AGENT_LOG           (fichier log)
  AGENT_HTTP_TIMEOUT  (sec, défaut 45)

DÉBOGAGE:
  - Logs très verbeux (logger "agent"). Cherche [DEBUG] dans la console.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Logging global très verbeux
# -----------------------------------------------------------------------------
logger = logging.getLogger("agent")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [agent] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# Dépendances optionnelles
# -----------------------------------------------------------------------------
try:
    import requests  # pour OpenAI-compatible proxy + http_get tool
except ImportError:  # pragma: no cover
    requests = None

# Wrappers MCP (fichier fourni séparément)
try:
    from mcp_tools import (
        mcp_filesystem_list_allowed_directories,
        mcp_supermemory_whoami,
        mcp_everything_echo,
        mcp_seqthink,
    )
    HAVE_MCP = True
    logger.debug("[INIT] mcp_tools importé avec succès (HAVE_MCP=True).")
except Exception as e:  # pragma: no cover
    HAVE_MCP = False
    logger.warning(f"[INIT] mcp_tools indisponible (HAVE_MCP=False): {e}")

# -----------------------------------------------------------------------------
# Sécurité et configuration
# -----------------------------------------------------------------------------

SANDBOX_ROOT = Path(os.environ.get("AGENT_SANDBOX_ROOT", ".")).resolve()
CHECKPOINT_PATH = Path(os.environ.get("AGENT_CHECKPOINT", ".agent_checkpoint.json")).resolve()
LOG_PATH = Path(os.environ.get("AGENT_LOG", ".agent_log.jsonl")).resolve()

OPENAI_BASE = os.environ.get("AGENT_OPENAI_BASE", "http://127.0.0.1:4000/v1").rstrip("/")
OPENAI_URL = f"{OPENAI_BASE}/chat/completions"
OPENAI_MODEL = os.environ.get("AGENT_MODEL", "deepseek-v3")
HTTP_TIMEOUT = int(os.environ.get("AGENT_HTTP_TIMEOUT", "45"))

# Whitelist de commandes autorisées (exécutées via /bin/sh -c)
ALLOWED_COMMANDS = [
    r"^echo\s+.*$",
    r"^cat\s+[\w\./\-]+$",
    r"^ls(\s+[-\w]+)?(\s+[\w\./\-]+)?$",
    r"^pwd$",
    r"^uname(\s+-[asnrvm]+)?$",
    r"^grep\s+.+\s+[\w\./\-]+$",
    r"^sed\s+.+\s+[\w\./\-]+$",
    r"^head\s+(-n\s+\d+\s+)?[\w\./\-]+$",
    r"^tail\s+(-n\s+\d+\s+)?[\w\./\-]+$",
    r"^wc\s+(-[lwc]+\s+)?[\w\./\-]+$",
    r"^python3?\s+[-\w\./]+(\s+.*)?$",
    r"^node\s+[-\w\./]+(\s+.*)?$",
    r"^npm\s+(run|install|ci|test)(\s+.*)?$",
    r"^pip3?\s+(install|list|show)(\s+.*)?$",
    r"^systemctl\s+(status|is-active|is-enabled)\s+[\w\-\.\@]+$",
]

# Limitation HTTP côté outils locaux (pour éviter exfiltration)
ALLOWED_HTTP_HOSTS = [
    "example.com",
    "api.github.com",
]

# -----------------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------------

def now() -> float:
    return time.time()

def append_log(entry: Dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover
        logger.debug(f"[LOG] append_log failure: {e}")

def is_path_in_sandbox(p: Path) -> bool:
    try:
        resolved = p.resolve()
        in_sb = SANDBOX_ROOT == resolved or SANDBOX_ROOT in resolved.parents
        logger.debug(f"[SANDBOX] check {resolved} in {SANDBOX_ROOT} -> {in_sb}")
        return in_sb
    except FileNotFoundError:
        try:
            parent = p.resolve().parent
            in_sb = SANDBOX_ROOT in parent.parents or SANDBOX_ROOT == parent
            logger.debug(f"[SANDBOX] check parent {parent} in {SANDBOX_ROOT} -> {in_sb}")
            return in_sb
        except Exception:
            return False

def is_command_allowed(cmd: str) -> bool:
    cmd = cmd.strip()
    allowed = any(re.match(pattern, cmd) for pattern in ALLOWED_COMMANDS)
    logger.debug(f"[SANDBOX] cmd allowed? {cmd} -> {allowed}")
    return allowed

def is_http_host_allowed(url: str) -> bool:
    m = re.match(r"^https?://([^/]+)/?.*$", url)
    if not m:
        return False
    host = m.group(1)
    allowed = any(host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_HTTP_HOSTS)
    logger.debug(f"[SANDBOX] http host allowed? {host} -> {allowed}")
    return allowed

# -----------------------------------------------------------------------------
# Contexte
# -----------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    objective: str
    memory: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # Historique pour l'agent LLM
    current_step: int = 0
    done: bool = False
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "memory": self.memory,
            "messages": self.messages,
            "current_step": self.current_step,
            "done": self.done,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ExecutionContext":
        return cls(
            objective=data["objective"],
            memory=data.get("memory", {}),
            messages=data.get("messages", []),
            current_step=data.get("current_step", 0),
            done=data.get("done", False),
            error=data.get("error"),
        )

def save_checkpoint(ctx: ExecutionContext) -> None:
    try:
        CHECKPOINT_PATH.write_text(json.dumps(ctx.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug(f"[CKPT] Sauvé {CHECKPOINT_PATH}")
    except Exception as e:  # pragma: no cover
        logger.debug(f"[CKPT] Échec sauvegarde: {e}")

def load_checkpoint() -> Optional[ExecutionContext]:
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        ctx = ExecutionContext.from_json(data)
        logger.debug(f"[CKPT] Chargé step={ctx.current_step} done={ctx.done} err={ctx.error}")
        return ctx
    except Exception as e:  # pragma: no cover
        logger.debug(f"[CKPT] Échec lecture: {e}")
        return None

# -----------------------------------------------------------------------------
# Outils sandbox
# -----------------------------------------------------------------------------

class Tooling:
    @staticmethod
    def read_file(path: str) -> Tuple[bool, str]:
        p = (SANDBOX_ROOT / path).resolve() if not path.startswith("/") else Path(path).resolve()
        logger.debug(f"[TOOL] read_file -> {p}")
        if not is_path_in_sandbox(p):
            return False, f"Accès refusé hors sandbox: {p}"
        try:
            content = p.read_text(encoding="utf-8")
            return True, content
        except Exception as e:
            return False, str(e)

    @staticmethod
    def write_file(path: str, content: str) -> Tuple[bool, str]:
        p = (SANDBOX_ROOT / path).resolve() if not path.startswith("/") else Path(path).resolve()
        logger.debug(f"[TOOL] write_file -> {p}")
        if not is_path_in_sandbox(p):
            return False, f"Accès refusé hors sandbox: {p}"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return True, f"Écrit: {p}"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def list_dir(path: str = ".") -> Tuple[bool, str]:
        p = (SANDBOX_ROOT / path).resolve() if not path.startswith("/") else Path(path).resolve()
        logger.debug(f"[TOOL] list_dir -> {p}")
        if not is_path_in_sandbox(p):
            return False, f"Accès refusé hors sandbox: {p}"
        try:
            if not p.exists():
                return False, f"N'existe pas: {p}"
            if not p.is_dir():
                return False, f"Pas un dossier: {p}"
            entries = [e.name + ("/" if e.is_dir() else "") for e in sorted(p.iterdir())]
            return True, "\n".join(entries)
        except Exception as e:
            return False, str(e)

    @staticmethod
    def run_cmd(cmd: str, timeout: int = 20) -> Tuple[bool, str]:
        cmd = cmd.strip()
        logger.debug(f"[TOOL] run_cmd -> '{cmd}' (timeout={timeout})")
        if not is_command_allowed(cmd):
            return False, f"Commande non autorisée (sandbox): {cmd}"
        try:
            proc = subprocess.run(
                cmd, shell=True, text=True, capture_output=True, timeout=timeout, cwd=str(SANDBOX_ROOT)
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            ok = proc.returncode == 0
            logger.debug(f"[TOOL] run_cmd rc={proc.returncode} bytes={len(out)}")
            return ok, out.strip()
        except subprocess.TimeoutExpired:
            return False, f"Timeout ({timeout}s): {cmd}"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def http_get(url: str, timeout: int = 15) -> Tuple[bool, str]:
        logger.debug(f"[TOOL] http_get -> {url} (timeout={timeout})")
        if requests is None:
            return False, "Le module 'requests' n'est pas installé"
        if not is_http_host_allowed(url):
            return False, f"Host HTTP non autorisé: {url}"
        try:
            r = requests.get(url, timeout=timeout)
            return True, f"HTTP {r.status_code}\n{r.text[:5000]}"
        except Exception as e:
            return False, str(e)

    # ---------- Wrappers MCP reliés aux tools ----------
    @staticmethod
    def mcp_fs_allowed() -> Tuple[bool, str]:
        logger.debug("[TOOL] mcp_fs_allowed()")
        if not HAVE_MCP:
            return False, "mcp_tools indisponible"
        res = mcp_filesystem_list_allowed_directories()
        return (True, json.dumps(res, ensure_ascii=False)) if res.get("ok") else (False, res.get("error", "Erreur"))

    @staticmethod
    def mcp_whoami() -> Tuple[bool, str]:
        logger.debug("[TOOL] mcp_whoami()")
        if not HAVE_MCP:
            return False, "mcp_tools indisponible"
        res = mcp_supermemory_whoami()
        return (True, json.dumps(res, ensure_ascii=False)) if res.get("ok") else (False, res.get("error", "Erreur"))

    @staticmethod
    def mcp_echo(message: str) -> Tuple[bool, str]:
        logger.debug(f"[TOOL] mcp_echo('{message}')")
        if not HAVE_MCP:
            return False, "mcp_tools indisponible"
        res = mcp_everything_echo(message)
        return (True, json.dumps(res, ensure_ascii=False)) if res.get("ok") else (False, res.get("error", "Erreur"))

    @staticmethod
    def mcp_seqthink(thought: str, total_thoughts: int = 1, next_thought_needed: bool = False) -> Tuple[bool, str]:
        logger.debug(f"[TOOL] mcp_seqthink(thought='{thought}', total={total_thoughts}, next_needed={next_thought_needed})")
        if not HAVE_MCP:
            return False, "mcp_tools indisponible"
        res = mcp_seqthink(thought, total_thoughts=total_thoughts, next_thought_needed=next_thought_needed)
        return (True, json.dumps(res, ensure_ascii=False)) if res.get("ok") else (False, res.get("error", "Erreur"))

# -----------------------------------------------------------------------------
# Schéma Tools (OpenAI) — locaux + MCP
# -----------------------------------------------------------------------------

OPENAI_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Lister le contenu d'un dossier sous la sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Chemin relatif ou absolu"},
                },
                "required": []
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lire un fichier texte sous la sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Écrire un fichier texte sous la sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Exécuter une commande shell autorisée (whitelist) dans la sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 120},
                },
                "required": ["cmd"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "Faire un HTTP GET (hosts autorisés uniquement)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 60},
                },
                "required": ["url"]
            },
        },
    },
]

# Ajout des tools MCP seulement si mcp_tools est dispo (évite les erreurs sur environnements incomplets)
if HAVE_MCP:
    OPENAI_TOOLS.extend([
        {
            "type": "function",
            "function": {
                "name": "mcp_fs_allowed",
                "description": "Appelle le MCP filesystem.list_allowed_directories et renvoie les dossiers autorisés.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_whoami",
                "description": "Appelle api-supermemory-ai.whoAmI pour récupérer l'identité connectée.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_echo",
                "description": "Appelle everything.echo pour renvoyer un message de test.",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_seqthink",
                "description": "Appelle sequential-thinking.sequentialthinking (outil de raisonnement séquentiel).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string"},
                        "total_thoughts": {"type": "integer", "minimum": 1, "default": 1},
                        "next_thought_needed": {"type": "boolean", "default": False},
                    },
                    "required": ["thought"],
                },
            },
        },
    ])

# -----------------------------------------------------------------------------
# LLM client (proxy OpenAI-compatible)
# -----------------------------------------------------------------------------

def call_openai(messages: List[Dict[str, Any]],
                tools: Optional[List[Dict[str, Any]]] = None,
                tool_choice: Optional[Any] = "auto",
                model: Optional[str] = None,
                temperature: float = 0.2,
                top_p: float = 1.0,
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """
    Appel non-streaming au /chat/completions (proxy local).
    Retour brut du JSON OpenAI.
    """
    if requests is None:
        raise RuntimeError("Le module 'requests' est requis pour le mode agent.")

    payload = {
        "model": model or OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    logger.debug(f"[OPENAI] POST {OPENAI_URL} model={payload['model']} tool_choice={tool_choice} msgs={len(messages)}")
    r = requests.post(OPENAI_URL, json=payload, timeout=HTTP_TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"OpenAI proxy HTTP {r.status_code}: {r.text[:1000]}")
    try:
        jr = r.json()
    except Exception:
        logger.debug(f"[OPENAI] Réponse non-JSON: {(r.text or '')[:400]}")
        raise
    logger.debug(f"[OPENAI] OK, keys={list(jr.keys())}")
    return jr

# -----------------------------------------------------------------------------
# Exécution d’un tool (côté agent)
# -----------------------------------------------------------------------------

def execute_tool_locally(name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
    logger.debug(f"[EXEC] tool={name} args={arguments}")
    try:
        if name == "list_dir":
            ok, msg = Tooling.list_dir(arguments.get("path", "."))
        elif name == "read_file":
            ok, msg = Tooling.read_file(arguments["path"])
        elif name == "write_file":
            ok, msg = Tooling.write_file(arguments["path"], arguments.get("content", ""))
        elif name == "run_cmd":
            ok, msg = Tooling.run_cmd(arguments["cmd"], timeout=int(arguments.get("timeout", 20)))
        elif name == "http_get":
            ok, msg = Tooling.http_get(arguments["url"], timeout=int(arguments.get("timeout", 15)))
        elif name == "mcp_fs_allowed" and HAVE_MCP:
            ok, msg = Tooling.mcp_fs_allowed()
        elif name == "mcp_whoami" and HAVE_MCP:
            ok, msg = Tooling.mcp_whoami()
        elif name == "mcp_echo" and HAVE_MCP:
            ok, msg = Tooling.mcp_echo(arguments.get("message", "hello from agent"))
        elif name == "mcp_seqthink" and HAVE_MCP:
            ok, msg = Tooling.mcp_seqthink(
                arguments["thought"],
                total_thoughts=int(arguments.get("total_thoughts", 1)),
                next_thought_needed=bool(arguments.get("next_thought_needed", False)),
            )
        else:
            return False, f"Tool inconnu ou indisponible: {name}"
        logger.debug(f"[EXEC] result ok={ok} size={len(msg) if isinstance(msg, str) else 'n/a'}")
        return ok, msg
    except KeyError as ke:
        return False, f"Argument manquant pour {name}: {ke}"
    except Exception as e:
        return False, f"Exception tool {name}: {e}"

# -----------------------------------------------------------------------------
# Prompt système
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """Tu es un agent outillé. Ton objectif est de réaliser la tâche en plusieurs étapes sûres.
- Utilise les TOOLS fournis via function calling quand c'est pertinent (tool_choice=auto).
- Respecte la sandbox: chemins sous la racine, commandes en whitelist, HTTP seulement sur hôtes autorisés.
- Tu disposes aussi d'outils MCP (filesystem, supermemory, everything, sequential-thinking) si disponibles.
- Après chaque outil, lis attentivement le résultat (role=tool) avant de décider la suite.
- Quand l'objectif est atteint, réponds avec un court résumé final et préfixe-le par: FINAL_ANSWER:
- Si une action risquée est demandée, propose une alternative safe.
"""

# -----------------------------------------------------------------------------
# Boucle agent LLM
# -----------------------------------------------------------------------------

def llm_agent_loop(ctx: ExecutionContext, max_tool_iters: int = 20) -> ExecutionContext:
    """
    Mode agent contrôlé par LLM via function-calling:
    - messages: [system, user], puis cycles assistant->tool->assistant...
    - s'arrête si: message assistant avec 'FINAL_ANSWER:' ou pas de tool_calls pendant 2 tours.
    """
    if not ctx.messages:
        ctx.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Objectif: {ctx.objective}\nMémoire: {json.dumps(ctx.memory, ensure_ascii=False)}"},
        ]
        logger.debug("[LOOP] initialisation messages system+user")

    no_tool_rounds = 0
    while not ctx.done and ctx.current_step < max_tool_iters:
        ctx.current_step += 1
        append_log({"ts": now(), "type": "agent_call", "step": ctx.current_step})
        logger.debug(f"[LOOP] step {ctx.current_step}")

        try:
            resp = call_openai(
                messages=ctx.messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                model=OPENAI_MODEL,
                temperature=0.2,
                top_p=1.0,
            )
        except Exception as e:
            ctx.error = f"Erreur LLM: {e}"
            append_log({"ts": now(), "type": "error", "error": ctx.error})
            logger.debug(f"[LOOP] erreur LLM: {e}")
            break

        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        logger.debug(f"[LOOP] tool_calls={len(tool_calls)}")

        # Si le modèle renvoie une réponse finale textuelle
        final_text = (message.get("content") or "").strip() if isinstance(message.get("content"), str) else ""
        if final_text:
            logger.debug(f"[LOOP] assistant text (len={len(final_text)})")
            if "FINAL_ANSWER:" in final_text:
                ctx.done = True
                ctx.error = None
                ctx.messages.append({"role": "assistant", "content": final_text})
                append_log({"ts": now(), "type": "final_text", "content": final_text})
                logger.debug("[LOOP] FINAL_ANSWER détecté -> arrêt")
                break
            # Pousser la pensée du modèle pour le tour suivant
            ctx.messages.append({"role": "assistant", "content": final_text})

        if not tool_calls:
            no_tool_rounds += 1
            logger.debug(f"[LOOP] no tool_calls round={no_tool_rounds}")
            if no_tool_rounds >= 2:
                # Pas d'outils deux fois de suite => fin
                ctx.done = True
                ctx.error = None
                append_log({"ts": now(), "type": "no_tool_termination"})
                logger.debug("[LOOP] arrêt par absence d'outils 2 tours de suite")
                break
            continue
        else:
            no_tool_rounds = 0

        # Exécuter chaque tool_call et ajouter la réponse role=tool
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            args_json = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_json)
            except Exception:
                args = {}
            logger.debug(f"[LOOP] executing tool '{name}' with args={args}")

            ok, result = execute_tool_locally(name, args)
            tool_content = json.dumps({
                "ok": ok,
                "result": result[:8000] if isinstance(result, str) else str(result)
            }, ensure_ascii=False)

            ctx.messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": tool_content
            })
            logger.debug(f"[LOOP] tool '{name}' -> ok={ok} size={len(tool_content)}")

        # Checkpoint périodique
        if ctx.current_step % 2 == 0:
            save_checkpoint(ctx)

    save_checkpoint(ctx)
    return ctx

# -----------------------------------------------------------------------------
# Fallback: planner simple local
# -----------------------------------------------------------------------------

def simple_planner_fallback(ctx: ExecutionContext, max_steps: int = 20) -> ExecutionContext:
    """
    Si le mode agent LLM est indisponible, on fait une séquence locale minimaliste.
    """
    steps = [
        {"action": "list_dir", "args": {"path": "."}, "label": "Lister dossier"},
        {"action": "read_file", "args": {"path": ctx.memory.get("file_path", "README.md")}, "label": "Lire fichier"},
    ]
    logger.debug(f"[FB] steps={steps}")
    for s in steps[:max_steps]:
        act = s["action"]; args = s.get("args", {})
        if act == "list_dir":
            ok, msg = Tooling.list_dir(args.get("path", "."))
        elif act == "read_file":
            ok, msg = Tooling.read_file(args["path"])
        else:
            ok, msg = False, f"Action inconnue: {act}"
        append_log({"ts": now(), "type": "fallback_step", "action": act, "ok": ok})
        logger.debug(f"[FB] {act} -> ok={ok} size={len(msg) if isinstance(msg, str) else 'n/a'}")
        if not ok:
            ctx.error = msg
            break
    ctx.done = ctx.error is None
    save_checkpoint(ctx)
    return ctx

# -----------------------------------------------------------------------------
# CLI / Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent outillé (function-calling) + sandbox + MCP")
    p.add_argument("--objective", "-o", type=str, help="Objectif naturel", required=False)
    p.add_argument("--resume", "-r", action="store_true", help="Reprendre depuis le checkpoint")
    p.add_argument("--set", "-s", action="append", default=[], help="Variable mémoire k=v (ex: --set file_path=README.md)")
    p.add_argument("--max-steps", type=int, default=40, help="Nombre maximum d’itérations tools")
    p.add_argument("--fallback", action="store_true", help="Forcer le mode fallback local (sans LLM)")
    return p.parse_args()

def parse_kv(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for pair in pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def main() -> None:
    args = parse_args()

    ctx: Optional[ExecutionContext] = None
    if args.resume:
        ctx = load_checkpoint()
        if ctx:
            print(f"[+] Reprise checkpoint: step={ctx.current_step}, done={ctx.done}, error={ctx.error}")
        else:
            print("[!] Aucun checkpoint, démarrage neuf.")

    if ctx is None:
        objective = args.objective or "Explorer le projet et proposer un plan bref."
        memory = parse_kv(args.set)
        ctx = ExecutionContext(objective=objective, memory=memory)

    print(f"Sandbox: {SANDBOX_ROOT}")
    print(f"Log: {LOG_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Model: {OPENAI_MODEL} via {OPENAI_URL}")
    print(f"Objectif: {ctx.objective}")
    if ctx.memory:
        print(f"Mémoire initiale: {ctx.memory}")

    logger.debug(f"[MAIN] HAVE_MCP={HAVE_MCP} OPENAI_BASE={OPENAI_BASE} MODEL={OPENAI_MODEL}")

    try:
        if args.fallback or requests is None:
            final_ctx = simple_planner_fallback(ctx)
        else:
            final_ctx = llm_agent_loop(ctx, max_tool_iters=args.max_steps)
    except KeyboardInterrupt:
        print("\n[!] Interrompu par l'utilisateur")
        sys.exit(130)

    status = "SUCCÈS" if final_ctx.done and not final_ctx.error else f"ERREUR: {final_ctx.error}"
    print(f"\n=== RÉSULTAT ===\n- done: {final_ctx.done}\n- step: {final_ctx.current_step}\n- status: {status}")
    print("Journal structuré:", LOG_PATH)
    print("Checkpoint:", CHECKPOINT_PATH)

if __name__ == "__main__":
    main()