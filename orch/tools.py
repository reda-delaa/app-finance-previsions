import subprocess, json, os, requests, textwrap

def shell(args):
    cmd = args.get("cmd")
    if not cmd: return {"ok": False, "error": "missing cmd"}
    cwd = args.get("cwd") or os.getcwd()
    try:
        # pylint: disable=subprocess-run-check
        # Using shell=True intentionally to support complex shell commands with piping/redirection
        out = subprocess.run(cmd, shell=True, cwd=cwd,  # nosec
                             capture_output=True, text=True, timeout=30)
        return {"ok": True, "code": out.returncode,
                "stdout": out.stdout[-4000:], "stderr": out.stderr[-2000:]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def read_file(args):
    p = args.get("path")
    if not p: return {"ok": False, "error": "missing path"}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = f.read()
        return {"ok": True, "content": data[:8000], "truncated": len(data) > 8000}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def write_file(args):
    p, c = args.get("path"), args.get("content")
    if not p or c is None: return {"ok": False, "error": "missing path/content"}
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(c)
        return {"ok": True, "msg": f"wrote {len(c)} bytes to {p}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def http_get(args):
    url = args.get("url")
    if not url: return {"ok": False, "error": "missing url"}
    try:
        r = requests.get(url, timeout=20)
        snippet = r.text[:8000]
        return {"ok": True, "status": r.status_code, "text": snippet, "truncated": len(r.text) > 8000}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def touch_file(args):
    # Handle touch command by creating empty file
    p = args.get("path")
    if not p: return {"ok": False, "error": "missing path"}
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    try:
        with open(p, "w", encoding="utf-8") as f:
            pass  # touch creates empty file
        return {"ok": True, "msg": f"touched {p}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def file_operation_dispatcher(args):
    # Handle file_operation calls from LLM
    command = args.get("command", "")
    if command.startswith("touch "):
        # Extract path from "touch filename.txt"
        path = command.replace("touch ", "").strip()
        return touch_file({"path": path})
    elif command.startswith("ls") or command.startswith("list"):
        # Map to shell command
        return shell({"cmd": "ls"})
    else:
        return {"ok": False, "error": f"unsupported file operation: {command}"}

TOOLS = {
    "shell": shell,
    "read_file": read_file,
    "write_file": write_file,
    "http_get": http_get,
    "touch_file": touch_file,
    "touch demo.txt": touch_file,  # Handle the specific command pattern
    "file_operation_dispatcher": file_operation_dispatcher
}
