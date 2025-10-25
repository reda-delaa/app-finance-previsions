from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import subprocess
import shlex


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _snapshot_lsof(rx: Optional[re.Pattern]) -> list[dict]:
    out: list[dict] = []
    try:
        cmd = "lsof -nP -iTCP -sTCP:ESTABLISHED"
        res = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=8)
        if res.returncode != 0 and not res.stdout:
            return out
        lines = res.stdout.splitlines()
        for ln in lines:
            if ln.startswith("COMMAND"):
                continue
            parts = ln.split()
            if len(parts) < 9:
                continue
            try:
                name = parts[0]
                pid = int(parts[1]) if parts[1].isdigit() else 0
                r_name = " ".join(parts[8:])
                if '->' not in r_name:
                    continue
                laddr, raddr = r_name.split('->', 1)
                raddr = raddr.split(' ', 1)[0]
                cmdline = None
                if pid:
                    try:
                        proc = psutil.Process(pid)
                        cmdline = ' '.join(proc.cmdline()[:6])
                    except Exception:
                        pass
                if rx and not ((name and rx.search(name)) or (cmdline and rx.search(cmdline))):
                    continue
                out.append({
                    'pid': pid,
                    'name': name,
                    'cmd': cmdline,
                    'laddr': laddr,
                    'raddr': raddr,
                    'status': 'ESTABLISHED',
                })
            except Exception:
                continue
    except Exception:
        return out
    return out


def observe(interval: float, outdir: Path, only_procs: Optional[str], samples: int) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"net_activity_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    rx = re.compile(only_procs) if only_procs else None
    seen = set()
    i = 0
    with p.open("a", encoding="utf-8") as f:
        while True:
            i += 1
            wrote = 0
            try:
                conns = []
                try:
                    conns = psutil.net_connections(kind="inet")
                except Exception:
                    for row in _snapshot_lsof(rx):
                        key = (row['pid'], row['laddr'], row['raddr'])
                        if key in seen:
                            continue
                        seen.add(key)
                        row['ts'] = _ts()
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        wrote += 1
                else:
                    for c in conns:
                        try:
                            if c.status != psutil.CONN_ESTABLISHED:
                                continue
                            pid = c.pid or 0
                            name = None
                            cmd = None
                            if pid:
                                try:
                                    proc = psutil.Process(pid)
                                    name = proc.name()
                                    cmd = " ".join(proc.cmdline()[:6])
                                except Exception:
                                    pass
                            if rx and not ((name and rx.search(name)) or (cmd and rx.search(cmd))):
                                continue
                            laddr = f"{getattr(c.laddr, 'ip', '')}:{getattr(c.laddr, 'port', '')}" if c.laddr else None
                            raddr = f"{getattr(c.raddr, 'ip', '')}:{getattr(c.raddr, 'port', '')}" if c.raddr else None
                            key = (pid, laddr, raddr)
                            if key in seen:
                                continue
                            seen.add(key)
                            row = {
                                "ts": _ts(),
                                "pid": pid,
                                "name": name,
                                "cmd": cmd,
                                "laddr": laddr,
                                "raddr": raddr,
                                "status": "ESTABLISHED",
                            }
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            wrote += 1
                        except Exception:
                            continue
            finally:
                if wrote:
                    f.flush()
            if samples > 0 and i >= samples:
                break
            time.sleep(max(0.2, float(interval)))
    return p


def main(argv: list[str] | None = None) -> int:
    pa = argparse.ArgumentParser(description="Observe active TCP connections and log per-process",
                                 epilog="Safe, read-only. No blocking; outputs JSONL under artifacts/net/ by default.")
    pa.add_argument("--interval", type=float, default=float(os.getenv("NET_INTERVAL", 5)), help="Seconds between snapshots")
    pa.add_argument("--outdir", type=str, default=os.getenv("NET_OUTDIR", "artifacts/net"), help="Output directory")
    pa.add_argument("--only-procs", type=str, default=os.getenv("NET_ONLY_PROCS", ""), help="Regex to filter process name/cmdline")
    pa.add_argument("--samples", type=int, default=int(os.getenv("NET_SAMPLES", 0)), help="Stop after N samples (0 = run until Ctrl+C)")
    args = pa.parse_args(argv)
    p = observe(args.interval, Path(args.outdir), args.only_procs or None, args.samples)
    print(json.dumps({"ok": True, "path": str(p)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

