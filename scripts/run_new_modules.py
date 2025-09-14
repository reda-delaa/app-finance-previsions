#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, subprocess, textwrap, importlib, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# Modules/fichiers récents à tester (ordre conseillé)
TARGETS = [
    # (friendly name, file path relative to REPO, module path)
    ("market_intel", "src/analytics/market_intel.py", "src.analytics.market_intel"),
    ("financials_ownership_client", "src/ingestion/financials_ownership_client.py", "src.ingestion.financials_ownership_client"),
    ("macro_derivatives_client", "src/ingestion/macro_derivatives_client.py", "src.ingestion.macro_derivatives_client"),
    ("finviz_client", "src/ingestion/finviz_client.py", "src.ingestion.finviz_client"),
    ("finviz_provider", "src/ingestion/finviz.py", "src.ingestion.finviz"),
]

def banner(title: str):
    print("\n" + "="*80)
    print(f"RUNNING: {title}")
    print("="*80)

def run_subprocess(cmd, env=None):
    res = subprocess.run(cmd, capture_output=True, text=True, env=env or os.environ.copy())
    print("CMD:", " ".join(cmd))
    print("\n--- STDOUT ---\n", res.stdout)
    if res.stderr:
        print("\n--- STDERR ---\n", res.stderr)
    return res.returncode

def try_run_as_file(name, rel_path):
    path = (REPO / rel_path).resolve()
    if not path.exists():
        print(f"[{name}] Fichier introuvable: {path}")
        return 127
    return run_subprocess([PY, str(path)])

def try_run_as_module(name, modpath):
    # On lance via `python -m`, en forçant PYTHONPATH=REPO
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    return run_subprocess([PY, "-m", modpath], env=env)

def try_import_smoke(name, modpath):
    print(f"[{name}] Tentative d’import simple: {modpath}")
    try:
        sys.path.insert(0, str(REPO))  # pour s'assurer que `src` est visible
        mod = importlib.import_module(modpath)
        attrs = [a for a in dir(mod) if not a.startswith("_")]
        print(f"[{name}] Import OK. Attributs publics (extrait): {attrs[:20]}")
        # Option: si un main/selftest existe sans args, on tente.
        for candidate in ("self_test", "smoke", "main", "run"):
            if hasattr(mod, candidate):
                fn = getattr(mod, candidate)
                if callable(fn):
                    print(f"[{name}] Appel de {candidate}() (sans arguments)…")
                    try:
                        out = fn()
                        print(f"[{name}] {candidate}() terminé. Retour: {out!r}")
                    except TypeError:
                        print(f"[{name}] {candidate}() semble nécessiter des arguments — on n’insiste pas.")
                    break
        return 0
    except Exception:
        print(f"[{name}] Import FAILED:\n{traceback.format_exc()}")
        return 1
    finally:
        if sys.path and sys.path[0] == str(REPO):
            sys.path.pop(0)

def main():
    print(f"Python: {PY}")
    print(f"Repo root: {REPO}")
    for name, rel, mod in TARGETS:
        banner(name)
        # 1) essai exécution directe du fichier
        code = try_run_as_file(name, rel)
        if code == 0:
            continue
        # 2) essai exécution en module
        code = try_run_as_module(name, mod)
        if code == 0:
            continue
        # 3) dernier recours: import smoke test
        _ = try_import_smoke(name, mod)

if __name__ == "__main__":
    main()