#!/usr/bin/env bash
set -euo pipefail

# Simple security scanning harness (best-effort).
# Writes results under artifacts/security/

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
ART_DIR="$ROOT_DIR/artifacts/security"
mkdir -p "$ART_DIR"

log(){ echo "[security-scan] $*"; }

# Python: pip-audit and safety
run_python_scans(){
  log "Python scans (pip-audit, safety)"
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
  if "$PY" -m pip show pip-audit >/dev/null 2>&1 || "$PY" -m pip install -q pip-audit; then
    log "Running pip-audit..."
    set +e
    "$PY" -m pip_audit -r "$ROOT_DIR/requirements.txt" -f json > "$ART_DIR/pip-audit.json" 2>/dev/null
    set -e
  else
    log "pip-audit not available"
  fi
  if "$PY" -m pip show safety >/dev/null 2>&1 || "$PY" -m pip install -q safety; then
    log "Running safety..."
    set +e
    "$PY" -m safety check -r "$ROOT_DIR/requirements.txt" --json > "$ART_DIR/safety.json" 2>/dev/null
    set -e
  else
    log "safety not available"
  fi
  # Bandit (code security linter)
  if "$PY" -m pip show bandit >/dev/null 2>&1 || "$PY" -m pip install -q bandit; then
    log "Running bandit..."
    set +e
    bandit -r "$ROOT_DIR/src" "$ROOT_DIR/scripts" -f json -o "$ART_DIR/bandit.json" 2>/dev/null
    set -e
  else
    log "bandit not available"
  fi
}

# Node: npm audit (if package.json exists)
run_node_scans(){
  if [ -f "$ROOT_DIR/package.json" ]; then
    if command -v npm >/dev/null 2>&1; then
      log "Running npm audit (project package.json) ..."
      set +e
      npm audit --json > "$ART_DIR/npm-audit.json" 2>/dev/null
      set -e
    else
      log "npm not available; skip npm audit"
    fi
  else
    log "No package.json detected; skip npm audit"
  fi
}

# Semgrep (code scanning) — optional
run_semgrep(){
  if command -v semgrep >/dev/null 2>&1; then
    log "Running semgrep..."
    set +e
    semgrep --quiet --config p/ci --json --output "$ART_DIR/semgrep.json" "$ROOT_DIR" 2>/dev/null
    set -e
  else
    log "semgrep not installed (optional)"
  fi
}

# Trivy filesystem scan — optional
run_trivy(){
  if command -v trivy >/dev/null 2>&1; then
    log "Running trivy fs scan..."
    set +e
    trivy fs --quiet --security-checks vuln,config --format json -o "$ART_DIR/trivy-fs.json" "$ROOT_DIR" 2>/dev/null
    set -e
  else
    log "trivy not installed (optional)"
  fi
}

run_secret_scan(){
  if command -v python3 >/dev/null 2>&1; then
    log "Running lightweight secret scan..."
    set +e
    python3 "$ROOT_DIR/ops/security/secret_scan.py" > "$ART_DIR/secret_scan.json" 2>/dev/null || true
    set -e
  fi
}

run_python_scans || true
run_node_scans || true
run_semgrep || true
run_trivy || true
run_secret_scan || true

log "Done. Reports under $ART_DIR"
