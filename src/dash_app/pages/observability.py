from __future__ import annotations

import os
import subprocess
from pathlib import Path
import json
import dash
import dash_bootstrap_components as dbc
import time
import requests
from dash import html, dcc, Output, Input, State, no_update


def _ui_health_card() -> dbc.Card:
    port = os.getenv("AF_DASH_PORT", "8050")
    return dbc.Card([
        dbc.CardHeader("UI — Santé (Dash)"),
        dbc.CardBody([
            html.Small(f"Dash port: {port}"), html.Br(),
            html.Div(id="dash-http-status", children=html.Small("HTTP: —")),
            dcc.Interval(id="dash-http-ping", interval=5000, n_intervals=0),
        ])
    ])


def _freshness_card() -> dbc.Card:
    parts = sorted(Path('data/quality').glob('dt=*/freshness.json'))
    body = []
    if parts:
        js = json.loads(parts[-1].read_text(encoding='utf-8'))
        checks = js.get('checks') or {}
        body.extend([
            html.Small(f"Forecasts aujourd'hui: {'Oui' if checks.get('forecasts_today') else 'Non'}"), html.Br(),
            html.Small(f"Final aujourd'hui: {'Oui' if checks.get('final_today') else 'Non'}"), html.Br(),
            html.Small(f"Macro aujourd'hui: {'Oui' if checks.get('macro_today') else 'Non'}"), html.Br(),
            html.Small(
                f"Couverture prix ≥5y: {int(checks.get('prices_5y_coverage_ratio')*100)}%"
                if isinstance(checks.get('prices_5y_coverage_ratio'), (int, float)) else "Couverture prix ≥5y: n/a"
            ),
        ])
    else:
        body.append(html.Small("Aucun freshness.json — exécutez `make update-monitor`."))
    return dbc.Card([dbc.CardHeader("Données — Fraîcheur"), dbc.CardBody(body)])


def _admin_controls_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("Actions (Admin) — UI Streamlit"),
        dbc.CardBody([
            html.Div([
                dbc.Button("Start (bg)", id="btn-start", color="success", size="sm", className="me-2"),
                dbc.Button("Restart (bg)", id="btn-restart", color="warning", size="sm", className="me-2"),
                dbc.Button("Stop", id="btn-stop", color="danger", size="sm"),
            ], className="mb-2"),
            dcc.Textarea(id="script-output", style={"width": "100%", "height": "120px"}),
        ])
    ])


def _logs_card() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader("Log en direct (Streamlit)"),
        dbc.CardBody([
            dcc.Interval(id="log-interval", interval=4000, n_intervals=0),
            html.Pre(id="log-view", style={"maxHeight": "300px", "overflowY": "auto"}),
        ])
    ])


def layout():
    return html.Div([
        html.H3("Observability (Dash)"),
        dbc.Row([
            dbc.Col(_ui_health_card(), md=6),
            dbc.Col(_freshness_card(), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(_admin_controls_card(), md=6),
            dbc.Col(_logs_card(), md=6),
        ]),
    ])


def _run_script(rel_path: str) -> str:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        script = str(repo_root / rel_path)
        out = subprocess.run(["bash", script], capture_output=True, text=True, timeout=45)
        stdout = out.stdout.strip() if out.stdout else ""
        stderr = out.stderr.strip() if out.stderr else ""
        return (stdout + ("\nSTDERR:\n" + stderr if stderr else "")).strip()
    except Exception as e:
        return f"Erreur: {e}"


@dash.callback(Output("script-output", "value"), Input("btn-start", "n_clicks"), prevent_initial_call=True)
def on_start(n):
    return _run_script("scripts/ui_start_bg.sh")


@dash.callback(Output("script-output", "value"), Input("btn-restart", "n_clicks"), prevent_initial_call=True)
def on_restart(n):
    return _run_script("scripts/ui_restart_bg.sh")


@dash.callback(Output("script-output", "value"), Input("btn-stop", "n_clicks"), prevent_initial_call=True)
def on_stop(n):
    return _run_script("scripts/ui_stop.sh")


@dash.callback(Output("log-view", "children"), Input("log-interval", "n_intervals"))
def refresh_log(_):
    try:
        repo_root = Path(__file__).resolve().parents[3]
        port = os.getenv("AF_UI_PORT", "5555")
        logfile = repo_root / 'logs' / 'ui' / f'streamlit_{port}.log'
        if logfile.exists():
            lines = logfile.read_text(encoding='utf-8', errors='ignore').splitlines()[-200:]
            return "\n".join(lines)
        return "(log introuvable)"
    except Exception as e:
        return f"(erreur lecture log: {e})"


@dash.callback(Output("dash-http-status", "children"), Input("dash-http-ping", "n_intervals"))
def ping_dash(_):
    try:
        t0 = time.perf_counter()
        port = int(os.getenv("AF_DASH_PORT", "8050"))
        r = requests.get(f"http://127.0.0.1:{port}", timeout=1.0)
        ms = int((time.perf_counter()-t0)*1000)
        return html.Small(f"HTTP: {r.status_code} ({ms} ms)")
    except Exception as e:
        return html.Small(f"HTTP: KO ({e})")
