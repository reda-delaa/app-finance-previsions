from __future__ import annotations

import os
from pathlib import Path
import json
import dash_bootstrap_components as dbc
from dash import html


def _ui_health() -> dbc.Card:
    port = os.getenv("AF_DASH_PORT", "8050")
    rows = [
        html.Small(f"Dash port: {port}"),
    ]
    return dbc.Card([
        dbc.CardHeader("UI — Santé (Dash)"),
        dbc.CardBody(rows),
    ])


def _freshness() -> dbc.Card:
    parts = sorted(Path('data/quality').glob('dt=*/freshness.json'))
    if not parts:
        return dbc.Card(dbc.CardBody([html.Small("Aucun freshness.json — exécutez `make update-monitor`.")]))
    js = json.loads(parts[-1].read_text(encoding='utf-8'))
    checks = js.get('checks') or {}
    items = [
        html.Small(f"Forecasts aujourd'hui: {'Oui' if checks.get('forecasts_today') else 'Non'}"),
        html.Br(),
        html.Small(f"Final aujourd'hui: {'Oui' if checks.get('final_today') else 'Non'}"),
        html.Br(),
        html.Small(f"Macro aujourd'hui: {'Oui' if checks.get('macro_today') else 'Non'}"),
        html.Br(),
        html.Small(f"Couverture prix ≥5y: {int(checks.get('prices_5y_coverage_ratio')*100)}%" if isinstance(checks.get('prices_5y_coverage_ratio'), (int,float)) else "Couverture prix ≥5y: n/a"),
    ]
    return dbc.Card([
        dbc.CardHeader("Données — Fraîcheur")
        , dbc.CardBody(items)
    ])


def layout():
    return html.Div([
        html.H3("Observability (Dash)"),
        dbc.Row([
            dbc.Col(_ui_health(), md=6),
            dbc.Col(_freshness(), md=6),
        ])
    ])

