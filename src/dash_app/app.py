from __future__ import annotations

import os
import json
import requests
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc


# Dash app (theme: dark Bootstrap)
external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    title="Finance Agent — Dash",
)
server = app.server


def sidebar() -> html.Div:
    return html.Div(
        [
            html.H4("Finance Agent", className="mt-3 mb-2"),
            html.Small("Analyse & Prévisions", className="text-muted"),
            dbc.Nav(
                [
                    dbc.NavLink("Dashboard", href="/dashboard", active="exact"),
                    dbc.NavLink("Signals", href="/signals", active="exact"),
                    dbc.NavLink("Portfolio", href="/portfolio", active="exact"),
                    dbc.NavLink("News", href="/news", active="exact"),
                    dbc.NavLink("Regimes", href="/regimes", active="exact"),
                    dbc.NavLink("Risk", href="/risk", active="exact"),
                    dbc.NavLink("Recession", href="/recession", active="exact"),
                ],
                vertical=True,
                pills=True,
                className="mb-3",
            ),
            html.Small("Administration", className="text-muted"),
            dbc.Nav(
                [
                    dbc.NavLink("Agents Status", href="/agents", active="exact"),
                    dbc.NavLink("Observability", href="/observability", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
            html.Div(id='global-status-badge', className="mt-3"),
            html.Small([dbc.NavLink("Détails", href="/agents", className="text-muted", style={"fontSize": "0.8rem"})], className="mb-2"),
        ],
        style={"padding": "0.75rem"},
    )


def _page_registry() -> Dict[str, Callable[[], html.Div]]:
    # Use absolute imports so running as script works with PYTHONPATH=src
    from dash_app.pages import dashboard, signals, portfolio, observability, agents_status, regimes, risk, recession, news

    return {
        "/": dashboard.layout,
        "/dashboard": dashboard.layout,
        "/signals": signals.layout,
        "/portfolio": portfolio.layout,
        "/regimes": regimes.layout,
        "/risk": risk.layout,
        "/recession": recession.layout,
        "/news": news.layout,
        "/agents": agents_status.layout,
        "/observability": observability.layout,
    }


app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        dbc.Row(
            [
                dbc.Col(sidebar(), width=2),
                dbc.Col(html.Div(id="page-content", style={"padding": "0.75rem"}), width=10),
            ],
            className="g-0",
        ),
        dcc.Interval(id='status-interval', interval=30*1000, n_intervals=0),  # refresh every 30s
    ],
    fluid=True,
)


@app.callback(dash.Output("page-content", "children"), dash.Input("url", "pathname"))
def render_page(pathname: str):
    try:
        pages = _page_registry()
        fn = pages.get(pathname, pages.get("/"))
        if not fn:
            return html.Div([html.H4("Page introuvable"), html.Small(pathname or "/")])
        try:
            return fn()
        except Exception as e:
            return html.Div([
                html.H4("Erreur lors du rendu de la page"),
                html.Small(str(e)),
            ])
    except Exception as e:
        return html.Div([
            html.H4("Erreur de navigation"),
            html.Small(str(e)),
        ])


@app.callback(dash.Output('global-status-badge', 'children'), dash.Input('status-interval', 'n_intervals'))
def update_global_status(n):
    try:
        # Check HTTP health
        port = int(os.getenv("AF_DASH_PORT", "8050"))
        url = f"http://127.0.0.1:{port}/"
        try:
            resp = requests.get(url, timeout=2)
            health_ok = resp.status_code == 200
        except:
            health_ok = False

        # Check freshness
        freshness_ok = True
        try:
            paths = sorted(Path('data/quality').glob('dt=*/freshness.json'))
            if paths:
                fresh = json.loads(paths[-1].read_text())
                now = pd.Timestamp.now()
                latest_dt = pd.to_datetime(fresh.get('latest_dt', '2000-01-01'))
                hours_diff = (now - latest_dt).total_seconds() / 3600
                freshness_ok = hours_diff < 25  # data du jour
        except:
            freshness_ok = False

        if health_ok and freshness_ok:
            return dbc.Badge("✓ OK", color="success")
        elif health_ok:
            return dbc.Badge("⚠ Données", color="warning")
        else:
            return dbc.Badge("✗ Box", color="danger")
    except:
        return dbc.Badge("? Err", color="secondary")


if __name__ == "__main__":
    port = int(os.getenv("AF_DASH_PORT", "8050"))
    debug = os.getenv("AF_DASH_DEBUG", "false").lower() == "true"
    # Dash >=3 replaced run_server with run
    app.run(host="0.0.0.0", port=port, debug=debug)
