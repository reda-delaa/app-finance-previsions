from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict

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
                ],
                vertical=True,
                pills=True,
                className="mb-3",
            ),
            html.Small("Administration", className="text-muted"),
            dbc.Nav(
                [
                    dbc.NavLink("Observability", href="/observability", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style={"padding": "0.75rem"},
    )


def _page_registry() -> Dict[str, Callable[[], html.Div]]:
    # Use absolute imports so running as script works with PYTHONPATH=src
    from dash_app.pages import dashboard, signals, portfolio, observability

    return {
        "/": dashboard.layout,
        "/dashboard": dashboard.layout,
        "/signals": signals.layout,
        "/portfolio": portfolio.layout,
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


if __name__ == "__main__":
    port = int(os.getenv("AF_DASH_PORT", "8050"))
    debug = os.getenv("AF_DASH_DEBUG", "false").lower() == "true"
    # Dash >=3 replaced run_server with run
    app.run(host="0.0.0.0", port=port, debug=debug)
