from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html


def _body() -> dbc.Card:
    try:
        parts = sorted(Path('data/macro/forecast').glob('dt=*/macro_forecast.parquet'))
        if not parts:
            return dbc.Card(dbc.CardBody([html.Small("Aucun macro_forecast.parquet trouvé.")]))
        df = pd.read_parquet(parts[-1])
        if df is None or df.empty:
            return dbc.Card(dbc.CardBody([html.Small("macro_forecast.parquet vide.")]))
        cols = [c for c in df.columns if c.lower() in {"regime","macro_regime","risk_regime","state"}]
        if not cols:
            # fallback: display head of key numeric columns if present
            cols = [c for c in df.columns if c.lower() in {"cpi_yoy","y10","y2","slope_10y_2y","recession_prob"}]
        if not cols:
            return dbc.Card(dbc.CardBody([html.Small("Colonnes de régime non trouvées.")]))
        out = df[cols].tail(10).reset_index(drop=True)
        table = dbc.Table.from_dataframe(out, striped=True, bordered=False, hover=True, size='sm')
        return dbc.Card([dbc.CardHeader("Regimes — Aperçu (derniers points)"), dbc.CardBody(table)])
    except Exception as e:
        return dbc.Card(dbc.CardBody([html.Small(f"Erreur Regimes: {e}")]))


def layout():
    return html.Div([
        html.H3("Regimes"),
        _body(),
    ])

