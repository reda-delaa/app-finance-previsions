from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html


def _recession_card() -> dbc.Card:
    try:
        parts = sorted(Path('data/macro/forecast').glob('dt=*/macro_forecast.parquet'))
        if not parts:
            return dbc.Card(dbc.CardBody([html.Small("Aucun macro_forecast.parquet trouvé.")]))
        df = pd.read_parquet(parts[-1])
        if df is None or df.empty:
            return dbc.Card(dbc.CardBody([html.Small("macro_forecast.parquet vide.")]))
        col = None
        for c in ["recession_prob","recession_probability","recession_12m"]:
            if c in df.columns:
                col = c
                break
        if not col:
            return dbc.Card(dbc.CardBody([html.Small("Probabilité de récession non disponible.")]))
        out = df[[col]].tail(12).reset_index(drop=True)
        table = dbc.Table.from_dataframe(out, striped=True, bordered=False, hover=True, size='sm')
        return dbc.Card([dbc.CardHeader("Recession — Probabilité (derniers 12 points)"), dbc.CardBody(table)])
    except Exception as e:
        return dbc.Card(dbc.CardBody([html.Small(f"Erreur Recession: {e}")]))


def layout():
    return html.Div([
        html.H3("Recession"),
        _recession_card(),
    ])

