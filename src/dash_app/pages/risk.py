from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html


def _risk_card() -> dbc.Card:
    try:
        parts = sorted(Path('data/macro/forecast').glob('dt=*/macro_forecast.parquet'))
        if not parts:
            return dbc.Card(dbc.CardBody([html.Small("Aucun macro_forecast.parquet trouvé.")]))
        df = pd.read_parquet(parts[-1])
        if df is None or df.empty:
            return dbc.Card(dbc.CardBody([html.Small("macro_forecast.parquet vide.")]))
        cols = [c for c in df.columns if c.lower() in {"vix","credit_spread","drawdown_prob","risk_index"}]
        if not cols:
            return dbc.Card(dbc.CardBody([html.Small("Aucun indicateur de risque détecté.")]))
        out = df[cols].tail(10).reset_index(drop=True)
        table = dbc.Table.from_dataframe(out, striped=True, bordered=False, hover=True, size='sm')
        return dbc.Card([dbc.CardHeader("Risk — Indicateurs (derniers points)"), dbc.CardBody(table)])
    except Exception as e:
        return dbc.Card(dbc.CardBody([html.Small(f"Erreur Risk: {e}")]))


def layout():
    return html.Div([
        html.H3("Risk"),
        _risk_card(),
    ])

