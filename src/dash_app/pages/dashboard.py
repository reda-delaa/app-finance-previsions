from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html


def _top_final() -> dbc.Card:
    parts = sorted(Path('data/forecast').glob('dt=*/final.parquet'))
    if not parts:
        return dbc.Card(dbc.CardBody([html.Small("Aucune donnée final.parquet trouvée." )]))
    df = pd.read_parquet(parts[-1])
    if df.empty:
        return dbc.Card(dbc.CardBody([html.Small("final.parquet vide.")]))
    top = df[df['horizon']=='1m'].sort_values('final_score', ascending=False).head(10)
    if top.empty:
        return dbc.Card(dbc.CardBody([html.Small("Pas de lignes horizon 1m.")]))
    table = dbc.Table.from_dataframe(top[['ticker','final_score']].reset_index(drop=True), striped=True, bordered=False, hover=True, size='sm')
    return dbc.Card([
        dbc.CardHeader("Top 10 (Final, 1m)"),
        dbc.CardBody(table),
    ])


def layout():
    return html.Div([
        html.H3("Dashboard — Top picks"),
        _top_final(),
    ])

