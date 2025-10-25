from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html


def _signals_table() -> dbc.Card:
    parts_f = sorted(Path('data/forecast').glob('dt=*/final.parquet'))
    parts_p = sorted(Path('data/forecast').glob('dt=*/forecasts.parquet'))
    if not parts_p:
        return dbc.Card(dbc.CardBody([html.Small("Aucune prévision disponible.")]))
    df = pd.read_parquet(parts_p[-1])
    if parts_f:
        ff = pd.read_parquet(parts_f[-1])
        df = df.merge(ff[['ticker','horizon','final_score']], on=['ticker','horizon'], how='left')
    cols = [c for c in ['ticker','horizon','final_score','direction','confidence','expected_return'] if c in df.columns]
    df = df[cols].sort_values(['final_score','confidence'], ascending=False, na_position='last').head(50)
    table = dbc.Table.from_dataframe(df.reset_index(drop=True), striped=True, bordered=False, hover=True, size='sm')
    return dbc.Card([
        dbc.CardHeader("Signals — Top (joint final/forecasts)"),
        dbc.CardBody(table),
    ])


def layout():
    return html.Div([
        html.H3("Signals"),
        _signals_table(),
    ])

