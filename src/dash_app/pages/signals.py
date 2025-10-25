from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table


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
    df = df[cols].sort_values(['final_score','confidence'], ascending=False, na_position='last').head(200)
    dt = dash_table.DataTable(
        id='signals-table',
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.reset_index(drop=True).to_dict('records'),
        sort_action='native',
        filter_action='native',
        page_size=20,
        export_format='csv',
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 13},
    )
    return dbc.Card([dbc.CardHeader("Signals — Top (joint final/forecasts)"), dbc.CardBody(dt)])


def layout():
    return html.Div([
        html.H3("Signals"),
        _signals_table(),
    ])
