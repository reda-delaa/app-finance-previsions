from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash import dash_table
import dash


def _read_signals() -> pd.DataFrame | None:
    parts_p = sorted(Path('data/forecast').glob('dt=*/forecasts.parquet'))
    if not parts_p:
        return None
    df = pd.read_parquet(parts_p[-1])
    try:
        parts_f = sorted(Path('data/forecast').glob('dt=*/final.parquet'))
        if parts_f:
            ff = pd.read_parquet(parts_f[-1])
            if {'ticker','horizon','final_score'}.issubset(ff.columns):
                df = df.merge(ff[['ticker','horizon','final_score']], on=['ticker','horizon'], how='left')
    except Exception:
        pass
    return df


def _watchlist() -> list[str]:
    wl = os.getenv('WATCHLIST', '').strip()
    if not wl:
        fp = Path('data/watchlist.txt')
        if fp.exists():
            try:
                wl = fp.read_text(encoding='utf-8')
            except Exception:
                wl = ''
    items = [x.strip().upper() for x in wl.replace('\n', ',').split(',') if x.strip()]
    return list(dict.fromkeys(items))


def _signals_table(horizon: str | None = None) -> dbc.Card:
    try:
        df = _read_signals()
        if df is None:
            return dbc.Card(dbc.CardBody([html.Small("Aucune prévision disponible.")]))
        if horizon and 'horizon' in df.columns:
            df = df[df['horizon'] == horizon]
        cols = [c for c in ['ticker','horizon','final_score','direction','confidence','expected_return'] if c in df.columns]
        if not cols:
            return dbc.Card(dbc.CardBody([html.Small("Colonnes attendues manquantes.")]))
        df = df[cols].sort_values(['final_score','confidence'] if 'final_score' in cols else ['confidence'], ascending=False, na_position='last').head(200)

        wl = _watchlist()
        styles = []
        for t in wl:
            styles.append({'if': {'filter_query': '{ticker} = "%s"' % t}, 'backgroundColor': '#2b3035', 'color': '#f8f9fa'})

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
            style_data_conditional=styles,
        )
        return dbc.Card([dbc.CardHeader("Signals — Top (joint final/forecasts)"), dbc.CardBody(dt)])
    except Exception as e:
        return dbc.Card(dbc.CardBody([html.Small(f"Erreur Signals: {e}")]))


def layout():
    controls = dbc.Row([
        dbc.Col([
            html.Small("Horizon ", className="me-2"),
            dcc.Dropdown(id='signals-horizon', options=[{"label":"1w","value":"1w"},{"label":"1m","value":"1m"},{"label":"1y","value":"1y"}], value='1m', clearable=False, style={"minWidth":"160px"})
        ], md=3),
        dbc.Col(html.Small("Watchlist (env WATCHLIST) mise en évidence."), md=6)
    ], className="mb-2")
    return html.Div([
        html.H3("Signals"),
        controls,
        _signals_table('1m'),
    ])


@dash.callback(dash.Output('signals-table','data'), dash.Input('signals-horizon','value'))
def on_horizon(h):
    try:
        df = _read_signals()
        if df is None:
            return []
        if h and 'horizon' in df.columns:
            df = df[df['horizon'] == h]
        cols = [c for c in ['ticker','horizon','final_score','direction','confidence','expected_return'] if c in df.columns]
        if not cols:
            return []
        df = df[cols].sort_values(['final_score','confidence'] if 'final_score' in cols else ['confidence'], ascending=False, na_position='last').head(200)
        return df.reset_index(drop=True).to_dict('records')
    except Exception:
        return []
