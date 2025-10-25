from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash


def _latest_final() -> pd.DataFrame | None:
    parts = sorted(Path('data/forecast').glob('dt=*/final.parquet'))
    if not parts:
        return None
    df = pd.read_parquet(parts[-1])
    if df is None or df.empty:
        return None
    return df


def _compute_proposal(n: int, mode: str) -> dbc.Card:
    try:
        df = _latest_final()
        if df is None:
            return dbc.Card(dbc.CardBody([html.Small("Aucun final.parquet disponible.")]))
        if df.empty:
            return dbc.Card(dbc.CardBody([html.Small("final.parquet vide.")]))
        cols_needed = {'ticker','horizon','final_score'}
        if not cols_needed.issubset(df.columns):
            return dbc.Card(dbc.CardBody([html.Small("Colonnes manquantes pour proposer un portefeuille.")]))
        sel = df[df['horizon']=='1m'].sort_values('final_score', ascending=False).head(max(1, int(n)))
        if sel.empty:
            return dbc.Card(dbc.CardBody([html.Small("Final (1m) vide.")]))
        if mode == 'proportional':
            vals = sel['final_score'].clip(lower=0)
            s = float(vals.sum()) if hasattr(vals, 'sum') else 0.0
            if s > 0:
                weights = (vals / s).round(4)
            else:
                weights = pd.Series([round(1.0/len(sel),4)]*len(sel), index=sel.index)
        else:
            weights = pd.Series([round(1.0/len(sel),4)]*len(sel), index=sel.index)
        out = pd.DataFrame({'ticker': sel['ticker'].values, 'proposed_weight': weights.values, 'final_score': sel['final_score'].values})
        table = dbc.Table.from_dataframe(out.reset_index(drop=True), striped=True, bordered=False, hover=True, size='sm')
        return dbc.Card([
            dbc.CardHeader(f"Portfolio — Proposition (Top‑{len(sel)}, {'égal' if mode!='proportional' else 'proportionnel'})"),
            dbc.CardBody(table),
        ])
    except Exception as e:
        return dbc.Card(dbc.CardBody([html.Small(f"Erreur Portfolio: {e}")]))


def layout():
    controls = dbc.Row([
        dbc.Col([
            html.Small("Top‑N ", className="me-2"),
            dcc.Slider(id='port-topn', min=1, max=25, step=1, value=5, marks={1:'1',5:'5',10:'10',15:'15',20:'20',25:'25'})
        ], md=6),
        dbc.Col([
            html.Small("Pondération ", className="me-2"),
            dcc.RadioItems(id='port-weight-mode', options=[{"label":"Égalitaire","value":"equal"},{"label":"Proportionnelle (score)","value":"proportional"}], value='equal', inline=True)
        ], md=6)
    ], className="mb-3")

    return html.Div([
        html.H3("Portfolio"),
        controls,
        html.Div(id='port-proposal', children=_compute_proposal(5, 'equal')),
    ])


@dash.callback(dash.Output('port-proposal','children'), dash.Input('port-topn','value'), dash.Input('port-weight-mode','value'))
def on_port_update(n, mode):
    try:
        n = int(n) if n is not None else 5
        mode = mode or 'equal'
        return _compute_proposal(n, mode)
    except Exception as e:
        return dbc.Card(dbc.CardBody([html.Small(f"Erreur: {e}")]))
