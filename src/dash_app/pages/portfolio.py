from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html


def _proposal() -> dbc.Card:
    try:
        parts = sorted(Path('data/forecast').glob('dt=*/final.parquet'))
        if not parts:
            return dbc.Card(dbc.CardBody([html.Small("Aucun final.parquet disponible.")]))
        df = pd.read_parquet(parts[-1])
        if df.empty:
            return dbc.Card(dbc.CardBody([html.Small("final.parquet vide.")]))
        cols_needed = {'ticker','horizon','final_score'}
        if not cols_needed.issubset(df.columns):
            return dbc.Card(dbc.CardBody([html.Small("Colonnes manquantes pour proposer un portefeuille.")]))
        sel = df[df['horizon']=='1m'].sort_values('final_score', ascending=False).head(5)
        if sel.empty:
            return dbc.Card(dbc.CardBody([html.Small("Final (1m) vide.")]))
        w = [round(1.0/len(sel), 4)]*len(sel)
        out = pd.DataFrame({'ticker': sel['ticker'], 'proposed_weight': w, 'final_score': sel['final_score']})
        table = dbc.Table.from_dataframe(out.reset_index(drop=True), striped=True, bordered=False, hover=True, size='sm')
        return dbc.Card([
            dbc.CardHeader("Portfolio — Proposition (Top‑5, égal‑pondéré)"),
            dbc.CardBody(table),
        ])
    except Exception as e:
        return dbc.Card(dbc.CardBody([html.Small(f"Erreur Portfolio: {e}")]))


def layout():
    return html.Div([
        html.H3("Portfolio"),
        _proposal(),
    ])
