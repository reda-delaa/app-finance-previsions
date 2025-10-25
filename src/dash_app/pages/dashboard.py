from __future__ import annotations

from pathlib import Path
import json
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
    # Optional alerts badge (from latest quality report)
    badge = None
    try:
        parts = sorted(Path('data/quality').glob('dt=*/report.json'))
        if parts:
            rep = json.loads(parts[-1].read_text(encoding='utf-8'))
            def _count(rep, sev):
                cnt = 0
                for sec in ['news','macro','prices','forecasts','features','events','freshness']:
                    s = rep.get(sec) or {}
                    for it in (s.get('issues') or []):
                        if str(it.get('sev','')).lower() == sev:
                            cnt += 1
                return cnt
            errs = _count(rep, 'error'); warns = _count(rep, 'warn')
            badge = dbc.Badge(f"Errors: {errs}  Warnings: {warns}", color=("danger" if errs else ("warning" if warns else "success")), className="ms-2")
    except Exception:
        pass

    header = html.Div([html.H3("Dashboard — Top picks"), badge] if badge else [html.H3("Dashboard — Top picks")])
    return html.Div([header, _top_final()])
