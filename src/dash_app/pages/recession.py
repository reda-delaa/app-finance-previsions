from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc


def _recession_chart(df: pd.DataFrame) -> dcc.Graph:
    if df is None or df.empty:
        return dcc.Graph()

    col = None
    label = "Probabilité de Rcession"
    for c in ["recession_prob", "recession_probability", "recession_12m", "usrec"]:
        if c in df.columns:
            col = c
            break

    if not col:
        return dcc.Graph(figure=go.Figure().add_annotation(text="Probabilité de récession non disponible", showarrow=False))

    if col == 'usrec':
        label = "Rcession US (0/1)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines+markers', name=label))

    fig.update_layout(
        title="Probabilité de Rcession Économique",
        xaxis_title="Période",
        yaxis_title="Probabilité",
        template='plotly_dark',
        yaxis=dict(range=[0, 1]) if col != 'usrec' else {}
    )
    return dcc.Graph(figure=fig)


def _recession_badge(last_val: float | None) -> dbc.Badge:
    if last_val is None:
        return dbc.Badge("N/A", color="secondary")

    if last_val > 0.7:
        color = "danger"
        text = "Risque élevé"
    elif last_val > 0.3:
        color = "warning"
        text = "Risque modéré"
    else:
        color = "success"
        text = "Risque faible"

    return dbc.Badge(f" Rcession: {last_val:.2f} ({text})", color=color)


def _recession_container() -> dbc.Container:
    try:
        parts = sorted(Path('data/macro/forecast').glob('dt=*/macro_forecast.parquet'))
        if not parts:
            return dbc.Container([dbc.Alert("Aucun macro_forecast.parquet trouvé.", color="warning")])

        df = pd.read_parquet(parts[-1])
        if df is None or df.empty:
            return dbc.Container([dbc.Alert("macro_forecast.parquet vide.", color="warning")])

        # Chart
        chart = _recession_chart(df)

        # Badge
        col = None
        for c in ["recession_prob", "recession_probability", "recession_12m", "usrec"]:
            if c in df.columns:
                col = c
                break
        badge = _recession_badge(df[col].iloc[-1] if col else None)

        # Table
        if col:
            out = df[[col]].tail(10).reset_index(drop=True)
            table = dbc.Table.from_dataframe(out, striped=True, bordered=False, hover=True, size='sm')
        else:
            table = html.Small("Probabilité non trouvée.")

        return dbc.Container([
            dbc.Row(dbc.Col(badge, className="mb-3")),
            dbc.Row(dbc.Col(chart, className="mb-3")),
            dbc.Row(dbc.Col([html.H5("Derniers points"), table]))
        ], fluid=True)
    except Exception as e:
        return dbc.Container([dbc.Alert(f"Erreur Recession: {e}", color="danger")])


def layout():
    return html.Div([
        html.H3("Recession"),
        _recession_container(),
    ])
