from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc


def _regimes_chart(df: pd.DataFrame) -> dcc.Graph:
    if df is None or df.empty:
        return dcc.Graph()

    # Detect available columns for plotting, map to actual data columns
    plot_cols = []
    if 'inflation_yoy' in df.columns:
        plot_cols.append(('Inflation YoY', 'inflation_yoy'))
    if 'yield_curve_slope' in df.columns:
        plot_cols.append(('Pente courbe taux', 'yield_curve_slope'))

    # If data is by horizon (not time series), plot bar for each horizon
    if 'horizon' in df.columns and len(df) <= 5:
        fig = go.Figure()
        for i, (label, col) in enumerate(plot_cols):
            fig.add_trace(go.Bar(
                x=df['horizon'],
                y=df[col],
                name=label,
                offsetgroup=i
            ))
        fig.update_layout(
            title="Indicateurs-Régimes par Horizon",
            xaxis_title="Horizon",
            yaxis_title="Valeur",
            template='plotly_dark',
            barmode='group'
        )
        return dcc.Graph(figure=fig)
    else:
        # Fallback to line chart if multiple time points
        fig = go.Figure()
        for label, col in plot_cols:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines+markers', name=label))
        fig.update_layout(
            title="Indicateurs de Régime Macroéconomique",
            xaxis_title="Période",
            yaxis_title="Valeur",
            template='plotly_dark'
        )
        return dcc.Graph(figure=fig)


def _trend_badge(last_val: float | None, label: str) -> dbc.Badge:
    if last_val is None:
        return dbc.Badge("N/A", color="secondary")

    # For recession prob, higher = worse
    if 'recession' in label.lower():
        color = "success" if last_val < 0.3 else "warning" if last_val < 0.7 else "danger"
    # For yields/spreads, negative = flight to quality (red)
    elif any(x in label.lower() for x in ['slope', 'spread']):
        color = "success" if last_val > 0 else "danger"
    # For inflation/risk, higher = warning
    elif any(x in label.lower() for x in ['inflation', 'risk', 'nfc']):
        color = "success" if last_val < 2 else "warning" if last_val < 5 else "danger"
    else:
        color = "secondary"

    return dbc.Badge(f"{label}: {last_val:.2f}", color=color)


def _body() -> dbc.Container:
    try:
        parts = sorted(Path('data/macro/forecast').glob('dt=*/macro_forecast.parquet'))
        if not parts:
            return dbc.Container([dbc.Alert("Aucun macro_forecast.parquet trouvé.", color="warning")])

        df = pd.read_parquet(parts[-1])
        if df is None or df.empty:
            return dbc.Container([dbc.Alert("macro_forecast.parquet vide.", color="warning")])

        # Chart section
        chart = _regimes_chart(df)

        # Badges
        badges = []
        if 'inflation_yoy' in df.columns:
            badges.append(_trend_badge(df['inflation_yoy'].iloc[-1], "Inflation"))
        if 'yield_curve_slope' in df.columns:
            badges.append(_trend_badge(df['yield_curve_slope'].iloc[-1], "Courbe"))

        # Table
        key_cols = ['cpi_yoy', 'slope_10y_2y', 'lei', 'pmi', 'ism', 'nfci']
        available_cols = [c for c in df.columns if c in key_cols]
        if available_cols:
            out = df[available_cols].tail(5).reset_index(drop=True)
            table = dbc.Table.from_dataframe(out, striped=True, bordered=False, hover=True, size='sm')
        else:
            table = html.Small("Colonnes clés non trouvées.")

        return dbc.Container([
            dbc.Row(dbc.Col(html.Div(badges), className="mb-3")),
            dbc.Row(dbc.Col(chart, className="mb-3")),
            dbc.Row(dbc.Col([html.H5("Derniers talons"), table]))
        ], fluid=True)
    except Exception as e:
        return dbc.Container([dbc.Alert(f"Erreur Regimes: {e}", color="danger")])


def layout():
    return html.Div([
        html.H3("Regimes"),
        _body(),
    ])
