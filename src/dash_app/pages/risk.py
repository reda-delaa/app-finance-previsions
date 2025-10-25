from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc


def _risk_chart(df: pd.DataFrame) -> dcc.Graph:
    if df is None or df.empty:
        return dcc.Graph()

    # Detect available columns for plotting
    plot_cols = []
    if 'vix' in df.columns:
        plot_cols.append(('VIX Index', 'vix'))
    if 'credit_spread' in df.columns or 'bamlh0a0hym2' in df.columns:
        col = 'credit_spread' if 'credit_spread' in df.columns else 'bamlh0a0hym2'
        plot_cols.append(('Spread Crédit', col))
    if 'drawdown_prob' in df.columns:
        plot_cols.append(('Prob. Drawdown', 'drawdown_prob'))
    if 'risk_index' in df.columns:
        plot_cols.append(('Indice Risque', 'risk_index'))
    if 'nfci' in df.columns or 'nfc_i' in df.columns:
        col = 'nfci' if 'nfci' in df.columns else 'nfc_i'
        plot_cols.append(('NFCI Risk', col))
    if 'unrate' in df.columns:
        plot_cols.append(('Taux Chômage (%)', 'unrate'))

    if not plot_cols:
        return dcc.Graph(figure=go.Figure().add_annotation(text="Aucun indicateur de risque détecté", showarrow=False))

    fig = go.Figure()
    for label, col in plot_cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=label))

    fig.update_layout(
        title="Indicateurs de Risque Macroéconomique",
        xaxis_title="Période",
        yaxis_title="Valeur",
        template='plotly_dark'
    )
    return dcc.Graph(figure=fig)


def _risk_badge(last_val: float | None, label: str) -> dbc.Badge:
    if last_val is None:
        return dbc.Badge("N/A", color="secondary")

    # VIX: low = green, high = danger
    if 'vix' in label.lower():
        color = "success" if last_val < 15 else "warning" if last_val < 30 else "danger"
    # Spread: low = green, high = danger
    elif 'spread' in label.lower():
        color = "success" if last_val < 3 else "warning" if last_val < 6 else "danger"
    # Drawdown/prob: low = green, high = danger
    elif 'drawdown' in label.lower() or 'prob' in label.lower():
        color = "success" if last_val < 0.3 else "warning" if last_val < 0.7 else "danger"
    # Unemployment low = green
    elif 'chômage' in label.lower():
        color = "success" if last_val < 5 else "warning" if last_val < 8 else "danger"
    # NFCI: negative = good, positive = bad
    elif 'nfci' in label.lower():
        color = "success" if last_val < 0 else "danger"
    else:
        color = "secondary"

    return dbc.Badge(f"{label}: {last_val:.2f}", color=color)


def _risk_container() -> dbc.Container:
    try:
        parts = sorted(Path('data/macro/forecast').glob('dt=*/macro_forecast.parquet'))
        if not parts:
            return dbc.Container([dbc.Alert("Aucun macro_forecast.parquet trouvé.", color="warning")])

        df = pd.read_parquet(parts[-1])
        if df is None or df.empty:
            return dbc.Container([dbc.Alert("macro_forecast.parquet vide.", color="warning")])

        # Chart
        chart = _risk_chart(df)

        # Badges
        badges = []
        if 'vix' in df.columns:
            badges.append(_risk_badge(df['vix'].iloc[-1], "VIX"))
        if 'credit_spread' in df.columns or 'bamlh0a0hym2' in df.columns:
            col = 'credit_spread' if 'credit_spread' in df.columns else 'bamlh0a0hym2'
            badges.append(_risk_badge(df[col].iloc[-1], "Spread Crédit"))
        if 'drawdown_prob' in df.columns:
            badges.append(_risk_badge(df['drawdown_prob'].iloc[-1], "Drawdown Prob"))
        if 'nfci' in df.columns or 'nfc_i' in df.columns:
            col = 'nfci' if 'nfci' in df.columns else 'nfc_i'
            badges.append(_risk_badge(df[col].iloc[-1], "NFCI"))
        if 'unrate' in df.columns:
            badges.append(_risk_badge(df['unrate'].iloc[-1], "Chômage"))

        # Table
        key_cols = ['vix', 'credit_spread', 'bamlh0a0hym2', 'drawdown_prob', 'risk_index', 'nfci', 'nfc_i']
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
        return dbc.Container([dbc.Alert(f"Erreur Risk: {e}", color="danger")])


def layout():
    return html.Div([
        html.H3("Risk"),
        _risk_container(),
    ])
