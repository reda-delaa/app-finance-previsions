from __future__ import annotations
from pathlib import Path
import pandas as pd
from dash import html, dcc, dash_table, callback, Output, Input
import dash_bootstrap_components as dbc
from utils.partitions import latest_dt, read_parquet_latest


def _load_forecast_data() -> pd.DataFrame:
    """Charge la dernière partition de prévisions depuis data/forecast.

    Retourne un DataFrame vide en cas d'erreur ou d'absence de données.
    """
    try:
        # détermine la date la plus récente et lit le fichier final.parquet
        _ = latest_dt("data/forecast")
        df = read_parquet_latest("data/forecast", "final.parquet")
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception:
        return pd.DataFrame()


def layout() -> html.Div:
    """Layout pour la page des prévisions multi-titres.

    Affiche des filtres par horizon et ticker et une table de données.
    """
    df = _load_forecast_data()
    horizons = sorted(df["horizon"].astype(str).unique()) if not df.empty else []
    tickers = sorted(df["ticker"].astype(str).unique()) if not df.empty else []
    return html.Div([
        html.H2("Prévisions multi-titres", className="mt-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Filtre par horizon"),
                dcc.Dropdown(
                    id="forecast-horizon-filter",
                    options=[{"label": "Tous", "value": ""}] + [{"label": h, "value": h} for h in horizons],
                    value="",
                    clearable=False,
                ),
            ], md=3),
            dbc.Col([
                html.Label("Filtre par ticker"),
                dcc.Dropdown(
                    id="forecast-ticker-filter",
                    options=[{"label": "Tous", "value": ""}] + [{"label": t, "value": t} for t in tickers],
                    value="",
                    clearable=False,
                    multi=False,
                ),
            ], md=3),
        ], className="mb-3"),
        dash_table.DataTable(
            id="forecast-table",
            columns=[{"name": c, "id": c} for c in df.columns],
            data=df.to_dict("records"),
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
            style_data_empty={"border": "1px solid #dee2e6", "height": "300px", "textAlign": "center"},
        ),
    ])


@callback(
    Output("forecast-table", "data"),
    Input("forecast-horizon-filter", "value"),
    Input("forecast-ticker-filter", "value"),
)
def update_table(selected_horizon: str, selected_ticker: str):
    """Met à jour la table en fonction des filtres sélectionnés."""
    df = _load_forecast_data()
    if df.empty:
        return []
    dff = df.copy()
    if selected_horizon:
        dff = dff[dff["horizon"].astype(str) == selected_horizon]
    if selected_ticker:
        dff = dff[dff["ticker"].astype(str) == selected_ticker]
    return dff.to_dict("records")