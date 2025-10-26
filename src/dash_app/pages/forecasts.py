from __future__ import annotations

from pathlib import Path
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, dash
import dash


def _load_forecasts_data() -> pd.DataFrame:
    """Load latest forecasts data"""
    try:
        parts = sorted(Path('data/forecast').glob('dt=*'))
        if parts:
            latest = parts[-1]
            final_path = latest / 'final.parquet'
            if final_path.exists():
                return pd.read_parquet(final_path)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame({'error': [f"Erreur chargement forecasts: {e}"]})


def layout():
    # Controls
    controls = dbc.Row([
        dbc.Col([
            html.Small("Filtre par horizon: "),
            dcc.Dropdown(
                id='forecasts-horizon-filter',
                options=[
                    {'label': 'Tous', 'value': 'all'},
                    {'label': '1 semaine', 'value': '1w'},
                    {'label': '1 mois', 'value': '1m'},
                    {'label': '1 an', 'value': '1y'}
                ],
                value='all',
                clearable=False,
                style={"minWidth": "150px"}
            )
        ], md=3),
        dbc.Col([
            html.Small("Recherche ticker: "),
            dcc.Input(id='forecasts-ticker-search', type='text', placeholder='AAPL, MSFT...', debounce=True)
        ], md=3),
        dbc.Col([
            html.Small("Trier par: "),
            dcc.Dropdown(
                id='forecasts-sort-by',
                options=[
                    {'label': 'Score final', 'value': 'final_score'},
                    {'label': 'Ticker', 'value': 'ticker'},
                    {'label': 'Horizon', 'value': 'horizon'}
                ],
                value='final_score',
                clearable=False,
                style={"minWidth": "150px"}
            )
        ], md=3),
    ], className="mb-3")

    return html.Div([
        html.H3("Forecasts - Prévisions Multi-Tickers"),
        controls,
        html.Div(id='forecasts-table', className="mb-3"),
        html.Div(id='forecasts-summary', className="mb-3"),
    ])


@dash.callback(
    dash.Output('forecasts-table', 'children'),
    dash.Output('forecasts-summary', 'children'),
    dash.Input('forecasts-horizon-filter', 'value'),
    dash.Input('forecasts-ticker-search', 'value'),
    dash.Input('forecasts-sort-by', 'value')
)
def update_forecasts(horizon, ticker_search, sort_by):
    try:
        df = _load_forecasts_data()

        if df.empty or 'error' in df.columns:
            return (
                dbc.Alert("Aucune prévision disponible.", color="warning"),
                dbc.Alert("Erreur lors du chargement des prévisions.", color="danger")
            )

        # Filter by horizon
        if horizon != 'all':
            df = df[df['horizon'] == horizon]

        # Filter by ticker search
        if ticker_search:
            tickers = [t.strip().upper() for t in ticker_search.split(',') if t.strip()]
            df = df[df['ticker'].isin(tickers)]

        if df.empty:
            return (
                dbc.Alert("Aucune prévision trouvée avec ces critères.", color="info"),
                html.Small("Aucune donnée à résumer.")
            )

        # Sort
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=(sort_by != 'final_score'))

        # Prepare display data
        display_df = df[['ticker', 'horizon', 'final_score', 'direction', 'confidence', 'expected_return']].copy()
        display_df['confidence'] = display_df['confidence'].fillna(0).apply(lambda x: f"{x:.1%}")
        display_df['expected_return'] = display_df['expected_return'].fillna(0).apply(lambda x: f"{x:.2%}")
        display_df['final_score'] = display_df['final_score'].fillna(0).apply(lambda x: f"{x:.2f}")

        # Create table
        table = dbc.Table.from_dataframe(
            display_df.reset_index(drop=True),
            striped=True, bordered=False, hover=True, size='sm'
        )

        table_card = dbc.Card([
            dbc.CardHeader(f"Prévisions ({len(df)} résultats)"),
            dbc.CardBody(table)
        ])

        # Summary
        summary_items = []
        summary_items.append(html.Li(f"Total prévisions: {len(df)}"))
        summary_items.append(html.Li(f"Tickers uniques: {df['ticker'].nunique()}"))
        summary_items.append(html.Li(f"Horizons: {', '.join(df['horizon'].unique())}"))

        if 'final_score' in df.columns:
            avg_score = df['final_score'].mean()
            summary_items.append(html.Li(f"Score moyen: {avg_score".2f"}"))

        summary_card = dbc.Card([
            dbc.CardHeader("Résumé"),
            dbc.CardBody(html.Ul(summary_items))
        ])

        return table_card, summary_card

    except Exception as e:
        return (
            dbc.Alert("Erreur affichage prévisions.", color="danger"),
            dbc.Alert(f"Erreur résumé: {e}", color="danger")
        )
