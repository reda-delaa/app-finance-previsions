from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html, dcc, dash
import dash


def _load_ticker_data(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load price data, forecasts, and news for a ticker"""
    ticker = ticker.strip().upper()

    # Load price data
    prices_path = Path(f'data/prices/ticker={ticker}/prices.parquet')
    prices_df = pd.DataFrame()
    if prices_path.exists():
        prices_df = pd.read_parquet(prices_path)
        if not prices_df.empty:
            prices_df = prices_df.set_index('date').sort_index()

    # Load forecasts for this ticker
    forecasts_df = pd.DataFrame()
    try:
        parts = sorted(Path('data/forecast').glob('dt=*'))
        if parts:
            latest = parts[-1]
            final_path = latest / 'final.parquet'
            if final_path.exists():
                final_df = pd.read_parquet(final_path)
                forecasts_df = final_df[final_df['ticker'] == ticker].copy()
    except Exception:
        pass

    # Load news for this ticker (placeholder)
    news_df = pd.DataFrame({
        'title': [f'News sample for {ticker}'],
        'summary': ['Sample news summary'],
        'sentiment': ['neutral'],
        'published': [pd.Timestamp.now()]
    })

    return prices_df, forecasts_df, news_df


def _create_price_chart(prices_df: pd.DataFrame) -> dcc.Graph:
    """Create 5-year price chart"""
    if prices_df.empty:
        return dcc.Graph(figure=go.Figure().add_annotation(text="Aucune donnée de prix disponible", showarrow=False))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices_df.index, y=prices_df['close'], mode='lines', name='Prix de clôture'))

    # Add simple moving averages if enough data
    if len(prices_df) >= 20:
        prices_df['SMA_20'] = prices_df['close'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=prices_df.index, y=prices_df['SMA_20'], mode='lines', name='SMA 20j'))

    if len(prices_df) >= 50:
        prices_df['SMA_50'] = prices_df['close'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=prices_df.index, y=prices_df['SMA_50'], mode='lines', name='SMA 50j'))

    fig.update_layout(
        title=f"Cours de l'action (5 ans)",
        xaxis_title="Date",
        yaxis_title="Prix",
        template='plotly_dark',
        height=400
    )

    return dcc.Graph(figure=fig)


def _create_forecasts_table(forecasts_df: pd.DataFrame) -> dbc.Table:
    """Create forecasts table"""
    if forecasts_df.empty:
        return dbc.Alert("Aucune prévision disponible pour ce ticker.", color="info")

    # Prepare display data
    display_df = forecasts_df[['horizon', 'final_score', 'direction', 'confidence', 'expected_return']].copy()
    display_df['confidence'] = display_df['confidence'].fillna(0).apply(lambda x: f"{x".1%"}")
    display_df['expected_return'] = display_df['expected_return'].fillna(0).apply(lambda x: f"{x".2%"}")

    return dbc.Table.from_dataframe(
        display_df.reset_index(drop=True),
        striped=True, bordered=False, hover=True, size='sm'
    )


def _create_news_section(news_df: pd.DataFrame) -> html.Div:
    """Create news section"""
    if news_df.empty:
        return dbc.Alert("Aucune actualité disponible pour ce ticker.", color="info")

    news_items = []
    for _, row in news_df.head(3).iterrows():
        sentiment_color = {
            'positive': 'success',
            'negative': 'danger',
            'neutral': 'secondary'
        }.get(row.get('sentiment', 'neutral'), 'secondary')

        news_items.append(
            dbc.Card([
                dbc.CardHeader([
                    html.Small(row.get('published', '').strftime('%Y-%m-%d') if hasattr(row.get('published', ''), 'strftime') else str(row.get('published', ''))),
                    dbc.Badge(row.get('sentiment', 'neutral').title(), color=sentiment_color, className="ms-2")
                ]),
                dbc.CardBody([
                    html.H6(row.get('title', ''), className="card-title"),
                    html.P(row.get('summary', ''), className="card-text")
                ])
            ], className="mb-2")
        )

    return html.Div([
        html.H5("Actualités récentes"),
        html.Div(news_items)
    ])


def layout():
    return html.Div([
        html.H3("Deep Dive - Analyse d'un titre"),
        dbc.Row([
            dbc.Col([
                html.Small("Entrez un ticker (ex: AAPL, MSFT): "),
                dcc.Input(id='deep-dive-ticker', type='text', placeholder='AAPL', debounce=True, style={"minWidth": "200px"}),
                html.Button("Analyser", id='deep-dive-analyze', n_clicks=0, className="ms-2 btn btn-primary")
            ], md=6)
        ], className="mb-3"),
        html.Div(id='deep-dive-content')
    ])


@dash.callback(
    dash.Output('deep-dive-content', 'children'),
    dash.Input('deep-dive-analyze', 'n_clicks'),
    dash.State('deep-dive-ticker', 'value')
)
def analyze_ticker(n_clicks, ticker):
    if n_clicks == 0 or not ticker:
        return dbc.Alert("Entrez un ticker et cliquez sur Analyser.", color="info")

    try:
        ticker = ticker.strip().upper()

        # Load data
        prices_df, forecasts_df, news_df = _load_ticker_data(ticker)

        if prices_df.empty and forecasts_df.empty and news_df.empty:
            return dbc.Alert(f"Aucune donnée disponible pour le ticker {ticker}.", color="warning")

        # Create content sections
        sections = []

        # Price chart section
        if not prices_df.empty:
            sections.append(dbc.Card([
                dbc.CardHeader("Cours de l'action (5 ans)"),
                dbc.CardBody(_create_price_chart(prices_df))
            ], className="mb-3"))

        # Forecasts section
        if not forecasts_df.empty:
            sections.append(dbc.Card([
                dbc.CardHeader(f"Prévisions pour {ticker}"),
                dbc.CardBody(_create_forecasts_table(forecasts_df))
            ], className="mb-3"))

        # News section
        sections.append(dbc.Card([
            dbc.CardHeader(f"Actualités pour {ticker}"),
            dbc.CardBody(_create_news_section(news_df))
        ], className="mb-3"))

        # Basic statistics
        if not prices_df.empty:
            stats_items = []
            if 'close' in prices_df.columns:
                current_price = prices_df['close'].iloc[-1]
                stats_items.append(html.Li(f"Prix actuel: ${current_price".2f"}"))

                if len(prices_df) > 1:
                    change_1d = prices_df['close'].iloc[-1] - prices_df['close'].iloc[-2]
                    stats_items.append(html.Li(f"Variation 1j: ${change_1d"+.2f"}"))

            if stats_items:
                sections.append(dbc.Card([
                    dbc.CardHeader("Statistiques de base"),
                    dbc.CardBody(html.Ul(stats_items))
                ], className="mb-3"))

        return html.Div(sections)

    except Exception as e:
        return dbc.Alert(f"Erreur lors de l'analyse du ticker {ticker}: {e}", color="danger")
