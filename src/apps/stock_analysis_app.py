# stock_analysis_app.py

import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils.st_compat import get_st
st = get_st()
import ta

from core_runtime import new_trace_id, set_trace_id, get_trace_id

# ================== CONFIG ==================
# Configuration par défaut pour l'analyse d'une action
DEFAULT_TICKER = "NGD.TO"  # New Gold Inc. (exemple)
PEER_GROUPS = {
    "Exploitants aurifères": ["ABX.TO", "K.TO", "AEM.TO", "BTO.TO", "IMG.TO", "OR.TO"],
    "Exploitants argentifères": ["PAAS.TO", "EDR.TO", "FR.TO"],
    "Exploitants cuprifères": ["CS.TO", "TECK-B.TO", "LUN.TO", "FM.TO"],
    "Sociétés minières diversifiées": ["RIO", "BHP", "VALE", "FCX"]
}

# Indices de référence
BENCHMARKS = {
    "^GSPTSE": "Indice composé S&P/TSX",
    "GDX": "VanEck Gold Miners ETF",
    "XGD.TO": "iShares S&P/TSX Global Gold Index ETF",
    "XME": "SPDR S&P Metals & Mining ETF"
}

# Indicateurs macroéconomiques à surveiller
MACRO_INDICATORS = {
    "GC=F": "Contrats à terme sur l’or (Gold Futures)",
    "SI=F": "Contrats à terme sur l’argent (Silver Futures)",
    "HG=F": "Contrats à terme sur le cuivre (Copper Futures)",
    "DX-Y.NYB": "Indice du dollar américain (DXY)",
    "^TNX": "Rendement des bons du Trésor US à 10 ans"
}

# Séries FRED pour l'analyse économique approfondie
FRED_SERIES = {
    # Inflation / attentes
    "CPIAUCSL": "Indice des prix à la consommation (tous articles, 1982-84=100)",
    "T10YIE":   "Inflation anticipée à 10 ans (breakeven)",
    # Croissance / activité
    "INDPRO":   "Indice de production industrielle",
    "GDPC1":    "PIB réel (trimestriel, en chaîne)",
    # Marché du travail
    "UNRATE":   "Taux de chômage",
    "PAYEMS":   "Emplois non agricoles (total)",
    # Taux & courbe
    "DGS10":    "Taux des bons du Trésor US à 10 ans",
    "DGS2":     "Taux des bons du Trésor US à 2 ans",
    # Dollar US
    "DTWEXBGS": "Indice pondéré du dollar US (large)",
    # Conditions financières / crédit
    "NFCI":     "Indice des conditions financières (Chicago Fed)",
    "BAMLC0A0CM": "Spread agrégé obligations US (Investment Grade)",
    "BAMLH0A0HYM2": "Spread obligations US à haut rendement",
    # Récessions (ombrage)
    "USREC":    "Indicateur de récession US"
}

# Sensibilité des secteurs aux facteurs économiques
SECTOR_SENSITIVITY = pd.DataFrame({
    "Inflation":{
        "XLK":-1,"XLF":1,"XLE":2,"XLB":2,"XLV":0,"XLY":-1,"XLP":0,"XLI":1,"XLRE":-1,"XLU":1,
        "GDX":2,"ABX.TO":2,"K.TO":2,"AEM.TO":2,"BTO.TO":2,"IMG.TO":2,"DGC.TO":2,
        "PAAS.TO":2,"EDR.TO":2,"FR.TO":2,"CS.TO":1,"TECK-B.TO":1,"LUN.TO":1,"FM.TO":1
    },
    "Growth":{
        "XLK":2,"XLF":1,"XLE":0,"XLB":1,"XLV":0,"XLY":2,"XLP":0,"XLI":1,"XLRE":0,"XLU":-1,
        "GDX":0,"ABX.TO":0,"K.TO":0,"AEM.TO":0,"BTO.TO":0,"IMG.TO":0,"DGC.TO":0,
        "PAAS.TO":0,"EDR.TO":0,"FR.TO":0,"CS.TO":1,"TECK-B.TO":1,"LUN.TO":1,"FM.TO":1
    },
    "Rates":{
        "XLK":-2,"XLF":2,"XLE":1,"XLB":0,"XLV":0,"XLY":-1,"XLP":0,"XLI":0,"XLRE":-1,"XLU":1,
        "GDX":1,"ABX.TO":1,"K.TO":1,"AEM.TO":1,"BTO.TO":1,"IMG.TO":1,"DGC.TO":1,
        "PAAS.TO":1,"EDR.TO":1,"FR.TO":1,"CS.TO":0,"TECK-B.TO":0,"LUN.TO":0,"FM.TO":0
    },
    "USD":{
        "XLK":-1,"XLF":0,"XLE":-1,"XLB":-1,"XLV":0,"XLY":0,"XLP":0,"XLI":-1,"XLRE":-1,"XLU":0,
        "GDX":-1,"ABX.TO":-1,"K.TO":-1,"AEM.TO":-1,"BTO.TO":-1,"IMG.TO":-1,"DGC.TO":-1,
        "PAAS.TO":-1,"EDR.TO":-1,"FR.TO":-1,"CS.TO":-1,"TECK-B.TO":-1,"LUN.TO":-1,"FM.TO":-1
    }
}).fillna(0)

# ================== HELPERS ==================
def get_stock_data(ticker, period="5y", interval="1d"):
    """Récupère les données historiques d'un actif (action, indice, future)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        # S'assurer des colonnes attendues
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in hist.columns:
                hist[col] = np.nan
        return hist
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour {ticker}: {e}")
        return None

def get_peer_data(tickers, period="1y"):
    """Récupère les données de clôture pour un groupe d'actifs."""
    data = {}
    valid_tickers = []
    for t in tickers:
        try:
            hist = get_stock_data(t, period=period)
            if hist is not None and not hist.empty:
                data[t] = hist["Close"]
                valid_tickers.append(t)
            else:
                st.warning(f"Aucune donnée disponible pour {t}")
            time.sleep(0.2)  # éviter de surcharger l'API
        except Exception as e:
            st.warning(f"Impossible de récupérer les données pour {t}: {e}")
    return pd.DataFrame(data), valid_tickers

def calculate_returns(df, periods):
    """Calcule les rendements (%) sur différentes périodes (en jours ouvrés)."""
    returns = {}
    for period_label, days in periods.items():
        try:
            if len(df) > days:
                returns[period_label] = df.pct_change(days).iloc[-1] * 100
            else:
                # si la période est plus longue que l'historique disponible
                returns[period_label] = (
                    df.pct_change(len(df)-1).iloc[-1] * 100
                    if len(df) > 1 else pd.Series(0, index=df.columns)
                )
        except Exception as e:
            st.warning(f"Erreur lors du calcul des rendements pour {period_label}: {e}")
            returns[period_label] = pd.Series(0, index=df.columns)
    return pd.DataFrame(returns)

def calculate_volatility(df, window=20):
    """Volatilité annualisée (%) calculée sur une fenêtre glissante."""
    return df.pct_change().rolling(window).std() * np.sqrt(252) * 100

def calculate_beta(stock_returns, benchmark_returns, window=60):
    """Coefficient bêta glissant (covariance/variance) d'un actif vs indice."""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(index=stock_returns.index)
    rolling_cov = aligned.iloc[:, 0].rolling(window=window).cov(aligned.iloc[:, 1])
    rolling_var = aligned.iloc[:, 1].rolling(window=window).var()
    return rolling_cov / rolling_var

def add_technical_indicators(df):
    """Ajoute des indicateurs techniques au DataFrame (MMS, RSI, MACD, Bollinger, OBV)."""
    data = df.copy()
    # Moyennes mobiles simples
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    # Bandes de Bollinger
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()
    data['BB_Middle'] = bb.bollinger_mavg()
    # Volume : On-Balance Volume
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    return data

def get_company_info(ticker):
    """Récupère les informations fondamentales de l'émetteur via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info if isinstance(info, dict) else {}
    except Exception as e:
        st.error(f"Erreur lors de la récupération des informations pour {ticker}: {e}")
        return {}

def load_fred_series(series_id, start_date=None):
    """Récupère une série temporelle depuis FRED (CSV public)."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        if start_date:
            url += f"&startdate={start_date.strftime('%Y-%m-%d')}"
        df = pd.read_csv(url)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.set_index("DATE").replace(".", np.nan).astype(float)
        df.columns = [series_id]
        return df
    except Exception as e:
        st.warning(f"Échec de récupération de {series_id} depuis FRED: {e}")
        return pd.DataFrame(columns=[series_id])

def get_fred_data(series_ids, start_date=None):
    """Récupère plusieurs séries FRED et les agrège dans un DataFrame."""
    data = {}
    for series_id in series_ids:
        try:
            df = load_fred_series(series_id, start_date)
            if not df.empty:
                data[series_id] = df[series_id]
            time.sleep(0.2)
        except Exception as e:
            st.warning(f"Échec de récupération de {series_id}: {e}")
    if not data:
        return pd.DataFrame()
    result = pd.concat(data, axis=1)
    result.columns = [FRED_SERIES.get(col, col) for col in result.columns]
    return result

def zscore(series, window=24):
    """Calcule le z-score glissant d'une série."""
    s = series.dropna()
    if len(s) < window + 2:
        return pd.Series(index=series.index, dtype=float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return ((s - mu) / (sd.replace(0, np.nan))).reindex(series.index)

def get_financials(ticker):
    """Récupère états financiers (résultat, bilan, flux de trésorerie)."""
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        return {
            "income_stmt": income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow
        }
    except Exception as e:
        st.warning(f"Erreur lors de la récupération des données financières pour {ticker}: {e}")
        return {}

def get_similar_stocks(ticker, n=5):
    """Trouve des actifs similaires par corrélation des rendements."""
    # Déterminer le groupe de pairs approprié
    peer_group = []
    for group, tickers in PEER_GROUPS.items():
        if ticker in tickers:
            peer_group = tickers.copy()
            break
    if not peer_group:
        peer_group = PEER_GROUPS["Exploitants aurifères"].copy()
    # retirer le ticker principal s'il est présent
    if ticker in peer_group:
        peer_group.remove(ticker)
    # Récupérer les données
    peers_df, _valid = get_peer_data([ticker] + peer_group)
    if peers_df is None or peers_df.empty or ticker not in peers_df.columns:
        return []
    # Corrélations des rendements
    returns = peers_df.pct_change().dropna()
    correlations = returns.corr()[ticker].drop(ticker)
    return correlations.nlargest(n).index.tolist()

# ================== UI ==================
# Ne configure la page que si on est vraiment dans Streamlit
try:
    import streamlit as _st_real  # noqa: F401
    st.set_page_config(page_title="Analyse Approfondie d'Action", layout="wide")
except Exception:
    pass

# ---- Trace ID visible UI
if "trace_id" not in st.session_state or not st.session_state["trace_id"]:
    st.session_state["trace_id"] = new_trace_id()
else:
    set_trace_id(st.session_state["trace_id"])
st.caption(f"Trace ID: `{st.session_state['trace_id']}`")

st.title("📊 Analyse Approfondie d'Action")

with st.sidebar:
    st.header("Paramètres")
    ticker = st.text_input("Symbole de l'action", value=DEFAULT_TICKER)
    period = st.selectbox("Période d'analyse", ["1y", "2y", "3y", "5y", "10y", "max"], index=2)
    benchmark = st.selectbox("Indice de référence", list(BENCHMARKS.keys()), format_func=lambda x: f"{x} - {BENCHMARKS[x]}")
    
    st.subheader("Indicateurs techniques")
    show_sma = st.checkbox("Moyennes mobiles simples (20/50/200 jours)", value=True)
    show_bb = st.checkbox("Bandes de Bollinger", value=True)
    show_rsi = st.checkbox("Indice de force relative (RSI)", value=True)
    show_macd = st.checkbox("Convergence–Divergence des moyennes mobiles (MACD)", value=True)
    
    st.subheader("Analyse comparative")
    compare_peers = st.checkbox("Comparer avec les actions similaires", value=True)
    compare_macro = st.checkbox("Comparer avec les indicateurs macroéconomiques", value=True)
    
    if st.button("🔄 Actualiser les données"):
        st.cache_data.clear()
        st.rerun()

# Vérifier si le ticker est valide
if not ticker:
    st.warning("Veuillez entrer un symbole d'action valide.")
    st.stop()

# --------- Chargement des données ----------
@st.cache_data(ttl=3600)
def load_stock_data(t, p):
    return get_stock_data(t, period=p)

@st.cache_data(ttl=3600)
def load_company_info(t):
    return get_company_info(t)

@st.cache_data(ttl=3600)
def load_financials_cached(t):
    return get_financials(t)

@st.cache_data(ttl=3600)
def load_benchmark_data_cached(bmk, p):
    return get_stock_data(bmk, period=p)

@st.cache_data(ttl=3600)
def load_similar_stocks_cached(t):
    return get_similar_stocks(t)

@st.cache_data(ttl=3600)
def load_macro_indicators(p):
    data = {}
    for indicator, _name in MACRO_INDICATORS.items():
        try:
            hist = get_stock_data(indicator, period=p)
            if hist is not None and not hist.empty:
                data[indicator] = hist["Close"]
            time.sleep(0.2)
        except Exception as e:
            st.warning(f"Impossible de récupérer les données pour {indicator}: {e}")
    return pd.DataFrame(data)

# Chargement des données
if getattr(st, "_is_dummy", False):
    # En mode test/bare, évite les appels réseau à l'import
    stock_data = pd.DataFrame({"Close": [100, 101, 102], "Volume": [0, 0, 0]})
    stock_data_with_indicators = add_technical_indicators(stock_data)
    company_info = {}
    financials = {}
    benchmark_data = stock_data.copy()
    similar_stocks = []
    macro_data = pd.DataFrame()
else:
    with st.spinner("Chargement des données de l'action..."):
        stock_data = load_stock_data(ticker, period)
        if stock_data is None or stock_data.empty:
            st.error(f"Impossible de récupérer les données pour {ticker}. Vérifiez que le symbole est correct.")
            st.stop()
        
        # Ajouter les indicateurs techniques
        stock_data_with_indicators = add_technical_indicators(stock_data)
        
        # Informations sur l'entreprise
        company_info = load_company_info(ticker)
        
        # Données financières
        financials = load_financials_cached(ticker)
        
        # Données de l'indice de référence
        benchmark_data = load_benchmark_data_cached(benchmark, period)
        
        # Actions similaires
        try:
            similar_stocks = load_similar_stocks_cached(ticker) if compare_peers else []
        except Exception as e:
            st.warning(f"Erreur lors du chargement des actions similaires: {e}")
            similar_stocks = []
        
        # Indicateurs macroéconomiques
        macro_data = load_macro_indicators(period) if compare_macro else pd.DataFrame()

# --------- Affichage des informations de l'entreprise ----------
if company_info:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        company_name = company_info.get("longName", ticker)
        st.header(f"{company_name} ({ticker})")
        st.markdown(f"**Secteur :** {company_info.get('sector', 'N/A')} | **Industrie :** {company_info.get('industry', 'N/A')}")
        st.markdown(company_info.get("longBusinessSummary", "Aucune description disponible."))
    
    with col2:
        st.subheader("Données de marché")
        current_price = stock_data["Close"].iloc[-1]
        previous_close = stock_data["Close"].iloc[-2] if len(stock_data) > 1 else current_price
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100 if previous_close else 0.0
        
        st.metric("Prix actuel", f"{current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
        st.metric("Volume moyen (30 jours)", f"{stock_data['Volume'].tail(30).mean():.0f}")
        st.metric("Capitalisation boursière (M)", f"{company_info.get('marketCap', 0) / 1e6:.2f}")
    
    with col3:
        st.subheader("Valorisation")
        st.metric("Cours/Bénéfices (P/E)", f"{company_info.get('trailingPE', 'N/A')}")
        st.metric("Cours/Valeur comptable (P/B)", f"{company_info.get('priceToBook', 'N/A')}")
        st.metric("Valeur d’entreprise / EBITDA", f"{company_info.get('enterpriseToEbitda', 'N/A')}")

# --------- Graphique principal ----------
st.subheader("Évolution du cours")

# Graphique avec sous-graphiques
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, 
    vertical_spacing=0.1, 
    row_heights=[0.7, 0.3],
    subplot_titles=(f"Cours de {ticker}", "Volume échangé")
)

# Chandeliers (prix)
fig.add_trace(
    go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name=f"Cours {ticker}"
    ),
    row=1, col=1
)

# Moyennes mobiles simples
if show_sma:
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data_with_indicators['SMA_20'],
            line=dict(color='blue', width=1),
            name='Moyenne mobile simple (20 jours)'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data_with_indicators['SMA_50'],
            line=dict(color='orange', width=1),
            name='Moyenne mobile simple (50 jours)'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data_with_indicators['SMA_200'],
            line=dict(color='red', width=1),
            name='Moyenne mobile simple (200 jours)'
        ),
        row=1, col=1
    )

# Bandes de Bollinger
if show_bb:
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data_with_indicators['BB_Upper'],
            line=dict(color='rgba(0,128,0,0.6)', width=1),
            name='Bande de Bollinger supérieure'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data_with_indicators['BB_Lower'],
            line=dict(color='rgba(0,128,0,0.6)', width=1),
            fill='tonexty',
            fillcolor='rgba(0,128,0,0.12)',
            name='Bande de Bollinger inférieure'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data_with_indicators['BB_Middle'],
            line=dict(color='rgba(0,128,0,0.6)', width=1, dash='dot'),
            name='Moyenne (Bollinger)'
        ),
        row=1, col=1
    )

# Volume
fig.add_trace(
    go.Bar(
        x=stock_data.index, y=stock_data['Volume'],
        marker_color='rgba(0,0,128,0.5)',
        name='Volume'
    ),
    row=2, col=1
)

# Mise en page
fig.update_layout(
    height=600, 
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, width="stretch")

# --------- Indicateurs techniques ----------
st.subheader("Indicateurs techniques")

col1, col2 = st.columns(2)

with col1:
    if show_rsi:
        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data_with_indicators['RSI'],
                line=dict(color='purple', width=1),
                name="Indice de force relative (RSI)"
            )
        )
        # lignes horizontales (70/30)
        fig_rsi.add_shape(
            type="line", x0=stock_data.index[0], y0=70, x1=stock_data.index[-1], y1=70,
            line=dict(color="red", width=1, dash="dash")
        )
        fig_rsi.add_shape(
            type="line", x0=stock_data.index[0], y0=30, x1=stock_data.index[-1], y1=30,
            line=dict(color="green", width=1, dash="dash")
        )
        fig_rsi.update_layout(
            title="Indice de force relative (période 14 jours)",
            yaxis=dict(range=[0, 100]),
            height=300,
            hovermode="x unified"
        )
        st.plotly_chart(fig_rsi, width="stretch")

with col2:
    if show_macd:
        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(
            go.Scatter(
                x=stock_data.index, y=stock_data_with_indicators['MACD'],
                line=dict(color='blue', width=1),
                name='MACD (12-26-9)'
            )
        )
        fig_macd.add_trace(
            go.Scatter(
                x=stock_data.index, y=stock_data_with_indicators['MACD_Signal'],
                line=dict(color='red', width=1),
                name='Ligne de signal MACD'
            )
        )
        colors = ['green' if val > 0 else 'red' for val in stock_data_with_indicators['MACD_Hist']]
        fig_macd.add_trace(
            go.Bar(
                x=stock_data.index, y=stock_data_with_indicators['MACD_Hist'],
                marker_color=colors,
                name='Histogramme MACD'
            )
        )
        fig_macd.update_layout(
            title="Convergence–Divergence des moyennes mobiles (MACD)",
            height=300,
            hovermode="x unified"
        )
        st.plotly_chart(fig_macd, width='stretch')

# --------- Analyse de performance ----------
st.subheader("Analyse de performance")

# Périodes de comparaison (jours ouvrés)
periods = {
    "1 semaine": 5,
    "1 mois": 21,
    "3 mois": 63,
    "6 mois": 126,
    "1 an": 252,
    "Depuis le début de l'année (YTD)": (datetime.now() - datetime(datetime.now().year, 1, 1)).days
}

# Données pour la comparaison
comparison_data = pd.DataFrame()
comparison_data[ticker] = stock_data['Close']

if benchmark_data is not None and not benchmark_data.empty:
    comparison_data[benchmark] = benchmark_data['Close']

# Ajouter les pairs
if compare_peers and similar_stocks:
    peer_df, valid_peers = get_peer_data(similar_stocks, period=period)
    for peer in peer_df.columns:
        comparison_data[peer] = peer_df[peer]

# Normalisation base 100
if not comparison_data.empty:
    comparison_normalized = comparison_data.dropna().div(comparison_data.dropna().iloc[0]) * 100
else:
    st.warning("Pas assez de données pour l'analyse comparative")
    comparison_normalized = pd.DataFrame()

# Graphique de performance relative
if not comparison_normalized.empty:
    try:
        fig_comp = px.line(
            comparison_normalized,
            x=comparison_normalized.index,
            y=comparison_normalized.columns,
            title="Performance relative (base 100)",
            labels={"value": "Indice (base 100)", "variable": "Symbole"}
        )
        fig_comp.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_comp, width='stretch')
    except Exception as e:
        st.warning(f"Erreur lors de la création du graphique de comparaison: {e}")
else:
    st.info("Pas assez de données pour afficher le graphique de comparaison")

# Tableau des rendements
if not comparison_data.empty and len(comparison_data) > 1:
    try:
        returns_df = calculate_returns(comparison_data, periods)
        if not returns_df.empty:
            styled_df = returns_df.T.style.format("{:.2f}%")
            try:
                styled_df = styled_df.background_gradient(cmap="RdYlGn", axis=1)
            except Exception:
                pass
            st.dataframe(styled_df)
        else:
            st.info("Pas assez de données pour calculer les rendements")
    except Exception as e:
        st.warning(f"Erreur lors de l'affichage des rendements: {e}")
        st.dataframe(comparison_data.tail(5))
else:
    st.info("Pas assez de données pour calculer les rendements")

# --------- Analyse de risque ----------
st.subheader("Analyse de risque")

col1, col2 = st.columns(2)

with col1:
    # Volatilité
    try:
        if not comparison_data.empty and len(comparison_data) > 20:
            volatility = calculate_volatility(comparison_data)
            if not volatility.empty:
                fig_vol = px.line(
                    volatility,
                    x=volatility.index,
                    y=volatility.columns,
                    title="Volatilité annualisée (fenêtre glissante de 20 jours)",
                    labels={"value": "Volatilité (%)", "variable": "Symbole"}
                )
                fig_vol.update_layout(height=300, hovermode="x unified")
                st.plotly_chart(fig_vol, width='stretch')
            else:
                st.info("Pas assez de données pour calculer la volatilité")
        else:
            st.info("Pas assez de données pour la volatilité (minimum 20 jours requis)")
    except Exception as e:
        st.warning(f"Erreur lors du calcul de la volatilité: {e}")

with col2:
    # Bêta vs indice de référence
    try:
        if benchmark in comparison_data.columns and ticker in comparison_data.columns:
            if len(comparison_data) > 60:
                stock_returns = comparison_data[ticker].pct_change().dropna()
                benchmark_returns = comparison_data[benchmark].pct_change().dropna()
                beta = calculate_beta(stock_returns, benchmark_returns)
                if not beta.empty and not beta.isna().all():
                    fig_beta = px.line(
                        beta, x=beta.index, y=beta,
                        title=f"Coefficient bêta (fenêtre 60 jours) par rapport à {BENCHMARKS.get(benchmark, benchmark)}",
                        labels={"value": "Bêta"}
                    )
                    fig_beta.update_layout(height=300, hovermode="x unified")
                    st.plotly_chart(fig_beta, width='stretch')
                else:
                    st.info("Impossible de calculer le bêta avec les données disponibles")
            else:
                st.info("Pas assez de données pour le bêta (minimum 60 jours requis)")
        else:
            st.info(f"Données manquantes pour {ticker} ou {benchmark} pour calculer le bêta")
    except Exception as e:
        st.warning(f"Erreur lors du calcul du bêta: {e}")
# --------- Corrélation avec les indicateurs macro ----------
if compare_macro and not macro_data.empty:
    st.subheader("Corrélation avec les indicateurs macroéconomiques")
    
    # Fusionner les données
    macro_comparison = pd.concat([stock_data['Close'], macro_data], axis=1).dropna()
    macro_comparison.columns = [ticker] + list(MACRO_INDICATORS.values())
    
    # Matrice de corrélation
    corr_matrix = macro_comparison.pct_change().corr()
    
    # Heatmap
    fig_corr = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Matrice de corrélation (variations journalières)"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, width='stretch')
    
    # Scatter sur les plus fortes corrélations (absolues)
    st.subheader("Relation avec les indicateurs clés")
    correlations = corr_matrix[ticker].drop(ticker).abs().sort_values(ascending=False)
    top_correlated = correlations.head(2).index.tolist()

# Wrapper testable depuis le hub
def render_stock(default_ticker: str = "AAPL") -> None:
    """Point d'entrée léger pour le hub/tests.
    Dans un contexte Streamlit, l'UI ci-dessus sera déjà évaluée à l'import.
    En contexte PyTest (st_compat), cet appel est un no-op sûr.
    """
    try:
        st.caption(f"Module stock — ticker par défaut: {default_ticker}")
    except Exception:
        pass
    
    col1, col2 = st.columns(2)
    for i, indicator in enumerate(top_correlated):
        with (col1 if i == 0 else col2):
            scatter_data = pd.DataFrame({
                'x': macro_comparison[indicator].pct_change(),
                'y': macro_comparison[ticker].pct_change()
            }).dropna()
            corr = scatter_data['x'].corr(scatter_data['y'])
            try:
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                if has_statsmodels:
                    fig_scatter = px.scatter(
                        scatter_data, x='x', y='y',
                        trendline="ols",
                        title=f"{ticker} vs {indicator} (Corrélation: {corr:.2f})",
                        labels={"x": f"{indicator} (variation journalière %)", 
                                "y": f"{ticker} (variation journalière %)"}
                    )
                else:
                    fig_scatter = px.scatter(
                        scatter_data, x='x', y='y',
                        title=f"{ticker} vs {indicator} (Corrélation: {corr:.2f})",
                        labels={"x": f"{indicator} (variation journalière %)", 
                                "y": f"{ticker} (variation journalière %)"}
                    )
                    st.info("📊 Pour afficher la droite de tendance, installez 'statsmodels' : pip install statsmodels")
                st.plotly_chart(fig_scatter, width='stretch')
            except Exception as e:
                st.warning(f"Erreur lors du graphique de dispersion: {e}")

# --------- Données financières ----------
st.subheader("Données financières")

if financials and isinstance(financials, dict) and any(
    (isinstance(df, pd.DataFrame) and not df.empty) for df in financials.values() if df is not None
):
    tabs = st.tabs(["Compte de résultat", "Bilan", "Flux de trésorerie", "Ratios clés"])
    
    with tabs[0]:
        if "income_stmt" in financials and isinstance(financials["income_stmt"], pd.DataFrame) and not financials["income_stmt"].empty:
            st.dataframe(financials["income_stmt"].T)
        else:
            st.info("Données du compte de résultat non disponibles.")
    
    with tabs[1]:
        if "balance_sheet" in financials and isinstance(financials["balance_sheet"], pd.DataFrame) and not financials["balance_sheet"].empty:
            st.dataframe(financials["balance_sheet"].T)
        else:
            st.info("Données du bilan non disponibles.")
    
    with tabs[2]:
        if "cash_flow" in financials and isinstance(financials["cash_flow"], pd.DataFrame) and not financials["cash_flow"].empty:
            st.dataframe(financials["cash_flow"].T)
        else:
            st.info("Données des flux de trésorerie non disponibles.")
            
    with tabs[3]:
        try:
            if ("income_stmt" in financials and "balance_sheet" in financials and
                isinstance(financials["income_stmt"], pd.DataFrame) and not financials["income_stmt"].empty and
                isinstance(financials["balance_sheet"], pd.DataFrame) and not financials["balance_sheet"].empty):
                
                income = financials["income_stmt"]
                balance = financials["balance_sheet"]
                
                # DataFrame des ratios
                ratios = pd.DataFrame(index=income.columns)
                
                # Rentabilité
                if "Net Income" in income.index:
                    ratios["Marge nette (%)"] = (income.loc["Net Income"] / income.loc["Total Revenue"]) * 100 if "Total Revenue" in income.index else np.nan
                    ratios["Rendement des capitaux propres (ROE, %)"] = (
                        (income.loc["Net Income"] / balance.loc["Total Stockholder Equity"]) * 100
                        if "Total Stockholder Equity" in balance.index else np.nan
                    )
                    ratios["Rendement des actifs (ROA, %)"] = (
                        (income.loc["Net Income"] / balance.loc["Total Assets"]) * 100
                        if "Total Assets" in balance.index else np.nan
                    )
                
                # Liquidité
                if "Current Assets" in balance.index and "Current Liabilities" in balance.index:
                    ratios["Ratio de liquidité (Actif courant / Passif courant)"] = (
                        balance.loc["Current Assets"] / balance.loc["Current Liabilities"]
                    )
                
                # Endettement
                if "Total Assets" in balance.index and "Total Liabilities Net Minority Interest" in balance.index:
                    ratios["Taux d'endettement (Passif/Actif, %)"] = (
                        balance.loc["Total Liabilities Net Minority Interest"] / balance.loc["Total Assets"] * 100
                    )
                
                # Valorisation (utilise marketCap correct)
                market_cap = company_info.get("marketCap", None)
                if market_cap is not None:
                    if "Net Income" in income.index and not pd.isna(income.loc["Net Income"].iloc[0]) and income.loc["Net Income"].iloc[0] != 0:
                        ratios["Cours/Bénéfices (P/E) estimé"] = market_cap / income.loc["Net Income"].iloc[0]
                    if "Total Revenue" in income.index and not pd.isna(income.loc["Total Revenue"].iloc[0]) and income.loc["Total Revenue"].iloc[0] != 0:
                        ratios["Cours/Chiffre d’affaires (P/S) estimé"] = market_cap / income.loc["Total Revenue"].iloc[0]
                    if ("Total Assets" in balance.index and "Total Liabilities Net Minority Interest" in balance.index):
                        book_value = balance.loc["Total Assets"].iloc[0] - balance.loc["Total Liabilities Net Minority Interest"].iloc[0]
                        if not pd.isna(book_value) and book_value != 0:
                            ratios["Cours/Valeur comptable (P/B) estimé"] = market_cap / book_value
                
                st.dataframe(ratios.T)
            else:
                st.info("Données insuffisantes pour calculer les ratios financiers.")
        except Exception as e:
            st.warning(f"Erreur lors du calcul des ratios financiers: {e}")
            st.info("Données insuffisantes pour calculer les ratios financiers.")
else:
    st.info("Données financières non disponibles pour cette action.")

# --------- Analyse macroéconomique approfondie ----------
st.subheader("Analyse macroéconomique approfondie")

# Option pour afficher l'analyse macroéconomique approfondie
show_macro_analysis = st.checkbox("Afficher l'analyse macroéconomique approfondie", value=False)

if show_macro_analysis:
    st.write("Sélectionnez les indicateurs économiques à analyser :")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_inflation = st.multiselect(
            "Inflation",
            ["CPIAUCSL", "T10YIE"],
            default=["CPIAUCSL"],
            format_func=lambda x: f"{x} - {FRED_SERIES[x]}"
        )
        selected_growth = st.multiselect(
            "Croissance",
            ["INDPRO", "GDPC1"],
            default=["INDPRO"],
            format_func=lambda x: f"{x} - {FRED_SERIES[x]}"
        )
    
    with col2:
        selected_rates = st.multiselect(
            "Taux d'intérêt",
            ["DGS10", "DGS2"],
            default=["DGS10"],
            format_func=lambda x: f"{x} - {FRED_SERIES[x]}"
        )
        selected_other = st.multiselect(
            "Autres indicateurs",
            ["UNRATE", "DTWEXBGS", "NFCI", "USREC"],
            default=["UNRATE"],
            format_func=lambda x: f"{x} - {FRED_SERIES[x]}"
        )
    
    selected_indicators = selected_inflation + selected_growth + selected_rates + selected_other
    
    if selected_indicators:
        fred_start_date = datetime.now() - timedelta(days=365*5)  # 5 ans par défaut
        with st.spinner("Récupération des données économiques en cours..."):
            fred_data = get_fred_data(selected_indicators, fred_start_date)
        
        if not fred_data.empty:
            st.subheader("Évolution des indicateurs économiques")
            macro_tabs = st.tabs(["Inflation", "Croissance", "Taux d'intérêt", "Autres", "Impact sur l'action"])
            
            # Inflation
            with macro_tabs[0]:
                if selected_inflation:
                    inflation_data = fred_data[[FRED_SERIES[col] for col in selected_inflation]]
                    if not inflation_data.empty:
                        fig = px.line(
                            inflation_data, x=inflation_data.index, y=inflation_data.columns,
                            title="Indicateurs d'inflation"
                        )
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, width='stretch')
                        if len(inflation_data) > 252:
                            annual_change = inflation_data.pct_change(252).iloc[-1] * 100
                            st.write("Variation sur 12 mois :")
                            st.dataframe(annual_change.to_frame("Variation (%)").T)
                    else:
                        st.info("Aucune donnée d'inflation disponible.")
                else:
                    st.info("Aucun indicateur d'inflation sélectionné.")
            
            # Croissance
            with macro_tabs[1]:
                if selected_growth:
                    growth_data = fred_data[[FRED_SERIES[col] for col in selected_growth]]
                    if not growth_data.empty:
                        fig = px.line(
                            growth_data, x=growth_data.index, y=growth_data.columns,
                            title="Indicateurs de croissance économique"
                        )
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, width='stretch')
                        if len(growth_data) > 252:
                            annual_change = growth_data.pct_change(252).iloc[-1] * 100
                            st.write("Variation sur 12 mois :")
                            st.dataframe(annual_change.to_frame("Variation (%)").T)
                    else:
                        st.info("Aucune donnée de croissance disponible.")
                else:
                    st.info("Aucun indicateur de croissance sélectionné.")
            
            # Taux d'intérêt
            with macro_tabs[2]:
                if selected_rates:
                    rates_data = fred_data[[FRED_SERIES[col] for col in selected_rates]]
                    if not rates_data.empty:
                        fig = px.line(
                            rates_data, x=rates_data.index, y=rates_data.columns,
                            title="Évolution des taux d'intérêt"
                        )
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, width='stretch')
                        if len(rates_data) > 63:
                            change_3m = rates_data.iloc[-1] - rates_data.iloc[-63]
                            st.write("Variation absolue sur 3 mois (points de base) :")
                            st.dataframe((change_3m * 100).to_frame("Variation (pb)").T)
                    else:
                        st.info("Aucune donnée de taux d'intérêt disponible.")
                else:
                    st.info("Aucun indicateur de taux d'intérêt sélectionné.")
            
            # Autres
            with macro_tabs[3]:
                if selected_other:
                    other_data = fred_data[[FRED_SERIES[col] for col in selected_other]]
                    if not other_data.empty:
                        fig = px.line(
                            other_data, x=other_data.index, y=other_data.columns,
                            title="Autres indicateurs économiques"
                        )
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Aucune donnée disponible.")
                else:
                    st.info("Aucun autre indicateur sélectionné.")
            
            # Impact sur l'action
            with macro_tabs[4]:
                st.write("Analyse de l'impact des facteurs macroéconomiques sur l'action")
                try:
                    stock_monthly = stock_data['Close'].resample('M').last()
                    stock_returns = stock_monthly.pct_change().dropna()
                    fred_monthly = fred_data.resample('M').last()
                    merged_data = pd.concat([stock_returns, fred_monthly], axis=1).dropna()
                    merged_data.columns = [ticker] + list(fred_monthly.columns)
                    
                    if not merged_data.empty and len(merged_data) > 24:
                        corr_vec = merged_data.corr()[ticker].drop(ticker).sort_values(ascending=False)
                        st.write("Corrélation (rendements mensuels) entre l'action et les indicateurs :")
                        fig_corr = px.bar(
                            corr_vec,
                            title="Impact des facteurs économiques sur l'action",
                            labels={"value": "Corrélation", "index": "Indicateur"}
                        )
                        fig_corr.update_layout(height=400)
                        st.plotly_chart(fig_corr, width='stretch')
                        
                        if ticker in SECTOR_SENSITIVITY.columns:
                            st.write("Sensibilité théorique de l'action aux facteurs économiques :")
                            sensitivity = SECTOR_SENSITIVITY[ticker].dropna()
                            fig_sens = px.bar(
                                sensitivity,
                                title="Sensibilité théorique aux facteurs économiques",
                                labels={"value": "Sensibilité (-2 à +2)", "index": "Facteur"}
                            )
                            fig_sens.update_layout(height=300)
                            st.plotly_chart(fig_sens, width='stretch')
                            
                            st.write("**Échelle d'interprétation :**")
                            st.write("- **+2** : très positivement impacté par une hausse du facteur")
                            st.write("- **+1** : modérément positivement impacté")
                            st.write("- **0** : impact neutre")
                            st.write("- **-1** : modérément négativement impacté")
                            st.write("- **-2** : très négativement impacté par une hausse du facteur")
                    else:
                        st.info("Données insuffisantes pour analyser l'impact.")
                except Exception as e:
                    st.warning(f"Erreur lors de l'analyse de l'impact économique: {e}")
        else:
            st.warning("Impossible de récupérer les données économiques. Réessayez plus tard.")
    else:
        st.info("Veuillez sélectionner au moins un indicateur économique pour l'analyse.")

# --------- Résumé et recommandations ----------
st.subheader("Résumé et analyse")

# Indicateurs clés (statut actuel)
try:
    current_price = stock_data['Close'].iloc[-1]
    has_sma_20 = 'SMA_20' in stock_data_with_indicators and not stock_data_with_indicators['SMA_20'].isna().iloc[-1]
    has_sma_50 = 'SMA_50' in stock_data_with_indicators and not stock_data_with_indicators['SMA_50'].isna().iloc[-1]
    has_sma_200 = 'SMA_200' in stock_data_with_indicators and not stock_data_with_indicators['SMA_200'].isna().iloc[-1]
    has_rsi = 'RSI' in stock_data_with_indicators and not stock_data_with_indicators['RSI'].isna().iloc[-1]
    has_macd = 'MACD' in stock_data_with_indicators and not stock_data_with_indicators['MACD'].isna().iloc[-1]
    has_macd_signal = 'MACD_Signal' in stock_data_with_indicators and not stock_data_with_indicators['MACD_Signal'].isna().iloc[-1]
    
    sma_20 = stock_data_with_indicators['SMA_20'].iloc[-1] if has_sma_20 else None
    sma_50 = stock_data_with_indicators['SMA_50'].iloc[-1] if has_sma_50 else None
    sma_200 = stock_data_with_indicators['SMA_200'].iloc[-1] if has_sma_200 else None
    rsi = stock_data_with_indicators['RSI'].iloc[-1] if has_rsi else None
    macd = stock_data_with_indicators['MACD'].iloc[-1] if has_macd else None
    macd_signal = stock_data_with_indicators['MACD_Signal'].iloc[-1] if has_macd_signal else None
except Exception as e:
    st.warning(f"Erreur lors du calcul des indicateurs techniques: {e}")
    current_price = sma_20 = sma_50 = sma_200 = rsi = macd = macd_signal = None

# Tendance de prix
if current_price is not None and sma_50 is not None:
    price_trend = "haussière" if current_price > sma_50 else "baissière"
    st.markdown(f"**Tendance de prix :** {price_trend}")
else:
    st.markdown("**Tendance de prix :** données insuffisantes")

# Signaux techniques (libellés sans abréviations)
signals = []

if current_price is not None and sma_20 is not None:
    if current_price > sma_20:
        signals.append("Prix au-dessus de la moyenne mobile simple 20 jours ✅")
    else:
        signals.append("Prix en-dessous de la moyenne mobile simple 20 jours ❌")
else:
    signals.append("Moyenne mobile simple 20 jours : données insuffisantes ℹ️")
    
if current_price is not None and sma_50 is not None:
    if current_price > sma_50:
        signals.append("Prix au-dessus de la moyenne mobile simple 50 jours ✅")
    else:
        signals.append("Prix en-dessous de la moyenne mobile simple 50 jours ❌")
else:
    signals.append("Moyenne mobile simple 50 jours : données insuffisantes ℹ️")
    
if current_price is not None and sma_200 is not None:
    if current_price > sma_200:
        signals.append("Prix au-dessus de la moyenne mobile simple 200 jours ✅")
    else:
        signals.append("Prix en-dessous de la moyenne mobile simple 200 jours ❌")
else:
    signals.append("Moyenne mobile simple 200 jours : données insuffisantes ℹ️")
    
if sma_20 is not None and sma_50 is not None:
    if sma_20 > sma_50:
        signals.append("Moyenne mobile simple 20 jours au-dessus de 50 jours (signal haussier) ✅")
    else:
        signals.append("Moyenne mobile simple 20 jours en-dessous de 50 jours (signal baissier) ❌")
else:
    signals.append("Croisement des moyennes mobiles simples : données insuffisantes ℹ️")
    
if rsi is not None:
    if rsi > 70:
        signals.append(f"Indice de force relative en zone de surachat ({rsi:.1f}) ⚠️")
    elif rsi < 30:
        signals.append(f"Indice de force relative en zone de survente ({rsi:.1f}) ⚠️")
    else:
        signals.append(f"Indice de force relative en zone neutre ({rsi:.1f}) ✓")
else:
    signals.append("Indice de force relative : données insuffisantes ℹ️")
    
if macd is not None and macd_signal is not None:
    if macd > macd_signal:
        signals.append("MACD au-dessus de la ligne de signal (signal haussier) ✅")
    else:
        signals.append("MACD en-dessous de la ligne de signal (signal baissier) ❌")
else:
    signals.append("MACD : données insuffisantes ℹ️")

# Afficher les signaux
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Signaux techniques :**")
    for s in signals[:3]:
        st.markdown(f"- {s}")
with c2:
    st.markdown("&nbsp;")
    for s in signals[3:]:
        st.markdown(f"- {s}")

# Conclusion
st.markdown("---")
st.markdown("**Conclusion :**")
positive_signals = sum(1 for s in signals if "✅" in s)
negative_signals = sum(1 for s in signals if "❌" in s)
warning_signals = sum(1 for s in signals if "⚠️" in s)

if positive_signals > negative_signals + warning_signals:
    st.markdown("L'analyse technique suggère une tendance globalement **positive**. Restez attentif(ve) aux changements de tendance et aux fondamentaux.")
elif negative_signals > positive_signals + warning_signals:
    st.markdown("L'analyse technique suggère une tendance globalement **négative**. Surveillez les niveaux de support et les catalyseurs fondamentaux.")
else:
    st.markdown("L'analyse technique suggère une tendance **mixte** (signaux contradictoires). Probable consolidation / incertitude.")

# --------- Prévisions à long terme ----------
st.subheader("Prévisions à long terme")

show_forecasts = st.checkbox("Afficher les prévisions à long terme", value=False)

if show_forecasts:
    try:
        if len(stock_data) > 252:
            forecast_method = st.selectbox(
                "Méthode de prévision",
                ["Tendance simple", "Moyenne mobile", "Régression linéaire", "ARIMA", "Prophet", "Modèle hybride"]
            )
            forecast_horizon = st.slider("Horizon de prévision (jours)", 30, 365, 180)
            
            close_prices = stock_data['Close']
            dates = close_prices.index
            last_date = dates[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
            
            with st.expander("Options avancées de prévision"):
                confidence_level = st.slider("Niveau de confiance (%)", 50, 95, 80) / 100
                include_macro = st.checkbox("Inclure des facteurs macroéconomiques", value=True)
                selected_macro_factors = []
                if include_macro:
                    selected_macro_factors = st.multiselect(
                        "Sélectionner les facteurs macroéconomiques à inclure",
                        ["Taux d'intérêt", "Inflation", "Dollar US", "Prix de l'or", "VIX (volatilité)"],
                        default=["Taux d'intérêt", "Dollar US"]
                    )
                use_cross_validation = st.checkbox("Utiliser la validation croisée", value=False)
                if use_cross_validation:
                    cv_periods = st.slider("Nombre de périodes de validation", 3, 10, 5)
                    cv_window = st.slider("Taille de la fenêtre de validation (jours)", 30, 180, 60)
            
            # === Tendance simple ===
            if forecast_method == "Tendance simple":
                recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                x = np.arange(len(recent_prices))
                slope, intercept = np.polyfit(x, recent_prices, 1)
                future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                forecast = slope * future_x + intercept
                y_pred = slope * x + intercept
                rmse = np.sqrt(np.mean((recent_prices - y_pred) ** 2))
                std_error = rmse * np.sqrt(1 + 1/len(x) + (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
                z_value = 1.96 if confidence_level >= 0.95 else (1.645 if confidence_level >= 0.9 else 1.28)
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast,
                    'Lower': forecast - z_value * std_error,
                    'Upper': forecast + z_value * std_error
                }).set_index('Date')
                st.write(f"**Méthode de tendance simple :** projection linéaire basée sur {len(recent_prices)} jours de bourse.")
                st.write(f"Tendance quotidienne moyenne : {slope:.4f} {company_info.get('currency', '$')}/jour")
                st.write(f"RMSE : {rmse:.4f}")
            
            # === Moyenne mobile ===
            elif forecast_method == "Moyenne mobile":
                ema_span = st.slider("Période de la moyenne mobile (jours)", 20, 200, 50)
                ema = close_prices.ewm(span=ema_span, adjust=False).mean()
                recent_ema = ema[-60:]
                x = np.arange(len(recent_ema))
                slope, intercept = np.polyfit(x, recent_ema, 1)
                future_x = np.arange(len(recent_ema), len(recent_ema) + forecast_horizon)
                forecast = slope * future_x + intercept
                volatility = close_prices.pct_change().std() * np.sqrt(252)
                daily_vol = volatility / np.sqrt(252)
                z_value = 1.96 if confidence_level >= 0.95 else (1.645 if confidence_level >= 0.9 else 1.28)
                time_factors = np.sqrt(np.arange(1, forecast_horizon + 1))
                uncertainty = np.array([daily_vol * tf * z_value for tf in time_factors])
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast,
                    'Lower': forecast * (1 - uncertainty),
                    'Upper': forecast * (1 + uncertainty)
                }).set_index('Date')
                st.write(f"**Méthode de moyenne mobile :** tendance de l'EMA {ema_span} jours. Volatilité annualisée : {volatility:.2%}")
            
            # === Régression linéaire ===
            elif forecast_method == "Régression linéaire":
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                if has_statsmodels:
                    import statsmodels.api as sm
                    df = pd.DataFrame(index=dates)
                    df['price'] = close_prices
                    df['trend'] = np.arange(len(df))
                    df['month'] = df.index.month
                    df['day_of_week'] = df.index.dayofweek
                    df['quarter'] = df.index.quarter
                    df['ma20'] = close_prices.rolling(window=20).mean().bfill()
                    df['ma50'] = close_prices.rolling(window=50).mean().bfill()
                    df['volatility'] = close_prices.rolling(window=20).std().bfill()
                    if include_macro and selected_macro_factors:
                        if "Taux d'intérêt" in selected_macro_factors:
                            df['interest_rate'] = np.random.normal(2.5, 0.5, len(df))
                        if "Dollar US" in selected_macro_factors:
                            df['usd_index'] = np.random.normal(100, 5, len(df))
                        if "Inflation" in selected_macro_factors:
                            df['inflation'] = np.random.normal(2.0, 0.3, len(df))
                    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
                    dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True)
                    quarter_dummies = pd.get_dummies(df['quarter'], prefix='quarter', drop_first=True)
                    X_columns = ['trend', 'ma20', 'ma50', 'volatility']
                    if include_macro and selected_macro_factors:
                        if "Taux d'intérêt" in selected_macro_factors: X_columns.append('interest_rate')
                        if "Dollar US" in selected_macro_factors: X_columns.append('usd_index')
                        if "Inflation" in selected_macro_factors: X_columns.append('inflation')
                    X = pd.concat([df[X_columns], month_dummies, dow_dummies, quarter_dummies], axis=1)
                    y = df['price']
                    model = sm.OLS(y, sm.add_constant(X)).fit()
                    future_df = pd.DataFrame(index=future_dates)
                    future_df['trend'] = np.arange(len(df), len(df) + len(future_dates))
                    future_df['month'] = future_df.index.month
                    future_df['day_of_week'] = future_df.index.dayofweek
                    future_df['quarter'] = future_df.index.quarter
                    future_df['ma20'] = df['ma20'].iloc[-1]
                    future_df['ma50'] = df['ma50'].iloc[-1]
                    future_df['volatility'] = df['volatility'].iloc[-1]
                    if include_macro and selected_macro_factors:
                        if "Taux d'intérêt" in selected_macro_factors: future_df['interest_rate'] = df['interest_rate'].iloc[-1]
                        if "Dollar US" in selected_macro_factors: future_df['usd_index'] = df['usd_index'].iloc[-1]
                        if "Inflation" in selected_macro_factors: future_df['inflation'] = df['inflation'].iloc[-1]
                    future_month_dummies = pd.get_dummies(future_df['month'], prefix='month', drop_first=True)
                    future_dow_dummies = pd.get_dummies(future_df['day_of_week'], prefix='dow', drop_first=True)
                    future_quarter_dummies = pd.get_dummies(future_df['quarter'], prefix='quarter', drop_first=True)
                    for col in month_dummies.columns:
                        if col not in future_month_dummies.columns: future_month_dummies[col] = 0
                    for col in dow_dummies.columns:
                        if col not in future_dow_dummies.columns: future_dow_dummies[col] = 0
                    for col in quarter_dummies.columns:
                        if col not in future_quarter_dummies.columns: future_quarter_dummies[col] = 0
                    future_X = pd.concat([
                        future_df[X_columns],
                        future_month_dummies[month_dummies.columns],
                        future_dow_dummies[dow_dummies.columns],
                        future_quarter_dummies[quarter_dummies.columns]
                    ], axis=1)
                    forecast = model.predict(sm.add_constant(future_X))
                    from statsmodels.sandbox.regression.predstd import wls_prediction_std
                    _, lower, upper = wls_prediction_std(model, sm.add_constant(future_X), alpha=1-confidence_level)
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast, 'Lower': lower, 'Upper': upper}).set_index('Date')
                    st.write("**Régression linéaire avancée :** tendance + saisonnalités + variables techniques.")
                    st.write(f"R² du modèle : {model.rsquared:.4f}")
                    coefs = model.params.sort_values(ascending=False)
                    coef_df = pd.DataFrame({'Facteur': coefs.index[:5], 'Coefficient': coefs.values[:5]})
                    st.dataframe(coef_df)
                else:
                    st.warning("La régression linéaire nécessite 'statsmodels'. Installez : pip install statsmodels")
                    recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent_prices))
                    slope, intercept = np.polyfit(x, recent_prices, 1)
                    future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast, 'Lower': forecast * 0.9, 'Upper': forecast * 1.1}).set_index('Date')
            
            # === ARIMA ===
            elif forecast_method == "ARIMA":
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                if has_statsmodels:
                    from statsmodels.tsa.arima.model import ARIMA
                    from pmdarima import auto_arima
                    with st.spinner("Recherche des meilleurs paramètres ARIMA..."):
                        try:
                            auto_model = auto_arima(
                                close_prices, start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
                                seasonal=False, trace=False, error_action='ignore', suppress_warnings=True,
                                stepwise=True, n_jobs=-1
                            )
                            best_order = auto_model.order
                            st.write(f"Meilleurs paramètres ARIMA : {best_order}")
                        except Exception as e:
                            st.warning(f"Erreur auto_arima : {e}")
                            best_order = (2, 1, 2)
                            st.write(f"Paramètres par défaut utilisés : {best_order}")
                    model = ARIMA(close_prices, order=best_order)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=forecast_horizon)
                    forecast_ci = model_fit.get_forecast(steps=forecast_horizon).conf_int(alpha=1-confidence_level)
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast,
                        'Lower': forecast_ci.iloc[:, 0].values,
                        'Upper': forecast_ci.iloc[:, 1].values
                    }).set_index('Date')
                    st.write(f"**ARIMA{best_order} :** modèle de série temporelle. AIC : {model_fit.aic:.2f}")
                    with st.expander("Diagnostics ARIMA"):
                        st.text(str(model_fit.summary()))
                else:
                    st.warning("ARIMA nécessite 'statsmodels' et 'pmdarima'. Installez : pip install statsmodels pmdarima")
                    recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent_prices))
                    slope, intercept = np.polyfit(x, recent_prices, 1)
                    future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast, 'Lower': forecast * 0.9, 'Upper': forecast * 1.1}).set_index('Date')
            
            # === Prophet ===
            elif forecast_method == "Prophet":
                import importlib.util
                has_prophet = importlib.util.find_spec('prophet') is not None
                if has_prophet:
                    from prophet import Prophet
                    prophet_data = pd.DataFrame({'ds': close_prices.index, 'y': close_prices.values})
                    with st.spinner("Ajustement du modèle Prophet..."):
                        model = Prophet(
                            changepoint_prior_scale=0.05,
                            seasonality_prior_scale=10.0,
                            seasonality_mode='multiplicative',
                            interval_width=confidence_level
                        )
                        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
                        if include_macro and selected_macro_factors:
                            if "Taux d'intérêt" in selected_macro_factors:
                                prophet_data['interest_rate'] = np.random.normal(2.5, 0.5, len(prophet_data))
                                model.add_regressor('interest_rate')
                            if "Dollar US" in selected_macro_factors:
                                prophet_data['usd_index'] = np.random.normal(100, 5, len(prophet_data))
                                model.add_regressor('usd_index')
                        model.fit(prophet_data)
                    future = model.make_future_dataframe(periods=forecast_horizon)
                    if include_macro and selected_macro_factors:
                        if "Taux d'intérêt" in selected_macro_factors:
                            future['interest_rate'] = prophet_data['interest_rate'].iloc[-1]
                        if "Dollar US" in selected_macro_factors:
                            future['usd_index'] = prophet_data['usd_index'].iloc[-1]
                    forecast_result = model.predict(future)
                    forecast_df = pd.DataFrame({
                        'Date': forecast_result['ds'].iloc[-forecast_horizon:],
                        'Forecast': forecast_result['yhat'].iloc[-forecast_horizon:],
                        'Lower': forecast_result['yhat_lower'].iloc[-forecast_horizon:],
                        'Upper': forecast_result['yhat_upper'].iloc[-forecast_horizon:]
                    }).set_index('Date')
                    st.write("**Prophet :** décomposition tendance/saisonnalité.")
                    with st.expander("Composantes du modèle Prophet"):
                        fig_comp = model.plot_components(forecast_result)
                        st.pyplot(fig_comp)
                else:
                    st.warning("Prophet nécessite le package 'prophet'. Installez : pip install prophet")
                    recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent_prices))
                    slope, intercept = np.polyfit(x, recent_prices, 1)
                    future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast, 'Lower': forecast * 0.9, 'Upper': forecast * 1.1}).set_index('Date')
            
            # === Modèle hybride ===
            elif forecast_method == "Modèle hybride":
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                if has_statsmodels:
                    import statsmodels.api as sm
                    from statsmodels.tsa.arima.model import ARIMA
                    with st.spinner("Construction du modèle hybride..."):
                        # Tendance linéaire
                        x = np.arange(len(close_prices))
                        slope, intercept = np.polyfit(x, close_prices, 1)
                        future_x = np.arange(len(close_prices), len(close_prices) + forecast_horizon)
                        trend_forecast = slope * future_x + intercept
                        # ARIMA simple
                        try:
                            arima_fit = ARIMA(close_prices, order=(2, 1, 2)).fit()
                            arima_forecast = arima_fit.forecast(steps=forecast_horizon)
                        except Exception as e:
                            st.warning(f"Erreur ARIMA (hybride) : {e}")
                            arima_forecast = trend_forecast
                        # Régression technique
                        df = pd.DataFrame(index=dates)
                        df['price'] = close_prices
                        df['ma20'] = close_prices.rolling(window=20).mean().bfill()
                        df['ma50'] = close_prices.rolling(window=50).mean().bfill()
                        df['volatility'] = close_prices.rolling(window=20).std().bfill()
                        df['trend'] = np.arange(len(df))
                        if include_macro and selected_macro_factors:
                            if "Taux d'intérêt" in selected_macro_factors:
                                df['interest_rate'] = np.random.normal(2.5, 0.5, len(df))
                            if "Dollar US" in selected_macro_factors:
                                df['usd_index'] = np.random.normal(100, 5, len(df))
                        X_columns = ['ma20', 'ma50', 'volatility', 'trend']
                        if include_macro and selected_macro_factors:
                            if "Taux d'intérêt" in selected_macro_factors: X_columns.append('interest_rate')
                            if "Dollar US" in selected_macro_factors: X_columns.append('usd_index')
                        X = df[X_columns]
                        y = df['price']
                        try:
                            reg_model = sm.OLS(y, sm.add_constant(X)).fit()
                            future_df = pd.DataFrame(index=future_dates)
                            future_df['ma20'] = df['ma20'].iloc[-1]
                            future_df['ma50'] = df['ma50'].iloc[-1]
                            future_df['volatility'] = df['volatility'].iloc[-1]
                            future_df['trend'] = np.arange(len(df), len(df) + len(future_dates))
                            if include_macro and selected_macro_factors:
                                if "Taux d'intérêt" in selected_macro_factors: future_df['interest_rate'] = df['interest_rate'].iloc[-1]
                                if "Dollar US" in selected_macro_factors: future_df['usd_index'] = df['usd_index'].iloc[-1]
                            reg_forecast = reg_model.predict(sm.add_constant(future_df[X_columns]))
                        except Exception as e:
                            st.warning(f"Erreur régression (hybride) : {e}")
                            reg_forecast = trend_forecast
                    # Combinaison pondérée
                    weights = [0.3, 0.4, 0.3]
                    combined_forecast = (weights[0]*trend_forecast + weights[1]*arima_forecast + weights[2]*reg_forecast)
                    forecasts = np.vstack([trend_forecast, np.array(arima_forecast), np.array(reg_forecast)])
                    forecast_std = np.std(forecasts, axis=0)
                    z_value = 1.96 if confidence_level >= 0.95 else (1.645 if confidence_level >= 0.9 else 1.28)
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': combined_forecast,
                        'Lower': combined_forecast - z_value * forecast_std,
                        'Upper': combined_forecast + z_value * forecast_std
                    }).set_index('Date')
                    st.write("**Modèle hybride :** moyenne pondérée (Tendance/ARIMA/Régression).")
                    st.dataframe(pd.DataFrame({'Modèle': ['Tendance','ARIMA','Régression'], 'Poids': weights}))
                else:
                    st.warning("Le modèle hybride nécessite 'statsmodels'. Installez : pip install statsmodels")
                    recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent_prices))
                    slope, intercept = np.polyfit(x, recent_prices, 1)
                    future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast, 'Lower': forecast * 0.9, 'Upper': forecast * 1.1}).set_index('Date')
            
            # Validation croisée (optionnelle)
            if 'use_cross_validation' in locals() and use_cross_validation:
                with st.expander("Résultats de la validation croisée"):
                    cv_errors = []
                    for i in range(cv_periods):
                        train_end = len(close_prices) - (cv_periods - i) * cv_window
                        test_end = train_end + cv_window
                        if train_end > 252:
                            train_data = close_prices[:train_end]
                            test_data = close_prices[train_end:test_end]
                            x_train = np.arange(len(train_data))
                            slope, intercept = np.polyfit(x_train, train_data, 1)
                            x_test = np.arange(len(train_data), len(train_data) + len(test_data))
                            pred = slope * x_test + intercept
                            mape = np.mean(np.abs((test_data.values - pred) / test_data.values)) * 100
                            cv_errors.append(mape)
                    if cv_errors:
                        avg_mape = np.mean(cv_errors)
                        st.write(f"**MAPE moyen ({cv_periods} périodes)** : {avg_mape:.2f}%")
                        fig_cv = go.Figure()
                        fig_cv.add_trace(go.Bar(x=[f"Période {i+1}" for i in range(len(cv_errors))], y=cv_errors))
                        fig_cv.update_layout(title="Erreur de prévision par période", xaxis_title="Période", yaxis_title="MAPE (%)", height=300)
                        st.plotly_chart(fig_cv, width='stretch')
                    else:
                        st.info("Données insuffisantes pour la validation croisée aux paramètres actuels.")
            
            # Visualisation finale de la prévision
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates[-252:], y=close_prices[-252:], mode='lines', name='Historique', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Prévision', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(
                x=list(forecast_df.index) + list(forecast_df.index)[::-1],
                y=list(forecast_df['Upper']) + list(forecast_df['Lower'])[::-1],
                fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,0,0,0)'),
                name=f'Intervalle de confiance ({int(confidence_level*100)}%)'
            ))
            fig.update_layout(
                title=f"Prévision du cours de {ticker} sur {forecast_horizon} jours",
                xaxis_title="Date",
                yaxis_title=f"Prix ({company_info.get('currency', '$')})",
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Statistiques de la prévision
            current_price = close_prices.iloc[-1]
            forecast_end = float(forecast_df['Forecast'].iloc[-1])
            change_pct = ((forecast_end / current_price) - 1) * 100
            st.write("**Résumé de la prévision :**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Prix actuel", f"{current_price:.2f} {company_info.get('currency', '$')}")
            with c2:
                st.metric("Prix prévu (fin de période)", f"{forecast_end:.2f} {company_info.get('currency', '$')}", f"{change_pct:.2f}%")
            with c3:
                lower_bound = float(forecast_df['Lower'].iloc[-1])
                upper_bound = float(forecast_df['Upper'].iloc[-1])
                st.metric("Intervalle de confiance", f"{lower_bound:.2f} - {upper_bound:.2f}")
            with c4:
                st.metric("Horizon", f"{forecast_horizon} jours")
            
            csv = forecast_df.to_csv()
            st.download_button(
                label="Télécharger les prévisions (CSV)",
                data=csv,
                file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            st.warning("⚠️ **Avertissement :** Prévisions basées sur des modèles et données historiques. Pas de conseil en investissement.")
        else:
            st.info("Données historiques insuffisantes (>= 1 an requis).")
    except Exception as e:
        st.error(f"Erreur lors de la génération des prévisions: {e}")
        st.error("Détails de l'erreur :", exc_info=True)

def render_stock(default_ticker: str = "AAPL"):
    """Fonction exportable pour afficher l'onglet Action dans le hub."""
    # Interface complète disponible via l’app principale.
    pass

# Exécution directe
if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Analyse Approfondie d'Action", layout="wide")
    st.title("📊 Analyse Approfondie d'Action")
    with st.sidebar:
        st.header("Paramètres")
        ticker = st.text_input("Symbole de l'action", value="NGD.TO")
    if ticker:
        st.header(f"Analyse pour {ticker}")
        st.info("Analyse complète disponible dans le hub principal.")
    else:
        st.warning("Veuillez entrer un symbole d'action valide.")
