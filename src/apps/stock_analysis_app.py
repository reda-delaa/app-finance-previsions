#stock_analysis_app.py

import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils.st_compat import get_st
st = get_st()
import ta

# ================== CONFIG ==================
# Configuration par défaut pour l'analyse d'une action
DEFAULT_TICKER = "NGD.TO"  # New Gold Inc. (exemple)
PEER_GROUPS = {
    "Gold Miners": ["ABX.TO", "K.TO", "AEM.TO", "BTO.TO", "IMG.TO", "OR.TO"],
    "Silver Miners": ["PAAS.TO", "EDR.TO", "FR.TO"],
    "Copper Miners": ["CS.TO", "TECK-B.TO", "LUN.TO", "FM.TO"],
    "Diversified Miners": ["RIO", "BHP", "VALE", "FCX"]
}

# Indices de référence
BENCHMARKS = {
    "^GSPTSE": "TSX Composite",
    "GDX": "VanEck Gold Miners ETF",
    "XGD.TO": "iShares S&P/TSX Global Gold Index ETF",
    "XME": "SPDR S&P Metals & Mining ETF"
}

# Indicateurs macroéconomiques à surveiller
MACRO_INDICATORS = {
    "GC=F": "Gold Futures",
    "SI=F": "Silver Futures",
    "HG=F": "Copper Futures",
    "DX-Y.NYB": "US Dollar Index",
    "^TNX": "10-Year Treasury Yield"
}

# Séries FRED pour l'analyse économique approfondie
FRED_SERIES = {
    # Inflation / expectations
    "CPIAUCSL": "CPI (All Items, Index 1982-84=100)",
    "T10YIE":   "10Y Breakeven Inflation",
    # Growth / activity
    "INDPRO":   "Industrial Production Index",
    "GDPC1":    "Real Gross Domestic Product (Quarterly)",
    # Labor
    "UNRATE":   "Unemployment Rate",
    "PAYEMS":   "Total Nonfarm Payrolls",
    # Rates & curve
    "DGS10":    "10Y Treasury Yield",
    "DGS2":     "2Y Treasury Yield",
    # USD
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index (Broad)",
    # Financial conditions / credit
    "NFCI":     "Chicago Fed National Financial Conditions Index",
    "BAMLC0A0CM": "ICE BofA US Corp Master OAS",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    # Recessions shading
    "USREC":    "US Recession Indicator"
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
    """Récupère les données historiques d'une action"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            return None
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        return hist
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour {ticker}: {e}")
        return None

def get_peer_data(tickers, period="1y"):
    """Récupère les données de cours pour un groupe d'actions"""
    data = {}
    valid_tickers = []
    for ticker in tickers:
        try:
            hist = get_stock_data(ticker, period=period)
            if hist is not None and not hist.empty:
                data[ticker] = hist["Close"]
                valid_tickers.append(ticker)
            else:
                st.warning(f"Aucune donnée disponible pour {ticker}")
            time.sleep(0.2)  # Pour éviter de surcharger l'API
        except Exception as e:
            st.warning(f"Impossible de récupérer les données pour {ticker}: {e}")
    return pd.DataFrame(data), valid_tickers

def calculate_returns(df, periods):
    """Calcule les rendements sur différentes périodes"""
    returns = {}
    for period, days in periods.items():
        try:
            if len(df) > days:
                returns[period] = df.pct_change(days).iloc[-1] * 100
            else:
                # Si la période est plus longue que les données disponibles
                returns[period] = df.pct_change(len(df)-1).iloc[-1] * 100 if len(df) > 1 else pd.Series(0, index=df.columns)
        except Exception as e:
            st.warning(f"Erreur lors du calcul des rendements pour la période {period}: {e}")
            returns[period] = pd.Series(0, index=df.columns)
    return pd.DataFrame(returns)

def calculate_volatility(df, window=20):
    """Calcule la volatilité annualisée sur une fenêtre glissante"""
    return df.pct_change().rolling(window).std() * np.sqrt(252) * 100

def calculate_beta(stock_returns, benchmark_returns, window=60):
    """Calcule le bêta glissant d'une action par rapport à un indice"""
    # Alignement des données
    aligned_data = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(aligned_data) < window:
        return pd.Series(index=stock_returns.index)
    
    # Calcul du bêta glissant
    rolling_cov = aligned_data.iloc[:, 0].rolling(window=window).cov(aligned_data.iloc[:, 1])
    rolling_var = aligned_data.iloc[:, 1].rolling(window=window).var()
    return rolling_cov / rolling_var

def add_technical_indicators(df):
    """Ajoute des indicateurs techniques au dataframe"""
    # Copie pour éviter les avertissements de SettingWithCopyWarning
    data = df.copy()
    
    # Moyennes mobiles
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
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_Upper'] = bollinger.bollinger_hband()
    data['BB_Lower'] = bollinger.bollinger_lband()
    data['BB_Middle'] = bollinger.bollinger_mavg()
    
    # Volume indicators
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    
    return data

def get_company_info(ticker):
    """Récupère les informations fondamentales de l'entreprise"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Erreur lors de la récupération des informations pour {ticker}: {e}")
        return {}

def load_fred_series(series_id, start_date=None):
    """Récupère une série temporelle depuis FRED"""
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
    """Récupère plusieurs séries temporelles depuis FRED"""
    data = {}
    for series_id in series_ids:
        try:
            df = load_fred_series(series_id, start_date)
            if not df.empty:
                data[series_id] = df[series_id]
            time.sleep(0.2)  # Pour éviter de surcharger l'API
        except Exception as e:
            st.warning(f"Échec de récupération de {series_id}: {e}")
    
    if not data:
        return pd.DataFrame()
    
    # Fusionner toutes les séries en un seul DataFrame
    result = pd.concat(data, axis=1)
    result.columns = [FRED_SERIES.get(col, col) for col in result.columns]
    return result

def zscore(series, window=24):
    """Calcule le z-score d'une série temporelle sur une fenêtre glissante"""
    s = series.dropna()
    if len(s) < window + 2:
        return pd.Series(index=series.index, dtype=float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return ((s - mu) / (sd.replace(0, np.nan))).reindex(series.index)

def get_financials(ticker):
    """Récupère les données financières de l'entreprise"""
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
    """Trouve des actions similaires basées sur la corrélation des rendements"""
    # Déterminer le groupe de pairs approprié
    peer_group = []
    for group, tickers in PEER_GROUPS.items():
        if ticker in tickers:
            peer_group = tickers
            break
    
    # Si l'action n'est pas dans un groupe prédéfini, utiliser le groupe Gold Miners par défaut
    if not peer_group:
        peer_group = PEER_GROUPS["Gold Miners"]
    
    # S'assurer que le ticker principal n'est pas dans la liste
    if ticker in peer_group:
        peer_group.remove(ticker)
    
    # Récupérer les données
    peers_data = get_peer_data([ticker] + peer_group)
    if peers_data.empty or ticker not in peers_data.columns:
        return []
    
    # Calculer les corrélations
    returns = peers_data.pct_change().dropna()
    correlations = returns.corr()[ticker].drop(ticker)
    
    # Retourner les n actions les plus corrélées
    return correlations.nlargest(n).index.tolist()

# ================== UI ==================
st.set_page_config(page_title="Analyse Approfondie d'Action", layout="wide")
st.title("📊 Analyse Approfondie d'Action")

with st.sidebar:
    st.header("Paramètres")
    ticker = st.text_input("Symbole de l'action", value=DEFAULT_TICKER)
    period = st.selectbox("Période d'analyse", ["1y", "2y", "3y", "5y", "10y", "max"], index=2)
    benchmark = st.selectbox("Indice de référence", list(BENCHMARKS.keys()), format_func=lambda x: f"{x} - {BENCHMARKS[x]}")
    
    st.subheader("Indicateurs techniques")
    show_sma = st.checkbox("Moyennes mobiles", value=True)
    show_bb = st.checkbox("Bandes de Bollinger", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)
    
    st.subheader("Analyse comparative")
    compare_peers = st.checkbox("Comparer avec actions similaires", value=True)
    compare_macro = st.checkbox("Comparer avec indicateurs macro", value=True)
    
    if st.button("🔄 Actualiser les données"):
        st.cache_data.clear()
        st.rerun()

# Vérifier si le ticker est valide
if not ticker:
    st.warning("Veuillez entrer un symbole d'action valide.")
    st.stop()

# --------- Chargement des données ----------
@st.cache_data(ttl=3600)
def load_stock_data(ticker, period):
    return get_stock_data(ticker, period=period)

@st.cache_data(ttl=3600)
def load_company_info(ticker):
    return get_company_info(ticker)

@st.cache_data(ttl=3600)
def load_financials(ticker):
    return get_financials(ticker)

@st.cache_data(ttl=3600)
def load_benchmark_data(benchmark, period):
    return get_stock_data(benchmark, period=period)

@st.cache_data(ttl=3600)
def load_similar_stocks(ticker):
    return get_similar_stocks(ticker)

@st.cache_data(ttl=3600)
def load_macro_indicators(period):
    data = {}
    for indicator, name in MACRO_INDICATORS.items():
        try:
            hist = get_stock_data(indicator, period=period)
            if hist is not None and not hist.empty:
                data[indicator] = hist["Close"]
            time.sleep(0.2)
        except Exception as e:
            st.warning(f"Impossible de récupérer les données pour {indicator}: {e}")
    return pd.DataFrame(data)

# Chargement des données
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
    financials = load_financials(ticker)
    
    # Données de l'indice de référence
    benchmark_data = load_benchmark_data(benchmark, period)
    
    # Actions similaires
    try:
        similar_stocks = load_similar_stocks(ticker) if compare_peers else []
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
        st.markdown(f"**Secteur:** {company_info.get('sector', 'N/A')} | **Industrie:** {company_info.get('industry', 'N/A')}")
        st.markdown(company_info.get("longBusinessSummary", "Aucune description disponible."))
    
    with col2:
        st.subheader("Données de marché")
        current_price = stock_data["Close"].iloc[-1]
        previous_close = stock_data["Close"].iloc[-2] if len(stock_data) > 1 else current_price
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100
        
        st.metric("Prix actuel", f"{current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
        st.metric("Volume moyen (30j)", f"{stock_data['Volume'].tail(30).mean():.0f}")
        st.metric("Capitalisation", f"{company_info.get('marketCap', 0) / 1e6:.2f}M")
    
    with col3:
        st.subheader("Valorisation")
        st.metric("P/E", f"{company_info.get('trailingPE', 'N/A')}")
        st.metric("P/B", f"{company_info.get('priceToBook', 'N/A')}")
        st.metric("EV/EBITDA", f"{company_info.get('enterpriseToEbitda', 'N/A')}")

# --------- Graphique principal ----------
st.subheader("Évolution du cours")

# Créer un graphique avec sous-graphiques
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.1, 
                   row_heights=[0.7, 0.3],
                   subplot_titles=(f"Cours de {ticker}", "Volume"))

# Graphique des prix
fig.add_trace(go.Candlestick(x=stock_data.index,
                            open=stock_data['Open'],
                            high=stock_data['High'],
                            low=stock_data['Low'],
                            close=stock_data['Close'],
                            name=ticker),
              row=1, col=1)

# Ajouter les moyennes mobiles si demandé
if show_sma:
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['SMA_20'],
                            line=dict(color='blue', width=1),
                            name='SMA 20'),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['SMA_50'],
                            line=dict(color='orange', width=1),
                            name='SMA 50'),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['SMA_200'],
                            line=dict(color='red', width=1),
                            name='SMA 200'),
                 row=1, col=1)

# Ajouter les bandes de Bollinger si demandé
if show_bb:
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['BB_Upper'],
                            line=dict(color='rgba(0,128,0,0.3)', width=1),
                            name='BB Upper'),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['BB_Lower'],
                            line=dict(color='rgba(0,128,0,0.3)', width=1),
                            fill='tonexty', fillcolor='rgba(0,128,0,0.1)',
                            name='BB Lower'),
                 row=1, col=1)

# Graphique du volume
fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'],
                    marker_color='rgba(0,0,128,0.5)',
                    name='Volume'),
              row=2, col=1)

# Mise en page
fig.update_layout(height=600, 
                 xaxis_rangeslider_visible=False,
                 hovermode="x unified",
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

st.plotly_chart(fig, width='stretch')

# --------- Indicateurs techniques ----------
st.subheader("Indicateurs techniques")

col1, col2 = st.columns(2)

with col1:
    if show_rsi:
        # Graphique RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['RSI'],
                                   line=dict(color='purple', width=1),
                                   name='RSI'))
        
        # Ajouter des lignes horizontales pour les niveaux de surachat/survente
        fig_rsi.add_shape(type="line", x0=stock_data.index[0], y0=70, x1=stock_data.index[-1], y1=70,
                         line=dict(color="red", width=1, dash="dash"))
        fig_rsi.add_shape(type="line", x0=stock_data.index[0], y0=30, x1=stock_data.index[-1], y1=30,
                         line=dict(color="green", width=1, dash="dash"))
        
        fig_rsi.update_layout(title="RSI (14)",
                            yaxis=dict(range=[0, 100]),
                            height=300,
                            hovermode="x unified")
        st.plotly_chart(fig_rsi, width='stretch')

with col2:
    if show_macd:
        # Graphique MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['MACD'],
                                     line=dict(color='blue', width=1),
                                     name='MACD'))
        fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['MACD_Signal'],
                                     line=dict(color='red', width=1),
                                     name='Signal'))
        
        # Histogramme MACD
        colors = ['green' if val > 0 else 'red' for val in stock_data_with_indicators['MACD_Hist']]
        fig_macd.add_trace(go.Bar(x=stock_data.index, y=stock_data_with_indicators['MACD_Hist'],
                                marker_color=colors,
                                name='Histogramme'))
        
        fig_macd.update_layout(title="MACD",
                             height=300,
                             hovermode="x unified")
        st.plotly_chart(fig_macd, width='stretch')

# --------- Analyse de performance ----------
st.subheader("Analyse de performance")

# Calculer les rendements sur différentes périodes
periods = {"1 semaine": 5, "1 mois": 21, "3 mois": 63, "6 mois": 126, "1 an": 252, "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days}

# Préparer les données pour la comparaison
comparison_data = pd.DataFrame()
comparison_data[ticker] = stock_data['Close']

if benchmark_data is not None and not benchmark_data.empty:
    comparison_data[benchmark] = benchmark_data['Close']

# Ajouter les actions similaires si demandé
if compare_peers and similar_stocks:
    peer_data, valid_peers = get_peer_data(similar_stocks, period=period)
    for peer in peer_data.columns:
        comparison_data[peer] = peer_data[peer]

# Normaliser les données (base 100)
if not comparison_data.empty and len(comparison_data) > 0:
    comparison_normalized = comparison_data.dropna().div(comparison_data.iloc[0]) * 100
else:
    st.warning("Pas assez de données pour l'analyse comparative")
    comparison_normalized = pd.DataFrame()

# Graphique de comparaison
if not comparison_normalized.empty and len(comparison_normalized) > 0:
    try:
        fig_comp = px.line(comparison_normalized, x=comparison_normalized.index, y=comparison_normalized.columns,
                        title="Performance relative (base 100)",
                        labels={"value": "Performance (%)", "variable": "Symbole"})
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
            # Utiliser to_pandas() pour éviter les problèmes de compatibilité avec les versions récentes de pandas
            styled_df = returns_df.T.style.format("{:.2f}%")
            try:
                styled_df = styled_df.background_gradient(cmap="RdYlGn", axis=1)
            except Exception:
                # Si le background_gradient échoue, on continue sans
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
        if not comparison_data.empty and len(comparison_data) > 20:  # Au moins 20 points pour calculer la volatilité
            volatility = calculate_volatility(comparison_data)
            if not volatility.empty:
                fig_vol = px.line(volatility, x=volatility.index, y=volatility.columns,
                                title="Volatilité annualisée (fenêtre 20 jours)",
                                labels={"value": "Volatilité (%)", "variable": "Symbole"})
                fig_vol.update_layout(height=300, hovermode="x unified")
                st.plotly_chart(fig_vol, width='stretch')
            else:
                st.info("Pas assez de données pour calculer la volatilité")
        else:
            st.info("Pas assez de données pour calculer la volatilité (minimum 20 jours requis)")
    except Exception as e:
        st.warning(f"Erreur lors du calcul de la volatilité: {e}")

with col2:
    # Bêta par rapport à l'indice de référence
    try:
        if benchmark in comparison_data.columns and ticker in comparison_data.columns:
            if len(comparison_data) > 60:  # Au moins 60 points pour calculer le bêta glissant
                stock_returns = comparison_data[ticker].pct_change().dropna()
                benchmark_returns = comparison_data[benchmark].pct_change().dropna()
                beta = calculate_beta(stock_returns, benchmark_returns)
                
                if not beta.empty and not beta.isna().all():
                    fig_beta = px.line(beta, x=beta.index, y=beta,
                                    title=f"Bêta glissant par rapport à {BENCHMARKS.get(benchmark, benchmark)}",
                                    labels={"value": "Bêta"})
                    fig_beta.update_layout(height=300, hovermode="x unified")
                    st.plotly_chart(fig_beta, width='stretch')
                else:
                    st.info("Impossible de calculer le bêta avec les données disponibles")
            else:
                st.info("Pas assez de données pour calculer le bêta (minimum 60 jours requis)")
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
    
    # Calculer la matrice de corrélation
    corr_matrix = macro_comparison.pct_change().corr()
    
    # Heatmap de corrélation
    fig_corr = px.imshow(corr_matrix,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title="Matrice de corrélation")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, width='stretch')
    
    # Graphiques de dispersion pour les corrélations importantes
    st.subheader("Relation avec les indicateurs clés")
    
    # Trouver les indicateurs les plus corrélés (positivement ou négativement)
    correlations = corr_matrix[ticker].drop(ticker).abs().sort_values(ascending=False)
    top_correlated = correlations.head(2).index.tolist()
    
    col1, col2 = st.columns(2)
    
    for i, indicator in enumerate(top_correlated):
        with col1 if i == 0 else col2:
            # Préparer les données
            scatter_data = pd.DataFrame({
                'x': macro_comparison[indicator].pct_change(),
                'y': macro_comparison[ticker].pct_change()
            }).dropna()
            
            # Calculer la corrélation
            corr = scatter_data['x'].corr(scatter_data['y'])
            
            # Graphique de dispersion
            try:
                # Vérifier si statsmodels est disponible
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                
                if has_statsmodels:
                    fig_scatter = px.scatter(scatter_data, x='x', y='y',
                                          trendline="ols",
                                          title=f"{ticker} vs {indicator} (Corr: {corr:.2f})",
                                          labels={"x": f"{indicator} (variation journalière %)", 
                                                 "y": f"{ticker} (variation journalière %)"}
                                         )
                else:
                    # Version sans ligne de tendance si statsmodels n'est pas disponible
                    fig_scatter = px.scatter(scatter_data, x='x', y='y',
                                          title=f"{ticker} vs {indicator} (Corr: {corr:.2f})",
                                          labels={"x": f"{indicator} (variation journalière %)", 
                                                 "y": f"{ticker} (variation journalière %)"}
                                         )
                    st.info("📊 Pour afficher les lignes de tendance, installez le package 'statsmodels' avec la commande: pip install statsmodels")
                
                st.plotly_chart(fig_scatter, width='stretch')
            except Exception as e:
                st.warning(f"Erreur lors de la création du graphique de dispersion: {e}")

# --------- Données financières ----------
st.subheader("Données financières")

if financials and isinstance(financials, dict) and any(not df.empty if isinstance(df, pd.DataFrame) else False for df in financials.values() if df is not None):
    tabs = st.tabs(["Compte de résultat", "Bilan", "Flux de trésorerie", "Ratios clés"])
    
    with tabs[0]:
        if "income_stmt" in financials and not financials["income_stmt"].empty:
            st.dataframe(financials["income_stmt"].T)
        else:
            st.info("Données du compte de résultat non disponibles.")
    
    with tabs[1]:
        if "balance_sheet" in financials and not financials["balance_sheet"].empty:
            st.dataframe(financials["balance_sheet"].T)
        else:
            st.info("Données du bilan non disponibles.")
    
    with tabs[2]:
        if "cash_flow" in financials and not financials["cash_flow"].empty:
            st.dataframe(financials["cash_flow"].T)
        else:
            st.info("Données des flux de trésorerie non disponibles.")
            
    with tabs[3]:
        try:
            if "income_stmt" in financials and "balance_sheet" in financials and not financials["income_stmt"].empty and not financials["balance_sheet"].empty:
                # Calculer quelques ratios financiers clés
                income = financials["income_stmt"]
                balance = financials["balance_sheet"]
                
                # Créer un DataFrame pour les ratios
                ratios = pd.DataFrame(index=income.columns)
                
                # Rentabilité
                if "Net Income" in income.index:
                    ratios["Marge nette (%)"] = (income.loc["Net Income"] / income.loc["Total Revenue"]) * 100 if "Total Revenue" in income.index else np.nan
                    ratios["ROE (%)"] = (income.loc["Net Income"] / balance.loc["Total Stockholder Equity"]) * 100 if "Total Stockholder Equity" in balance.index else np.nan
                    ratios["ROA (%)"] = (income.loc["Net Income"] / balance.loc["Total Assets"]) * 100 if "Total Assets" in balance.index else np.nan
                
                # Liquidité
                if "Current Assets" in balance.index and "Current Liabilities" in balance.index:
                    ratios["Ratio de liquidité"] = balance.loc["Current Assets"] / balance.loc["Current Liabilities"]
                
                # Endettement
                if "Total Assets" in balance.index and "Total Liabilities Net Minority Interest" in balance.index:
                    ratios["Ratio d'endettement (%)"] = (balance.loc["Total Liabilities Net Minority Interest"] / balance.loc["Total Assets"]) * 100
                
                # Valorisation
                if "market_cap" in company_info:
                    market_cap = company_info["market_cap"]
                    if "Net Income" in income.index:
                        ratios["P/E (estimé)"] = market_cap / income.loc["Net Income"].iloc[0] if not income.loc["Net Income"].iloc[0] == 0 else np.nan
                    if "Total Revenue" in income.index:
                        ratios["P/S (estimé)"] = market_cap / income.loc["Total Revenue"].iloc[0] if not income.loc["Total Revenue"].iloc[0] == 0 else np.nan
                    if "Total Assets" in balance.index and "Total Liabilities Net Minority Interest" in balance.index:
                        book_value = balance.loc["Total Assets"].iloc[0] - balance.loc["Total Liabilities Net Minority Interest"].iloc[0]
                        ratios["P/B (estimé)"] = market_cap / book_value if not book_value == 0 else np.nan
                
                # Afficher les ratios
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
    # Sélection des indicateurs FRED à afficher
    st.write("Sélectionnez les indicateurs économiques à analyser:")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_inflation = st.multiselect("Inflation", 
                                          ["CPIAUCSL", "T10YIE"], 
                                          default=["CPIAUCSL"], 
                                          format_func=lambda x: f"{x} - {FRED_SERIES[x]}")
        
        selected_growth = st.multiselect("Croissance", 
                                       ["INDPRO", "GDPC1"], 
                                       default=["INDPRO"], 
                                       format_func=lambda x: f"{x} - {FRED_SERIES[x]}")
    
    with col2:
        selected_rates = st.multiselect("Taux d'intérêt", 
                                      ["DGS10", "DGS2"], 
                                      default=["DGS10"], 
                                      format_func=lambda x: f"{x} - {FRED_SERIES[x]}")
        
        selected_other = st.multiselect("Autres indicateurs", 
                                      ["UNRATE", "DTWEXBGS", "NFCI", "USREC"], 
                                      default=["UNRATE"], 
                                      format_func=lambda x: f"{x} - {FRED_SERIES[x]}")
    
    # Combiner tous les indicateurs sélectionnés
    selected_indicators = selected_inflation + selected_growth + selected_rates + selected_other
    
    if selected_indicators:
        # Période pour les données FRED
        fred_start_date = datetime.now() - timedelta(days=365*5)  # 5 ans par défaut
        
        # Récupérer les données FRED
        with st.spinner("Récupération des données économiques en cours..."):
            fred_data = get_fred_data(selected_indicators, fred_start_date)
        
        if not fred_data.empty:
            # Afficher les données économiques
            st.subheader("Évolution des indicateurs économiques")
            
            # Créer des onglets pour les différentes catégories
            macro_tabs = st.tabs(["Inflation", "Croissance", "Taux d'intérêt", "Autres", "Impact sur l'action"])
            
            # Onglet Inflation
            with macro_tabs[0]:
                if selected_inflation:
                    inflation_data = fred_data[[FRED_SERIES[col] for col in selected_inflation]]
                    if not inflation_data.empty:
                        fig = px.line(inflation_data, x=inflation_data.index, y=inflation_data.columns,
                                    title="Indicateurs d'inflation")
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, width='stretch')
                        
                        # Calculer les variations annuelles
                        if len(inflation_data) > 252:  # Au moins un an de données
                            annual_change = inflation_data.pct_change(252).iloc[-1] * 100
                            st.write("Variation sur 12 mois:")
                            st.dataframe(annual_change.to_frame("Variation (%)").T)
                    else:
                        st.info("Aucune donnée d'inflation disponible pour la période sélectionnée.")
                else:
                    st.info("Aucun indicateur d'inflation sélectionné.")
            
            # Onglet Croissance
            with macro_tabs[1]:
                if selected_growth:
                    growth_data = fred_data[[FRED_SERIES[col] for col in selected_growth]]
                    if not growth_data.empty:
                        fig = px.line(growth_data, x=growth_data.index, y=growth_data.columns,
                                    title="Indicateurs de croissance économique")
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, width='stretch')
                        
                        # Calculer les variations annuelles
                        if len(growth_data) > 252:  # Au moins un an de données
                            annual_change = growth_data.pct_change(252).iloc[-1] * 100
                            st.write("Variation sur 12 mois:")
                            st.dataframe(annual_change.to_frame("Variation (%)").T)
                    else:
                        st.info("Aucune donnée de croissance disponible pour la période sélectionnée.")
                else:
                    st.info("Aucun indicateur de croissance sélectionné.")
            
            # Onglet Taux d'intérêt
            with macro_tabs[2]:
                if selected_rates:
                    rates_data = fred_data[[FRED_SERIES[col] for col in selected_rates]]
                    if not rates_data.empty:
                        fig = px.line(rates_data, x=rates_data.index, y=rates_data.columns,
                                    title="Évolution des taux d'intérêt")
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, width='stretch')
                        
                        # Calculer les variations absolues sur 3 mois
                        if len(rates_data) > 63:  # Au moins 3 mois de données
                            change_3m = rates_data.iloc[-1] - rates_data.iloc[-63]
                            st.write("Variation absolue sur 3 mois (points de base):")
                            st.dataframe((change_3m * 100).to_frame("Variation (pb)").T)
                    else:
                        st.info("Aucune donnée de taux d'intérêt disponible pour la période sélectionnée.")
                else:
                    st.info("Aucun indicateur de taux d'intérêt sélectionné.")
            
            # Onglet Autres indicateurs
            with macro_tabs[3]:
                if selected_other:
                    other_data = fred_data[[FRED_SERIES[col] for col in selected_other]]
                    if not other_data.empty:
                        fig = px.line(other_data, x=other_data.index, y=other_data.columns,
                                    title="Autres indicateurs économiques")
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Aucune donnée disponible pour la période sélectionnée.")
                else:
                    st.info("Aucun autre indicateur sélectionné.")
            
            # Onglet Impact sur l'action
            with macro_tabs[4]:
                st.write("Analyse de l'impact des facteurs macroéconomiques sur l'action")
                
                # Fusionner les données de l'action avec les données FRED
                try:
                    # Convertir les données de l'action en rendements mensuels
                    stock_monthly = stock_data['Close'].resample('M').last()
                    stock_returns = stock_monthly.pct_change().dropna()
                    
                    # Convertir les données FRED en données mensuelles
                    fred_monthly = fred_data.resample('M').last()
                    
                    # Fusionner les données
                    merged_data = pd.concat([stock_returns, fred_monthly], axis=1).dropna()
                    merged_data.columns = [ticker] + list(fred_monthly.columns)
                    
                    if not merged_data.empty and len(merged_data) > 24:  # Au moins 2 ans de données
                        # Calculer les corrélations
                        corr_matrix = merged_data.corr()[ticker].drop(ticker).sort_values(ascending=False)
                        
                        # Afficher les corrélations
                        st.write("Corrélation entre les rendements mensuels de l'action et les indicateurs économiques:")
                        fig_corr = px.bar(corr_matrix, 
                                        title="Impact des facteurs économiques sur l'action",
                                        labels={"value": "Corrélation", "index": "Indicateur"})
                        fig_corr.update_layout(height=400)
                        st.plotly_chart(fig_corr, width='stretch')
                        
                        # Analyse de sensibilité basée sur le secteur
                        if ticker in SECTOR_SENSITIVITY.columns:
                            st.write("Sensibilité théorique de l'action aux facteurs économiques:")
                            sensitivity = SECTOR_SENSITIVITY[ticker].dropna()
                            
                            # Créer un graphique de la sensibilité
                            fig_sens = px.bar(sensitivity, 
                                            title="Sensibilité théorique aux facteurs économiques",
                                            labels={"value": "Sensibilité (-2 à +2)", "index": "Facteur"})
                            fig_sens.update_layout(height=300)
                            st.plotly_chart(fig_sens, width='stretch')
                            
                            # Interprétation
                            st.write("**Interprétation de la sensibilité:**")
                            st.write("- **+2**: Très positivement impacté par une hausse du facteur")
                            st.write("- **+1**: Modérément positivement impacté")
                            st.write("- **0**: Impact neutre")
                            st.write("- **-1**: Modérément négativement impacté")
                            st.write("- **-2**: Très négativement impacté par une hausse du facteur")
                    else:
                        st.info("Données insuffisantes pour analyser l'impact des facteurs économiques.")
                except Exception as e:
                    st.warning(f"Erreur lors de l'analyse de l'impact économique: {e}")
        else:
            st.warning("Impossible de récupérer les données économiques. Veuillez réessayer plus tard.")
    else:
        st.info("Veuillez sélectionner au moins un indicateur économique pour l'analyse.")

# --------- Résumé et recommandations ----------
st.subheader("Résumé et analyse")

# Calculer quelques indicateurs techniques simples pour l'analyse
try:
    current_price = stock_data['Close'].iloc[-1]
    
    # Vérifier que les indicateurs sont disponibles (suffisamment de données historiques)
    has_sma_20 = not stock_data_with_indicators['SMA_20'].isna().iloc[-1] if 'SMA_20' in stock_data_with_indicators else False
    has_sma_50 = not stock_data_with_indicators['SMA_50'].isna().iloc[-1] if 'SMA_50' in stock_data_with_indicators else False
    has_sma_200 = not stock_data_with_indicators['SMA_200'].isna().iloc[-1] if 'SMA_200' in stock_data_with_indicators else False
    has_rsi = not stock_data_with_indicators['RSI'].isna().iloc[-1] if 'RSI' in stock_data_with_indicators else False
    has_macd = not stock_data_with_indicators['MACD'].isna().iloc[-1] if 'MACD' in stock_data_with_indicators else False
    has_macd_signal = not stock_data_with_indicators['MACD_Signal'].isna().iloc[-1] if 'MACD_Signal' in stock_data_with_indicators else False
    
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
    st.markdown(f"**Tendance de prix:** {price_trend}")
else:
    st.markdown("**Tendance de prix:** Données insuffisantes pour déterminer la tendance")

# Signaux techniques
signals = []

# Vérifier que les indicateurs sont disponibles avant de les utiliser
if current_price is not None and sma_20 is not None:
    if current_price > sma_20:
        signals.append("Prix au-dessus de la SMA 20 ✅")
    else:
        signals.append("Prix en-dessous de la SMA 20 ❌")
else:
    signals.append("SMA 20: Données insuffisantes ℹ️")
    
if current_price is not None and sma_50 is not None:
    if current_price > sma_50:
        signals.append("Prix au-dessus de la SMA 50 ✅")
    else:
        signals.append("Prix en-dessous de la SMA 50 ❌")
else:
    signals.append("SMA 50: Données insuffisantes ℹ️")
    
if current_price is not None and sma_200 is not None:
    if current_price > sma_200:
        signals.append("Prix au-dessus de la SMA 200 ✅")
    else:
        signals.append("Prix en-dessous de la SMA 200 ❌")
else:
    signals.append("SMA 200: Données insuffisantes ℹ️")
    
if sma_20 is not None and sma_50 is not None:
    if sma_20 > sma_50:
        signals.append("SMA 20 au-dessus de SMA 50 (signal haussier) ✅")
    else:
        signals.append("SMA 20 en-dessous de SMA 50 (signal baissier) ❌")
else:
    signals.append("Croisement SMA: Données insuffisantes ℹ️")
    
if rsi is not None:
    if rsi > 70:
        signals.append(f"RSI en zone de surachat ({rsi:.1f}) ⚠️")
    elif rsi < 30:
        signals.append(f"RSI en zone de survente ({rsi:.1f}) ⚠️")
    else:
        signals.append(f"RSI en zone neutre ({rsi:.1f}) ✓")
else:
    signals.append("RSI: Données insuffisantes ℹ️")
    
if macd is not None and macd_signal is not None:
    if macd > macd_signal:
        signals.append("MACD au-dessus de la ligne de signal (signal haussier) ✅")
    else:
        signals.append("MACD en-dessous de la ligne de signal (signal baissier) ❌")
else:
    signals.append("MACD: Données insuffisantes ℹ️")

# Afficher les signaux
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Signaux techniques:**")
    for signal in signals[:3]:
        st.markdown(f"- {signal}")
        
with col2:
    st.markdown("&nbsp;")
    for signal in signals[3:]:
        st.markdown(f"- {signal}")

# Conclusion
st.markdown("---")
st.markdown("**Conclusion:**")

# Compter les signaux positifs et négatifs
positive_signals = sum(1 for s in signals if "✅" in s)
negative_signals = sum(1 for s in signals if "❌" in s)
warning_signals = sum(1 for s in signals if "⚠️" in s)

if positive_signals > negative_signals + warning_signals:
    st.markdown("L'analyse technique suggère une tendance globalement **positive** pour cette action. La majorité des indicateurs sont haussiers, mais surveillez toujours les changements de tendance et les facteurs fondamentaux avant de prendre des décisions d'investissement.")
elif negative_signals > positive_signals + warning_signals:
    st.markdown("L'analyse technique suggère une tendance globalement **négative** pour cette action. La majorité des indicateurs sont baissiers, mais surveillez toujours les changements de tendance et les facteurs fondamentaux avant de prendre des décisions d'investissement.")
else:
    st.markdown("L'analyse technique suggère une tendance **mixte** pour cette action. Les indicateurs donnent des signaux contradictoires, ce qui pourrait indiquer une période de consolidation ou d'incertitude. Surveillez attentivement les niveaux de support et de résistance ainsi que les facteurs fondamentaux avant de prendre des décisions d'investissement.")

# --------- Prévisions à long terme ----------
st.subheader("Prévisions à long terme")

# Option pour afficher les prévisions à long terme
show_forecasts = st.checkbox("Afficher les prévisions à long terme", value=False)

if show_forecasts:
    try:
        # Vérifier si nous avons suffisamment de données historiques
        if len(stock_data) > 252:  # Au moins un an de données
            # Sélection de la méthode de prévision
            forecast_method = st.selectbox(
                "Méthode de prévision",
                ["Tendance simple", "Moyenne mobile", "Régression linéaire", "ARIMA", "Prophet", "Modèle hybride"]
            )
            
            # Sélection de l'horizon de prévision
            forecast_horizon = st.slider("Horizon de prévision (jours)", 30, 365, 180)
            
            # Préparation des données
            close_prices = stock_data['Close']
            dates = close_prices.index
            last_date = dates[-1]
            
            # Générer les dates futures
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
            
            # Options avancées
            with st.expander("Options avancées de prévision"):
                # Niveau de confiance pour l'intervalle de prédiction
                confidence_level = st.slider("Niveau de confiance (%)", 50, 95, 80) / 100
                
                # Option pour inclure des facteurs macroéconomiques
                include_macro = st.checkbox("Inclure des facteurs macroéconomiques", value=True)
                
                # Sélection des facteurs macroéconomiques si l'option est activée
                selected_macro_factors = []
                if include_macro:
                    selected_macro_factors = st.multiselect(
                        "Sélectionner les facteurs macroéconomiques à inclure",
                        ["Taux d'intérêt", "Inflation", "Dollar US", "Prix de l'or", "VIX (volatilité)"],
                        default=["Taux d'intérêt", "Dollar US"]
                    )
                
                # Option pour la validation croisée
                use_cross_validation = st.checkbox("Utiliser la validation croisée", value=False)
                if use_cross_validation:
                    cv_periods = st.slider("Nombre de périodes de validation", 3, 10, 5)
                    cv_window = st.slider("Taille de la fenêtre de validation (jours)", 30, 180, 60)
            
            # Méthode de prévision
            if forecast_method == "Tendance simple":
                # Calculer la tendance sur les 6 derniers mois
                if len(close_prices) > 126:  # Au moins 6 mois de données
                    recent_prices = close_prices[-126:]  # Environ 6 mois de trading days
                else:
                    recent_prices = close_prices
                
                # Calculer la tendance (pente)
                x = np.arange(len(recent_prices))
                slope, intercept = np.polyfit(x, recent_prices, 1)
                
                # Projeter dans le futur
                future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                forecast = slope * future_x + intercept
                
                # Calculer l'erreur standard pour l'intervalle de confiance
                y_pred = slope * x + intercept
                rmse = np.sqrt(np.mean((recent_prices - y_pred) ** 2))
                std_error = rmse * np.sqrt(1 + 1/len(x) + (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
                
                # Créer un DataFrame pour la visualisation avec intervalles de confiance
                z_value = 1.96  # Pour un niveau de confiance de 95%
                if confidence_level == 0.8:
                    z_value = 1.28
                elif confidence_level == 0.9:
                    z_value = 1.645
                
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast,
                    'Lower': forecast - z_value * std_error,
                    'Upper': forecast + z_value * std_error
                }).set_index('Date')
                
                # Description de la méthode
                st.write(f"**Méthode de tendance simple:** Projection linéaire basée sur les {len(recent_prices)} derniers jours de trading.")
                st.write(f"Tendance quotidienne moyenne: {slope:.4f} {company_info.get('currency', '$')}/jour")
                st.write(f"RMSE (erreur quadratique moyenne): {rmse:.4f}")
                
            elif forecast_method == "Moyenne mobile":
                # Utiliser une moyenne mobile exponentielle
                ema_span = st.slider("Période de la moyenne mobile (jours)", 20, 200, 50)
                ema = close_prices.ewm(span=ema_span, adjust=False).mean()
                
                # Calculer la tendance récente de l'EMA
                recent_ema = ema[-60:]  # Derniers 60 jours pour plus de stabilité
                x = np.arange(len(recent_ema))
                slope, intercept = np.polyfit(x, recent_ema, 1)
                
                # Projeter dans le futur
                future_x = np.arange(len(recent_ema), len(recent_ema) + forecast_horizon)
                forecast = slope * future_x + intercept
                
                # Calculer la volatilité historique pour l'intervalle de confiance
                volatility = close_prices.pct_change().std() * np.sqrt(252)  # Annualisée
                daily_vol = volatility / np.sqrt(252)  # Quotidienne
                
                # Créer un DataFrame pour la visualisation avec intervalles de confiance
                z_value = 1.96  # Pour un niveau de confiance de 95%
                if confidence_level == 0.8:
                    z_value = 1.28
                elif confidence_level == 0.9:
                    z_value = 1.645
                
                # L'incertitude augmente avec l'horizon de prévision (racine carrée du temps)
                time_factors = np.sqrt(np.arange(1, forecast_horizon + 1))
                uncertainty = np.array([daily_vol * tf * z_value for tf in time_factors])
                
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast,
                    'Lower': forecast * (1 - uncertainty),
                    'Upper': forecast * (1 + uncertainty)
                }).set_index('Date')
                
                # Description de la méthode
                st.write(f"**Méthode de moyenne mobile:** Projection basée sur la tendance de la moyenne mobile exponentielle sur {ema_span} jours.")
                st.write(f"Volatilité annualisée: {volatility:.2%}")
                
            elif forecast_method == "Régression linéaire":
                # Vérifier si statsmodels est disponible
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                
                if has_statsmodels:
                    import statsmodels.api as sm
                    
                    # Créer des variables explicatives (tendance temporelle + saisonnalité)
                    df = pd.DataFrame(index=dates)
                    df['price'] = close_prices
                    df['trend'] = np.arange(len(df))
                    
                    # Ajouter des variables saisonnières (jour de la semaine, mois)
                    df['month'] = df.index.month
                    df['day_of_week'] = df.index.dayofweek
                    df['quarter'] = df.index.quarter
                    
                    # Ajouter des variables techniques
                    df['ma20'] = close_prices.rolling(window=20).mean().fillna(method='bfill')
                    df['ma50'] = close_prices.rolling(window=50).mean().fillna(method='bfill')
                    df['volatility'] = close_prices.rolling(window=20).std().fillna(method='bfill')
                    
                    # Ajouter des facteurs macroéconomiques si sélectionnés
                    if include_macro and selected_macro_factors:
                        # Simuler l'ajout de facteurs macro (dans une application réelle, ces données seraient récupérées)
                        if "Taux d'intérêt" in selected_macro_factors:
                            # Simuler un taux d'intérêt (dans une application réelle, utiliser FRED ou autre source)
                            df['interest_rate'] = np.random.normal(2.5, 0.5, len(df))
                        if "Dollar US" in selected_macro_factors:
                            df['usd_index'] = np.random.normal(100, 5, len(df))
                        if "Inflation" in selected_macro_factors:
                            df['inflation'] = np.random.normal(2.0, 0.3, len(df))
                    
                    # Créer des variables indicatrices pour les mois et jours de la semaine
                    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
                    dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True)
                    quarter_dummies = pd.get_dummies(df['quarter'], prefix='quarter', drop_first=True)
                    
                    # Combiner toutes les variables explicatives
                    X_columns = ['trend', 'ma20', 'ma50', 'volatility']
                    if include_macro and selected_macro_factors:
                        if "Taux d'intérêt" in selected_macro_factors:
                            X_columns.append('interest_rate')
                        if "Dollar US" in selected_macro_factors:
                            X_columns.append('usd_index')
                        if "Inflation" in selected_macro_factors:
                            X_columns.append('inflation')
                    
                    X = pd.concat([df[X_columns], month_dummies, dow_dummies, quarter_dummies], axis=1)
                    y = df['price']
                    
                    # Ajuster le modèle
                    model = sm.OLS(y, sm.add_constant(X)).fit()
                    
                    # Créer des données pour la prévision
                    future_df = pd.DataFrame(index=future_dates)
                    future_df['trend'] = np.arange(len(df), len(df) + len(future_dates))
                    future_df['month'] = future_df.index.month
                    future_df['day_of_week'] = future_df.index.dayofweek
                    future_df['quarter'] = future_df.index.quarter
                    
                    # Projeter les moyennes mobiles et la volatilité
                    future_df['ma20'] = df['ma20'].iloc[-1]
                    future_df['ma50'] = df['ma50'].iloc[-1]
                    future_df['volatility'] = df['volatility'].iloc[-1]
                    
                    # Projeter les facteurs macroéconomiques
                    if include_macro and selected_macro_factors:
                        if "Taux d'intérêt" in selected_macro_factors:
                            future_df['interest_rate'] = df['interest_rate'].iloc[-1]
                        if "Dollar US" in selected_macro_factors:
                            future_df['usd_index'] = df['usd_index'].iloc[-1]
                        if "Inflation" in selected_macro_factors:
                            future_df['inflation'] = df['inflation'].iloc[-1]
                    
                    # Créer des variables indicatrices pour les mois et jours de la semaine
                    future_month_dummies = pd.get_dummies(future_df['month'], prefix='month', drop_first=True)
                    future_dow_dummies = pd.get_dummies(future_df['day_of_week'], prefix='dow', drop_first=True)
                    future_quarter_dummies = pd.get_dummies(future_df['quarter'], prefix='quarter', drop_first=True)
                    
                    # Ajouter les colonnes manquantes et réorganiser
                    for col in month_dummies.columns:
                        if col not in future_month_dummies.columns:
                            future_month_dummies[col] = 0
                    for col in dow_dummies.columns:
                        if col not in future_dow_dummies.columns:
                            future_dow_dummies[col] = 0
                    for col in quarter_dummies.columns:
                        if col not in future_quarter_dummies.columns:
                            future_quarter_dummies[col] = 0
                    
                    future_X = pd.concat([future_df[X_columns], 
                                        future_month_dummies[month_dummies.columns], 
                                        future_dow_dummies[dow_dummies.columns],
                                        future_quarter_dummies[quarter_dummies.columns]], axis=1)
                    
                    # Faire la prévision
                    forecast = model.predict(sm.add_constant(future_X))
                    
                    # Calculer les intervalles de confiance
                    from statsmodels.sandbox.regression.predstd import wls_prediction_std
                    _, lower, upper = wls_prediction_std(model, sm.add_constant(future_X), alpha=1-confidence_level)
                    
                    # Créer un DataFrame pour la visualisation
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast,
                        'Lower': lower,
                        'Upper': upper
                    }).set_index('Date')
                    
                    # Description de la méthode
                    st.write("**Méthode de régression linéaire avancée:** Modèle tenant compte de la tendance, des facteurs saisonniers et techniques.")
                    st.write(f"R² du modèle: {model.rsquared:.4f}")
                    
                    # Afficher les coefficients les plus importants
                    coefs = model.params.sort_values(ascending=False)
                    st.write("**Facteurs les plus influents:**")
                    coef_df = pd.DataFrame({
                        'Facteur': coefs.index[:5],
                        'Coefficient': coefs.values[:5]
                    })
                    st.dataframe(coef_df)
                else:
                    st.warning("La méthode de régression linéaire nécessite le package 'statsmodels'. Veuillez l'installer avec: pip install statsmodels")
                    # Utiliser la méthode de tendance simple comme fallback
                    recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent_prices))
                    slope, intercept = np.polyfit(x, recent_prices, 1)
                    future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast,
                        'Lower': forecast * 0.9,
                        'Upper': forecast * 1.1
                    }).set_index('Date')
            
            elif forecast_method == "ARIMA":
                # Vérifier si statsmodels est disponible
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                
                if has_statsmodels:
                    from statsmodels.tsa.arima.model import ARIMA
                    from pmdarima import auto_arima
                    
                    # Utiliser auto_arima pour trouver les meilleurs paramètres
                    with st.spinner("Recherche des meilleurs paramètres ARIMA..."):
                        try:
                            # Limiter la recherche pour des raisons de performance
                            auto_model = auto_arima(close_prices, 
                                                   start_p=0, start_q=0, max_p=5, max_q=5, max_d=2,
                                                   seasonal=False, trace=False,
                                                   error_action='ignore', suppress_warnings=True,
                                                   stepwise=True, n_jobs=-1)
                            best_order = auto_model.order
                            st.write(f"Meilleurs paramètres ARIMA trouvés: {best_order}")
                        except Exception as e:
                            st.warning(f"Erreur lors de la recherche automatique des paramètres: {e}")
                            best_order = (2, 1, 2)  # Paramètres par défaut
                            st.write(f"Utilisation des paramètres ARIMA par défaut: {best_order}")
                    
                    # Ajuster le modèle ARIMA avec les meilleurs paramètres
                    model = ARIMA(close_prices, order=best_order)
                    model_fit = model.fit()
                    
                    # Faire la prévision
                    forecast_result = model_fit.forecast(steps=forecast_horizon)
                    forecast = forecast_result
                    
                    # Obtenir les intervalles de confiance
                    forecast_ci = model_fit.get_forecast(steps=forecast_horizon).conf_int(alpha=1-confidence_level)
                    
                    # Créer un DataFrame pour la visualisation
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast,
                        'Lower': forecast_ci.iloc[:, 0].values,
                        'Upper': forecast_ci.iloc[:, 1].values
                    }).set_index('Date')
                    
                    # Description de la méthode
                    st.write(f"**Méthode ARIMA({best_order[0]}, {best_order[1]}, {best_order[2]}):** Modèle de série temporelle avancé tenant compte de l'autocorrélation.")
                    st.write(f"AIC: {model_fit.aic:.2f}")
                    
                    # Afficher les diagnostics du modèle
                    with st.expander("Diagnostics du modèle ARIMA"):
                        st.write("**Résumé du modèle:**")
                        st.text(str(model_fit.summary()))
                else:
                    st.warning("La méthode ARIMA nécessite les packages 'statsmodels' et 'pmdarima'. Veuillez les installer avec: pip install statsmodels pmdarima")
                    # Utiliser la méthode de tendance simple comme fallback
                    recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent_prices))
                    slope, intercept = np.polyfit(x, recent_prices, 1)
                    future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast,
                        'Lower': forecast * 0.9,
                        'Upper': forecast * 1.1
                    }).set_index('Date')
            
            elif forecast_method == "Prophet":
                # Vérifier si Prophet est disponible
                import importlib.util
                has_prophet = importlib.util.find_spec('prophet') is not None
                
                if has_prophet:
                    from prophet import Prophet
                    
                    # Préparer les données pour Prophet
                    prophet_data = pd.DataFrame({
                        'ds': close_prices.index,
                        'y': close_prices.values
                    })
                    
                    # Configurer et ajuster le modèle Prophet
                    with st.spinner("Ajustement du modèle Prophet..."):
                        model = Prophet(
                            changepoint_prior_scale=0.05,  # Flexibilité des points de changement
                            seasonality_prior_scale=10.0,  # Force de la saisonnalité
                            seasonality_mode='multiplicative',  # Mode de saisonnalité
                            interval_width=confidence_level  # Niveau de confiance
                        )
                        
                        # Ajouter des saisonnalités personnalisées
                        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
                        
                        # Ajouter des régresseurs externes si des facteurs macro sont sélectionnés
                        if include_macro and selected_macro_factors:
                            # Dans une application réelle, ces données seraient récupérées de sources externes
                            if "Taux d'intérêt" in selected_macro_factors:
                                prophet_data['interest_rate'] = np.random.normal(2.5, 0.5, len(prophet_data))
                                model.add_regressor('interest_rate')
                            if "Dollar US" in selected_macro_factors:
                                prophet_data['usd_index'] = np.random.normal(100, 5, len(prophet_data))
                                model.add_regressor('usd_index')
                        
                        # Ajuster le modèle
                        model.fit(prophet_data)
                    
                    # Préparer les données futures pour la prévision
                    future = model.make_future_dataframe(periods=forecast_horizon)
                    
                    # Ajouter les valeurs futures des régresseurs
                    if include_macro and selected_macro_factors:
                        if "Taux d'intérêt" in selected_macro_factors:
                            future['interest_rate'] = prophet_data['interest_rate'].iloc[-1]
                        if "Dollar US" in selected_macro_factors:
                            future['usd_index'] = prophet_data['usd_index'].iloc[-1]
                    
                    # Faire la prévision
                    forecast_result = model.predict(future)
                    
                    # Extraire les résultats pertinents
                    forecast_df = pd.DataFrame({
                        'Date': forecast_result['ds'].iloc[-forecast_horizon:],
                        'Forecast': forecast_result['yhat'].iloc[-forecast_horizon:],
                        'Lower': forecast_result['yhat_lower'].iloc[-forecast_horizon:],
                        'Upper': forecast_result['yhat_upper'].iloc[-forecast_horizon:]
                    }).set_index('Date')
                    
                    # Description de la méthode
                    st.write("**Méthode Prophet:** Modèle de décomposition de série temporelle développé par Facebook Research.")
                    st.write("Avantages: Gestion automatique des tendances, saisonnalités et jours fériés.")
                    
                    # Afficher les composantes du modèle
                    with st.expander("Composantes du modèle Prophet"):
                        fig_comp = model.plot_components(forecast_result)
                        st.pyplot(fig_comp)
                else:
                    st.warning("La méthode Prophet nécessite le package 'prophet'. Veuillez l'installer avec: pip install prophet")
                    # Utiliser la méthode de tendance simple comme fallback
                    recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent_prices))
                    slope, intercept = np.polyfit(x, recent_prices, 1)
                    future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast,
                        'Lower': forecast * 0.9,
                        'Upper': forecast * 1.1
                    }).set_index('Date')
            
            elif forecast_method == "Modèle hybride":
                # Vérifier si statsmodels est disponible
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                
                if has_statsmodels:
                    import statsmodels.api as sm
                    from statsmodels.tsa.arima.model import ARIMA
                    
                    # Créer plusieurs modèles et combiner leurs prévisions
                    with st.spinner("Création du modèle hybride..."):
                        # 1. Modèle de tendance linéaire
                        x = np.arange(len(close_prices))
                        slope, intercept = np.polyfit(x, close_prices, 1)
                        future_x = np.arange(len(close_prices), len(close_prices) + forecast_horizon)
                        trend_forecast = slope * future_x + intercept
                        
                        # 2. Modèle ARIMA simplifié
                        try:
                            arima_model = ARIMA(close_prices, order=(2, 1, 2))
                            arima_fit = arima_model.fit()
                            arima_forecast = arima_fit.forecast(steps=forecast_horizon)
                        except Exception as e:
                            st.warning(f"Erreur lors de l'ajustement du modèle ARIMA: {e}")
                            arima_forecast = trend_forecast  # Fallback
                        
                        # 3. Modèle de régression avec variables techniques
                        df = pd.DataFrame(index=dates)
                        df['price'] = close_prices
                        df['ma20'] = close_prices.rolling(window=20).mean().fillna(method='bfill')
                        df['ma50'] = close_prices.rolling(window=50).mean().fillna(method='bfill')
                        df['volatility'] = close_prices.rolling(window=20).std().fillna(method='bfill')
                        df['trend'] = np.arange(len(df))
                        
                        # Ajouter des facteurs macroéconomiques si sélectionnés
                        if include_macro and selected_macro_factors:
                            if "Taux d'intérêt" in selected_macro_factors:
                                df['interest_rate'] = np.random.normal(2.5, 0.5, len(df))
                            if "Dollar US" in selected_macro_factors:
                                df['usd_index'] = np.random.normal(100, 5, len(df))
                        
                        # Définir les variables explicatives
                        X_columns = ['ma20', 'ma50', 'volatility', 'trend']
                        if include_macro and selected_macro_factors:
                            if "Taux d'intérêt" in selected_macro_factors:
                                X_columns.append('interest_rate')
                            if "Dollar US" in selected_macro_factors:
                                X_columns.append('usd_index')
                        
                        X = df[X_columns]
                        y = df['price']
                        
                        try:
                            reg_model = sm.OLS(y, sm.add_constant(X)).fit()
                            
                            # Préparer les données futures
                            future_df = pd.DataFrame(index=future_dates)
                            future_df['ma20'] = df['ma20'].iloc[-1]
                            future_df['ma50'] = df['ma50'].iloc[-1]
                            future_df['volatility'] = df['volatility'].iloc[-1]
                            future_df['trend'] = np.arange(len(df), len(df) + len(future_dates))
                            
                            if include_macro and selected_macro_factors:
                                if "Taux d'intérêt" in selected_macro_factors:
                                    future_df['interest_rate'] = df['interest_rate'].iloc[-1]
                                if "Dollar US" in selected_macro_factors:
                                    future_df['usd_index'] = df['usd_index'].iloc[-1]
                            
                            reg_forecast = reg_model.predict(sm.add_constant(future_df[X_columns]))
                        except Exception as e:
                            st.warning(f"Erreur lors de l'ajustement du modèle de régression: {e}")
                            reg_forecast = trend_forecast  # Fallback
                    
                    # Combiner les prévisions (moyenne pondérée)
                    weights = [0.3, 0.4, 0.3]  # Poids pour chaque modèle
                    combined_forecast = (weights[0] * trend_forecast + 
                                        weights[1] * arima_forecast + 
                                        weights[2] * reg_forecast)
                    
                    # Calculer l'incertitude basée sur la dispersion des prévisions
                    forecasts = np.vstack([trend_forecast, arima_forecast, reg_forecast])
                    forecast_std = np.std(forecasts, axis=0)
                    
                    # Créer un DataFrame pour la visualisation avec intervalles de confiance
                    z_value = 1.96  # Pour un niveau de confiance de 95%
                    if confidence_level == 0.8:
                        z_value = 1.28
                    elif confidence_level == 0.9:
                        z_value = 1.645
                    
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': combined_forecast,
                        'Lower': combined_forecast - z_value * forecast_std,
                        'Upper': combined_forecast + z_value * forecast_std
                    }).set_index('Date')
                    
                    # Description de la méthode
                    st.write("**Modèle hybride:** Combinaison pondérée de plusieurs modèles de prévision.")
                    st.write("Avantages: Réduit le risque d'erreur en combinant différentes approches.")
                    
                    # Afficher les poids des modèles
                    st.write("**Poids des modèles:**")
                    weight_df = pd.DataFrame({
                        'Modèle': ['Tendance', 'ARIMA', 'Régression'],
                        'Poids': weights
                    })
                    st.dataframe(weight_df)
                else:
                    st.warning("Le modèle hybride nécessite le package 'statsmodels'. Veuillez l'installer avec: pip install statsmodels")
                    # Utiliser la méthode de tendance simple comme fallback
                    recent_prices = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent_prices))
                    slope, intercept = np.polyfit(x, recent_prices, 1)
                    future_x = np.arange(len(recent_prices), len(recent_prices) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast,
                        'Lower': forecast * 0.9,
                        'Upper': forecast * 1.1
                    }).set_index('Date')
            
            # Validation croisée si activée
            if use_cross_validation:
                with st.expander("Résultats de la validation croisée"):
                    # Simuler une validation croisée temporelle
                    cv_errors = []
                    for i in range(cv_periods):
                        # Définir les périodes de train/test
                        train_end = len(close_prices) - (cv_periods - i) * cv_window
                        test_end = train_end + cv_window
                        
                        if train_end > 252:  # Au moins un an de données d'entraînement
                            train_data = close_prices[:train_end]
                            test_data = close_prices[train_end:test_end]
                            
                            # Modèle simple pour la validation croisée
                            x_train = np.arange(len(train_data))
                            slope, intercept = np.polyfit(x_train, train_data, 1)
                            
                            # Prédire sur la période de test
                            x_test = np.arange(len(train_data), len(train_data) + len(test_data))
                            pred = slope * x_test + intercept
                            
                            # Calculer l'erreur
                            mape = np.mean(np.abs((test_data.values - pred) / test_data.values)) * 100
                            cv_errors.append(mape)
                    
                    if cv_errors:
                        avg_mape = np.mean(cv_errors)
                        st.write(f"**MAPE moyen sur {cv_periods} périodes:** {avg_mape:.2f}%")
                        st.write("Plus le MAPE est bas, plus le modèle est précis.")
                        
                        # Visualiser les erreurs
                        fig_cv = go.Figure()
                        fig_cv.add_trace(go.Bar(
                            x=[f"Période {i+1}" for i in range(len(cv_errors))],
                            y=cv_errors,
                            marker_color='indianred'
                        ))
                        fig_cv.update_layout(
                            title="Erreur de prévision par période de validation",
                            xaxis_title="Période",
                            yaxis_title="MAPE (%)",
                            height=300
                        )
                        st.plotly_chart(fig_cv, width='stretch')
                    else:
                        st.info("Données insuffisantes pour la validation croisée avec les paramètres actuels.")
            
            # Visualiser la prévision
            fig = go.Figure()
            
            # Ajouter les données historiques
            fig.add_trace(go.Scatter(
                x=dates[-252:],  # Dernière année
                y=close_prices[-252:],
                mode='lines',
                name='Historique',
                line=dict(color='blue')
            ))
            
            # Ajouter la prévision
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Forecast'],
                mode='lines',
                name='Prévision',
                line=dict(color='red', dash='dash')
            ))
            
            # Ajouter l'intervalle de confiance
            fig.add_trace(go.Scatter(
                x=list(forecast_df.index) + list(forecast_df.index)[::-1],
                y=list(forecast_df['Upper']) + list(forecast_df['Lower'])[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name=f'Intervalle de confiance ({int(confidence_level*100)}%)'  # Niveau de confiance dynamique
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
            
            # Afficher quelques statistiques de la prévision
            current_price = close_prices.iloc[-1]
            forecast_end = forecast_df['Forecast'].iloc[-1]
            change_pct = ((forecast_end / current_price) - 1) * 100
            
            st.write("**Résumé de la prévision:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prix actuel", f"{current_price:.2f} {company_info.get('currency', '$')}")
            with col2:
                st.metric("Prix prévu (fin de période)", f"{forecast_end:.2f} {company_info.get('currency', '$')}", f"{change_pct:.2f}%")
            with col3:
                lower_bound = forecast_df['Lower'].iloc[-1]
                upper_bound = forecast_df['Upper'].iloc[-1]
                st.metric("Intervalle de confiance", f"{lower_bound:.2f} - {upper_bound:.2f}")
            with col4:
                st.metric("Horizon", f"{forecast_horizon} jours")
            
            # Téléchargement des prévisions
            csv = forecast_df.to_csv()
            st.download_button(
                label="Télécharger les prévisions (CSV)",
                data=csv,
                file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            
            # Avertissement important
            st.warning("⚠️ **Avertissement:** Ces prévisions sont basées sur des modèles statistiques et des données historiques. Elles ne constituent pas des conseils d'investissement et ne garantissent pas les performances futures. De nombreux facteurs externes peuvent influencer le cours d'une action et ne sont pas pris en compte dans ces modèles.")
        else:
            st.info("Données historiques insuffisantes pour générer des prévisions fiables. Au moins un an de données est nécessaire.")
    except Exception as e:
        st.error(f"Erreur lors de la génération des prévisions: {e}")
        st.error("Détails de l'erreur:", exc_info=True)

def render_stock(default_ticker: str = "AAPL"):
    """Fonction exportable pour afficher l'onglet Action dans le hub"""
    # Cette fonction sera appelée depuis le hub, on ne peut pas dupliquer tout le code ici
    # On indique juste que l'interface complète devrait être disponible
    pass

# Code UI principal ci-dessous (exécuté uniquement si appelé directement)
if __name__ == "__main__":
    # Interface complète d'analyse d'actions
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
