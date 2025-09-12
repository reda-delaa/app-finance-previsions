# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import streamlit as st
import ta

# ================== CONFIG (améliorée) ==================
# - Retrait de DGC.TO (delistée)
# - Pairs limités pour lisibilité/perf
DEFAULT_TICKER = "NGD.TO"

PEER_GROUPS = {
    "Gold Miners": ["ABX.TO", "K.TO", "AEM.TO", "BTO.TO", "IMG.TO", "PAAS.TO", "FR.TO"],
    "Silver Miners": ["PAAS.TO", "EDR.TO", "FR.TO"],
    "Copper Miners": ["CS.TO", "TECK-B.TO", "LUN.TO", "FM.TO"],
    "Diversified Miners": ["RIO", "BHP", "VALE", "FCX"]
}

BENCHMARKS = {
    "^GSPTSE": "TSX Composite",
    "GDX": "VanEck Gold Miners ETF",
    "XGD.TO": "iShares S&P/TSX Global Gold Index ETF",
    "XME": "SPDR S&P Metals & Mining ETF"
}

MACRO_INDICATORS = {
    "GC=F": "Gold Futures",
    "SI=F": "Silver Futures",
    "HG=F": "Copper Futures",
    "DX-Y.NYB": "US Dollar Index",
    "^TNX": "10-Year Treasury Yield"
}

FRED_SERIES = {
    "CPIAUCSL": "CPI (All Items, Index 1982-84=100)",
    "T10YIE":   "10Y Breakeven Inflation",
    "INDPRO":   "Industrial Production Index",
    "GDPC1":    "Real Gross Domestic Product (Quarterly)",
    "UNRATE":   "Unemployment Rate",
    "PAYEMS":   "Total Nonfarm Payrolls",
    "DGS10":    "10Y Treasury Yield",
    "DGS2":     "2Y Treasury Yield",
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index (Broad)",
    "NFCI":     "Chicago Fed National Financial Conditions Index",
    "BAMLC0A0CM": "ICE BofA US Corp Master OAS",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "USREC":    "US Recession Indicator"
}

SECTOR_SENSITIVITY = pd.DataFrame({
    "Inflation":{
        "XLK":-1,"XLF":1,"XLE":2,"XLB":2,"XLV":0,"XLY":-1,"XLP":0,"XLI":1,"XLRE":-1,"XLU":1,
        "GDX":2,"ABX.TO":2,"K.TO":2,"AEM.TO":2,"BTO.TO":2,"IMG.TO":2,
        "PAAS.TO":2,"EDR.TO":2,"FR.TO":2,"CS.TO":1,"TECK-B.TO":1,"LUN.TO":1,"FM.TO":1
    },
    "Growth":{
        "XLK":2,"XLF":1,"XLE":0,"XLB":1,"XLV":0,"XLY":2,"XLP":0,"XLI":1,"XLRE":0,"XLU":-1,
        "GDX":0,"ABX.TO":0,"K.TO":0,"AEM.TO":0,"BTO.TO":0,"IMG.TO":0,
        "PAAS.TO":0,"EDR.TO":0,"FR.TO":0,"CS.TO":1,"TECK-B.TO":1,"LUN.TO":1,"FM.TO":1
    },
    "Rates":{
        "XLK":-2,"XLF":2,"XLE":1,"XLB":0,"XLV":0,"XLY":-1,"XLP":0,"XLI":0,"XLRE":-1,"XLU":1,
        "GDX":1,"ABX.TO":1,"K.TO":1,"AEM.TO":1,"BTO.TO":1,"IMG.TO":1,
        "PAAS.TO":1,"EDR.TO":1,"FR.TO":1,"CS.TO":0,"TECK-B.TO":0,"LUN.TO":0,"FM.TO":0
    },
    "USD":{
        "XLK":-1,"XLF":0,"XLE":-1,"XLB":-1,"XLV":0,"XLY":0,"XLP":0,"XLI":-1,"XLRE":-1,"XLU":0,
        "GDX":-1,"ABX.TO":-1,"K.TO":-1,"AEM.TO":-1,"BTO.TO":-1,"IMG.TO":-1,
        "PAAS.TO":-1,"EDR.TO":-1,"FR.TO":-1,"CS.TO":-1,"TECK-B.TO":-1,"LUN.TO":-1,"FM.TO":-1
    }
}).fillna(0)

# ================== UTILITAIRES ==================

def _has_matplotlib() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("matplotlib") is not None
    except Exception:
        return False

def _fmt_mcap(x: Any) -> str:
    try:
        x = float(x)
        if np.isnan(x) or x == 0:
            return "N/A"
        units = [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]
        for u, v in units:
            if abs(x) >= v:
                return f"{x/v:.2f}{u}"
        return f"{x:.0f}"
    except Exception:
        return "N/A"

def get_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Récupère des données historiques (robuste aux index tz)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        return hist
    except Exception as e:
        st.warning(f"⚠️ Données indisponibles pour {ticker}: {e}")
        return None

def get_peer_data(tickers: List[str], period: str = "1y") -> Tuple[pd.DataFrame, List[str]]:
    """Récupère les Close pour un groupe d'actions, filtre celles vides."""
    data, valid = {}, []
    for t in tickers:
        try:
            hist = get_stock_data(t, period=period)
            if hist is not None and not hist.empty:
                s = hist["Close"].dropna()
                if s.empty:
                    st.caption(f"• {t}: données vides (ignoré)")
                else:
                    data[t] = s
                    valid.append(t)
            else:
                st.caption(f"• {t}: aucune donnée (ignoré)")
            time.sleep(0.1)
        except Exception as e:
            st.caption(f"• {t}: erreur ({e})")
    if not data:
        return pd.DataFrame(), []
    # aligne sur l'intersection de dates
    df = pd.DataFrame(data).dropna(how="all")
    return df, valid

def calculate_returns(df: pd.DataFrame, periods: Dict[str, int]) -> pd.DataFrame:
    """Rendements (%) multi-horizons, tolérant aux séries courtes."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = {}
    for label, days in periods.items():
        try:
            if isinstance(days, int) and len(df) > days >= 1:
                out[label] = df.pct_change(days).iloc[-1] * 100
            else:
                # YTD ou séries plus courtes
                if isinstance(days, int) and days <= 0:
                    out[label] = pd.Series(0, index=df.columns)
                else:
                    out[label] = df.pct_change(max(1, len(df) - 1)).iloc[-1] * 100
        except Exception:
            out[label] = pd.Series(0, index=df.columns)
    return pd.DataFrame(out)

def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volatilité annualisée (%) rolling."""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        return df.pct_change().rolling(window).std() * np.sqrt(252) * 100
    except Exception:
        return pd.DataFrame()

def calculate_beta(stock_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.Series:
    """Bêta glissant simple."""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(index=stock_returns.index, dtype=float)
    rolling_cov = aligned.iloc[:, 0].rolling(window).cov(aligned.iloc[:, 1])
    rolling_var = aligned.iloc[:, 1].rolling(window).var()
    beta = (rolling_cov / rolling_var).replace([np.inf, -np.inf], np.nan)
    return beta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute SMA, RSI, MACD, BB, OBV (ta)."""
    data = df.copy()
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()
    data['BB_Middle'] = bb.bollinger_mavg()
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    return data

def get_company_info(ticker: str) -> Dict[str, Any]:
    """Récupère info yfinance (robuste aux versions)."""
    try:
        t = yf.Ticker(ticker)
        info = {}
        if hasattr(t, "get_info"):
            try:
                info = t.get_info() or {}
            except Exception:
                info = {}
        if not info:
            try:
                info = t.info or {}
            except Exception:
                info = {}
        try:
            fi = getattr(t, "fast_info", None)
            if fi:
                info.setdefault("marketCap", getattr(fi, "market_cap", None))
                info.setdefault("regularMarketPrice", getattr(fi, "last_price", None))
                info.setdefault("currency", getattr(fi, "currency", None))
        except Exception:
            pass
        return info or {}
    except Exception as e:
        st.warning(f"⚠️ Impossible de récupérer les infos pour {ticker}: {e}")
        return {}

def load_fred_series(series_id: str, start_date: Optional[datetime] = None) -> pd.DataFrame:
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
        st.caption(f"• FRED {series_id} indisponible: {e}")
        return pd.DataFrame(columns=[series_id])

def get_fred_data(series_ids: List[str], start_date: Optional[datetime] = None) -> pd.DataFrame:
    data = {}
    for sid in series_ids:
        df = load_fred_series(sid, start_date)
        if not df.empty:
            data[sid] = df[sid]
        time.sleep(0.05)
    if not data:
        return pd.DataFrame()
    res = pd.concat(data, axis=1)
    res.columns = [FRED_SERIES.get(c, c) for c in res.columns]
    return res

def zscore(series: pd.Series, window: int = 24) -> pd.Series:
    s = series.dropna()
    if len(s) < window + 2:
        return pd.Series(index=series.index, dtype=float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return ((s - mu) / (sd.replace(0, np.nan))).reindex(series.index)

def get_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        return {
            "income_stmt": getattr(t, "income_stmt", pd.DataFrame()),
            "balance_sheet": getattr(t, "balance_sheet", pd.DataFrame()),
            "cash_flow": getattr(t, "cashflow", pd.DataFrame())
        }
    except Exception as e:
        st.caption(f"• Données financières indisponibles: {e}")
        return {"income_stmt": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cash_flow": pd.DataFrame()}

def _peer_candidates_for(ticker: str) -> List[str]:
    for _g, lst in PEER_GROUPS.items():
        if ticker in lst:
            return [x for x in lst if x != ticker]
    return PEER_GROUPS["Gold Miners"][:]

def get_similar_stocks(ticker: str, n: int = 5) -> List[str]:
    """Pairs via corrélation (sur 1 an) avec filtrage NaN/outliers."""
    base = _peer_candidates_for(ticker)
    base = [x for x in base if x != ticker]
    if not base:
        base = PEER_GROUPS["Gold Miners"][:]
    df_peers, _ = get_peer_data([ticker] + base, period="1y")
    if df_peers.empty or ticker not in df_peers.columns:
        return base[:n]
    ret = df_peers.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if ret.empty:
        return base[:n]
    corr = ret.corr()[ticker].drop(ticker).dropna()
    # filtre extrêmes improbables |corr|>0.999 (dégénéré)
    corr = corr[(corr.abs() < 0.999)]
    top = corr.nlargest(n).index.tolist() if not corr.empty else base[:n]
    return top

# ====== Outils risque / fair value / reco ======
def safe_info_number(info: dict, *keys, default=np.nan):
    cur = info or {}
    for k in keys:
        if k in cur and cur[k] is not None:
            try:
                return float(cur[k])
            except Exception:
                return default
    return default

def compute_drawdown(series: pd.Series) -> Tuple[pd.Series, float]:
    if series is None or series.empty:
        return pd.Series(dtype=float), np.nan
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return dd, float(dd.min()) if not dd.empty else np.nan

def compute_short_term_signals(df_wi: pd.DataFrame) -> dict:
    sig = {}
    if df_wi is None or df_wi.empty:
        return {"signals": {}, "score": 0.0}
    last = df_wi.iloc[-1]
    rsi = last.get('RSI', np.nan)
    if pd.notna(rsi):
        if rsi < 30:   sig['RSI'] = +0.3
        elif rsi > 70: sig['RSI'] = -0.3
        else:          sig['RSI'] =  0.0
    macd = last.get('MACD', np.nan); msig = last.get('MACD_Signal', np.nan)
    if pd.notna(macd) and pd.notna(msig):
        sig['MACD'] = +0.25 if macd > msig else -0.25
    up = last.get('BB_Upper', np.nan); lo = last.get('BB_Lower', np.nan); px = last.get('Close', np.nan)
    if pd.notna(up) and pd.notna(lo) and pd.notna(px):
        width = up - lo
        if width > 0:
            z = (px - (lo + up)/2) / (width/2)
            if z < -1: sig['BB'] = +0.15
            elif z > 1: sig['BB'] = -0.15
            else: sig['BB'] = 0.0
    clos = df_wi['Close'].dropna()
    if len(clos) >= 20:
        last20_hi = clos.iloc[-20:].max()
        last20_lo = clos.iloc[-20:].min()
        if px >= last20_hi: sig['Breakout20'] = +0.25
        elif px <= last20_lo: sig['Breakout20'] = -0.25
        else: sig['Breakout20'] = 0.0
    sma20 = last.get('SMA_20', np.nan); sma50 = last.get('SMA_50', np.nan)
    if pd.notna(sma20) and pd.notna(sma50):
        sig['Slope_20_50'] = +0.2 if sma20 > sma50 else -0.2
    score = float(np.clip(sum(sig.values()), -1.0, 1.0)) if sig else 0.0
    return {"signals": sig, "score": score}

def compute_medium_term_signals(df_wi: pd.DataFrame, bench_close: Optional[pd.Series]=None) -> dict:
    sig = {}
    if df_wi is None or df_wi.empty:
        return {"signals": {}, "score": 0.0}
    clos = df_wi['Close'].dropna()
    if len(clos) >= 126:
        m3 = clos.pct_change(63).iloc[-1]
        m6 = clos.pct_change(126).iloc[-1]
        sig['Momentum'] = float(np.clip((0.6*m6 + 0.4*m3)*3, -0.6, 0.6))
    sma200 = df_wi['SMA_200'].iloc[-1] if 'SMA_200' in df_wi else np.nan
    px = df_wi['Close'].iloc[-1]
    if pd.notna(sma200):
        sig['Above_SMA200'] = +0.25 if px > sma200 else -0.25
    if bench_close is not None and not bench_close.empty:
        idx = clos.index.intersection(bench_close.index)
        if len(idx) > 63:
            er_stock = clos.loc[idx].pct_change(63).iloc[-1]
            er_bench = bench_close.loc[idx].pct_change(63).iloc[-1]
            sig['Excess_3m'] = float(np.clip((er_stock - er_bench)*3, -0.4, 0.4))
    score = float(np.clip(sum(sig.values()), -1.0, 1.0)) if sig else 0.0
    return {"signals": sig, "score": score}

def detect_regime(df_wi: pd.DataFrame) -> str:
    clos = df_wi['Close'].dropna() if df_wi is not None else pd.Series(dtype=float)
    if len(clos) < 220 or 'SMA_200' not in df_wi:
        return "Range"
    sma200 = df_wi['SMA_200'].dropna()
    if len(sma200) < 50:
        return "Range"
    x = np.arange(len(sma200.tail(50)))
    slope, _ = np.polyfit(x, sma200.tail(50).values, 1)
    dd, dd_min = compute_drawdown(clos)
    if slope > 0 and dd.iloc[-1] > -0.15:
        return "Bull"
    if slope < 0 and dd_min < -0.2:
        return "Bear"
    return "Range"

def _clean_peer_metrics(dfm: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage des multiples pairs (supprime NaN et extrêmes)."""
    if dfm.empty:
        return dfm
    for col in ["pe", "ps", "ev_ebitda"]:
        if col in dfm:
            # remove negatives et extrêmes
            dfm.loc[(dfm[col] <= 0) | (dfm[col] > 200), col] = np.nan
    dfm = dfm.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    # Au moins 2 non-NaN sur l’ensemble
    if dfm[["pe","ps","ev_ebitda"]].count().sum() < 2:
        return pd.DataFrame(columns=dfm.columns)
    return dfm

def peer_valuation(ticker: str, peers: List[str], info_main: dict) -> dict:
    """Fair value via médiane des pairs (filtrée)."""
    metrics = []
    tick_all = [t for t in peers if t != ticker][:8]  # limite visuelle/perf
    for p in tick_all:
        try:
            inf = yf.Ticker(p).info
            metrics.append({
                "ticker": p,
                "pe": safe_info_number(inf, "trailingPE", "forwardPE"),
                "ps": safe_info_number(inf, "priceToSalesTrailing12Months"),
                "ev_ebitda": safe_info_number(inf, "enterpriseToEbitda"),
            })
        except Exception:
            pass
        time.sleep(0.05)
    dfm = pd.DataFrame(metrics)
    dfm = _clean_peer_metrics(dfm)
    out = {"peers_used": dfm["ticker"].dropna().tolist() if not dfm.empty else []}
    if dfm.empty:
        out["fair_value"] = {}
        return out

    peer_mult = {
        "pe": float(dfm["pe"].median(skipna=True)) if "pe" in dfm else np.nan,
        "ps": float(dfm["ps"].median(skipna=True)) if "ps" in dfm else np.nan,
        "ev_ebitda": float(dfm["ev_ebitda"].median(skipna=True)) if "ev_ebitda" in dfm else np.nan,
    }

    price   = safe_info_number(info_main, "currentPrice", "regularMarketPrice")
    shares  = safe_info_number(info_main, "sharesOutstanding", "floatShares")
    revenue = safe_info_number(info_main, "totalRevenue")
    ebitda  = safe_info_number(info_main, "ebitda")
    eps_ttm = safe_info_number(info_main, "trailingEps", "forwardEps")

    fair = {}
    if eps_ttm and peer_mult["pe"] and eps_ttm > 0 and peer_mult["pe"] > 0:
        fair["PE_based"] = float(eps_ttm * peer_mult["pe"])
    if revenue and shares and peer_mult["ps"] and peer_mult["ps"] > 0:
        fair["PS_based"] = float((revenue * peer_mult["ps"]) / shares)
    if ebitda and ebitda > 0 and peer_mult["ev_ebitda"] and peer_mult["ev_ebitda"] > 0 and shares:
        fair["EV_EBITDA_based"] = float((peer_mult["ev_ebitda"] * ebitda) / shares)

    vals = [v for v in fair.values() if v and v > 0]
    if vals:
        fair["composite_fair_price"] = float(np.median(vals))

    out["fair_value"] = fair
    out["peer_multiples_median"] = peer_mult
    out["current_price"] = price
    if "composite_fair_price" in fair and price:
        out["upside_%"] = (fair["composite_fair_price"] / price - 1.0) * 100.0
    return out

def risk_pack(price_series: pd.Series, bench_series: Optional[pd.Series]=None) -> dict:
    out = {}
    if price_series is None or price_series.empty:
        return out
    ret = price_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if not ret.empty:
        out["vol_annual_%"] = float(ret.std() * np.sqrt(252) * 100)
        out["VaR95_%"] = float(np.percentile(ret, 5) * 100)
    dd, dd_min = compute_drawdown(price_series)
    out["max_drawdown_%"] = float(dd_min * 100) if pd.notna(dd_min) else np.nan
    if bench_series is not None and not bench_series.empty:
        aligned = pd.concat([price_series.pct_change(), bench_series.pct_change()], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) >= 60:
            cov = aligned.iloc[:,0].rolling(60).cov(aligned.iloc[:,1])
            var = aligned.iloc[:,1].rolling(60).var()
            b = (cov/var).dropna()
            out["beta_60d"] = float(b.iloc[-1]) if not b.empty else np.nan
    return out

def build_recommendation(short_sig: dict, med_sig: dict, regime: str, fair: dict, risk: dict) -> dict:
    ct = short_sig.get("score", 0.0)
    mt = med_sig.get("score", 0.0)
    def to_view(s):
        if s >= 0.35: return "Haussier", 0.6
        if s <= -0.35: return "Baissier", 0.6
        if s >= 0.15: return "Lég. haussier", 0.5
        if s <= -0.15: return "Lég. baissier", 0.5
        return "Neutre", 0.45
    view_ct, conf_ct = to_view(ct)
    view_mt, conf_mt = to_view(mt)

    fv_up = fair.get("upside_%")
    if fv_up is not None and not np.isnan(fv_up):
        if fv_up > 15: view_lt, conf_lt = "Haussier (val.)", 0.55
        elif fv_up < -10: view_lt, conf_lt = "Baissier (val.)", 0.55
        else: view_lt, conf_lt = "Neutre (val.)", 0.5
    else:
        view_lt, conf_lt = ("Bull",0.52) if regime=="Bull" else (("Bear",0.52) if regime=="Bear" else ("Range",0.48))

    drivers = []
    for k,v in short_sig.get("signals", {}).items():
        if abs(v) >= 0.2:
            drivers.append(f"CT {k} {'+' if v>0 else '-'}")
    for k,v in med_sig.get("signals", {}).items():
        if abs(v) >= 0.25:
            drivers.append(f"MT {k} {'+' if v>0 else '-'}")

    risk_notes = []
    if risk.get("max_drawdown_%", 0) < -35:
        risk_notes.append("DD historique profond")
    if risk.get("VaR95_%") is not None and abs(risk.get("VaR95_%")) > 3.5:
        risk_notes.append("VaR(95) élevée")
    if abs(risk.get("beta_60d", 0)) > 1.3:
        try:
            risk_notes.append(f"Bêta élevé ({risk.get('beta_60d'):.2f})")
        except Exception:
            risk_notes.append("Bêta élevé")

    return {
        "short_term": {"view": view_ct, "confidence": conf_ct, "score": ct},
        "medium_term": {"view": view_mt, "confidence": conf_mt, "score": mt},
        "long_term": {"view": view_lt, "confidence": conf_lt, "valuation_upside_%": fv_up},
        "regime": regime,
        "drivers": drivers[:6],
        "risk_flags": risk_notes[:4],
        "fair_value": fair.get("fair_value", {}),
        "peer_multiples_median": fair.get("peer_multiples_median", {}),
    }

def pick_first_index(df: pd.DataFrame, *candidates):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    idx = set(map(str, df.index))
    for c in candidates:
        if c in idx:
            return c
    low = {str(i).lower(): str(i) for i in df.index}
    for c in candidates:
        key = str(c).lower()
        if key in low:
            return low[key]
    return None

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

if not ticker:
    st.warning("Veuillez entrer un symbole d'action valide.")
    st.stop()

# --------- Chargement des données (cache) ----------
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
def load_similar_stocks_cached(ticker):
    return get_similar_stocks(ticker)

@st.cache_data(ttl=3600)
def load_macro_indicators(period):
    data = {}
    for indicator, name in MACRO_INDICATORS.items():
        hist = get_stock_data(indicator, period=period)
        if hist is not None and not hist.empty:
            data[indicator] = hist["Close"]
        time.sleep(0.05)
    return pd.DataFrame(data)

with st.spinner("Chargement des données de l'action..."):
    stock_data = load_stock_data(ticker, period)
    if stock_data is None or stock_data.empty:
        st.error(f"Impossible de récupérer les données pour {ticker}. Vérifiez que le symbole est correct.")
        st.stop()
    stock_data_with_indicators = add_technical_indicators(stock_data)
    company_info = load_company_info(ticker)
    financials = load_financials(ticker)
    benchmark_data = load_benchmark_data(benchmark, period)
    try:
        similar_stocks = load_similar_stocks_cached(ticker) if compare_peers else []
    except Exception as e:
        st.caption(f"Pairs similaires indisponibles: {e}")
        similar_stocks = []
    macro_data = load_macro_indicators(period) if compare_macro else pd.DataFrame()

# --------- Informations entreprise ----------
if company_info:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        company_name = company_info.get("longName", ticker)
        st.header(f"{company_name} ({ticker})")
        st.markdown(f"**Secteur:** {company_info.get('sector', 'N/A')} | **Industrie:** {company_info.get('industry', 'N/A')}")
        st.markdown(company_info.get("longBusinessSummary", "Aucune description disponible."))
    with col2:
        st.subheader("Données de marché")
        current_price = float(stock_data["Close"].iloc[-1])
        previous_close = float(stock_data["Close"].iloc[-2]) if len(stock_data) > 1 else current_price
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100 if previous_close else 0.0
        st.metric("Prix actuel", f"{current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
        st.metric("Volume moyen (30j)", f"{stock_data['Volume'].tail(30).mean():.0f}")
        st.metric("Capitalisation", f"{_fmt_mcap(company_info.get('marketCap', 0))}")
    with col3:
        st.subheader("Valorisation")
        st.metric("P/E", f"{company_info.get('trailingPE', 'N/A')}")
        st.metric("P/B", f"{company_info.get('priceToBook', 'N/A')}")
        st.metric("EV/EBITDA", f"{company_info.get('enterpriseToEbitda', 'N/A')}")

# --------- Graphique principal ----------
st.subheader("Évolution du cours")
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3],
    subplot_titles=(f"Cours de {ticker}", "Volume")
)
fig.add_trace(go.Candlestick(
    x=stock_data.index, open=stock_data['Open'], high=stock_data['High'],
    low=stock_data['Low'], close=stock_data['Close'], name=ticker
), row=1, col=1)

if show_sma:
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['SMA_20'], line=dict(width=1), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['SMA_50'], line=dict(width=1), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['SMA_200'], line=dict(width=1), name='SMA 200'), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['BB_Upper'], line=dict(width=1), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['BB_Lower'], line=dict(width=1), fill='tonexty', fillcolor='rgba(0,128,0,0.08)', name='BB Lower'), row=1, col=1)

fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'), row=2, col=1)
fig.update_layout(height=600, xaxis_rangeslider_visible=False, hovermode="x unified",
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# --------- Indicateurs techniques ----------
st.subheader("Indicateurs techniques")
col1, col2 = st.columns(2)
with col1:
    if show_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['RSI'], line=dict(width=1), name='RSI'))
        fig_rsi.add_shape(type="line", x0=stock_data.index[0], y0=70, x1=stock_data.index[-1], y1=70, line=dict(width=1, dash="dash"))
        fig_rsi.add_shape(type="line", x0=stock_data.index[0], y0=30, x1=stock_data.index[-1], y1=30, line=dict(width=1, dash="dash"))
        fig_rsi.update_layout(title="RSI (14)", yaxis=dict(range=[0, 100]), height=300, hovermode="x unified")
        st.plotly_chart(fig_rsi, use_container_width=True)
with col2:
    if show_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['MACD'], line=dict(width=1), name='MACD'))
        fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data_with_indicators['MACD_Signal'], line=dict(width=1), name='Signal'))
        colors = ['green' if val > 0 else 'red' for val in stock_data_with_indicators['MACD_Hist']]
        fig_macd.add_trace(go.Bar(x=stock_data.index, y=stock_data_with_indicators['MACD_Hist'], marker_color=colors, name='Histogramme'))
        fig_macd.update_layout(title="MACD", height=300, hovermode="x unified")
        st.plotly_chart(fig_macd, use_container_width=True)

# --------- Analyse de performance ----------
st.subheader("Analyse de performance")
periods = {
    "1 semaine": 5,
    "1 mois": 21,
    "3 mois": 63,
    "6 mois": 126,
    "1 an": 252,
    "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days
}

comparison_data = pd.DataFrame()
comparison_data[ticker] = stock_data['Close']
if benchmark_data is not None and not benchmark_data.empty:
    comparison_data[benchmark] = benchmark_data['Close']

# Peers “corrélés” (limités à 5)
if compare_peers:
    peers = get_similar_stocks(ticker, n=5)
    if peers:
        peer_df, valid_peers = get_peer_data(peers, period=period)
        for peer in valid_peers:
            comparison_data[peer] = peer_df[peer]

if not comparison_data.empty:
    comparison_normalized = comparison_data.dropna().div(comparison_data.iloc[0]) * 100
    try:
        fig_comp = px.line(
            comparison_normalized,
            x=comparison_normalized.index,
            y=comparison_normalized.columns,
            title="Performance relative (base 100)",
            labels={"value": "Performance (%)", "variable": "Symbole"}
        )
        fig_comp.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_comp, use_container_width=True)
    except Exception as e:
        st.caption(f"Graphique comparaison indisponible: {e}")
else:
    st.info("Pas assez de données pour l'analyse comparative")

# Tableau de rendements (%), fallback si matplotlib absent
if not comparison_data.empty and len(comparison_data) > 1:
    try:
        returns_df = calculate_returns(comparison_data, periods)
        if not returns_df.empty:
            styler = returns_df.T.style.format("{:.2f}%")
            if _has_matplotlib():
                try:
                    styler = styler.background_gradient(cmap="RdYlGn", axis=1)
                except Exception:
                    pass
            else:
                st.caption("Astuce: installez matplotlib pour la mise en couleur des rendements.")
            st.dataframe(styler, use_container_width=True)
        else:
            st.info("Pas assez de données pour calculer les rendements")
    except Exception as e:
        st.caption(f"Affichage rendements — fallback: {e}")
        st.dataframe(returns_df.T if 'returns_df' in locals() else comparison_data.tail(5))
else:
    st.info("Pas assez de données pour calculer les rendements")

# --------- Analyse de risque ----------
st.subheader("Analyse de risque")
col1, col2 = st.columns(2)

with col1:
    try:
        if not comparison_data.empty and len(comparison_data) > 20:
            volatility = calculate_volatility(comparison_data)
            if not volatility.empty:
                fig_vol = px.line(
                    volatility,
                    x=volatility.index,
                    y=volatility.columns,
                    title="Volatilité annualisée (fenêtre 20 jours)",
                    labels={"value": "Volatilité (%)", "variable": "Symbole"}
                )
                fig_vol.update_layout(height=300, hovermode="x unified")
                st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.info("Pas assez de données pour calculer la volatilité")
        else:
            st.info("Pas assez de données pour calculer la volatilité (≥ 20 jours requis)")
    except Exception as e:
        st.caption(f"Volatilité — indisponible ({e})")

with col2:
    try:
        if benchmark in comparison_data.columns and ticker in comparison_data.columns and len(comparison_data) > 60:
            stock_returns = comparison_data[ticker].pct_change().dropna()
            benchmark_returns = comparison_data[benchmark].pct_change().dropna()
            beta = calculate_beta(stock_returns, benchmark_returns)
            if not beta.empty and not beta.isna().all():
                beta_df = beta.to_frame(name="beta")
                fig_beta = px.line(
                    beta_df, x=beta_df.index, y="beta",
                    title=f"Bêta glissant vs {BENCHMARKS.get(benchmark, benchmark)}",
                    labels={"beta": "Bêta", "index": "Date"}
                )
                fig_beta.update_layout(height=300, hovermode="x unified")
                st.plotly_chart(fig_beta, use_container_width=True)
            else:
                st.info("Impossible de calculer le bêta avec les données disponibles")
        else:
            st.info("Pas assez de données pour le bêta (≥ 60 jours et benchmark)")
    except Exception as e:
        st.caption(f"Bêta — indisponible ({e})")

# --------- Corrélation avec les indicateurs macro ----------
if compare_macro:
    st.subheader("Corrélation avec les indicateurs macroéconomiques")

if compare_macro and 'Close' in stock_data and not stock_data['Close'].empty:
    macro_data = macro_data if isinstance(macro_data, pd.DataFrame) else pd.DataFrame()
    if not macro_data.empty:
        macro_comparison = pd.concat([stock_data['Close'], macro_data], axis=1).dropna()
        macro_comparison.columns = [ticker] + list(MACRO_INDICATORS.values())
        try:
            corr_matrix = macro_comparison.pct_change().corr()
            fig_corr = px.imshow(
                corr_matrix, x=corr_matrix.columns, y=corr_matrix.columns,
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Matrice de corrélation"
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.caption(f"Matrice corrélation indisponible: {e}")

        st.subheader("Relation avec les indicateurs clés")
        try:
            correlations = corr_matrix[ticker].drop(ticker).abs().sort_values(ascending=False)
            top_correlated = correlations.head(2).index.tolist()
        except Exception:
            top_correlated = []

        col1, col2 = st.columns(2)
        for i, indicator in enumerate(top_correlated):
            with (col1 if i == 0 else col2):
                try:
                    scatter_data = pd.DataFrame({
                        'x': macro_comparison[indicator].pct_change(),
                        'y': macro_comparison[ticker].pct_change()
                    }).dropna()
                    if len(scatter_data) > 5:
                        corr = scatter_data['x'].corr(scatter_data['y'])
                        fig_scatter = px.scatter(
                            scatter_data, x='x', y='y',
                            title=f"{ticker} vs {indicator} (Corr: {corr:.2f})",
                            labels={"x": f"{indicator} (variation journalière %)",
                                    "y": f"{ticker} (variation journalière %)"}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                except Exception as e:
                    st.caption(f"Dispersion {indicator} indisponible: {e}")
    else:
        st.info("Indicateurs macro indisponibles pour la période sélectionnée.")

# --------- Données financières ----------
st.subheader("Données financières")
has_any_fin = False
if isinstance(financials, dict):
    tabs = st.tabs(["Compte de résultat", "Bilan", "Flux de trésorerie", "Ratios clés"])
    with tabs[0]:
        df = financials.get("income_stmt", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            has_any_fin = True
            st.dataframe(df.T, use_container_width=True)
        else:
            st.info("Données du compte de résultat non disponibles.")
    with tabs[1]:
        df = financials.get("balance_sheet", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            has_any_fin = True
            st.dataframe(df.T, use_container_width=True)
        else:
            st.info("Données du bilan non disponibles.")
    with tabs[2]:
        df = financials.get("cash_flow", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            has_any_fin = True
            st.dataframe(df.T, use_container_width=True)
        else:
            st.info("Données des flux de trésorerie non disponibles.")
    with tabs[3]:
        try:
            income = financials.get("income_stmt", pd.DataFrame())
            balance = financials.get("balance_sheet", pd.DataFrame())
            if isinstance(income, pd.DataFrame) and isinstance(balance, pd.DataFrame) and not income.empty and not balance.empty:
                has_any_fin = True
                ratios = pd.DataFrame(index=income.columns)

                net_income_row = pick_first_index(income, "Net Income", "NetIncome", "Net Income Common Stockholders")
                revenue_row    = pick_first_index(income, "Total Revenue", "TotalRevenue", "Revenue")
                equity_row     = pick_first_index(balance, "Total Stockholder Equity", "Total Stockholders Equity", "Stockholders Equity")
                assets_row     = pick_first_index(balance, "Total Assets")
                curr_assets    = pick_first_index(balance, "Current Assets")
                curr_liabs     = pick_first_index(balance, "Current Liabilities")
                tot_liabs_row  = pick_first_index(balance, "Total Liabilities Net Minority Interest", "Total Liabilities")

                if net_income_row and revenue_row:
                    ratios["Marge nette (%)"] = (income.loc[net_income_row] / income.loc[revenue_row]) * 100
                if net_income_row and equity_row:
                    ratios["ROE (%)"] = (income.loc[net_income_row] / balance.loc[equity_row]) * 100
                if net_income_row and assets_row:
                    ratios["ROA (%)"] = (income.loc[net_income_row] / balance.loc[assets_row]) * 100

                if curr_assets and curr_liabs:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        ratios["Ratio de liquidité"] = balance.loc[curr_assets] / balance.loc[curr_liabs]

                if assets_row and tot_liabs_row:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        ratios["Ratio d'endettement (%)"] = (balance.loc[tot_liabs_row] / balance.loc[assets_row]) * 100

                mcap = company_info.get("marketCap", np.nan)
                if isinstance(mcap, (int, float)) and mcap and not np.isnan(mcap):
                    try:
                        if net_income_row and income.loc[net_income_row].iloc[0] not in (0, np.nan):
                            ratios["P/E (estimé)"] = mcap / income.loc[net_income_row].iloc[0]
                    except Exception:
                        pass
                    try:
                        if revenue_row and income.loc[revenue_row].iloc[0] not in (0, np.nan):
                            ratios["P/S (estimé)"] = mcap / income.loc[revenue_row].iloc[0]
                    except Exception:
                        pass
                    try:
                        if assets_row and tot_liabs_row:
                            book_value = balance.loc[assets_row].iloc[0] - balance.loc[tot_liabs_row].iloc[0]
                            ratios["P/B (estimé)"] = (mcap / book_value) if (isinstance(book_value, (int, float)) and book_value not in (0, np.nan)) else np.nan
                    except Exception:
                        pass

                st.dataframe(ratios.T, use_container_width=True)
            else:
                st.info("Données insuffisantes pour calculer les ratios financiers.")
        except Exception as e:
            st.caption(f"Ratios — indisponibles ({e})")
else:
    st.info("Données financières non disponibles pour cette action.")

# --------- Analyse macroéconomique approfondie ----------
st.subheader("Analyse macroéconomique approfondie")
show_macro_analysis = st.checkbox("Afficher l'analyse macroéconomique approfondie", value=False)

if show_macro_analysis:
    st.write("Sélectionnez les indicateurs économiques à analyser:")
    col1, col2 = st.columns(2)
    with col1:
        selected_inflation = st.multiselect("Inflation", ["CPIAUCSL", "T10YIE"], default=["CPIAUCSL"],
                                            format_func=lambda x: f"{x} - {FRED_SERIES[x]}")
        selected_growth = st.multiselect("Croissance", ["INDPRO", "GDPC1"], default=["INDPRO"],
                                         format_func=lambda x: f"{x} - {FRED_SERIES[x]}")
    with col2:
        selected_rates = st.multiselect("Taux d'intérêt", ["DGS10", "DGS2"], default=["DGS10"],
                                        format_func=lambda x: f"{x} - {FRED_SERIES[x]}")
        selected_other = st.multiselect("Autres indicateurs", ["UNRATE", "DTWEXBGS", "NFCI", "USREC"], default=["UNRATE"],
                                        format_func=lambda x: f"{x} - {FRED_SERIES[x]}")

    selected_indicators = selected_inflation + selected_growth + selected_rates + selected_other
    if selected_indicators:
        fred_start_date = datetime.now() - timedelta(days=365*5)
        with st.spinner("Récupération des données économiques en cours..."):
            fred_data = get_fred_data(selected_indicators, fred_start_date)
        if not fred_data.empty:
            st.subheader("Évolution des indicateurs économiques")
            macro_tabs = st.tabs(["Inflation", "Croissance", "Taux d'intérêt", "Autres", "Impact sur l'action"])
            with macro_tabs[0]:
                if selected_inflation:
                    cols = [FRED_SERIES[col] for col in selected_inflation if FRED_SERIES.get(col) in fred_data.columns]
                    inflation_data = fred_data[cols] if cols else pd.DataFrame()
                    if not inflation_data.empty:
                        fig = px.line(inflation_data, x=inflation_data.index, y=inflation_data.columns, title="Indicateurs d'inflation")
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                        if len(inflation_data) > 252:
                            annual_change = inflation_data.pct_change(252).iloc[-1] * 100
                            st.write("Variation sur 12 mois:")
                            st.dataframe(annual_change.to_frame("Variation (%)").T, use_container_width=True)
                    else:
                        st.info("Aucune donnée d'inflation disponible pour la période sélectionnée.")
                else:
                    st.info("Aucun indicateur d'inflation sélectionné.")
            with macro_tabs[1]:
                if selected_growth:
                    cols = [FRED_SERIES[col] for col in selected_growth if FRED_SERIES.get(col) in fred_data.columns]
                    growth_data = fred_data[cols] if cols else pd.DataFrame()
                    if not growth_data.empty:
                        fig = px.line(growth_data, x=growth_data.index, y=growth_data.columns, title="Indicateurs de croissance économique")
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                        if len(growth_data) > 252:
                            annual_change = growth_data.pct_change(252).iloc[-1] * 100
                            st.write("Variation sur 12 mois:")
                            st.dataframe(annual_change.to_frame("Variation (%)").T, use_container_width=True)
                    else:
                        st.info("Aucune donnée de croissance disponible pour la période sélectionnée.")
                else:
                    st.info("Aucun indicateur de croissance sélectionné.")
            with macro_tabs[2]:
                if selected_rates:
                    cols = [FRED_SERIES[col] for col in selected_rates if FRED_SERIES.get(col) in fred_data.columns]
                    rates_data = fred_data[cols] if cols else pd.DataFrame()
                    if not rates_data.empty:
                        fig = px.line(rates_data, x=rates_data.index, y=rates_data.columns, title="Évolution des taux d'intérêt")
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                        if len(rates_data) > 63:
                            change_3m = rates_data.iloc[-1] - rates_data.iloc[-63]
                            st.write("Variation absolue sur 3 mois (points de base):")
                            st.dataframe((change_3m * 100).to_frame("Variation (pb)").T, use_container_width=True)
                    else:
                        st.info("Aucune donnée de taux d'intérêt disponible pour la période sélectionnée.")
                else:
                    st.info("Aucun indicateur de taux d'intérêt sélectionné.")
            with macro_tabs[3]:
                if selected_other:
                    cols = [FRED_SERIES[col] for col in selected_other if FRED_SERIES.get(col) in fred_data.columns]
                    other_data = fred_data[cols] if cols else pd.DataFrame()
                    if not other_data.empty:
                        fig = px.line(other_data, x=other_data.index, y=other_data.columns, title="Autres indicateurs économiques")
                        fig.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Aucune donnée disponible pour la période sélectionnée.")
                else:
                    st.info("Aucun autre indicateur sélectionné.")
            with macro_tabs[4]:
                st.write("Analyse de l'impact des facteurs macroéconomiques sur l'action")
                try:
                    stock_monthly = stock_data['Close'].resample('M').last()
                    stock_returns_m = stock_monthly.pct_change().dropna()
                    fred_monthly = fred_data.resample('M').last()
                    merged_data = pd.concat([stock_returns_m, fred_monthly], axis=1).dropna()
                    merged_data.columns = [ticker] + list(fred_monthly.columns)
                    if not merged_data.empty and len(merged_data) > 24:
                        corr_matrix = merged_data.corr()[ticker].drop(ticker).sort_values(ascending=False)
                        st.write("Corrélation entre les rendements mensuels de l'action et les indicateurs économiques:")
                        fig_corr = px.bar(corr_matrix, title="Impact des facteurs économiques sur l'action",
                                          labels={"value": "Corrélation", "index": "Indicateur"})
                        fig_corr.update_layout(height=400)
                        st.plotly_chart(fig_corr, use_container_width=True)

                        if ticker in SECTOR_SENSITIVITY.columns:
                            st.write("Sensibilité théorique de l'action aux facteurs économiques:")
                            sensitivity = SECTOR_SENSITIVITY[ticker].dropna()
                            fig_sens = px.bar(sensitivity,
                                              title="Sensibilité théorique aux facteurs économiques",
                                              labels={"value": "Sensibilité (-2 à +2)", "index": "Facteur"})
                            fig_sens.update_layout(height=300)
                            st.plotly_chart(fig_sens, use_container_width=True)
                            st.write("**Interprétation:** +2 très positif / -2 très négatif")
                    else:
                        st.info("Données insuffisantes pour analyser l'impact des facteurs économiques.")
                except Exception as e:
                    st.caption(f"Impact macro — indisponible ({e})")
        else:
            st.warning("Impossible de récupérer les données économiques. Réessayez plus tard.")
    else:
        st.info("Veuillez sélectionner au moins un indicateur économique pour l'analyse.")

# --------- Recommandation (expérimentale) ----------
st.subheader("🎯 Recommandation (expérimentale)")
bench_close = benchmark_data['Close'] if (isinstance(benchmark_data, pd.DataFrame) and not benchmark_data.empty and 'Close' in benchmark_data) else None
short_sig = compute_short_term_signals(stock_data_with_indicators)
med_sig   = compute_medium_term_signals(stock_data_with_indicators, bench_close)
regime    = detect_regime(stock_data_with_indicators)
peer_list = similar_stocks if (compare_peers and similar_stocks) else _peer_candidates_for(ticker)
fair      = peer_valuation(ticker, peer_list, company_info)
risk      = risk_pack(stock_data['Close'], bench_close)
reco      = build_recommendation(short_sig, med_sig, regime, fair, risk)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Court terme", f"{reco['short_term']['view']}", f"score {reco['short_term']['score']:+.2f}")
with c2:
    st.metric("Moyen terme", f"{reco['medium_term']['view']}", f"score {reco['medium_term']['score']:+.2f}")
with c3:
    lt_lab = reco['long_term']['view']
    up = reco.get('long_term',{}).get('valuation_upside_%')
    delta = f"{up:+.1f}%" if (up is not None and not pd.isna(up)) else "n/a"
    st.metric("Long terme", lt_lab, delta=delta)

st.write(f"**Régime détecté** : `{reco['regime']}`")
if reco["drivers"]:
    st.write("**Drivers** :", " · ".join(reco["drivers"]))
if reco["risk_flags"]:
    st.write("**Drapeaux risque** :", " · ".join(reco["risk_flags"]))

fv = reco.get("fair_value", {})
if fv:
    st.markdown("**Fair value (multiples pairs)**")
    st.json(fv)
    pm = reco.get("peer_multiples_median", {})
    if pm:
        try:
            st.caption(f"Multiples pairs (médian): PE={pm.get('pe', np.nan):.2f}, EV/EBITDA={pm.get('ev_ebitda', np.nan):.2f}, P/S={pm.get('ps', np.nan):.2f}")
        except Exception:
            pass

# --------- Résumé et analyse “classique” ----------
st.subheader("Résumé et analyse")
try:
    current_price = float(stock_data['Close'].iloc[-1])
    sma_20 = float(stock_data_with_indicators['SMA_20'].iloc[-1]) if 'SMA_20' in stock_data_with_indicators and not stock_data_with_indicators['SMA_20'].isna().iloc[-1] else None
    sma_50 = float(stock_data_with_indicators['SMA_50'].iloc[-1]) if 'SMA_50' in stock_data_with_indicators and not stock_data_with_indicators['SMA_50'].isna().iloc[-1] else None
    sma_200 = float(stock_data_with_indicators['SMA_200'].iloc[-1]) if 'SMA_200' in stock_data_with_indicators and not stock_data_with_indicators['SMA_200'].isna().iloc[-1] else None
    rsi = float(stock_data_with_indicators['RSI'].iloc[-1]) if 'RSI' in stock_data_with_indicators and not stock_data_with_indicators['RSI'].isna().iloc[-1] else None
    macd = float(stock_data_with_indicators['MACD'].iloc[-1]) if 'MACD' in stock_data_with_indicators and not stock_data_with_indicators['MACD'].isna().iloc[-1] else None
    macd_signal = float(stock_data_with_indicators['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in stock_data_with_indicators and not stock_data_with_indicators['MACD_Signal'].isna().iloc[-1] else None
except Exception as e:
    st.caption(f"Indicateurs techniques — indisponibles ({e})")
    current_price = sma_20 = sma_50 = sma_200 = rsi = macd = macd_signal = None

if current_price is not None and sma_50 is not None:
    price_trend = "haussière" if current_price > sma_50 else "baissière"
    st.markdown(f"**Tendance de prix:** {price_trend}")
else:
    st.markdown("**Tendance de prix:** Données insuffisantes")

signals = []
if current_price is not None and sma_20 is not None:
    signals.append("Prix au-dessus de la SMA 20 ✅" if current_price > sma_20 else "Prix en-dessous de la SMA 20 ❌")
else:
    signals.append("SMA 20: Données insuffisantes ℹ️")
if current_price is not None and sma_50 is not None:
    signals.append("Prix au-dessus de la SMA 50 ✅" if current_price > sma_50 else "Prix en-dessous de la SMA 50 ❌")
else:
    signals.append("SMA 50: Données insuffisantes ℹ️")
if current_price is not None and sma_200 is not None:
    signals.append("Prix au-dessus de la SMA 200 ✅" if current_price > sma_200 else "Prix en-dessous de la SMA 200 ❌")
else:
    signals.append("SMA 200: Données insuffisantes ℹ️")
if sma_20 is not None and sma_50 is not None:
    signals.append("SMA 20 au-dessus de SMA 50 (signal haussier) ✅" if sma_20 > sma_50 else "SMA 20 en-dessous de SMA 50 (signal baissier) ❌")
else:
    signals.append("Croisement SMA: Données insuffisantes ℹ️")
if rsi is not None:
    if rsi > 70: signals.append(f"RSI en zone de surachat ({rsi:.1f}) ⚠️")
    elif rsi < 30: signals.append(f"RSI en zone de survente ({rsi:.1f}) ⚠️")
    else: signals.append(f"RSI en zone neutre ({rsi:.1f}) ✓")
else:
    signals.append("RSI: Données insuffisantes ℹ️")
if macd is not None and macd_signal is not None:
    signals.append("MACD au-dessus de la ligne de signal (signal haussier) ✅" if macd > macd_signal else "MACD en-dessous de la ligne de signal (signal baissier) ❌")
else:
    signals.append("MACD: Données insuffisantes ℹ️")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Signaux techniques:**")
    for s in signals[:3]:
        st.markdown(f"- {s}")
with c2:
    st.markdown("&nbsp;")
    for s in signals[3:]:
        st.markdown(f"- {s}")

st.markdown("---")
st.markdown("**Conclusion:**")
positive_signals = sum(1 for s in signals if "✅" in s)
negative_signals = sum(1 for s in signals if "❌" in s)
warning_signals = sum(1 for s in signals if "⚠️" in s)
if positive_signals > negative_signals + warning_signals:
    st.markdown("L'analyse technique suggère une tendance globalement **positive**. La majorité des indicateurs sont haussiers, mais surveillez la dynamique et les fondamentaux.")
elif negative_signals > positive_signals + warning_signals:
    st.markdown("L'analyse technique suggère une tendance globalement **négative**. La majorité des indicateurs sont baissiers ; prudence et confirmation requises.")
else:
    st.markdown("L'analyse technique suggère une tendance **mixte**. Les signaux sont partagés, possiblement une phase de consolidation/incertitude.")

# --------- Prévisions à long terme ----------
st.subheader("Prévisions à long terme")
show_forecasts = st.checkbox("Afficher les prévisions à long terme", value=False)
if show_forecasts:
    try:
        if len(stock_data) > 252:
            forecast_method = st.selectbox("Méthode de prévision",
                                           ["Tendance simple", "Moyenne mobile", "Régression linéaire", "ARIMA", "Prophet", "Modèle hybride"])
            forecast_horizon = st.slider("Horizon de prévision (jours)", 30, 365, 180)
            close_prices = stock_data['Close']
            dates = close_prices.index
            last_date = dates[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)

            with st.expander("Options avancées de prévision"):
                confidence_level = st.slider("Niveau de confiance (%)", 50, 95, 80) / 100
                include_macro = st.checkbox("Inclure des facteurs macroéconomiques (placeholders)", value=False)
                selected_macro_factors = []
                if include_macro:
                    selected_macro_factors = st.multiselect("Sélectionner les facteurs macroéconomiques à inclure",
                                                            ["Taux d'intérêt", "Inflation", "Dollar US"], default=["Taux d'intérêt"])
                use_cross_validation = st.checkbox("Utiliser la validation croisée (démo)", value=False)
                if use_cross_validation:
                    _ = st.slider("Nombre de périodes de validation", 3, 10, 5)
                    _ = st.slider("Taille de la fenêtre de validation (jours)", 30, 180, 60)

            forecast_df = pd.DataFrame()

            if forecast_method == "Tendance simple":
                recent = close_prices[-126:] if len(close_prices) > 126 else close_prices
                x = np.arange(len(recent))
                slope, intercept = np.polyfit(x, recent, 1)
                future_x = np.arange(len(recent), len(recent) + forecast_horizon)
                forecast = slope * future_x + intercept
                y_pred = slope * x + intercept
                rmse = float(np.sqrt(np.mean((recent - y_pred) ** 2)))
                std_error = rmse * np.sqrt(1 + 1/len(x) + (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
                z_value = 1.96 if confidence_level >= 0.95 else (1.645 if confidence_level >= 0.9 else 1.28)
                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast,
                                            'Lower': forecast - z_value * std_error,
                                            'Upper': forecast + z_value * std_error}).set_index('Date')
                st.write(f"**Méthode de tendance simple:** basée sur {len(recent)} derniers jours.")
                st.write(f"Tendance quotidienne moyenne: {slope:.4f} {company_info.get('currency', '$')}/jour")
                st.write(f"RMSE: {rmse:.4f}")

            elif forecast_method == "Moyenne mobile":
                ema_span = st.slider("Période EMA (jours)", 20, 200, 50)
                ema = close_prices.ewm(span=ema_span, adjust=False).mean()
                recent_ema = ema[-60:]
                x = np.arange(len(recent_ema))
                slope, intercept = np.polyfit(x, recent_ema, 1)
                future_x = np.arange(len(recent_ema), len(recent_ema) + forecast_horizon)
                forecast = slope * future_x + intercept
                volatility = close_prices.pct_change().std() * np.sqrt(252)
                daily_vol = float(volatility) / np.sqrt(252) if pd.notna(volatility) else 0.02
                z_value = 1.96 if confidence_level >= 0.95 else (1.645 if confidence_level >= 0.9 else 1.28)
                time_factors = np.sqrt(np.arange(1, forecast_horizon + 1))
                uncertainty = np.array([daily_vol * tf * z_value for tf in time_factors])
                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast,
                                            'Lower': forecast * (1 - uncertainty),
                                            'Upper': forecast * (1 + uncertainty)}).set_index('Date')
                st.write(f"**EMA {ema_span}j :** tendance + bande d'incertitude proportionnelle à la vol.")

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
                    df['ma20'] = close_prices.rolling(window=20).mean().fillna(method='bfill')
                    df['ma50'] = close_prices.rolling(window=50).mean().fillna(method='bfill')
                    df['volatility'] = close_prices.rolling(window=20).std().fillna(method='bfill')
                    if include_macro and selected_macro_factors:
                        if "Taux d'intérêt" in selected_macro_factors: df['interest_rate'] = np.random.normal(2.5, 0.5, len(df))
                        if "Dollar US" in selected_macro_factors: df['usd_index'] = np.random.normal(100, 5, len(df))
                        if "Inflation" in selected_macro_factors: df['inflation'] = np.random.normal(2.0, 0.3, len(df))
                    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
                    dow_dummies   = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True)
                    quarter_dummies = pd.get_dummies(df['quarter'], prefix='quarter', drop_first=True)
                    X_columns = ['trend','ma20','ma50','volatility']
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
                    future_dow_dummies   = pd.get_dummies(future_df['day_of_week'], prefix='dow', drop_first=True)
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
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast,
                                                'Lower': lower, 'Upper': upper}).set_index('Date')
                    st.write("**Régression linéaire (avec dummies calendrier et features simples)**")
                    st.write(f"R² du modèle: {model.rsquared:.4f}")
                else:
                    st.info("`statsmodels` non disponible — fallback Tendance simple.")
                    recent = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent))
                    slope, intercept = np.polyfit(x, recent, 1)
                    future_x = np.arange(len(recent), len(recent) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast,
                                                'Lower': forecast * 0.9, 'Upper': forecast * 1.1}).set_index('Date')

            elif forecast_method == "ARIMA":
                import importlib.util
                has_statsmodels = importlib.util.find_spec('statsmodels') is not None
                if has_statsmodels:
                    from statsmodels.tsa.arima.model import ARIMA
                    best_order = (2, 1, 2)
                    try:
                        from pmdarima import auto_arima
                        with st.spinner("Sélection des paramètres ARIMA..."):
                            auto_model = auto_arima(close_prices, seasonal=False, error_action='ignore',
                                                    suppress_warnings=True, stepwise=True, n_jobs=-1)
                            best_order = auto_model.order
                            st.caption(f"Paramètres ARIMA sélectionnés: {best_order}")
                    except Exception:
                        st.caption("`pmdarima` indisponible — ARIMA(2,1,2) par défaut.")

                    model = ARIMA(close_prices, order=best_order)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=forecast_horizon)
                    forecast_ci = model_fit.get_forecast(steps=forecast_horizon).conf_int(alpha=1-confidence_level)
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast,
                                                'Lower': forecast_ci.iloc[:, 0].values,
                                                'Upper': forecast_ci.iloc[:, 1].values}).set_index('Date')
                    st.caption(f"ARIMA{best_order} — AIC: {model_fit.aic:.2f}")
                else:
                    st.info("`statsmodels` indisponible — fallback Tendance simple.")
                    recent = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent))
                    slope, intercept = np.polyfit(x, recent, 1)
                    future_x = np.arange(len(recent), len(recent) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast,
                                                'Lower': forecast * 0.9, 'Upper': forecast * 1.1}).set_index('Date')

            elif forecast_method == "Prophet":
                import importlib.util
                has_prophet = importlib.util.find_spec('prophet') is not None
                if has_prophet:
                    from prophet import Prophet
                    prophet_df = close_prices.reset_index().rename(columns={"Date": "ds", "Close": "y"})
                    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
                    model.fit(prophet_df)
                    future = pd.DataFrame({"ds": future_dates})
                    forecast = model.predict(future)
                    forecast_df = forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].rename(
                        columns={"yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"}
                    )
                    st.write("**Prophet (saisonnalité annuelle)**")
                else:
                    st.info("Prophet non installé — fallback Tendance simple.")
                    recent = close_prices[-126:] if len(close_prices) > 126 else close_prices
                    x = np.arange(len(recent))
                    slope, intercept = np.polyfit(x, recent, 1)
                    future_x = np.arange(len(recent), len(recent) + forecast_horizon)
                    forecast = slope * future_x + intercept
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast,
                                                'Lower': forecast * 0.9, 'Upper': forecast * 1.1}).set_index('Date')

            else:  # Modèle hybride
                try:
                    recent = close_prices[-60:]
                    x = np.arange(len(recent))
                    slope, intercept = np.polyfit(x, recent, 1)
                    base_future = slope * np.arange(len(recent), len(recent) + forecast_horizon) + intercept
                    ema20 = close_prices.ewm(span=20, adjust=False).mean().iloc[-1]
                    drift = (recent.iloc[-1] - ema20) * 0.15
                    forecast = base_future + drift
                    daily_vol = close_prices.pct_change().std()
                    daily_vol = float(daily_vol) if pd.notna(daily_vol) and daily_vol > 0 else 0.02
                    z = 1.96 if confidence_level >= 0.95 else (1.645 if confidence_level >= 0.9 else 1.28)
                    bands = np.sqrt(np.arange(1, forecast_horizon + 1)) * daily_vol * z * recent.iloc[-1]
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast,
                                                'Lower': forecast - bands, 'Upper': forecast + bands}).set_index('Date')
                    st.write("**Modèle hybride (tendance 60j + drift EMA + bandes vol)**")
                except Exception as e:
                    st.caption(f"Hybride — indisponible ({e})")

            # Affichage prévisions
            if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
                hist_df = stock_data[['Close']].rename(columns={'Close': 'Historique'})
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Historique'], name="Historique", mode="lines"))
                fig_f.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], name="Prévision", mode="lines"))
                if 'Lower' in forecast_df.columns and 'Upper' in forecast_df.columns:
                    fig_f.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper'], name="Upper", mode="lines",
                                               line=dict(width=0), showlegend=False))
                    fig_f.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower'], name="Lower", mode="lines",
                                               fill='tonexty', fillcolor='rgba(0, 0, 200, 0.1)', line=dict(width=0),
                                               showlegend=False))
                fig_f.update_layout(title="Projection de prix", hovermode="x unified", height=450)
                st.plotly_chart(fig_f, use_container_width=True)
            else:
                st.info("Aucune prévision calculée.")
        else:
            st.info("Données insuffisantes (≥ 252 jours) pour calculer des prévisions.")
    except Exception as e:
        st.caption(f"Prévisions — indisponibles ({e})")

st.caption("⚠️ Ceci n'est pas un conseil financier. Faites vos propres recherches (DYOR).")