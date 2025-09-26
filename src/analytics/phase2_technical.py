# phase2_technical.py
# -*- coding: utf-8 -*-
"""
Phase 2 — Analyse Technique & Backtests (actions US/CA, daily par défaut)

Dépendances:
    pip install yfinance pandas numpy ta

Fonctions clés:
- load_prices() : OHLCV via yfinance
- compute_indicators() : ajoute un large set d'indicateurs
- technical_signals() : signaux élémentaires + score composite
- detect_regime() : Bull/Bear/Range + régime de volatilité
- risk_stats() : vol annualisée, VaR(95), max drawdown
- backtest() : moteur vectorisé avec R:R, stops, trailing, slippage, fees
- walk_forward_backtest() : évalue la robustesse par fenêtres temporelles

Les sorties utilisent des dataclasses (sérialisables en dict).

Auteurs: toi + IA (2025)
Licence: MIT (à adapter)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
import ta

# -----------------------------------------------------------------------------#
#                                    Config                                    #
# -----------------------------------------------------------------------------#

YF_SLEEP = 0.1
DEFAULT_PERIOD = "3y"
DEFAULT_INTERVAL = "1d"
TRADING_DAYS = 252

# -----------------------------------------------------------------------------#
#                                  Dataclasses                                 #
# -----------------------------------------------------------------------------#

@dataclass
class IndicatorSet:
    df: pd.DataFrame  # OHLCV + indicateurs
    meta: Dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return self.df

    def to_dict(self) -> Dict[str, Any]:
        m = dict(self.meta)
        m["columns"] = list(self.df.columns)
        m["rows"] = len(self.df)
        return m


@dataclass
class TechnicalSignals:
    score: float
    components: Dict[str, float]
    labels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"score": float(self.score), "components": dict(self.components), "labels": list(self.labels)}


@dataclass
class RegimeInfo:
    trend: str           # "Bull" | "Bear" | "Range"
    vol_regime: str      # "LowVol" | "HighVol"
    slope200: Optional[float] = None
    drawdown_last: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskStats:
    vol_ann_pct: Optional[float]
    var95_pct: Optional[float]
    max_dd_pct: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


@dataclass
class TradeResult:
    entries: int
    win_rate_pct: float
    avg_ret_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_dd_pct: float
    expectancy_pct: float
    exposure_pct: float
    final_equity: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BacktestReport:
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    summary: TradeResult
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "params": dict(self.params),
            "n_trades": int(len(self.trades)),
            "equity_start": float(self.equity_curve.iloc[0]) if len(self.equity_curve) else None,
            "equity_end": float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else None,
        }


@dataclass
class WalkForwardReport:
    folds: List[BacktestReport]
    blended_summary: TradeResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folds": [f.summary.to_dict() for f in self.folds],
            "blended": self.blended_summary.to_dict()
        }

# -----------------------------------------------------------------------------#
#                                Data Loading                                  #
# -----------------------------------------------------------------------------#

def load_prices(ticker: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Télécharge OHLCV et nettoie l'index (timezone-naive)."""
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    # s'assure des colonnes
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

# -----------------------------------------------------------------------------#
#                              Technical Indicators                             #
# -----------------------------------------------------------------------------#

def _safe_series(x) -> pd.Series:
    return pd.Series(x).replace([np.inf, -np.inf], np.nan)

def compute_indicators(px: pd.DataFrame) -> IndicatorSet:
    """
    Ajoute un large set d'indicateurs dans un DataFrame unique.
    Colonnes ajoutées (extrait):
      SMA_20/50/200, EMA_12/26, RSI_14, MACD/Signal/Hist,
      Stoch_%K/%D, ADX_14, ATR_14, BB_Upper/Middle/Lower,
      Keltner_* , Donchian_20_Up/Down, OBV, ROC_63, Volatility_20,
      Trend_200_slope (lignearly regressed), etc.
    """
    df = px.copy()
    close = df["Close"]

    # MAs
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
    df["EMA_12"] = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"] = ta.trend.ema_indicator(close, window=26)

    # Momentum / oscillateurs
    df["RSI_14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=close, window=14, smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    df["ROC_63"] = ta.momentum.roc(close, window=63)

    # Volatilité / bandes
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2.0)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["ATR_14"] = ta.volatility.average_true_range(df["High"], df["Low"], close, window=14)
    df["Volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS) * 100.0

    # Directional / trend
    df["ADX_14"] = ta.trend.adx(df["High"], df["Low"], close, window=14)
    # pente SMA200 (régression linéaire simple sur 50 derniers points de SMA200)
    sma200 = df["SMA_200"].dropna()
    slope200 = pd.Series(index=df.index, dtype=float)
    if len(sma200) >= 50:
        x = np.arange(50)
        for i in range(49, len(sma200)):
            y = sma200.iloc[i-49:i+1].values
            m, _ = np.polyfit(x, y, 1)
            slope200.loc[sma200.index[i]] = m
    df["Trend200_slope"] = slope200.reindex(df.index)

    # Keltner Channel
    ema20 = ta.trend.ema_indicator(close, window=20)
    df["Keltner_Upper"] = ema20 + 2 * df["ATR_14"]
    df["Keltner_Lower"] = ema20 - 2 * df["ATR_14"]

    # Donchian
    def rolling_max(s, w): return s.rolling(w).max()
    def rolling_min(s, w): return s.rolling(w).min()
    df["Donchian20_Up"] = rolling_max(df["High"], 20)
    df["Donchian20_Down"] = rolling_min(df["Low"], 20)

    # Volume-based
    df["OBV"] = ta.volume.on_balance_volume(close, df["Volume"])

    # Nettoyage final
    meta = {"indicator_set": "v1", "rows": len(df)}
    return IndicatorSet(df=df, meta=meta)

# -----------------------------------------------------------------------------#
#                                 Signals & Score                               #
# -----------------------------------------------------------------------------#

def technical_signals(ind: IndicatorSet) -> TechnicalSignals:
    """
    Crée un score composite (-1..+1) basé sur:
      - Mom short/mid (RSI, MACD, EMA12>EMA26, SMA20>SMA50)
      - Trend long (Close>SMA200, slope200>0)
      - Breakout/Range (Donchian, Bollinger position)
      - Volatility filter (ADX/ATR)
    """
    df = ind.df.copy()
    last = df.iloc[-1]

    comp: Dict[str, float] = {}
    labels: List[str] = []

    # Momentum court
    rsi = last.get("RSI_14", np.nan)
    if np.isfinite(rsi):
        if rsi < 30: comp["RSI"] = +0.20; labels.append(f"RSI oversold {rsi:.1f}")
        elif rsi > 70: comp["RSI"] = -0.20; labels.append(f"RSI overbought {rsi:.1f}")
        else: comp["RSI"] = 0.05 * (rsi - 50) / 50.0

    macd, sig = last.get("MACD", np.nan), last.get("MACD_Signal", np.nan)
    if np.isfinite(macd) and np.isfinite(sig):
        comp["MACD"] = +0.15 if macd > sig else -0.10
        labels.append("MACD bull" if macd > sig else "MACD bear")

    ema12, ema26 = last.get("EMA_12", np.nan), last.get("EMA_26", np.nan)
    if np.isfinite(ema12) and np.isfinite(ema26):
        comp["EMA_Cross"] = +0.10 if ema12 > ema26 else -0.10

    sma20, sma50 = last.get("SMA_20", np.nan), last.get("SMA_50", np.nan)
    if np.isfinite(sma20) and np.isfinite(sma50):
        comp["MA_20_50"] = +0.10 if sma20 > sma50 else -0.10

    # Trend long
    sma200, close = last.get("SMA_200", np.nan), last.get("Close", np.nan)
    if np.isfinite(sma200) and np.isfinite(close):
        comp["Above_200"] = +0.15 if close > sma200 else -0.15
    slope200 = last.get("Trend200_slope", np.nan)
    if np.isfinite(slope200):
        comp["Slope200"] = +0.10 if slope200 > 0 else -0.10

    # Breakout / position BB / Donchian
    up, lo = last.get("BB_Upper", np.nan), last.get("BB_Lower", np.nan)
    if np.isfinite(up) and np.isfinite(lo) and np.isfinite(close):
        width = up - lo
        if width > 0:
            z = (close - (lo + up)/2) / (width/2)
            comp["BB_Z"] = np.clip(z*0.07, -0.15, 0.15)

    don_up, don_dn = last.get("Donchian20_Up", np.nan), last.get("Donchian20_Down", np.nan)
    if np.isfinite(don_up) and np.isfinite(don_dn) and np.isfinite(close):
        if close >= don_up: comp["Breakout20"] = +0.15; labels.append("Breakout 20d ↑")
        elif close <= don_dn: comp["Breakout20"] = -0.15; labels.append("Breakdown 20d ↓")
        else: comp["Breakout20"] = 0.0

    # ADX (force de tendance)
    adx = last.get("ADX_14", np.nan)
    if np.isfinite(adx):
        if adx >= 25: comp["ADX"] = +0.05; labels.append("Trend strong (ADX≥25)")
        else: comp["ADX"] = 0.0

    score = float(np.clip(sum(comp.values()), -1.0, +1.0))
    return TechnicalSignals(score=score, components=comp, labels=labels)

# -----------------------------------------------------------------------------#
#                                  Regime Logic                                 #
# -----------------------------------------------------------------------------#

def _drawdown(series: pd.Series) -> pd.Series:
    rollmax = series.cummax()
    dd = series / rollmax - 1.0
    return dd

def detect_regime(ind: IndicatorSet) -> RegimeInfo:
    df = ind.df.copy()
    close = df["Close"].dropna()
    if len(close) < 220:
        return RegimeInfo(trend="Range", vol_regime="LowVol", slope200=0.0, drawdown_last=0.0)

    sma200 = df["SMA_200"].dropna()
    if len(sma200) < 60:
        return RegimeInfo(trend="Range", vol_regime="LowVol", slope200=0.0, drawdown_last=0.0)

    slope = df["Trend200_slope"].dropna()
    sl = slope.iloc[-1] if len(slope) else np.nan
    dd = _drawdown(close)
    dd_last = float(dd.iloc[-1]) if len(dd) else np.nan

    # Trend
    if np.isfinite(sl):
        # Seuils prudents: une très faible pente ~0 est traitée comme Range
        thr = 0.01
        if sl > thr and dd_last > -0.15:
            trend = "Bull"
        elif sl < -thr and dd.min() < -0.20:
            trend = "Bear"
        else:
            trend = "Range"
    else:
        trend = "Range"

    # Vol regime via ATR/Close (normalisé), tolère absence d'ATR
    vol_regime = "LowVol"
    if "ATR_14" in df.columns:
        atr = df["ATR_14"].dropna()
        if len(atr) and len(close):
            atr_pct = float(atr.iloc[-1] / close.iloc[-1] * 100.0)
            if atr_pct >= 3.0:
                vol_regime = "HighVol"

    return RegimeInfo(trend=trend, vol_regime=vol_regime, slope200=float(sl) if np.isfinite(sl) else 0.0,
                      drawdown_last=float(dd_last) if np.isfinite(dd_last) else 0.0)

# -----------------------------------------------------------------------------#
#                                   Risk Stats                                  #
# -----------------------------------------------------------------------------#

def risk_stats(px: pd.DataFrame) -> RiskStats:
    close = px["Close"].dropna()
    if len(close) < 30:
        return RiskStats(None, None, None)

    ret = close.pct_change().dropna()
    vol_ann = float(ret.std() * np.sqrt(TRADING_DAYS) * 100.0)
    var95 = float(np.percentile(ret, 5) * 100.0)
    dd = _drawdown(close)
    max_dd = float(dd.min() * 100.0)
    return RiskStats(vol_ann_pct=vol_ann, var95_pct=var95, max_dd_pct=max_dd)

# -----------------------------------------------------------------------------#
#                                   Backtester                                  #
# -----------------------------------------------------------------------------#

def _fees_slippage(returns: pd.Series, fee_bps: float = 0.0, slippage_bps: float = 0.0,
                   signals: Optional[pd.Series] = None) -> pd.Series:
    """
    Applique coûts proportionnels lors des changements de position (rotation).
    fee_bps/slippage_bps en points de base par 'trade'.
    """
    if signals is None or signals.empty:
        return returns
    turns = signals.diff().abs().fillna(0.0)
    cost = (fee_bps + slippage_bps) / 10000.0
    # coût appliqué le jour du changement (approx)
    ret_net = returns - turns * cost
    return ret_net

def _position_sizer(vol_target_ann: Optional[float], daily_vol: pd.Series,
                    kelly_frac: Optional[float]) -> pd.Series:
    """
    Calcule un levier (0..x) via ciblage de volatilité (annualisée) et/ou Kelly fraction.
    vol_target_ann en %, daily_vol = rolling std daily (en %).
    """
    lev = pd.Series(1.0, index=daily_vol.index)
    if vol_target_ann and vol_target_ann > 0:
        vol_daily_target = vol_target_ann / np.sqrt(TRADING_DAYS)
        with np.errstate(divide='ignore', invalid='ignore'):
            lev = (vol_daily_target / daily_vol.replace(0, np.nan)).clip(upper=5.0).fillna(0.0)
    if kelly_frac is not None:
        # On applique en multiplicatif (borne 0..1.5 pour prudence)
        lev = (lev * float(kelly_frac)).clip(0.0, 1.5)
    return lev

def _entry_exit_from_rules(ind: IndicatorSet, rules: Dict[str, Any]) -> pd.Series:
    """
    Construit un signal d'exposition (0/1 ou -1/0/1) à partir de règles simples.
    Exemples de rules:
      {
        "long_when": ["EMA12>EMA26", "Close>SMA200", "MACD>Signal"],
        "flat_when": ["RSI>80"],
        "short_when": ["EMA12<EMA26", "Close<SMA200"],  # optionnel si on veut short
        "confirm_with": ["ADX>=20"],
      }
    """
    df = ind.df
    sig = pd.Series(0.0, index=df.index)

    def cond(name: str) -> pd.Series:
        name = name.strip().lower()
        if name == "ema12>ema26":
            return (df["EMA_12"] > df["EMA_26"])
        if name == "ema12<ema26":
            return (df["EMA_12"] < df["EMA_26"])
        if name == "close>sma200":
            return (df["Close"] > df["SMA_200"])
        if name == "close<sma200":
            return (df["Close"] < df["SMA_200"])
        if name == "sma20>sma50":
            return (df["SMA_20"] > df["SMA_50"])
        if name == "sma20<sma50":
            return (df["SMA_20"] < df["SMA_50"])
        if name == "macd>signal":
            return (df["MACD"] > df["MACD_Signal"])
        if name == "macd<signal":
            return (df["MACD"] < df["MACD_Signal"])
        if name == "rsi<30":
            return (df["RSI_14"] < 30)
        if name == "rsi>70":
            return (df["RSI_14"] > 70)
        if name == "adx>=20":
            return (df["ADX_14"] >= 20)
        if name == "breakout20":
            return (df["Close"] >= df["Donchian20_Up"])
        if name == "breakdown20":
            return (df["Close"] <= df["Donchian20_Down"])
        # fallback: col1>col2 generic
        if ">" in name:
            c1, c2 = [c.strip().upper() for c in name.split(">")]
            return (df[c1] > df[c2])
        if "<" in name:
            c1, c2 = [c.strip().upper() for c in name.split("<")]
            return (df[c1] < df[c2])
        raise ValueError(f"Règle inconnue: {name}")

    long_when = rules.get("long_when", [])
    short_when = rules.get("short_when", [])
    flat_when = rules.get("flat_when", [])
    confirms = rules.get("confirm_with", [])

    if long_when:
        cond_long = pd.concat([cond(c) for c in long_when], axis=1).all(axis=1)
    else:
        cond_long = pd.Series(False, index=df.index)
    if short_when:
        cond_short = pd.concat([cond(c) for c in short_when], axis=1).all(axis=1)
    else:
        cond_short = pd.Series(False, index=df.index)
    if flat_when:
        cond_flat = pd.concat([cond(c) for c in flat_when], axis=1).any(axis=1)
    else:
        cond_flat = pd.Series(False, index=df.index)

    # confirmations
    if confirms:
        conf_all = pd.concat([cond(c) for c in confirms], axis=1).all(axis=1)
        cond_long &= conf_all
        cond_short &= conf_all

    sig[cond_long] = 1.0
    sig[cond_short] = -1.0
    sig[cond_flat] = 0.0
    sig = sig.ffill().fillna(0.0)
    return sig

def _apply_stops_and_trailing(df: pd.DataFrame, signal: pd.Series,
                              sl_pct: Optional[float], tp_pct: Optional[float],
                              atr_mult_trail: Optional[float]) -> pd.Series:
    """
    Transforme un signal d'expo en 'position' en appliquant sorties anticipées via SL/TP/Trailing.
    Approche vectorisée approximée sur données daily (pas intraday-exact).
    """
    pos = signal.copy().astype(float)
    if sl_pct is None and tp_pct is None and atr_mult_trail is None:
        return pos

    close = df["Close"]
    atr = df["ATR_14"]
    entry_price = None
    trail_stop = None

    for i in range(1, len(pos)):
        if pos.iloc[i] != pos.iloc[i-1]:  # changement d'état → (ré)initialisation
            entry_price = close.iloc[i]
            trail_stop = None

        if pos.iloc[i] == 0.0:
            continue

        # SL/TP
        if entry_price is not None:
            r = (close.iloc[i] / entry_price - 1.0) * (1 if pos.iloc[i] > 0 else -1)
            hit_sl = (sl_pct is not None) and (r <= -abs(sl_pct)/100.0)
            hit_tp = (tp_pct is not None) and (r >= abs(tp_pct)/100.0)

            # Trailing ATR
            if atr_mult_trail and np.isfinite(atr.iloc[i]) and np.isfinite(close.iloc[i]):
                if pos.iloc[i] > 0:
                    ts = close.iloc[i] - atr_mult_trail * atr.iloc[i]
                    trail_stop = max(trail_stop or ts, ts)
                    if close.iloc[i] <= trail_stop:
                        hit_sl = True
                else:
                    ts = close.iloc[i] + atr_mult_trail * atr.iloc[i]
                    trail_stop = min(trail_stop or ts, ts)
                    if close.iloc[i] >= trail_stop:
                        hit_sl = True

            if hit_sl or hit_tp:
                pos.iloc[i] = 0.0  # flat après coup
                entry_price = None
                trail_stop = None

    return pos

def backtest(ind: IndicatorSet,
             rules: Dict[str, Any],
             initial_equity: float = 100_000.0,
             fee_bps: float = 1.0,
             slippage_bps: float = 1.0,
             sl_pct: Optional[float] = None,
             tp_pct: Optional[float] = None,
             atr_mult_trail: Optional[float] = None,
             vol_target_ann: Optional[float] = None,   # ex: 15 (%)
             kelly_frac: Optional[float] = None) -> BacktestReport:
    """
    Backtest vectorisé daily:
      - 'rules' -> signal brut (-1/0/1)
      - stops/trailing -> position
      - vol targeting / kelly -> levier
      - coûts appliqués à chaque rotation
    """
    df = ind.df
    close = df["Close"]
    ret = close.pct_change().fillna(0.0)

    signal = _entry_exit_from_rules(ind, rules)
    position = _apply_stops_and_trailing(df, signal, sl_pct, tp_pct, atr_mult_trail)

    # volatilité rolling pour sizing
    daily_vol = ret.rolling(20).std().fillna(method="bfill").replace(0, np.nan)
    lev = _position_sizer(vol_target_ann, daily_vol * 100.0, kelly_frac) if (vol_target_ann or kelly_frac) else pd.Series(1.0, index=ret.index)

    gross = position.shift(1).fillna(0.0) * lev.shift(1).fillna(1.0) * ret  # entrée au close précédent
    net = _fees_slippage(gross, fee_bps=fee_bps, slippage_bps=slippage_bps, signals=position)

    # equity curve
    equity = (1.0 + net).cumprod() * initial_equity
    daily_rets = net

    # trades simples (chgt de signe)
    turns = position.diff().fillna(0.0)
    trade_starts = turns[turns != 0].index
    trades = []
    last_state = 0.0
    entry_idx = None
    entry_price = None
    for i, dt in enumerate(df.index):
        st = position.loc[dt]
        if last_state == 0.0 and st != 0.0:
            entry_idx = dt
            entry_price = close.loc[dt]
        if last_state != 0.0 and st == 0.0 and entry_idx is not None:
            exit_idx = dt
            exit_price = close.loc[dt]
            side = "LONG" if last_state > 0 else "SHORT"
            ret_pct = (exit_price / entry_price - 1.0) * (1 if side == "LONG" else -1)
            trades.append([entry_idx, exit_idx, side, float(ret_pct*100.0)])
            entry_idx = None
            entry_price = None
        last_state = st
    trades_df = pd.DataFrame(trades, columns=["entry", "exit", "side", "ret_pct"])

    # stats
    def _cagr(series: pd.Series) -> float:
        if len(series) < TRADING_DAYS:
            return (series.iloc[-1] / series.iloc[0]) - 1.0
        years = len(series) / TRADING_DAYS
        return (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1.0

    def _sharpe(rets: pd.Series) -> float:
        if rets.std() == 0 or rets.std() != rets.std():
            return 0.0
        return float(np.sqrt(TRADING_DAYS) * rets.mean() / rets.std())

    def _sortino(rets: pd.Series) -> float:
        neg = rets[rets < 0]
        sd = neg.std()
        if sd == 0 or not np.isfinite(sd):
            return 0.0
        return float(np.sqrt(TRADING_DAYS) * rets.mean() / sd)

    wins = (trades_df["ret_pct"] > 0).sum() if not trades_df.empty else 0
    entries = len(trades_df)
    win_rate = (wins / entries * 100.0) if entries > 0 else 0.0
    avg_ret = trades_df["ret_pct"].mean() if entries > 0 else 0.0
    expectancy = (trades_df["ret_pct"].mean()) if entries > 0 else 0.0

    dd = _drawdown(equity)
    max_dd_pct = float(dd.min() * 100.0) if len(dd) else 0.0

    exposure = position.abs().mean() * 100.0
    summary = TradeResult(
        entries=int(entries),
        win_rate_pct=float(win_rate),
        avg_ret_pct=float(avg_ret),
        cagr_pct=float(_cagr(equity) * 100.0),
        sharpe=float(_sharpe(daily_rets)),
        sortino=float(_sortino(daily_rets)),
        max_dd_pct=float(max_dd_pct),
        expectancy_pct=float(expectancy),
        exposure_pct=float(exposure),
        final_equity=float(equity.iloc[-1])
    )

    params = dict(rules=rules, fee_bps=fee_bps, slippage_bps=slippage_bps,
                  sl_pct=sl_pct, tp_pct=tp_pct, atr_mult_trail=atr_mult_trail,
                  vol_target_ann=vol_target_ann, kelly_frac=kelly_frac)

    return BacktestReport(
        equity_curve=equity,
        daily_returns=daily_rets,
        trades=trades_df,
        summary=summary,
        params=params
    )

# -----------------------------------------------------------------------------#
#                             Walk-Forward Backtest                             #
# -----------------------------------------------------------------------------#

def walk_forward_backtest(ind: IndicatorSet,
                          rules: Dict[str, Any],
                          n_folds: int = 3,
                          min_points_per_fold: int = 252,
                          **bt_kwargs) -> WalkForwardReport:
    """
    Split temporel en n_folds (consécutifs), backtest sur chaque fold.
    """
    df = ind.df
    N = len(df)
    if N < n_folds * min_points_per_fold:
        n_folds = max(1, N // min_points_per_fold)

    folds = []
    for k in range(n_folds):
        start = int(k * N / n_folds)
        end = int((k+1) * N / n_folds)
        seg = IndicatorSet(df=df.iloc[start:end].copy(), meta={"fold": k+1})
        if len(seg.df) < min_points_per_fold:
            continue
        rep = backtest(seg, rules, **bt_kwargs)
        folds.append(rep)

    # Blend summary (pondéré par longueur)
    if folds:
        weights = [len(r.daily_returns) for r in folds]
        total_w = sum(weights)
        def wavg(attr):
            vals = [getattr(r.summary, attr) for r in folds]
            return float(np.average(vals, weights=weights))
        blended = TradeResult(
            entries=int(np.average([f.summary.entries for f in folds], weights=weights)),
            win_rate_pct=wavg("win_rate_pct"),
            avg_ret_pct=wavg("avg_ret_pct"),
            cagr_pct=wavg("cagr_pct"),
            sharpe=wavg("sharpe"),
            sortino=wavg("sortino"),
            max_dd_pct=wavg("max_dd_pct"),
            expectancy_pct=wavg("expectancy_pct"),
            exposure_pct=wavg("exposure_pct"),
            final_equity=float(np.average([f.summary.final_equity for f in folds], weights=weights))
        )
    else:
        blended = TradeResult(0,0,0,0,0,0,0,0,0,0)

    return WalkForwardReport(folds=folds, blended_summary=blended)

# -----------------------------------------------------------------------------#
#                              API "haut niveau"                                #
# -----------------------------------------------------------------------------#

def compute_technical_features(ticker: str, window: int = 180) -> Dict[str, Any]:
    """
    Compute technical features for a given ticker using the build_technical_view function.
    This is a wrapper function to match the expected signature from app.py.

    Args:
        ticker (str): Stock ticker symbol
        window (int): Lookback window in days (default: 180)

    Returns:
        Dict: Technical features suitable for conversion to dict
    """
    try:
        # Map window to appropriate period and interval
        if window <= 30:
            period = "1mo"
            interval = "1d"
        elif window <= 90:
            period = "3mo"
            interval = "1d"
        elif window <= 180:
            period = "6mo"
            interval = "1d"
        elif window <= 365:
            period = "1y"
            interval = "1d"
        else:
            period = f"{max(1, window // 365)}y"
            interval = "1d"

        result = build_technical_view(ticker, period=period, interval=interval)

        # Convert to dict if it's not already
        if hasattr(result, 'to_dict'):
            out = result.to_dict()
            out.setdefault('window', window)
            return out
        elif isinstance(result, dict):
            result.setdefault('window', window)
            return result
        else:
            # Fallback conversion
            return {
                'ticker': ticker,
                'last_price': getattr(result, 'last_price', None),
                'signals': getattr(result, 'signals', {}),
                'regime': getattr(result, 'regime', {}),
                'risk': getattr(result, 'risk', {}),
                'window': window
            }

    except Exception as e:
        # Return error dict
        return {
            'ticker': ticker,
            'error': f"Failed to compute technical features: {str(e)}",
            'signals': {},
            'regime': {},
            'risk': {},
            'window': window
        }


def build_technical_view(ticker: str,
                         period: str = DEFAULT_PERIOD,
                         interval: str = DEFAULT_INTERVAL) -> Dict[str, Any]:
    """
    Pipeline rapide (sans backtest):
      - charge prix
      - compute_indicators
      - technical_signals + detect_regime + risk_stats
    Retourne un dict compact prêt à afficher/logguer.
    """
    px = load_prices(ticker, period=period, interval=interval)
    if px.empty:
        return {"ticker": ticker, "error": "No price data"}

    ind = compute_indicators(px)
    sig = technical_signals(ind)
    regime = detect_regime(ind)
    risk = risk_stats(px)

    return {
        "ticker": ticker,
        "last_price": float(px["Close"].iloc[-1]),
        "signals": sig.to_dict(),
        "regime": regime.to_dict(),
        "risk": risk.to_dict()
    }

# -----------------------------------------------------------------------------#
#                                Exemple local                                  #
# -----------------------------------------------------------------------------#

if __name__ == "__main__":
    import json

    TICKER = "AAPL"
    px = load_prices(TICKER, period="5y")
    if px.empty:
        print("Prix introuvables.")
        raise SystemExit(0)

    ind = compute_indicators(px)
    sig = technical_signals(ind)
    regime = detect_regime(ind)
    risk = risk_stats(px)

    print("=== SNAPSHOT TECH ===")
    print(json.dumps({
        "ticker": TICKER,
        "signals": sig.to_dict(),
        "regime": regime.to_dict(),
        "risk": risk.to_dict()
    }, indent=2))

    # Backtest exemple — tendance/momentum classique
    rules = {
        "long_when": ["EMA12>EMA26", "Close>SMA200", "MACD>Signal"],
        "flat_when": ["RSI>80"],
        "confirm_with": ["ADX>=20"]
    }
    rep = backtest(
        ind, rules,
        initial_equity=100_000,
        fee_bps=1.0, slippage_bps=1.0,
        sl_pct=10.0, tp_pct=None, atr_mult_trail=2.0,
        vol_target_ann=15.0, kelly_frac=0.8
    )

    print("\n=== BACKTEST SUMMARY ===")
    print(json.dumps(rep.summary.to_dict(), indent=2))

    # Walk-forward 3 folds
    wfr = walk_forward_backtest(
        ind, rules, n_folds=3, min_points_per_fold=300,
        initial_equity=100_000, fee_bps=1.0, slippage_bps=1.0,
        sl_pct=10.0, atr_mult_trail=2.0, vol_target_ann=15.0, kelly_frac=0.8
    )
    print("\n=== WALK-FORWARD BLENDED ===")
    print(json.dumps(wfr.blended_summary.to_dict(), indent=2))
