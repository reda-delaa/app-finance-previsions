# phase1_fundamental.py
# -*- coding: utf-8 -*-
"""
Phase 1 — Fondamental “pro” (Fair Value dynamique) pour actions US/CA.
Mode DEBUG détaillé activable: --log DEBUG ou PHASE1_DEBUG=1

Dépendances:
    pip install yfinance pandas numpy python-dateutil
"""
from __future__ import annotations

import os
import math
import time
import json
import platform
import logging
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from datetime import datetime

# ----------------------------- Logging & Debug --------------------------------

LOGGER_NAME = "phase1_fundamental"
logger = logging.getLogger(LOGGER_NAME)

def _init_logging_from_env(default_level: str = "INFO"):
    """Appelé depuis __main__ ou quand importé sans CLI."""
    lvl = os.getenv("PHASE1_DEBUG")
    level = logging.DEBUG if (lvl and lvl.strip() not in ("0", "", "false", "False")) else getattr(logging, default_level)
    if not logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    logger.setLevel(level)

def _brief(x: Any, max_chars: int = 160) -> str:
    """Résumé compact et sûr pour logs."""
    try:
        if x is None:
            return "None"
        if isinstance(x, (int, float, str, bool)):
            s = str(x)
            return s if len(s) <= max_chars else s[:max_chars] + "…"
        if isinstance(x, dict):
            keys = list(x.keys())
            return f"dict(keys={len(keys)}: {keys[:6]}{'…' if len(keys)>6 else ''})"
        if isinstance(x, pd.DataFrame):
            cols = list(x.columns)[:6]
            return f"DataFrame{tuple(x.shape)} cols={cols}{'…' if x.shape[1]>6 else ''}"
        if isinstance(x, pd.Series):
            return f"Series(len={len(x)}) name={x.name}"
        if isinstance(x, (list, tuple, set)):
            elems = list(x)[:6]
            return f"{type(x).__name__}(len={len(x)}): {elems}{'…' if len(x)>6 else ''}"
        return f"{type(x).__name__}"
    except Exception:
        return f"{type(x).__name__}"

def _brief_dict(d: Dict[str, Any]) -> str:
    try:
        return "{ " + ", ".join([f"{k}: {_brief(v)}" for k, v in list(d.items())[:8]]) + (" … }" if len(d)>8 else " }")
    except Exception:
        return "dict(?)"

def debug_io(name: Optional[str] = None):
    """
    Decorator: trace les entrées (types/shapes), la durée, et la sortie.
    N’affiche le détail que si le logger est en DEBUG.
    """
    def deco(func):
        fname = name or func.__name__
        def wrapper(*args, **kwargs):
            if logger.isEnabledFor(logging.DEBUG):
                args_brief = ", ".join(_brief(a) for a in args)
                kwargs_brief = _brief_dict(kwargs) if kwargs else ""
                logger.debug(f"→ {fname}({args_brief}{', ' if kwargs else ''}{kwargs_brief})")
            t0 = time.time()
            try:
                out = func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"× {fname} a échoué: {e}")
                raise
            dt = time.time() - t0
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"← {fname} OK ({dt:.3f}s) → {_brief(out)}")
            return out
        return wrapper
    return deco

# Init logging si importé sans passer par __main__
_init_logging_from_env()

# ----------------------------- Config & Constantes -----------------------------

DEFAULT_BENCHMARK = "^GSPC"
DEFAULT_RF_TICKER = "^TNX"
DEFAULT_FX_USDCAD = "CAD=X"

DEFAULT_MRP = 0.05
DEFAULT_TAX_RATE = 0.25
DEFAULT_WACC_MIN = 0.06
DEFAULT_WACC_MAX = 0.12
DEFAULT_PERPET_G_MIN = 0.0
DEFAULT_PERPET_G_MAX = 0.03
MAX_PEERS = 15
YF_SLEEP = 0.15

# ----------------------------- Dataclasses sorties -----------------------------

@dataclass
class HealthRatios:
    roe_pct: Optional[float] = None
    roic_pct: Optional[float] = None
    fcf_yield_pct: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    current_ratio: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)

@dataclass
class PeerMultiples:
    pe_median: Optional[float] = None
    ev_ebitda_median: Optional[float] = None
    ps_median: Optional[float] = None
    sample_size: int = 0
    tickers_used: List[str] = None
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ComparableZScores:
    pe_z: Optional[float] = None
    ev_ebitda_z: Optional[float] = None
    ps_z: Optional[float] = None
    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)

@dataclass
class DCFResult:
    fv_central: Optional[float] = None
    fv_low: Optional[float] = None
    fv_high: Optional[float] = None
    wacc_used: Optional[float] = None
    g_used: Optional[float] = None
    confidence: float = 0.0
    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)

@dataclass
class FairValueAggregate:
    fv_comparable: Optional[float] = None
    fv_dcf: Optional[float] = None
    fv_composite: Optional[float] = None
    upside_pct: Optional[float] = None
    confidence: float = 0.0
    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)

@dataclass
class FundamentalView:
    ticker: str
    currency: str
    price: Optional[float]
    health: HealthRatios
    peers: PeerMultiples
    zscores: ComparableZScores
    dcf: DCFResult
    fair_value: FairValueAggregate
    notes: List[str]
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ----------------------------- Utils sécurisés --------------------------------

def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        val = float(x)
        if math.isfinite(val):
            return val
        return default
    except Exception:
        return default

def _get_series_last(s: pd.Series) -> Optional[float]:
    try:
        v = s.dropna().iloc[-1]
        return float(v)
    except Exception:
        return None

@debug_io()
def _yf_info(t: yf.Ticker) -> Dict[str, Any]:
    """Robuste: tente get_info(), sinon .info, puis complète avec fast_info."""
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

@debug_io()
def _currency_of(ticker: str) -> Optional[str]:
    try:
        t = yf.Ticker(ticker)
        info = _yf_info(t)
        return info.get("currency")
    except Exception:
        return None

@debug_io()
def _load_fx_usdcad() -> Optional[float]:
    try:
        fx = yf.Ticker(DEFAULT_FX_USDCAD).history(period="5d", interval="1d", auto_adjust=True)
        if fx.empty:
            return None
        return float(fx["Close"].dropna().iloc[-1])
    except Exception:
        return None

def convert_to_currency(value: Optional[float], from_ccy: Optional[str], to_ccy: Optional[str]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    if not from_ccy or not to_ccy or from_ccy == to_ccy:
        return float(value)
    if from_ccy == "USD" and to_ccy == "CAD":
        fx = _load_fx_usdcad()
        return float(value) * fx if fx else value
    if from_ccy == "CAD" and to_ccy == "USD":
        fx = _load_fx_usdcad()
        return float(value) / fx if fx else value
    return float(value)

# ----------------------------- Chargement des données --------------------------

@debug_io()
def load_prices(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV historiques."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            logger.warning(f"history vide pour {ticker} ({period},{interval})")
            return pd.DataFrame()
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        logger.exception(f"load_prices({ticker}) a échoué: {e}")
        return pd.DataFrame()

@debug_io()
def load_fundamentals(ticker: str) -> Dict[str, pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        out = {
            "income_stmt": t.income_stmt if isinstance(t.income_stmt, pd.DataFrame) else pd.DataFrame(),
            "balance_sheet": t.balance_sheet if isinstance(t.balance_sheet, pd.DataFrame) else pd.DataFrame(),
            "cash_flow": t.cashflow if isinstance(t.cashflow, pd.DataFrame) else pd.DataFrame(),
        }
        # INFO-level résumé qualité
        logger.info(f"{ticker} fundamentals: "
                    f"income={out['income_stmt'].shape} | "
                    f"balance={out['balance_sheet'].shape} | "
                    f"cashflow={out['cash_flow'].shape}")
        return out
    except Exception as e:
        logger.exception(f"load_fundamentals({ticker}) a échoué: {e}")
        return {"income_stmt": pd.DataFrame(), "balance_sheet": pd.DataFrame(), "cash_flow": pd.DataFrame()}

@debug_io()
def load_info(ticker: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(ticker)
        info = _yf_info(t)
        logger.info(f"{ticker} info: {len(info)} clés")
        return info
    except Exception:
        return {}

@debug_io()
def load_rf_rate() -> Optional[float]:
    try:
        df = yf.Ticker(DEFAULT_RF_TICKER).history(period="10d")
        if df.empty:
            logger.warning("^TNX vide → rf=None")
            return None
        last = float(df["Close"].dropna().iloc[-1])
        return last / 100.0 if last > 1.0 else last
    except Exception:
        return None

@debug_io()
def estimate_beta(ticker: str, benchmark: str = DEFAULT_BENCHMARK, period: str = "3y") -> Optional[float]:
    try:
        px_i = load_prices(ticker, period=period)
        px_m = load_prices(benchmark, period=period)
        if px_i.empty or px_m.empty:
            return None
        df = pd.concat([px_i["Close"].pct_change(), px_m["Close"].pct_change()], axis=1).dropna()
        df.columns = ["ri", "rm"]
        cov = df["ri"].cov(df["rm"])
        var = df["rm"].var()
        return float(cov / var) if var and var > 0 else None
    except Exception:
        return None

# ----------------------------- Localisation lignes -----------------------------

def pick_first_index(df: pd.DataFrame, *candidates) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    idx_set = set(map(str, df.index))
    for c in candidates:
        if c in idx_set:
            return c
    low = {str(i).lower(): str(i) for i in df.index}
    for c in candidates:
        key = str(c).lower()
        if key in low:
            return low[key]
    return None

# ----------------------------- Ratios “santé” ---------------------------------

@debug_io()
def compute_health_ratios(ticker: str,
                          fundamentals: Dict[str, pd.DataFrame],
                          info: Dict[str, Any]) -> HealthRatios:
    income = fundamentals.get("income_stmt", pd.DataFrame())
    balance = fundamentals.get("balance_sheet", pd.DataFrame())
    cashflw = fundamentals.get("cash_flow", pd.DataFrame())

    # Localisation robuste
    net_income_row = pick_first_index(income, "Net Income", "NetIncome", "Net Income Common Stockholders")
    revenue_row    = pick_first_index(income, "Total Revenue", "TotalRevenue", "Revenue")
    ebit_row       = pick_first_index(income, "EBIT", "Ebit")
    ebitda_row     = pick_first_index(income, "EBITDA", "Ebitda")
    gross_profit_row = pick_first_index(income, "Gross Profit", "GrossProfit")
    op_income_row  = pick_first_index(income, "Operating Income", "OperatingIncome")

    equity_row          = pick_first_index(balance, "Total Stockholder Equity", "Total Stockholders Equity", "Stockholders Equity")
    total_assets_row    = pick_first_index(balance, "Total Assets")
    current_assets_row  = pick_first_index(balance, "Current Assets")
    current_liab_row    = pick_first_index(balance, "Current Liabilities")
    total_liab_row      = pick_first_index(balance, "Total Liabilities Net Minority Interest", "Total Liabilities")

    cfo_row   = pick_first_index(cashflw, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex_row = pick_first_index(cashflw, "Capital Expenditure", "Capital Expenditures")

    def last_row(df: pd.DataFrame, row_name: Optional[str]) -> Optional[float]:
        try:
            if row_name and row_name in df.index:
                return _get_series_last(df.loc[row_name])
            return None
        except Exception:
            return None

    net_income   = last_row(income, net_income_row)
    revenue      = last_row(income, revenue_row)
    ebit         = last_row(income, ebit_row)
    ebitda       = last_row(income, ebitda_row)
    gross_profit = last_row(income, gross_profit_row)
    op_income    = last_row(income, op_income_row)

    equity         = last_row(balance, equity_row)
    total_assets   = last_row(balance, total_assets_row)
    current_assets = last_row(balance, current_assets_row)
    current_liab   = last_row(balance, current_liab_row)
    total_liab     = last_row(balance, total_liab_row)

    cfo   = last_row(cashflw, cfo_row)
    capex = last_row(cashflw, capex_row)
    fcf   = (cfo + capex) if (cfo is not None and capex is not None) else None

    mcap = _safe_float(info.get("marketCap"), default=np.nan)

    roe = (net_income / equity * 100.0) if (net_income and equity and equity != 0) else None
    roic = None
    try:
        if ebit is not None and total_assets is not None:
            invested_capital = total_assets - (current_liab if current_liab else 0.0)
            if invested_capital:
                nopat = ebit * (1 - DEFAULT_TAX_RATE)
                roic = (nopat / invested_capital) * 100.0
    except Exception:
        roic = None

    fcf_yield = (fcf / mcap * 100.0) if (fcf and mcap and mcap > 0) else None

    net_debt_to_ebitda = None
    try:
        cash_row = pick_first_index(balance, "Cash And Cash Equivalents",
                                    "Cash And Cash Equivalents, And Short Term Investments", "Cash")
        cash = last_row(balance, cash_row) if cash_row else None
        net_debt = (total_liab - (cash if cash else 0.0)) if total_liab else None
        if net_debt is not None and ebitda and ebitda != 0:
            net_debt_to_ebitda = float(net_debt / ebitda)
    except Exception:
        net_debt_to_ebitda = None

    current_ratio = (current_assets / current_liab) if (current_assets and current_liab and current_liab != 0) else None
    gross_margin = (gross_profit / revenue * 100.0) if (gross_profit and revenue) else None
    op_margin    = (op_income / revenue * 100.0) if (op_income and revenue) else None
    net_margin   = (net_income / revenue * 100.0) if (net_income and revenue) else None

    # trace valeurs clés au niveau INFO pour QA rapide
    logger.info(
        f"[{ticker}] health inputs: revenue={revenue}, net_income={net_income}, "
        f"ebitda={ebitda}, equity={equity}, CFO={cfo}, Capex={capex}, shares={info.get('sharesOutstanding')}"
    )

    return HealthRatios(
        roe_pct=_safe_float(roe, None),
        roic_pct=_safe_float(roic, None),
        fcf_yield_pct=_safe_float(fcf_yield, None),
        net_debt_to_ebitda=_safe_float(net_debt_to_ebitda, None),
        current_ratio=_safe_float(current_ratio, None),
        gross_margin_pct=_safe_float(gross_margin, None),
        operating_margin_pct=_safe_float(op_margin, None),
        net_margin_pct=_safe_float(net_margin, None),
    )

# ----------------------------- Pairs / comparables -----------------------------

def infer_peers_from_info(ticker: str, info: Dict[str, Any], limit: int = MAX_PEERS) -> List[str]:
    _ = (ticker, info, limit)
    return []

@debug_io()
def build_peer_set(ticker: str, info: Dict[str, Any], fallback_peers: Optional[List[str]] = None) -> List[str]:
    peers = infer_peers_from_info(ticker, info)
    if not peers and fallback_peers:
        peers = fallback_peers.copy()
    peers = [p for p in peers if p and p.upper() != ticker.upper()]
    peers = list(dict.fromkeys(peers))[:MAX_PEERS]
    logger.info(f"[{ticker}] peers: {peers}")
    return peers

@debug_io()
def fetch_peer_multiples(peers: List[str]) -> pd.DataFrame:
    rows = []
    for p in peers:
        try:
            t = yf.Ticker(p)
            info = _yf_info(t)
            rows.append({
                "ticker": p,
                "trailingPE": _safe_float(info.get("trailingPE"), np.nan),
                "forwardPE": _safe_float(info.get("forwardPE"), np.nan),
                "ps_ttm": _safe_float(info.get("priceToSalesTrailing12Months"), np.nan),
                "ev_ebitda": _safe_float(info.get("enterpriseToEbitda"), np.nan),
            })
            time.sleep(YF_SLEEP)
        except Exception:
            logger.warning(f"peer {p}: info indisponible")
            continue
    df = pd.DataFrame(rows)
    logger.info(f"peers multiples: shape={df.shape}")
    return df

@debug_io()
def summarize_peer_multiples(df: pd.DataFrame) -> PeerMultiples:
    if df is None or df.empty:
        return PeerMultiples(pe_median=None, ev_ebitda_median=None, ps_median=None, sample_size=0, tickers_used=[])
    pe_series = df[["trailingPE", "forwardPE"]].apply(
        lambda r: r["trailingPE"] if np.isfinite(r["trailingPE"]) else r["forwardPE"], axis=1
    )
    pe_median = np.nanmedian(pe_series) if np.isfinite(pe_series).any() else np.nan
    ev_ebitda_median = np.nanmedian(df["ev_ebitda"]) if "ev_ebitda" in df and np.isfinite(df["ev_ebitda"]).any() else np.nan
    ps_median = np.nanmedian(df["ps_ttm"]) if "ps_ttm" in df and np.isfinite(df["ps_ttm"]).any() else np.nan
    used = df["ticker"].dropna().tolist()
    out = PeerMultiples(
        pe_median=float(pe_median) if math.isfinite(pe_median) else None,
        ev_ebitda_median=float(ev_ebitda_median) if math.isfinite(ev_ebitda_median) else None,
        ps_median=float(ps_median) if math.isfinite(ps_median) else None,
        sample_size=len(used),
        tickers_used=used
    )
    logger.info(f"peers summary: {out.to_dict()}")
    return out

@debug_io()
def compute_company_multiples(info: Dict[str, Any], fundamentals: Dict[str, pd.DataFrame]) -> Dict[str, Optional[float]]:
    price = _safe_float(info.get("regularMarketPrice"), np.nan)
    eps_ttm = _safe_float(info.get("trailingEps"), np.nan)
    fwd_eps = _safe_float(info.get("forwardEps"), np.nan)
    pe_trailing = (price / eps_ttm) if (price and eps_ttm and eps_ttm != 0) else np.nan
    pe_forward = (price / fwd_eps) if (price and fwd_eps and fwd_eps != 0) else np.nan
    ps_ttm = _safe_float(info.get("priceToSalesTrailing12Months"), np.nan)
    ev_ebitda = _safe_float(info.get("enterpriseToEbitda"), np.nan)
    out = {
        "pe_trailing": float(pe_trailing) if math.isfinite(pe_trailing) else None,
        "pe_forward": float(pe_forward) if math.isfinite(pe_forward) else None,
        "ps_ttm": float(ps_ttm) if math.isfinite(ps_ttm) else None,
        "ev_ebitda": float(ev_ebitda) if math.isfinite(ev_ebitda) else None,
    }
    logger.info(f"company multiples: {out}")
    return out

@debug_io()
def compute_zscores_company_vs_peers(company: Dict[str, Optional[float]], peers_df: pd.DataFrame) -> ComparableZScores:
    def robust_z(x, series):
        s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty or x is None or not math.isfinite(x):
            return None
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        med = s.median()
        if iqr and iqr > 0:
            return float((x - med) / iqr)
        std = s.std()
        if std and std > 0:
            return float((x - s.mean()) / std)
        return None

    pe_series = peers_df[["trailingPE", "forwardPE"]].apply(
        lambda r: r["trailingPE"] if np.isfinite(r["trailingPE"]) else r["forwardPE"], axis=1
    ) if not peers_df.empty else pd.Series(dtype=float)

    out = ComparableZScores(
        pe_z=robust_z(company.get("pe_trailing") or company.get("pe_forward"), pe_series),
        ev_ebitda_z=robust_z(company.get("ev_ebitda"), peers_df["ev_ebitda"] if "ev_ebitda" in peers_df else []),
        ps_z=robust_z(company.get("ps_ttm"), peers_df["ps_ttm"] if "ps_ttm" in peers_df else []),
    )
    logger.info(f"zscores: {out.to_dict()}")
    return out

@debug_io()
def fair_value_from_comparables(info: Dict[str, Any], peers: PeerMultiples) -> Optional[float]:
    price   = _safe_float(info.get("regularMarketPrice"), np.nan)
    shares  = _safe_float(info.get("sharesOutstanding") or info.get("floatShares"), np.nan)
    eps_ttm = _safe_float(info.get("trailingEps"), np.nan)
    revenue = _safe_float(info.get("totalRevenue"), np.nan)
    ebitda  = _safe_float(info.get("ebitda"), np.nan)
    net_debt = None
    try:
        total_debt = _safe_float(info.get("totalDebt"), np.nan)
        total_cash = _safe_float(info.get("totalCash"), np.nan)
        if math.isfinite(total_debt) and math.isfinite(total_cash):
            net_debt = total_debt - total_cash
    except Exception:
        net_debt = None

    estimates = []
    if eps_ttm and peers.pe_median and peers.pe_median > 0:
        estimates.append(eps_ttm * peers.pe_median)
    if revenue and shares and peers.ps_median and peers.ps_median > 0:
        rev_per_share = revenue / shares
        estimates.append(rev_per_share * peers.ps_median)
    if ebitda and shares and peers.ev_ebitda_median and peers.ev_ebitda_median > 0:
        ev = peers.ev_ebitda_median * ebitda
        equity_value = (ev - net_debt) if net_debt is not None else np.nan
        if math.isfinite(equity_value):
            estimates.append(equity_value / shares)

    estimates = [x for x in estimates if x and math.isfinite(x) and x > 0]
    fv = float(np.median(estimates)) if estimates else None
    logger.info(f"fv_comparables: {fv} (estimates={len(estimates)})")
    return fv

# ----------------------------- DCF simplifié -----------------------------------

@debug_io()
def dcf_simplified(fundamentals: Dict[str, pd.DataFrame],
                   info: Dict[str, Any],
                   years: int = 5,
                   growth_g: Optional[float] = 0.02,
                   wacc: Optional[float] = None,
                   tax_rate: float = DEFAULT_TAX_RATE,
                   g_low: float = DEFAULT_PERPET_G_MIN,
                   g_high: float = DEFAULT_PERPET_G_MAX,
                   wacc_low: float = DEFAULT_WACC_MIN,
                   wacc_high: float = DEFAULT_WACC_MAX) -> DCFResult:
    cashflw = fundamentals.get("cash_flow", pd.DataFrame())
    shares = _safe_float(info.get("sharesOutstanding") or info.get("floatShares"), np.nan)
    price  = _safe_float(info.get("regularMarketPrice"), np.nan)
    if not (shares and shares > 0 and price and price > 0):
        logger.warning("DCF: shares/price manquants → confidence=0")
        return DCFResult(confidence=0.0)

    cfo_row   = pick_first_index(cashflw, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex_row = pick_first_index(cashflw, "Capital Expenditure", "Capital Expenditures")
    try:
        cfo   = _get_series_last(cashflw.loc[cfo_row]) if cfo_row else None
        capex = _get_series_last(cashflw.loc[capex_row]) if capex_row else None
        if cfo is None or capex is None:
            logger.warning("DCF: CFO/Capex manquants")
            return DCFResult(confidence=0.0)
        fcf0 = cfo + capex
    except Exception:
        return DCFResult(confidence=0.0)

    if wacc is None:
        rf = load_rf_rate() or 0.04
        beta = estimate_beta(info.get("symbol") or "", DEFAULT_BENCHMARK) or 1.0
        wacc = float(np.clip(rf + beta * DEFAULT_MRP, wacc_low, wacc_high))

    def pv_of_cashflows(fcf0: float, g: float, w: float, n: int) -> float:
        if w <= g:
            w = g + 0.005
        fcf = fcf0
        pv = 0.0
        for t in range(1, n + 1):
            fcf *= (1.0 + g)
            pv += fcf / ((1.0 + w) ** t)
        fcf_terminal = fcf * (1.0 + g)
        tv = fcf_terminal / (w - g)
        pv_tv = tv / ((1.0 + w) ** n)
        return pv + pv_tv

    try:
        pv_central = pv_of_cashflows(fcf0, growth_g or 0.0, wacc, years)
        fv_central = pv_central / shares
    except Exception:
        return DCFResult(confidence=0.0)

    try:
        low  = pv_of_cashflows(fcf0, g_low,  wacc_high, years) / shares
        high = pv_of_cashflows(fcf0, g_high, wacc_low,  years) / shares
    except Exception:
        low, high = None, None

    conf = 0.7 if all([shares, fcf0, wacc]) and years >= 5 else 0.5
    if low is None or high is None:
        conf = min(conf, 0.6)

    out = DCFResult(
        fv_central=float(fv_central) if math.isfinite(fv_central) else None,
        fv_low=float(low) if (low is not None and math.isfinite(low)) else None,
        fv_high=float(high) if (high is not None and math.isfinite(high)) else None,
        wacc_used=float(wacc),
        g_used=float(growth_g or 0.0),
        confidence=float(conf),
    )
    logger.info(f"DCF: {out.to_dict()}")
    return out

# ----------------------------- Agrégation Fair Value ---------------------------

@debug_io()
def aggregate_fair_value(current_price: Optional[float],
                         fv_cmp: Optional[float],
                         dcf_res: DCFResult) -> FairValueAggregate:
    weights, values = [], []
    if fv_cmp and math.isfinite(fv_cmp) and fv_cmp > 0:
        weights.append(0.5); values.append(fv_cmp)
    if dcf_res and dcf_res.fv_central and math.isfinite(dcf_res.fv_central) and dcf_res.fv_central > 0:
        w = 0.5 * max(0.4, min(1.0, dcf_res.confidence))
        weights.append(w); values.append(dcf_res.fv_central)

    fv_composite = None
    if values and weights:
        wsum = sum(weights)
        if wsum > 0:
            fv_composite = float(np.average(values, weights=weights))

    upside = None
    if current_price and fv_composite and current_price > 0:
        upside = (fv_composite / current_price - 1.0) * 100.0

    conf = 0.0
    if fv_cmp:
        conf += 0.4
    if dcf_res and dcf_res.confidence:
        conf += 0.6 * dcf_res.confidence

    out = FairValueAggregate(
        fv_comparable=float(fv_cmp) if fv_cmp else None,
        fv_dcf=float(dcf_res.fv_central) if (dcf_res and dcf_res.fv_central) else None,
        fv_composite=float(fv_composite) if fv_composite else None,
        upside_pct=float(upside) if upside is not None else None,
        confidence=float(min(1.0, conf))
    )
    logger.info(f"FairValue aggregate: {out.to_dict()}")
    return out

# ----------------------------- Vue principale Phase 1 --------------------------

@debug_io()
def build_fundamental_view(ticker: str,
                           fallback_peers: Optional[List[str]] = None,
                           dcf_years: int = 5,
                           dcf_g: float = 0.02,
                           dcf_wacc: Optional[float] = None) -> FundamentalView:
    notes: List[str] = []
    info = load_info(ticker)
    info["symbol"] = info.get("symbol") or ticker

    px = load_prices(ticker, period="1y")
    if not px.empty:
        price = float(px["Close"].dropna().iloc[-1])
    else:
        p = _safe_float(info.get("regularMarketPrice"), np.nan)
        price = float(p) if math.isfinite(p) else None
        if price is None:
            notes.append("Prix introuvable: utilisation partielle des modèles.")
    currency = info.get("currency") or _currency_of(ticker) or "USD"

    fundamentals = load_fundamentals(ticker)
    health = compute_health_ratios(ticker, fundamentals, info)

    peer_list = build_peer_set(ticker, info, fallback_peers=fallback_peers)
    peer_df = fetch_peer_multiples(peer_list) if peer_list else pd.DataFrame()
    peers_summary = summarize_peer_multiples(peer_df)

    company_mult = compute_company_multiples(info, fundamentals)
    zscores = compute_zscores_company_vs_peers(company_mult, peer_df) if not peer_df.empty else ComparableZScores()

    fv_cmp = fair_value_from_comparables(info, peers_summary) if peers_summary.sample_size > 0 else None
    dcf_res = dcf_simplified(fundamentals, info, years=dcf_years, growth_g=dcf_g, wacc=dcf_wacc)
    fair = aggregate_fair_value(price, fv_cmp, dcf_res)

    if peers_summary.sample_size < 5:
        notes.append("Échantillon de pairs limité — confiance comparables réduite.")
    if dcf_res.confidence < 0.6:
        notes.append("DCF basé sur infos partielles — utiliser avec prudence.")
    if health.fcf_yield_pct is None:
        notes.append("FCF introuvable — DCF peut sous/over-estimer la valeur.")
    if fair.upside_pct is not None and fair.upside_pct < -20:
        notes.append("Upside négatif marqué — vérifier hypothèses et cyclicité.")
    if fair.upside_pct is not None and fair.upside_pct > 30:
        notes.append("Upside positif élevé — suspecter risque de 'value trap' ou données incomplètes.")

    view = FundamentalView(
        ticker=ticker,
        currency=currency,
        price=price,
        health=health,
        peers=peers_summary,
        zscores=zscores,
        dcf=dcf_res,
        fair_value=fair,
        notes=notes
    )
    logger.info(f"[{ticker}] SUMMARY: price={price} | currency={currency} | "
                f"peers_n={peers_summary.sample_size} | fv={fair.fv_composite} | upside={fair.upside_pct}%")
    return view

# ----------------------------- CLI / Exécution directe ------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    import argparse
    parser = argparse.ArgumentParser(description="Phase1 fondamental — avec logs détaillés (mode debug).")
    parser.add_argument("--ticker", "-t", default="AAPL")
    parser.add_argument("--peers", "-p", nargs="*", default=["MSFT","GOOGL","AMZN","META","NVDA"])
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--g", type=float, default=0.02)
    parser.add_argument("--wacc", type=float, default=None)
    parser.add_argument("--log", default=os.getenv("PHASE1_LOG","INFO"),
                        choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                        help="Niveau de logs. DEBUG montre toutes les étapes & inputs.")
    args = parser.parse_args()

    # init logging selon --log (prend le dessus sur PHASE1_DEBUG)
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(getattr(logging, args.log))

    pyver = os.sys.version.split()[0]
    logger.info(f"Python: {pyver} | NumPy: {np.__version__} | Pandas: {pd.__version__} | yfinance: {getattr(yf, '__version__', '?')}")
    logger.info(f"Ticker: {args.ticker} | Peers: {args.peers} | years={args.years} g={args.g} wacc={args.wacc}")

    view = build_fundamental_view(args.ticker, fallback_peers=args.peers, dcf_years=args.years, dcf_g=args.g, dcf_wacc=args.wacc)

    print(json.dumps({
        "ticker": view.ticker,
        "currency": view.currency,
        "price": view.price,
        "health": view.health.to_dict(),
        "peers": view.peers.to_dict(),
        "zscores": view.zscores.to_dict(),
        "dcf": view.dcf.to_dict(),
        "fair_value": view.fair_value.to_dict(),
        "notes": view.notes
    }, indent=2, ensure_ascii=False))