# phase3_macro.py
# -*- coding: utf-8 -*-
"""
Phase 3 — Macro & Nowcasting pour actions US/CA

Gratuit et robuste:
- FRED: téléchargement CSV public (sans clé) via fredgraph.csv
- yfinance: proxies marchés (USD, Or, WTI, Copper, 10Y)

Fonctions clés:
- fetch_fred_series(), get_us_macro_bundle()
- resample_align(), macro_nowcast()
- build_macro_factors(), rolling_betas(), factor_model()
- macro_regime(), scenario_impact()

Dépendances:
    pip install pandas numpy yfinance
    (optionnel) pip install statsmodels

Auteur: toi + IA (2025) — Licence MIT (à adapter)
"""
from __future__ import annotations

import io
import math
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf

# statsmodels (optionnel)
try:
    import statsmodels.api as sm  # type: ignore
    HAS_SM = True
except Exception:
    HAS_SM = False

TRADING_DAYS = 252

# ----------------------------- Dataclasses ---------------------------------- #

@dataclass
class MacroBundle:
    """Conteneur de séries macro alignées (mensuelles par défaut)."""
    data: pd.DataFrame
    meta: Dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return self.data

    def to_dict(self) -> Dict[str, Any]:
        m = dict(self.meta)
        m["columns"] = list(self.data.columns)
        m["rows"] = int(len(self.data))
        return m


@dataclass
class NowcastView:
    """
    Indices synthétiques standardisés (z-scores): Growth, Inflation, Policy, USD, Commodities.
    """
    scores: Dict[str, float]
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"scores": dict(self.scores), "components": dict(self.components)}


@dataclass
class ExposureReport:
    """Expositions macro d’un titre (rolling β, OLS multi-facteurs)."""
    rolling_betas: pd.DataFrame
    ols_loadings: Dict[str, float]
    r2: float
    stability: Dict[str, float]  # ex: std(β) / |mean(β)|

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ols_loadings": {k: float(v) for k, v in self.ols_loadings.items()},
            "r2": float(self.r2),
            "stability": {k: float(v) for k, v in self.stability.items()},
            "rolling_beta_cols": list(self.rolling_betas.columns),
        }


@dataclass
class MacroRegimeView:
    """Classification de régime macro agrégée."""
    label: str
    growth_z: float
    inflation_z: float
    policy_z: float
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["growth_z"] = float(self.growth_z)
        d["inflation_z"] = float(self.inflation_z)
        d["policy_z"] = float(self.policy_z)
        return d


@dataclass
class ScenarioImpact:
    """Projection d’impact (%) sur le titre pour des chocs macro instantanés."""
    deltas: Dict[str, float]   # choc (ex: {"USD": +0.05, "10Y": +0.005})
    expected_return_pct: float
    detail: Dict[str, float]   # contribution par facteur (%)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deltas": {k: float(v) for k, v in self.deltas.items()},
            "expected_return_pct": float(self.expected_return_pct),
            "detail": {k: float(v) for k, v in self.detail.items()},
        }

# ---------------------------- Utils & Fetchers ------------------------------- #

def _fred_csv(series_id: str, start: Optional[str] = None) -> pd.Series:
    """
    Télécharge une série FRED via CSV public.
    start: 'YYYY-MM-DD' (optionnel)
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    if start:
        url += f"&startdate={start}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        raw = resp.read()
    df = pd.read_csv(io.BytesIO(raw))
    if "DATE" not in df.columns or series_id not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_datetime(df["DATE"])
    v = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
    out = pd.Series(v.values, index=s, name=series_id).sort_index()
    return out.dropna()


def fetch_fred_series(series: List[str], start: Optional[str] = None, sleep: float = 0.15) -> pd.DataFrame:
    """Batch FRED (tolérant aux échecs)."""
    data = {}
    for sid in series:
        try:
            s = _fred_csv(sid, start=start)
            if not s.empty:
                data[sid] = s
        except Exception:
            pass
        time.sleep(sleep)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def fetch_market_proxies(period: str = "10y") -> pd.DataFrame:
    """
    Proxies marchés via yfinance (journaliers):
      - DXY: "DX-Y.NYB" (USD Index)
      - Gold: "GC=F"
      - WTI: "CL=F"
      - Copper: "HG=F"
      - US10Y: "^TNX" (rendement 10Y *100, donc /100 pour décimal)
    """
    tickers = ["DX-Y.NYB", "GC=F", "CL=F", "HG=F", "^TNX"]
    frames = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period=period, auto_adjust=True)
            if getattr(h.index, "tz", None) is not None:
                h.index = h.index.tz_localize(None)
            if not h.empty:
                col = "Close"
                if t == "^TNX":
                    # ^TNX = yield * 100
                    frames.append(h[[col]].rename(columns={col: "US10Y"}).assign(US10Y=lambda x: x["US10Y"] / 100.0))
                else:
                    frames.append(h[[col]].rename(columns={col: t}))
        except Exception:
            pass
        time.sleep(0.1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).dropna(how="all")
    return df


def resample_align(df: pd.DataFrame, freq: str = "M", method: str = "last") -> pd.DataFrame:
    """
    Standardise la fréquence (mensuelle par défaut).
    method: 'last' (par défaut), 'mean'
    """
    if df.empty:
        return df
    if method == "mean":
        out = df.resample(freq).mean()
    else:
        out = df.resample(freq).last()
    # supprime colonnes full-NaN
    return out.dropna(how="all")


def pct_chg(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return df.pct_change(periods=periods)


def yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Croissance annuelle (YoY) pour données mensuelles/quarterly."""
    return df.pct_change(12)


def zscore_df(df: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """Z-score par colonne, winsorisé pour robustesse."""
    x = df.copy()
    mu = x.rolling(24, min_periods=12).mean()
    sd = x.rolling(24, min_periods=12).std()
    z = (x - mu) / sd.replace(0, np.nan)
    z = z.clip(lower=-winsor, upper=winsor)
    return z

# ----------------------------- Macro bundle US ------------------------------- #

def get_us_macro_bundle(start: str = "2000-01-01",
                        monthly: bool = True) -> MacroBundle:
    """
    Récupère un panier de séries macro US 'core' :
      - Growth: INDPRO, PAYEMS, RETAIL SALES (RSAFS), ISM MANUFACTURING PMI (NAPM)
      - Inflation: CPIAUCSL, CORE CPI (CPILFESL), 5y5y (T5YIFR) proxy: T10YIE (breakeven 10Y)
      - Policy: Fed Funds Rate (FEDFUNDS), 10Y (DGS10), 2Y (DGS2)
      - USD: DTWEXBGS (Dollar broad)
      - Credit/Stress: NFCI
    Et proxies marchés (Gold, DXY, WTI, Copper, US10Y).
    """
    fred_ids = [
        # Growth
        "INDPRO", "PAYEMS", "RSAFS", "NAPM",
        # Inflation
        "CPIAUCSL", "CPILFESL", "T10YIE",
        # Policy & rates
        "FEDFUNDS", "DGS10", "DGS2",
        # USD broad
        "DTWEXBGS",
        # Stress
        "NFCI"
    ]
    fred = fetch_fred_series(fred_ids, start=start)
    # markets daily
    mkt = fetch_market_proxies(period="max")
    # resample
    if monthly:
        fred_m = resample_align(fred, "M", "last")
        mkt_m = resample_align(mkt, "M", "last")
    else:
        fred_m = resample_align(fred, "W", "last")
        mkt_m = resample_align(mkt, "W", "last")

    data = pd.concat([fred_m, mkt_m], axis=1)
    data = data.dropna(how="all")
    meta = {"country": "US", "freq": "M" if monthly else "W", "source": "FRED+yfinance"}
    return MacroBundle(data=data, meta=meta)

# ------------------------------- Nowcasting ---------------------------------- #

def macro_nowcast(bundle: MacroBundle) -> NowcastView:
    """
    Construit des scores synthétiques à partir de YoY (growth, inflation)
    et niveaux / spreads (policy), ainsi que proxies (USD, commodities).
    Méthode:
      - Growth: z-score moyenne de {INDPRO_yoy, PAYEMS_yoy, RSAFS_yoy, NAPM (demean)}
      - Inflation: z-score moyenne de {CPIAUCSL_yoy, CPILFESL_yoy, T10YIE (demean)}
      - Policy: z-score moyenne de {FEDFUNDS, (DGS10-DGS2) inversé pour restrictive tilt}
      - USD: z-score de DTWEXBGS_yoy (ou DXY si dispo)
      - Commodities: z-score moyenne de {gold_yoy, wti_yoy, copper_yoy}
    """
    df = bundle.data.copy()

    # YoY calculable pour séries > 12 mois
    yoy_cols = {}
    def _safe_yoy(col):
        if col in df.columns:
            return df[[col]].pct_change(12).rename(columns={col: col + "_YoY"})
        return pd.DataFrame()

    parts = []
    for c in ["INDPRO", "PAYEMS", "RSAFS"]:
        parts.append(_safe_yoy(c))
    growth = pd.concat(parts + [df[["NAPM"]].apply(lambda s: s - s.rolling(24, min_periods=12).mean()) if "NAPM" in df else pd.DataFrame()], axis=1)

    infl_parts = []
    for c in ["CPIAUCSL", "CPILFESL"]:
        infl_parts.append(_safe_yoy(c))
    # T10YIE (niveau, recentré)
    if "T10YIE" in df:
        infl_parts.append((df[["T10YIE"]] - df["T10YIE"].rolling(24, min_periods=12).mean()).rename(columns={"T10YIE": "T10YIE_dev"}))
    inflation = pd.concat(infl_parts, axis=1)

    # Policy: FedFunds (niveau recentré) + slope yield (2s10s) inversé (plus inversé → plus restrictif)
    pol_parts = []
    if "FEDFUNDS" in df:
        pol_parts.append((df[["FEDFUNDS"]] - df["FEDFUNDS"].rolling(24, min_periods=12).mean()).rename(columns={"FEDFUNDS": "FEDFUNDS_dev"}))
    if "DGS10" in df and "DGS2" in df:
        slope = (df["DGS10"] - df["DGS2"]).to_frame("Slope_10y_2y")
        pol_parts.append((-slope).rename(columns={"Slope_10y_2y": "Policy_Tightness"}))
    policy = pd.concat(pol_parts, axis=1)

    # USD
    usd_parts = []
    if "DTWEXBGS" in df:
        usd_parts.append(df[["DTWEXBGS"]].pct_change(12).rename(columns={"DTWEXBGS": "DTWEXBGS_YoY"}))
    if "DX-Y.NYB" in df:
        usd_parts.append(df[["DX-Y.NYB"]].pct_change(12).rename(columns={"DX-Y.NYB": "DXY_YoY"}))
    usd = pd.concat(usd_parts, axis=1)

    # Commodities
    com_parts = []
    for col, out in [("GC=F", "Gold_YoY"), ("CL=F", "WTI_YoY"), ("HG=F", "Copper_YoY")]:
        if col in df:
            com_parts.append(df[[col]].pct_change(12).rename(columns={col: out}))
    commodities = pd.concat(com_parts, axis=1)

    # z-scores
    growth_z = zscore_df(growth).mean(axis=1)
    infl_z = zscore_df(inflation).mean(axis=1)
    policy_z = zscore_df(policy).mean(axis=1)
    usd_z = zscore_df(usd).mean(axis=1)
    com_z = zscore_df(commodities).mean(axis=1)

    latest = pd.concat([
        growth_z.rename("GrowthZ"),
        infl_z.rename("InflationZ"),
        policy_z.rename("PolicyZ"),
        usd_z.rename("USDZ"),
        com_z.rename("CommoditiesZ")
    ], axis=1).dropna().iloc[-1]

    # composants (derniers dispo)
    comps = {
        "INDPRO_YoY": float(growth.filter(like="INDPRO").iloc[-1]) if not growth.empty and growth.filter(like="INDPRO").shape[1] else np.nan,
        "PAYEMS_YoY": float(growth.filter(like="PAYEMS").iloc[-1]) if growth.filter(like="PAYEMS").shape[1] else np.nan,
        "CPI_YoY": float(inflation.filter(like="CPIAUCSL").iloc[-1]) if inflation.filter(like="CPIAUCSL").shape[1] else np.nan,
        "CoreCPI_YoY": float(inflation.filter(like="CPILFESL").iloc[-1]) if inflation.filter(like="CPILFESL").shape[1] else np.nan,
        "Breakeven_dev": float(inflation.filter(like="T10YIE_dev").iloc[-1]) if inflation.filter(like="T10YIE_dev").shape[1] else np.nan,
        "FedFunds_dev": float(policy.filter(like="FEDFUNDS_dev").iloc[-1]) if policy.filter(like="FEDFUNDS_dev").shape[1] else np.nan,
        "YieldSlope_Tight": float(policy.filter(like="Policy_Tightness").iloc[-1]) if policy.filter(like="Policy_Tightness").shape[1] else np.nan,
        "USD_YoY": float(usd.mean(axis=1).iloc[-1]) if not usd.empty else np.nan,
        "Commodities_YoY": float(commodities.mean(axis=1).iloc[-1]) if not commodities.empty else np.nan
    }

    scores = {
        "Growth": float(latest["GrowthZ"]) if "GrowthZ" in latest else np.nan,
        "Inflation": float(latest["InflationZ"]) if "InflationZ" in latest else np.nan,
        "Policy": float(latest["PolicyZ"]) if "PolicyZ" in latest else np.nan,
        "USD": float(latest["USDZ"]) if "USDZ" in latest else np.nan,
        "Commodities": float(latest["CommoditiesZ"]) if "CommoditiesZ" in latest else np.nan,
    }
    return NowcastView(scores=scores, components=comps)

# --------------------------- Facteurs & Expositions -------------------------- #

def build_macro_factors(bundle: MacroBundle) -> pd.DataFrame:
    """
    Produit un set compact de facteurs (mensuels):
      - GRW: moyenne z de {INDPRO_yoy, PAYEMS_yoy, RSAFS_yoy, NAPM_dev}
      - INF: moyenne z de {CPI_yoy, CORE_yoy, T10YIE_dev}
      - POL: z de FedFunds_dev + (−)slope(10y-2y)
      - USD: z de broad/dxy yoy
      - CMD: z de yoy {gold, wti, copper}
      - RATE10: variation de US10Y (Δ, mensuel)
    """
    df = bundle.data.copy()
    # utiliser les mêmes constructions que nowcast
    nc = macro_nowcast(bundle)
    # On reconstitue les séries sous-jacentes pour tout l'historique
    # Growth block
    g_parts = []
    for c in ["INDPRO", "PAYEMS", "RSAFS"]:
        if c in df:
            g_parts.append(df[[c]].pct_change(12).rename(columns={c: c + "_YoY"}))
    if "NAPM" in df:
        g_parts.append((df[["NAPM"]] - df["NAPM"].rolling(24, min_periods=12).mean()).rename(columns={"NAPM": "NAPM_dev"}))
    G = zscore_df(pd.concat(g_parts, axis=1)).mean(axis=1).rename("GRW")

    # Inflation block
    i_parts = []
    for c in ["CPIAUCSL", "CPILFESL"]:
        if c in df:
            i_parts.append(df[[c]].pct_change(12).rename(columns={c: c + "_YoY"}))
    if "T10YIE" in df:
        i_parts.append((df[["T10YIE"]] - df["T10YIE"].rolling(24, min_periods=12).mean()).rename(columns={"T10YIE": "T10YIE_dev"}))
    I = zscore_df(pd.concat(i_parts, axis=1)).mean(axis=1).rename("INF")

    # Policy block
    p_parts = []
    if "FEDFUNDS" in df:
        p_parts.append((df[["FEDFUNDS"]] - df["FEDFUNDS"].rolling(24, min_periods=12).mean()).rename(columns={"FEDFUNDS": "FEDFUNDS_dev"}))
    if "DGS10" in df and "DGS2" in df:
        slope = (df["DGS10"] - df["DGS2"]).to_frame("Slope")
        p_parts.append((-slope).rename(columns={"Slope": "Policy_Tight"}))
    P = zscore_df(pd.concat(p_parts, axis=1)).mean(axis=1).rename("POL")

    # USD
    u_parts = []
    if "DTWEXBGS" in df:
        u_parts.append(df[["DTWEXBGS"]].pct_change(12).rename(columns={"DTWEXBGS": "DTWEXBGS_YoY"}))
    if "DX-Y.NYB" in df:
        u_parts.append(df[["DX-Y.NYB"]].pct_change(12).rename(columns={"DX-Y.NYB": "DXY_YoY"}))
    U = zscore_df(pd.concat(u_parts, axis=1)).mean(axis=1).rename("USD")

    # Commodities
    c_parts = []
    for col in ["GC=F", "CL=F", "HG=F"]:
        if col in df:
            c_parts.append(df[[col]].pct_change(12).rename(columns={col: col + "_YoY"}))
    C = zscore_df(pd.concat(c_parts, axis=1)).mean(axis=1).rename("CMD")

    # Rate10 delta (niveau → Δ mensuel)
    RATE10 = None
    if "US10Y" in df:
        RATE10 = df["US10Y"].diff().rename("RATE10")
    elif "DGS10" in df:
        RATE10 = (df["DGS10"] / 100.0).diff().rename("RATE10")

    facs = pd.concat([G, I, P, U, C, RATE10], axis=1)
    return facs.dropna(how="all")


def _align_stock_factors(ticker: str,
                         factors: pd.DataFrame,
                         period: str = "10y") -> Tuple[pd.Series, pd.DataFrame]:
    """
    Aligne les rendements mensuels (ou hebdo) du ticker sur le dataframe de facteurs.
    """
    px = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    if px.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    # Mensualisation
    pr_m = px["Close"].resample("M").last()
    ret_m = pr_m.pct_change().dropna()
    fac_m = factors.copy()
    common = ret_m.index.intersection(fac_m.index)
    return ret_m.loc[common], fac_m.loc[common]


def rolling_betas(ret: pd.Series, facs: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Rolling OLS: ret_t ~ a + b*GRW + b*INF + b*POL + b*USD + b*CMD + b*RATE10
    Fallback numpy si statsmodels absent.
    """
    cols = [c for c in ["GRW", "INF", "POL", "USD", "CMD", "RATE10"] if c in facs.columns]
    X = facs[cols].copy()
    Y = ret.copy()
    out = pd.DataFrame(index=Y.index, columns=cols, dtype=float)

    for i in range(window, len(Y)+1):
        yi = Y.iloc[i-window:i]
        xi = X.iloc[i-window:i]
        xi = xi.dropna()
        yi = yi.loc[xi.index]
        if len(xi) < window * 0.8:
            continue
        # OLS
        if HAS_SM:
            xx = sm.add_constant(xi.values)
            model = sm.OLS(yi.values, xx, missing="drop").fit()
            betas = model.params[1:]  # sans constante
        else:
            xx = np.c_[np.ones(len(xi)), xi.values]
            try:
                betas = np.linalg.lstsq(xx, yi.values, rcond=None)[0][1:]
            except Exception:
                continue
        out.loc[yi.index[-1], cols] = betas
    return out


def factor_model(ret: pd.Series, facs: pd.DataFrame) -> ExposureReport:
    """
    OLS global pour obtenir des loadings 'moyens' + R².
    Calcule aussi la stabilité: std(β) / |mean(β)| en rolling.
    """
    cols = [c for c in ["GRW", "INF", "POL", "USD", "CMD", "RATE10"] if c in facs.columns]
    X = facs[cols].dropna()
    Y = ret.loc[X.index]
    if len(Y) < 24:
        return ExposureReport(pd.DataFrame(), {}, np.nan, {})

    # OLS global
    if HAS_SM:
        Xc = sm.add_constant(X.values)
        model = sm.OLS(Y.values, Xc).fit()
        load = dict(zip(cols, model.params[1:].tolist()))
        r2 = float(model.rsquared)
    else:
        Xc = np.c_[np.ones(len(X)), X.values]
        params = np.linalg.lstsq(Xc, Y.values, rcond=None)[0]
        load = dict(zip(cols, params[1:].tolist()))
        # R2
        yhat = Xc @ params
        ssr = np.sum((Y.values - yhat) ** 2)
        sst = np.sum((Y.values - Y.values.mean()) ** 2)
        r2 = 1.0 - ssr / sst if sst > 0 else np.nan

    # Rolling betas pour stabilité
    rb = rolling_betas(Y, X, window=24)
    stability = {}
    for c in cols:
        if c in rb and rb[c].dropna().size:
            m = rb[c].mean()
            s = rb[c].std()
            stability[c] = float(s / abs(m)) if m not in (0, np.nan) and abs(m) > 1e-9 else np.inf

    return ExposureReport(rolling_betas=rb, ols_loadings=load, r2=r2, stability=stability)

# --------------------------- Macro Regime Classifier ------------------------- #

def macro_regime(nc: NowcastView) -> MacroRegimeView:
    """
    Classification simple par règles sur z-scores:
      - Reflation: Growth>0 & Inflation>0 & Policy accommodante (Policy<0)
      - Goldilocks: Growth>0 & Inflation<0 (désinflation avec croissance)
      - Stagflation: Growth<0 & Inflation>0
      - Désinflation restrictive: Inflation<0 & Policy>0 (resserrement)
      Sinon: 'Transition'
    """
    g = nc.scores.get("Growth", np.nan)
    i = nc.scores.get("Inflation", np.nan)
    p = nc.scores.get("Policy", np.nan)

    if np.isfinite(g) and np.isfinite(i) and np.isfinite(p):
        if g > 0 and i > 0 and p < 0:
            lab = "Reflation"
        elif g > 0 and i < 0:
            lab = "Goldilocks"
        elif g < 0 and i > 0:
            lab = "Stagflation"
        elif i < 0 and p > 0:
            lab = "Désinflation restrictive"
        else:
            lab = "Transition"
    else:
        lab = "Inconnu"

    extra = {"USD": nc.scores.get("USD"), "Commodities": nc.scores.get("Commodities")}
    return MacroRegimeView(label=lab, growth_z=g, inflation_z=i, policy_z=p, extra=extra)

# ------------------------------- Scénarios ----------------------------------- #

def scenario_impact(exposure: ExposureReport,
                    deltas: Dict[str, float]) -> ScenarioImpact:
    """
    Estime l'impact instantané (%) sur le titre pour des chocs de facteurs.
    Convention des deltas:
      - GRW, INF, POL, USD, CMD: variation en 'z' (écarts-types) — si vous donnez des %,
        convertissez-les d'abord en z vs. historique, sinon supposez 1 z ≈ move 'normal'.
      - RATE10: choc en points décimaux de taux (ex: +0.005 = +50 pb)
    Exemple:
        {"USD": +1.0, "RATE10": +0.005, "CMD": -0.5}
    """
    if not exposure.ols_loadings:
        return ScenarioImpact(deltas=deltas, expected_return_pct=np.nan, detail={})

    load = exposure.ols_loadings
    detail: Dict[str, float] = {}
    ret = 0.0
    for k, dv in deltas.items():
        if k in load:
            contrib = load[k] * dv * 100.0  # en %
            detail[k] = float(contrib)
            ret += contrib
        else:
            # facteur absent → contribution nulle
            detail[k] = 0.0
    return ScenarioImpact(deltas=deltas, expected_return_pct=float(ret), detail=detail)

# --------------------------- API "haut niveau" --------------------------------#

def get_macro_features() -> Dict[str, Any]:
    """
    Get current macro features using the macro nowcast function.
    This is a wrapper function to match the expected signature from app.py.

    Returns:
        Dict: Current macro features suitable for conversion to dict
    """
    try:
        # Get the US macro bundle
        bundle = get_us_macro_bundle(start="2000-01-01", monthly=True)

        # Generate the macro nowcast
        nowcast = macro_nowcast(bundle)

        # Return as dict
        return {
            "macro_nowcast": nowcast.to_dict(),
            "timestamp": bundle.data.index[-1] if not bundle.data.empty else None,
            "meta": bundle.meta
        }

    except Exception as e:
        # Return error dict
        return {
            "error": f"Failed to get macro features: {str(e)}",
            "macro_nowcast": {
                "scores": {},
                "components": {}
            },
            "meta": {}
        }


def build_macro_view(ticker: str,
                     start: str = "2000-01-01",
                     period_stock: str = "15y") -> Dict[str, Any]:
    """
    Pipeline complet:
      1) Récupère macro US (mensuel) + proxies
      2) Nowcast (scores z)
      3) Facteurs agrégés
      4) Aligne rendements mensuels du ticker
      5) Expositions (OLS + rolling β) & R²
      6) Régime macro
    Retourne un dict compact (prêt à intégrer dans l’app).
    """
    bundle = get_us_macro_bundle(start=start, monthly=True)
    nc = macro_nowcast(bundle)
    facs = build_macro_factors(bundle)
    ret_m, facs_m = _align_stock_factors(ticker, facs, period=period_stock)
    expo = factor_model(ret_m, facs_m)
    reg = macro_regime(nc)

    # Résumé "drivers"
    drivers = sorted(expo.ols_loadings.items(), key=lambda kv: abs(kv[1]), reverse=True) if expo.ols_loadings else []
    top_drivers = [f"{k}:{v:+.2f}" for k, v in drivers[:4]]

    return {
        "ticker": ticker,
        "macro_meta": bundle.meta,
        "nowcast": nc.to_dict(),
        "regime": reg.to_dict(),
        "exposure": expo.to_dict(),
        "top_drivers": top_drivers
    }

# --------------------------------- Exemple ----------------------------------- #

if __name__ == "__main__":
    import json

    TICKER = "AEM.TO"   # exemple Canada (or)
    view = build_macro_view(TICKER, start="2003-01-01", period_stock="20y")
    print("=== MACRO VIEW ===")
    print(json.dumps(view, indent=2))

    # Scénario: USD +1 écart-type, 10Y +50 pb, Commodities -0.5 z
    # (interprétation: choc restrictif, dollar fort, matières premières en repli)
    # → Utilise les loadings OLS estimés
    from pprint import pprint
    bundle = get_us_macro_bundle(start="2003-01-01", monthly=True)
    facs = build_macro_factors(bundle)
    ret_m, facs_m = _align_stock_factors(TICKER, facs, period="20y")
    expo = factor_model(ret_m, facs_m)

    scen = scenario_impact(expo, {"USD": +1.0, "RATE10": +0.005, "CMD": -0.5})
    print("\n=== SCENARIO IMPACT (instantané %) ===")
    pprint(scen.to_dict())
