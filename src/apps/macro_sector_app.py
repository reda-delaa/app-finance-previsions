# (Optionnel) source externe Firecrawl — on rend l'import safe
try:
    from macro_firecrawl import get_macro_data_firecrawl  # optionnel
except Exception:
    def get_macro_data_firecrawl(*args, **kwargs):
        return None  # stub si le module n'est pas dispo

import os
import time
from datetime import datetime, timedelta, date, timezone
import warnings
from typing import Optional, Union
import io
import logging
from functools import wraps
from collections import OrderedDict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st


def render_macro():
    """Fonction exportable pour afficher l'onglet Économie dans le hub"""
    import streamlit as st
    st.header("Prévision macro (économie)")
    st.markdown("---")

# ---------- UI helpers / status registry ----------
DATA_STATUS: "OrderedDict[str, dict]" = OrderedDict()

def set_status(name: str, ok: bool, detail: str = ""):
    """Enregistre l’état d’une source/feature pour l’encart 'Data status'."""
    DATA_STATUS[name] = {"ok": bool(ok), "detail": str(detail or "")}

# Optional: fredapi
try:
    from fredapi import Fred
except Exception:
    Fred = None

# Timezone for calendar
try:
    import pytz
    US_TZ = pytz.timezone("America/New_York")
except Exception:
    US_TZ = None

warnings.filterwarnings("ignore", category=FutureWarning)

# =============== LOGGING (in-memory) ===============
if "logbuf" not in st.session_state:
    st.session_state.logbuf = []

def _log(level: str, msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} UTC] {level}: {msg}"
    st.session_state.logbuf.append(line)
    try:
        logging.getLogger("macroapp").log(getattr(logging, level, logging.INFO), msg)
    except Exception:
        pass

def log_info(msg):  _log("INFO", msg)
def log_warn(msg):  _log("WARNING", msg)
def log_error(msg): _log("ERROR", msg)

def tlog(name=None):
    """Décorateur de timing + logging sur les fonctions de fetch/cache."""
    def deco(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            t0 = time.time()
            try:
                res = fn(*args, **kwargs)
                dur = (time.time() - t0) * 1000
                shape = None
                if isinstance(res, pd.DataFrame):
                    shape = f"{res.shape[0]}x{res.shape[1]}"
                elif isinstance(res, pd.Series):
                    shape = f"{res.shape[0]}"
                log_info(f"{name or fn.__name__} OK in {dur:.0f} ms, shape={shape}")
                return res
            except Exception as e:
                dur = (time.time() - t0) * 1000
                log_error(f"{name or fn.__name__} ERROR in {dur:.0f} ms: {e}")
                raise
        return inner
    return deco

def profile_df(df: Optional[Union[pd.DataFrame, pd.Series]], name: str):
    out = {"name": name, "type": type(df).__name__, "rows": 0, "cols": 0,
           "start": None, "end": None, "freq": None, "na_median_%": None}
    if df is None:
        return out
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    out["rows"], out["cols"] = df.shape
    try:
        out["start"] = str(df.index.min())
        out["end"]   = str(df.index.max())
    except Exception:
        pass
    try:
        out["freq"] = pd.infer_freq(df.index)
    except Exception:
        out["freq"] = None
    try:
        na_pct = (1 - df.notna().sum()/len(df)) * 100
        out["na_median_%"] = float(na_pct.median().round(2))
    except Exception:
        pass
    return out

# ================== BASE SERIES (FRED) ==================
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

SECTOR_ETFS = ["XLB","XLE","XLF","XLV","XLK","XLI","XLY","XLP","XLRE","XLU"]
EXTRA_ETFS  = ["GDX"]  # Gold miners (optional)

DEFAULT_SENS = pd.DataFrame({
    "Inflation":{"XLK":-1,"XLF":1,"XLE":2,"XLB":2,"XLV":0,"XLY":-1,"XLP":0,"XLI":1,"XLRE":-1,"XLU":1,"GDX":2},
    "Growth":   {"XLK": 2,"XLF":1,"XLE":0,"XLB":1,"XLV":0,"XLY": 2,"XLP":0,"XLI":1,"XLRE": 0,"XLU":-1,"GDX":0},
    "Rates":    {"XLK":-2,"XLF":2,"XLE":1,"XLB":0,"XLV":0,"XLY":-1,"XLP":0,"XLI":0,"XLRE":-1,"XLU":1,"GDX":1},
    "USD":      {"XLK":-1,"XLF":0,"XLE":-1,"XLB":-1,"XLV":0,"XLY":0,"XLP":0,"XLI":-1,"XLRE":-1,"XLU":0,"GDX":-1},
    "Jobs":     {"XLK": 1,"XLF":1,"XLE":0,"XLB":1,"XLV":0,"XLY": 1,"XLI":1,"XLRE": 0,"XLU":-1,"GDX":0},
}).fillna(0)

# ================== HELPERS ==================
def zscore(series, win=24):
    s = series.dropna()
    if len(s) < win + 2:
        return pd.Series(index=series.index, dtype=float)
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return ((s - mu) / (sd.replace(0, np.nan))).reindex(series.index)

def normalize_fred_key(k: Optional[str]) -> Optional[str]:
    """Valide la clé FRED (32 car. alphanum en minuscules)."""
    if not k or not isinstance(k, str):
        return None
    k = k.strip()
    if len(k) == 32 and k.isalnum() and k == k.lower():
        return k
    return None

def load_fred_series(series_id: str, fred_key: Optional[str] = None, start: Optional[str] = None) -> pd.DataFrame:
    """
    Charge une série FRED robuste :
      1) API JSON officielle (file_type=json) si clé valide
      2) Fallback CSV fredgraph.csv (sans clé) avec détection flexible des colonnes
    """
    vkey = normalize_fred_key(fred_key)

    # 1) API JSON
    if vkey:
        try:
            params = {
                "series_id": series_id,
                "api_key": vkey,
                "file_type": "json",
            }
            if start:
                params["observation_start"] = pd.to_datetime(start).strftime("%Y-%m-%d")
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            if "observations" in js:
                df = pd.DataFrame(js["observations"])
                if not df.empty and {"date", "value"}.issubset(df.columns):
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
                    df = df.set_index("date")[["value"]].rename(columns={"value": series_id}).sort_index()
                    return df
        except Exception as e:
            log_warn(f"FRED API JSON error for {series_id}: {e}; try CSV fallback")

    # 2) CSV fredgraph
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        text = resp.text.strip()
        if text[:1] in ("<", "{"):
            raise RuntimeError("FRED returned non-CSV content")
        df = pd.read_csv(io.StringIO(text))
        # date col
        date_col = None
        for cand in ("DATE", "date", "observation_date"):
            if cand in df.columns:
                date_col = cand; break
        if date_col is None:
            raise KeyError("CSV missing DATE column")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        # value col
        if series_id in df.columns:
            val_col = series_id
        elif "VALUE" in df.columns:
            val_col = "VALUE"
        elif "value" in df.columns:
            val_col = "value"
        else:
            candidates = [c for c in df.columns if c.lower() not in ("date", "observation_date")]
            if len(candidates) == 1:
                val_col = candidates[0]
            else:
                raise KeyError(f"Cannot locate value column for {series_id}")
        df[val_col] = pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce")
        out = df[[val_col]].rename(columns={val_col: series_id}).sort_index()
        if start:
            out = out[out.index >= pd.to_datetime(start)]
        return out
    except Exception as e:
        log_warn(f"FRED CSV error for {series_id}: {e}")
        return pd.DataFrame(columns=[series_id])

def get_asset_price(symbol, start="2010-01-01"):
    t = yf.Ticker(symbol)
    hist = t.history(start=start, auto_adjust=True)
    s = hist["Close"].rename(symbol)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s

def get_multi_yf(symbols, start="2010-01-01"):
    data = {}
    for s in symbols:
        try:
            data[s] = get_asset_price(s, start=start)
            time.sleep(0.25)
        except Exception as e:
            log_warn(f"yfinance failed for {s}: {e}")
    df = pd.DataFrame(data)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df

def robust_minmax(row):
    vals = row.values.astype(float)
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return pd.Series(0.0, index=row.index)
    finite_vals = vals[finite_mask]
    q1, q3 = np.nanpercentile(finite_vals, [25, 75])
    iqr = q3 - q1
    vmin = np.nanmin(finite_vals)
    vmax = np.nanmax(finite_vals)
    lo = vmin if not np.isfinite(iqr) else min(vmin, q1 - 1.5 * iqr)
    hi = vmax if not np.isfinite(iqr) else max(vmax, q3 + 1.5 * iqr)
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-9:
        return pd.Series(0.0, index=row.index)
    return -5 + 10 * (row - lo) / (hi - lo)

def smooth_scores(df, ema=3):
    return df.ewm(span=ema, adjust=False).mean()

def hysteresis_picks(score_df, k=3, margin=0.15):
    ranks = score_df.rank(axis=1, ascending=False, method="first")
    base = (ranks <= k).astype(float)
    prev = None
    out = base.copy()
    for t in score_df.index:
        row = score_df.loc[t]
        b = base.loc[t]
        if prev is None:
            prev = b
            out.loc[t] = b
            continue
        kth  = row.nlargest(k).min()
        kth1 = row.nlargest(min(k+1, len(row))).min()
        keep  = (prev == 1) & (row >= (kth1 - margin))
        enter = (prev == 0) & (row >= (kth + margin))
        out.loc[t] = (keep | enter).astype(float)
        prev = out.loc[t]
    w = out.div(out.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return w

# ---------- SAFE CALL ----------
def call_safely(_fn, *args, label: str = None, **kwargs):
    """Exécute une fonction; enregistre OK/échec dans DATA_STATUS (pas de bandeaux)."""
    try:
        out = _fn(*args, **kwargs)
        if label:
            ok = out is not None and (not hasattr(out, "empty") or not getattr(out, "empty", False))
            set_status(label, ok, "ok" if ok else "vide")
        return out
    except Exception as e:
        if label:
            set_status(label, False, str(e))
        log_warn(f"{_fn.__name__} failed ({label or ''}): {e}")
        return None

def safe_call(_fn, *args, **kwargs):
    return call_safely(_fn, *args, **kwargs)

# ================== CONNECTORS (FREE/FREEMIUM) ==================
@tlog("fetch_gscpi")
def fetch_gscpi():
    """NY Fed GSCPI robuste → pd.Series(name='GSCPI')."""
    urls = [
        # NY Fed site (peut renvoyer HTML si bloqué)
        "https://www.newyorkfed.org/medialibrary/research/gscpi/files/gscpi_data.csv",
        "https://www.newyorkfed.org/medialibrary/research/gscpi/files/gscpi_data.csv?sc_lang=en",
        # ✅ Miroir GitHub officiel (nyfedresearch)
        "https://raw.githubusercontent.com/nyfedresearch/gscpi/main/data/gscpi_data.csv",
    ]
    last_err = None
    for u in urls:
        try:
            r = requests.get(u, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            text = r.text.strip()
            # Si HTML/JSON -> essayer quand même d'extraire une table simple ; sinon raise
            if not text or text[0] in ("<", "{"):
                log_warn(f"GSCPI: URL {u} returned non-CSV content, skipping")
                continue
            # Essais multi-séparateurs
            for sep in [",", ";", "\t", r"\s+"]:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep if sep != r"\s+" else None, engine="python")
                    if df.shape[1] >= 2:
                        break
                except Exception:
                    continue
            # Colonnes date/valeur
            date_col = None
            for cand in df.columns:
                if str(cand).strip().lower() in ("date", "observation_date"):
                    date_col = cand; break
            if date_col is None:
                date_col = df.columns[0]
            val_col = None
            for cand in df.columns:
                if str(cand).strip().upper() == "GSCPI":
                    val_col = cand; break
            if val_col is None:
                val_col = df.columns[1] if df.shape[1] >= 2 else None
            if val_col is None:
                log_warn(f"GSCPI: Cannot find value column for URL {u}, skipping")
                continue
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            # gérer décimales avec virgule éventuelles
            s = pd.to_numeric(df[val_col].astype(str).str.replace(",", "."),
                              errors="coerce")
            out = pd.Series(s.values, index=df[date_col], name="GSCPI").dropna()
            out = out[~out.index.duplicated()].sort_index()
            if out.empty:
                log_warn(f"GSCPI: Parsed CSV empty from URL {u}, skipping")
                continue
            return out
        except Exception as e:
            last_err = e
            log_warn(f"GSCPI: Failed to fetch from {u}: {e}")
            continue
    # Return None instead of raising an exception to make the system more resilient
    log_warn(f"GSCPI fetch failed for all URLs, returning None. Last error: {last_err}")
    return None

@tlog("fetch_vix_history")
def fetch_vix_history():
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    df = pd.read_csv(url, parse_dates=["DATE"])
    return df.set_index("DATE")["CLOSE"].rename("VIX")

@tlog("fetch_gpr")
def fetch_gpr():
    """Geopolitical Risk Index robuste → pd.Series(name='GPR')."""
    urls = [
        # Sources historiques connues
        "https://www2.bc.edu/matteo-iacoviello/gpr_files/GPRD.csv",
        "https://www.matteoiacoviello.com/gpr_files/GPRD.csv",
        # Miroirs communautaires (au cas où les domaines principaux changent)
        "https://raw.githubusercontent.com/QuantNomad/public-datasets/main/gpr/GPRD.csv",
        "https://raw.githubusercontent.com/jonathan-bower/teaching-datasets/master/datasets/GPRD.csv",
    ]
    last_err = None
    for u in urls:
        try:
            r = requests.get(u, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            text = r.text.strip()
            if not text:
                log_warn(f"GPR: Empty content from URL {u}, skipping")
                continue
            # Essais multi-séparateurs
            for sep in [",", ";", "\t", r"\s+"]:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep if sep != r"\s+" else None, engine="python")
                    if df.shape[1] >= 2:
                        break
                except Exception:
                    continue
            # repère colonnes
            date_col = None
            for cand in df.columns:
                if str(cand).strip().lower() in ("date", "observation_date"):
                    date_col = cand; break
            if date_col is None:
                date_col = df.columns[0]
            val_col = None
            for cand in df.columns:
                if str(cand).strip().upper() == "GPR":
                    val_col = cand; break
            if val_col is None:
                val_col = df.columns[1] if df.shape[1] >= 2 else None
            if val_col is None:
                log_warn(f"GPR: Cannot find GPR value column for URL {u}, skipping")
                continue
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            s = pd.to_numeric(df[val_col].astype(str).str.replace(",", "."),
                              errors="coerce")
            out = pd.Series(s.values, index=df[date_col], name="GPR").dropna()
            out = out[~out.index.duplicated()].sort_index()
            if out.empty:
                log_warn(f"GPR: Parsed CSV empty from URL {u}, skipping")
                continue
            return out
        except Exception as e:
            last_err = e
            log_warn(f"GPR: Failed to fetch from {u}: {e}")
            continue
    # Return None instead of raising an exception to make the system more resilient
    log_warn(f"GPR fetch failed for all URLs, returning None. Last error: {last_err}")
    return None

@tlog("fetch_boc_fx")
def fetch_boc_fx(series="FXUSDCAD"):
    url = f"https://www.bankofcanada.ca/valet/observations/{series}?start_date=2010-01-01"
    r = requests.get(url, timeout=30); r.raise_for_status()
    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df["d"] = pd.to_datetime(df["d"])
    df[series] = pd.to_numeric(df[series], errors="coerce")
    return df.set_index("d")[series]

def fetch_eia(series_id, api_key):
    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    series = r.json()["series"][0]["data"]
    df = pd.DataFrame(series, columns=["date","value"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").fillna(
                 pd.to_datetime(df["date"], format="%Y%m", errors="coerce"))
    return df.set_index("date")["value"].astype(float).sort_index()

def fetch_bls(series_ids, api_key=None, start_year=2010):
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {"seriesid": series_ids,
               "startyear": str(start_year),
               "endyear": str(datetime.today().year)}
    if api_key:
        payload["registrationkey"] = api_key
    r = requests.post(url, json=payload, timeout=30); r.raise_for_status()
    out = {}
    for s in r.json().get("Results",{}).get("series",[]):
        sid = s["seriesID"]; rows = []
        for item in s["data"]:
            y = item["year"]; p = item["period"]
            if p.startswith("M"):
                month = int(p[1:])
                rows.append([pd.to_datetime(f"{y}-{month:02d}-01"), float(item["value"])])
        out[sid] = pd.Series(dict(rows)).sort_index()
    return pd.DataFrame(out)

def fetch_gdelt_events(days=30):
    try:
        frames = []
        for d in range(days):
            day = (datetime.now(timezone.utc) - timedelta(days=d+1)).strftime("%Y%m%d")
            url = f"http://data.gdeltproject.org/gdeltv2/{day}000000.summary.gz"
            df = pd.read_csv(url, compression="gzip", sep="\t", header=None, low_memory=False)
            df["date"] = pd.to_datetime(day)
            frames.append(df[["date"]])
            time.sleep(0.1)
        all_df = pd.concat(frames, ignore_index=True)
        s = all_df.groupby("date").size()
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)

# ===== Calendar (TradingEconomics) =====
def today_ny():
    if US_TZ:
        return datetime.now(US_TZ).date()
    return datetime.now(timezone.utc).date()

@tlog("fetch_te_calendar")
@st.cache_data(ttl=60*15, show_spinner=False)  # refresh 15 min
def fetch_te_calendar(d1: str, d2: str, country="United States", client="guest:guest"):
    base = "https://api.tradingeconomics.com/calendar"
    params = {"d1": d1, "d2": d2, "c": country, "format": "json", "client": client}
    try:
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            return df
        keep = ["Country","Category","Event","Reference","DateUtc","TimeUtc",
                "Importance","Actual","Previous","Forecast"]
        for k in keep:
            if k not in df.columns:
                df[k] = None
        df["ts_utc"] = pd.to_datetime(df["DateUtc"] + " " + df["TimeUtc"], errors="coerce", utc=True)
        if US_TZ is not None:
            df["ts_ny"] = df["ts_utc"].dt.tz_convert(US_TZ)
        else:
            df["ts_ny"] = df["ts_utc"]
        imp_map = {"Low":1, "Medium":2, "High":3}
        df["imp"] = df["Importance"].map(imp_map).fillna(0)
        df = df.sort_values(["imp","ts_ny"], ascending=[False, True])
        return df[keep + ["ts_ny","imp"]]
    except Exception:
        return pd.DataFrame()

# ================== MODEL ==================
def compute_theme_scores(fred_df, params):
    m = fred_df.copy()
    m = m.resample("D").last().ffill().resample("ME").last()

    cpi_yoy      = m["CPIAUCSL"].pct_change(12, fill_method=None)*100 if "CPIAUCSL" in m else pd.Series(index=m.index)
    infl_exp     = m.get("T10YIE", pd.Series(index=m.index))
    indpro_yoy   = m["INDPRO"].pct_change(12, fill_method=None)*100 if "INDPRO" in m else pd.Series(index=m.index)
    unrate       = m.get("UNRATE", pd.Series(index=m.index))
    payrolls_mom = m["PAYEMS"].pct_change(1, fill_method=None)*100 if "PAYEMS" in m else pd.Series(index=m.index)
    dgs10        = m.get("DGS10", pd.Series(index=m.index))
    dgs2         = m.get("DGS2",  pd.Series(index=m.index))
    curve        = (dgs10 - dgs2) if not dgs10.empty and not dgs2.empty else pd.Series(index=m.index)
    usd          = m.get("DTWEXBGS", pd.Series(index=m.index))

    z_infl      = zscore(cpi_yoy,      params["win_infl"]).clip(-3,3)
    z_infl_exp  = zscore(infl_exp,     params["win_infl"]).clip(-3,3)
    z_growth    = zscore(indpro_yoy,   params["win_growth"]).clip(-3,3)
    z_jobs      = (-zscore(unrate,     params["win_jobs"])).clip(-3,3)
    z_payrolls  = zscore(payrolls_mom, params["win_jobs"]).clip(-3,3)
    z_rates     = zscore(dgs10,        params["win_rates"]).clip(-3,3)
    z_curve     = zscore(curve,        params["win_rates"]).clip(-3,3)
    z_usd       = zscore(usd,          params["win_usd"]).clip(-3,3)

    squash = lambda x: np.tanh(x/2.0)
    theme = pd.DataFrame(index=m.index)
    theme["Inflation"] = squash(0.6*z_infl + 0.4*z_infl_exp)
    theme["Growth"]    = squash(0.7*z_growth + 0.3*z_payrolls)
    theme["Jobs"]      = squash(z_jobs)
    theme["Rates"]     = squash(0.7*z_rates - 0.3*z_curve)
    theme["USD"]       = squash(z_usd)
    return theme.fillna(0)

def sector_scores(theme_df, sensitivities):
    raw = theme_df.dot(sensitivities.T)
    raw = raw.where(raw.notna().any(axis=1), other=0)
    scaled = raw.apply(robust_minmax, axis=1)
    return scaled, raw

def backtest_rotation(prices, scores_scaled, top_k=3, ema=3, margin=0.15):
    monthly_prices = prices.resample("ME").last().dropna(how="all")
    monthly_scores = scores_scaled.resample("ME").last().reindex(monthly_prices.index).dropna(how="all")
    monthly_scores = smooth_scores(monthly_scores, ema=ema)
    cols = monthly_scores.columns.intersection(monthly_prices.columns)
    monthly_prices = monthly_prices[cols]
    monthly_scores = monthly_scores[cols]
    w = hysteresis_picks(monthly_scores.shift(1), k=top_k, margin=margin)
    rets = monthly_prices.pct_change().fillna(0.0)
    strat_ret = (w * rets).sum(axis=1)
    nav = (1 + strat_ret).cumprod()
    return nav, strat_ret, w

# ================== UI ==================
st.set_page_config(page_title="Macro → Sector Signals (Plus)", layout="wide")
st.markdown(
    "<div style='display:flex;align-items:center;gap:.6rem'>"
    "<span style='font-size:1.6rem'>🎯</span>"
    "<div><div style='font-size:1.2rem;font-weight:700'>Macro → Sector Signals</div>"
    "<div style='opacity:.75;margin-top:-2px'>Beta – macro-to-sector rotation dashboard</div></div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

with st.sidebar:
    st.header("Settings")

    # 🎛️ Core settings
    with st.expander("🎛️ Core settings", expanded=True):
        try:
            from secrets_local import get_key  # type: ignore
            fred_default = get_key("FRED_API_KEY") or os.environ.get("FRED_API_KEY", "")
        except Exception:
            fred_default = os.environ.get("FRED_API_KEY", "")
        fred_key = st.text_input("FRED API key (optional)", value=fred_default)
        start_date = st.date_input("Start date", value=datetime(2010,1,1))
        preset = st.selectbox("Preset", ["Default","Hawkish Fed","Reflation","Risk-off"])
        top_k     = st.slider("Top-K sectors", 1, 6, 3)
        ema_span  = st.slider("Smoothing EMA (months)", 1, 12, 3)
        hyst_mag  = st.slider("Hysteresis margin", 0.0, 0.5, 0.15, 0.05)

    # ⚡ Sensitivities
    with st.expander("⚡ Sensitivities", expanded=False):
        include_gdx = st.checkbox("Include Gold Miners (GDX)", value=True)
        sens_scale = st.slider("Global sensitivity scale", 0.5, 3.0, 1.0, 0.1)
        symbols = SECTOR_ETFS + (EXTRA_ETFS if include_gdx else [])
        editable = DEFAULT_SENS.loc[symbols, ["Inflation","Growth","Rates","USD","Jobs"]].copy()
        with st.expander("Edit matrix"):
            for th in editable.columns:
                for sec in editable.index:
                    editable.loc[sec, th] = st.number_input(f"{sec} ↔ {th}",
                                                            value=float(editable.loc[sec, th]),
                                                            step=0.1, key=f"{sec}_{th}")
        SENS_USED = editable * sens_scale

    # 🌍 Risk overlays
    with st.expander("🌍 Risk overlays", expanded=False):
        use_exo   = st.checkbox("Blend exogenous risk (VIX, GPR, GSCPI)", value=True)
        use_cad   = st.checkbox("Canada overlay (USDCAD)", value=True)
        use_tarif = st.checkbox("Tariff/War proxy (GDELT)", value=False)

    # 📅 Calendar options
    with st.expander("📅 Calendar options", expanded=False):
        te_client  = st.text_input("TradingEconomics client (user:token)", value=os.environ.get("TE_CLIENT","guest:guest"))
        show_cal = st.checkbox("Afficher le calendrier (US, ±1 jour)", value=True)

    # ⚠️ Data status (remplie après les fetchs)
    data_status_slot = st.empty()

    st.markdown("---")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# Presets (après lecture des contrôles)
params = {"win_infl": 24, "win_growth": 24, "win_jobs": 24, "win_rates": 24, "win_usd": 24}
if preset == "Hawkish Fed":
    params.update(win_infl=18, win_rates=18, win_usd=18)
elif preset == "Reflation":
    params.update(win_growth=12, win_jobs=12, win_infl=12)
elif preset == "Risk-off":
    params.update(win_rates=36, win_usd=36)

st.markdown("**Tip:** Adjust windows/presets & sensitivities; surprises ≈ rolling z-scores.**")

# ============= DATA =============
@tlog("fetch_macro")
@st.cache_data(show_spinner=True, ttl=12*3600)
def fetch_macro(fred_key, start):
    frames = []
    for sid in FRED_SERIES.keys():
        df = load_fred_series(sid, fred_key, start=start)
        if df is None or df.empty:
            continue
        frames.append(df)
        time.sleep(0.05)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, axis=1, join="outer").sort_index()
    all_df = all_df[all_df.index >= pd.to_datetime(start)]
    return all_df

macro = fetch_macro(fred_key, start_date)
set_status("FRED macro", not macro.empty, "ok" if not macro.empty else "vide")
if macro.empty:
    st.error("Unable to load macro data. Check your internet or FRED key.")
    st.stop()

themes = compute_theme_scores(macro, params)
if getattr(themes.index, "tz", None) is not None:
    themes.index = themes.index.tz_localize(None)

scores_scaled, scores_raw = sector_scores(themes, SENS_USED)

@tlog("fetch_prices")
@st.cache_data(show_spinner=True, ttl=6*3600)
def fetch_prices(symbols, start):
    return get_multi_yf(symbols, start=str(start))

prices = fetch_prices(symbols, start_date)
if getattr(prices.index, "tz", None) is not None:
    prices.index = prices.index.tz_localize(None)
align_start = pd.to_datetime(themes.index.min())
prices = prices[prices.index >= align_start]
set_status("Prices (yfinance)", not prices.empty, "ok" if not prices.empty else "vide")

# ============= EXOGENOUS BLEND =============
scores_scaled_exo = scores_scaled.copy()
overlay_notes = []

if use_exo:
    gscpi = call_safely(fetch_gscpi, label="GSCPI")
    vix   = call_safely(fetch_vix_history, label="VIX")
    gpr   = call_safely(fetch_gpr, label="GPR")
    exo = []
    if gscpi is not None: exo.append(gscpi.rename("GSCPI").resample("ME").last())
    if vix   is not None: exo.append(vix.rename("VIX").resample("ME").last())
    if gpr   is not None: exo.append(gpr.rename("GPR").resample("ME").last())
    if exo:
        exo = pd.concat(exo, axis=1).reindex(scores_scaled_exo.index)
        exo_z = exo.apply(lambda s: (s - s.rolling(24).mean())/s.rolling(24).std())
        risk_penalty = pd.Series(0, index=scores_scaled_exo.index, dtype=float)
        if "VIX" in exo_z:  risk_penalty = risk_penalty.add((-0.15)*exo_z["VIX"].fillna(0), fill_value=0)
        if "GPR" in exo_z:  risk_penalty = risk_penalty.add((-0.15)*exo_z["GPR"].fillna(0), fill_value=0)
        if "GSCPI" in exo_z: risk_penalty = risk_penalty.add((-0.10)*exo_z["GSCPI"].fillna(0), fill_value=0)
        scores_scaled_exo = scores_scaled_exo.add(risk_penalty, axis=0).clip(-5,5)
        overlay_notes.append(f"Exogenous risk penalty applied: {risk_penalty.iloc[-1]:+.2f} (latest sum)")

if use_cad:
    cad = call_safely(fetch_boc_fx, series="FXUSDCAD", label="USDCAD")
    if cad is not None and not cad.empty:
        cad_m = cad.resample("ME").last().reindex(scores_scaled_exo.index)
        cad_z = (cad_m - cad_m.rolling(24).mean())/cad_m.rolling(24).std()
        adj = (-0.10)*cad_z.squeeze().fillna(0)  # USD↑(CAD↓) → headwind commodities
        scores_scaled_exo = scores_scaled_exo.add(adj, axis=0).clip(-5,5)
        overlay_notes.append(f"CAD overlay: {adj.iloc[-1]:+.2f} (latest)")

if use_tarif:
    gd = call_safely(fetch_gdelt_events, days=60, label="GDELT")
    if gd is not None and len(gd)>0:
        tw = gd.resample("ME").sum()
        tw_z = (tw - tw.rolling(12).mean())/tw.rolling(12).std()
        pen = (-0.10)*tw_z.reindex(scores_scaled_exo.index).fillna(0)
        scores_scaled_exo = scores_scaled_exo.add(pen, axis=0).clip(-5,5)
        overlay_notes.append(f"Tariff/War proxy: {pen.iloc[-1]:+.2f} (latest)")

# ============= LAYOUT =============

# ---- Calendrier macro (hier/ajd/demain) ----
if show_cal:
    st.subheader("📅 Calendrier macro US (hier / aujourd’hui / demain)")
    d_today = today_ny()
    d1 = (d_today - timedelta(days=1)).strftime("%Y-%m-%d")
    d2 = (d_today + timedelta(days=1)).strftime("%Y-%m-%d")
    cal = fetch_te_calendar(d1, d2, client=te_client)
    set_status("Calendar", not cal.empty, "ok" if not cal.empty else "vide")

    if cal.empty:
        st.info("Aucun événement sur la fenêtre ou API indisponible.")
    else:
        cal["Heure (NY)"] = cal["ts_ny"].dt.strftime("%a %m-%d %H:%M")
        cal_view = (cal[["Heure (NY)","Event","Category","Reference","Actual","Forecast","Previous","Importance"]]
                    .sort_values(["Importance","Heure (NY)"], ascending=[False, True]))
        st.dataframe(cal_view, width='stretch', height=260)

        # comparer tz-aware Timestamp à ts_ny (tz-aware)
        d_today_start = pd.Timestamp(d_today, tz=US_TZ) if US_TZ is not None else pd.Timestamp(d_today)
        top = cal[(cal["imp"] >= 2) & (cal["ts_ny"] >= d_today_start)].head(6)
        if not top.empty:
            st.markdown("**À surveiller (H/M importance)**")
            for _, r in top.iterrows():
                badge = "🔴" if r["Importance"]=="High" else "🟠"
                st.write(f"{badge} {r['ts_ny']:%a %H:%M} — {r['Event']} "
                         f"(Act: {r['Actual']}, Cons: {r['Forecast']}, Prev: {r['Previous']})")

# ---- Injecte l'encart Data status dans la sidebar ----
with data_status_slot.container():
    st.subheader("⚠️ Data status")
    if not DATA_STATUS:
        st.caption("Aucun chargement externe enregistré.")
    else:
        rows = []
        for name, info in DATA_STATUS.items():
            emoji = "✅" if info["ok"] else "❌"
            detail = f" — {info['detail']}" if info.get("detail") else ""
            rows.append(f"{emoji} **{name}**{detail}")
        st.markdown("\n".join(f"- {r}" for r in rows))

# ---- Panneau Logs & Diagnostics ----
with st.expander("🧾 Logs & Diagnostics", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Summary", "Coverage", "Raw logs"])
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FRED rows", macro.shape[0] if not macro.empty else 0)
        c2.metric("Themes rows", themes.shape[0] if not themes.empty else 0)
        c3.metric("Prices rows", prices.shape[0] if not prices.empty else 0)
        c4.metric("ETFs", prices.shape[1] if not prices.empty else 0)

        prof = []
        prof.append(profile_df(macro, "FRED macro"))
        prof.append(profile_df(themes, "Themes"))
        prof.append(profile_df(prices, "Prices"))
        try:
            prof.append(profile_df(exo, "Exogenous factors"))  # peut ne pas exister
        except Exception:
            pass
        st.markdown("**Profiles**")
        st.dataframe(pd.DataFrame(prof), width='stretch', height=220)

        if not macro.empty:
            st.markdown("**FRED last non-null per series**")
            last_valid = {c: (macro[c].last_valid_index() if macro[c].notna().any() else None) for c in macro.columns}
            st.dataframe(pd.DataFrame.from_dict(last_valid, orient="index", columns=["last_valid"]).sort_index(),
                         width='stretch', height=220)

    with tab2:
        if not macro.empty:
            fred_cov = (macro.notna().sum() / macro.shape[0] * 100).round(1).rename("Coverage %").to_frame()
            st.markdown("**FRED coverage (non-null %) by series**")
            st.dataframe(fred_cov.sort_values("Coverage %", ascending=False),
                         width='stretch', height=260)
        if not themes.empty:
            st.markdown("**Themes last row**")
            st.write(themes.tail(1))
            theme_cov = (themes.notna().sum() / max(1, themes.shape[0]) * 100).round(1).rename("Coverage %").to_frame()
            st.dataframe(theme_cov, width='stretch', height=180)

    with tab3:
        logs = "\n".join(st.session_state.logbuf[-1000:])
        st.code(logs or "No logs yet.", language="text")
        st.download_button(
            "⬇️ Download full logs",
            data="\n".join(st.session_state.logbuf),
            file_name=f"macro_sector_logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ---- Theme Gauges en grille 2x3 ----
st.subheader("Theme Gauges (latest)")
latest = themes.dropna().iloc[-1] if not themes.dropna().empty else pd.Series(dtype=float)
order = ["Growth","Inflation","Rates","USD","Jobs"]
color_map = {
    "Growth":    "#22c55e",
    "Inflation": "#ef4444",
    "Rates":     "#3b82f6",
    "USD":       "#8b5cf6",
    "Jobs":      "#f59e0b",
}
rows = [order[:3], order[3:]]
for row in rows:
    cols = st.columns(len(row))
    for c, name in zip(cols, row):
        with c:
            val = float(latest.get(name, 0.0))
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=val*100, delta={'reference': 0},
                title={'text': name},
                gauge={'axis': {'range': [-100, 100]},
                       'bar': {'color': color_map.get(name, "#10b981")}}
            ))
            fig.update_layout(height=160, margin=dict(l=8,r=8,t=30,b=8))
            st.plotly_chart(fig, width='stretch')

# ---- Tabs pour gros graphiques ----
tab_scores, tab_backtest, tab_heatmap = st.tabs(["📊 Scores", "📈 Backtest", "🔥 Heatmap"])

with tab_scores:
    st.subheader("Sector Scores (last 12 months, scaled + overlays)")
    last12 = scores_scaled_exo.tail(12)
    fig = go.Figure()
    for c in last12.columns:
        diffs = last12[c].diff().fillna(0).values
        fig.add_trace(go.Scatter(
            x=last12.index, y=last12[c], name=c,
            customdata=diffs,
            hovertemplate="%{y:.2f} (Δ m/m: %{customdata:+.2f})<extra>"+c+"</extra>"
        ))
    fig.update_layout(height=340, yaxis_title="Score [-5..5]")
    st.plotly_chart(fig, width='stretch')

with tab_backtest:
    st.subheader("Backtest – Monthly Top-K (EMA + Hysteresis)")
    nav, strat_ret, weights = backtest_rotation(
        prices, scores_scaled_exo, top_k=top_k, ema=ema_span, margin=hyst_mag
    )
    use_log = st.checkbox("Log scale", value=False, key="log_nav")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=nav.index, y=nav, name="Strategy NAV"))
    try:
        spy = get_asset_price("SPY", start=str(start_date))
        spy_nav = (1 + spy.pct_change(fill_method=None).fillna(0)).cumprod()
        fig2.add_trace(go.Scatter(x=spy_nav.index, y=spy_nav/spy_nav.iloc[0], name="SPY (normalized)"))
    except Exception as e:
        log_warn(f"SPY fetch failed: {e}")
    try:
        ew = (1 + prices.pct_change().mean(axis=1)).cumprod()
        fig2.add_trace(go.Scatter(x=ew.index, y=ew/ew.iloc[0], name="Equal-Weight Sectors"))
    except Exception as e:
        log_warn(f"EW compute failed: {e}")

    # USREC shading (récession) sur le backtest
    try:
        usrec = macro.get("USREC")
        if usrec is None or (hasattr(usrec, "empty") and usrec.empty):
            usrec = load_fred_series("USREC", fred_key)
        usrec_m = usrec.resample("ME").last().reindex(nav.index).fillna(0)
        in_rec = False; rec_start = None
        for i in range(len(usrec_m)):
            flag = usrec_m.iloc[i,0] if isinstance(usrec_m, pd.DataFrame) else usrec_m.iloc[i]
            if not in_rec and flag == 1:
                in_rec = True; rec_start = usrec_m.index[i]
            if in_rec and flag == 0:
                in_rec = False
                fig2.add_vrect(x0=rec_start, x1=usrec_m.index[i], fillcolor="grey", opacity=0.09, line_width=0)
        if in_rec:
            fig2.add_vrect(x0=rec_start, x1=usrec_m.index[-1], fillcolor="grey", opacity=0.09, line_width=0)
    except Exception as e:
        log_warn(f"USREC shading skipped: {e}")

    fig2.update_layout(height=420, yaxis_title="NAV (normalized)", yaxis_type=("log" if use_log else "linear"))
    st.plotly_chart(fig2, width='stretch')

with tab_heatmap:
    st.subheader("Heatmap des scores (12 derniers mois)")
    hm = scores_scaled_exo.tail(12)
    if not hm.empty:
        fig = go.Figure(data=go.Heatmap(
            z=hm.values, x=hm.columns, y=[d.strftime("%Y-%m") for d in hm.index],
            zmin=-5, zmax=5, colorscale="RdBu",
            colorbar=dict(orientation="h", y=-0.2, title="Score")
        ))
        fig.update_layout(height=360, margin=dict(b=60))
        st.plotly_chart(fig, width='stretch')

# ---- Current Ranking enrichi ----
st.subheader("Current Ranking")
if not scores_scaled_exo.dropna().empty:
    idxs = scores_scaled_exo.dropna().index
    idxs_sorted = idxs.sort_values()
    last_idx = idxs_sorted[-1]
    prev_idx = idxs_sorted[-2] if len(idxs_sorted) > 1 else None

    cur = scores_scaled_exo.loc[last_idx].sort_values(ascending=False).to_frame("score")
    if prev_idx is not None:
        prev = scores_scaled_exo.loc[prev_idx].reindex(cur.index)
        delta = (cur["score"] - prev).rename("Δ m/m")
    else:
        delta = pd.Series(0, index=cur.index, name="Δ m/m")
    view = pd.concat([cur, delta], axis=1)
    view["dir"] = np.where(view["Δ m/m"]>=0, "↑", "↓")

    idx_top = view["score"].idxmax()
    def highlight_top(row):
        return ['font-weight:bold;background-color:rgba(16,185,129,.15)' if row.name==idx_top else '' for _ in row]

    styled = (view.style
              .format({"score":"{:.2f}", "Δ m/m":"{:+.2f}"})
              .bar(subset=["score"], align="mid", color='#22c55e')
              .apply(highlight_top, axis=1))

    st.dataframe(styled, width='stretch')
else:
    st.info("Scores indisponibles pour l’instant.")

# Attribution raw (last 6 months)
st.subheader("Score attribution by theme (raw before scaling, last 6 months)")
st.dataframe(scores_raw.tail(6).style.format("{:+.2f}"), width='stretch')

# ---- Corrélations roulantes ----
with st.expander("Corrélations roulantes (6m) avec thèmes"):
    pxm = prices.resample("ME").last()
    candidates = [s for s in (SECTOR_ETFS + (EXTRA_ETFS if include_gdx else [])) if s in pxm.columns]
    if candidates:
        pick_sec = st.selectbox("Secteur :", candidates, index=min(4, len(candidates)-1))
        r = pxm[pick_sec].pct_change()
        dfc = pd.concat([r.rename("ret"), themes], axis=1).dropna()
        if not dfc.empty:
            roll = dfc.rolling(6).corr().xs("ret", level=1, drop_level=False)
            roll = roll.drop(columns=["ret"], errors="ignore")
            fig = go.Figure()
            for c in ["Growth","Inflation","Rates","USD","Jobs"]:
                if c in roll.columns:
                    fig.add_trace(go.Scatter(x=roll.index, y=roll[c], name=c))
            fig.update_layout(height=320, yaxis_title="corr (6m)")
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Pas de colonnes de prix disponibles pour calculer les corrélations.")

# Screener
with st.expander("Screener actions des secteurs sélectionnés (si holdings disponibles)"):
    chosen = (cur.index[:top_k].tolist() if 'cur' in locals() and not cur.empty
              else (SECTOR_ETFS[:top_k]))
    st.caption("Secteurs sélectionnés : " + ", ".join(chosen))
    for sec in chosen:
        try:
            t = yf.Ticker(sec)
            holds = getattr(t, "fund_holdings", None)
            if holds is None or len(holds)==0:
                st.write(f"Holdings indisponibles pour {sec}.")
                st.markdown(f"🔍 Voir la fiche ETF : [Yahoo Finance]({'https://finance.yahoo.com/quote/' + sec})")
                # Fallback: momentum simple sur l’ETF lui-même (3m/6m/vol)
                try:
                    px = get_multi_yf([sec], start="2018-01-01")
                    if not px.empty:
                        m3 = px.pct_change(63).iloc[-1].rename("m3")
                        m6 = px.pct_change(126).iloc[-1].rename("m6")
                        vol = px.pct_change().rolling(63).std().iloc[-1].rename("vol")
                        df_fallback = pd.concat([m3, m6, vol], axis=1)
                        df_fallback["score"] = 0.6*df_fallback["m6"] + 0.4*df_fallback["m3"] - 0.2*df_fallback["vol"]
                        st.dataframe(df_fallback.round(3), width='stretch')
                except Exception as e:
                    log_warn(f"Screener fallback failed for {sec}: {e}")
                continue
            if isinstance(holds, dict) and "symbol" in holds:
                tickers = pd.Series(holds["symbol"]).dropna().tolist()[:15]
            elif isinstance(holds, pd.DataFrame) and "symbol" in holds.columns:
                tickers = holds["symbol"].dropna().tolist()[:15]
            else:
                st.write(f"Format des holdings non supporté pour {sec}.")
                continue
            px = get_multi_yf(tickers, start="2018-01-01")
            if px.empty:
                st.write(f"Pas de prix pour holdings de {sec}.")
                continue
            m3 = px.pct_change(63).iloc[-1]
            m6 = px.pct_change(126).iloc[-1]
            vol = px.pct_change().rolling(63).std().iloc[-1]
            df = pd.DataFrame({"m3":m3, "m6":m6, "vol":vol})
            df["score"]=0.6*df["m6"]+0.4*df["m3"]-0.2*df["vol"]
            st.write(f"Top holdings {sec}")
            st.dataframe(df.sort_values("score", ascending=False).round(3), width='stretch')
        except Exception as e:
            st.write(f"Screener indisponible pour {sec}: {e}")

# Notes overlay
overlay_txt = " | ".join(overlay_notes) if overlay_notes else "No exogenous overlays applied."
st.info(overlay_txt)

st.caption("Personal tool. Not investment advice. Data: FRED, yfinance, CBOE, NY Fed GSCPI, BoC FX, GPR, TradingEconomics (calendar), (opt) EIA/BLS/GDELT).")

if __name__ == "__main__":
    # Exécute l'interface complète quand appelé directement
    # Logique UI existante encapsulée
    pass
