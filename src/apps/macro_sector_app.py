# src/apps/macro_sector_app.py
# ======================================================================================
# Tableau de bord macro → secteurs (version lisible, sans abréviations, tout visible)
# - AUCUNE action UI n’est exécutée à l’import : tout est encapsulé dans render_macro()
# - Libellés développés (pas d’abréviations opaques)
# - Pas d’expanders : tout est affiché afin de repérer facilement les problèmes
# ======================================================================================

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
from utils import get_st, warn_once
from core_runtime import (SESSION, df_fingerprint, write_entry, with_span,
                          log, get_trace_id, new_trace_id, set_trace_id, get_dataset_log_latest)

st = get_st()

# ---------- Éviter les avertissements courants ----------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ================== LOGGING (en mémoire pour l’écran) ==================
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
                log_info(f"{name or fn.__name__} OK en {dur:.0f} ms, shape={shape}")
                return res
            except Exception as e:
                dur = (time.time() - t0) * 1000
                log_error(f"{name or fn.__name__} ERREUR en {dur:.0f} ms: {e}")
                raise
        return inner
    return deco

def profile_df(df: Optional[Union[pd.DataFrame, pd.Series]], name: str):
    out = {"Nom": name, "Type": type(df).__name__, "Lignes": 0, "Colonnes": 0,
           "Début": None, "Fin": None, "Fréquence inférée": None, "Valeurs manquantes (médiane, %)": None}
    if df is None:
        return out
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    out["Lignes"], out["Colonnes"] = df.shape
    try:
        out["Début"] = str(df.index.min())
        out["Fin"]   = str(df.index.max())
    except Exception:
        pass
    try:
        out["Fréquence inférée"] = pd.infer_freq(df.index)
    except Exception:
        out["Fréquence inférée"] = None
    try:
        na_pct = (1 - df.notna().sum()/len(df)) * 100
        out["Valeurs manquantes (médiane, %)"] = float(na_pct.median().round(2))
    except Exception:
        pass
    return out

# ================== SÉRIES MACRO (FRED) ==================
FRED_SERIES = {
    # Inflation / anticipations
    "CPIAUCSL": "Indice des prix à la consommation (niveau, base 1982-84=100)",
    "T10YIE":   "Inflation implicite à 10 ans (breakeven, attentes de marché)",
    # Activité
    "INDPRO":   "Indice de production industrielle",
    "GDPC1":    "Produit intérieur brut réel (trimestriel)",
    # Emploi
    "UNRATE":   "Taux de chômage",
    "PAYEMS":   "Emplois non agricoles (niveau)",
    # Taux d’intérêt & courbe
    "DGS10":    "Taux du Trésor US à 10 ans",
    "DGS2":     "Taux du Trésor US à 2 ans",
    # Dollar
    "DTWEXBGS": "Indice large du dollar américain (taux de change pondéré)",
    # Conditions financières / crédit
    "NFCI":       "Indice des conditions financières (Fed de Chicago)",
    "BAMLC0A0CM": "Écart de crédit obligations d’entreprises (investment grade)",
    "BAMLH0A0HYM2": "Écart de crédit obligations à haut rendement",
    # Récessions (indicateur binaire)
    "USREC":    "Indicateur de récession (NBER, 1 = en récession)",
}

SECTOR_ETFS = ["XLB","XLE","XLF","XLV","XLK","XLI","XLY","XLP","XLRE","XLU"]
EXTRA_ETFS  = ["GDX"]  # Mines d'or (optionnel)

# Sensibilités par thème (positif = thème favorable au secteur)
DEFAULT_SENS = pd.DataFrame({
    "Inflation":{"XLK":-1,"XLF":1,"XLE":2,"XLB":2,"XLV":0,"XLY":-1,"XLP":0,"XLI":1,"XLRE":-1,"XLU":1,"GDX":2},
    "Croissance":   {"XLK": 2,"XLF":1,"XLE":0,"XLB":1,"XLV":0,"XLY": 2,"XLP":0,"XLI":1,"XLRE": 0,"XLU":-1,"GDX":0},
    "Taux d’intérêt":    {"XLK":-2,"XLF":2,"XLE":1,"XLB":0,"XLV":0,"XLY":-1,"XLP":0,"XLI":0,"XLRE":-1,"XLU":1,"GDX":1},
    "Dollar américain":      {"XLK":-1,"XLF":0,"XLE":-1,"XLB":-1,"XLV":0,"XLY":0,"XLP":0,"XLI":-1,"XLRE":-1,"XLU":0,"GDX":-1},
    "Marché du travail":     {"XLK": 1,"XLF":1,"XLE":0,"XLB":1,"XLV":0,"XLY": 1,"XLI":1,"XLRE": 0,"XLU":-1,"GDX":0},
}).fillna(0)

# ================== HELPERS ANALYTIQUES ==================
def zscore(series, win=24):
    s = series.dropna()
    if len(s) < win + 2:
        return pd.Series(index=series.index, dtype=float)
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return ((s - mu) / (sd.replace(0, np.nan))).reindex(series.index)

def normalize_fred_key(k: Optional[str]) -> Optional[str]:
    """Valide une clé FRED (32 caractères alphanumériques en minuscules)."""
    if not k or not isinstance(k, str):
        return None
    k = k.strip()
    if len(k) == 32 and k.isalnum() and k == k.lower():
        return k
    return None

def load_fred_series(series_id: str, fred_key: Optional[str] = None, start: Optional[str] = None) -> pd.DataFrame:
    """
    Charge une série FRED robuste :
      1) API JSON officielle (si clé valide)
      2) Fallback CSV fredgraph.csv (sans clé)
    """
    vkey = normalize_fred_key(fred_key)

    # 1) API JSON
    if vkey:
        try:
            params = {"series_id": series_id, "api_key": vkey, "file_type": "json"}
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
            log_warn(f"API FRED JSON en échec pour {series_id}: {e}; essai CSV")

    # 2) CSV fredgraph
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        text = resp.text.strip()
        if text[:1] in ("<", "{"):
            raise RuntimeError("FRED a renvoyé un contenu non-CSV")
        df = pd.read_csv(io.StringIO(text))
        # colonne date
        date_col = None
        for cand in ("DATE", "date", "observation_date"):
            if cand in df.columns:
                date_col = cand; break
        if date_col is None:
            raise KeyError("CSV sans colonne DATE")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        # colonne valeur
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
                raise KeyError(f"Impossible d’identifier la colonne valeur pour {series_id}")
        df[val_col] = pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce")
        out = df[[val_col]].rename(columns={val_col: series_id}).sort_index()
        if start:
            out = out[out.index >= pd.to_datetime(start)]
        return out
    except Exception as e:
        log_warn(f"FRED CSV en échec pour {series_id}: {e}")
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
            log_warn(f"yfinance en échec pour {s}: {e}")
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

# ---------- Indicateur d’état de données (affiché en sidebar) ----------
DATA_STATUS: "OrderedDict[str, dict]" = OrderedDict()

def set_status(name: str, ok: bool, detail: str = ""):
    DATA_STATUS[name] = {"ok": bool(ok), "detail": str(detail or "")}

def call_safely(_fn, *args, label: str = None, **kwargs):
    """Exécute une fonction; enregistre OK/échec dans DATA_STATUS (affiché en UI)."""
    try:
        out = _fn(*args, **kwargs)
        if label:
            ok = out is not None and (not hasattr(out, "empty") or not getattr(out, "empty", False))
            set_status(label, ok, "ok" if ok else "vide")
        return out
    except Exception as e:
        if label:
            set_status(label, False, str(e))
        log_warn(f"{_fn.__name__} en échec ({label or ''}): {e}")
        return None

def render_sources_state():
    df = get_dataset_log_latest()
    st.subheader("État des sources (dernière collecte)")
    st.dataframe(df, width='stretch')

def badge(available: bool, name: str, fallback: str = ""):
    """Affiche un badge indiquant l'état d'une source de données"""
    if available:
        st.success(f"✅ {name}")
    else:
        st.error(f"❌ {name} - {fallback}" if fallback else f"❌ {name}")

# ================== CONNECTEURS EXTERNES ==================
log = logging.getLogger("macroapp")

@with_span("fetch_gscpi")
def fetch_gscpi():
    """
    Charge GSCPI depuis plusieurs miroirs.
    Gère les lignes d'en-tête/metadata et CSV non standard.
    """
    import io
    import re
    URLS = [
        # miroir Fed (HTML/CSV), attention: peut servir du CSV "sale"
        "https://www.newyorkfed.org/medialibrary/research/gscpi/files/gscpi_data.csv",
        # miroir communautaire
        "https://raw.githubusercontent.com/QUANTAXIS/QAData/main/gscpi_data.csv",
        # backup quantnomad (si dispo)
        "https://raw.githubusercontent.com/QuantNomad/public-datasets/main/gscpi/gscpi_data.csv",
    ]
    last_err = None
    for url in URLS:
        try:
            r = SESSION.get(url, timeout=10)
            r.raise_for_status()
            raw = r.text

            # Nettoyage basique: enlever BOM, lignes vides, heading non-CSV
            raw = raw.lstrip("\ufeff")
            # Certaines versions contiennent des lignes de commentaire/texte
            lines = [ln for ln in raw.splitlines() if ln.strip()]
            # Garder à partir de la 1ère ligne qui contient une virgule ou un ';'
            start = 0
            for i, ln in enumerate(lines):
                if ("," in ln) or (";" in ln):
                    start = i
                    break
            cleaned = "\n".join(lines[start:])

            # Essais de parsing: coma puis point-virgule
            for sep in (",", ";"):
                try:
                    df = pd.read_csv(io.StringIO(cleaned), sep=sep)
                    break
                except Exception:
                    df = None
            if df is None or df.empty:
                raise ValueError("CSV vide / illisible")

            # Normaliser colonnes
            cols_l = {c.lower(): c for c in df.columns}
            dcol = cols_l.get("date")
            if not dcol:
                # certaines variantes utilisent 'observation_date'
                dcol = cols_l.get("observation_date")
            if not dcol:
                raise ValueError("colonne date manquante")

            gcol = None
            for c in df.columns:
                if re.sub(r"[^a-z]", "", c.lower()) in ("gscpi", "global_supply_chain_pressureindex"):
                    gcol = c
                    break
            if not gcol:
                # parfois colonne 1
                gcol = df.columns[1] if len(df.columns) > 1 else None
            if not gcol:
                raise ValueError("colonne GSCPI manquante")

            df = df[[dcol, gcol]].rename(columns={dcol: "date", gcol: "GSCPI"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            df["GSCPI"] = pd.to_numeric(df["GSCPI"], errors="coerce")
            df = df.dropna(subset=["GSCPI"])

            meta = df_fingerprint(df)
            write_entry("GSCPI", url, "ok", meta, "v1")
            log.info("gscpi_ok", extra={"rows": meta["rows"], "min_date": meta["min_date"], "max_date": meta["max_date"]})
            return df

        except Exception as e:
            last_err = str(e)
            log.warning("gscpi_try_fail", extra={"url": url, "err": last_err})
            continue

    write_entry("GSCPI", ";".join(URLS), "fail", df_fingerprint(pd.DataFrame()), "v1")
    log.error("gscpi_fail", extra={"err": last_err})
    return None

@tlog("fetch_vix_history")
def fetch_vix_history():
    """Indice de volatilité CBOE (VIX) quotidien → pd.Series(name='VIX')."""
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    df = pd.read_csv(url, parse_dates=["DATE"])
    return df.set_index("DATE")["CLOSE"].rename("VIX")

@with_span("fetch_gpr")
def fetch_gpr():
    """
    Indice de risque géopolitique (Iacoviello).
    Plusieurs miroirs + arrêt propre si tous échouent.
    """
    import io
    URLS = [
        "https://www2.bc.edu/matteo-iacoviello/gpr_files/GPRD.csv",
        "https://www.matteoiacoviello.com/gpr_files/GPRD.csv",
        "https://raw.githubusercontent.com/QuantNomad/public-datasets/main/gpr/GPRD.csv",
        "https://raw.githubusercontent.com/pmorissette/gpr-data/main/GPRD.csv",
    ]
    last_err = None
    for url in URLS:
        try:
            r = SESSION.get(url, timeout=10)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # normaliser
            dcol = next((c for c in df.columns if c.lower() in ("date","month")), None)
            if not dcol:
                raise ValueError("GPR: colonne date/mois manquante")
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
            gcol = "GPR" if "GPR" in df.columns else next((c for c in df.columns if c.upper()=="GPR"), None)
            if gcol:
                df = df[[gcol]].astype(float).rename(columns={gcol:"GPR"})
            else:
                # certaines versions ont 'gpr' en lower
                gcol = next((c for c in df.columns if c.lower()=="gpr"), None)
                if gcol:
                    df = df[[gcol]].astype(float).rename(columns={gcol:"GPR"})
            if df.empty:
                raise ValueError("GPR vide")
            meta = df_fingerprint(df)
            write_entry("GPR", url, "ok", meta, "v1")
            log.info("gpr_ok", extra={"rows": meta["rows"], "min_date": meta["min_date"], "max_date": meta["max_date"]})
            return df
        except Exception as e:
            last_err = str(e)
            log.warning("gpr_try_fail", extra={"url": url, "err": last_err})
            continue

    write_entry("GPR", ";".join(URLS), "fail", df_fingerprint(pd.DataFrame()), "v1")
    log.error("gpr_fail", extra={"err": last_err})
    return None

@tlog("fetch_boc_fx")
def fetch_boc_fx(series="FXUSDCAD"):
    """Taux de change Banque du Canada (USD/CAD)."""
    url = f"https://www.bankofcanada.ca/valet/observations/{series}?start_date=2010-01-01"
    r = requests.get(url, timeout=30); r.raise_for_status()
    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df["d"] = pd.to_datetime(df["d"])
    df[series] = pd.to_numeric(df[series], errors="coerce")
    return df.set_index("d")[series]

def fetch_eia(series_id, api_key):
    """Séries EIA (énergie)."""
    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    series = r.json()["series"][0]["data"]
    df = pd.DataFrame(series, columns=["date","value"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").fillna(
                 pd.to_datetime(df["date"], format="%Y%m", errors="coerce"))
    return df.set_index("date")["value"].astype(float).sort_index()

def fetch_bls(series_ids, api_key=None, start_year=2010):
    """Bureau of Labor Statistics (ex: séries d’emploi, salaires…)."""
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
    """Proxy de chocs ‘tarifs/guerre’ basé sur le volume d’événements GDELT."""
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

# ===== Fuseau horaire New York pour le calendrier =====
try:
    import pytz
    US_TZ = pytz.timezone("America/New_York")
except Exception:
    US_TZ = None

def today_ny():
    if US_TZ:
        return datetime.now(US_TZ).date()
    return datetime.now(timezone.utc).date()

@tlog("fetch_te_calendar")
@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_te_calendar(d1: str, d2: str, country="United States", client="guest:guest"):
    """Calendrier des publications économiques (TradingEconomics)"""
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

# ================== MODÈLES / SCORES ==================
def compute_theme_scores(fred_df, params):
    """Calcule des scores mensuels par grands thèmes macro, normalisés [-1..1]."""
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
    theme["Inflation"]           = squash(0.6*z_infl + 0.4*z_infl_exp)
    theme["Croissance"]          = squash(0.7*z_growth + 0.3*z_payrolls)
    theme["Marché du travail"]   = squash(z_jobs)
    theme["Taux d’intérêt"]      = squash(0.7*z_rates - 0.3*z_curve)
    theme["Dollar américain"]    = squash(z_usd)
    return theme.fillna(0)

def sector_scores(theme_df, sensitivities):
    """Score des secteurs = combinaison linéaire (thèmes × sensibilités), puis mise à l’échelle robuste [-5..5]."""
    raw = theme_df.dot(sensitivities.T)
    raw = raw.where(raw.notna().any(axis=1), other=0)
    scaled = raw.apply(robust_minmax, axis=1)
    return scaled, raw

def backtest_rotation(prices, scores_scaled, top_k=3, ema=3, margin=0.15):
    """Stratégie mensuelle : prendre les K meilleurs secteurs (avec lissage et hystérésis)."""
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

# ======================================================================================
# RENDU PRINCIPAL (TOUT est exécuté ici, rien à l’import)
# ======================================================================================
def render_macro():
    # ---- Trace ID visible UI
    if "trace_id" not in st.session_state or not st.session_state["trace_id"]:
        st.session_state["trace_id"] = new_trace_id()
    else:
        set_trace_id(st.session_state["trace_id"])
    st.caption(f"Trace ID: `{st.session_state['trace_id']}`")

    # En-tête
    st.header("Prévision macroéconomique et rotation sectorielle (affichage complet)")
    st.caption("Tous les éléments sont affichés sans abréviations ambiguës. Les intitulés précisent ce que montre chaque graphique ou tableau.")

    # -------------------- RÉGLAGES (toujours visibles) --------------------
    st.subheader("Réglages de l’analyse")
    colA, colB, colC = st.columns([1.2, 1, 1])
    with colA:
        # Clé FRED (si disponible en env / secrets)
        try:
            from secrets_local import get_key  # type: ignore
            fred_default = get_key("FRED_API_KEY") or os.environ.get("FRED_API_KEY", "")
        except Exception:
            fred_default = os.environ.get("FRED_API_KEY", "")
        fred_key = st.text_input("Clé API FRED (facultatif)", value=fred_default, help="Laissez vide si vous n’en avez pas. On basculera sur une méthode alternative.")
    with colB:
        start_date = st.date_input("Date de début des séries macro", value=datetime(2010,1,1))
    with colC:
        preset = st.selectbox("Profil d’analyse (fenêtres de calcul)", ["Standard","Banque centrale restrictive","Reflation","Aversion au risque"])

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k     = st.slider("Nombre de secteurs retenus simultanément", 1, 6, 3, help="K meilleurs secteurs retenus chaque mois.")
    with col2:
        ema_span  = st.slider("Lissage exponentiel (nombre de mois)", 1, 12, 3, help="Plus la valeur est élevée, plus les scores sont lissés.")
    with col3:
        hyst_mag  = st.slider("Marge d’hystérésis (stabilité de sélection)", 0.0, 0.5, 0.15, 0.05, help="Évite de changer trop souvent de secteurs proches du seuil.")

    st.markdown("---")

    # Sensibilités
    st.subheader("Sensibilités des secteurs aux thèmes macroéconomiques")
    include_gdx = st.checkbox("Inclure le secteur 'mines d’or' (ETF GDX)", value=True)
    sens_scale = st.slider("Échelle globale de sensibilité", 0.5, 3.0, 1.0, 0.1,
                           help="Ajuste l’intensité des sensibilités entre thèmes et secteurs.")
    symbols = SECTOR_ETFS + (EXTRA_ETFS if include_gdx else [])
    editable = DEFAULT_SENS.loc[symbols, ["Inflation","Croissance","Taux d’intérêt","Dollar américain","Marché du travail"]].copy()
    st.caption("Modifie si besoin la matrice de sensibilités (positif = thème favorable au secteur).")
    grid_cols = st.columns(len(editable.columns))
    for j, th in enumerate(editable.columns):
        with grid_cols[j]:
            st.markdown(f"**{th}**")
            for sec in editable.index:
                editable.loc[sec, th] = st.number_input(f"{sec} ↔ {th}", value=float(editable.loc[sec, th]), step=0.1, key=f"{sec}_{th}")
    SENS_USED = editable * sens_scale

    st.markdown("---")

    # Facteurs de risque externes
    st.subheader("Facteurs de risque externes à intégrer aux scores")
    colR1, colR2, colR3 = st.columns(3)
    with colR1:
        use_exo = st.checkbox("Ajuster par risques externes (volatilité, géopolitique, chaînes logistiques)", value=True,
                              help="Utilise VIX (volatilité), GPR (risque géopolitique), GSCPI (pressions d’approvisionnement).")
    with colR2:
        use_cad = st.checkbox("Ajuster par taux de change dollar US / dollar canadien (USD/CAD)", value=True,
                              help="Impact sur les matières premières.")
    with colR3:
        use_tarif = st.checkbox("Ajuster par proxy 'tarifs/guerre' (GDELT)", value=False,
                                help="Basé sur le volume d’événements GDELT.")

    # Calendrier macro
    st.subheader("Paramètre du calendrier des publications économiques")
    te_client  = st.text_input("Identifiants TradingEconomics (utilisateur:clé) — facultatif", value=os.environ.get("TE_CLIENT","guest:guest"))
    show_cal = st.checkbox("Afficher le calendrier économique des États-Unis (hier, aujourd’hui, demain)", value=True)

    st.markdown("---")

    # Bouton de rechargement
    if st.button("Recharger toutes les données"):
        st.cache_data.clear()
        st.rerun()

    # -------------------- PARAMÈTRES D’ANALYSE --------------------
    params = {"win_infl": 24, "win_growth": 24, "win_jobs": 24, "win_rates": 24, "win_usd": 24}
    if preset == "Banque centrale restrictive":
        params.update(win_infl=18, win_rates=18, win_usd=18)
    elif preset == "Reflation":
        params.update(win_growth=12, win_jobs=12, win_infl=12)
    elif preset == "Aversion au risque":
        params.update(win_rates=36, win_usd=36)

    st.caption("Les ‘surprises’ sont estimées via des z-scores roulants sur des fenêtres exprimées en mois.")

    # -------------------- CHARGEMENT DES DONNÉES --------------------
    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_fetch_macro(fred_key, start):
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

    macro = cached_fetch_macro(fred_key, start_date)
    set_status("Données macroéconomiques (FRED)", not macro.empty, "ok" if not macro.empty else "vide")
    if macro.empty:
        st.error("Impossible de charger les séries macroéconomiques. Vérifie ta connexion ou la clé FRED.")
        st.stop()

    themes = compute_theme_scores(macro, params)
    if getattr(themes.index, "tz", None) is not None:
        themes.index = themes.index.tz_localize(None)

    scores_scaled, scores_raw = sector_scores(themes, SENS_USED)

    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_fetch_prices(symbols, start):
        return get_multi_yf(symbols, start=str(start))

    prices = cached_fetch_prices(symbols, start_date)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    align_start = pd.to_datetime(themes.index.min())
    prices = prices[prices.index >= align_start]
    set_status("Prix des secteurs (Yahoo Finance)", not prices.empty, "ok" if not prices.empty else "vide")

    # -------------------- AJUSTEMENTS EXOGÈNES --------------------
    scores_scaled_exo = scores_scaled.copy()
    overlay_notes = []

    if use_exo:
        gscpi = call_safely(fetch_gscpi, label="Indice de pression des chaînes d’approvisionnement (GSCPI)")
        vix   = call_safely(fetch_vix_history, label="Indice de volatilité CBOE (VIX)")
        gpr   = call_safely(fetch_gpr, label="Indice de risque géopolitique (GPR)")
        exo = []
        if gscpi is not None: exo.append(gscpi.rename("GSCPI").resample("ME").last())
        if vix   is not None: exo.append(vix.rename("VIX").resample("ME").last())
        if gpr   is not None: exo.append(gpr.rename("GPR").resample("ME").last())
        if exo:
            exo = pd.concat(exo, axis=1).reindex(scores_scaled_exo.index)
            exo_z = exo.apply(lambda s: (s - s.rolling(24).mean())/s.rolling(24).std())
            risk_penalty = pd.Series(0, index=scores_scaled_exo.index, dtype=float)
            if "VIX" in exo_z:   risk_penalty = risk_penalty.add((-0.15)*exo_z["VIX"].fillna(0), fill_value=0)
            if "GPR" in exo_z:   risk_penalty = risk_penalty.add((-0.15)*exo_z["GPR"].fillna(0), fill_value=0)
            if "GSCPI" in exo_z: risk_penalty = risk_penalty.add((-0.10)*exo_z["GSCPI"].fillna(0), fill_value=0)
            scores_scaled_exo = scores_scaled_exo.add(risk_penalty, axis=0).clip(-5,5)
            try:
                overlay_notes.append(f"Ajustement ‘risques externes’ appliqué (somme la plus récente : {risk_penalty.iloc[-1]:+.2f})")
            except Exception:
                pass

    render_sources_state()

    if gscpi is None:
        st.error(f"❌ GSCPI indisponible — voir 'État des sources’. Trace ID: `{get_trace_id()}`")
    if gpr is None:
        st.warning(f"⚠️ GPR indisponible — trace `{get_trace_id()}`")

    if use_cad:
        cad = call_safely(fetch_boc_fx, series="FXUSDCAD",
                          label="Taux de change USD/CAD (Banque du Canada)")
        if cad is not None and not cad.empty:
            cad_m = cad.resample("ME").last().reindex(scores_scaled_exo.index)
            cad_z = (cad_m - cad_m.rolling(24).mean())/cad_m.rolling(24).std()
            adj = (-0.10)*cad_z.squeeze().fillna(0)  # USD↑(CAD↓) → vent contraire pour matières premières
            scores_scaled_exo = scores_scaled_exo.add(adj, axis=0).clip(-5,5)
            try:
                overlay_notes.append(f"Ajustement USD/CAD : {adj.iloc[-1]:+.2f} (valeur la plus récente)")
            except Exception:
                pass

    if use_tarif:
        gd = call_safely(fetch_gdelt_events, days=60, label="Proxy ‘tarifs/guerre’ (GDELT)")
        if gd is not None and len(gd)>0:
            tw = gd.resample("ME").sum()
            tw_z = (tw - tw.rolling(12).mean())/tw.rolling(12).std()
            pen = (-0.10)*tw_z.reindex(scores_scaled_exo.index).fillna(0)
            scores_scaled_exo = scores_scaled_exo.add(pen, axis=0).clip(-5,5)
            try:
                overlay_notes.append(f"Proxy ‘tarifs/guerre’ : {pen.iloc[-1]:+.2f} (valeur la plus récente)")
            except Exception:
                pass

    # -------------------- CALENDRIER DES PUBLICATIONS --------------------
    if show_cal:
        st.subheader("Calendrier économique des États-Unis (hier, aujourd’hui, demain)")
        d_today = today_ny()
        d1 = (d_today - timedelta(days=1)).strftime("%Y-%m-%d")
        d2 = (d_today + timedelta(days=1)).strftime("%Y-%m-%d")
        cal = fetch_te_calendar(d1, d2, client=te_client)
        set_status("Calendrier des publications (TradingEconomics)", not cal.empty, "ok" if not cal.empty else "vide")

        if cal.empty:
            st.info("Aucun événement sur la fenêtre choisie ou service temporairement indisponible.")
        else:
            cal["Heure (New York)"] = cal["ts_ny"].dt.strftime("%a %m-%d %H:%M")
            cal_view = (cal[["Heure (New York)","Event","Category","Reference","Actual","Forecast","Previous","Importance"]]
                        .sort_values(["Importance","Heure (New York)"], ascending=[False, True]))
            st.dataframe(cal_view, width='stretch', height=280)

            # Focus événements importants à venir aujourd’hui
            try:
                d_today_start = pd.Timestamp(d_today, tz=US_TZ) if US_TZ is not None else pd.Timestamp(d_today)
                top = cal[(cal["imp"] >= 2) & (cal["ts_ny"] >= d_today_start)].head(6)
                if not top.empty:
                    st.markdown("**À surveiller (importance moyenne/élevée, heures de New York)**")
                    for _, r in top.iterrows():
                        imp_badge = "🔴" if r["Importance"] == "High" else "🟠"
                        st.write(
                            f"{imp_badge} {r['ts_ny']:%a %H:%M} — {r['Event']} "
                            f"(Actuel : {r['Actual']}, Consensus : {r['Forecast']}, Précédent : {r['Previous']})"
                        )
            except Exception as e:
                log_warn(f"Top events cal échoué: {e}")

    st.markdown("---")

    # -------------------- ÉTAT DES SOURCES DE DONNÉES --------------------
    st.subheader("État des sources de données (chargements externes)")
    if not DATA_STATUS:
        st.caption("Aucun chargement externe enregistré pour l'instant.")
    else:
        rows = []
        for name, info in DATA_STATUS.items():
            emoji = "✅" if info["ok"] else "❌"
            detail = f" — {info['detail']}" if info.get("detail") else ""
            rows.append(f"{emoji} **{name}**{detail}")
        st.markdown("\n".join(f"- {r}" for r in rows))

    st.markdown("---")

    # Add source status badges
    badge(gscpi is not None and not gscpi.empty if 'gscpi' in locals() else False,
          "Indice de pression des chaînes d'approvisionnement (GSCPI)",
          "source indisponible, remplacé par placeholder" if gscpi is None or gscpi.empty else "")

    badge(gpr is not None and not gpr.empty if 'gpr' in locals() else False,
          "Indice de risque géopolitique (GPR)",
          "source indisponible, saute automatiquement" if gpr is None or gpr.empty else "")

    # -------------------- DIAGNOSTICS ET LOGS --------------------
    st.subheader("Diagnostics (couverture, profils) et journal d’exécution")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre de lignes — séries FRED", macro.shape[0] if not macro.empty else 0)
    c2.metric("Nombre de lignes — thèmes macro", themes.shape[0] if not themes.empty else 0)
    c3.metric("Nombre de lignes — prix des secteurs", prices.shape[0] if not prices.empty else 0)
    c4.metric("Nombre de secteurs analysés", prices.shape[1] if not prices.empty else 0)

    prof = []
    prof.append(profile_df(macro, "Séries FRED (macro)"))
    prof.append(profile_df(themes, "Thèmes macro calculés"))
    prof.append(profile_df(prices, "Prix des secteurs (yfinance)"))
    try:
        # ‘exo’ peut exister si use_exo True
        if "exo" in locals():
            prof.append(profile_df(exo, "Facteurs externes (VIX, GPR, GSCPI)"))
    except Exception:
        pass
    st.markdown("**Profils des tableaux de données**")
    st.dataframe(pd.DataFrame(prof), width='stretch', height=240)

    if not macro.empty:
        st.markdown("**Dernière date non-nulle par série FRED**")
        last_valid = {c: (macro[c].last_valid_index() if macro[c].notna().any() else None) for c in macro.columns}
        st.dataframe(pd.DataFrame.from_dict(last_valid, orient="index", columns=["Dernière date non nulle"]).sort_index(),
                     width='stretch', height=240)

    # Journal (dernier millier de lignes)
    st.markdown("**Journal des opérations (extraits récents)**")
    logs = "\n".join(st.session_state.logbuf[-1000:])
    st.code(logs or "Aucun log pour l’instant.", language="text")

    st.markdown("---")

    # -------------------- INDICATEURS DE THÈMES (dernière valeur) --------------------
    st.subheader("Indicateurs de thèmes macroéconomiques — dernière valeur (échelle −100 à +100)")
    latest = themes.dropna().iloc[-1] if not themes.dropna().empty else pd.Series(dtype=float)
    order = ["Croissance","Inflation","Taux d’intérêt","Dollar américain","Marché du travail"]
    color_map = {
        "Croissance":        "#22c55e",
        "Inflation":         "#ef4444",
        "Taux d’intérêt":    "#3b82f6",
        "Dollar américain":  "#8b5cf6",
        "Marché du travail": "#f59e0b",
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
                fig.update_layout(height=170, margin=dict(l=8,r=8,t=30,b=8))
                st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # -------------------- GRANDS GRAPHIQUES --------------------
    st.subheader("Scores des secteurs (12 derniers mois) — après normalisation et ajustements externes")
    last12 = scores_scaled_exo.tail(12)
    fig = go.Figure()
    for c in last12.columns:
        diffs = last12[c].diff().fillna(0).values
        fig.add_trace(go.Scatter(
            x=last12.index, y=last12[c], name=c,
            customdata=diffs,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Score normalisé: %{y:.2f}<br>Variation mensuelle: %{customdata:+.2f}<extra>"+c+"</extra>"
        ))
    fig.update_layout(height=360, yaxis_title="Score normalisé (de −5 à +5)")
    st.plotly_chart(fig, width='stretch')

    # Simulation historique (backtest)
    st.subheader("Simulation historique (backtest) — stratégie mensuelle Top-K avec lissage et hystérésis")
    nav, strat_ret, weights = backtest_rotation(
        prices, scores_scaled_exo, top_k=top_k, ema=ema_span, margin=hyst_mag
    )
    use_log = st.checkbox("Afficher l’échelle logarithmique pour la courbe de performance", value=False, key="log_nav")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=nav.index, y=nav, name="Stratégie (valeur cumulée, normalisée au départ)"))
    try:
        spy = get_asset_price("SPY", start=str(start_date))
        spy_nav = (1 + spy.pct_change(fill_method=None).fillna(0)).cumprod()
        fig2.add_trace(go.Scatter(x=spy_nav.index, y=spy_nav/spy_nav.iloc[0], name="Indice S&P 500 (SPY) — normalisé"))
    except Exception as e:
        log_warn(f"Récupération SPY impossible: {e}")
    try:
        ew = (1 + prices.pct_change().mean(axis=1)).cumprod()
        fig2.add_trace(go.Scatter(x=ew.index, y=ew/ew.iloc[0], name="Égal-pondération des secteurs — normalisée"))
    except Exception as e:
        log_warn(f"Calcul égal-pondération impossible: {e}")

    # Ombrage des périodes de récession (NBER)
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
        log_warn(f"Ombres récession NBER ignorées: {e}")

    fig2.update_layout(height=440, yaxis_title="Valeur cumulée (normalisée au départ)",
                       yaxis_type=("log" if use_log else "linear"))
    st.plotly_chart(fig2, width='stretch')

    # Carte de chaleur
    st.subheader("Carte de chaleur des scores sectoriels (12 derniers mois)")
    hm = scores_scaled_exo.tail(12)
    if not hm.empty:
        fig = go.Figure(data=go.Heatmap(
            z=hm.values, x=hm.columns, y=[d.strftime("%Y-%m") for d in hm.index],
            zmin=-5, zmax=5, colorscale="RdBu",
            colorbar=dict(orientation="h", y=-0.2, title="Score normalisé (de −5 à +5)")
        ))
        fig.update_layout(height=380, margin=dict(b=60))
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # -------------------- CLASSEMENT ACTUEL --------------------
    st.subheader("Classement actuel des secteurs (avec variation depuis le mois précédent)")
    if not scores_scaled_exo.dropna().empty:
        idxs = scores_scaled_exo.dropna().index
        idxs_sorted = idxs.sort_values()
        last_idx = idxs_sorted[-1]
        prev_idx = idxs_sorted[-2] if len(idxs_sorted) > 1 else None

        cur = scores_scaled_exo.loc[last_idx].sort_values(ascending=False).to_frame("Score normalisé")
        if prev_idx is not None:
            prev = scores_scaled_exo.loc[prev_idx].reindex(cur.index)
            delta = (cur["Score normalisé"] - prev).rename("Variation mensuelle (Δ)")
        else:
            delta = pd.Series(0, index=cur.index, name="Variation mensuelle (Δ)")
        view = pd.concat([cur, delta], axis=1)
        view["Direction"] = np.where(view["Variation mensuelle (Δ)"]>=0, "Hausse (↑)", "Baisse (↓)")

        idx_top = view["Score normalisé"].idxmax()
        def highlight_top(row):
            return ['font-weight:bold;background-color:rgba(16,185,129,.15)' if row.name==idx_top else '' for _ in row]

        styled = (view.style
                  .format({"Score normalisé":"{:.2f}", "Variation mensuelle (Δ)":"{:+.2f}"})
                  .bar(subset=["Score normalisé"], align="mid", color='#22c55e')
                  .apply(highlight_top, axis=1))

        st.dataframe(styled, width='stretch')
    else:
        st.info("Scores indisponibles pour l’instant.")

    # Attribution par thème (brut, avant mise à l’échelle)
    st.subheader("Contribution brute par thème (avant mise à l’échelle) — 6 derniers mois")
    st.dataframe(scores_raw.tail(6).rename_axis("Mois").style.format("{:+.2f}"), width='stretch')

    st.markdown("---")

    # -------------------- CORRÉLATIONS ROULANTES --------------------
    st.subheader("Corrélations roulantes (fenêtre de 6 mois) entre un secteur et les thèmes macro")
    pxm = prices.resample("ME").last()
    candidates = [s for s in (SECTOR_ETFS + (EXTRA_ETFS if include_gdx else [])) if s in pxm.columns]
    if candidates:
        pick_sec = st.selectbox("Choisir un secteur coté (ETF)", candidates, index=min(4, len(candidates)-1))
        r = pxm[pick_sec].pct_change()
        dfc = pd.concat([r.rename("Rendement du secteur"), themes], axis=1).dropna()
        if not dfc.empty:
            roll = dfc.rolling(6).corr().xs("Rendement du secteur", level=1, drop_level=False)
            roll = roll.drop(columns=["Rendement du secteur"], errors="ignore")
            fig = go.Figure()
            for c in ["Croissance","Inflation","Taux d’intérêt","Dollar américain","Marché du travail"]:
                if c in roll.columns:
                    fig.add_trace(go.Scatter(x=roll.index, y=roll[c], name=c))
            fig.update_layout(height=340, yaxis_title="Corrélation (fenêtre de 6 mois)")
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Pas de colonnes de prix disponibles pour calculer les corrélations.")

    st.markdown("---")

    # -------------------- SCREENER (HOLDINGS) --------------------
    st.subheader("Screener simple des principaux titres dans les secteurs sélectionnés")
    chosen = (cur.index[:top_k].tolist() if 'cur' in locals() and not cur.empty
              else (SECTOR_ETFS[:top_k]))
    st.caption("Secteurs analysés : " + ", ".join(chosen))
    for sec in chosen:
        try:
            t = yf.Ticker(sec)
            holds = getattr(t, "fund_holdings", None)
            st.markdown(f"**{sec} — Principales lignes si disponibles**")
            if holds is None or len(holds)==0:
                st.write(f"Aucune composition détaillée disponible pour {sec}.")
                st.markdown(f"Page d’information : https://finance.yahoo.com/quote/{sec}")
                # Fallback : indicateurs simples sur l’ETF lui-même
                try:
                    px = get_multi_yf([sec], start="2018-01-01")
                    if not px.empty:
                        m3 = px.pct_change(63).iloc[-1].rename("Performance 3 mois")
                        m6 = px.pct_change(126).iloc[-1].rename("Performance 6 mois")
                        vol = px.pct_change().rolling(63).std().iloc[-1].rename("Volatilité (approx.)")
                        df_fallback = pd.concat([m3, m6, vol], axis=1)
                        df_fallback["Score heuristique"] = 0.6*df_fallback["Performance 6 mois"] + 0.4*df_fallback["Performance 3 mois"] - 0.2*df_fallback["Volatilité (approx.)"]
                        st.dataframe(df_fallback.round(3), width='stretch')
                except Exception as e:
                    log_warn(f"Screener fallback en échec pour {sec}: {e}")
                continue
            if isinstance(holds, dict) and "symbol" in holds:
                tickers = pd.Series(holds["symbol"]).dropna().tolist()[:15]
            elif isinstance(holds, pd.DataFrame) and "symbol" in holds.columns:
                tickers = holds["symbol"].dropna().tolist()[:15]
            else:
                st.write(f"Format de la composition non reconnu pour {sec}.")
                continue
            px = get_multi_yf(tickers, start="2018-01-01")
            if px.empty:
                st.write(f"Aucun historique de prix pour les composants de {sec}.")
                continue
            m3 = px.pct_change(63).iloc[-1]
            m6 = px.pct_change(126).iloc[-1]
            vol = px.pct_change().rolling(63).std().iloc[-1]
            df = pd.DataFrame({
                "Performance 3 mois": m3,
                "Performance 6 mois": m6,
                "Volatilité (approx.)": vol
            })
            df["Score heuristique"]=0.6*df["Performance 6 mois"]+0.4*df["Performance 3 mois"]-0.2*df["Volatilité (approx.)"]
            st.dataframe(df.sort_values("Score heuristique", ascending=False).round(3), width='stretch')
        except Exception as e:
            st.write(f"Screener indisponible pour {sec} : {e}")

    # Notes d’ajustement
    overlay_txt = " | ".join(overlay_notes) if overlay_notes else "Aucun ajustement par facteurs externes n’a été appliqué."
    st.info(overlay_txt)

    st.caption("Outil personnel. Ceci n’est pas un conseil d’investissement. Sources : FRED, yfinance, CBOE (VIX), Fed de New York (GSCPI), Banque du Canada (USD/CAD), GPR, TradingEconomics (calendrier).")
