from pathlib import Path
import sys as _sys
import streamlit as st
import pandas as pd
import numpy as np
import duckdb

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Evaluation — Finance Agent", layout="wide")
st.title("📈 Evaluation — Forecast Quality")

HORIZON_TO_DAYS = {"1w": 5, "1m": 21, "1y": 252}

with st.sidebar:
    st.header("Parameters")
    horizon = st.selectbox("Horizon", ["1w","1m","1y"], index=1)
    top_n = st.slider("Top N per day", 1, 15, 5)
    days_back = st.slider("Days back", 30, 365, 120, step=15)
    run = st.button("Compute")

def _load_forecasts(h: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute("select * from read_parquet('data/forecast/dt=*/forecasts.parquet') where horizon=$1 order by dt", [h]).fetch_df()
    try:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    except Exception:
        pass
    return df

def _score(df: pd.DataFrame) -> pd.Series:
    dir_base = df["direction"].map({"up": 1.0, "down": -1.0}).fillna(0.0)
    if "ml_return" in df.columns:
        mlc = df.get("ml_conf", 0.6)
        return dir_base*df["confidence"].astype(float) + 0.4*df["expected_return"].fillna(0.0).astype(float) + 0.3*df["ml_return"].fillna(0.0).astype(float)*mlc
    return dir_base*df["confidence"].astype(float) + 0.5*df["expected_return"].fillna(0.0).astype(float)

def _cached_prices(ticker: str) -> pd.DataFrame | None:
    p = Path("data/prices")/f"ticker={ticker}"/"prices.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.set_index('date')
        return df
    except Exception:
        return None

def _realized(ticker: str, dt: pd.Timestamp, days: int) -> float | None:
    dfp = _cached_prices(ticker)
    if dfp is None or dfp.empty or 'Close' not in dfp.columns:
        return None
    try:
        idx = dfp.index.get_loc(dt, method='nearest')
    except Exception:
        after = dfp[dfp.index >= dt]
        if after.empty: return None
        idx = dfp.index.get_loc(after.index[0])
    j = min(len(dfp)-1, idx+days)
    return float(dfp['Close'].iloc[j]/dfp['Close'].iloc[idx] - 1.0)

if run:
    days = HORIZON_TO_DAYS.get(horizon, 21)
    df = _load_forecasts(horizon)
    if df.empty:
        st.warning("No forecasts found.")
    else:
        end = df['dt'].max()
        start = end - pd.Timedelta(days=days_back)
        df = df[(df['dt']>=start) & (df['dt']<=end)].copy()
        if df.empty:
            st.warning("No forecasts in selected window.")
        else:
            df['score'] = _score(df)
            details = []
            basket = []
            for d, sdf in df.groupby(df['dt'].dt.date):
                sdf = sdf.sort_values('score', ascending=False).head(top_n)
                rets = []
                for _, row in sdf.iterrows():
                    rr = _realized(str(row['ticker']), pd.Timestamp(d), days)
                    if rr is None: continue
                    rets.append(rr)
                    details.append({
                        'dt': str(d), 'ticker': row['ticker'], 'horizon': horizon,
                        'score': float(row['score']), 'direction': row['direction'], 'confidence': float(row['confidence']),
                        'expected_return': float(row.get('expected_return') or 0.0),
                        'ml_return': float(row.get('ml_return') or 0.0) if 'ml_return' in row else None,
                        'ml_conf': float(row.get('ml_conf') or 0.0) if 'ml_conf' in row else None,
                        'realized_return': float(rr),
                    })
                if rets:
                    basket.append(np.mean(rets))
            if basket:
                s = pd.Series(basket)
                sharpe = float(s.mean()/s.std(ddof=1)) if s.std(ddof=1) and len(s)>1 else 0.0
                st.subheader("Summary")
                st.write({
                    'count_days': int(s.count()),
                    'avg_basket_return': float(s.mean()),
                    'median': float(s.median()),
                    'stdev': float(s.std(ddof=1)) if len(s)>1 else 0.0,
                    'sharpe_like': sharpe,
                })
                st.subheader("Details")
                st.dataframe(pd.DataFrame(details), use_container_width=True)
            else:
                st.info("Insufficient price cache to compute realized returns.")

