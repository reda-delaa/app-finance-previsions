from __future__ import annotations

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
            daily_perf = []
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
                    avg = float(np.mean(rets))
                    basket.append(avg)
                    daily_perf.append({'dt': str(d), 'mean_return': avg})
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
                # Friendly narrative
                try:
                    perf_df = pd.DataFrame(daily_perf)
                    cum_val = None
                    if not perf_df.empty:
                        perf_df['dt'] = pd.to_datetime(perf_df['dt'], errors='coerce')
                        perf_df = perf_df.sort_values('dt')
                        perf_df['cum_return'] = (1.0 + perf_df['mean_return']).cumprod() - 1.0
                        if not perf_df['cum_return'].empty:
                            cum_val = float(perf_df['cum_return'].iloc[-1])
                    st.markdown(
                        f"En clair: sur {int(s.count())} jours, un panier quotidien égal‑pondéré des {top_n} meilleures idées a généré en moyenne {float(s.mean())*100:.2f}% par jour, pour une médiane de {float(s.median())*100:.2f}% et une variabilité (écart‑type) de {(float(s.std(ddof=1)) if len(s)>1 else 0.0)*100:.2f}%. "
                        + (f"Le cumul sur la période atteint {cum_val*100:.2f}% (ré‑investi chaque jour)." if cum_val is not None else "")
                    )
                except Exception:
                    pass
                st.subheader("Details")
                det_df = pd.DataFrame(details)
                st.dataframe(det_df, use_container_width=True)
                # CSV export
                try:
                    csv_bytes = det_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Télécharger les détails (CSV)", data=csv_bytes, file_name=f"eval_details_{horizon}.csv", mime="text/csv")
                except Exception:
                    pass
                # Daily cumulative performance chart
                try:
                    if 'perf_df' not in locals():
                        perf_df = pd.DataFrame(daily_perf)
                        if not perf_df.empty:
                            perf_df['dt'] = pd.to_datetime(perf_df['dt'], errors='coerce')
                            perf_df = perf_df.sort_values('dt')
                            perf_df['cum_return'] = (1.0 + perf_df['mean_return']).cumprod() - 1.0
                    if not perf_df.empty:
                        st.subheader("Cumulative Performance (Top N basket)")
                        perf_plot = perf_df.set_index('dt')[['cum_return']]
                        st.area_chart(perf_plot)
                except Exception:
                    pass
                # Calibration (probability-of-up proxy)
                try:
                    st.subheader("Confidence Calibration (simple)")
                    det = det_df.copy()
                    if not det.empty:
                        # Predicted up probability proxy
                        det['p_up'] = np.where(det['direction']=='up', det['confidence'], 1.0 - det['confidence'])
                        det['realized_up'] = (det['realized_return'] > 0).astype(int)
                        bins = [0.0,0.5,0.6,0.7,0.8,0.9,1.0]
                        labels = ['≤0.5','0.5–0.6','0.6–0.7','0.7–0.8','0.8–0.9','>0.9']
                        det['p_bin'] = pd.cut(det['p_up'].clip(0,1), bins=bins, labels=labels, include_lowest=True)
                        calib = det.groupby('p_bin')['realized_up'].mean().reset_index().rename(columns={'realized_up':'actual_up_rate'})
                        st.bar_chart(calib.set_index('p_bin'))
                        st.caption("If well calibrated, higher confidence bins should map to higher actual up rates. This is a simple proxy; consider a dedicated probability model for rigorous calibration.")
                except Exception:
                    pass
            else:
                st.info("Insufficient price cache to compute realized returns.")
