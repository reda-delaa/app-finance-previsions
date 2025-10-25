from pathlib import Path
import sys as _sys
import streamlit as st
import pandas as pd
import datetime as dt

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from core.data_store import have_files, query_duckdb
from core.market_data import get_price_history
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Backtests — Finance Agent", layout="wide")
st.title("📐 Backtests — Top-N (demo)")

with st.sidebar:
    st.header("Paramètres")
    horizon = st.selectbox("Horizon", ["1w","1m","1y"], index=1)
    top_n = st.slider("Top N", 1, 10, 5)
    start_dt = st.text_input("Start dt (YYYYMMDD)", value="20250101")
    end_dt = st.text_input("End dt (YYYYMMDD)", value=dt.datetime.utcnow().strftime("%Y%m%d"))
    run = st.button("Run backtest")

def _score_df(h):
    df = query_duckdb(f"""
        select *,
               (case direction when 'up' then 1.0 when 'down' then -1.0 else 0.0 end) as dir_base,
               ((case direction when 'up' then 1.0 when 'down' then -1.0 else 0.0 end) * cast(confidence as double)
                 + 0.5 * coalesce(cast(expected_return as double), 0.0)) as score
        from read_parquet('data/forecast/dt=*/forecasts.parquet')
        where horizon = '{h}' and substr(dt, 1, 10) is not null
    """)
    return df

if run:
    if not have_files("data/forecast/dt=*/forecasts.parquet"):
        st.info("Aucune prévision disponible pour la période. Consultez Admin → Agents Status pour l'état du pipeline.")
    else:
        df = _score_df(horizon)
        df = df[(df["dt"] >= pd.to_datetime(start_dt)) & (df["dt"] <= pd.to_datetime(end_dt))]
        if df.empty:
            st.info("No forecasts in the selected window.")
        else:
            # For each dt, pick top N tickers, then compute realized horizon returns via yfinance (best-effort)
            dts = sorted(df["dt"].dt.strftime("%Y-%m-%d").unique().tolist())
            basket_returns = []
            details = []
            ahead_days = {"1w": 5, "1m": 21, "1y": 252}[horizon]
            for d in dts:
                sdf = df[df["dt"].dt.strftime("%Y-%m-%d") == d].sort_values("score", ascending=False).head(top_n)
                if sdf.empty:
                    continue
                rets = []
                for t in sdf["ticker"].tolist():
                    try:
                        # Prefer cached parquet if present
                        pfile = Path("data/prices") / f"ticker={t}" / "prices.parquet"
                        if pfile.exists():
                            hist = pd.read_parquet(pfile)
                            # try to standardize columns
                            if 'date' in hist.columns:
                                hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
                                hist = hist.set_index('date')
                        else:
                            # fetch around the date and forward window
                            start = (pd.to_datetime(d) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                            end = (pd.to_datetime(d) + pd.Timedelta(days=ahead_days+5)).strftime("%Y-%m-%d")
                            hist = get_price_history(t, start=start, end=end)
                        if hist is None or hist.empty:
                            continue
                        idx = hist.index.get_loc(pd.to_datetime(d), method="nearest")
                        i2 = min(len(hist)-1, idx + ahead_days)
                        r = float(hist["Close"].iloc[i2] / hist["Close"].iloc[idx] - 1.0)
                        rets.append(r)
                        details.append({"dt": d, "ticker": t, "horizon": horizon, "realized_return": r})
                    except Exception:
                        pass
                if rets:
                    basket_returns.append(pd.Series(rets).mean())
            if basket_returns:
                ser = pd.Series(basket_returns)
                st.subheader("Results")
                st.write({
                    "count": int(ser.count()),
                    "avg_basket_return": float(ser.mean()),
                    "median": float(ser.median()),
                    "stdev": float(ser.std(ddof=1)) if ser.count() > 1 else 0.0,
                })
                st.subheader("Details")
                st.dataframe(pd.DataFrame(details), use_container_width=True)
            else:
                st.info("Insufficient price data to compute realized returns.")
