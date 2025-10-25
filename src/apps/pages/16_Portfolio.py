from __future__ import annotations

from pathlib import Path
import sys as _sys
import json
import pandas as pd
import streamlit as st
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Portfolio — Finance Agent", layout="wide")
page_header(active="user")
st.subheader("💼 Portfolio — Poids & Propositions")

def _load_holdings() -> pd.DataFrame:
    p = Path('data/portfolio/holdings.json')
    if not p.exists():
        return pd.DataFrame(columns=['ticker','weight'])
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame(columns=['ticker','weight'])

def _latest_final_parquet() -> Path | None:
    parts = sorted(Path('data/forecast').glob('dt=*/final.parquet'))
    return parts[-1] if parts else None

with st.sidebar:
    st.header("Actions")
    if st.button("Créer un fichier holdings d'exemple"):
        p = Path('data/portfolio'); p.mkdir(parents=True, exist_ok=True)
        sample = [{"ticker":"AAPL","weight":0.2},{"ticker":"MSFT","weight":0.2}]
        (p/'holdings.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')
        st.success("data/portfolio/holdings.json créé")
    top_n = st.slider("Top N (proposition)", 1, 10, 5)
    mode = st.selectbox("Méthode de pondération", ["Égal‑pondéré","Proportionnel au score"], index=0)
    st.divider()
    st.subheader("Scenario tilt (macro)")
    do_tilt = st.checkbox("Activer le tilt macro", value=False)
    tilt_strength = st.slider("Intensité tilt (%)", 0, 20, 5)
    tilt_list = st.text_input("Tickers à favoriser (optionnel)", value="GDX, AEM.TO, ABX.TO, K.TO, NGD.TO")
    st.caption("Sans liste, l'app tentera d'inférer des tickers via presets (data/config/tilt_presets.json) puis heuristique or/énergie.")
    st.divider()
    st.subheader("Rebalance simulator")
    import datetime as _dt
    start_date = st.date_input("Date de départ (perf)", value=(_dt.date.today() - _dt.timedelta(days=21)))

hold = _load_holdings()
st.subheader("Positions actuelles")
if hold.empty:
    st.info("Aucune position. Utilisez l'action dans la barre latérale pour créer un exemple, ou ajoutez data/portfolio/holdings.json.")
else:
    st.dataframe(hold, use_container_width=True)

st.subheader("Proposition (Top‑N par score final)")
fp = _latest_final_parquet()
if not fp:
    st.info("Aucune consolidation de prévisions disponible. Consultez Admin → Agents Status pour l'état du pipeline.")
else:
    df = pd.read_parquet(fp)
    if df.empty:
        st.info("final.parquet vide")
    else:
        top = df[df['horizon']=='1m'].sort_values('final_score', ascending=False).head(top_n)
        if top.empty:
            st.info("Pas de données 1m")
        else:
            if mode.startswith("Égal"):
                weights = [round(1.0/len(top), 6)]*len(top)
            else:
                sc = top['final_score'].clip(lower=0).astype(float)
                ssum = float(sc.sum()) or 1.0
                weights = [round(float(v/ssum), 6) for v in sc]
            # scenario tilt (optional)
            if do_tilt:
                # try regime
                try:
                    import json as _json
                    from pathlib import Path as _P
                    reg = sorted(_P('data/macro/regime').glob('dt=*/regime.json'))
                    tilt_names = [x.strip().upper() for x in tilt_list.split(',') if x.strip()]
                    if reg:
                        robj = _json.loads(_P(reg[-1]).read_text(encoding='utf-8'))
                        probs = (robj.get('probs') or {})
                        # favor commodities in inflation regime
                        infl_p = float(probs.get('inflation') or 0.0)
                        favor_set = set(tilt_names)
                        # if not specified, try presets
                        if not favor_set:
                            try:
                                cfgp = _P('data/config/tilt_presets.json')
                                if cfgp.exists():
                                    presets = _json.loads(cfgp.read_text(encoding='utf-8'))
                                    lst = presets.get('inflation') or []
                                    favor_set = set([str(x).upper() for x in lst if isinstance(x, (str,))])
                            except Exception:
                                pass
                        if not favor_set:
                            favor_set = set([t for t in top['ticker'] if any(k in str(t).upper() for k in ['GDX','GLD','XLE','AEM','ABX','K.TO','NGD'])])
                        if favor_set:
                            add = (tilt_strength/100.0) * infl_p
                            # add equally across favored tickers
                            n = sum(1 for t in top['ticker'] if str(t).upper() in favor_set) or 1
                            weights = [max(0.0, w + (add/n if str(t).upper() in favor_set else 0.0)) for t,w in zip(top['ticker'], weights)]
                            # renormalize
                            ssum = sum(weights) or 1.0
                            weights = [round(w/ssum, 6) for w in weights]
                except Exception:
                    pass
            prop = pd.DataFrame({'ticker': top['ticker'], 'proposed_weight': weights, 'final_score': top['final_score'].round(4)})
            st.dataframe(prop, use_container_width=True)
            st.caption("Poids égal‑pondéré; ajustez selon votre profil de risque.")
            # export buttons
            try:
                csv_bytes = prop.to_csv(index=False).encode('utf-8')
                st.download_button("Exporter pondérations (CSV)", data=csv_bytes, file_name="portfolio_proposed.csv", mime="text/csv")
                out_json = prop.to_dict(orient='records')
                st.download_button("Exporter pondérations (JSON)", data=json.dumps(out_json, ensure_ascii=False, indent=2), file_name="portfolio_proposed.json", mime="application/json")
            except Exception:
                pass

            # Rebalance simulator: compare current vs proposed returns since start_date
            try:
                import pandas as _pd
                def _ret_since(t: str, d0) -> float | None:
                    p = Path('data/prices')/f"ticker={t}"/'prices.parquet'
                    if not p.exists(): return None
                    df = _pd.read_parquet(p)
                    if 'date' in df.columns:
                        df['date'] = _pd.to_datetime(df['date'], errors='coerce'); df = df.set_index('date')
                    if df.empty or 'Close' not in df.columns: return None
                    d0 = _pd.to_datetime(d0)
                    try:
                        idx0 = df.index.get_loc(d0, method='nearest')
                    except Exception:
                        aft = df[df.index>=d0];
                        if aft.empty: return None
                        idx0 = df.index.get_loc(aft.index[0])
                    r = float(df['Close'].iloc[-1]/df['Close'].iloc[idx0] - 1.0)
                    return r
                # current
                cur = hold.copy()
                if not cur.empty:
                    cur['ret'] = [ _ret_since(t, start_date) for t in cur['ticker'] ]
                    cur = cur.dropna(subset=['ret','weight'])
                    ssum = float(cur['weight'].sum()) or 1.0
                    cur['w'] = cur['weight']/ssum
                    ret_cur = float((cur['w']*cur['ret']).sum()) if not cur.empty else None
                else:
                    ret_cur = None
                # proposed
                pr = prop.copy(); pr['ret'] = [ _ret_since(t, start_date) for t in pr['ticker'] ]
                pr = pr.dropna(subset=['ret','proposed_weight'])
                ssum = float(pr['proposed_weight'].sum()) or 1.0
                pr['w'] = pr['proposed_weight']/ssum
                ret_prop = float((pr['w']*pr['ret']).sum()) if not pr.empty else None
                st.subheader("Rebalance simulator")
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("Portefeuille actuel", f"{(ret_cur*100):.2f}%" if ret_cur is not None else "n/a")
                with c2: st.metric("Proposition", f"{(ret_prop*100):.2f}%" if ret_prop is not None else "n/a")
                with c3: 
                    if ret_cur is not None and ret_prop is not None:
                        st.metric("Différence", f"{(ret_prop-ret_cur)*100:.2f}%")
            except Exception:
                pass
page_footer()
