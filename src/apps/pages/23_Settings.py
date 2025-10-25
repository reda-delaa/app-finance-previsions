from pathlib import Path
import sys as _sys
import json
import streamlit as st
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Settings — Finance Agent", layout="wide")
st.title("⚙️ Settings — Presets & Seuils")
page_header(active="admin")

base_cfg = Path('data/config')
base_cfg.mkdir(parents=True, exist_ok=True)

# Tilt presets
st.subheader("Tilt Presets (macro → tickers)")
tilt_path = base_cfg/'tilt_presets.json'
default_presets = {
    "inflation": ["GDX","GLD","XLE"],
    "slowdown": ["XLU","XLV"],
    "deflation": ["TLT","IEF"],
}
try:
    presets = json.loads(tilt_path.read_text(encoding='utf-8')) if tilt_path.exists() else default_presets
except Exception:
    presets = default_presets
text = json.dumps(presets, ensure_ascii=False, indent=2)
new_text = st.text_area("tilt_presets.json", value=text, height=180)
if st.button("Enregistrer presets"):
    try:
        obj = json.loads(new_text)
        tilt_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        st.success(f"Enregistré: {tilt_path}")
    except Exception as e:
        st.warning(f"Erreur: {e}")

st.divider()
st.subheader("Seuils Alerts")
alerts_path = base_cfg/'alerts.json'
default_alerts = {"move_abs_pct": 1.0}
try:
    alerts = json.loads(alerts_path.read_text(encoding='utf-8')) if alerts_path.exists() else default_alerts
except Exception:
    alerts = default_alerts
mv = st.slider("Mouvement (%, 1j) minimum", 0.0, 5.0, float(alerts.get('move_abs_pct', 1.0)), 0.1)
if st.button("Enregistrer seuils"):
    try:
        alerts_path.write_text(json.dumps({"move_abs_pct": mv}, ensure_ascii=False, indent=2), encoding='utf-8')
        st.success(f"Enregistré: {alerts_path}")
    except Exception as e:
        st.warning(f"Erreur: {e}")

page_footer()
