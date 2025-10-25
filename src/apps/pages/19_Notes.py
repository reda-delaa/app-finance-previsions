from pathlib import Path
import sys as _sys
import datetime as dt
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Notes — Finance Agent", layout="wide")
st.title("📝 Notes — Journal personnel")

today = dt.datetime.utcnow().strftime('%Y%m%d')
base = Path('data/notes')
base.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Dates")
    dates = sorted([p.name for p in base.glob('dt=*')], reverse=True)
    chosen = st.selectbox("Dossier date", dates, index=0 if dates else None)
    if st.button("Nouveau (aujourd'hui)"):
        ndir = base/f"dt={today}"; ndir.mkdir(parents=True, exist_ok=True)
        if not (ndir/'notes.md').exists():
            (ndir/'notes.md').write_text("", encoding='utf-8')
        st.session_state['__notes_chosen'] = ndir.name

target_dir = base/(chosen or f"dt={today}")
target_dir.mkdir(parents=True, exist_ok=True)
path = target_dir/'notes.md'
text = ""
if path.exists():
    text = path.read_text(encoding='utf-8')

st.subheader(f"Éditer: {target_dir.name}/notes.md")
new_text = st.text_area("", value=text, height=320)
if st.button("Enregistrer"):
    path.write_text(new_text or "", encoding='utf-8')
    st.success("Enregistré")

st.subheader("Aperçu")
st.markdown(new_text or text)

