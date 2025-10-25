from pathlib import Path
import sys as _sys
import json
import datetime as dt
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Events — Finance Agent", layout="wide")
st.title("📅 Upcoming Events — Calendrier Macro")

with st.sidebar:
    st.header("Source")
    base = Path("data/events")
    dates = sorted([p.name for p in base.glob("dt=*")], reverse=True)
    chosen = st.selectbox("Dossier date", dates, index=0 if dates else None)
    st.caption("Les dates sont générées de façon approximative (heuristiques). Vérifiez toujours les sources officielles.")

def _fmt_date(s: str) -> str:
    try:
        d = dt.date.fromisoformat(s)
        return d.strftime("%A %d %B %Y")
    except Exception:
        return s

if chosen:
    ddir = Path("data/events") / chosen
    f = ddir / "events.json"
    if not f.exists():
        st.info("Aucun fichier d'événements pour cette date.")
    else:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"Lecture impossible: {e}")
            obj = None
        if obj:
            events = obj.get("events") or []
            st.subheader("Événements à venir (14 jours par défaut)")
            if not events:
                st.info("Aucun événement détecté dans la fenêtre.")
            else:
                # Display as a friendly list
                for ev in sorted(events, key=lambda x: x.get("date","")):
                    st.markdown(f"- { _fmt_date(ev.get('date','')) } — **{ ev.get('name','') }**\n  ")
                    if ev.get("impact"):
                        st.caption(ev.get("impact"))
            with st.expander("Voir le JSON brut"):
                st.json(obj)
else:
    st.info("Sélectionnez une date pour afficher le calendrier des événements.")

