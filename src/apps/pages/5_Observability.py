from pathlib import Path
import sys as _sys
import os
import streamlit as st
import subprocess
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Observability — Finance Agent", layout="wide")
page_header(active="admin")
st.subheader("🛠️ Observability — Santé & Clés")

keys = [
    "FIRECRAWL_API_KEY",
    "SERPER_API_KEY",
    "TAVILY_API_KEY",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
]

st.markdown("#### Clés d'API (présence seulement)")
rows = []
for i, k in enumerate(keys, start=1):
    v = os.getenv(k)
    rows.append({"clé": f"Key #{i}", "présente": bool(v)})
st.dataframe(rows, use_container_width=True)

st.markdown("#### UI — Santé")
try:
    import json as _json
    import time as _time
    port = os.getenv("AF_UI_PORT", "5555")
    logdir = Path('logs/ui')
    pidfile = logdir / f'streamlit_{port}.pid'
    logfile = logdir / f'streamlit_{port}.log'
    pid = None
    alive = False
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip() or '0')
            # Probe process existence
            import os as _os, signal as _signal
            if pid > 0:
                _os.kill(pid, 0)
                alive = True
        except Exception:
            alive = False
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Port UI", port)
    with c2: st.metric("Process vivant", "Oui" if alive else "Non")
    with c3: st.metric("PID", str(pid or '—'))
    if logfile.exists():
        st.caption(f"Dernières lignes du log ({logfile}):")
        try:
            tail = logfile.read_text(encoding='utf-8', errors='ignore').splitlines()[-10:]
            st.code("\n".join(tail) or "(vide)", language='bash')
            st.download_button("Télécharger le log", data=logfile.read_bytes(), file_name=f"streamlit_{port}.log")
        except Exception:
            pass
    else:
        st.caption("Aucun log trouvé pour l'UI (lancé via ui_start_bg/ui_restart_bg ?)")
except Exception:
    st.caption("Section UI indisponible (erreur d'accès système).")

st.markdown("#### Action (Admin) — Redémarrer l'UI")
with st.expander("Redémarrer l'interface (arrière‑plan)", expanded=False):
    st.caption("Cette action stoppe l'instance Streamlit courante puis la relance en arrière‑plan avec journaux. L'interface sera indisponible quelques secondes.")
    with st.form("ui_restart_form"):
        confirm = st.checkbox("Je confirme le redémarrage immédiat de l'UI")
        submitted = st.form_submit_button("Redémarrer maintenant (bg)")
        if submitted:
            if not confirm:
                st.warning("Cochez la case de confirmation avant de redémarrer.")
            else:
                try:
                    env = dict(**os.environ)
                    env.setdefault("AF_UI_PORT", os.getenv("AF_UI_PORT", "5555"))
                    out = subprocess.run(["bash", "scripts/ui_restart_bg.sh"], capture_output=True, text=True, env=env, timeout=30)
                    st.info("Redémarrage demandé. L'UI peut se couper puis revenir; rechargez cette page après 2–3s.")
                    if out.stdout:
                        st.code(out.stdout.strip(), language='bash')
                    if out.stderr:
                        st.caption("stderr:")
                        st.code(out.stderr.strip(), language='bash')
                except Exception as e:
                    st.error(f"Échec du redémarrage: {e}")

st.markdown("#### Processus")
st.write("UI principale et pages chargées. Consultez les logs dans le dossier logs/ si nécessaire.")
page_footer()
