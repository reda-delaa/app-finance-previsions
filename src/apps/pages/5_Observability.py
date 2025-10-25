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
    # Repo root = <repo>
    repo_root = Path(__file__).resolve().parents[3]
    logdir = repo_root / 'logs' / 'ui'
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
    # HTTP probe
    http_ok = False
    http_ms = None
    try:
        import time, requests
        t0 = time.perf_counter();
        r = requests.get(f"http://127.0.0.1:{port}", timeout=1.0)
        http_ok = (r.status_code == 200)
        http_ms = int((time.perf_counter() - t0) * 1000)
    except Exception:
        http_ok = False
        http_ms = None

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Port UI", port)
    with c2: st.metric("Process vivant", "Oui" if alive else "Non")
    with c3: st.metric("PID", str(pid or '—'))
    with c4: st.metric("HTTP", ("200" if http_ok else "KO") + (f" ({http_ms} ms)" if http_ms is not None else ""))
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

st.markdown("#### Log en direct (manuel)")
with st.expander("Afficher/rafraîchir", expanded=False):
    repo_root = Path(__file__).resolve().parents[3]
    port = os.getenv("AF_UI_PORT", "5555")
    logfile = repo_root / 'logs' / 'ui' / f'streamlit_{port}.log'
    n_lines = st.slider("Nombre de lignes (tail)", 10, 400, 120, step=10)
    if logfile.exists():
        try:
            lines = logfile.read_text(encoding='utf-8', errors='ignore').splitlines()
            st.code("\n".join(lines[-n_lines:]) or "(vide)", language='bash')
            if st.button("Rafraîchir"):
                pass  # rerender
        except Exception as e:
            st.warning(f"Lecture log impossible: {e}")
    else:
        st.caption("Log introuvable — lancez l'UI en arrière-plan pour créer le log.")

st.markdown("#### Données — Fraîcheur (qualité)")
try:
    repo_root = Path(__file__).resolve().parents[3]
    parts = sorted((repo_root/'data'/'quality').glob('dt=*/freshness.json'))
    if parts:
        import json as _json
        js = _json.loads(parts[-1].read_text(encoding='utf-8'))
        checks = js.get('checks') or {}
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Forecasts (aujourd'hui)", "Oui" if checks.get('forecasts_today') else "Non")
        with c2: st.metric("Final (aujourd'hui)", "Oui" if checks.get('final_today') else "Non")
        with c3: st.metric("Macro (aujourd'hui)", "Oui" if checks.get('macro_today') else "Non")
        with c4:
            r = checks.get('prices_5y_coverage_ratio')
            st.metric("Couverture prix ≥5y", f"{int(r*100)}%" if isinstance(r, (int,float)) else "n/a")
    else:
        st.caption("Aucun freshness.json — exécutez `make update-monitor`.")
except Exception:
    st.caption("Section fraîcheur indisponible (lecture).")

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
                    root = Path(__file__).resolve().parents[3]
                    script = str(root/"scripts"/"ui_restart_bg.sh")
                    out = subprocess.run(["bash", script], capture_output=True, text=True, env=env, timeout=45)
                    st.info("Redémarrage demandé. L'UI peut se couper puis revenir; rechargez cette page après 2–3s.")
                    if out.stdout:
                        st.code(out.stdout.strip(), language='bash')
                    if out.stderr:
                        st.caption("stderr:")
                        st.code(out.stderr.strip(), language='bash')
                except Exception as e:
                    st.error(f"Échec du redémarrage: {e}")

st.markdown("#### Actions (Admin) — Démarrer / Arrêter l'UI")
cols = st.columns(2)
with cols[0]:
    with st.form("ui_start_form"):
        st.caption("Démarrer l'interface en arrière‑plan (si aucune instance n'écoute).")
        start_confirm = st.checkbox("Je confirme le démarrage de l'UI")
        start_submit = st.form_submit_button("Démarrer (bg)")
        if start_submit:
            if not start_confirm:
                st.warning("Cochez la case de confirmation avant de démarrer.")
            else:
                try:
                    env = dict(**os.environ)
                    env.setdefault("AF_UI_PORT", os.getenv("AF_UI_PORT", "5555"))
                    root = Path(__file__).resolve().parents[3]
                    script = str(root/"scripts"/"ui_start_bg.sh")
                    out = subprocess.run(["bash", script], capture_output=True, text=True, env=env, timeout=45)
                    st.info("Démarrage demandé. L'UI devrait être disponible sous peu.")
                    if out.stdout:
                        st.code(out.stdout.strip(), language='bash')
                    if out.stderr:
                        st.caption("stderr:")
                        st.code(out.stderr.strip(), language='bash')
                except Exception as e:
                    st.error(f"Échec du démarrage: {e}")

with cols[1]:
    with st.form("ui_stop_form"):
        st.caption("Arrêter l'interface courante (best‑effort).")
        stop_confirm = st.checkbox("Je confirme l'arrêt de l'UI")
        stop_submit = st.form_submit_button("Arrêter")
        if stop_submit:
            if not stop_confirm:
                st.warning("Cochez la case de confirmation avant d'arrêter.")
            else:
                try:
                    root = Path(__file__).resolve().parents[3]
                    script = str(root/"scripts"/"ui_stop.sh")
                    out = subprocess.run(["bash", script], capture_output=True, text=True, timeout=30)
                    st.info("Arrêt demandé.")
                    if out.stdout:
                        st.code(out.stdout.strip(), language='bash')
                    if out.stderr:
                        st.caption("stderr:")
                        st.code(out.stderr.strip(), language='bash')
                except Exception as e:
                    st.error(f"Échec de l'arrêt: {e}")

st.markdown("#### Processus")
st.write("UI principale et pages chargées. Consultez les logs dans le dossier logs/ si nécessaire.")
page_footer()
