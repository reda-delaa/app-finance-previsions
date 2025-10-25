from __future__ import annotations

from pathlib import Path
import datetime as dt
import json
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html


def _latest(path_glob: str) -> tuple[str | None, Path | None]:
    parts = sorted(Path('.').glob(path_glob))
    if not parts:
        return (None, None)
    p = parts[-1]
    # try to parse dt=YYYYMMDD
    try:
        name = p.parent.name if p.name.endswith('.parquet') or p.name.endswith('.json') else p.name
        if name.startswith('dt='):
            return (name.split('=',1)[-1], p)
    except Exception:
        pass
    return (None, p)


def _file_info(p: Path | None) -> dict:
    if not p or not p.exists():
        return {"exists": False, "path": str(p) if p else "-", "size": 0, "mtime": "-"}
    st = p.stat()
    mtime = dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')
    return {"exists": True, "path": str(p), "size": st.st_size, "mtime": mtime}


def _freshness_summary() -> dict:
    try:
        parts = sorted(Path('data/quality').glob('dt=*/freshness.json'))
        if not parts:
            return {"exists": False}
        js = json.loads(parts[-1].read_text(encoding='utf-8'))
        return {"exists": True, "checks": js.get('checks') or {}, "path": str(parts[-1])}
    except Exception:
        return {"exists": False}


def layout():
    # Forecasts / Final
    last_f_dt, last_f = _latest('data/forecast/dt=*/forecasts.parquet')
    last_final_dt, last_final = _latest('data/forecast/dt=*/final.parquet')
    # Macro
    last_macro_dt, last_macro = _latest('data/macro/forecast/dt=*/macro_forecast.parquet')
    # Freshness
    fresh = _freshness_summary()

    rows = []
    fi_f = _file_info(last_f)
    fi_final = _file_info(last_final)
    fi_macro = _file_info(last_macro)
    rows.append(["forecasts.parquet", last_f_dt or "-", fi_f["mtime"], "Oui" if fi_f["exists"] else "Non"]) 
    rows.append(["final.parquet", last_final_dt or "-", fi_final["mtime"], "Oui" if fi_final["exists"] else "Non"]) 
    rows.append(["macro_forecast.parquet", last_macro_dt or "-", fi_macro["mtime"], "Oui" if fi_macro["exists"] else "Non"]) 
    last_fresh_dt = fresh.get("path", "-")
    if "dt=" in last_fresh_dt:
        last_fresh_dt = last_fresh_dt.split("dt=")[-1].split("/")[0]
    else:
        last_fresh_dt = "-"
    rows.append(["freshness.json", last_fresh_dt, "-", "Oui" if fresh.get("exists") else "Non"]) 

    df = pd.DataFrame(rows, columns=["Ressource", "Dernière dt", "Dernière modif", "Présent"])
    table = dbc.Table.from_dataframe(df, striped=True, bordered=False, hover=True, size='sm')

    checks = fresh.get("checks") or {}
    body = [
        html.Small(f"Forecasts aujourd'hui: {'Oui' if checks.get('forecasts_today') else 'Non'}"), html.Br(),
        html.Small(f"Final aujourd'hui: {'Oui' if checks.get('final_today') else 'Non'}"), html.Br(),
        html.Small(f"Macro aujourd'hui: {'Oui' if checks.get('macro_today') else 'Non'}"), html.Br(),
    ]

    return html.Div([
        html.H3("Agents Status"),
        dbc.Card([
            dbc.CardHeader("Fichiers clés — Derniers états"),
            dbc.CardBody(table),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader("Freshness (résumé)"),
            dbc.CardBody(body),
        ]),
    ])
