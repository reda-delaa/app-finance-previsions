# test_phase1_live.py
# -*- coding: utf-8 -*-
"""
Test live de la Phase 1 (fondamentaux) — sans mock.
- Log les appels et tailles de jeux de données
- Détecte et appelle automatiquement les fonctions de phase1_fundamental si présentes
- Fallback sur yfinance pour vérifier la connectivité et la disponibilité des états financiers
- Sauvegarde des artefacts (JSON/CSV) par ticker

Usage:
  python test_phase1_live.py --ticker NGD.TO --log DEBUG
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import inspect
import math

# ------ Logging --------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Optionnel: verbeux pour les libs HTTP (utile pour REST internes)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("curl_cffi").setLevel(logging.INFO)
    logging.getLogger("yfinance").setLevel(logging.INFO)

log = logging.getLogger("phase1_live")

# ------ Environnement --------------------------------------------------------

def log_env():
    import platform
    try:
        import numpy as np
        npv = np.__version__
    except Exception:
        npv = "N/A"
    try:
        import pandas as pd
        pdv = pd.__version__
    except Exception:
        pdv = "N/A"
    try:
        import yfinance as yf  # noqa
        yfv = getattr(sys.modules["yfinance"], "__version__", "unknown")
    except Exception:
        yfv = "N/A"

    log.info("Python: %s", sys.version.split()[0])
    log.info("Platform: %s %s (%s)", platform.system(), platform.release(), platform.machine())
    log.info("NumPy: %s | Pandas: %s | yfinance: %s", npv, pdv, yfv)
    log.info("Executable: %s", sys.executable)

# ------ Helpers fichiers -----------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dump_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def dump_df(df, path_csv: str) -> None:
    if df is None:
        return
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            df.to_csv(path_csv)
    except Exception:
        pass

# ------ Import phase1_fundamental -------------------------------------------

def import_phase1() -> Optional[Any]:
    try:
        import phase1_fundamental as P1
        log.info("Module 'phase1_fundamental' importé.")
        return P1
    except Exception as e:
        log.warning("Impossible d'importer phase1_fundamental: %r", e)
        log.debug("Trace:\n%s", traceback.format_exc())
        return None

def list_funcs(P1) -> List[str]:
    if P1 is None:
        return []
    attrs = dir(P1)
    funcs = [a for a in attrs if callable(getattr(P1, a))]
    log.info("Fonctions detectées dans phase1_fundamental: %s", ", ".join(sorted(funcs)))
    return funcs

# ------ Appels robustes & nettoyage pairs -----------------------------------

def try_call(P1, func_name: str, *args, **kwargs):
    """Appelle P1.func_name si dispo. Retourne (ok, result/None)."""
    if P1 is None:
        return False, None
    fn = getattr(P1, func_name, None)
    if not callable(fn):
        return False, None
    try:
        log.info("→ Appel %s(%s %s)", func_name, args, kwargs if kwargs else "")
        out = fn(*args, **kwargs)
        log.info("← %s OK", func_name)
        return True, out
    except Exception as e:
        log.error("× %s a échoué: %r", func_name, e)
        log.debug("Trace:\n%s", traceback.format_exc())
        return True, None  # fonction existait mais a échoué

def _smart_call(P1, func_name: str, ctx: Dict[str, Any], extra: Dict[str, Any] | None = None):
    """
    Appelle P1.func_name en construisant des kwargs à partir des noms de paramètres.
    Évite les erreurs d'ordre d'arguments et passe seulement ce qui est attendu.
    """
    if P1 is None:
        return False, None
    fn = getattr(P1, func_name, None)
    if not callable(fn):
        return False, None

    bank = {
        "ticker": ctx.get("ticker"),
        "info": ctx.get("info"),
        "fundamentals": ctx.get("fundamentals"),
        "prices": ctx.get("prices"),
        "beta": ctx.get("beta"),
        "peer_set": ctx.get("peer_set"),
        "peers": ctx.get("peers_df") or ctx.get("peers"),
        "peers_df": ctx.get("peers_df") or ctx.get("peers"),
        "fv_cmp": ctx.get("fv_cmp"),
        "dcf_res": ctx.get("dcf_res"),
    }
    if extra:
        bank.update(extra)

    try:
        sig = inspect.signature(fn)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name in bank and bank[name] is not None:
                kwargs[name] = bank[name]
        log.info("→ Appel %s(**%s)", func_name, {k: type(v).__name__ for k, v in kwargs.items()})
        out = fn(**kwargs)
        log.info("← %s OK", func_name)
        return True, out
    except Exception as e:
        log.error("× %s a échoué: %r", func_name, e)
        log.debug("Trace:\n%s", traceback.format_exc())
        return True, None

def _is_all_nan_row(row, metric_cols: list[str]) -> bool:
    for c in metric_cols:
        v = row.get(c)
        if v is None:
            continue
        try:
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                return False
        except Exception:
            return False
    return True

def clean_peers_df(peers_df, subject_ticker: str):
    """Supprime les tickers obviously invalides et les lignes sans métriques utilisables."""
    try:
        import pandas as pd
        if not isinstance(peers_df, pd.DataFrame) or peers_df.empty:
            return peers_df, {"dropped_invalid": 0, "dropped_all_nan": 0}

        df = peers_df.copy()
        if "ticker" not in df.columns:
            for alt in ["symbol", "Symbols", "tickers"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "ticker"})
                    break

        before = len(df)
        df = df[df["ticker"].astype(str).str.len() > 1]
        df = df[df["ticker"].astype(str).str.strip() != "."]
        df = df[df["ticker"].notna()]
        df = df[df["ticker"].astype(str).str.upper() != str(subject_ticker).upper()]
        dropped_invalid = before - len(df)

        metric_cols = [c for c in ["trailingPE", "forwardPE", "ps_ttm", "ev_ebitda"] if c in df.columns]
        if metric_cols:
            mask_all_nan = df[metric_cols].apply(lambda r: all(pd.isna(r.values)), axis=1)
            dropped_all_nan = int(mask_all_nan.sum())
            df = df[~mask_all_nan]
        else:
            dropped_all_nan = 0

        df["__nonnull"] = df[metric_cols].notna().sum(axis=1) if metric_cols else 0
        df = df.sort_values("__nonnull", ascending=False).drop(columns="__nonnull", errors="ignore")

        return df, {"dropped_invalid": int(dropped_invalid), "dropped_all_nan": int(dropped_all_nan)}
    except Exception:
        log.debug("clean_peers_df: fallback (pas Pandas ?)")
        return peers_df, {"dropped_invalid": 0, "dropped_all_nan": 0}

# ------ Sauvegarde intelligente ---------------------------------------------

def save_any(name: str, outdir: str, obj: Any) -> None:
    """Sauve intelligemment dict/df/liste."""
    if obj is None:
        return
    try:
        import pandas as pd
    except Exception:
        pd = None

    path_base = os.path.join(outdir, name)
    if isinstance(obj, dict):
        try:
            dump_json(obj, path_base + ".json")
        except TypeError:
            # dict contenant DataFrames -> on bascule en .txt repr()
            with open(path_base + ".txt", "w", encoding="utf-8") as f:
                f.write(repr(obj))
    elif pd is not None and isinstance(obj, pd.DataFrame):
        dump_df(obj, path_base + ".csv")
    elif isinstance(obj, list):
        dump_json(obj, path_base + ".json")
    else:
        try:
            dump_json(obj, path_base + ".json")
        except Exception:
            with open(path_base + ".txt", "w", encoding="utf-8") as f:
                f.write(repr(obj))

# ------ Fallback yfinance (connectivité réelle) ------------------------------

def probe_yfinance(ticker: str, outdir: str) -> Dict[str, Any]:
    """
    Récupère des blocs essentiels avec yfinance pour s'assurer que:
     - le réseau fonctionne
     - les états financiers sont récupérables
    """
    import pandas as pd
    import yfinance as yf

    t = yf.Ticker(ticker)
    pack: Dict[str, Any] = {"ticker": ticker}

    # INFO/PROFILE
    try:
        info = t.fast_info  # plus robuste que .info
        pack["fast_info"] = dict(info) if info else {}
        dump_json(pack["fast_info"], os.path.join(outdir, "yf_fast_info.json"))
        log.info("yfinance.fast_info: %d clés", len(pack["fast_info"]))
    except Exception as e:
        log.warning("fast_info KO: %r", e)

    # PRICE HISTORY (3y daily)
    try:
        hist = t.history(period="3y", auto_adjust=True)
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        pack["history_shape"] = list(hist.shape)
        dump_df(hist, os.path.join(outdir, "yf_history_3y.csv"))
        log.info("history(3y): %s", hist.shape)
    except Exception as e:
        log.error("history KO: %r", e)

    # FINANCIALS (annual & quarterly)
    fin_keys = [
        ("financials", "yf_income_annual.csv"),
        ("quarterly_financials", "yf_income_quarterly.csv"),
        ("balance_sheet", "yf_balance_annual.csv"),
        ("quarterly_balance_sheet", "yf_balance_quarterly.csv"),
        ("cashflow", "yf_cashflow_annual.csv"),
        ("quarterly_cashflow", "yf_cashflow_quarterly.csv"),
        ("earnings", "yf_earnings_annual.csv"),
        ("quarterly_earnings", "yf_earnings_quarterly.csv"),
    ]
    for attr, fname in fin_keys:
        try:
            df = getattr(t, attr)
            import pandas as pd  # ensure in scope
            if isinstance(df, pd.DataFrame) and not df.empty:
                dump_df(df, os.path.join(outdir, fname))
                log.info("%s: %s", attr, df.shape)
            else:
                log.warning("%s: vide", attr)
        except Exception as e:
            log.error("%s KO: %r", attr, e)

    return pack

# ------ Runner par ticker ----------------------------------------------------

ESSENTIAL_MIN_ROWS = 5  # seuil minimal "raisonnable" pour juger non vide

def run_phase1_live_for_ticker(ticker: str, root_out: str, P1) -> int:
    outdir = os.path.join(root_out, ticker.replace("/", "_"))
    ensure_dir(outdir)
    log.info("=== PHASE1 LIVE • %s ===", ticker)

    _ = list_funcs(P1)  # log des fonctions disponibles

    # Contexte partagé entre appels
    called_any = False
    ctx: Dict[str, Any] = {"ticker": ticker}

    # 1) Appel haut niveau principal si dispo
    ok, result = try_call(P1, "build_fundamental_view", ticker)
    if ok:
        called_any = True
        save_any("p1_build_fundamental_view", outdir, result)
        try:
            if isinstance(result, dict):
                for k in ["info", "fundamentals", "peers_df", "peers", "dcf", "fair_value", "fast_info"]:
                    if k in result:
                        ctx[k if k != "peers" else "peers_df"] = result[k]
                log.info("build_fundamental_view keys: %s", list(result.keys()))
        except Exception:
            pass

    # 2) Charges de base
    ok, res = _smart_call(P1, "load_info", ctx)
    if ok and res is not None:
        called_any = True
        ctx["info"] = res
        save_any("p1_load_info", outdir, res)

    ok, res = _smart_call(P1, "load_fundamentals", ctx)
    if ok and res is not None:
        called_any = True
        ctx["fundamentals"] = res
        save_any("p1_load_fundamentals", outdir, res)
        # sanity et sauvegarde des sous-DF si présents
        if isinstance(res, dict):
            for key in ["income_stmt", "balance_sheet", "cash_flow"]:
                if key in res:
                    save_any(f"p1_fund_{key}", outdir, res[key])
                else:
                    log.warning("fundamentals['%s'] manquant", key)

    ok, res = _smart_call(P1, "load_prices", ctx)
    if ok and res is not None:
        called_any = True
        ctx["prices"] = res
        save_any("p1_load_prices", outdir, res)

    ok, res = _smart_call(P1, "estimate_beta", ctx)
    if ok and res is not None:
        called_any = True
        ctx["beta"] = res
        save_any("p1_estimate_beta", outdir, res)

    # 2.1 Peers
    ok, res = _smart_call(P1, "build_peer_set", ctx)
    if ok and res is not None:
        called_any = True
        ctx["peer_set"] = res
        save_any("p1_build_peer_set", outdir, res)

    ok, res = _smart_call(P1, "fetch_peer_multiples", ctx)
    if ok and res is not None:
        called_any = True
        ctx["peers_df_raw"] = res
        cleaned, stats = clean_peers_df(res, ticker)
        ctx["peers_df"] = cleaned
        save_any("p1_fetch_peer_multiples_raw", outdir, res)
        save_any("p1_fetch_peer_multiples_clean", outdir, cleaned)
        shape = getattr(cleaned, "shape", ("?", "?"))
        log.info("Peers nettoyés: -invalid=%s -all_nan=%s ; restants=%s", stats["dropped_invalid"], stats["dropped_all_nan"], shape)

    # 2.2 Multiples / Santé / Zscores
    ok, res = _smart_call(P1, "compute_company_multiples", ctx)
    if ok and res is not None:
        called_any = True
        ctx["company_multiples"] = res
        save_any("p1_compute_company_multiples", outdir, res)

    ok, res = _smart_call(P1, "compute_health_ratios", ctx)
    if ok and res is not None:
        called_any = True
        ctx["health"] = res
        save_any("p1_compute_health_ratios", outdir, res)

    ok, res = _smart_call(P1, "compute_zscores_company_vs_peers", ctx)
    if ok and res is not None:
        called_any = True
        ctx["zscores"] = res
        save_any("p1_compute_zscores_company_vs_peers", outdir, res)

    # 2.3 DCF, Comparables, Agrégat
    ok, res = _smart_call(P1, "dcf_simplified", ctx)
    if ok and res is not None:
        called_any = True
        ctx["dcf_res"] = res
        save_any("p1_dcf_simplified", outdir, res)

    ok, res = _smart_call(P1, "fair_value_from_comparables", ctx)
    if ok and res is not None:
        called_any = True
        ctx["fv_cmp"] = res
        save_any("p1_fair_value_from_comparables", outdir, res)

    ok, res = _smart_call(P1, "aggregate_fair_value", ctx)
    if ok and res is not None:
        called_any = True
        ctx["fair_value"] = res
        save_any("p1_aggregate_fair_value", outdir, res)

    # 3) Fallback/connectivité: yfinance direct
    yf_pack = probe_yfinance(ticker, outdir)

    # 4) Sanity: vérifie qu’on a des données essentielles non vides
    import pandas as pd
    essentials_ok = False

    hist_csv = os.path.join(outdir, "yf_history_3y.csv")
    fin_candidates = [
        "yf_income_annual.csv",
        "yf_balance_annual.csv",
        "yf_cashflow_annual.csv",
        "yf_earnings_annual.csv",
    ]
    fin_ok = False
    for fn in fin_candidates:
        p = os.path.join(outdir, fn)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, index_col=0)
                if len(df) >= 1:
                    fin_ok = True
                    break
            except Exception:
                pass

    hist_ok = False
    if os.path.exists(hist_csv):
        try:
            dfh = pd.read_csv(hist_csv, index_col=0, parse_dates=True)
            hist_ok = len(dfh) >= ESSENTIAL_MIN_ROWS
        except Exception:
            hist_ok = False

    essentials_ok = hist_ok and fin_ok

    # 5) Résumé lisible
    summary = {
        "ticker": ticker,
        "called_phase1_functions": called_any,
        "history_ok": hist_ok,
        "financials_ok": fin_ok,
        "output_dir": outdir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(summary, os.path.join(outdir, "summary.json"))
    log.info("Résumé: %s", json.dumps(summary, indent=2, ensure_ascii=False))

    # Code retour: 0 si OK, 2 si essentiel manquant
    return 0 if essentials_ok else 2

# ------ Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Test live Phase 1 (sans mock)")
    ap.add_argument("--ticker", nargs="+", required=True, help="Un ou plusieurs tickers (ex: NGD.TO AEM.TO)")
    ap.add_argument("--log", default="INFO", help="Niveau de log (DEBUG, INFO, WARNING)")
    ap.add_argument("--outdir", default="artifacts_phase1", help="Répertoire de sortie")
    args = ap.parse_args()

    setup_logging(args.log)
    log_env()
    ensure_dir(args.outdir)

    P1 = import_phase1()

    exit_code = 0
    for tk in args.ticker:
        try:
            rc = run_phase1_live_for_ticker(tk, args.outdir, P1)
            exit_code = max(exit_code, rc)
        except KeyboardInterrupt:
            log.warning("Interrompu par l’utilisateur")
            return 130
        except Exception as e:
            log.error("Échec sur %s: %r", tk, e)
            log.debug("Trace:\n%s", traceback.format_exc())
            exit_code = max(exit_code, 1)

    if exit_code == 0:
        log.info("✅ Phase 1 OK pour tous les tickers.")
    elif exit_code == 2:
        log.error("❌ Données essentielles manquantes (history/financials) pour au moins un ticker.")
    else:
        log.error("❌ Des erreurs sont survenues.")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()