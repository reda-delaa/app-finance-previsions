# sanity_runner.py
# -*- coding: utf-8 -*-
"""
Sanity Runner & Minimal Tests for Phases 1..5

Objectif:
- Importer les modules phase1_data, phase2_factor_models, phase3_macro, phase4_sentiment, phase5_fusion
- Exécuter un pipeline end-to-end (ou des mocks/degrafés si modules manquants)
- Logger les sorties & vérifier des invariants simples (smoke tests)
- Fournir quelques "unit tests" légers sans dépendre d'Internet (mode --offline)

Usage:
  python sanity_runner.py --ticker NGD.TO --log INFO
  python sanity_runner.py --ticker AAPL --offline
"""

from __future__ import annotations
import argparse
import logging
import sys
import json
import time
from typing import Any, Dict, Optional
from dataclasses import asdict, is_dataclass

# ===== Logging =====
def setup_logger(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

log = logging.getLogger("sanity")

# ===== Imports souples avec drapeaux =====
HAS_P1 = HAS_P2 = HAS_P3 = HAS_P4 = HAS_P5 = False

try:
    import phase1_data as P1
    HAS_P1 = True
except Exception as e:
    log.debug("Phase1 indisponible: %s", e)

try:
    import phase2_factor_models as P2
    HAS_P2 = True
except Exception as e:
    log.debug("Phase2 indisponible: %s", e)

try:
    import phase3_macro as P3
    HAS_P3 = True
except Exception as e:
    log.debug("Phase3 indisponible: %s", e)

try:
    import phase4_sentiment as P4
    HAS_P4 = True
except Exception as e:
    log.debug("Phase4 indisponible: %s", e)

try:
    import phase5_fusion as P5
    HAS_P5 = True
except Exception as e:
    log.debug("Phase5 indisponible: %s", e)


# ===== Mocks légers si une phase manque ou si --offline =====
def mock_phase1_payload(ticker: str) -> Dict[str, Any]:
    # Simule les signaux CT/MT + risk pack
    return {
        "short_sig": {"score": +0.2, "signals": {"RSI": +0.3, "MACD": +0.25, "Breakout20": +0.25, "Slope_20_50": +0.2}},
        "med_sig":   {"score": +0.6, "signals": {"Momentum": +0.5, "Above_SMA200": +0.25, "Excess_3m": +0.3}},
        "regime": "Bull",
        "risk": {"vol_annual_%": 45.0, "VaR95_%": -3.8, "max_drawdown_%": -42.0, "beta_60d": 1.35},
        "stock_data": None,  # on n’injecte pas de DataFrame en offline
    }

def mock_phase2_payload(ticker: str) -> Dict[str, Any]:
    # Simule fondamentaux: ratios, comparables, qualité
    return {
        "ratios": {"net_margin_pct": 12.5, "roe_pct": 16.0, "roa_pct": 7.5,
                   "eps_cagr_3y_pct": 9.0, "rev_cagr_3y_pct": 6.0},
        "peer_comps": {"pe_rel_sector_pct": 35.0, "ev_ebitda_rel_sector_pct": 40.0, "psales_rel_sector_pct": 45.0},
        "quality": {"fcf_margin_pct": 8.5, "gross_margin_stability": 75.0, "net_debt_to_ebitda": 1.8},
    }

def mock_phase3_payload(ticker: str) -> Dict[str, Any]:
    # Simule macro: régime + vents + sector fit
    return {
        "regime": {"label": "Goldilocks", "zscore": +0.6},
        "factor_tailwinds": {"inflation": +0.2, "rates": -0.1, "usd": -0.2, "commodities": +0.3},
        "sector_fit": +0.35,
    }

def mock_phase4_payload(ticker: str) -> Dict[str, Any]:
    # Simule sentiment agrége
    return {
        "summary": {"mean_sent_7d": +0.18, "mean_sent_30d": +0.05, "shock_score": 0.4, "drift_score": +0.12},
        "signals": ["Guidance positive", "Couverture analystes en hausse"],
        "aggregates": {"risk_flags": ["Litige mineur clos favorablement"]},
        "_df_by_day": None,
        "_df_by_week": None,
    }


# ===== Petits helpers =====
def dataclass_to_dict_safe(obj):
    if is_dataclass(obj):
        return asdict(obj)
    return obj


# ===== Smoke tests (fonctionnels) =====
def smoke_phase1(ticker: str, offline: bool) -> Dict[str, Any]:
    log.info("SMOKE P1: %s", "offline mock" if (offline or not HAS_P1) else "live")
    if offline or not HAS_P1:
        payload = mock_phase1_payload(ticker)
    else:
        # Essaye d’utiliser les fonctions de ta phase 1
        stock = P1.get_stock_data(ticker, period="1y")
        bench = "^GSPTSE" if ticker.endswith(".TO") else "SPY"
        bench_data = P1.get_stock_data(bench, period="1y")
        with_ind = P1.add_technical_indicators(stock)
        short_sig = P1.compute_short_term_signals(with_ind)
        med_sig = P1.compute_medium_term_signals(with_ind, bench_data["Close"] if bench_data is not None else None)
        regime = P1.detect_regime(with_ind)
        risk = P1.risk_pack(stock["Close"], bench_data["Close"] if bench_data is not None else None)
        payload = {
            "short_sig": short_sig,
            "med_sig": med_sig,
            "regime": regime,
            "risk": risk,
            "stock_data": stock,
        }
    log.debug("P1 payload: %s", json.dumps(payload, default=dataclass_to_dict_safe, ensure_ascii=False))
    # assertions minimales
    assert "short_sig" in payload and "med_sig" in payload and "regime" in payload
    return payload


def smoke_phase2(ticker: str, offline: bool) -> Dict[str, Any]:
    log.info("SMOKE P2: %s", "offline mock" if (offline or not HAS_P2) else "live")
    if offline or not HAS_P2:
        payload = mock_phase2_payload(ticker)
    else:
        payload = P2.build_fundamental_snapshot(ticker)
    # validations simples
    assert isinstance(payload, dict)
    assert "ratios" in payload
    return payload


def smoke_phase3(ticker: str, offline: bool) -> Dict[str, Any]:
    log.info("SMOKE P3: %s", "offline mock" if (offline or not HAS_P3) else "live")
    if offline or not HAS_P3:
        payload = mock_phase3_payload(ticker)
    else:
        payload = P3.build_macro_view(ticker)
    assert isinstance(payload, dict)
    assert "regime" in payload
    return payload


def smoke_phase4(ticker: str, offline: bool) -> Dict[str, Any]:
    log.info("SMOKE P4: %s", "offline mock" if (offline or not HAS_P4) else "live")
    if offline or not HAS_P4:
        payload = mock_phase4_payload(ticker)
    else:
        payload = P4.build_sentiment_view(ticker)
    assert isinstance(payload, dict)
    assert "summary" in payload
    return payload


def smoke_phase5(ticker: str, p1, p2, p3, p4) -> Any:
    if not HAS_P5:
        raise RuntimeError("Phase5 (fusion) introuvable — ajoute phase5_fusion.py")
    log.info("SMOKE P5: run_fusion(...)")
    out = P5.run_fusion(
        ticker=ticker,
        p1_payload=p1,
        p2_payload=p2,
        p3_payload=p3,
        p4_view=p4,
        compute_missing_with_phases=False,  # on fournit nos payloads déjà
    )
    # validation de base
    assert hasattr(out, "total_score")
    assert 0.0 <= float(out.total_score) <= 100.0
    assert hasattr(out, "pillar_scores")
    assert hasattr(out, "recommendation")
    log.debug("P5 output (compact): %s", out.to_json(compact=True))
    log.info("Score global: %.1f | Reco: %s", out.total_score, out.recommendation)
    return out


# ===== Micro “unit tests” (logiques) =====
def unit_tests_format(p1, p2, p3, p4, p5_out) -> None:
    log.info("UNIT: formats & bornes basiques")

    # P1: scores -1..+1
    for k in ("short_sig", "med_sig"):
        sc = p1.get(k, {}).get("score", 0.0)
        assert -1.001 <= float(sc) <= 1.001, f"P1 {k} score out of range: {sc}"

    # P2: ratios clés présents
    req_rat = ("net_margin_pct", "roe_pct", "rev_cagr_3y_pct")
    for r in req_rat:
        assert r in (p2.get("ratios", {}) or {}), f"P2 ratio manquant: {r}"

    # P3: regime label str + zscore
    reg = p3.get("regime", {})
    assert isinstance(reg.get("label", ""), str), "P3 regime label invalide"
    assert isinstance(reg.get("zscore", 0.0), (int, float)), "P3 regime zscore invalide"

    # P4: sentiment [-1,1]
    s7 = p4.get("summary", {}).get("mean_sent_7d", 0.0)
    s30 = p4.get("summary", {}).get("mean_sent_30d", 0.0)
    assert -1.001 <= float(s7) <= 1.001, f"P4 sent7 hors bornes {s7}"
    assert -1.001 <= float(s30) <= 1.001, f"P4 sent30 hors bornes {s30}"

    # P5: total & piliers
    tot = float(p5_out.total_score)
    assert 0.0 <= tot <= 100.0, f"P5 total_score hors bornes {tot}"
    ps = p5_out.pillar_scores.as_dict()
    for k, v in ps.items():
        assert 0.0 <= float(v) <= 100.0, f"P5 pillar {k} hors bornes {v}"

    log.info("UNIT: ✅ formats OK")


def unit_tests_behavior(p5_out) -> None:
    log.info("UNIT: règles de reco usuelles")
    tot = float(p5_out.total_score)
    reco = str(p5_out.recommendation).lower()
    if tot >= 68:
        assert "acheter" in reco, f"Reco attendue 'Acheter' pour score {tot}, obtenu: {reco}"
    elif tot >= 55:
        assert ("neutre" in reco) or ("conserver" in reco), f"Reco attendue 'Neutre/Conserver' (score {tot})"
    else:
        assert ("vendre" in reco) or ("éviter" in reco) or ("eviter" in reco), f"Reco attendue 'Vendre/Éviter' (score {tot})"
    log.info("UNIT: ✅ reco/score cohérents")


# ===== Runner principal =====
def main():
    parser = argparse.ArgumentParser(description="Sanity runner for phases 1..5")
    parser.add_argument("--ticker", type=str, default="NGD.TO", help="Symbole (ex: NGD.TO, AAPL)")
    parser.add_argument("--log", type=str, default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--offline", action="store_true", help="N’utilise aucun accès data; force les mocks")
    args = parser.parse_args()

    setup_logger(args.log)
    log.info("=== Sanity Runner (ticker=%s, offline=%s) ===", args.ticker, args.offline)

    t0 = time.time()
    try:
        p1 = smoke_phase1(args.ticker, args.offline)
        p2 = smoke_phase2(args.ticker, args.offline)
        p3 = smoke_phase3(args.ticker, args.offline)
        p4 = smoke_phase4(args.ticker, args.offline)
        p5_out = smoke_phase5(args.ticker, p1, p2, p3, p4)

        # Unit-like tests
        unit_tests_format(p1, p2, p3, p4, p5_out)
        unit_tests_behavior(p5_out)

        # Résumé final
        print("\n=== Résumé compact ===")
        print("Ticker:", p5_out.ticker)
        print("Score global:", f"{p5_out.total_score:.1f}/100")
        print("Reco:", p5_out.recommendation)
        print("Piliers:", json.dumps(p5_out.pillar_scores.as_dict(), ensure_ascii=False))
        print("Drivers:", " · ".join(p5_out.drivers))
        print("Risques:", " · ".join(p5_out.risk_flags))

        # JSON pour CI/CD
        print("\nJSON:", p5_out.to_json(compact=True))

        elapsed = time.time() - t0
        log.info("=== Terminé en %.2fs ===", elapsed)
        return 0
    except AssertionError as ae:
        log.error("Test/Assertion échouée: %s", ae)
        return 2
    except Exception as e:
        log.exception("Erreur fatale: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())