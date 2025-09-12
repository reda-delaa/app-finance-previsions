# phase5_fusion.py
# -*- coding: utf-8 -*-
"""
Phase 5 — Fusion, Scoring Global & Orchestration

Assemble les sorties des Phases 1→4 pour produire:
- Score global (0..100)
- Scores par pilier: Fondamental / Technique / Macro / Sentiment
- Ajustement de risque (volatilité, VaR, bêta, DD)
- Recommandation (Acheter / Neutre / Vendre) + drivers + flags
- Paquet "prêt UI": dictionnaire + DataFrames optionnels

Licence: MIT (adapter à votre projet)
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

# ============ Imports Phases 1→4 (avec fallbacks doux) ============

# PHASE 1 — Data helpers (tu peux mapper vers tes fonctions existantes)
try:
    import phase1_data as P1
    HAS_P1 = True
except Exception:
    HAS_P1 = False

# PHASE 2 — Fondamental / Facteurs / Multiples
try:
    import phase2_factor_models as P2
    HAS_P2 = True
except Exception:
    HAS_P2 = False

# PHASE 3 — Macro / Régimes / Élasticités
try:
    import phase3_macro as P3
    HAS_P3 = True
except Exception:
    HAS_P3 = False

# PHASE 4 — Sentiment / News
try:
    import phase4_sentiment as P4
    HAS_P4 = True
except Exception:
    HAS_P4 = False


# ===================== Dataclasses de sortie ======================

@dataclass
class PillarScores:
    fundamental: float     # 0..100
    technical: float       # 0..100
    macro: float           # 0..100
    sentiment: float       # 0..100

    def as_dict(self) -> Dict[str, float]:
        return {
            "fundamental": float(self.fundamental),
            "technical": float(self.technical),
            "macro": float(self.macro),
            "sentiment": float(self.sentiment),
        }


@dataclass
class RiskMetrics:
    vol_annual_pct: Optional[float] = None
    var95_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    beta_60d: Optional[float] = None

    def risk_malus(self) -> float:
        """
        Convertit le profil de risque en malus (-20..0).
        - Volatilité élevée, VaR négative importante, DD profond, bêta > 1.3 ⇒ baisse de score
        """
        penalty = 0.0
        # Vol annualisée
        if self.vol_annual_pct is not None:
            if self.vol_annual_pct > 60:
                penalty -= 12
            elif self.vol_annual_pct > 40:
                penalty -= 8
            elif self.vol_annual_pct > 30:
                penalty -= 5
        # VaR 95% (valeur négative = perte)
        if self.var95_pct is not None:
            # var95_pct arrive souvent en %, ex: -4.5
            if self.var95_pct < -5:
                penalty -= 6
            elif self.var95_pct < -3.5:
                penalty -= 4
        # Max Drawdown
        if self.max_drawdown_pct is not None:
            if self.max_drawdown_pct < -55:
                penalty -= 8
            elif self.max_drawdown_pct < -35:
                penalty -= 5
        # Bêta
        if self.beta_60d is not None:
            if abs(self.beta_60d) > 1.6:
                penalty -= 6
            elif abs(self.beta_60d) > 1.3:
                penalty -= 4
        return float(np.clip(penalty, -20, 0))

    def flags(self) -> List[str]:
        f = []
        if self.vol_annual_pct and self.vol_annual_pct > 40:
            f.append(f"Volatilité élevée ({self.vol_annual_pct:.1f}%)")
        if self.var95_pct is not None and self.var95_pct < -3.5:
            f.append(f"VaR(95) défavorable ({self.var95_pct:.1f}%)")
        if self.max_drawdown_pct is not None and self.max_drawdown_pct < -35:
            f.append(f"Drawdown historique profond ({self.max_drawdown_pct:.1f}%)")
        if self.beta_60d is not None and abs(self.beta_60d) > 1.3:
            f.append(f"Bêta élevé ({self.beta_60d:.2f})")
        return f


@dataclass
class FusionOutput:
    ticker: str
    horizon_label: str
    total_score: float              # 0..100
    pillar_scores: PillarScores
    risk: RiskMetrics
    recommendation: str            # "Acheter", "Neutre", "Vendre"
    drivers: List[str]
    risk_flags: List[str]
    diagnostics: Dict[str, Any]    # détails par phase, pour UI
    # DataFrames optionnels (None si non fournis)
    df_perf: Optional[pd.DataFrame] = None
    df_macro: Optional[pd.DataFrame] = None
    df_sent_day: Optional[pd.DataFrame] = None
    df_sent_week: Optional[pd.DataFrame] = None

    def to_json(self, compact: bool = True) -> str:
        base = {
            "ticker": self.ticker,
            "horizon_label": self.horizon_label,
            "total_score": float(self.total_score),
            "pillar_scores": self.pillar_scores.as_dict(),
            "risk": asdict(self.risk),
            "recommendation": self.recommendation,
            "drivers": list(self.drivers),
            "risk_flags": list(self.risk_flags),
            "diagnostics": self.diagnostics,
        }
        if compact:
            # n’embarque pas les DF volumineux
            return json.dumps(base, ensure_ascii=False, indent=2)
        # sinon: ajoute tailles DF pour debugging
        base["_df_shapes"] = {
            "df_perf": None if self.df_perf is None else list(self.df_perf.shape),
            "df_macro": None if self.df_macro is None else list(self.df_macro.shape),
            "df_sent_day": None if self.df_sent_day is None else list(self.df_sent_day.shape),
            "df_sent_week": None if self.df_sent_week is None else list(self.df_sent_week.shape),
        }
        return json.dumps(base, ensure_ascii=False, indent=2)


# ====================== Normalisations utilitaires ======================

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _scale_to_0_100(x: float) -> float:
    return float(np.clip(x * 100.0, 0.0, 100.0))

def _z_to_0_100(z: float, lo: float = -2.0, hi: float = 2.0) -> float:
    # map z in [lo,hi] to [0,100]
    zc = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    return float(100.0 * zc)

def _pct_to_score(p: float, good_high: bool = True, ptiles: Tuple[float, float] = (0.2, 0.8)) -> float:
    """
    Convertit un pourcentage (ex: marge, ROE, etc.) en score 0..100 via seuils heuristiques.
    good_high: True si une valeur plus haute est meilleure, False sinon (ex: leverage).
    """
    if np.isnan(p):
        return 50.0
    lo, hi = ptiles
    # heuristique simple
    if good_high:
        if p <= lo * 100: return 20.0
        if p >= hi * 100: return 90.0
        # interpolation
        return 20.0 + (p/100.0 - lo) / (hi - lo) * 70.0
    else:
        # plus bas = mieux
        if p <= lo * 100: return 90.0
        if p >= hi * 100: return 20.0
        return 90.0 - (p/100.0 - lo) / (hi - lo) * 70.0


# ====================== Fusions partielles par pilier ======================

def fuse_fundamental(p2_payload: Optional[Dict[str, Any]]) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Agrège un score fondamental sur 0..100.
    Attend en entrée la sortie de phase2_factor_models (ou dict similaire):
      - ratios (ROE, marge nette, leverage, growth EPS/Rev)
      - valuation (PE, EV/EBITDA, P/S) vs. pairs / secteur
      - qualité (FCF, marge, stabilité)
    """
    drivers = []
    diag = {}
    if not HAS_P2 or not p2_payload:
        return 55.0, ["Fallback fondamental (données limitées)"], {"note": "phase2 manquante ou payload vide"}

    try:
        ratios = p2_payload.get("ratios", {})
        peers = p2_payload.get("peer_comps", {})
        quality = p2_payload.get("quality", {})

        # Exemples: convertis en scores simples
        sc_profit = np.nanmean([
            _pct_to_score(ratios.get("net_margin_pct", np.nan), good_high=True),
            _pct_to_score(ratios.get("roe_pct", np.nan), good_high=True),
            _pct_to_score(ratios.get("roa_pct", np.nan), good_high=True),
        ])

        sc_growth = np.nanmean([
            _pct_to_score(ratios.get("eps_cagr_3y_pct", np.nan), good_high=True),
            _pct_to_score(ratios.get("rev_cagr_3y_pct", np.nan), good_high=True),
        ])

        # valorisation: plus bas (vs pairs) = mieux
        sc_val = np.nanmean([
            _pct_to_score(peers.get("pe_rel_sector_pct", np.nan), good_high=False),
            _pct_to_score(peers.get("ev_ebitda_rel_sector_pct", np.nan), good_high=False),
            _pct_to_score(peers.get("psales_rel_sector_pct", np.nan), good_high=False),
        ])

        sc_quality = np.nanmean([
            _pct_to_score(quality.get("fcf_margin_pct", np.nan), good_high=True),
            _pct_to_score(quality.get("gross_margin_stability", np.nan), good_high=True),
            _pct_to_score(quality.get("net_debt_to_ebitda", np.nan), good_high=False),
        ])

        parts = [sc_profit, sc_growth, sc_val, sc_quality]
        sc = float(np.nanmean(parts))
        if np.isnan(sc): sc = 55.0

        # Drivers
        if sc_profit >= 65: drivers.append("Rentabilité solide")
        if sc_growth >= 65: drivers.append("Croissance attractive")
        if sc_val >= 65: drivers.append("Valorisation attractive vs pairs")
        if sc_quality >= 65: drivers.append("Qualité/FCF favorables")
        if not drivers: drivers.append("Fondamentaux mitigés")

        diag = {"profit": sc_profit, "growth": sc_growth, "value": sc_val, "quality": sc_quality}
        return sc, drivers, diag
    except Exception as e:
        return 55.0, [f"Fallback fondamental ({e})"], {"error": str(e)}


def fuse_technical(p1_payload: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Utilise les signaux CT/MT déjà présents dans ton app (RSI/MACD/SMA/Breakout…).
    p1_payload attend:
      - short_sig: {"score": -1..+1, "signals": {...}}
      - med_sig:   {"score": -1..+1, "signals": {...}}
      - regime: "Bull/Bear/Range"
    """
    drivers = []
    diag = {}
    try:
        short_sig = p1_payload.get("short_sig", {}) or {}
        med_sig = p1_payload.get("med_sig", {}) or {}
        regime = p1_payload.get("regime", "Range")

        ct = float(short_sig.get("score", 0.0))
        mt = float(med_sig.get("score", 0.0))

        # Map -1..+1 → 0..100 (50 = neutre)
        def s_map(x: float) -> float:
            return float(50.0 + 50.0 * np.clip(x, -1, 1))

        sc_ct = s_map(ct)
        sc_mt = s_map(mt)

        # léger bonus/malus selon régime
        reg_bonus = {"Bull": +5, "Bear": -5, "Range": 0}.get(regime, 0)

        sc = float(np.clip(0.6 * sc_mt + 0.4 * sc_ct + reg_bonus, 0, 100))

        # Drivers
        # collecter 2-4 signaux directionnels forts
        for k, v in (short_sig.get("signals", {}) or {}).items():
            if abs(v) >= 0.2:
                drivers.append(f"CT {k} {'+' if v>0 else '-'}")
        for k, v in (med_sig.get("signals", {}) or {}).items():
            if abs(v) >= 0.25:
                drivers.append(f"MT {k} {'+' if v>0 else '-'}")
        if regime:
            drivers.append(f"Régime: {regime}")
        drivers = drivers[:5] if drivers else ["Technique neutre"]

        diag = {"short": sc_ct, "medium": sc_mt, "regime_bonus": reg_bonus}
        return sc, drivers, diag
    except Exception as e:
        return 50.0, [f"Fallback technique ({e})"], {"error": str(e)}


def fuse_macro(p3_payload: Optional[Dict[str, Any]]) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Convertit l’état macro en score 0..100.
    p3_payload attendu (phase3_macro):
      - regime: {"label": "Goldilocks/Slowflation/Stagflation/Overheat", "zscore": ...}
      - factor_tailwinds: {"inflation": +/-, "rates": +/-, "usd": +/-, ...} sur -1..+1
      - sector_fit: score -1..+1 à +1 pro-secteur
      - nowcasts / surprises optionnels
    """
    drivers = []
    diag = {}
    if not HAS_P3 or not p3_payload:
        return 55.0, ["Fallback macro"], {"note": "phase3 manquante ou payload vide"}

    try:
        regime = p3_payload.get("regime", {}) or {}
        tail = p3_payload.get("factor_tailwinds", {}) or {}
        sector_fit = float(p3_payload.get("sector_fit", 0.0))

        # Score regime basé sur z-score normalisé
        reg_z = float(regime.get("zscore", 0.0))
        sc_reg = _z_to_0_100(reg_z, lo=-1.5, hi=1.5)

        # Vent arrière/face agrégé
        if tail:
            tv = np.mean([np.clip(float(v), -1, 1) for v in tail.values()])
        else:
            tv = 0.0
        sc_tail = float(50.0 + 35.0 * tv)  # -1..+1 → 15..85

        # Sector fit
        sc_sector = float(50.0 + 40.0 * np.clip(sector_fit, -1, 1))

        sc = float(np.clip(0.5 * sc_reg + 0.3 * sc_tail + 0.2 * sc_sector, 0, 100))

        # Drivers
        if regime.get("label"):
            drivers.append(f"Régime macro: {regime['label']}")
        if tv > 0.2: drivers.append("Vents macro favorables")
        if tv < -0.2: drivers.append("Vents macro défavorables")
        if sector_fit > 0.2: drivers.append("Secteur bien positionné")
        if sector_fit < -0.2: drivers.append("Secteur sous pression")
        if not drivers: drivers.append("Macro neutre")

        diag = {"regime_score": sc_reg, "tailwinds_score": sc_tail, "sector_fit_score": sc_sector, "tailwinds_raw": tail}
        return sc, drivers, diag
    except Exception as e:
        return 55.0, [f"Fallback macro ({e})"], {"error": str(e)}


def fuse_sentiment(p4_view: Optional[Dict[str, Any]]) -> Tuple[float, List[str], Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convertit la vue sentiment/news (phase4) en score 0..100.
    Utilise mean_sent_7d / 30d + shock/drift + risk flags.
    """
    drivers = []
    diag = {}
    df_day = None
    df_week = None
    if not HAS_P4 or not p4_view:
        return 55.0, ["Fallback sentiment"], {"note": "phase4 manquante ou vide"}, df_day, df_week

    try:
        s7 = p4_view.get("summary", {}).get("mean_sent_7d", np.nan)
        s30 = p4_view.get("summary", {}).get("mean_sent_30d", np.nan)
        shock = float(p4_view.get("summary", {}).get("shock_score", 0.0))
        drift = float(p4_view.get("summary", {}).get("drift_score", 0.0))
        signals = p4_view.get("signals", []) or []

        # sentiment HF/VADER est -1..+1 ⇒ map vers 0..100
        def smap(x):
            if x is None or np.isnan(x):
                return 50.0
            return float(50.0 + 50.0 * np.clip(x, -1, 1))

        sc_7 = smap(s7)
        sc_30 = smap(s30)
        sc_base = float(0.6 * sc_7 + 0.4 * sc_30)

        # bonus/malus shock & drift (petit poids)
        bonus = 0.0
        if shock > 1.5:
            bonus += 2.0 * np.clip(shock, 0, 3)  # max ~6
            drivers.append("News shock détecté")
        if drift > 0.1:
            bonus += 3.0
            drivers.append("Drift post-annonce positif")
        elif drift < -0.1:
            bonus -= 3.0
            drivers.append("Drift post-annonce négatif")

        sc = float(np.clip(sc_base + bonus, 0, 100))

        if signals:
            # garder 2-3 signaux lisibles
            drivers.extend(signals[:2])

        # DF pour UI si dispo
        df_day = p4_view.get("_df_by_day")
        df_week = p4_view.get("_df_by_week")

        diag = {"sent7": sc_7, "sent30": sc_30, "shock": shock, "drift": drift, "base": sc_base, "bonus": bonus}
        if not drivers:
            drivers = ["Sentiment neutre"]
        return sc, drivers, diag, df_day, df_week
    except Exception as e:
        return 55.0, [f"Fallback sentiment ({e})"], {"error": str(e)}, df_day, df_week


# ====================== Score global & Reco ======================

def combine_scores(pillars: PillarScores, risk: RiskMetrics,
                   weights: Dict[str, float] = None) -> Tuple[float, List[str]]:
    """
    Combine les scores par pilier + malus risque pour produire total (0..100).
    """
    if weights is None:
        # pondérations par défaut — tu peux ajuster
        weights = {"fundamental": 0.35, "technical": 0.30, "macro": 0.20, "sentiment": 0.15}
    wsum = sum(weights.values())
    if wsum <= 0:
        weights = {k: 1/4 for k in ["fundamental", "technical", "macro", "sentiment"]}

    raw = (weights["fundamental"] * pillars.fundamental +
           weights["technical"]   * pillars.technical +
           weights["macro"]       * pillars.macro +
           weights["sentiment"]   * pillars.sentiment) / sum(weights.values())

    malus = risk.risk_malus()  # -20..0
    total = float(np.clip(raw + malus, 0, 100))

    drivers = []
    # mise en avant des piliers dominants
    comp = [
        ("Fondamental", pillars.fundamental),
        ("Technique", pillars.technical),
        ("Macro", pillars.macro),
        ("Sentiment", pillars.sentiment),
    ]
    comp.sort(key=lambda x: x[1], reverse=True)
    for name, val in comp[:2]:
        if val >= 65:
            drivers.append(f"{name} favorable")
    # Si le meilleur pilier < 60, message mitigé
    if not drivers:
        drivers.append("Aucun pilier franchement dominant")

    # Si malus notable
    if malus <= -8:
        drivers.append("Profil de risque pénalisant")

    return total, drivers


def make_recommendation(total_score: float, horizon: str = "6–12m") -> str:
    """
    Règle simple — adapte à tes préférences.
    """
    if total_score >= 68:
        return f"Acheter (horizon {horizon})"
    if total_score >= 55:
        return f"Neutre / Conserver (horizon {horizon})"
    return f"Vendre / Éviter (horizon {horizon})"


# ====================== Orchestrateur principal ======================

def run_fusion(
    ticker: str,
    # --- payloads pré-calculés (si tu les as déjà dans l’app) :
    p1_payload: Optional[Dict[str, Any]] = None,  # doit contenir short_sig, med_sig, regime, risk_pack, etc.
    p2_payload: Optional[Dict[str, Any]] = None,
    p3_payload: Optional[Dict[str, Any]] = None,
    p4_view: Optional[Dict[str, Any]] = None,
    # --- sinon, calcule à la volée via phases si dispo :
    compute_missing_with_phases: bool = False,
    horizon_label: str = "6–12m",
) -> FusionOutput:
    """
    Assemble les 4 piliers + risque => score global & reco.
    Si compute_missing_with_phases=True et que certains payloads manquent,
    tente de les calculer via P1..P4 (si importés).
    """
    diag: Dict[str, Any] = {}

    # 1) Obtenir les payloads manquants si demandé
    if compute_missing_with_phases:
        # Phase 1 (technique + risque de base)
        if (p1_payload is None) and HAS_P1:
            try:
                # Exemples — mappe sur tes propres fonctions/objets
                stock_data = P1.get_stock_data(ticker, period="3y")
                bench = "^GSPTSE" if ticker.endswith(".TO") else "SPY"
                benchmark_data = P1.get_stock_data(bench, period="3y")
                with_ind = P1.add_technical_indicators(stock_data)
                short_sig = P1.compute_short_term_signals(with_ind)
                med_sig = P1.compute_medium_term_signals(with_ind, benchmark_data["Close"] if benchmark_data is not None else None)
                regime = P1.detect_regime(with_ind)
                risk = P1.risk_pack(stock_data["Close"], benchmark_data["Close"] if benchmark_data is not None else None)
                p1_payload = {
                    "short_sig": short_sig,
                    "med_sig": med_sig,
                    "regime": regime,
                    "risk": risk,
                    "stock_data": stock_data,
                }
            except Exception as e:
                diag["p1_error"] = str(e)

        # Phase 2 (fondamental)
        if (p2_payload is None) and HAS_P2:
            try:
                p2_payload = P2.build_fundamental_snapshot(ticker)  # à implémenter en phase 2
            except Exception as e:
                diag["p2_error"] = str(e)

        # Phase 3 (macro)
        if (p3_payload is None) and HAS_P3:
            try:
                p3_payload = P3.build_macro_view(ticker)  # à implémenter en phase 3
            except Exception as e:
                diag["p3_error"] = str(e)

        # Phase 4 (sentiment)
        if (p4_view is None) and HAS_P4:
            try:
                p4_view = P4.build_sentiment_view(ticker)
            except Exception as e:
                diag["p4_error"] = str(e)

    # 2) Scores par pilier
    fund_sc, fund_drv, fund_diag = fuse_fundamental(p2_payload)
    tech_sc, tech_drv, tech_diag = fuse_technical(p1_payload or {})
    macro_sc, macro_drv, macro_diag = fuse_macro(p3_payload)
    sent_sc, sent_drv, sent_diag, df_sent_day, df_sent_week = fuse_sentiment(p4_view)

    pillars = PillarScores(
        fundamental=fund_sc,
        technical=tech_sc,
        macro=macro_sc,
        sentiment=sent_sc,
    )

    # 3) Risque
    risk = RiskMetrics()
    if p1_payload and isinstance(p1_payload.get("risk", {}), dict):
        r = p1_payload["risk"]
        risk.vol_annual_pct = r.get("vol_annual_%")
        risk.var95_pct = r.get("VaR95_%")
        risk.max_drawdown_pct = r.get("max_drawdown_%")
        risk.beta_60d = r.get("beta_60d")

    # 4) Score global + reco
    total, score_drivers = combine_scores(pillars, risk)
    reco = make_recommendation(total, horizon=horizon_label)

    # 5) Drivers & flags
    drivers = []
    drivers.extend([f"[FOND] {d}" for d in fund_drv[:2]])
    drivers.extend([f"[TECH] {d}" for d in tech_drv[:2]])
    drivers.extend([f"[MACRO] {d}" for d in macro_drv[:2]])
    drivers.extend([f"[SENT] {d}" for d in sent_drv[:2]])
    drivers.extend([f"[SCORE] {d}" for d in score_drivers])
    # unique et court
    uniq = []
    for d in drivers:
        if d not in uniq:
            uniq.append(d)
    drivers = uniq[:8]

    risk_flags = risk.flags()
    if isinstance(p4_view, dict):
        rf = p4_view.get("aggregates", {}).get("risk_flags", [])
        if rf:
            risk_flags.extend([f"News: {x}" for x in rf])
    # unique
    runiq = []
    for f in risk_flags:
        if f not in runiq:
            runiq.append(f)
    risk_flags = runiq[:6]

    # 6) Diagnostics
    diag.update({
        "fundamental": fund_diag,
        "technical": tech_diag,
        "macro": macro_diag,
        "sentiment": sent_diag,
        "pillars": pillars.as_dict(),
        "total_after_risk": total,
    })

    # 7) DataFrames optionnels pour UI
    df_perf = None
    if p1_payload and isinstance(p1_payload.get("stock_data", None), pd.DataFrame):
        # Petite table de performance normalisée (base 100), utile pour graphique
        close = p1_payload["stock_data"]["Close"]
        if len(close) > 0:
            df_perf = pd.DataFrame({"Price": close / close.iloc[0] * 100.0})

    df_macro = None
    if HAS_P3 and isinstance(p3_payload, dict):
        # si phase3 renvoie un tableau exploitable (ex: nowcasts)
        maybe_df = p3_payload.get("_df_macro")
        if isinstance(maybe_df, pd.DataFrame):
            df_macro = maybe_df

    out = FusionOutput(
        ticker=ticker.upper(),
        horizon_label=horizon_label,
        total_score=total,
        pillar_scores=pillars,
        risk=risk,
        recommendation=reco,
        drivers=drivers,
        risk_flags=risk_flags,
        diagnostics=diag,
        df_perf=df_perf,
        df_macro=df_macro,
        df_sent_day=df_sent_day,
        df_sent_week=df_sent_week,
    )
    return out


# ====================== Exemple d’exécution ======================

if __name__ == "__main__":
    tk = "NGD.TO"
    # Exécution "auto" si phases dispos
    out = run_fusion(tk, compute_missing_with_phases=True)
    print(out.to_json())
    # Pour debugging: afficher les 8 drivers et flags
    print("\nDrivers:", " | ".join(out.drivers))
    print("Flags:", " | ".join(out.risk_flags))