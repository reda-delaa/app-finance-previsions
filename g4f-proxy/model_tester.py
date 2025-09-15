# -*- coding: utf-8 -*-
"""
model_tester.py — Scanner parallèle des providers/modèles g4f réellement fonctionnels
- Conçu pour s'intégrer au proxy Flask existant.
- Lancement on-demand via endpoint /v1/g4f/scan
- Résultats consultables via /v1/working-models
- Cache JSON local pour éviter de rescanner tout le temps

ENV (optionnels):
  G4F_TEST_PROMPT       : prompt court à utiliser pour tester (défaut: "ping")
  G4F_TEST_TIMEOUT      : timeout par appel (sec, défaut: 6)
  G4F_TEST_PARALLEL     : nb. workers (défaut: 12)
  G4F_TEST_SAMPLE       : nb. max de candidats à tester (0 = tous)
  G4F_LIVE_CACHE        : chemin du cache (défaut: .g4f_live.json)
  G4F_VARIANTS_PER_CAND : nb. de variantes à tenter par candidat (défaut: 4)
"""

from __future__ import annotations
import os
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Iterable, Optional

import g4f

DEFAULT_TEST_PROMPT = os.getenv("G4F_TEST_PROMPT", "ping")
DEFAULT_TIMEOUT = float(os.getenv("G4F_TEST_TIMEOUT", "6"))
DEFAULT_PARALLEL = int(os.getenv("G4F_TEST_PARALLEL", "12"))
DEFAULT_SAMPLE = int(os.getenv("G4F_TEST_SAMPLE", "0"))  # 0 = tous
LIVE_CACHE = os.getenv("G4F_LIVE_CACHE", ".g4f_live.json")
VARIANTS_PER_CAND = int(os.getenv("G4F_VARIANTS_PER_CAND", "4"))

def _now() -> int:
    return int(time.time())

def _candidate_variants(candidate: str) -> List[str]:
    """
    Reproduit la logique du proxy: on tente plusieurs écritures du même modèle.
    """
    # Cas "Provider:Model"
    if ":" in candidate:
        provider, model = candidate.split(":", 1)
        base = [
            f"{provider}:{model}",
            model,
        ]
    else:
        # modèle brut → différentes graphies communes
        base = [
            candidate,
            candidate.replace("/", "-"),
            candidate.split("/")[-1],
        ]
    # Un extra "deepseek-v3" utile comme variante générique
    if candidate.lower().startswith(("deepseek", "deepseek-ai/")) and "deepseek-v3" not in base:
        base.append("deepseek-v3")
    # Limite configurable
    out = []
    for v in base:
        if v not in out:
            out.append(v)
        if len(out) >= VARIANTS_PER_CAND:
            break
    return out

def _try_one_variant(variant: str, prompt: str, timeout: float) -> Tuple[bool, str]:
    """
    Teste 1 variant précis. Retourne (ok, message) où message contient le provider/model effectif testé ou l'erreur.
    """
    try:
        # essai en mode "messages"
        try:
            gen = g4f.ChatCompletion.create(
                model=variant,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                timeout=timeout,  # certains backends honorent 'timeout'
            )
            txt = str(gen)
            if txt and "gated by API key" not in txt.lower():
                return True, f"OK(messages) {variant}"
        except Exception as e1:
            # fallback prompt unique
            gen = g4f.ChatCompletion.create(
                model=variant,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                timeout=timeout,
            )
            txt = str(gen)
            if txt and "gated by API key" not in txt.lower():
                return True, f"OK(prompt) {variant}"
            return False, f"fail(prompt) {variant}: {txt[:120]}"
    except Exception as e:
        return False, f"EXC {variant}: {e}"

def _test_one_candidate(candidate: str, prompt: str, timeout: float) -> Dict[str, Any]:
    """
    Essaie plusieurs variants pour un candidat donné.
    Renvoie un dict avec la réussite et la meilleure forme.
    """
    for variant in _candidate_variants(candidate):
        ok, msg = _try_one_variant(variant, prompt, timeout)
        if ok:
            return {"candidate": candidate, "variant": variant, "ok": True, "detail": msg}
    return {"candidate": candidate, "variant": None, "ok": False, "detail": "all variants failed or gated"}

def scan_candidates(candidates: List[str],
                    prompt: str = DEFAULT_TEST_PROMPT,
                    timeout: float = DEFAULT_TIMEOUT,
                    max_workers: int = DEFAULT_PARALLEL,
                    sample: int = DEFAULT_SAMPLE) -> Dict[str, Any]:
    """
    Lance un scan parallèle sur la liste 'candidates'.
    """
    started_at = _now()
    if sample and sample > 0:
        to_test = candidates[:sample]
    else:
        to_test = list(candidates)

    results: List[Dict[str, Any]] = []
    ok_models: List[Dict[str, str]] = []

    # Exécution parallèle
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for cand in to_test:
            futures[ex.submit(_test_one_candidate, cand, prompt, timeout)] = cand

        for fut in as_completed(futures):
            cand = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"candidate": cand, "ok": False, "detail": f"EXCEPTION: {e}"}
            results.append(res)
            if res.get("ok"):
                ok_models.append({"candidate": res["candidate"], "variant": res["variant"]})

    duration = _now() - started_at
    summary = {
        "ts": started_at,
        "duration_sec": duration,
        "total_tested": len(to_test),
        "ok_count": len(ok_models),
        "ok": ok_models,
        "results": results,
        "prompt": prompt,
        "timeout": timeout,
        "max_workers": max_workers,
        "sample": sample,
    }

    # Sauvegarde cache
    try:
        with open(LIVE_CACHE, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return summary

def load_last_scan() -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(LIVE_CACHE):
            with open(LIVE_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None
