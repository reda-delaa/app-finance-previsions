# --- G4F verified models integration (uses g4f-working) ----------------------
# Dépendances: pip install g4f requests (ou urllib si tu préfères)
import os, re, json, time, urllib.request
from typing import List, Dict, Optional

try:
    import g4f
except Exception as e:
    raise RuntimeError("g4f non installé: pip install -U g4f") from e

G4F_MODELS_TXT = "https://raw.githubusercontent.com/maruf009sultan/g4f-working/main/working/models.txt"
G4F_TEST_JSON  = "https://raw.githubusercontent.com/maruf009sultan/g4f-working/main/working/test_results.json"
G4F_CACHE_PATH = os.path.join(".cache", "g4f_verified_cache.json")

SOTA_PAT = re.compile(
    r"(deepseek[-_ ]?(r1|v3|prover))|"
    r"(qwen[-_/ ]?3.*(235b|next|480b|coder))|"
    r"(llama[-_ ]?3\.?3.*70b)|"
    r"(glm[-_ ]?4\.?5)|"
    r"(phi[-_ ]?4(\b|-)|phi[-_ ]?4[-_ ]?reasoning)|"
    r"(gemma[-_ ]?3.*(27b|12b))|"
    r"(kimi|moonshot)|"
    r"(hermes[-_ ]?3.*405b)|"
    r"(qwq[-_ ]?32b)|"
    r"(devstral|llama[-_ ]?4[-_ ]?(maverick|scout))",
    flags=re.I
)

PREFERRED_CHAIN = ["OpenRouter", "DeepInfra", "Liaobots", "You", "Phind", "Meta", "Google", "Replicate", "PollinationsAI"]

def _http_get(url: str, timeout=20) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

def _ensure_cache_dir():
    d = os.path.dirname(G4F_CACHE_PATH)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_verified_from_cache(max_age_sec=3600) -> Optional[dict]:
    if not os.path.exists(G4F_CACHE_PATH):
        return None
    st = os.stat(G4F_CACHE_PATH)
    if time.time() - st.st_mtime > max_age_sec:
        return None
    with open(G4F_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_verified_to_cache(data: dict):
    _ensure_cache_dir()
    with open(G4F_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_models_txt(text: str) -> List[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", line)
        if m:
            name, caps = m.group(1).strip(), [c.strip().lower() for c in m.group(2).split(",")]
            out.append({"model": name, "caps": caps})
        else:
            out.append({"model": line, "caps": ["text"]})
    return out

def load_verified_models(refresh=False) -> dict:
    if not refresh:
        cache = load_verified_from_cache()
        if cache: 
            return cache
    models_txt = _http_get(G4F_MODELS_TXT)
    test_json  = _http_get(G4F_TEST_JSON)
    data = {
        "models": parse_models_txt(models_txt),
        "tests": json.loads(test_json)
    }
    save_verified_to_cache(data)
    return data

def _pass_rate(tests: dict, model: str) -> Optional[float]:
    x = tests.get(model) or {}
    p, f = x.get("pass", 0), x.get("fail", 0)
    tot = p + f
    return (p / tot) if tot else None

def select_verified_models(caps_need=("text",), min_pass=0.30, only_sota=True, limit=30, refresh=False) -> List[dict]:
    src = load_verified_models(refresh=refresh)
    models = src["models"]
    tests  = src["tests"]
    need = set([c.lower() for c in caps_need or []])

    sel = []
    for it in models:
        caps = set(it.get("caps", []))
        if need and not need.issubset(caps):
            continue
        name = it["model"]
        if only_sota and not SOTA_PAT.search(name):
            continue
        pr = _pass_rate(tests, name)
        if (pr is None) or (pr >= min_pass):
            prov_hint = name.split(":")[0] if ":" in name else None
            sel.append({"model": name, "caps": sorted(caps), "pass_rate": pr, "hint": prov_hint})

    # unique + tri (fiabilité décroissante, puis alpha)
    uniq = {x["model"]: x for x in sel}
    out = list(uniq.values())
    out.sort(key=lambda x: (-(x["pass_rate"] or 0), x["model"].lower()))
    return out[:limit]

def provider_candidates_for(model_name: str, hint: Optional[str]) -> List[str]:
    if hint:
        first = [hint]
    else:
        first = []
    low = model_name.lower()
    # quelques directs
    if "pollinationsai:" in low:
        first = ["PollinationsAI"]
    # compléter
    chain = first + [p for p in PREFERRED_CHAIN if p not in first]
    return chain

def g4f_chat_once(provider: str, prompt: str, model: Optional[str]=None, system: Optional[str]=None, temperature: float=0.2, max_tokens: int=2048, timeout: int=45) -> Optional[str]:
    """
    Appel simple g4f → renvoie str ou None si échec.
    NB: suivant la version g4f, la signature peut varier (client.ChatCompletion.create vs client.chat.completions.create).
    Adapte si besoin à ta version exacte.
    """
    try:
        # API "haute compat" de g4f (évolue parfois)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # g4f.Client() (v2) style:
        client = g4f.Client()
        kwargs = dict(
            model=model or "auto",
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            provider=provider
        )
        # certains providers n’acceptent pas provider=… → on essaie, sinon fallback
        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            # ancienne signature (sans 'provider') ?
            kwargs.pop("provider", None)
            resp = client.chat.completions.create(**kwargs)

        # normaliser la sortie
        if hasattr(resp, "choices") and resp.choices:
            return resp.choices[0].message.content
        if isinstance(resp, dict):
            ch = resp.get("choices") or []
            if ch and ch[0].get("message"):
                return ch[0]["message"].get("content")
        return None
    except Exception:
        return None

def llm_ask(prompt: str, system: Optional[str]=None, caps="text", min_pass=0.30, only_sota=True, refresh=False, temperature=0.2, max_tokens=2048, tries_per_model=2, providers_per_model=4) -> dict:
    """
    Sélectionne un modèle 'verified' et tente des providers en cascade.
    Retourne {model, provider, pass_rate, text} (ou text=None si tout a échoué).
    """
    models = select_verified_models(caps_need=(caps,), min_pass=min_pass, only_sota=only_sota, limit=40, refresh=refresh)
    for m in models:
        provs = provider_candidates_for(m["model"], m["hint"])
        used = []
        for prov in provs[:providers_per_model]:
            used.append(prov)
            for _ in range(tries_per_model):
                out = g4f_chat_once(provider=prov, prompt=prompt, model=m["model"], system=system, temperature=temperature, max_tokens=max_tokens)
                if out and isinstance(out, str) and out.strip():
                    return {
                        "model": m["model"],
                        "provider": prov,
                        "pass_rate": m["pass_rate"],
                        "text": out.strip()
                    }
    return {"model": None, "provider": None, "pass_rate": None, "text": None}
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Petit exemple CLI rapide
    q = "Explique en 5 puces la différence entre DeepSeek-R1 et Qwen3-235B pour du raisonnement financier."
    ans = llm_ask(q, system="Tu es un analyste quant rigoureux.", caps="text", min_pass=0.30, only_sota=True, refresh=False)
    print("[MODEL]", ans["model"])
    print("[PROVIDER]", ans["provider"])
    print("[PASS_RATE]", ans["pass_rate"])
    print("\n", ans["text"] or "(aucune réponse)")

# ======================= TOP-5 VERIFIED MODELS — TEST SUITE =======================
import time
import argparse

def ask_with_specific_model(model_name: str, prompt: str, system: str = None,
                            temperature: float = 0.2, max_tokens: int = 1024,
                            providers_per_model: int = 4, tries_per_model: int = 2,
                            timeout: int = 45) -> dict:
    """Force l’usage d’un modèle précis, avec cascade de providers."""
    provs = provider_candidates_for(model_name, hint=model_name.split(":")[0] if ":" in model_name else None)
    messages = {
        "finance": prompt,
    }
    for prov in provs[:providers_per_model]:
        for _ in range(tries_per_model):
            t0 = time.time()
            out = g4f_chat_once(provider=prov, prompt=prompt, model=model_name,
                                 system=system, temperature=temperature,
                                 max_tokens=max_tokens, timeout=timeout)
            dt = time.time() - t0
            if out and isinstance(out, str) and out.strip():
                return {
                    "ok": True,
                    "model": model_name,
                    "provider": prov,
                    "latency_s": round(dt, 2),
                    "text": out.strip()
                }
    return {"ok": False, "model": model_name, "provider": None, "latency_s": None, "text": None}


def run_top5_tests(refresh=False, min_pass=0.30, only_sota=True):
    print("\n[+] Sélection des 5 meilleurs modèles ‘verified’...")
    top5 = select_verified_models(caps_need=("text",), min_pass=min_pass, only_sota=only_sota, limit=5, refresh=refresh)
    if not top5:
        print("[-] Aucun modèle ‘verified’ trouvé (vérifie la connexion ou baisse min_pass).")
        return

    print("\n[Top-5]")
    for i, m in enumerate(top5, 1):
        print(f"  {i}. {m['model']}  (pass_rate={None if m['pass_rate'] is None else round(m['pass_rate']*100,1)}%)")

    # ---------- 5 PROMPTS COMPLÉMENTAIRES ----------
    prompts = {
        "TEST1_finance": {
            "system": "Tu es un analyste sell-side expérimenté, factuel et concis.",
            "prompt": (
                "Rédige un mémo d’investissement en 5 puces sur Orange S.A. (ORA). "
                "Inclure: (1) thèse, (2) catalyseurs 3-6 mois, (3) risques clés, "
                "(4) valorisation rapide (multiples comparables), (5) point d’entrée technique (niveaux). "
                "Réponse en FR, puces courtes."
            ),
            "temperature": 0.2,
            "max_tokens": 500
        },
        "TEST2_code_event_study": {
            "system": "Tu es un data scientist financier. Tu écris du code Python clair et robuste.",
            "prompt": (
                "Écris une fonction Python `event_study_abnormal_returns(prices, market, events, pre=5, post=5)` "
                "qui calcule les rendements anormaux (modèle de marché simple), et renvoie un dict avec: "
                "`car_by_event` (CAR par événement), `avg_ar` (AR moyens par jour relatif), `avg_car` (CAR moyens). "
                "Utilise uniquement numpy/pandas. Garde le code autonome (imports inclus) et ajoute un court exemple d’appel."
            ),
            "temperature": 0.1,
            "max_tokens": 900
        },
        "TEST3_reasoning_math": {
            "system": "Tu es un assistant de raisonnement rigoureux. Montre tes étapes brièvement.",
            "prompt": (
                "Un portefeuille a deux actifs A et B. E[R_A]=8%, E[R_B]=12%, Var(A)=0.04, Var(B)=0.09, "
                "Cov(A,B)=0.012. Quel poids w (dans A) minimise la variance du portefeuille? Donne w, Var_min et E[R] "
                "correspondant. Réponse concise avec 3 lignes: w, Var_min, E[R]."
            ),
            "temperature": 0.0,
            "max_tokens": 300
        },
        "TEST4_translation_summary": {
            "system": "Tu es un traducteur financier. Tu résumes en FR de manière fidèle.",
            "prompt": (
                "Traduis et résume en 4 puces cette note anglaise: "
                "\"IEA slashes clean hydrogen outlook as costs rise and projects stall; regulatory uncertainty and demand "
                "signals remain weak, risking underinvestment through 2030 despite strategic importance for heavy industry.\""
            ),
            "temperature": 0.2,
            "max_tokens": 300
        },
        "TEST5_ie_extraction": {
            "system": "Tu es un extracteur d’information financière (schema JSON compact).",
            "prompt": (
                "Extrait en JSON compact {event_type, issuer, tickers[], region, impact, horizon} depuis: "
                "\"Ukrainian drones hit Russia’s Primorsk oil port; Brent jumps to $68; US pushing secondary sanctions on "
                "India/China; near-term supply risks rise.\" Retiens des valeurs simples (impact ∈ {bearish,bullish,neutral})."
            ),
            "temperature": 0.1,
            "max_tokens": 200
        }
    }

    print("\n[+] Exécution des 5 tests sur chaque modèle…\n")
    results = []
    for m in top5:
        name = m["model"]
        pr = None if m["pass_rate"] is None else round(m["pass_rate"]*100, 1)
        print(f"=== {name}  (pass_rate={pr}%) ===")
        for test_name, cfg in prompts.items():
            print(f"\n-- {test_name} --")
            t0 = time.time()
            resp = ask_with_specific_model(
                model_name=name,
                prompt=cfg["prompt"],
                system=cfg["system"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                providers_per_model=4,
                tries_per_model=2,
                timeout=60
            )
            dt = time.time() - t0
            ok = "OK" if resp["ok"] else "FAIL"
            provider = resp["provider"] or "-"
            print(f"[{ok}] provider={provider}  latency={resp['latency_s']}s  elapsed={round(dt,2)}s")
            if resp["text"]:
                # affiche un extrait pour éviter des floods en console
                snippet = resp["text"][:800]
                print(snippet + ("\n...[troncature]..." if len(resp["text"]) > 800 else ""))
            results.append({
                "model": name,
                "pass_rate": pr,
                "test": test_name,
                "ok": resp["ok"],
                "provider": provider,
                "latency_s": resp["latency_s"],
                "excerpt": (resp["text"][:200] if resp["text"] else None)
            })
        print("\n")

    # résumé
    print("\n==================== RÉSUMÉ ====================")
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], {"ok":0, "total":0})
        by_model[r["model"]]["total"] += 1
        by_model[r["model"]]["ok"] += int(r["ok"])
    for model, agg in by_model.items():
        rate = 100.0 * agg["ok"] / max(1, agg["total"])
        print(f"{model}: {agg['ok']}/{agg['total']} tests OK  → {rate:.0f}%")

    # option: écrire un log jsonl
    try:
        os.makedirs("artifacts_g4f", exist_ok=True)
        with open("artifacts_g4f/top5_test_results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("\n[+] Log écrit: artifacts_g4f/top5_test_results.jsonl")
    except Exception as e:
        print(f"[!] Écriture log échouée: {e}")


# ----- CLI hook -----
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-top5", action="store_true", help="Lance la batterie de tests sur les 5 meilleurs modèles ‘verified’.")
    p.add_argument("--refresh", action="store_true", help="Ignore le cache local et refetch la liste verified.")
    p.add_argument("--min-pass", type=float, default=0.30, help="Seuil minimal de pass_rate.")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.test_top5:
        run_top5_tests(refresh=args.refresh, min_pass=args.min_pass, only_sota=True)
    else:
        # Comportement existant éventuel… ou exemple rapide:
        demo = llm_ask(
            "Donne 3 risques macro pour l’Europe Q4 (puces courtes).",
            system="Réponds en FR, style télégraphique.",
            caps="text", min_pass=0.30, only_sota=True, refresh=False,
            temperature=0.2, max_tokens=256
        )
        print("[DEMO MODEL]", demo["model"])
        print("[DEMO PROVIDER]", demo["provider"])
        print("\n", demo["text"] or "(aucune réponse)")
# =============================================================================== 