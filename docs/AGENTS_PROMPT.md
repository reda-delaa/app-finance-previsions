Agents Prompt (System/Project)

Objectif
- Construire et opérer une « usine d’agents IA » (texte uniquement) pour un investisseur débutant.
- Répondre aux cas d’usage: une action précise, une commodité (ex: or), ou l’économie (ex: risque de récession).
- Maintenir des données fraîches et un historique ≥ 5 ans pour des prévisions robustes.

Principes
- gpt4free (texte), sélection dynamique: DeepSeek R1/V3 → Qwen3 235B Thinking → GLM‑4.5 → Llama‑3.3‑70B → gpt‑oss‑120b.
- Robustesse data: fraîcheur, couverture ≥ 5 ans, JSON/Parquet propres (NaN/Inf nettoyés), schémas stables.
- UX débutant: langage simple, éviter le jargon, tooltips/explications courtes, exports CSV/JSON.
- Anti‑duplication: ne pas recréer un agent/module existant — enrichir l’existant.
- Transparence: expliquer « pourquoi » (drivers, signaux, risques, limites).
- Itération continue: si pas d’idée, se mettre à la place d’un investisseur débutant et proposer ce qui manque.

Rôles d’agents (usine)
- Ingestion (news/macro/prices/fundamentals): maj Parquet; watchlist override; cadence configurable.
- Backfill 5 ans (prices/macro clés): garantit la couverture historique.
- Qualité Data: scanne schémas/NaN/vide/fraîcheur/couverture; écrit rapport JSON; seuils en settings.
- Rapports de Problèmes: lit la qualité et produit des actions/corrections priorisées.
- Fix Planner: propose un plan de correction (patchs/scripts), sans dupliquer.
- QA Exécution: exécute tests/scripts; rapporte échecs/régressions; propose corrections.
- Architecture: analyse structure/dépendances; recommande refactors ciblés; identifie risques techniques.
- Modèles LLM: watcher (sources verified/official), probe (latence/ok), Scoreboard (usage/accord/latence/source).
- Prévision: baseline (rule+ML) + Fusion (rule/ML/LLM consensus); Ensemble LLM + Arbitre (Investigations/Topics/Memos).
- Macro: Macro Regime, Risk Monitor, (à venir) Recession Risk (probabilité & drivers).
- Calendriers: événements macro (CPI/NFP/FOMC), (à venir) earnings watchlist.
- Orchestration sans orchestrator central: Makefile/cron; UI status via dernières partitions/JSON.
- What Changed: compare partitions (regime/risk/topN/brief) et synthétise les deltas.

Exigences data
- Fraîcheur: news/forecasts récentes; qualité signale news>3j et forecasts>2j.
- Couverture: prix ≥ 5y pour tickers; macro clés (CPI, DGS10, DGS2) ≥ 5y — qualité signale les manques.
- Stockage: Parquet (dt=YYYYMMDD) + JSON propre; watchlist via env ou data/watchlist.json.

Politique LLM
- gpt4free uniquement (texte).
- Watcher: G4F_SOURCE=official|verified|both; working.json: provider/latency/source.
- Scoreboard: usages, avg_agreement, provider, latency, source; exports CSV.
- Ordre reasoning‑first et ensemble + arbitre pour analyses multi-modèles.

UX & sorties
- UI Streamlit: Dashboard (Final Top‑N, régime, 90j metrics, mini-alertes), Signals (poids Rule/ML/LLM + CSV), Portfolio (tilt macro presets, rebalance simulator, exports), Regimes, Risk, Memos, Alerts, Watchlist, Changes, Settings, Notes, Scoreboard.
- Exports CSV/JSON sur les tables clés; langage simple, explications brèves.

Discipline Git & Docs (obligatoire)
- Après chaque ajout/patch: git add → git commit (message clair) → git push.
- Mettre à jour docs/PROGRESS.md (fait/ajouts récents/next/how-to-run).
- Toujours redémarrer l’UI après un changement de l’interface: `make ui-restart` (port canonique 5555). Ne jamais laisser plusieurs instances Streamlit tourner sur des ports différents.
- Effectuer des commits atomiques après chaque ajout/correction visible (message explicite). En cas de doute, commit avant changement risqué.
- Documenter les nouvelles cibles Makefile et scripts.
- Ne pas casser les cibles existantes (factory-run, refresh, probe, fuse, risk, memos, backfill).
- Idempotence: écrire sous data/.../dt=YYYYMMDD/ sans écraser l’historique.

Validation & Qualité
- Après modification: exécuter la cible la plus proche (make backfill-prices, make fuse-forecasts, make risk-monitor…), vérifier que les pages UI s’ouvrent et exportent.
- Qualité: lancer le scanner; corriger NaN/Inf; vérifier couverture ≥ 5 ans pour tickers/macro clés.
- Ajouter une UI minimaliste si une nouvelle sortie n’a pas de page.

Orchestration
- Makefile/cron = panneau de contrôle:
  - factory-run: harvester → refresh models → llm-agents → macro regime → fuse.
  - g4f-refresh / g4f-refresh-official / g4f-probe-api / g4f-fetch-official
  - risk-monitor / memos / backfill-prices
- UI « status » = pages lisant les dernières partitions/JSON; badge d’alertes sur le Dashboard.
 - Ne pas utiliser `orch/orchestrator.py` dans l’exécution; préférer les scripts séquentiels et Makefile.

Règles d’évolution
- Ne pas dupliquer un agent; étendre l’existant.
- Si pas d’idée: se mettre à la place d’un investisseur débutant:
  - Action: “Que regarder aujourd’hui ?” → Top‑N, régimes, risques, changements clés.
  - Titre: “Est‑ce intéressant ?” → Memo clair (thèse, catalyseurs, risques).
  - Commodité: “Et l’or ?” → Lens: dollar/taux réels/momentum/miners.
  - Macro: “Risque de récession ?” → Probabilité, drivers, limites.

Critères d’acceptation (exemples)
- Données: couverture ≥ 5 ans; fraîcheur OK; JSON propre.
- Prévisions: final.parquet publié; UI Signals/Portfolio lisent et exportent.
- Macro: Regimes + Risk lisibles; Changes montre les deltas.
- LLM: working.json à jour; Scoreboard visible (uses/avg_agreement/provider/latency/source).
- Docs: PROGRESS.md à jour; cibles Makefile documentées.

Sorties attendues
- Fichiers sous data/<domaine>/dt=YYYYMMDD/*.{json,parquet}; exports UI (CSV/JSON).
- Commits fréquents; PROGRESS.md à jour; code/UI robustes pour un investisseur débutant; pas de duplication d’agents.
