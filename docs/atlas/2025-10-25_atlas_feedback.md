# ATLAS — Feedback UI & Code (capturé)

Source: Atlas Navigator (accès UI + code via GitHub)
Date de capture: $(date +%F)

## Vérification du dépôt

Le dépôt `app-finance-previsions` comporte une structure claire : on y trouve un dossier `src/agents` contenant les agents actuels (harvester, qualité, calendrier des résultats, watcher LLM, mémo d’investissement, régime macro, récession et surveillance des risques). Le fichier `AGENTS.md` rappelle les instructions centrales : suivre le prompt cadre `docs/AGENTS_PROMPT.md`, tenir à jour le journal d’avancement dans `docs/PROGRESS.md`, ne pas dupliquer un agent existant, écrire les sorties dans des sous‑répertoires datés (`data/.../dt=YYYYMMDD/`), et passer par les scripts/Makefile pour l’orchestration.

## Suggestions d’évolution du code selon la vision

1) Garder le socle existant sans duplication
- Les modules actuels couvrent déjà ingestion (`data_harvester.py`), qualité (`data_quality.py`), mémos (`investment_memo_agent.py`), régimes macro (`macro_regime_agent.py`), récession (`recession_agent.py`), risque (`risk_monitor_agent.py`), watcher LLM (`g4f_model_watcher.py`), calendrier des résultats (`earnings_calendar_agent.py`). Avant d’ajouter, vérifier non‑duplication.

2) Nouveaux agents spécialisés (F4F)
- `equity_forecast_agent.py`: distributions rendements/volatilité (1j/1s/1m), ≥5 ans d’historique → `data/forecast/dt=*/equities.parquet`.
- `commodity_forecast_agent.py`: matières premières (or, pétrole…), variables macro (offre/demande/dollar) + 5 ans.
- `macro_forecast_agent.py`: prévisions macro (croissance, inflation, taux), étendre `macro_regime_agent.py` si possible.
- `forecast_aggregator_agent.py`: agrège (Équité/Commodités/Macro + ML existants), score confiance, écrit `final.parquet` (éviter de modifier `scripts/fuse_forecasts.py`).
- `update_monitor_agent.py`: vérifie fraîcheur + déclenche ingestion/backfill pour ≥5 ans.
- `explanation_memo_agent.py`: synthèses FR (scénarios/risques/drivers) pour Memos/Notes.
- `backtest_agent.py`: stratégies simples (top‑N, tilt), écrit sous `data/backtests/dt=*`.
- `evaluation_agent.py`: MAE/RMSE/couverture, historique de performance, alimente page Évaluation.
- `sentiment_agent.py`: score sentiment par actif/thème (news/transcriptions), distinct des modules actuels.
- Règle: idempotence par `dt=YYYYMMDD`, documenter dans `docs/PROGRESS.md`.

3) Pipeline & orchestration
- Facteur de couverture 5 ans: s’assurer que `make backfill-prices` est déclenché par `update_monitor_agent.py` et contrôle trous/dups (via `data_quality.py`).
- Lien agents↔UI: pour chaque agent, page/section lisant ses sorties dans « Analyse & Prévisions » (ex. « Prévisions actions » filtrable, ou consolidation Dashboard).
- UI sans shell: retirer invites « Lancez scripts/... »; déclencher via agents/Makefile hors UI.

4) Qualité & sécurité
- Masquer secrets: pas de noms exacts de clés dans les sorties; identifiants génériques (« Clé API A: ✅ »).
- Config centralisée: watchlist/seuils/presets via Settings/`config.yaml` accessible par `src/core/config.py`.
- Journalisation: chaque agent logue début/fin, fichiers écrits, anomalies → `logs/` (Observability).

5) Documentation
- Continuer à mettre à jour `docs/PROGRESS.md` et `docs/AGENTS_PROMPT.md`; ajouter une section décrivant les nouveaux agents (rôle, I/O, modèle F4F).
- Ajouter README dans `src/agents` (architecture/extension du système).

## Analyse UI et proposition d’organisation globale

Deux espaces:
- « Analyse & Prévisions » (investisseur): Dashboard, Deep‑Dive/Recherche instrument, Régimes & Risques, Backtests & Évaluation, Mémos & Notes, Alertes.
- « Usine & Administration »: Observabilité & Qualité (fusion), Gestion des agents (inclut LLM Scoreboard), Settings & Watchlist, Mise à jour des données.

Principes UX
- FR unifié (labels/dates), clarté (éviter JSON brut; expander « Voir JSON »), feedback (success/error après actions), sélecteur de date en haut des pages dépendantes, états vides utiles, sécurité (pas d’instruction shell; masquer noms de clés).

Nouveaux agents F4F (récap)
- Prévision Équité, Prévision Commodités, Prévision Macro/Récession, Agrégateur/Arbitre, Qualité des données, Mises à jour & Historique, Explications & Mémos, Backtest & Simulation, Évaluation de modèles, Sentiment & Actualités.

Maintien des données
- Marché ≥5 ans (actions/indices/commodities/crypto), Macro clés, Historique d’événements (earnings/FOMC), News/transcriptions. Un agent Qualité garantit complétude; un agent Mise à jour notifie et déclenche ingestion.

Conclusion
- Recentrer UI (Analyse vs Usine), uniformiser FR/ergonomie, ajouter agents F4F spécialisés. L’app devient un assistant de portefeuille: prévisions probabilistes, explications claires, surveillance continue de la qualité, historique suffisant.
