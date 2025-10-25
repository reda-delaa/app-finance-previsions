# AGENTS.md — Instructions centrales pour les agents

Résumé
- Tous les agents (ingestion, qualité, prévision, LLM, UI) doivent se conformer au prompt cadre : `docs/AGENTS_PROMPT.md`.
- Le journal d’avancement et les prochaines étapes sont dans `docs/PROGRESS.md`.
- Les commandes de lancement et d’orchestration sont dans le `Makefile`.

Bonnes pratiques (obligatoires)
- Après chaque petit ajout/patch: `git add` → `git commit -m "…"` → `git push`.
- Mettre à jour `docs/PROGRESS.md` (fait/ajouts récents/prochaines étapes/how-to-run).
- Ne pas dupliquer un agent existant; étendre l’existant.
- Idempotence: écrire sous `data/.../dt=YYYYMMDD/` sans écraser les historiques.

Points d’entrée clés
- Prompt cadre: `docs/AGENTS_PROMPT.md`
- Avancement: `docs/PROGRESS.md`
- App UI: `src/apps/agent_app.py` (multi-pages)
- Qualité: `src/agents/data_quality.py` + page Alerts + Settings
- Modèles LLM: `src/agents/g4f_model_watcher.py` + page Scoreboard
- Prévisions fusion: `scripts/fuse_forecasts.py` → `data/forecast/dt=*/final.parquet`

Makefile (extraits utiles)
- `make factory-run` — pipeline unitaire (harvester → refresh models → llm-agents → macro regime → fuse)
- `make g4f-refresh` / `make g4f-refresh-official` / `make g4f-probe-api` / `make g4f-fetch-official`
- `make risk-monitor` / `make memos` / `make backfill-prices`

Si vous ne savez pas quoi faire
- Lisez `docs/AGENTS_PROMPT.md`, puis proposez de petites étapes utiles pour un investisseur débutant (texte simple, exports CSV/JSON, pages UI dédiées, qualité data et 5 ans d’historique).

