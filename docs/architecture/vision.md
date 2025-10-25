# Vision Architecture Cible

Objectifs
- Réutiliser les modules existants et clarifier le découpage en couches.
- Offrir 5 vues: Macro, News, Deep Dive Action, Prévisions/Backtest, Observabilité.
 - Opérations lisibles sans orchestrateur central (Makefile/cron) ; UI lit les dernières partitions.

Couches
- Domain: `src/core/models.py`, `src/core/io_utils.py`, `src/core/market_data.py`, `src/core/config.py`.
- Application: orchestrations `src/analytics/*` (market_intel, phase2_technical, phase3_macro, econ_llm_agent) et use-cases.
- Adapters: Ingestion `src/ingestion/*`, Data sources `src/analytics/data_sources/*`.
- UI (actuelle): Dash `src/dash_app/*`.
- UI (ancienne/legacy): Streamlit `src/apps/*`, utilitaires `src/utils/st_ui.py`.
- Ops: scripts `scripts/*.py`, cibles `Makefile`, et pages Admin « status ».

Contraintes & Standards
- Contrats de données explicites (Pydantic dans models.py) entre couches.
- I/O isolés; pas d’accès réseau direct depuis UI.
- Caching au bord des adapters, invalidation TTL.
 - Partitions immuables: `data/<domaine>/dt=YYYYMMDD/` (Parquet/JSON), idempotence par date.
 - UI ne suggère pas d’exécuter des scripts; ces conseils vivent dans docs/Admin.

Pages UI (Dash — actuelle)
- Sidebar Analyse/Admin, thème Bootstrap sombre (Cyborg), navigation multipage.
- Dashboard Macro, Signals, Portfolio, Observability (équivalents migrés depuis Streamlit).
- États vides FR cohérents; sélecteur de date in‑page pour pages partitionnées.
- Pas de prompts d’exécution de scripts dans les pages (docs/Admin seulement).

Ancienne UI (Streamlit — legacy)
- Conservée uniquement pour référence durant la migration. Aucune nouvelle fonctionnalité n’y sera implémentée.
- Scripts `scripts/ui_*` et docs peuvent aider à la maintenance/validation ponctuelle.

Données & Intégrations
- FRED via MCP + `src/analytics/data_sources/*` internes.
- Finnhub (si clés) pour quotes/fondamentaux/News en temps réel.
- Crawlers pour documents/actualités (Tavily/Firecrawl/Serper).
 - SearXNG local pour recherche robuste: `ops/web/searxng-local` (port 8082).

Sécurité & Posture MCP
- MCP activés: browser (@playwright/mcp), architecture analyzer, filesystem, memory, mermaid, sqlite, spec‑workflow, taskmanager, FRED, Finnhub, Serper/Tavily/Firecrawl.
- Désactiver les MCP non officiels non indispensables; pins de versions et timeouts.
- UI smoke automatisé (Playwright) pour vérifier navbar/footer, pages clés, codes 200.
