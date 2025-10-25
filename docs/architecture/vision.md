# Vision Architecture Cible

Objectifs
- Réutiliser les modules existants et clarifier le découpage en couches.
- Offrir 5 vues: Macro, News, Deep Dive Action, Prévisions/Backtest, Observabilité.
 - Opérations lisibles sans orchestrateur central (Makefile/cron) ; UI lit les dernières partitions.

Couches
- Domain: `src/core/models.py`, `src/core/io_utils.py`, `src/core/market_data.py`, `src/core/config.py`.
- Application: orchestrations `src/analytics/*` (market_intel, phase2_technical, phase3_macro, econ_llm_agent) et use-cases.
- Adapters: Ingestion `src/ingestion/*`, Data sources `src/analytics/data_sources/*`, UI `src/apps/*`, `src/utils/st_ui.py`.
 - Ops: scripts `scripts/*.py`, cibles `Makefile`, et pages Admin « status ».

Contraintes & Standards
- Contrats de données explicites (Pydantic dans models.py) entre couches.
- I/O isolés; pas d’accès réseau direct depuis UI.
- Caching au bord des adapters, invalidation TTL.
 - Partitions immuables: `data/<domaine>/dt=YYYYMMDD/` (Parquet/JSON), idempotence par date.
 - UI ne suggère pas d’exécuter des scripts; ces conseils vivent dans docs/Admin.

Pages UI (Streamlit)
- Top‑nav sticky + footer (port 5555 visible), séparation claire « Prévisions » vs « Administration ».
- Dashboard Macro (FRED, GSCPI, GPR) avec comparaisons temporelles.
- News & Sentiment (Tavily/Firecrawl + résumés) avec états vides explicites.
- Stock Deep Dive (fondamentaux, technique, pairs) — JSON détaillé sous expander.
- Prévisions & Backtest (marché/secteurs/titres, résultats + incertitudes) ; exports CSV.
- Observabilité (logs récents, santé des APIs, latence) — pas d’exposition de noms de clés sensibles.
- Pages dépendant d’une date (Régimes/Risque/Récession/Mémos): sélecteur de date en page (fallback = dernière partition).

Données & Intégrations
- FRED via MCP + `src/analytics/data_sources/*` internes.
- Finnhub (si clés) pour quotes/fondamentaux/News en temps réel.
- Crawlers pour documents/actualités (Tavily/Firecrawl/Serper).
 - SearXNG local pour recherche robuste: `ops/web/searxng-local` (port 8082).

Sécurité & Posture MCP
- MCP activés: browser (@playwright/mcp), architecture analyzer, filesystem, memory, mermaid, sqlite, spec‑workflow, taskmanager, FRED, Finnhub, Serper/Tavily/Firecrawl.
- Désactiver les MCP non officiels non indispensables; pins de versions et timeouts.
- UI smoke automatisé (Playwright) pour vérifier navbar/footer, pages clés, codes 200.
