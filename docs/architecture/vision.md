# Vision Architecture Cible

Objectifs
- Réutiliser les modules existants et clarifier le découpage en couches.
- Offrir 5 vues: Macro, News, Deep Dive Action, Prévisions/Backtest, Observabilité.

Couches
- Domain: `src/core/models.py`, `src/core/io_utils.py`, `src/core/market_data.py`, `src/core/config.py`.
- Application: orchestrations `src/analytics/*` (market_intel, phase2_technical, phase3_macro, econ_llm_agent) et use-cases.
- Adapters: Ingestion `src/ingestion/*`, Data sources `src/analytics/data_sources/*`, UI `src/apps/*`, `src/utils/st_ui.py`.

Contraintes & Standards
- Contrats de données explicites (Pydantic dans models.py) entre couches.
- I/O isolés; pas d’accès réseau direct depuis UI.
- Caching au bord des adapters, invalidation TTL.

Pages UI (Streamlit)
- Dashboard Macro (FRED, GSCPI, GPR) avec comparaisons temporelles.
- News & Sentiment (Tavily/Firecrawl + résumés).
- Stock Deep Dive (fondamentaux, technique, pairs).
- Prévisions & Backtest (marché/secteurs/titres, résultats + incertitudes).
- Observabilité (logs récents, santé des APIs, latence, clés manquantes).

Données & Intégrations
- FRED via MCP + `src/analytics/data_sources/*` internes.
- Finnhub (si clés) pour quotes/fondamentaux/News en temps réel.
- Crawlers pour documents/actualités (Tavily/Firecrawl/Serper).
