# Plan de Refactor (3 itérations)

Itération 1: Stabilisation & Observabilité
- Centraliser logging (src/hub/logging_setup.py), harmoniser warn_log (src/core/warn_log.py, src/utils/warn_log.py).
- Normaliser config (.env, src/core/config.py, secrets_local.py) et clés MCP.
- Ajouter tests de fumée runners (existants) aux CI locales.

Itération 2: Couches & Ingestion
- Introduire interfaces ingestion (quotes, fundamentals, macro, news) + impls basées sur modules existants (finviz, clients, FRED MCP).
- Déplacer logique “collée UI” dans application services.
- Contrats Pydantic pour inputs/outputs analytics.

Itération 3: UI Modulaire
- Pages Streamlit (dashboard macro, news, deep dive, forecasts/backtest, observabilité).
- Composants communs (tables, charts, selectors) dans src/utils/st_ui.py, st_compat.py.
- Caching sélectif sur chargements coûteux.

Livrables
- Docs C4 + backlog à jour, scripts d’ingestion/forecast unifiés, UI refaite et testée.
