# Backlog produit (EPICs, User Stories, AC)

## EPIC: Information Marchés & Macro
- US1: En tant qu’utilisateur, je veux un tableau de bord macro (inflation, taux, courbes, FRED séries clés) pour comprendre le contexte.
  - AC: Affiche CPI, Core CPI, 10Y, 2Y, 10Y-2Y, Unemployment; graphiques sur 1Y/5Y/Max; liens source.
  - Source: FRED MCP, gscpi.py, gpr.py.

- US2: En tant qu’utilisateur, je veux une veille actualités/événements (macro/secteurs) avec résumé.
  - AC: Feed quotidien agrégé (sources configurables), résumé auto, filtres par secteur, liens cliquables.
  - Source: tavily-mcp, firecrawl-mcp, finnews.py, nlp_enrich.py.

## EPIC: Analyse Action (Deep Dive)
- US3: En tant qu’utilisateur, je veux une fiche action (profil, fondamentaux, ratios, ownership, dérivés).
  - AC: Page action unique, sections Fundamentals, Estimates, Ownership, Derivatives; export CSV.
  - Source: finviz.py, financials_ownership_client.py, macro_derivatives_client.py.

- US4: En tant qu’utilisateur, je veux une analyse technique synthétique (tendances, niveaux, indicateurs) et pairs.
  - AC: Indicateurs clés (MA, RSI, MACD), niveaux support/résistance, liste de pairs sectoriels.
  - Source: phase2_technical.py, peers_finder.py.

## EPIC: Prévisions & Stratégies
- US5: En tant qu’utilisateur, je veux des prévisions (marché global, secteurs, titres) et explications.
  - AC: Prévisions 1W/1M configurable, incertitude, drivers; logs et modèle affichés.
  - Source: market_intel.py, econ_llm_agent.py, market_data.py, models.py.

- US6: En tant qu’utilisateur, je veux backtester une stratégie simple (règles, période) et voir les métriques.
  - AC: Paramètres stratégie (entrée/sortie), période, métriques (CAGR, MaxDD, Sharpe), equity curve.
  - Source: analytics/backtest_news_impact.py, market_data.py.

## EPIC: Observabilité & Qualité
- US7: En tant qu’owner, je veux des logs clairs et reproductibilité pour diagnostiquer.
  - AC: logging_setup.py centralisé; warn_log harmonisé; tests runners stables.

- US8: En tant qu’owner, je veux une UI cohérente (pages, navigation, perf) avec cache et état persistant.
  - AC: Arborescence de pages claire, tabs, caches avec TTL; zero “widget chaos”.

## Tâches Techniques clés
- Uniformiser ingestion via interfaces (src/ingestion/*) et data contracts (src/core/models.py).
- Séparer couches: domain (models, io), application (use-cases), adapters (ingestion, UI).
- Refactor UI Streamlit: pages modulaires, `st_ui.py` utilitaires, composants communs.
- Normaliser config/env: src/core/config.py, .env, secrets_local.py.
