# Backlog produit (EPICs, User Stories, AC)

## EPIC: Dash — Stabilisation & Migration (Phase 1)
- US22: En tant qu’utilisateur, je veux que les pages Dash n’échouent jamais en absence de données afin de naviguer sans erreurs.
  - AC: Plus de « Callback error updating page-content.children »; si `final.parquet`/`forecasts.parquet`/`freshness.json` manquent, la page affiche une carte d’état vide FR; testé avec `make dash-smoke` et navigation manuelle.

- US23: En tant qu’analyste, je veux choisir la partition de date (dt=YYYYMMDD) sur le Dashboard pour comparer des journées différentes.
  - AC: Sélecteur en page listant `data/forecast/dt=*`; défaut = dernière partition; fallback explicite si aucune partition.

- US24: En tant qu’analyste, je veux filtrer les signaux par horizon (1w/1m/1y) et surligner ma watchlist pour focaliser mon analyse.
  - AC: Contrôle horizon en-tête; DataTable filtrée; `WATCHLIST` (env ou fichier) mise en évidence via style; export CSV reste disponible.

- US25: En tant que gérant, je veux paramétrer Top‑N et le mode de pondération (égal/proportionnel) pour obtenir une proposition de portefeuille.
  - AC: Slider N, radio « égal/proportionnel »; calcul des poids et contrôle somme=100%; états vides FR si pas de `final.parquet`.

- US26: En tant qu’admin, je veux piloter l’UI Streamlit depuis la page Observability Dash et visualiser la santé HTTP des deux UIs.
  - AC: Boutons Start/Restart/Stop connectés aux scripts `scripts/ui_*`; cartes santé HTTP pour Dash (port AF_DASH_PORT) et Streamlit (port AF_UI_PORT); log live tail consultable.

- US27: En tant qu’utilisateur, je veux voir des KPIs Macro (CPI YoY, pente 10Y‑2Y, prob. récession) sur le Dashboard.
  - AC: Lecture `data/macro/forecast/dt=*/macro_forecast.parquet`; affichage sous le Top‑10; valeurs « n/a » gérées proprement.

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

## EPIC: UI & Ergonomie (FR, vides, feedback)
- US9: En tant qu’utilisateur, je veux des états « pas de données » utiles sur les pages vides.
  - AC: Messages d’explication, liens d’aide; pas de bouton « Stop » sans contexte.

- US10: En tant qu’utilisateur, je veux un sélecteur de date directement dans les pages dépendantes d’une partition.
  - AC: Sélecteur in‑page; fallback = dernière partition; pas de dépendance à la sidebar.

- US11: En tant qu’utilisateur, je veux des confirmations après sauvegarde (watchlist, notes, réglages).
  - AC: st.success/st.error explicites; erreurs gérées.

- US12: En tant qu’utilisateur, je veux une langue unifiée (FR) et une top‑nav claire.
  - AC: Libellés FR; date formatting FR; top‑nav sticky + footer port 5555.

## EPIC: Admin & Sécurité UI
- US13: En tant qu’owner, je veux retirer des pages toute instruction « lancez make … ».
  - AC: Ces conseils déplacés vers « Agents Status » (expander Admin) et docs.

- US14: En tant qu’owner, je veux éviter d’exposer des noms de variables sensibles en Observability.
  - AC: Noms génériques (Key #1/2...) + booléens présent/absent.

## Tâches Techniques clés
- Uniformiser ingestion via interfaces (src/ingestion/*) et data contracts (src/core/models.py).
- Séparer couches: domain (models, io), application (use-cases), adapters (ingestion, UI).
- Refactor UI Streamlit: pages modulaires, `st_ui.py` utilitaires, composants communs.
- Normaliser config/env: src/core/config.py, .env, secrets_local.py.
 - Automatiser UI smoke (Playwright MCP) + sécurité (pip‑audit/safety/bandit/secret‑scan).


## EPIC: Usine à Prévisions F4F
- US15: Agent Prévision Équité (equity_forecast_agent) — génère distributions (1j/1s/1m), écrit equities.parquet.
  - AC: ≥5 ans d'historique; colonnes standardisées (ticker, horizon, mean, vol, p_up).
- US16: Agent Prévision Commodités (commodity_forecast_agent) — inclut drivers macro.
  - AC: actifs or/pétrole initiaux; champs alignés à US15.
- US17: Agent Macro Forecast (macro_forecast_agent) — croissance/inflation/taux.
  - AC: prob. récession/12m; séries publiées.
- US18: Agrégateur/Arbitre (forecast_aggregator_agent) — construit final.parquet.
  - AC: pondération par performance; export CSV.
- US19: Backtest Agent — top‑N basé prévisions; métriques.
  - AC: Sharpe, CAGR, MaxDD; équity curve.
- US20: Évaluation modèles — MAE/RMSE/couverture; classement.
  - AC: tableau UI Évaluation; export.
- US21: Sentiment & News — score par actif/thème.
  - AC: feed enrichi; score [-1,1].
