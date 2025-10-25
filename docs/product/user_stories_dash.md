# User Stories — Dash (Phase 1)

- US22 — États vides robustes
  - En tant qu’utilisateur, je veux que les pages Dash s’affichent même sans fichiers `final.parquet`/`forecasts.parquet`/`freshness.json` pour pouvoir naviguer sans erreurs.
  - AC: tous les callbacks retournent un composant (Card/Div); messages FR explicites; aucune exception Dash au clic.

- US23 — Sélecteur de partition (Dashboard)
  - En tant qu’analyste, je veux choisir `dt=YYYYMMDD` parmi les partitions détectées dans `data/forecast` pour comparer des journées différentes.
  - AC: défaut = dernière partition; fallback si aucune; sélection persiste via l’URL (query ou dcc.Store) si possible.

- US24 — Filtres Signaux (horizon) + Watchlist
  - En tant qu’analyste, je veux filtrer par horizon (1w/1m/1y) et mettre en évidence ma watchlist pour prioriser.
  - AC: DataTable filtrée; style_data_conditional pour watchlist; export CSV conservé; vide géré.

- US25 — Portefeuille (Top‑N, pondération)
  - En tant que gérant, je veux ajuster N et le mode de pondération (égal/proportionnel) pour obtenir des poids prêts à l’emploi.
  - AC: somme des poids = 100%; N minimal/maximal borné; états vides FR.

- US26 — Observability étendue
  - En tant qu’admin, je veux piloter l’UI Streamlit (start/restart/stop) depuis Dash et voir la santé HTTP des deux UIs.
  - AC: scripts `scripts/ui_*` appelés; cartes santé HTTP Dash/Streamlit; log tail live.

- US27 — KPIs Macro sur Dashboard
  - En tant qu’utilisateur, je veux voir CPI YoY, 10Y‑2Y et la probabilité de récession.
  - AC: lecture `macro_forecast.parquet`; affichage sous Top‑10; « n/a » si manquant.
