# UI Audit & Redesign Plan

Problèmes observés
- Couche UI mêlée à la logique métier, widgets non structurés, navigation confuse.
- Chargements lents (manque de cache), gestion d’état inconsistante.

Actions
- Découper en pages: Macro, News, Deep Dive, Forecast/Backtest, Observabilité.
- Extraire composants communs dans src/utils/st_ui.py (tables, cards, KPIs, layouts).
- Uniformiser couleurs/thème, palettes cohérentes (test_colors_* existants).
- Ajouter cache TTL par datasource (quotes, fundamentals, macro).
- Ajouter “state” léger (session_state) pour sélections utilisateur.

Validation
- Définir critères UX: TTFB page < 2s (données cache), interactions fluides, lisibilité.
