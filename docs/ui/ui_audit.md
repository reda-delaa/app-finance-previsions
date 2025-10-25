# UI Audit & Redesign Plan (v2)

Constats fonctionnels (observés)
- Pages vides/non implémentées: News Agrégation, Deep Dive, Forecasts multi‑tickers, Backtests demo, Evaluation, Advisor affichent titres mais aucun contenu utile.
- Dépendances non satisfaites visibles: invites « Lancez scripts/... » ou « make ... » dans Portfolio, Alerts, Scoreboard, Earnings, Agents Status; ces messages doivent être déplacés hors des pages utilisateurs.
- Données manquantes: Agents/Memos/Risk n’affichent rien sans partitions; fournir des états vides explicites et actions sûres.

Ergonomie & cohérence
- Navigation: menu latéral très long; top‑nav ajoutée (sticky) avec séparation « Prévisions » vs « Administration ».
- Commutateur « Beginner mode »: accessibilité à vérifier (DOM/test UI smoke).
- Affichage brut: JSON détaillé sous expander (Reports/Deep Dive) au lieu d’être in‑line.
- Mélange de langues: uniformiser en français, dates FR, libellés FR.
- Éléments sans feedback: Watchlist/Notes/Settings doivent confirmer sauvegarde.
- Bandeau supérieur gris: harmonisé avec thème sombre (.streamlit/config.toml).
- Bouton « Stop » sans contexte: à retirer des pages vides.

Décisions & actions
- Sélecteurs de date en page pour Regimes, Risk, Recession, Memos (fallback = dernière partition).
- États vides utiles: texte d’explication + liens d’aide; masquer commandes internes.
- Observability: ne pas lister de noms de clés; afficher « Key #N présente: Oui/Non ».
- Head/Footer communs: `ui.shell.page_header/footer` intégrés sur les pages clés.
- Exports: privilégier CSV pour tableaux synthétiques; JSON détaillé dans expander.
- Sécurité: éviter prompt‑injection via messages « lancez make ... » dans UI; déplacer vers « Agents Status » (expander Admin) et docs.

Run discipline
- Port canonique: 5555 (scripts/ui_start.sh). Une seule instance Streamlit.
- Après tout changement UI: `make ui-restart`. Watcher: `make ui-watch`.

Validation (automatisée)
- UI smoke (Playwright MCP): vérifie pages clés (HTTP 200), présence navbar/footer, labels FR, aucun secret.
- Tests runners: stabilité des pages lisant dernières partitions.

Statut
- Fait: top‑nav + footer, thème sombre, Scoreboard revu (CSV export), pages Admin dédiées (Earnings, Agents Status), SearXNG local, audit sécurité.
- À faire prochain: déplacer les sélecteurs de date en page; ajouter états vides; unifier libellés FR; confirmations de sauvegarde Notes/Settings.


Référence
- Voir le retour structuré: docs/atlas/2025-10-25_atlas_feedback.md
