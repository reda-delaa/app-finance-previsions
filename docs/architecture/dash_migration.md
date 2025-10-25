# Plan de migration vers Dash (pour le développeur Codex)

Pour transformer l’interface actuelle en une application Dash plus professionnelle tout en respectant la vision d’une « usine de prévisions » pour investisseur privé, voici un plan de migration structuré. Il vise à minimiser les interruptions, à réutiliser le maximum de logique existante et à garder la séparation « Analyse » vs « Admin ».

## 🎯 Objectifs de la migration

- Améliorer l’UX/UI : obtenir un rendu plus abouti, avec des composants interactifs avancés et un thème homogène (par exemple Bootstrap sombre).
- Conserver la logique de données : les agents continuent d’écrire leurs sorties (Parquet/JSON) dans `data/…/dt=YYYYMMDD/`. L’interface Dash consommera ces fichiers via des callbacks.
- Faciliter l’extension future : permettre l’ajout d’onglets, de graphiques, de filtres et d’alertes sans hacks HTML, tout en restant en Python.

## Phase 0 – Préparation (analyse et choix techniques)

1. Lister les pages et fonctionnalités existantes : Dashboard (Top‑N, signaux), Forecasts, Signals, Portfolio, Alerts, Regimes, Risk, Recession, News, Reports, Deep Dive, Observability, Agents Status, Settings…
2. Définir une charte graphique : choisir un thème Bootstrap (par exemple « Cyborg » ou « Slate ») via `dash-bootstrap-components`.
3. Configurer l’environnement :

```bash
pip install dash dash-bootstrap-components pandas pyarrow plotly
```

Créer un nouveau package `src/dash_app/` qui contiendra l’application Dash.

4. Décider de la structure du backend :
- Option 1 : garder Streamlit pour les agents et l’orchestration, et utiliser Dash uniquement pour le front. Démarrage via `python src/dash_app/app.py`.
- Option 2 : supprimer la partie Streamlit une fois la migration terminée. Les scripts d’observabilité et de gestion des agents resteront côté backend (scripts shell / Makefile).

## Phase 1 – Maquette et squelette de l’appli Dash

1. Créer `src/dash_app/app.py` (squelette):

```python
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = dbc.Container([
    dcc.Location(id="url"),
    dbc.Row([
        dbc.Col(html.Div(id="sidebar"), width=2),
        dbc.Col(html.Div(id="page-content"), width=10),
    ], className="g-0"),
], fluid=True)
```

2. Implémenter une barre latérale (sidebar) avec deux sections :
- Analyse : Dashboard, Signals, Forecasts, Portfolio, Regimes, Risk, Recession, News, Reports, Deep Dive.
- Administration : Observability, Agents Status, Settings, Quality, LLM Models.

3. Mettre en place le routage multipages via `dcc.Location` + callback pour remplir `page-content`.

4. Créer un module par page dans `src/dash_app/pages/` exposant `layout()`.

## Phase 2 – Migration des pages essentielles

1. Dashboard :
- Lire `data/forecast/dt=<today>/final.parquet` et `forecasts.parquet` ; table Top‑10 (DataTable) par `final_score` ; fallback composantes.
- Badge d’alertes (dbc.Badge) et filtres (watchlist/date).

2. Signals :
- Joindre `final.parquet`/`forecasts.parquet` ; DataTable triable/filtrable ; export CSV.

3. Portfolio :
- Top‑N 1m à partir de `final.parquet` ; sliders (N, pondération) ; export CSV/JSON.

4. Observability :
- Cartes (dbc.Card) pour Port/PID/HTTP/latence ; boutons Start/Stop/Restart (subprocess côté serveur) ; panneau “Log en direct” (dcc.Interval ou bouton Rafraîchir).

5. Agents Status :
- Tableau état agents (dernière exécution, OK/KO) à partir d’un JSON de statut.

## Phase 3 – Migration des autres pages

- Regimes, Risk, Recession : consommer `macro_forecast` et afficher graphiques Plotly + tableaux.
- Backtests et Évaluation : visualiser performances cumulées et métriques.
- News & Deep Dive : accordion (dbc.Accordion) pour les articles/rapports (JSON détaillé repliable).
- LLM Models, Quality, Settings : transposition fidèle, sans invites techniques.

## Phase 4 – Ajustements et fin de migration

- Thème et branding (logo, couleurs, typo). Tests UI (dash.testing/Selenium). Doc: `docs/PROGRESS.md`, README (procédure `python src/dash_app/app.py`). Décommission Streamlit quand prêt.

## Phase 5 – Extensions futures

- Brancher les pages macro dès que macro_forecast/update_monitor sont stables.
- Notifications/toasts ; authentification (le cas échéant) ; déploiement (Docker/PaaS).

---

Ce plan permet une migration progressive vers Dash sans interrompre le flux actuel, en restant aligné avec l’approche « usine de prévisions » et une séparation claire Analyse vs Admin.
