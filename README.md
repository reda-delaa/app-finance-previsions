Désolé pour la confusion : vous avez raison, la documentation actuelle est obsolète car elle mélange l’ancien Hub Streamlit et la nouvelle application Dash. Voici un README simplifié et révisé qui se concentre uniquement sur l’architecture **Dash** de `app-finance-previsions`, sans duplication.

---

## 📊 App Finance Prévisions — Interface Dash

Cette application est un assistant de prévisions économiques et de sélection d’actions destiné à un investisseur privé. Elle combine des indicateurs macroéconomiques, techniques et fondamentaux avec des analyses de news, puis synthétise ces informations via des agents d’IA pour fournir des recommandations.

---

### 🚀 Commandes essentielles (Dash)

* **Démarrer l’UI Dash en arrière‑plan** :

  ```bash
  make dash-start-bg
  ```
* **Générer les données (agents)** :

  ```bash
  make equity-forecast
  make forecast-aggregate
  make macro-forecast
  make update-monitor
  ```
* **Redémarrer l’UI Dash après modifications** :

  ```bash
  make dash-restart-bg
  ```
* **Tests de connectivité** :

  ```bash
  make dash-smoke     # vérifie le HTTP 200 sur toutes les routes
  make dash-mcp-test  # (après correction du script) lance le test UX via web-eval-agent
  ```

L’interface est accessible sur [http://localhost:8050](http://localhost:8050).

---

### 🧠 Architecture (vue d’ensemble)

```
[Sources]
   ├─ FRED (macro)        → ingestion JSON/CSV
   ├─ yfinance (actions)  → prix OHLCV & fondamentaux
   ├─ RSS/News            → pipeline de normalisation
   └─ Autres (Finviz, ...)

[Agents]
   ├─ equity_forecast_agent      → prévisions actions (1w/1m/1y)
   ├─ macro_forecast_agent       → prévisions macro (croissance, inflation, taux)
   ├─ forecast_aggregator_agent  → agrégation des prévisions (score final)
   ├─ update_monitor_agent       → surveillance de la fraîcheur et backfill 5 ans
   └─ (à venir) commodities, backtests, évaluation, sentiment...

[UI Dash]
   ├─ Dashboard         → Top‑N final, KPIs macro, filtre Watchlist
   ├─ Signals           → DataTable triable/exportable des signaux par horizon
   ├─ Portfolio         → Propositions Top‑N avec pondération
   ├─ Regimes/Risk/Recession → Visualisations macro multivariées et badges de tendance
   ├─ Agents Status     → Présence et date des dernières partitions (forecasts, final, macro)
   └─ Observability     → Santé du serveur, fraîcheur des données, badge global (✓/⚠/✗)
```

---

### 🛠 Installation

1. **Cloner le dépôt** :

   ```bash
   git clone [URL_DU_REPO]
   cd app-finance-previsions
   ```

2. **Créer et activer un environnement virtuel** :

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/Mac
   # ou
   .\.venv\Scripts\activate       # Windows
   ```

3. **Installer les dépendances** :

   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les variables d’environnement** :

   ```bash
   cp .env.example .env
   # puis éditer .env avec vos clés API
   ```

---

### 📈 Pages & fonctionnalités principales

* **Dashboard** : vue d’ensemble avec Top‑10 (basé sur `final.parquet`), indicateurs macro (CPI YoY, pente 10Y‑2Y, probabilité de récession), sélecteur de date et filtre Watchlist.
* **Signals** : tableau interactif de signaux par horizon (1w/1m/1y) avec tri, filtres et export CSV. Surligne les tickers de la watchlist.
* **Portfolio** : propose un Top‑N (paramétrable) de titres selon le score final, avec pondération égalitaire ou proportionnelle.
* **Regimes / Risk / Recession** : graphiques Plotly multivariés et badges de tendance (inflation, courbe des taux, LEI, PMI/ISM, VIX, spreads, drawdown, chômage). Tables récapitulatives des derniers points. États vides explicites si certaines colonnes sont absentes.
* **Agents Status** : liste les dernières partitions (`forecasts.parquet`, `final.parquet`, `macro_forecast.parquet`, `freshness.json`) avec date/heure et présence. Résumé de fraîcheur (aujourd’hui ou non).
* **Observability** : affiche le port, le PID, la latence HTTP ; badge global (✓ vert si tout est frais, ⚠ jaune si données périmées, ✗ rouge si serveur down) ; lien vers Agents Status ; actions d’administration de l’ancienne UI (legacy).

---

### 🧪 Tests & QA

* **Tests unitaires** : exécuter `pytest -q` pour valider la logique métier.
* **Smoke tests** : `make dash-smoke` vérifie que toutes les routes (`/dashboard`, `/signals`, `/portfolio`, `/regimes`, `/risk`, `/recession`, `/agents`, `/observability`) renvoient HTTP 200.
* **Tests UX MCP** : `make dash-mcp-test` utilise le web‑eval‑agent (lorsqu’il sera corrigé) pour évaluer l’interface via IA.
* **Procédure QA** : vérifier les commits récents, redémarrer l’UI, tester chaque page, examiner Observability et Agents Status, rédiger un bilan. Voir la doc détaillée dans `docs/PROGRESS.md`.

---

### 🔒 Sécurité & bonnes pratiques

* Ne jamais committer vos clés API ou données sensibles. `.gitignore` exclut les fichiers `.env`, les secrets locaux et les répertoires `data/`.
* Utiliser des identifiants génériques dans l’UI (par ex. « Clé API A : ✅ » au lieu du nom de la clé).
* Centraliser la configuration dans `config.yaml` (watchlist, seuils d’alertes) et l’importer via `src/core/config.py`.

---

### 📚 Documentation utile

* **`docs/AGENTS_PROMPT.md`** : guide générique pour la création d’agents (à lire en premier).
* **`docs/PROGRESS.md`** : suivi des sprints, tâches livrées et à venir.
* **`docs/architecture/dash_overview.md`** : architecture détaillée de l’interface Dash, organisation des pages, commandes de démarrage et d’orchestration.
* **`docs/README.md`** : index général de la documentation.