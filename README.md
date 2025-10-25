# Analyse Financière — Hub IA (Macro · Actions · News)

Suite pro d'outils d’analyse financière combinant signaux macro (FRED), technique (yfinance), fondamentaux, et news agrégées — puis synthétisés par un arbitre/IA pour orienter la décision (rotation sectorielle, couverture FX/taux, focus titres).

TL;DR (3 commandes utiles)
- App principale (Hub IA):
  - `PYTHONPATH=src streamlit run src/apps/app.py`
- Tests d’intégration réseau (FRED/yfinance) avec venv:
  - `make it-integration-venv`
- News snapshot (CLI):
  - `python -m src.analytics.market_intel run --regions US,INTL --window last_week --ticker AAPL --stdout`

## Architecture (vue d’ensemble)

```
[Sources]
  FRED  yfinance  RSS/News  Finviz?  MacroDerivs?  ➜  Ingestion & Normalisation
   |       |         |         |         |
   |       |         |         |         └─ (optionnels, best-effort)
   |       |         |         └─ finviz_client (company/options/futures)
   |       |         └─ finnews (run_pipeline) → news normalisées
   |       └─ get_stock_data / OHLCV
   └─ fetch_fred_series (API JSON + fallback CSV)

[Analytics]
  - phase3_macro: nowcast macro (z-scores + composants + fraîcheur séries)
  - phase2_technical: indicateurs techniques (SMA/RSI/MACD/BB, etc.)
  - market_intel: agrégation news → features (sentiment, événements, secteurs)

[Features Bundle]
  macro + technical + fundamentals + news  →  ctx['features'] pour IA/Arbitre

[Décision]
  - econ_llm_agent (arbitre) → synthèse / orientation
  - nlp_enrich (IA) → explications et pistes d’actions

[UI Streamlit]
  apps/app.py (Hub): Macro synthèse + Actions + News + IA/Arbitre + Diagnostics
```

## Installation — du plus important

1. Cloner le repository
```bash
git clone [URL_DU_REPO]
cd analyse-financiere
```

2. Créer et activer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement
```bash
cp .env.example .env
# Éditer .env avec vos clés API
```

## Comment l’App aide la prévision (3–6 mois)

- Le bloc “Synthèse macro” expose des z‑scores robustes (Croissance/Inflation/Politique/USD/Commodities) + composants (YoY, slope), lisibles et datés.
- Les news sont agrégées (sentiment/événements) et fusionnées avec les signaux macro → `features` unifié passé à l’IA et à l’arbitre.
- L’IA vulgarise le contexte, l’arbitre propose une orientation (rotation sectorielle/couverture FX/taux) avec un rationnel synthétique.

## Scénarios d'Utilisation Principaux

### 1. Collecte et Analyse d'Impact des Nouvelles

```bash
# Collecter les nouvelles pour une entreprise
python src/ingestion/finnews.py --company "Orange" --ticker ORA

# Analyser l'impact des nouvelles
python src/analytics/backtest_news_impact.py --news data/processed/news_enriched.jsonl
```

### 2. Application d'Analyse Macroéconomique

```bash
streamlit run src/apps/macro_sector_app.py
```

### 3. Application d'Analyse d'Actions

```bash
streamlit run src/apps/stock_analysis_app.py
```

## Tests

```bash
pytest -q
```

## Flux d’intégration (détaillé)

```
1) Macro
   phase3_macro.get_macro_features()
     ├─ FRED JSON (clé) → observations; fallback CSV (fredgraph) si besoin
     ├─ z-scores (GRW, INF, POL, USD, CMD)
     └─ composants + fraîcheur séries (AAAA‑MM)

2) News
   finnews.run_pipeline(...) → items normalisés
   market_intel.collect_news + build_unified_features → features agrégées

3) Actions
   get_stock_data + phase2_technical → indicateurs
   fondamentaux (yfinance) → ratios de base

4) Décision
   ctx['features'] = {macro, news, [technical, fundamentals]}
   econ_llm_agent.analyze(ctx)  → orientation & drivers
```

## Structure des Données

### data/raw/
- Données brutes (fichiers RSS, dumps JSONL, données yfinance)
- Non versionnées dans git

### data/interim/
- Données transformées intermédiaires
- Ex: nouvelles enrichies avant agrégation finale

### data/processed/
- Features finales prêtes pour les modèles
- Données nettoyées et validées

### artifacts/
- Sorties générées (figures, rapports CSV/JSON)
- Organisées par date (YYYY-MM-DD_description/)

## Maintenance

- Logs dans `logs/`
- Cache dans `cache/`
- Artifacts datés dans `artifacts/`
- Tests unitaires dans `tests/`

## Roadmap (intégration continue)

- Formaliser un FeatureBundle (dataclass) {macro, technical, fundamentals, news} avec as_dict/from_*.
- Ajouter poids/horizon paramétrables pour l’arbitre (config.yaml).
- Étendre la couverture tests d’intégration (ajout d’autres sources réseau marquées @integration).
- Ajouter un petit bandeau “état des sources” (🟢/🟠/🔴) en haut de l’app.

## Sécurité & Secrets

- Ne commitez jamais vos clés: `.gitignore` ignore `.env`, `src/secrets_local.py`, `*.key`, `*.pem`, etc.
- Pour purger l’index si déjà commis: `git rm --cached src/secrets_local.py && git commit -m "remove secrets_local"`.
### 4. Hub (macro + actions + news + IA)

```bash
PYTHONPATH=src streamlit run src/apps/app.py

## Documentation centrale

- docs/AGENTS_PROMPT.md — prompt cadre central pour tous les agents (à lire en premier)
- docs/PROGRESS.md — statut, ce qui est fait, ce qui manque, comment lancer
- docs/README.md — index de la documentation détaillée
```
## DOC for QA/PO/MANAGER Agent 

### Procédure détaillée pour tester l’UI et valider les dernières modifications

En tant qu’ATLAS, voici un guide pas à pas pour s’assurer que l’interface est testée sur la version la plus récente du code et pour vérifier l’impact des derniers commits avant de proposer de nouvelles priorités.

#### 1. Vérifier les derniers commits

1. **Ouvrir la liste des commits** :

   * Rendez-vous sur la page GitHub du projet : `https://github.com/DelaaReda/app-finance-previsions`.
   * Cliquez sur l’onglet **Code** puis sur **commits** ou directement sur `https://github.com/DelaaReda/app-finance-previsions/commits/main`.
   * Consultez les messages et les horodatages des derniers commits pour identifier les nouveautés et les corrections récentes.

2. **Étudier les modifications pertinentes** :

   * Pour un commit particulier, cliquez sur son hash (par ex. `d54f221`) pour voir les fichiers modifiés.
   * Notez les pages ou scripts concernés (par exemple : ajout d’un bouton Start/Stop dans Observability, refactoring de `ui_start_bg.sh`, etc.).
   * Cette étape permet de savoir quelles fonctionnalités tester spécifiquement dans l’UI.

#### 2. Redémarrer l’UI pour utiliser la dernière version

Afin d’être certain(e) que l’interface reflète les dernières modifications, il est indispensable de redémarrer l’UI après chaque mise à jour de code.

* **Méthode via la ligne de commande (depuis la racine du dépôt)** :

  ```bash
  make ui-restart-bg    # redémarre l’UI en arrière-plan et écrit les logs dans logs/ui
  make ui-status        # affiche le port, le PID et la fin du log
  ```

  Ces commandes utilisent les scripts `scripts/ui_restart_bg.sh` et `scripts/ui_status.sh`, qui sont agnostiques du répertoire courant grâce à `REPO_ROOT`.

* **Méthode via l’interface Observability** :

  1. Ouvrez l’UI (voir section suivante) et allez dans **Observability** via le menu latéral.
  2. Dans la section « Action (Admin) — Redémarrer l’UI », cliquez sur l’icône d’expansion « ▶ » pour dévoiler le bouton.
  3. Cochez la case de confirmation (si elle est présente) et cliquez sur **Redémarrer l’interface (arrière‑plan)**.
  4. Patientez le temps que l’UI redémarre (la sonde de santé essaie jusqu’à 15 fois par défaut). Le résultat du script s’affiche dans l’UI.
  5. Vérifiez ensuite la santé de l’UI dans la section « UI — Santé » (port, PID, statut).

#### 3. Accéder à l’interface utilisateur

1. **Ouvrir l’URL** : l’UI est accessible via `http://localhost:5555` (ou `http://localhost:5555/Dashboard` pour atterrir directement sur le tableau de bord).
2. **Attendre le chargement** : si l’interface ne s’affiche pas immédiatement, patienter quelques secondes ; en cas d’erreur 200/404, assurez-vous que l’UI a bien redémarré comme décrit ci-dessus.
3. **Naviguer dans les pages** : utilisez le menu latéral pour accéder aux différentes sections (Dashboard, News, Deep Dive, Forecasts, Observability, Backtests, Reports, etc.).

#### 4. Tester l’Observability et les contrôles UI

1. **Accéder à Observability** : dans le menu latéral, cliquez sur *Observability*.

2. **Examiner la santé de l’UI** : en haut de la page, le tableau des métriques affiche le port (`5555`), l’état du processus (vivant ou non) et le PID.

3. **Redémarrer / Démarrer / Arrêter l’UI** :

   * Pour **redémarrer** l’UI en arrière‑plan, utilisez le panneau « Redémarrer l’interface (arrière‑plan) » (icône ▶ à déplier). Une confirmation est demandée.
   * Pour **démarrer** l’UI s’il n’y a pas d’instance, cochez la case « Je confirme le démarrage de l’UI » puis cliquez sur **Démarrer (bg)**.
   * Pour **arrêter** l’UI en cours d’exécution, cochez la case « Je confirme l’arrêt de l’UI » puis cliquez sur **Arrêter**.
   * Les scripts se déclenchent en arrière‑plan et leurs sorties sont affichées dans l’interface. Un message confirme la réussite ou l’échec.
   * La section « Processus » indique que l’UI principale et les pages sont chargées et rappelle de consulter les logs si nécessaire.

4. **Vérifier les API Keys** : Observability présente aussi un tableau « Clés d’API (présence seulement) » pour indiquer si les clés nécessaires sont présentes sans en afficher les valeurs.

#### 5. Tester les autres pages après redémarrage

1. **Dashboard** : vérifier l’affichage de la page « Dashboard — Résumé & Picks ». Si elle indique qu’aucun fichier de prévisions n’est trouvé, ne pas exécuter de script depuis l’UI, mais noter ce manque de données et vérifier que les agents de prévision ont bien été exécutés.

2. **Pages “Deep Dive”, “Forecasts”, “Signals”, “Portfolio”, “Alerts”** : s’assurer que les pages ne sont plus vides et qu’elles n’affichent pas d’instructions techniques (ex. : exécuter un script). Si les données sont absentes, un message explicatif et des états vides sûrs doivent s’afficher.

3. **Observabilité** : après avoir redémarré l’UI, retourner sur Observability et confirmer que les métriques (port, PID, process vivant) reflètent bien l’état courant.

4. **Backtests et Evaluation** : si ces pages ont été développées, vérifier qu’elles chargent correctement et qu’elles affichent les résultats des agents ou des messages d’état vide.

#### 6. Synthèse et priorisation

Une fois ces étapes complétées, rédigez un bilan :

* Quelles pages fonctionnent comme prévu ?
* Quelles pages affichent encore des JSON bruts ou des messages techniques ?
* Y a‑t‑il des erreurs (ex. : fichier manquant, processus non détecté) ?
* Les boutons Start/Stop/Restart de l’UI fonctionnent-ils correctement (avec confirmation) ?

Sur cette base, établissez les prochaines priorités de développement et d’amélioration, en tenant compte des objectifs stratégiques (mise en place des agents F4F, amélioration des pages principales, raffinement d’Observability, etc.).

---

En appliquant cette procédure à chaque itération (vérification des commits, redémarrage de l’UI, test des pages et bilan), vous vous assurerez que les décisions de priorisation reposent sur l’état réel du produit et que l’UI est toujours évaluée sur la version la plus récente du code.
