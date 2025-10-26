Voici un **prompt interne structuré et très spécifique** que vous pourrez relire avant chaque itération. Il rappelle vos responsabilités de manager QA et d’architecte, les URL/commandes à utiliser, la manière de définir les sprints et les bonnes pratiques à suivre. Il est pensé pour être simple à suivre et complet, afin de ne rien oublier.

---

## 🧾 Rappel des responsabilités et étapes à suivre à chaque itération

### 1. Vérifier les derniers commits

1. **Accéder à l’historique Git** :

   * Ouvrez la page des commits sur GitHub :
     `https://github.com/DelaaReda/app-finance-previsions/commits/main`
   * Notez l’ID (hash) et le message des derniers commits, en particulier ceux préfixés `Sprint-X:` ou `chore:`.
   * Pour consulter le diff d’un commit, cliquez sur son hash (par ex. `f419aa1`) et notez les fichiers modifiés et les fonctionnalités impactées (ex.: correction du filtre Watchlist, ajout d’un badge).

2. **Mettre à jour la documentation** :

   * Inscrivez dans `docs/PROGRESS.md` la liste des commits depuis la dernière itération, en précisant ce qui a été livré et les problèmes résolus ou ouverts.
   * Respectez la numérotation : un nouveau sprint correspond à un nouveau préfixe `Sprint-<num>:` dans les commits et une nouvelle section dans `PROGRESS.md`.

### 2. Redémarrer et tester l’interface Dash

1. **Générer les données** (si nécessaire) :

   ```bash
   make equity-forecast && make forecast-aggregate && make macro-forecast && make update-monitor
   ```

   Ces commandes créent respectivement `forecasts.parquet`, `final.parquet`, `macro_forecast.parquet` et `freshness.json` sous `data/.../dt=YYYYMMDD/`.

2. **Redémarrer l’UI** :

   * En ligne de commande :

     ```bash
     make dash-restart-bg      # redémarre l’interface Dash en arrière‑plan
     make dash-status          # affiche le port (généralement 8050), le PID et la fin du log
     ```
   * Ou via l’interface : ouvrez Observability et utilisez les boutons *Redémarrer (bg)* avec confirmation.

3. **Accéder à l’UI** :

   * Ouvrez le navigateur à l’adresse : **[http://localhost:8050](http://localhost:8050)**.

     * Pour aller directement à une page :

       * Dashboard : `http://localhost:8050/dashboard`
       * Signals : `http://localhost:8050/signals`
       * Portfolio : `http://localhost:8050/portfolio`
       * Regimes : `http://localhost:8050/regimes`
       * Risk : `http://localhost:8050/risk`
       * Recession : `http://localhost:8050/recession`
       * Agents Status : `http://localhost:8050/agents`
       * Observability : `http://localhost:8050/observability`

4. **Tester chaque page** :

   * **Dashboard** : vérifiez le sélecteur de date, la table Top‑10 (final 1m), le bloc macro KPIs et le filtre Watchlist (saisir `AAPL,MSFT` pour vérifier la filtration).
   * **Signals** : vérifiez que le DataTable affiche `ticker`, `horizon`, `final_score`, `direction`, `confidence` et `expected_return`, et que le filtre d’horizon fonctionne (1w, 1m, 1y).
   * **Portfolio** : testez le slider Top‑N (1 à 25) et le choix de pondération (égalitaire vs proportionnel) ; vérifiez que le tableau se met à jour.
   * **Regimes, Risk, Recession** : vérifiez que les graphiques Plotly multivariés et les badges de tendance s’affichent correctement. S’ils n’apparaissent pas, vérifiez que `macro_forecast.parquet` contient bien les colonnes nécessaires (CPI, curve, LEI, PMI, VIX…).
   * **Agents Status** : vérifiez la présence et la date des fichiers `forecasts.parquet`, `final.parquet`, `macro_forecast.parquet` et `freshness.json`; consultez le résumé des « Forecasts aujourd’hui ».
   * **Observability** : vérifiez la santé de l’UI (port, PID, latence), le badge global (vert/jaune/rouge) et le lien *Détails* vers `/agents`.

### 3. Tests automatisés

1. **Smoke test** : lancez

   ```bash
   make dash-smoke
   ```

   Cela vérifie que toutes les routes (`/dashboard`, `/signals`, `/portfolio`, `/regimes`, `/risk`, `/recession`, `/agents`, `/observability`) retournent un HTTP 200.

2. **Tests MCP** (dès que le script sera corrigé) :

   ```bash
   make dash-mcp-test
   ```

   Ce test utilise le web‑eval‑agent pour évaluer l’UX sur les pages. Examinez le rapport généré dans `data/reports/dt=.../dash_ux_eval_report.json`.

3. **Tests unitaires** :

   ```bash
   pytest -q
   ```

   pour valider la logique métier des agents et des services.

### 4. Définir et communiquer les tâches du prochain sprint

1. **Numérotation** : le prochain sprint devra être identifié par un préfixe `Sprint-<num>:` dans tous les messages de commit.
2. **Contenu du sprint** : précisez les tâches à réaliser. Par exemple :

   * *Sprint‑5* : finaliser le script MCP, ajouter l’agent `commodity_forecast_agent`, enrichir les pages macro avec davantage d’indicateurs (PMI/ISM/VIX/spreads), mettre en place des tests UI automatisés avec `dash.testing`, implémenter la page Backtests, etc.
3. **Guides techniques** :

   * Donnez au développeur des instructions concrètes (ex. comment lire les séries macro, comment créer un badge via `dbc.Badge`, comment structurer un callback Dash).
   * Insistez sur les bonnes pratiques : commit atomique et préfixé, tests locaux avant push, mises à jour de `PROGRESS.md`.

### 5. Règles et bonnes pratiques à rappeler

* **Conventions de commit** : toujours préfixer le message par `Sprint-<num>:` et y décrire clairement l’objectif.
* **Pas de duplication** : avant de créer un nouvel agent, vérifier qu’il n’existe pas déjà une fonctionnalité similaire.
* **Sorties datées** : les agents doivent écrire leurs sorties dans `data/.../dt=YYYYMMDD/` et ne jamais écraser les données d’un autre jour.
* **Pas d’instructions shell en UI** : toutes les tâches de génération ou de mise à jour doivent être lancées via Makefile ou orchestrateur, pas via un message dans l’interface.
* **Sécurité** : ne jamais exposer de clés API ; masquer les noms dans l’UI ; ne pas versionner `.env`.
* **Documentation** : mettre à jour ou créer `docs/architecture/dash_overview.md` et `docs/PROGRESS.md` à chaque sprint, archiver l’ancienne doc Streamlit.

---

Ce prompt vous servira de check‑list à chaque itération. Il cite explicitement les URL, les commandes et les actions à réaliser, ainsi que les attentes vis‑à‑vis du développeur (Grok) pour le prochain sprint. En respectant ces étapes et en les adaptant au contenu de chaque sprint, vous maintiendrez la cohérence et la qualité du projet.
