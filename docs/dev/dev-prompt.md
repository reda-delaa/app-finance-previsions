📣 PROMPT GÉNÉRAL DEV — App Finance Prévisions (Dash + Agents)

🎯 Vision & Principes
	•	Tu construis une usine d’agents qui ingèrent, contrôlent la qualité, prévoient (multi-horizons), agrègent et exposent leurs sorties dans une UI Dash pour un investisseur privé.
	•	Jamais de duplication d’agent ni de module. Tout agent écrit ses sorties dans des partitions datées : data/<domaine>/dt=YYYYMMDD/<fichier>.
	•	Historique ≥ 5 ans pour toute série utilisée par des modèles/agents de prévision.
	•	États vides FR, pas de JSON brut (mets-le dans un expander « Voir JSON »), labels FR unifiés.

⸻

🧭 Repères & URLs
	•	Repo GitHub : https://github.com/DelaaReda/app-finance-previsions
	•	IDE web (GitHub.dev) : https://gloomy-superstition-5g7p5rvvqp5934gjx.github.dev
	•	UI Dash (local) : http://localhost:8050
Routes clés : /dashboard, /signals, /portfolio, /regimes, /risk, /recession, /agents, /observability, /news
(à ajouter/valider : /deep_dive, /forecasts, /backtests, /evaluation)

⸻

🛠️ Environnement & Lancement
	1.	Créer l’environnement Python

python -m venv .venv
source .venv/bin/activate     # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt

	2.	Générer les données minimales

make equity-forecast && make forecast-aggregate
make macro-forecast && make update-monitor

	3.	Démarrer/relancer Dash

make dash-start-bg         # ou: make dash-restart-bg
make dash-status && make dash-logs
# Ouvre http://localhost:8050

⚠️ Ne jamais committer de données : data/ est ignoré par .gitignore. Si un fichier apparaît tracké, le retirer.

⸻

📁 Organisation du code (où coder)
	•	UI Dash : src/dash_app/
	•	App & routing : src/dash_app/app.py
	•	Pages : src/dash_app/pages/*.py (Dashboard, Signals, Portfolio, Regimes, Risk, Recession, AgentsStatus, Observability, News, Deep Dive, Forecasts, Backtests, Evaluation)
	•	Agents : src/agents/*.py → sorties sous data/**/dt=YYYYMMDD/…
	•	Ops/Tests :
	•	Makefile : dash-start-bg, dash-restart-bg, dash-status, dash-logs, dash-stop, dash-smoke, dash-mcp-test
	•	MCP runner : ops/ui/mcp_dash_smoke.mjs (stderr visible)

⸻

🧪 Tests — Manuel, Smoke, MCP, Unitaires

A. Manuel (à faire avant chaque push)
	•	Relancer l’UI : make dash-restart-bg
	•	Vérifier les pages clés (200/OK et rendu attendu) :
http://localhost:8050/{dashboard,signals,portfolio,regimes,risk,recession,agents,observability,news}
	•	Vérifier badge statut global (sidebar) :
	•	🟢 = HTTP 200 + fraîcheur ≤ 25h
	•	🟡 = HTTP 200 mais données trop anciennes
	•	🔴 = serveur down

B. Smoke (automatisé)

make dash-smoke   # attend HTTP 200 sur toutes les routes connues

C. MCP (évaluation UX automatisée)

make dash-mcp-test
# Le script Node 'ops/ui/mcp_dash_smoke.mjs' génère un rapport et affiche les erreurs stderr

	•	Si erreur TypeError [ERR_INVALID_ARG_TYPE]: "file" must be string : vérifier les chemins d’artifacts/screenshot dans le script MCP, et que name → fname n’est jamais undefined.

D. Tests unitaires / UI
	•	pytest (si présent) : pytest -q
	•	dash.testing : ajoute des assertions simples (existence d’un DataTable/Graph sur une route) dans tests/.

⸻

🧩 Checklists d’implémentation

✅ Ajouter / modifier une PAGE Dash
	1.	Créer le fichier dans src/dash_app/pages/xxx.py avec une fonction layout() et éventuels callbacks.
	2.	Enregistrer la route dans src/dash_app/app.py (sidebar + dcc.Location → layout).
	3.	Données : lire la dernière partition via un utilitaire (ex. utils/partitions.py) :
	•	latest_dt("data/forecast") ⇒ YYYYMMDD
	•	read_parquet_latest("data/forecast", "final.parquet")
	4.	UI : DataTable/Graph Plotly, filtres (ticker/horizon/date), état vide FR.
	5.	Tests :
	•	Manuel : URL dédiée http://localhost:8050/xxx
	•	make dash-smoke doit inclure la route
	•	make dash-mcp-test doit produire un rapport sans erreur bloquante

✅ Ajouter / modifier un AGENT
	1.	Créer src/agents/<nom>_agent.py
	•	Entrées : fichiers existants (ex. forecasts.parquet, prix historiques)
	•	Sortie datée : data/<domaine>/dt=YYYYMMDD/<nom>.parquet|json
	•	logging (début/fin, volumes, anomalies) → logs/ ; pas de print()
	2.	Makefile : ajouter une cible (ex. backtest, evaluation) avec PYTHONPATH=src python -m src.agents.<nom>_agent
	3.	Idempotence : ne réécris pas d’anciennes partitions ; écris uniquement dans dt=today
	4.	Qualité : si trous/duplicats → produit un rapport qualité (data/quality/dt=…/freshness.json ou anomalies)
	5.	Validation : exécute la cible Make, vérifie le fichier, ajoute lecture côté page correspondante

⸻

🧹 Code style & bonnes pratiques
	•	Python : 3.10+ ; typing strict ; docstrings (Google/Numpy) ; fonctions pures quand possible.
	•	Erreurs : gérer avec try/except + logs ; feedback clair à l’utilisateur (Alert FR) sans stacktrace.
	•	Chemins : agnostiques du CWD (utilise REPO_ROOT/absolus dans scripts), pas de chemins relatifs fragiles.
	•	Logs : logging.getLogger(__name__) ; niveau INFO/ERROR ; timestamps.
	•	Sécurité : jamais de clés API en clair ni en log ; .env non versionné ; .gitignore protège data/, logs/ si volumineux.
	•	Perf/UX : pas de calcul lourd dans les callbacks Dash → pré-calcul côté agent et lis la sortie.

⸻

✍️ Commits & Docs
	•	Conventional Commits :
	•	feat(ui): add forecasts page with table & filters
	•	fix(agents): handle missing final.parquet gracefully
	•	chore(ops): improve dash-smoke checks
	•	docs(readme): update dash overview
	•	Atomiques : 1 commit = 1 étape visible (crée page → branche les données → ajoute filtres → tests).
	•	Documentation :
	•	Mets à jour docs/PROGRESS.md (Delivered / In progress / Next + comment valider : URLs + commandes).
	•	Documente rapidement dans docs/architecture/dash_overview.md toute nouvelle page/agent (I/O + fichiers lus/écrits).

⸻

🧯 Dépannage rapide (FAQ)
	•	Port 8050 occupé : make dash-stop puis make dash-start-bg
	•	Manque numpy/requests : active .venv + pip install -r requirements.txt
	•	Pas de final.parquet : make equity-forecast && make forecast-aggregate
	•	Badge global reste 🟡 : relance make update-monitor (fraîcheur > 25h)
	•	MCP erreur “file undefined” : vérifier ops/ui/mcp_dash_smoke.mjs (génération des noms de fichiers & répertoires de sortie), relancer avec stderr visible (déjà configuré)
	•	UI ne reflète pas le dernier code : make dash-restart-bg + vérifier make dash-status et make dash-logs

⸻

✅ Definition of Done (DoD)
	•	Fonction réellement implémentée (pas seulement docs) et accessible via URL locale.
	•	Smoke OK (make dash-smoke) et MCP OK (make dash-mcp-test sans erreur bloquante).
	•	Observability vert et partitions du jour présentes.
	•	Docs à jour (PROGRESS + dash_overview) avec étapes de validation.
	•	Commits propres, atomiques, conventionnels.

⸻

Garde ce prompt ouvert dans l’IDE (pane Markdown) et suis-le à la lettre. S’il manque une donnée, affiche un état vide FR et note le besoin dans docs/PROGRESS.md. Lorsque tu termines une étape : tests → commit → push → PROGRESS.md.