# DOC pour l’Agent QA (ATLAS)

Ce guide décrit la procédure de test UI à appliquer à chaque sprint pour valider les dernières modifications et prioriser la suite.

## 1) Vérifier les derniers commits
- Ouvrir GitHub: `https://github.com/DelaaReda/app-finance-previsions/commits/main`.
- Lire messages/horodatages, noter les zones impactées (pages, scripts, Observability, etc.).
- Pour un commit précis, ouvrir le diff et identifier ce qui doit être testé en UI.

## 2) Redémarrer l’UI (toujours après mise à jour)
- Ligne de commande (racine du dépôt):
```bash
make ui-restart-bg   # redémarre l’UI en arrière-plan et log sous logs/ui
make ui-status       # affiche port, PID, extrait du log
```
- Via Observability (UI):
  1. Ouvrir Observability (menu).
  2. Déplier « Action (Admin) — Redémarrer l’UI ».
  3. Cocher la confirmation, cliquer Redémarrer (bg).
  4. Attendre la sonde (jusqu’à 15 tentatives). Vérifier « UI — Santé ».

## 3) Accéder à l’interface
- URL: `http://localhost:5555` (ou `http://localhost:5555/Dashboard`).
- Si la page tarde, attendre quelques secondes; en cas d’erreur, vérifier le redémarrage (étape 2).

## 4) Tester Observability et contrôles UI
- Observability → « UI — Santé »: port, PID, Process vivant, HTTP 200 (latence).
- Redémarrer / Démarrer / Arrêter l’UI:
  - Redémarrer (bg) via panneau repliable, confirmation requise.
  - Démarrer (bg) si aucune instance n’écoute, avec confirmation.
  - Arrêter l’UI avec confirmation.
  - Les scripts tournent en bg; sortie affichée dans l’UI; logs sous `logs/ui`.
- Clés d’API: tableau « présence seulement », sans exposer de valeurs.

## 5) Tester les autres pages
- Dashboard: s’affiche sans invites techniques. Si pas de données, message d’état vide (pas d’instruction shell) et pointer Admin → Agents Status.
- Deep Dive / Forecasts / Signals / Portfolio / Alerts: pas de pages vides; pas d’instructions shell; états vides sûrs si nécessaire.
- Regimes / Risk / Recession: sélecteur de date en page; messages clairs si aucune partition.
- News / Reports: JSON bruts sous expanders; résumé lisible en premier.
- Backtests / Evaluation: chargement et état vide clairs (si pas de données).

## 6) Synthèse et priorisation
- Quelles pages OK ?
- JSON bruts visibles non désirés ?
- Erreurs (fichier manquant, process non détecté) ?
- Start/Stop/Restart (UI) OK avec confirmation ?

Produire une liste courte de priorités: suppression des invites techniques restantes, complétion des états vides, agents F4F à implémenter (équité, macro, agrégateur, update/quality), améliorations Observability (live log/badge).

---

En appliquant cette procédure à chaque itération (commits → restart UI → tests UI → bilan), on garantit que l’UI est évaluée sur la dernière version et que la priorisation reflète l’état réel du produit.
