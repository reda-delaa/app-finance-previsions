# Analyse Financière

Suite d'outils d'analyse financière combinant analyse fondamentale, technique, macroéconomique et sentiment.

## Structure du Projet

```
analyse-financiere/
├─ src/                    # Code principal
│  ├─ core/                # Briques génériques réutilisables
│  ├─ ingestion/          # Collecte de données
│  ├─ taxonomy/           # Classifications et lexiques
│  ├─ analytics/          # Logique d'analyse
│  ├─ apps/              # Applications Streamlit
│  ├─ research/          # Scripts exploratoires
│  └─ runners/           # Exécuteurs batch
└─ [autres dossiers...]
```

## Installation

1. Cloner le repository
```bash
git clone [URL_DU_REPO]
cd analyse-financiere
```

2. Créer et activer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
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
python -m pytest tests/
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
