# Data Flow — Prévisions & UI

Objectif: illustrer le flux de données et responsabilités sans orchestrateur central (Makefile/cron + UI en lecture).

```mermaid
flowchart LR
  subgraph External[External Data & Web]
    fred[FRED]
    finnhub[Finnhub]
    search[SearX / Serper]
    crawl[Tavily / Firecrawl]
  end
  subgraph Adapters[Adapters / Ingestion]
    ds1[src/analytics/data_sources/*]
    ing[src/ingestion/*]
  end
  subgraph Domain[Domain & Core]
    models[src/core/models.py]
    io[src/core/io_utils.py]
    config[src/core/config.py]
    mkt[src/core/market_data.py]
  end
  subgraph Agents[Forecast & Quality Agents]
    eq[src/agents/equity_forecast_agent]
    agg[src/agents/forecast_aggregator_agent]
    mac[src/agents/macro_forecast_agent]
    mon[src/agents/update_monitor_agent]
  end
  subgraph DataLake[Data Partitions]
    fcast["data/forecast/dt=YYYYMMDD\nforecasts.parquet, final.parquet"]
    macro["data/macro/forecast/dt=YYYYMMDD\nmacro_forecast.parquet"]
    qual["data/quality/dt=YYYYMMDD\nfreshness.json"]
  end
  subgraph UI[UI]
    st[Streamlit]
    dash[Dash]
  end
  user[(Users)]

  fred --> ds1
  finnhub --> ds1
  search --> ds1
  crawl --> ds1
  ds1 --> ing
  ing --> mkt
  mkt --> eq
  eq --> fcast
  fcast --> agg
  agg --> fcast
  mac --> macro
  mon --> qual
  fcast --> dash
  fcast --> st
  macro --> dash
  macro --> st
  qual --> dash
  qual --> st
  user --> st
  user --> dash
```

Notes
- Les agents écrivent des partitions immuables par date (idempotence par dt).
- L’UI lit la « dernière partition » par défaut mais peut sélectionner une date (US23).
- Pas d’orchestrateur: cibles Makefile/cron; UI ne déclenche pas de calculs réseau.
