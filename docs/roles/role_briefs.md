Role Briefs for Finance Agent

1) Investment Expert
- Objectives: Forecast 1w/1m/1y for indices, sectors (gold), and equities; outputs: P(up), expected_return, confidence, drivers.
- Signals: Macro (DXY, 2Y/10Y, real yields, CPI, growth, credit), Commodities (Gold/Silver/Copper), Fundamentals (cap, PE, beta, dividend; miners: AISC/leverage when available), Technicals (SMA20/50/200, RSI, MACD, vol), News/Sentiment (pos/neg ratios, events: earnings/M&A/geopolitics).
- Evaluation: hit‑rate, IR on ranked lists, calibration (Brier), explainable drivers consistency.
- Constraints: diversification, liquidity, event risk filters; refresh daily; intraday light refresh for news.

2) Architect
- Data Lake: Parquet + DuckDB (forecasts, features_flat, macro, news, prices, fundamentals).
- Ingestion: Harvester daemon (news live+backfill Tavily, macro FRED, prices/fundamentals cache, LLM investigations).
- Features: Standardized FeatureBundle; persist selected numeric features to features_flat.parquet.
- Forecasting: Baseline SMA+sentiment → add ML baseline (ridge/GBM) + LLM scenario blend; write ml_return/ml_conf.
- Serving: Streamlit multi‑page (Dashboard, News, Deep Dive, Forecasts, Backtests, Reports, Observability).
- Reliability: retries/backoff, rate‑limit caps, state + logs; deterministic backtests use cached prices.

3) Product Owner (PO)
- EPIC Daily Brief: Macro deltas + Top 10 picks with reasons; show on Dashboard.
- EPIC Forecast Quality: Train/evaluate ML baseline + blend; report IR/hit‑rate/calibration over time.
- EPIC News Intelligence: 1y backfill, topic trends (LLM), sector filters; parquet for fast scans.
- EPIC Backtests: Use cached prices; strategy templates; KPIs (CAGR, MaxDD, Sharpe, IR).
- EPIC Ops: Harvester daemonization; key health; cost/quotas; cache TTLs.

4) Integration Analyst
- Linkage: harvester → lake (news/macro/prices/funda) → features → forecasts → UI (Dash/DeepDive/Backtests/Reports).
- LLM usage: investigations + topic discovery attach macro deltas, watchlist, and this brief to guide outputs.
- Contracts: forecasts.parquet (dt,ticker,horizon,direction,confidence,expected_return,drivers_json,ml_*), features_flat.parquet (dt,ticker,selected numeric features), news parquet normalized.
- Scheduling: daily full run; intraday news/price refresh; backfill tasks guarded by quotas.

