# Agents — Architecture et extension

Principes
- Ne pas dupliquer: étendre agents existants si possible.
- Idempotence: sorties sous data/<domaine>/dt=YYYYMMDD/.
- Journalisation: logs sous logs/.

Agents existants (exemples)
- data_harvester, data_quality, g4f_model_watcher, earnings_calendar_agent, recession_agent, macro_regime_agent, risk_monitor_agent.

Nouveaux agents (proposés, F4F)
- equity_forecast_agent, commodity_forecast_agent, macro_forecast_agent, forecast_aggregator_agent, update_monitor_agent, explanation_memo_agent, backtest_agent, evaluation_agent, sentiment_agent.

Contrats de données
- Utiliser Pydantic/structures stables dans src/core/models.py; partitions datées.
