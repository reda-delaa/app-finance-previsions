Project Docs — Finance Agent

Quickstart
- UI canonical port: `5555`. Start with `make ui-start`.
- After any UI change: `make ui-restart` (single instance policy).
- Background mode: `make ui-start-bg` / `make ui-restart-bg` (non‑blocking; logs under `logs/ui/streamlit_5555.log`).
- Check status/logs: `make ui-status` / `make ui-logs`.
- Keep UI always up: `make ui-watch` (auto‑restart if port is down).
- Optional search backend: `make searx-up` then `export SEARXNG_LOCAL_URL=http://localhost:8082`.

MCP & Codex CLI
- Codex config: `~/.codex/config.toml` (browser MCP, architecture analyzer, filesystem, memory, mermaid, sqlite, serper/tavily/firecrawl, spec‑workflow, taskmanager, FRED/Finnhub, Playwright MCP).
- First time: Node/npm required; Playwright MCP downloads browsers on first run.
- Runbook and prompts: `runbook/codex_playbook.md`.

Docs Map
- Product: `product/backlog.md` — EPICs, user stories, acceptance criteria.
- Architecture: `architecture/vision.md`, `architecture/c4.md`, `architecture/refactor_plan.md`.
- UI: `ui/ui_audit.md` — audit, decisions, and action plan.
- Progress: `PROGRESS.md` — what’s done, what’s next, run discipline.
- QA: `qa/ATLAS_QA.md` — procedure for ATLAS to verify commits, restart UI, test pages, and report.

Principles
- No central orchestrator in runtime UI. Pipelines run via `Makefile`/cron; UI reads latest partitions under `data/**/dt=YYYYMMDD/`.
- Safe UI by default: no shell/make prompts in user flows; admin‑only guidance lives in Agents Status/docs.
- French language first; consistent copy, friendly empty states, and confirmations after writes.
