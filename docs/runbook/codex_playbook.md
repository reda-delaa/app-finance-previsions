# Codex Playbook (MCP)

Prérequis
- Exporter les clés: `TAVILY_API_KEY`, `FIRECRAWL_API_KEY`, `SERPER_API_KEY`, (optionnel) `FINNHUB_API_KEY`.
- Lancer Codex dans le repo: `codex --cd /Users/venom/Documents/analyse-financiere --sandbox danger-full-access --ask-for-approval never`.
- Config MCP: `~/.codex/config.toml` (browser @playwright/mcp, architecture‑analyzer, filesystem, memory, mermaid, sqlite, spec‑workflow, taskmanager, serper/tavily/firecrawl, FRED/Finnhub).
- Première exécution: Playwright MCP télécharge les navigateurs (Node requis).

Étapes recommandées
- Exigences (spec-workflow-mcp + memory)
  - Prompt: « Capture Vision, Personas, EPICs/US + AC pour l’app; persiste dans Memory MCP et écris docs/product/backlog.md ».
- Architecture (architecture-analyzer-mcp + mcp-mermaid)
  - Prompt: « Analyse couplages/cycles, propose couches; génère C4 et sauvegarde docs/architecture/c4.md ».
- Données (FRED/Finnhub/News)
  - Prompt: « Liste datasets macro requis (FRED), champs quotes/fundamentaux (Finnhub), pipeline News; propose contrats Pydantic ».
- UI (filesystem + sequential-thinking)
  - Prompt: « Conçois pages Streamlit, composants communs, plan de refactor; créer squelettes de pages sans casser l’existant ».
- Prévisions & Backtest
  - Prompt: « Conçois flux “forecast + backtest” modulaires, inputs/outputs, métriques, et tests de non-régression ».
 - Ops & CI (security/ui‑smoke)
   - Prompt: « Ajoute cibles Makefile ui‑start/stop/restart/watch et ui‑smoke; pipeline CI: tests + ui smoke + security audit; docs PROGRESS.md ».

Notes
- Utiliser Browser MCP pour audits rapides de sources/benchmarks.
- Tenir le backlog à jour à chaque itération de refactor.
- Toujours redémarrer l’UI après changement d’interface: `make ui-restart` (port 5555).
