Legacy (non‑runtime) assets

Purpose
- Keep historical shims/tests that are not part of the application runtime.
- Avoid confusion by excluding this directory from default pytest collection (`pytest.ini: norecursedirs=ops`).

Contents
- `searxng-local/finnews.py`: thin compatibility wrapper used by legacy tests.
- `tests/test_finnews_import.py`: verifies the shim’s minimal API; not run by default.

Notes
- The runtime uses `src/research/web_navigator.py` for SearXNG (public instances), not this shim.
- If a local SearXNG instance is needed in the future, integrate it as an on‑demand tool (e.g., MCP) under `ops/` instead of `src/`.

