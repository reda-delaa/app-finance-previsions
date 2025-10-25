from __future__ import annotations

from ui.nav import render_top_nav
from ui.footer import render_footer


def page_header(active: str | None = None) -> None:
    render_top_nav(active=active)


def page_footer() -> None:
    render_footer()

