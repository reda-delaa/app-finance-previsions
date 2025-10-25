from __future__ import annotations

from src.agents.earnings_calendar_agent import run


if __name__ == '__main__':
    p = run()
    print({'ok': True, 'path': str(p)})

