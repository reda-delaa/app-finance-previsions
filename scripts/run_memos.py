from __future__ import annotations

from src.agents.investment_memo_agent import run_all

if __name__ == '__main__':
    paths = run_all()
    print({'ok': True, 'count': len(paths), 'paths': paths})

