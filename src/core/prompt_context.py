from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def load_role_briefs() -> str:
    p = Path("docs/roles/role_briefs.md")
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""

@lru_cache(maxsize=1)
def load_next_steps() -> str:
    p = Path("docs/roles/next_steps.md")
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""

