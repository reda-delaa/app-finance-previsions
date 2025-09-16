# src/utils/st_compat.py
from __future__ import annotations
import types

def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def get_st():
    if _in_streamlit():
        import streamlit as _st
        return _st

    # Dummy "st" pour le mode bare/pytest : pas d'effets de bord
    d = types.SimpleNamespace()
    d.session_state = {}
    def _noop(*a, **k): pass
    # attributs/func basiques utilisés dans ton code
    for name in [
        "set_page_config", "title", "sidebar", "write", "warning",
        "error", "info", "success", "caption", "code", "json", "text_input",
        "slider", "button", "subheader", "tabs", "expander"
    ]:
        setattr(d, name, _noop)

    # sous-objets minimaux
    d.sidebar = types.SimpleNamespace(
        checkbox=_noop, write=_noop, expander=lambda *a, **k: _DummyCtx()
    )

    # composants qui renvoient un "context manager"
    d.expander = lambda *a, **k: _DummyCtx()
    d.tabs     = lambda labels: [ _DummyCtx() for _ in labels ]

    return d

class _DummyCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
