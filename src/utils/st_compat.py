# src/utils/st_compat.py
from __future__ import annotations
import types

class _SessionState:
    def __init__(self):
        self._data = {}
    # mapping-style
    def __contains__(self, key):
        return key in self._data
    def __getitem__(self, key):
        return self._data[key]
    def __setitem__(self, key, value):
        self._data[key] = value
    # attribute-style
    def __getattr__(self, name):
        if name == "_data":
            return super().__getattribute__(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        if name == "_data":
            return super().__setattr__(name, value)
        self._data[name] = value

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
    d.session_state = _SessionState()
    def _noop(*a, **k): pass
    # attributs/func basiques utilisés dans ton code
    for name in [
        "set_page_config", "title", "sidebar", "write", "warning",
        "error", "info", "success", "caption", "code", "json", "text_input",
        "text_area", "slider", "button", "checkbox", "subheader", "header", "tabs", "expander",
        "plotly_chart"
    ]:
        setattr(d, name, _noop)

    # sous-objets minimaux
    class _Sidebar:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def checkbox(self, *a, **k): return False
        def write(self, *a, **k): pass
        def expander(self, *a, **k): return _DummyCtx()
    d.sidebar = _Sidebar()

    # composants qui renvoient un "context manager"
    d.expander = lambda *a, **k: _DummyCtx()
    d.tabs     = lambda labels: [ _DummyCtx() for _ in labels ]
    def _columns(spec, **k):
        try:
            n = int(spec)
        except Exception:
            try:
                n = len(spec)
            except Exception:
                n = 1
        return [ _DummyCtx() for _ in range(n) ]
    d.columns  = _columns
    d.multiselect = lambda label, options, default=None, **k: (default or [])
    d.selectbox = lambda label, options, index=0, format_func=lambda x: x, **k: (options[index] if options else None)

    # cache_data decorator (no-op)
    def _cache_data(**kwargs):
        def _deco(fn):
            return fn
        return _deco
    d.cache_data = _cache_data

    return d

class _DummyCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
