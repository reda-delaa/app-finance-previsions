"""
Unified utils module.

Consolidates prior small modules under utils/ into a single import surface:
- get_st: Streamlit compatibility shim (dummy in non-Streamlit contexts)
- show_fig: helper to render Plotly/Matplotlib figures
- warn_once: warning helper emitted once per key
- get_cfg: compatibility shim to access project config and check API keys

Import examples:
    from utils import get_st, show_fig, warn_once, get_cfg
"""

from __future__ import annotations

import types
import warnings
import os
from typing import Any


# =============================
# Streamlit compatibility layer
# =============================

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
    """Return Streamlit module if running inside Streamlit, else a safe dummy.

    The dummy implements commonly used attributes to allow imports/tests to run
    without side effects when Streamlit runtime is not active.
    """
    if _in_streamlit():
        import streamlit as _st
        return _st

    # Dummy "st" for bare/pytest mode: no side effects
    d = types.SimpleNamespace()
    d.session_state = _SessionState()

    def _noop(*a, **k):
        return None

    # Basic attributes/functions used across the code
    for name in [
        "set_page_config", "title", "sidebar", "write", "warning",
        "error", "info", "success", "caption", "code", "json", "text_input",
        "text_area", "slider", "button", "checkbox", "subheader", "header",
        "tabs", "expander", "plotly_chart", "markdown", "pyplot", "dataframe",
        "metric",
    ]:
        setattr(d, name, _noop)

    # sub-objects minimal
    class _DummyCtx:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def __getattr__(self, name):
            def _noop_attr(*a, **k):
                return None
            return _noop_attr

    class _Sidebar:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def checkbox(self, *a, **k):
            return False
        def write(self, *a, **k):
            pass
        def expander(self, *a, **k):
            return _DummyCtx()

    d.sidebar = _Sidebar()
    d.expander = lambda *a, **k: _DummyCtx()
    d.spinner = lambda *a, **k: _DummyCtx()
    d.container = lambda *a, **k: _DummyCtx()
    d.tabs = lambda labels: [_DummyCtx() for _ in labels]

    def _columns(spec, **k):
        try:
            n = int(spec)
        except Exception:
            try:
                n = len(spec)
            except Exception:
                n = 1
        return [_DummyCtx() for _ in range(n)]

    d.columns = _columns
    d.multiselect = lambda label, options, default=None, **k: (default or [])
    d.selectbox = (
        lambda label, options, index=0, format_func=lambda x: x, **k:
        (options[index] if options else None)
    )

    # cache_data decorator (no-op)
    class _CacheData:
        def __call__(self, **kwargs):
            def _deco(fn):
                return fn
            return _deco
        def clear(self):
            pass

    d.cache_data = _CacheData()

    # misc widgets/utilities
    import datetime as _dt
    d.date_input = lambda label, value=None, **k: (value or _dt.datetime(2000, 1, 1))
    d.number_input = lambda label, value=0.0, **k: value
    d.rerun = lambda: None
    d.stop = lambda: None
    d._is_dummy = True

    return d


# ===============================
# Unified UI helpers (fig render)
# ===============================

def show_fig(fig: Any) -> None:
    """Unified figure display helper for Plotly/Matplotlib figures.

    Automatically uses correct width handling for Streamlit when available.
    """
    st = get_st()
    if fig is None:
        return
    # Plotly figures typically have class names starting with Figure/FigureWidget
    cls = getattr(fig, "__class__", type(None))
    name = getattr(cls, "__name__", "")
    if name.startswith(("Figure", "FigureWidget")) and hasattr(st, "plotly_chart"):
        st.plotly_chart(fig, width="stretch")
    else:
        # Assume matplotlib-like
        if hasattr(st, "pyplot"):
            st.pyplot(fig, width="content")


# ===========================
# Warn once helper (keyed)
# ===========================

_emitted_once = set()


def warn_once(logger, key: str, message: str, wcat=UserWarning):
    """Emit a warning only once per unique key and log it via the logger."""
    if key in _emitted_once:
        return
    _emitted_once.add(key)
    warnings.warn(message, wcat)
    try:
        logger.warning(message)
    except Exception:
        pass


# ==============================================
# Config compatibility (utils.get_cfg shim)
# ==============================================

def get_cfg():
    """Return a lightweight config proxy with has_any_fin_api().

    Bridges older imports (utils.config.get_cfg) to the current core.config.
    """
    try:
        from core.config import Config, config as _config  # type: ignore
    except Exception:
        # Fallback: derive from environment only
        class _Cfg:
            def has_any_fin_api(self) -> bool:
                return any([
                    os.getenv("FRED_API_KEY"),
                    os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY"),
                    os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_KEY"),
                    os.getenv("YAHOO_API_KEY"),
                    os.getenv("TE_USER") and os.getenv("TE_KEY"),
                ])

        return _Cfg()

    # Use instantiated config if available, else create one
    cfg_obj = _config if _config is not None else Config()

    class _CfgProxy:
        def __init__(self, inner):
            self._inner = inner

        def has_any_fin_api(self) -> bool:
            return any([
                bool(getattr(self._inner, "finnhub_key", None)),
                bool(getattr(self._inner, "alpha_vantage_key", None)),
                bool(getattr(self._inner, "yahoo_api_key", None)),
                bool(os.getenv("FRED_API_KEY")),
                bool(os.getenv("TE_USER") and os.getenv("TE_KEY")),
            ])

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _CfgProxy(cfg_obj)

