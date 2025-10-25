from __future__ import annotations

from pathlib import Path
import sys as _sys
import streamlit as st

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))


def render_top_nav(active: str | None = None) -> None:
    """Render a simple top navigation bar separating User vs Admin sections.

    active: optional key ("user" or "admin") to highlight the section.
    """
    st.markdown(
        """
        <style>
        .af-navbar {background: #1B1F2A; padding: 0.5rem 0.8rem; border-bottom: 1px solid #2E77D0;
                    position: sticky; top: 0; z-index: 999;}
        .af-nav-title {font-weight: 600; margin-right: 1rem; color: #E5E7EB;}
        .af-pill {display:inline-block; padding: 0.25rem 0.6rem; margin-right: 0.4rem; border-radius: 9999px; font-size: 0.9rem;}
        .af-pill.user {background: #22314a; color: #dce8ff;}
        .af-pill.admin {background: #2a1f1f; color: #ffdcdc;}
        .af-active {outline: 2px solid #2E77D0;}
        .af-links a {text-decoration: none; margin-right: 0.6rem; color: #cbd5e1;}
        .af-links a:hover {color: #ffffff;}
        /* add top padding so content doesn't jump under sticky navbar */
        section.main > div.block-container {padding-top: 0.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown('<div class="af-navbar">', unsafe_allow_html=True)
        cols = st.columns([1.2, 3, 6])
        with cols[0]:
            st.markdown('<span class="af-nav-title">Finance Agent</span>', unsafe_allow_html=True)
        with cols[1]:
            u_cls = "af-pill user af-active" if active == "user" else "af-pill user"
            a_cls = "af-pill admin af-active" if active == "admin" else "af-pill admin"
            c1, c2 = st.columns([1,1])
            with c1:
                st.page_link("pages/00_Home.py", label="🏠 Home")
            with c2:
                st.markdown(f'<span class="{u_cls}">Prévisions</span> <span class="{a_cls}">Admin</span>', unsafe_allow_html=True)
        with cols[2]:
            l1, l2, l3, l4 = st.columns([1,1,1,3])
            with l1: st.page_link("pages/1_Dashboard.py", label="📊 Dashboard")
            with l2: st.page_link("pages/4_Forecasts.py", label="📈 Forecasts")
            with l3: st.page_link("pages/23_Settings.py", label="⚙ Settings")
            with l4: st.page_link("pages/27_Agents_Status.py", label="🛰 Agents")
        st.markdown('</div>', unsafe_allow_html=True)
