from __future__ import annotations

from datetime import datetime
import os
import streamlit as st


def render_footer() -> None:
    port = os.getenv("AF_UI_PORT", "5555")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"""
        <style>
        .af-footer {{position: fixed; bottom: 0; left: 0; right: 0; background: #1B1F2A;
                     border-top: 1px solid #2E77D0; color: #9aa4af; font-size: 0.85rem;}}
        .af-footer .inner {{padding: 0.25rem 0.8rem; display: flex; gap: 1rem;}}
        .af-footer a {{color: #cbd5e1; text-decoration: none;}}
        .af-footer a:hover {{color: #ffffff; text-decoration: underline;}}
        /* add bottom padding so content not hidden under footer */
        section.main > div.block-container {{padding-bottom: 2.0rem;}}
        </style>
        <div class="af-footer"><div class="inner">
          <span>Port UI: {port}</span>
          <span>Horodatage: {ts}</span>
          <span><a href="/">Accueil</a></span>
        </div></div>
        """,
        unsafe_allow_html=True,
    )

