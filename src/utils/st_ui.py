"""
UI helpers for Streamlit compatibility - unify chart display across apps
"""

from typing import Any
import streamlit as st

def show_fig(fig: Any) -> None:
    """
    Unified figure display helper that handles both Plotly and Matplotlib figures.
    Automatically uses correct width handling for Streamlit.
    """
    if hasattr(st, "plotly_chart") and fig is not None:
        # Check if it's a Plotly figure
        if fig.__class__.__name__.startswith(("Figure", "FigureWidget")):
            st.plotly_chart(fig, width="stretch")
        else:
            # Assume matplotlib
            st.pyplot(fig, width='content')

def get_st():
    """Get streamlit module with error handling."""
    return st
