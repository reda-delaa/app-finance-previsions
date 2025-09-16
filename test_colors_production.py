#!/usr/bin/env python3
"""Test de coloriage des logs en production (même environnement que l'app Streamlit)"""

from pathlib import Path
import sys as _sys

# sys.path bootstrap EXACTEMENT comme dans l'app
_SRC_ROOT = Path(__file__).resolve().parents[1]   # .../analyse-financiere/src
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

# Simulation du setup comme dans app.py
print("=== Testing Color Fix Production ===")

# Import du setup logging
from hub.logging_setup import get_logger

# Test des logs
logger = get_logger("hub")

logger.info("Testing INFO message from test_colors")
logger.warning("Testing WARNING message from test_colors")
logger.error("Testing ERROR message from test_colors")

# Test du logger secondaire comme dans macroapp
macro_logger = get_logger("macroapp")
macro_logger.info("Testing INFO message from macroapp")
macro_logger.warning("Testing WARNING message from macroapp")
macro_logger.error("Testing ERROR message from macroapp")

print("✅ Color test completed - check the colored output above")
