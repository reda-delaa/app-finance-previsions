# src/core/warn_log.py
"""
Helper for better warning capture in tests via caplog
"""

import logging
import warnings
from typing import Type


def warn_log(logger: logging.Logger,
              name: str,
              category: type[Warning],
              msg: str):
    """
    Helper function to log warnings that can be captured by both:
    - caplog (pytest)
    - warnings system

    Args:
        logger: Logger instance
        name: Context name for the warning
        category: Warning category class
        msg: Warning message
    """
    # 1) Emit Python warning (caplog can capture this)
    warnings.warn(f"[{name}] {msg}", category)

    # 2) Emit logger warning (normal logging)
    logger.warning("WARN %s: %s", name, msg)
