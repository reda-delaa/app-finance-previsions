from __future__ import annotations
# AJOUT
from core_runtime import configure_logging, new_trace_id, set_trace_id, get_trace_id
configure_logging()  # force notre handler JSON sur le root

import os, sys, logging, warnings, atexit
from logging import Handler, LogRecord
from loguru import logger as _loguru
from pathlib import Path
import coloredlogs

# --- Dossier / fichier
ROOT = Path(__file__).resolve().parents[2]  # .../analyse-financiere
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "hub_app.log"

# --- Intercepteur: route tous les logs stdlib -> loguru
class InterceptHandler(Handler):
    def emit(self, record: LogRecord) -> None:
        try:
            level = _loguru.level(record.levelname).name
        except Exception:
            level = record.levelno
        # Trouver le bon depth pour afficher la bonne origine
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _loguru.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

_CONFIGURED = False

def setup_logging(level: str | int = "DEBUG"):
    """Idempotent: configure Loguru + pont stdlib + capture warnings."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    # 1) Loguru: purge sinks par défaut
    _loguru.remove()

    # 2) Console colorée (dev)
    #    coloredlogs ne gère pas loguru directement, mais on garde un format loguru lisible + couleurs terminal
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    _loguru.add(sys.stdout, level=level, enqueue=True, backtrace=False, diagnose=False, format=console_format)

    # 3) Fichier tournant (rotation + retention)
    file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    _loguru.add(
        LOG_FILE,
        level=level,
        rotation="5 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=file_format,
    )

    # 4) Rediriger stdlib logging -> loguru (pour caplog pytest + libs tierces)
    logging.root.handlers = []
    logging.root.setLevel(logging.DEBUG)  # capter tout, filtrage côté sinks
    logging.root.propagate = False  # Prevent propagation of sub-module loggers to root
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG)

    # calmer le bruit de libs
    for noisy in [
        "asyncio", "urllib3", "requests", "yfinance", "matplotlib", "PIL",
        "peewee", "watchdog", "watchdog.observers", "watchdog.observers.fsevents",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 5) Capturer warnings Python en logs (IMPORTANT pour tes tests)
    warnings.simplefilter("default")
    logging.captureWarnings(True)   # Warning->logging (logger 'py.warnings')

    # 6) atexit: flush propre
    @atexit.register
    def _shutdown():
        try:
            _loguru.complete()
        except Exception:
            pass

    _CONFIGURED = True

# Helper public pour obtenir un logger nommé "proprement"
def get_logger(name: str = "hub"):
    setup_logging()
    # Ne plus utiliser bind() car cela casse le formatage %s avec args positionnels
    # Retourner plutôt un wrapper qui utilise le logger de base
    class BoundLogger:
        def __init__(self, name):
            self.name = name

        def log(self, level, message, *args, **kwargs):
            # Use main logger directly to preserve colors perfectly
            # The tradeoff is all loggers will show the same module name, but colors work

            # Si pas d'args, logger le message directement
            if not args:
                _loguru.log(level, message)
            else:
                # Avec args, utiliser le formatage comme Python standard logging
                try:
                    formatted_message = message % args if args else message
                    _loguru.log(level, formatted_message)
                except (TypeError, ValueError):
                    # Fallback si le formatage %s échoue
                    _loguru.log(level, message)

        def debug(self, message, *args, **kwargs):
            self.log("DEBUG", message, *args, **kwargs)

        def info(self, message, *args, **kwargs):
            self.log("INFO", message, *args, **kwargs)

        def warning(self, message, *args, **kwargs):
            self.log("WARNING", message, *args, **kwargs)

        def error(self, message, *args, **kwargs):
            self.log("ERROR", message, *args, **kwargs)

        def critical(self, message, *args, **kwargs):
            self.log("CRITICAL", message, *args, **kwargs)

        def exception(self, message, *args, **kwargs):
            self.log("ERROR", message, *args, **kwargs)

    return BoundLogger(name)

def ensure_trace():
    if get_trace_id() is None:
        new_trace_id()

# Backward compatibility - wrapper for old configure_logging function
def configure_logging(level=logging.INFO, logfile="logs/hub_app.log"):
    """Wrapper for backward compatibility with existing code."""
    setup_logging(level)
    # Return empty function for compatibility - original function returned nothing
    return lambda: None
