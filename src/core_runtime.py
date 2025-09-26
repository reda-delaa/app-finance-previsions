# src/core_runtime.py
import json, sys, time, uuid, sqlite3, hashlib, logging, contextvars
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------- Logging unifié + Correlation IDs + Tracing --------
_trace_id = contextvars.ContextVar("trace_id", default=None)
_span_id  = contextvars.ContextVar("span_id",  default=None)

def get_trace_id(): return _trace_id.get()
def set_trace_id(val): _trace_id.set(val); return val
def new_trace_id(): return set_trace_id(uuid.uuid4().hex)

def get_span_id(): return _span_id.get()
def set_span_id(val): _span_id.set(val); return val
def new_span_id(): return set_span_id(uuid.uuid4().hex[:16])

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "trace_id": getattr(record, "trace_id", None) or get_trace_id(),
            "span_id":  getattr(record, "span_id",  None) or get_span_id(),
        }
        # Merge any dict passed in extra={"ctx":{...}} or flat extras
        if hasattr(record, "ctx") and isinstance(record.ctx, dict):
            base.update(record.ctx)
        # Ajoute extras plats si présents (ticker, ui_page, ui_action, etc.)
        for k in ("ui_page","ui_action","ticker","url","where"):
            if hasattr(record, k):
                base[k] = getattr(record, k)
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def _json_formatter(record):
    base = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
        "level": record.levelname.lower(),
        "msg": record.getMessage(),
        "trace_id": get_trace_id(),
        "span_id": _span_id.get(),
        "logger": record.name,
        "module": record.module,
    }
    if record.exc_info:
        import traceback
        base["exc"] = "".join(traceback.format_exception(*record.exc_info))[-2000:]
    return base

class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # attache toujours trace/span même si non fournis
        if not hasattr(record, "trace_id") or record.trace_id is None:
            record.trace_id = get_trace_id()
        if not hasattr(record, "span_id") or record.span_id is None:
            # ne touche pas aux spans des libs, juste assure une valeur
            if _span_id.get() is None:
                _span_id.set(uuid.uuid4().hex[:16])
            record.span_id = _span_id.get()
        return True

class JsonHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            import json
            s = json.dumps(_json_formatter(record), ensure_ascii=False)
        except Exception:
            s = record.getMessage()
        self.stream.write(s + "\n")
        self.flush()

def configure_logging(level=logging.INFO):
    # un seul handler root, en JSON + capture warnings
    logging.captureWarnings(True)  # route warnings -> logging
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    h = JsonHandler(stream=sys.stdout)
    h.addFilter(CorrelationIdFilter())
    root.addHandler(h)
    # Option: réduire le bruit de libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("streamlit").propagate = True  # fait remonter dans root
    return logging.getLogger("af")

log = configure_logging()

@contextmanager
def with_span(name: str, **ctx):
    """Pour les fonctions backend (fetch_xxx, parse_xxx, etc.)."""
    parent = get_span_id()
    cur = new_span_id()
    log.info(f"{name}.start", extra={"span_id": cur, "ctx": ctx})
    try:
        yield
        log.info(f"{name}.ok", extra={"span_id": cur})
    except Exception:
        log.exception(f"{name}.fail", extra={"span_id": cur})
        raise
    finally:
        set_span_id(parent)

@contextmanager
def ui_event(action: str, ui_page: str = None, **ctx):
    """À utiliser dans l'UI : encadre chaque clic/chargement."""
    # si pas de trace encore, créer ici (séance/clic)
    if not get_trace_id():
        new_trace_id()
    parent = get_span_id()
    cur = new_span_id()
    extra = {"span_id": cur, "ui_action": action}
    if ui_page: extra["ui_page"] = ui_page
    if ctx:     extra["ctx"] = ctx
    log.info("ui_event.start", extra=extra)
    try:
        yield
        log.info("ui_event.ok", extra={"span_id": cur})
    except Exception:
        log.exception("ui_event.fail", extra={"span_id": cur})
        raise
    finally:
        set_span_id(parent)

# -------- Session HTTP robuste --------
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "AF/1.0 (+analyse-financiere)"})
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s

SESSION = make_session()

# -------- Fingerprint + provenance (SQLite) --------
# Le chemin de la base peut être surchargé par les variables d'env:
# - AF_DB_PATH: chemin fichier complet
# - AF_DB_DIR:  dossier contenant la base (fichier: af_provenance.sqlite)
# Par défaut on tente ~/.af_provenance.sqlite. Si non accessible (ex: sandbox),
# on bascule automatiquement vers <repo_root>/logs/af_provenance.sqlite.
def _compute_default_db_path() -> Path:
    # 1) Env overrides
    import os
    af_db_path = os.getenv("AF_DB_PATH")
    if af_db_path:
        return Path(af_db_path)
    af_db_dir = os.getenv("AF_DB_DIR")
    if af_db_dir:
        return Path(af_db_dir) / "af_provenance.sqlite"

    # 2) Home by default
    return Path.home() / ".af_provenance.sqlite"

def _project_logs_path() -> Path:
    # src/core_runtime.py -> repo_root = parents[1]
    repo_root = Path(__file__).resolve().parents[1]
    # dossier logs dans le repo courant
    logs = (repo_root / "logs").resolve()
    logs.mkdir(parents=True, exist_ok=True)
    return logs / "af_provenance.sqlite"

DB = _compute_default_db_path()

def _init_db():
    global DB
    target_paths = [DB]
    # En cas d'échec sur HOME, on tente le fallback projet
    fallback = _project_logs_path()
    if fallback not in target_paths:
        target_paths.append(fallback)

    last_err = None
    for path in target_paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as cx:
                cx.execute(
                    """CREATE TABLE IF NOT EXISTS dataset_log(
                        ts INTEGER, dataset TEXT, source_url TEXT, status TEXT,
                        rows INTEGER, min_date TEXT, max_date TEXT, checksum TEXT,
                        schema_version TEXT, trace_id TEXT
                    )"""
                )
            DB = path  # Utiliser ce chemin validé
            return
        except Exception as e:
            last_err = e
            continue
    # Si tout échoue, on remonte l'erreur originale
    raise RuntimeError(f"Impossible d'initialiser la base de provenance: {last_err}")

_init_db = _init_db()  # init au chargement

def df_fingerprint(df: pd.DataFrame):
    rows, cols = df.shape
    min_date = max_date = None
    if isinstance(df.index, pd.DatetimeIndex) and rows:
        min_date = df.index.min().strftime("%Y-%m-%d")
        max_date = df.index.max().strftime("%Y-%m-%d")
    h = hashlib.sha256()
    h.update((",".join(map(str, df.columns))).encode())
    if rows:
        try:
            arr = df.head(1000).select_dtypes(include=[np.number]).to_numpy(copy=False)
        except Exception:
            arr = df.head(1000).to_numpy(dtype=float, copy=False)
        h.update(np.nan_to_num(arr).tobytes())
    return {
        "rows": rows, "cols": cols,
        "min_date": min_date, "max_date": max_date,
        "checksum": h.hexdigest()
    }

def write_entry(dataset:str, source_url:str, status:str, meta:dict, schema_version:str):
    with sqlite3.connect(DB) as cx:
        cx.execute("""INSERT INTO dataset_log
            (ts,dataset,source_url,status,rows,min_date,max_date,checksum,schema_version,trace_id)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (int(time.time()), dataset, source_url, status,
             meta.get("rows",0), meta.get("min_date"), meta.get("max_date"),
             meta.get("checksum"), schema_version, get_trace_id()))
