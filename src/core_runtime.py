# src/core_runtime.py
import json, sys, time, uuid, hashlib, logging, contextvars
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

# -------- Fingerprint + provenance (JSONL) --------
def _project_logs_path() -> Path:
    # src/core_runtime.py -> repo_root = parents[1]
    repo_root = Path(__file__).resolve().parents[1]
    # dossier logs dans le repo courant
    logs = (repo_root / "logs").resolve()
    logs.mkdir(parents=True, exist_ok=True)
    return logs

# Chemin du journal JSONL des datasets
_LOGS_DIR = _project_logs_path()
DATASET_LOG = _LOGS_DIR / "dataset_log.jsonl"

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
    """Append a provenance entry to a JSONL log instead of SQLite."""
    DATASET_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": int(time.time()),
        "dataset": dataset,
        "source_url": source_url,
        "status": status,
        "rows": int(meta.get("rows", 0) or 0),
        "min_date": meta.get("min_date"),
        "max_date": meta.get("max_date"),
        "checksum": meta.get("checksum"),
        "schema_version": schema_version,
        "trace_id": get_trace_id(),
    }
    try:
        with DATASET_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger("af").exception("write_entry failed", extra={"where": "write_entry"})


def get_dataset_log_latest() -> pd.DataFrame:
    """Return the latest entry per dataset from the JSONL log as a DataFrame."""
    cols = ["dataset","status","rows","min_date","max_date","ts","trace_id"]
    if not DATASET_LOG.exists():
        return pd.DataFrame(columns=cols)
    records = []
    try:
        with DATASET_LOG.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                r = {
                    "dataset": rec.get("dataset"),
                    "status": rec.get("status"),
                    "rows": rec.get("rows"),
                    "min_date": rec.get("min_date"),
                    "max_date": rec.get("max_date"),
                    "ts": rec.get("ts"),
                    "trace_id": rec.get("trace_id"),
                }
                if r["dataset"]:
                    records.append(r)
    except Exception:
        logging.getLogger("af").exception("read dataset_log.jsonl failed")
        return pd.DataFrame(columns=cols)

    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(records)
    # Keep latest per dataset
    df = df.sort_values("ts").groupby("dataset", as_index=False).tail(1)
    # Convert ts to datetime
    try:
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
    except Exception:
        pass
    return df[cols]
