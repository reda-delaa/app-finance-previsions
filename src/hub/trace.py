import functools, logging, json
logger = logging.getLogger("hub")

def _short(obj, limit=400):
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        s = str(obj)
    return (s[:limit] + "...") if len(s) > limit else s

def trace_call(name=None):
    def deco(fn):
        nm = name or fn.__name__
        @functools.wraps(fn)
        def wrap(*a, **k):
            logger.debug(f"→ {nm} args={_short(a)} kwargs={_short(k)}")
            res = fn(*a, **k)
            logger.debug(f"← {nm} result={_short(res)}")
            return res
        return wrap
    return deco
