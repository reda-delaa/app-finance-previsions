from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import logging

import sys
_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from research import web_navigator as wn


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + 'Z'


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: List[str] = []
    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:
            pass


def run_probe(n_runs: int = 20, sleep_s: float = 1.0) -> Dict[str, Any]:
    logger = logging.getLogger('web_navigator')
    logger.setLevel(logging.DEBUG)
    handler = _ListHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s|%(message)s'))
    logger.addHandler(handler)

    queries = [
        'site:reuters.com Fed inflation outlook',
        'site:ft.com earnings outlook',
        'US CPI release date',
        'ECB meeting schedule',
        'gold price forecast',
        'Treasury yields 10y',
        'AAPL earnings forecast',
        'NVDA guidance next quarter',
    ]

    results: List[Dict[str, Any]] = []
    success = 0
    failures = 0
    instances_hits: Dict[str, int] = {}
    rate_limits = 0
    forbids = 0
    non_json = 0

    for i in range(n_runs):
        q = queries[i % len(queries)]
        t0 = time.time()
        ok = False
        used_instance = None
        err = None
        try:
            out = wn.search_searxng(query=q, num=8, engines=wn.SEARXNG_DEFAULT_ENGINES, logger_=logger)
            items = (out or {}).get('results') or []
            ok = bool(items)
        except Exception as e:
            err = str(e)
            ok = False
        dt = time.time() - t0

        # inspect recent log lines to find instance or error categories
        recent = handler.records[-12:]
        for ln in reversed(recent):
            msg = ln.split('|', 1)[-1]
            if 'Succeeded on ' in msg:
                # e.g., Succeeded on https://example via POST.
                try:
                    base = msg.split('Succeeded on ', 1)[1].split(' ', 1)[0]
                    used_instance = base.strip()
                    instances_hits[used_instance] = instances_hits.get(used_instance, 0) + 1
                    break
                except Exception:
                    pass
            if '429' in msg:
                rate_limits += 1
            if '403' in msg:
                forbids += 1
            if 'non-JSON' in msg:
                non_json += 1

        if ok:
            success += 1
        else:
            failures += 1
        results.append({'q': q, 'ok': ok, 'dt_s': round(dt, 3), 'instance': used_instance, 'err': err})
        time.sleep(max(0.0, float(sleep_s)))

    rep = {
        'asof': _now_iso(),
        'runs': n_runs,
        'success': success,
        'failures': failures,
        'rate_limits': rate_limits,
        'forbidden': forbids,
        'non_json': non_json,
        'instances': instances_hits,
        'results': results,
    }
    outdir = Path('data/reports') / f"dt={datetime.utcnow().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / 'searxng_probe.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    return rep


def main() -> int:
    import argparse, os
    pa = argparse.ArgumentParser(description='Probe SearXNG public instances reliability')
    pa.add_argument('--runs', type=int, default=int(os.getenv('SEARX_PROBE_RUNS', 20)))
    pa.add_argument('--sleep', type=float, default=float(os.getenv('SEARX_PROBE_SLEEP', 0.5)))
    args = pa.parse_args()
    rep = run_probe(n_runs=args.runs, sleep_s=args.sleep)
    print(json.dumps({'ok': True, 'runs': args.runs, 'success': rep['success'], 'failures': rep['failures'], 'instances': rep['instances']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
