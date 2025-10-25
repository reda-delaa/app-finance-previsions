"""
Data Quality Scanner — checks datasets and writes a report.

Scans:
- News parquet (columns, NaNs, duplicates)
- Macro series parquet (presence, recency)
- Prices parquet (positivity, outliers)
- Forecasts parquet (schema, NaNs, ranges)
- Features parquet (presence, numeric columns)
- Events JSON (presence, recency)

Writes: data/quality/dt=YYYYMMDD/report.json
CLI:   python -m src.agents.data_quality --scan
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import os

import pandas as pd


def _today_dt() -> str:
    return datetime.utcnow().strftime('%Y%m%d')


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _glob(paths: str) -> List[Path]:
    return [Path(p) for p in Path().glob(paths)]


def scan_news(days_back: int = 30) -> Dict[str, Any]:
    base = Path('data/news')
    issues: List[Dict[str, Any]] = []
    if not base.exists():
        return {"ok": False, "issues": [{"sev": "warn", "msg": "data/news absent"}]}
    import datetime as dt
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    files: List[Path] = []
    for d in pd.date_range(start, end):
        files += list((base / f"dt={d.date()}").glob("*.parquet"))
    if not files:
        return {"ok": False, "issues": [{"sev": "warn", "msg": "aucun parquet news sur la période"}]}
    sample = pd.concat([_safe_read_parquet(f).head(1000) for f in files[:20] if f.exists()], ignore_index=True) if files else pd.DataFrame()
    if sample.empty:
        issues.append({"sev": "warn", "msg": "news parquet lisible mais vide"})
    else:
        needed = ["ts","source","title","link","summary"]
        for c in needed:
            if c not in sample.columns:
                issues.append({"sev": "error", "msg": f"colonne manquante: {c}"})
        if 'link' in sample.columns:
            dups = int(sample['link'].dropna().duplicated().sum())
            if dups > 0:
                issues.append({"sev": "info", "msg": f"liens dupliqués: {dups}"})
        null_rate = float(sample.isna().mean().mean()) if not sample.empty else 0.0
        if null_rate > 0.4:
            issues.append({"sev": "warn", "msg": f"taux de valeurs manquantes élevé: {null_rate:.2f}"})
    return {"ok": not any(i['sev']=='error' for i in issues), "issues": issues}


def scan_macro(recency_days: int = 7) -> Dict[str, Any]:
    base = Path('data/macro')
    issues: List[Dict[str, Any]] = []
    if not base.exists():
        return {"ok": False, "issues": [{"sev": "warn", "msg": "data/macro absent"}]}
    series_dirs = list(base.glob('series_id=*/series.parquet'))
    if not series_dirs:
        return {"ok": False, "issues": [{"sev": "warn", "msg": "aucune série FRED trouvée"}]}
    # Check recency for a few key series
    for sid in ['DTWEXBGS','DGS10','CPIAUCSL']:
        p = base / f"series_id={sid}" / "series.parquet"
        if not p.exists():
            issues.append({"sev": "warn", "msg": f"série absente: {sid}"})
            continue
        df = _safe_read_parquet(p)
        if df.empty:
            issues.append({"sev": "warn", "msg": f"série vide: {sid}"})
            continue
        try:
            last = pd.to_datetime(df['date'] if 'date' in df.columns else df.index).max()
            if (datetime.utcnow() - pd.to_datetime(last).to_pydatetime()).days > recency_days*6:
                issues.append({"sev": "info", "msg": f"série {sid} peu récente (ok si fréquence mensuelle)"})
        except Exception:
            pass
    return {"ok": not any(i['sev']=='error' for i in issues), "issues": issues}


def scan_prices() -> Dict[str, Any]:
    base = Path('data/prices')
    issues: List[Dict[str, Any]] = []
    if not base.exists():
        return {"ok": False, "issues": [{"sev": "warn", "msg": "data/prices absent"}]}
    for p in base.glob('ticker=*/prices.parquet'):
        df = _safe_read_parquet(p)
        if df.empty:
            issues.append({"sev": "warn", "msg": f"vide: {p.parent.name}"})
            continue
        if 'Close' in df.columns and (df['Close'] <= 0).any():
            issues.append({"sev": "error", "msg": f"prix non positifs: {p.parent.name}"})
    return {"ok": not any(i['sev']=='error' for i in issues), "issues": issues}


def scan_forecasts() -> Dict[str, Any]:
    parts = list(Path('data/forecast').glob('dt=*/forecasts.parquet'))
    issues: List[Dict[str, Any]] = []
    if not parts:
        return {"ok": False, "issues": [{"sev": "warn", "msg": "aucun forecasts.parquet"}]}
    df = pd.concat([_safe_read_parquet(p).head(2000) for p in parts[-5:]], ignore_index=True)
    need_cols = ["dt","ticker","horizon","direction","confidence","expected_return"]
    for c in need_cols:
        if c not in df.columns:
            issues.append({"sev": "error", "msg": f"colonne manquante: {c}"})
    try:
        bad_conf = int(((df['confidence']<0)|(df['confidence']>1)).sum())
        if bad_conf:
            issues.append({"sev": "warn", "msg": f"confidence hors [0,1]: {bad_conf}"})
    except Exception:
        pass
    try:
        bad_er = int(((df['expected_return']<-0.5)|(df['expected_return']>0.5)).sum())
        if bad_er:
            issues.append({"sev": "info", "msg": f"expected_return extrême: {bad_er}"})
    except Exception:
        pass
    return {"ok": not any(i['sev']=='error' for i in issues), "issues": issues}


def scan_features() -> Dict[str, Any]:
    parts = list(Path('data/features').glob('dt=*/features_flat.parquet'))
    issues: List[Dict[str, Any]] = []
    if not parts:
        return {"ok": False, "issues": [{"sev": "warn", "msg": "aucun features_flat.parquet"}]}
    df = _safe_read_parquet(parts[-1])
    nums = ["news_count","mean_sentiment","y_pe","y_beta"]
    for c in nums:
        if c not in df.columns:
            issues.append({"sev": "info", "msg": f"colonne feature absente: {c}"})
    return {"ok": not any(i['sev']=='error' for i in issues), "issues": issues}


def scan_events(recency_days: int = 14) -> Dict[str, Any]:
    parts = sorted(Path('data/events').glob('dt=*/events.json'))
    issues: List[Dict[str, Any]] = []
    if not parts:
        return {"ok": False, "issues": [{"sev": "warn", "msg": "aucun events.json"}]}
    try:
        last = parts[-1]
        obj = json.loads(last.read_text(encoding='utf-8'))
        evs = obj.get('events') or []
        if not evs:
            issues.append({"sev": "info", "msg": "aucun événement dans la fenêtre"})
    except Exception as e:
        issues.append({"sev": "error", "msg": f"lecture échouée: {e}"})
    return {"ok": not any(i['sev']=='error' for i in issues), "issues": issues}


def scan_all() -> Dict[str, Any]:
    out = {
        'asof': datetime.utcnow().isoformat()+'Z',
        'news': scan_news(),
        'macro': scan_macro(),
        'prices': scan_prices(),
        'forecasts': scan_forecasts(),
        'features': scan_features(),
        'events': scan_events(),
    }
    # Freshness checks (simple age thresholds)
    try:
        freshness: Dict[str, Any] = {"issues": []}
        # news: expect something in last 3 days
        import pandas as _pd
        from glob import glob
        from datetime import timedelta as _td
        now = datetime.utcnow()
        # news
        news_parts = sorted(Path('data/news').glob('dt=*'))
        if news_parts:
            last = news_parts[-1].name.replace('dt=','')
            try:
                last_dt = _pd.to_datetime(last).to_pydatetime()
                if (now - last_dt).days > 3:
                    freshness['issues'].append({'sev':'warn','msg':'news partitions older than 3 days'})
            except Exception:
                pass
        # forecasts
        fc_parts = sorted(Path('data/forecast').glob('dt=*'))
        if fc_parts:
            last = fc_parts[-1].name.replace('dt=','')
            try:
                last_dt = _pd.to_datetime(last).to_pydatetime()
                if (now - last_dt).days > 2:
                    freshness['issues'].append({'sev':'warn','msg':'forecasts partitions older than 2 days'})
            except Exception:
                pass
        out['freshness'] = {'ok': len(freshness['issues'])==0, 'issues': freshness['issues']}
    except Exception:
        out['freshness'] = {'ok': True, 'issues': []}
    # Coverage check (≥5y)
    try:
        out['coverage'] = scan_coverage(min_years=int(os.getenv('COVERAGE_MIN_YEARS','5')))
    except Exception:
        out['coverage'] = {'ok': True, 'issues': []}
    ok = all(v.get('ok', False) for v in out.values() if isinstance(v, dict))
    out['ok'] = ok
    return out

def scan_coverage(min_years: int = 5) -> Dict[str, Any]:
    """Check that prices and selected macro series cover at least min_years."""
    issues = []
    import pandas as pd
    min_days = int(min_years * 365 * 0.98)  # allow small slack
    today = pd.Timestamp.utcnow().normalize()
    # prices
    for p in Path('data/prices').glob('ticker=*/prices.parquet'):
        try:
            df = pd.read_parquet(p)
            col_date = 'date' if 'date' in df.columns else None
            if col_date:
                df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
                dmin, dmax = df[col_date].min(), df[col_date].max()
                if pd.isna(dmin) or pd.isna(dmax):
                    continue
                span = (dmax - dmin).days
                if span < min_days:
                    issues.append({'sev':'warn','msg': f'coverage< {min_years}y for {p.parent.name} ({span} days)'})
        except Exception:
            continue
    # macro core series
    core = ['DGS10','DGS2','CPIAUCSL']
    for sid in core:
        sp = Path('data/macro')/f'series_id={sid}'/'series.parquet'
        if not sp.exists():
            issues.append({'sev':'warn','msg': f'macro series missing: {sid}'})
            continue
        try:
            df = pd.read_parquet(sp)
            col_date = 'date' if 'date' in df.columns else None
            if col_date:
                df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
                dmin, dmax = df[col_date].min(), df[col_date].max()
                span = (dmax - dmin).days if dmin is not None and dmax is not None else 0
                if span < min_days:
                    issues.append({'sev':'warn','msg': f'macro {sid} coverage< {min_years}y ({span} days)'})
        except Exception:
            continue
    return {'ok': len([i for i in issues if i.get('sev')=='error'])==0, 'issues': issues}


def write_report(obj: Dict[str, Any]) -> Path:
    outdir = Path('data/quality')/f"dt={_today_dt()}"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir/'report.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


def main(argv: List[str] | None = None) -> int:
    import argparse
    pa = argparse.ArgumentParser(description='Data Quality Scanner')
    pa.add_argument('--scan', action='store_true', help='Run scan and write report')
    args = pa.parse_args(argv)
    if args.scan:
        rep = scan_all()
        p = write_report(rep)
        print(json.dumps({'ok': rep.get('ok'), 'path': str(p)}, ensure_ascii=False))
        return 0 if rep.get('ok') else 1
    print('Use --scan to run checks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
