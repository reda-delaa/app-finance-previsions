from __future__ import annotations

import os
import sys
import time
import requests


def check(path: str, base: str) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    r = requests.get(base + path, timeout=3)
    ms = (time.perf_counter() - t0) * 1000
    return (path, r.status_code, ms)


def main() -> int:
    base = os.getenv("DASH_BASE", f"http://127.0.0.1:{os.getenv('AF_DASH_PORT', '8050')}")
    targets = ["/", "/dashboard", "/signals", "/portfolio", "/agents", "/observability", "/regimes", "/risk", "/recession"]
    ok = True
    for p in targets:
        try:
            path, code, ms = check(p, base)
            print(f"{path}: {code} ({ms:.0f} ms)")
            ok = ok and (code == 200)
        except Exception as e:
            print(f"{p}: ERROR {e}")
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
