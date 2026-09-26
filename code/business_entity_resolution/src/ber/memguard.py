"""Memory guard: abort the process cleanly if its resident set size goes over a limit.

A daemon thread polls the current RSS (via `ps`, no extra dependency) and exits with status 3
and a clear message when it exceeds the limit, so a job that does not fit on the 8 GB laptop
stops instead of swapping. Heavy jobs then move to Colab.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

DEFAULT_LIMIT_GIB = float(os.environ.get("BER_MEM_LIMIT_GIB", 5.0))


def rss_gib(pid: int | None = None) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid or os.getpid())], capture_output=True, text=True)
    return int(out.stdout.strip() or 0) / 2**20  # ps reports KiB


def start(limit_gib: float = DEFAULT_LIMIT_GIB, interval_s: float = 1.0) -> None:
    def watch() -> None:
        while True:
            r = rss_gib()
            if r > limit_gib:
                print(f"\nMEMORY GUARD: RSS {r:.2f} GiB > {limit_gib} GiB, aborting.", file=sys.stderr, flush=True)
                os._exit(3)
            time.sleep(interval_s)

    threading.Thread(target=watch, daemon=True).start()
