"""
Pull the full intraday sample: 1,209 symbols, 2018-05 to 2024-12, `bbo-1m`.

Chunked by (venue, calendar year), not issued as one request
------------------------------------------------------------
The whole pull is ~55 GB.  A single `get_range` for that is one long-lived
stream: if the connection drops at 80% there is nothing to show for it, and
nothing to resume from.  Twenty-one venue-year chunks each land in their own
cache file, so a re-run skips what already succeeded and the cost of any
failure is one chunk.

It also makes the pull observable.  A silent 55 GB download is
indistinguishable from a hang; this prints a line per chunk with its size and
elapsed time.

Ordering: most recent years first.  If the pull has to be stopped early, the
years that survive are the ones that matter most — Databento's history starts
in 2018 precisely because that is where the daily effect is already dead, and
the migration hypothesis is tested against the most recent regime.

Disk guard
----------
Each chunk checks free space before downloading and stops cleanly rather than
filling the volume.  DBN is zstd-compressed, so on-disk size is well below the
billable figure, but the margin is checked rather than assumed.

    uv run --with databento --with pandas --with pyarrow --with python-dotenv \
        --with psycopg2-binary python scripts/pull_intraday.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.data.databento_fetch import (  # noqa: E402
    CACHE_DIR,
    PullSpec,
    estimate_size,
    fetch,
)

SCHEMA = "bbo-1m"
HISTORY_START = "2018-05-01"
END = "2024-12-31"
YEARS = list(range(2024, 2017, -1))          # newest first — see the docstring
MIN_FREE_GB = 25.0                            # stop before the volume is tight

RESULTS = PROJECT_ROOT / "results"
_t0 = time.time()


def step(msg: str) -> None:
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


symbols = pd.read_parquet(RESULTS / "p1_intraday_symbols.parquet")
step(f"symbol list: {symbols['ticker'].nunique():,} tickers, "
     f"{symbols['permno'].nunique():,} permnos, "
     f"{symbols['ff49'].nunique()} industries")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
step(f"cache: {CACHE_DIR}  ({free_gb(CACHE_DIR):.0f} GB free)")

log: list[dict[str, object]] = []
total_bytes = 0

for venue, chunk in symbols.groupby("venue"):
    tickers = tuple(sorted(chunk["ticker"].unique()))
    for year in YEARS:
        start = max(f"{year}-01-01", HISTORY_START)
        stop = min(f"{year}-12-31", END)
        if start > stop:
            continue

        spec = PullSpec(
            dataset=str(venue), schema=SCHEMA, symbols=tickers,
            start=start, end=stop,
        )
        path = spec.cache_path()
        if path.exists():
            size = path.stat().st_size
            total_bytes += size
            step(f"{venue} {year}: cached ({size / 1e9:.2f} GB)")
            log.append({"venue": venue, "year": year, "gb_on_disk": size / 1e9,
                        "status": "cached"})
            continue

        if free_gb(CACHE_DIR) < MIN_FREE_GB:
            step(f"STOPPING: only {free_gb(CACHE_DIR):.0f} GB free "
                 f"(floor {MIN_FREE_GB:.0f} GB). Completed chunks are intact; "
                 f"re-run after freeing space to continue.")
            break

        try:
            est = estimate_size(spec)
            t_chunk = time.time()
            step(f"{venue} {year}: pulling {len(tickers)} symbols, "
                 f"est {est['gb']:.2f} GB ...")
            path = fetch(spec)
            size = path.stat().st_size
            total_bytes += size
            step(f"{venue} {year}: done, {size / 1e9:.2f} GB on disk "
                 f"({size / max(est['bytes'], 1):.0%} of billable) "
                 f"in {time.time() - t_chunk:.0f}s")
            log.append({"venue": venue, "year": year,
                        "gb_billable": est["gb"], "gb_on_disk": size / 1e9,
                        "seconds": round(time.time() - t_chunk),
                        "status": "ok"})
        except Exception as exc:  # noqa: BLE001
            # One bad chunk must not kill a multi-hour run.  Record and move
            # on; re-running picks up whatever is still missing.
            step(f"{venue} {year}: FAILED — {type(exc).__name__}: {exc}")
            log.append({"venue": venue, "year": year, "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"})

pd.DataFrame(log).to_csv(RESULTS / "p1_intraday_pull_log.csv", index=False)
step(f"total on disk {total_bytes / 1e9:.1f} GB; "
     f"{free_gb(CACHE_DIR):.0f} GB free remaining")
step("done")
