"""
Part 4 prep: compute every shifted input on the full CRSP panel, then keep the
follower rows.

    PYTHONPATH=src python scripts/prep_part4.py 2>&1 | tee cache.nosync/prep_part4.log

Requires the WRDS cache (cache.nosync/crsp_daily_v2_*.parquet, delistings,
factors) and results/p1_follower_panel.parquet from scripts/run_part1.py.

Writes:
    cache.nosync/p4_inputs.parquet    one row per follower-day (stock-level,
                                      CRSP-derived: never commit)
    results/p4_input_coverage.csv     share of follower-days with each input,
                                      by year (aggregate, safe to commit)
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.data.daily_returns import adjusted_daily_returns  # noqa: E402
from lead_lag.data.wrds_fetch import (  # noqa: E402
    load_or_fetch_crsp_daily,
    load_or_fetch_crsp_delist,
    load_or_fetch_factors_daily,
)
from lead_lag.portfolio import full_panel_inputs  # noqa: E402

START, END = "1996-01-01", "2024-12-31"
RESULTS = PROJECT_ROOT / "results"
CACHE = PROJECT_ROOT / "cache.nosync"
N_CHUNKS = 8
EXTRA_COLS = ["date", "permno", "openprc", "bid", "ask", "cfacpr", "quote_only", "retx"]

_t0 = time.time()


def step(label: str) -> None:
    print(f"[{time.time() - _t0:7.1f}s] {label}", flush=True)


fp_path = RESULTS / "p1_follower_panel.parquet"
if not fp_path.exists():
    raise FileNotFoundError(f"{fp_path} not found -- run scripts/run_part1.py first.")

step("loading cached CRSP daily, delistings, factors ...")
daily = load_or_fetch_crsp_daily(None, START, END, verbose=False)
delist = load_or_fetch_crsp_delist(None, START, END)
factors = load_or_fetch_factors_daily(None, START, END)

step("delisting-adjusted returns on the full panel (part 1's function) ...")
returns, _ = adjusted_daily_returns(daily, delist)
ret = returns.frame[["date", "permno", "prc", "ret", "dlret"]].copy()
extras = daily.frame[EXTRA_COLS].copy()
del daily, returns, delist
gc.collect()
step(f"full panel: {len(ret):,} rows, {ret['permno'].nunique():,} stocks")

keys = pd.read_parquet(fp_path, columns=["date", "permno"])
keys["date"] = pd.to_datetime(keys["date"])
step(f"follower keys: {len(keys):,}")

# every computation is within permno, so chunking by permno is exact
permnos = np.sort(ret["permno"].unique())
chunks = np.array_split(permnos, N_CHUNKS)
parts = []
for i, chunk in enumerate(chunks, 1):
    r = ret[ret["permno"].isin(chunk)]
    e = extras[extras["permno"].isin(chunk)]
    out = full_panel_inputs(r, e, factors.frame)
    out = keys[keys["permno"].isin(chunk)].merge(out, on=["date", "permno"], how="left")
    parts.append(out)
    step(f"chunk {i}/{N_CHUNKS}: {len(out):,} follower rows")
    del r, e, out
    gc.collect()

inputs = pd.concat(parts, ignore_index=True).sort_values(["date", "permno"])
inputs.to_parquet(CACHE / "p4_inputs.parquet", index=False)
step(f"wrote cache.nosync/p4_inputs.parquet ({len(inputs):,} rows)")

cov = inputs.assign(year=inputs["date"].dt.year).groupby("year").agg(
    rows=("permno", "size"),
    ret_A=("ret_A", lambda s: s.notna().mean()),
    ret_B=("ret_B", lambda s: s.notna().mean()),
    ret_C=("ret_C", lambda s: s.notna().mean()),
    beta=("beta", lambda s: s.notna().mean()),
    own_lag=("own_lag", lambda s: s.notna().mean()),
    mid_ret=("mid_ret_valid", "mean"),
    rel_spread_lag=("rel_spread_lag", lambda s: s.notna().mean()),
)
cov.to_csv(RESULTS / "p4_input_coverage.csv")
print(cov.round(4).to_string())
step("done")
