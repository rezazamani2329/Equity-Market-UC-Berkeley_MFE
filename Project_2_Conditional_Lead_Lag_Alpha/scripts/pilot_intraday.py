"""
Pilot the intraday leg: three industries, one month, end to end.

Purpose is to prove the pipeline before committing to a 55 GB pull — that the
request works, the DBN parses, the minute grid is genuinely synchronised
across symbols, and the intraday lead-lag regression runs and produces a
number.  Whether that number is interesting is not the point here.

    uv run --with databento --with pandas --with pyarrow --with statsmodels \
        python scripts/pilot_intraday.py

Everything lands in `cache.nosync/databento/` (data, git-ignored) and
`results/` (small tables, committed).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.data.databento_fetch import (  # noqa: E402
    PullSpec,
    estimate_size,
    fetch,
    load,
    minute_panel,
    minute_returns,
)

RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Three industries with unambiguous, stable leaders over the whole window, so
# the pilot is not also a test of the leader-assignment edge cases:
#   35 Computers (AAPL), 45 Banking (JPM), 31 Utilities (NEE)
PILOT_INDUSTRIES = [35, 45, 31]
PILOT_START, PILOT_END = "2023-06-01", "2023-06-30"
N_FOLLOWERS = 6

_t0 = time.time()


def step(msg: str) -> None:
    print(f"[{time.time() - _t0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- symbols
symbols = pd.read_parquet(RESULTS / "p1_intraday_symbols.parquet")
pilot = symbols.loc[symbols["ff49"].isin(PILOT_INDUSTRIES)].copy()

leaders = pilot.loc[pilot["ever_leader"]]
followers = (
    pilot.loc[~pilot["ever_leader"]]
    .sort_values("active_months", ascending=False)
    .groupby("ff49")
    .head(N_FOLLOWERS)
)
chosen = pd.concat([leaders, followers], ignore_index=True)

step(f"pilot universe: {len(chosen)} symbols across {chosen['ff49'].nunique()} industries")
for ff49, chunk in chosen.groupby("ff49"):
    lead = chunk.loc[chunk["ever_leader"], "ticker"].tolist()
    foll = chunk.loc[~chunk["ever_leader"], "ticker"].tolist()
    print(f"    ff49 {ff49:>2}  leader(s) {lead}  followers {foll}")

# ------------------------------------------------------------------ pull
frames = []
for venue, chunk in chosen.groupby("venue"):
    spec = PullSpec(
        dataset=str(venue),
        schema="bbo-1m",
        symbols=tuple(sorted(chunk["ticker"].unique())),
        start=PILOT_START,
        end=PILOT_END,
    )
    est = estimate_size(spec)
    step(f"{spec.describe()}  ->  {est['gb']:.3f} GB, ${est['cost_usd']:.2f}")

    path = fetch(spec)
    step(f"  cached at {path.name} ({path.stat().st_size / 1e6:.1f} MB on disk)")

    raw = load(path)
    step(f"  {len(raw):,} raw records, columns: {sorted(raw.columns)[:8]} ...")
    frames.append(raw)

raw_all = pd.concat(frames, ignore_index=True)

# ------------------------------------------------------- synchronise + returns
panel = minute_panel(raw_all)
step(f"minute panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, "
     f"{panel['date'].nunique()} sessions")

# The property the whole intraday leg rests on: every symbol observed on the
# same grid.  If this is ragged, the non-synchronous problem has followed us.
per_minute = panel.groupby("minute")["symbol"].nunique()
step(f"symbols per minute: median {per_minute.median():.0f}, "
     f"min {per_minute.min()}, max {per_minute.max()} "
     f"(of {panel['symbol'].nunique()} in the pilot)")

rets5 = minute_returns(panel, freq_minutes=5)
step(f"5-minute midpoint returns: {len(rets5):,} observations")

# -------------------------------------------------------------- lead-lag
ticker_to_ff49 = dict(zip(chosen["ticker"], chosen["ff49"]))
leader_tickers = set(chosen.loc[chosen["ever_leader"], "ticker"])

rets5["ff49"] = rets5["symbol"].map(ticker_to_ff49)
rets5["is_leader"] = rets5["symbol"].isin(leader_tickers)

# An industry can have more than one leader over the window (leader turnover
# is ~5% a month, and Banking has both C and JPM here).  Average them rather
# than `drop_duplicates`, which would keep whichever row happened to come
# first and make the leader series depend on row order.
lead = (
    rets5.loc[rets5["is_leader"]]
    .groupby(["ts", "ff49"], as_index=False)
    .agg(leader_ret=("ret", "mean"), n_leaders=("symbol", "nunique"))
)
foll = rets5.loc[~rets5["is_leader"]].merge(lead, on=["ts", "ff49"], how="inner")

# Lag the leader by one sampling interval, within symbol and within session.
foll = foll.sort_values(["symbol", "ts"])
foll["leader_lag"] = foll.groupby(["symbol", "date"])["leader_ret"].shift(1)
foll["own_lag"] = foll.groupby(["symbol", "date"])["ret"].shift(1)

reg = foll[["ret", "leader_lag", "own_lag", "ts"]].dropna()
step(f"regression sample: {len(reg):,} follower-intervals")

import statsmodels.api as sm  # noqa: E402

x = sm.add_constant(reg[["leader_lag", "own_lag"]])
fit = sm.OLS(reg["ret"], x).fit(cov_type="cluster", cov_kwds={"groups": reg["ts"]})

print()
print("  intraday lead-lag, 5-minute intervals, clustered by timestamp")
print(f"    leader_lag  beta {fit.params['leader_lag']:+.5f}  t {fit.tvalues['leader_lag']:+.2f}")
print(f"    own_lag     beta {fit.params['own_lag']:+.5f}  t {fit.tvalues['own_lag']:+.2f}")
print(f"    n {int(fit.nobs):,}   R2 {fit.rsquared:.5f}")
print()
print("  for comparison, the DAILY unconditional result on 1996-2024:")
print("    lag 1  beta_FM +0.00928  t +5.03     lag 2  beta_FM +0.00122  t +0.72")
print("    and by year the daily effect is confined to 1996-2006.")

summary = pd.DataFrame(
    [{
        "window": f"{PILOT_START}..{PILOT_END}",
        "industries": chosen["ff49"].nunique(),
        "symbols": chosen["ticker"].nunique(),
        "interval_min": 5,
        "n_obs": int(fit.nobs),
        "beta_leader_lag": float(fit.params["leader_lag"]),
        "t_leader_lag": float(fit.tvalues["leader_lag"]),
        "beta_own_lag": float(fit.params["own_lag"]),
        "t_own_lag": float(fit.tvalues["own_lag"]),
        "median_rel_spread": float(panel["rel_spread"].median()),
    }]
)
summary.to_csv(RESULTS / "p1_intraday_pilot.csv", index=False)
step(f"wrote {RESULTS / 'p1_intraday_pilot.csv'}")
