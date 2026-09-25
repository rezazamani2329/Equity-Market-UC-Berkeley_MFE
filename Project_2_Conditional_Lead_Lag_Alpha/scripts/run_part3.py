"""
Run part 3 end to end: the conditional push signal (part 2's decomposition,
combined per hypotheses 1 & 2), validated by rank IC / horizon-IC and
quantile long-short portfolios, benchmarked against two baselines part 1/2
already established -- the unconditional leader return and naive own-return
reversal.

    PYTHONPATH=src python scripts/run_part3.py 2>&1 | tee cache.nosync/part3.log

Requires, in order:
    1. `uv run python scripts/run_part1.py`  -> results/p1_follower_panel.parquet
    2. `uv run python scripts/run_part2.py`  -> cache.nosync/p2_shocks_rolling.parquet
Both are gitignored (regenerate-only); this script fails fast with a clear
message naming which one is missing rather than a bare FileNotFoundError.

What "beat the baseline" means here, concretely (docs/STATUS.md, part 3's
unblocked note): the conditional_signal's IC / quantile spread must exceed
BOTH the unconditional leader signal (part 1's baseline -- expected to be
weak/near zero pooled, per p1_coefficient_by_year.csv's post-2007 flatline)
and naive own-return reversal (-own_lag) -- not just be different from zero
in isolation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.shocks.decomposition import LeaderShocks  # noqa: E402
from lead_lag.signals import (  # noqa: E402
    conditional_signal_panel,
    forward_returns,
    horizon_ic,
    quantile_portfolios,
)

RESULTS = PROJECT_ROOT / "results"
CACHE = PROJECT_ROOT / "cache.nosync"
HORIZONS = (1, 5, 10, 20)
RESULTS.mkdir(exist_ok=True)

_t0 = time.time()


def step(label: str) -> None:
    print(f"\n[{time.time() - _t0:7.1f}s] {label}", flush=True)


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `uv run python {produced_by}` first."
        )
    return path


# --------------------------------------------------------------- load
step("loading part 1's follower panel and part 2's rolling decomposition ...")
fp_path = _require(RESULTS / "p1_follower_panel.parquet", "scripts/run_part1.py")
shocks_path = _require(CACHE / "p2_shocks_rolling.parquet", "scripts/run_part2.py")

fp = pd.read_parquet(fp_path)
shocks = LeaderShocks.from_raw(pd.read_parquet(shocks_path))
print(f"follower_panel: {len(fp):,} rows, {fp['permno'].nunique():,} permnos")
print(f"shocks (rolling, point-in-time): {len(shocks.frame):,} industry-days")

# --------------------------------------------------------- build the signal
step("building the conditional signal panel (lag=1) ...")
signal_panel = conditional_signal_panel(fp, shocks, lag=1, ret_col="ret")
signal_panel["reversal_signal"] = -signal_panel["own_lag"]
n_have_signal = signal_panel["conditional_signal"].notna().sum()
print(f"follower-days with a non-missing conditional_signal: {n_have_signal:,} "
      f"({n_have_signal / len(signal_panel):.1%} of {len(signal_panel):,})")

# ------------------------------------------------------------ forward returns
step(f"computing forward returns, horizons={HORIZONS} ...")
fwd = forward_returns(fp, horizons=HORIZONS, ret_col="ret")

# ------------------------------------------------------------------- IC
SIGNALS = {
    "conditional_signal (common - shock)": "conditional_signal",
    "common_lag (continuation leg)": "common_lag",
    "shock_lag (reversion leg, raw sign)": "shock_lag",
    "leader_ret_lag (part 1 baseline, unconditional)": "leader_ret_lag",
    "reversal_signal (naive, -own_lag)": "reversal_signal",
}

step("horizon IC for every signal (the central exhibit) ...")
ic_tables = {}
for label, col in SIGNALS.items():
    table = horizon_ic(signal_panel, fwd, signal_col=col, horizons=HORIZONS)
    ic_tables[label] = table
    print(f"\n-- {label} --")
    print(table.round(5).to_string())

all_ic_rows = []
for label, table in ic_tables.items():
    for h, row in table.iterrows():
        all_ic_rows.append({"signal": label, "horizon": h, **row.to_dict()})
ic_summary = pd.DataFrame(all_ic_rows)
ic_summary.to_csv(RESULTS / "p3_horizon_ic.csv", index=False)

# -------------------------------------------------------- quantile portfolios
step("quintile long-short spread at h=1 for every signal ...")
merged_h1 = signal_panel.merge(
    fwd[["date", "permno", "fwd_ret_1d"]], on=["date", "permno"], how="left"
)

q_rows = []
for label, col in SIGNALS.items():
    q = quantile_portfolios(merged_h1, signal_col=col, ret_col="fwd_ret_1d", n_quantiles=5)
    print(f"\n-- {label} --")
    print(q.summary.round(6).to_string())
    print(f"long-short spread: mean={q.spread_mean:.4%}  t={q.spread_t_stat:.2f}  "
          f"n={len(q.spread)}")
    q_rows.append(
        {
            "signal": label,
            "spread_mean": q.spread_mean,
            "spread_t_stat": q.spread_t_stat,
            "n_periods": len(q.spread),
        }
    )
q_summary = pd.DataFrame(q_rows)
q_summary.to_csv(RESULTS / "p3_quintile_spread_h1.csv", index=False)

# ----------------------------------------------------------------- headline
step("headline comparison: h=1 mean IC and t-stat, every signal ===")
h1 = ic_summary.loc[ic_summary["horizon"] == 1, ["signal", "mean_ic", "t_stat", "n_periods"]]
print(h1.to_string(index=False))
print()
print(q_summary.to_string(index=False))

beats_baseline = (
    ic_tables["conditional_signal (common - shock)"].loc[1, "mean_ic"]
    > max(
        ic_tables["leader_ret_lag (part 1 baseline, unconditional)"].loc[1, "mean_ic"],
        ic_tables["reversal_signal (naive, -own_lag)"].loc[1, "mean_ic"],
    )
)
step(f"conditional_signal IC(h=1) beats both baselines: {beats_baseline}")
print("If False, hypothesis 3 (conditioning beats undifferentiated reversal) does")
print("not hold at lag 1 in this sample -- report that plainly rather than reaching")
print("for a different horizon or subsample until it does; see docs/STATUS.md's")
print("note on the 20-look horizon scan for why that would be p-hacking.")

for f in sorted(RESULTS.glob("p3_*")):
    print(f"  {f.name:<34} {f.stat().st_size / 1e6:8.3f} MB")
step("done")
