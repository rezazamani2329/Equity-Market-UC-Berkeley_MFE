"""
Run part 2 end to end on the real sample, headless: rebuild the eligible
daily panel and roles (same steps `run_part1.py` uses, steps 3-8), then the
ex-leader industry index, the leader-shock decomposition (both estimators),
and the conditional-response regression that is part 2's own headline
exhibit.

    PYTHONPATH=src python scripts/run_part2.py 2>&1 | tee cache.nosync/part2.log

Why this re-does steps 3-8 instead of reading `results/p1_*.parquet`
----------------------------------------------------------------------
`run_part1.py` persists `p1_universe.parquet`, `p1_roles.parquet`,
`p1_industry_returns.parquet`, and `p1_follower_panel.parquet` -- but not the
full eligible-daily panel (`eligible_daily` output), which
`shocks.assemble_leader_frame` needs to build the ex-leader industry index.
That panel is comparable in size to the follower panel itself and gitignored
by the same `*.parquet` rule (see `.gitignore`'s comment: nothing
CRSP-derived at the security level belongs in a public repo). Rebuilding it
here from the already-warm WRDS cache costs the transform time, not a pull,
so it is cheap to redo rather than to persist and version.

Everything this script writes lands in `results/` (the small CSVs) and
`cache.nosync/` (the large, regenerate-only `p2_shocks_rolling.parquet`, so
part 3 does not have to redo this script's work every run).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.data.daily_returns import adjusted_daily_returns  # noqa: E402
from lead_lag.data.industry_map import attach_industry, load_siccodes49  # noqa: E402
from lead_lag.data.information_set import attach_lagged  # noqa: E402
from lead_lag.data.leaders import LeaderRule, assign_roles, follower_panel  # noqa: E402
from lead_lag.data.universe import UniverseRules, build_universe, eligible_daily  # noqa: E402
from lead_lag.data.wrds_fetch import (  # noqa: E402
    load_or_fetch_crsp_daily,
    load_or_fetch_crsp_delist,
    load_or_fetch_factors_daily,
)
from lead_lag.shocks.decomposition import (  # noqa: E402
    DecompositionSpec,
    assemble_leader_frame,
    decompose_full_sample,
    decompose_rolling,
)
from lead_lag.shocks.regression import conditional_horizon_profile  # noqa: E402

START, END = "1996-01-01", "2024-12-31"
RESULTS = PROJECT_ROOT / "results"
CACHE = PROJECT_ROOT / "cache.nosync"
RESULTS.mkdir(exist_ok=True)

_t0 = time.time()


def step(label: str) -> None:
    print(f"[{time.time() - _t0:7.1f}s] {label}", flush=True)


# ------------------------------------------------------ 3-6. rebuild the panel
# Identical to run_part1.py's steps 3-6: warm-cache pulls, delisting merge,
# industry attach, d-1 information set. See run_part1.py for the reasoning;
# this is not re-derived, just re-run, so part 2 uses the exact same panel
# part 1 validated against (same audit, same blocking checks).
step("loading cached CRSP daily / delist / factors ...")
daily = load_or_fetch_crsp_daily(None, START, END, verbose=False)
delist = load_or_fetch_crsp_delist(None, START, END)
factors = load_or_fetch_factors_daily(None, START, END)
step(f"daily {len(daily):,} rows / {daily.frame['permno'].nunique():,} stocks; "
     f"delist {len(delist):,}; factors {len(factors):,}")

step("merging delisting events ...")
returns, _report = adjusted_daily_returns(daily, delist)
del daily

step("attaching Fama-French 49 industries and the d-1 information set ...")
siccodes = load_siccodes49()
panel = attach_industry(returns.frame, siccodes)
del returns
panel = attach_lagged(panel)

step("building the monthly universe and leader/follower roles ...")
universe, _attrition = build_universe(panel, UniverseRules())
roles = assign_roles(universe, LeaderRule())
step(f"{universe.frame['eligible'].sum():,} eligible stock-months; "
     f"{(roles.frame['role'] == 'leader').sum():,} leader-months")

step("building the eligible daily panel (needed for the ex-leader index) ...")
eligible = eligible_daily(panel, universe)
step(f"eligible daily rows: {len(eligible):,}")

step("building the regression panel (for part 2's own conditional-response exhibit) ...")
fp = follower_panel(panel, roles).merge(
    factors.frame[["date", "mktrf"]], on="date", how="left"
)
del panel

# ------------------------------------------------------------- 7. assemble
step("assembling the leader frame: leader_ret + ex-leader industry index + mkt ...")
frame = assemble_leader_frame(eligible, roles, factors.frame, weight="mktcap_lag", market_col="mktrf")
step(f"leader-industry-days assembled: {len(frame):,}")

# ------------------------------------------------------------ 8. decompose
spec = DecompositionSpec()  # market + ex-leader industry, no sync lags, 60-day min
step(f"decomposing (full-sample, descriptive): {spec.regressor_note()} ...")
shocks_full = decompose_full_sample(frame, spec)
var_shares = shocks_full.variance_shares()
step("variance shares (top 10 by specific_share):")
print(var_shares.head(10).round(4).to_string())
var_shares.to_csv(RESULTS / "p2_variance_shares.csv")

step("decomposing (rolling, point-in-time -- the version a signal may use) ...")
shocks_rolling = decompose_rolling(frame, spec)
step(f"rolling decomposition rows: {len(shocks_rolling):,} "
     f"(first {spec.window} days of each industry are burn-in and absent)")

# Sanity check the residual identity part 2's module docstring promises:
# common + shock == leader_ret, exactly, by construction.
resid_check = (
    (shocks_full.frame["common"] + shocks_full.frame["shock"] - shocks_full.frame["leader_ret"])
    .abs().max()
)
step(f"max |common + shock - leader_ret| (full-sample): {resid_check:.2e} (should be ~0)")

# ----------------------------------------------------- 9. conditional response
step("part 2's own headline exhibit: b_common(k) vs b_shock(k), lags 1-10 ...")
cond_profile = conditional_horizon_profile(fp, shocks_rolling, lags=range(1, 11), market_col="mktrf")
print(cond_profile.round(5).to_string())
cond_profile.to_csv(RESULTS / "p2_conditional_response_horizon.csv")

# ------------------------------------------------------------- 10. handoff
step("writing handoff artefacts ...")
shocks_rolling.frame.to_parquet(CACHE / "p2_shocks_rolling.parquet", index=False)
shocks_full.frame.to_parquet(CACHE / "p2_shocks_full.parquet", index=False)
step(f"wrote {CACHE / 'p2_shocks_rolling.parquet'} "
     f"({(CACHE / 'p2_shocks_rolling.parquet').stat().st_size / 1e6:.1f} MB) "
     f"-- part 3 reads this one (point-in-time, tradeable)")

for f in sorted(RESULTS.glob("p2_*")):
    print(f"  {f.name:<40} {f.stat().st_size / 1e6:8.3f} MB")
step("done")
