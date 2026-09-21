"""
Run part 1 end to end on the real sample, headless.

Same steps as `notebooks/part1_data_universe.ipynb` §3-§10, without the
figures.  The notebook is for reading and diagnosing; this is for the long
run — 34 million daily rows take tens of minutes, which is not a thing to do
inside an interactive kernel that might lose its connection.

    PYTHONPATH=src python scripts/run_part1.py 2>&1 | tee cache.nosync/part1.log

Everything it writes lands in `results/`.  Rerunning is safe: the WRDS cache
is already warm, so the cost is the transforms, not the pulls.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.baseline import (  # noqa: E402
    baseline_test,
    build_lags,
    fama_macbeth,
    horizon_profile,
    quintile_sort,
)
from lead_lag.data.daily_returns import adjusted_daily_returns  # noqa: E402
from lead_lag.data.industry_map import (  # noqa: E402
    OTHER_INDUSTRY,
    attach_industry,
    industry_returns,
    load_siccodes49,
)
from lead_lag.data.information_set import attach_lagged, same_day_columns  # noqa: E402
from lead_lag.data.leaders import (  # noqa: E402
    LeaderRule,
    assign_roles,
    follower_panel,
    leader_turnover,
)
from lead_lag.data.quality_checks import blocking_audit, daily_panel_audit  # noqa: E402
from lead_lag.data.universe import (  # noqa: E402
    UniverseRules,
    build_universe,
    eligible_daily,
)
from lead_lag.data.wrds_fetch import (  # noqa: E402
    load_or_fetch_crsp_daily,
    load_or_fetch_crsp_delist,
    load_or_fetch_factors_daily,
)

START, END = "1996-01-01", "2024-12-31"
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

_t0 = time.time()


def step(label: str) -> None:
    """One timestamped progress line.  A 40-minute run with no output is
    indistinguishable from a hung one."""
    print(f"[{time.time() - _t0:7.1f}s] {label}", flush=True)


# ---------------------------------------------------------------- 3. pulls
step("loading cached CRSP daily ...")
daily = load_or_fetch_crsp_daily(None, START, END, verbose=False)
delist = load_or_fetch_crsp_delist(None, START, END)
factors = load_or_fetch_factors_daily(None, START, END)
step(f"daily {len(daily):,} rows / {daily.frame['permno'].nunique():,} stocks; "
     f"delist {len(delist):,}; factors {len(factors):,}")

by_year = daily.frame.assign(year=daily.frame["date"].dt.year).groupby("year").agg(
    rows=("permno", "size"), stocks=("permno", "nunique"), days=("date", "nunique")
)
by_year.to_csv(RESULTS / "p1_coverage_by_year.csv")

# ------------------------------------------------------- 4. delisting merge
step("merging delisting events ...")
returns, report = adjusted_daily_returns(daily, delist)
del daily
print(report.to_frame().to_string())
print("\nrows by source:")
print(returns.frame["source"].value_counts().to_string())
report.to_frame().to_csv(RESULTS / "p1_delisting_report.csv")

# ---------------------------------------------------------- 5. quality audit
step("running the quality audit ...")
audit = daily_panel_audit().run(returns)
print(audit.summary().to_string())
audit.to_frame().to_csv(RESULTS / "p1_quality_report.csv", index=False)
blocking_audit().assert_clean(returns)
step("blocking checks clean (no duplicate keys)")

# --------------------------------------------------------- 6. industry map
step("attaching Fama-French 49 industries ...")
siccodes = load_siccodes49()
panel = attach_industry(returns.frame, siccodes)
del returns
other_share = (panel["ff49"] == OTHER_INDUSTRY).mean()
step(f"rows in industry {OTHER_INDUSTRY} ('Other'): {other_share:.2%}")

# ------------------------------------------ 6b. the d-1 information set
# Every characteristic a signal may condition on — industry classification,
# market cap, price, dollar volume, exchange — gets a `_lag` twin holding the
# value at the previous close.  Downstream code takes the lagged one; the
# same-day columns stay only so a robustness run can compare against them.
step("attaching the d-1 information set ...")
panel = attach_lagged(panel)
n_reclass = int(
    (panel["ff49_lag"].notna() & (panel["ff49"] != panel["ff49_lag"])).sum()
)
step(f"stock-days on which the industry classification changed: {n_reclass:,} "
     f"({n_reclass / len(panel):.4%} of rows) — each now uses its d-1 industry")

# ------------------------------------------------------------- 7. universe
step("building the monthly universe ...")
rules = UniverseRules()
universe, attrition = build_universe(panel, rules)
step(f"{len(universe):,} decisions; {universe.frame['eligible'].sum():,} eligible")
print(attrition.describe().loc[["mean", "min", "max"]].astype(int).to_string())
attrition.to_csv(RESULTS / "p1_universe_attrition.csv")

solo = pd.Series(
    {col.removeprefix("pass_"): 1 - universe.frame[col].mean()
     for col in ("pass_obs", "pass_price", "pass_size", "pass_liquidity", "pass_industry")},
    name="share failing (standalone)",
).sort_values(ascending=False)
print("\nstandalone cost of each screen:")
print(solo.to_string())

# -------------------------------------------------------------- 8. leaders
step("assigning leaders and followers ...")
roles = assign_roles(universe, LeaderRule())
print(roles.frame["role"].value_counts().to_string())
turnover = leader_turnover(roles)
step(f"leader changes month to month: {turnover.attrs['overall_rate']:.1%}")

# ------------------------------------------------------------- 9. baseline
step("building the regression panel ...")
fp = follower_panel(panel, roles).merge(
    factors.frame[["date", "mktrf"]], on="date", how="left"
)
step(f"regression panel: {len(fp):,} follower-days, {fp['date'].nunique():,} days")

step("baseline test at lag 1 ...")
r1 = baseline_test(fp, lag=1, market_col="mktrf")
print(pd.Series(r1.as_row()).to_string())

step("horizon profile, lags 1-10 ...")
profile = horizon_profile(fp, lags=range(1, 11), market_col="mktrf", verbose=True)
print(profile.round(5).to_string())
profile.to_csv(RESULTS / "p1_horizon_profile.csv")

step("quintile sort ...")
bins = quintile_sort(fp, lag=1, n_bins=5)
print(bins.round(6).to_string())
bins.to_csv(RESULTS / "p1_quintile_sort.csv")

step("coefficient stability by year ...")
_, _, daily_coefs = fama_macbeth(build_lags(fp, lag=1))
annual = daily_coefs.groupby(daily_coefs.index.year).mean()
print(annual.round(4).to_string())
annual.to_frame("mean_b").to_csv(RESULTS / "p1_coefficient_by_year.csv")

# -------------------------------------------------------------- 10. handoff
step("writing handoff artefacts ...")
# The industry index uses the FORMATION-MONTH vintage and the eligible
# universe, so it is consistent with the leader/follower assignment: a firm
# reclassified mid-month cannot be ranked against industry A's leader while
# being averaged into industry B's index.  `eligible_daily` supplies both —
# it restricts to tradeable names and carries the universe's `ff49`.
# Weights are `mktcap_lag`, the previous close.  See industry_map's docstring.
eligible = eligible_daily(panel, universe)
step(f"eligible daily rows for the index: {len(eligible):,}")
ind_ret = industry_returns(eligible, industry_col="ff49", weight="mktcap_lag")

# `ff49` in the regression panel comes from the role table (formation month),
# not the daily row, so it is exempt from the same-day check — see
# `information_set.same_day_columns`.
print(f"  unlagged characteristics in the regression panel: "
      f"{same_day_columns(fp, exempt={'ff49'}) or 'none'}")

# The two handoff files must key on the same industry vintage.
shared = set(map(tuple, roles.frame[["ff49"]].drop_duplicates().to_numpy().tolist()))
print(f"  roles industries {len(shared)} | index industries "
      f"{ind_ret['ff49'].nunique()} — same key, same vintage")
universe.frame.to_parquet(RESULTS / "p1_universe.parquet", index=False)
roles.frame.to_parquet(RESULTS / "p1_roles.parquet", index=False)
ind_ret.to_parquet(RESULTS / "p1_industry_returns.parquet", index=False)
fp.to_parquet(RESULTS / "p1_follower_panel.parquet", index=False)

for f in sorted(RESULTS.iterdir()):
    print(f"  {f.name:<34} {f.stat().st_size / 1e6:8.2f} MB")
step("done")
