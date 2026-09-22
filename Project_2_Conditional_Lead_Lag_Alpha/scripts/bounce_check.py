"""
Is the daily lead-lag effect information, or is it bid-ask bounce?

This is the single most important robustness question in the project, and it
can be answered now that the v2 pull carries quotes and split factors.

The setup.  The unconditional effect is beta_FM = +0.0093 (t = 5.03) at one
day, and by year it is confined to 1996-2006.  Over exactly that period the
median relative spread fell from 3.23% to 0.11%.  Roll (1984) shows that
close-to-close returns of a stock quoted with spread s carry a spurious
negatively-autocorrelated component of scale s/2 — so the era with the effect
is also the era where the artefact had 30x the room to operate.

The test.  Rebuild the follower panel on MIDPOINT returns instead of closes,
holding the universe and the leader assignment fixed, and re-run the same
regression.  The midpoint does not bounce.  Three outcomes:

    effect survives on midpoints        -> it is not bounce; the project has
                                           something to condition on
    effect disappears on midpoints      -> it was the spread, and the honest
                                           conclusion is "do not implement"
    effect shrinks but stays            -> partly artefact; report both

Whatever comes out, this table goes in the report.  The comparison is
apples-to-apples because only the return measure changes: same stocks, same
leaders, same dates, same regression.

One caveat to state alongside the result.  A midpoint return is a PRICE
return; it excludes dividends.  The close-based comparison here therefore uses
`retx` (CRSP's price return) rather than `ret`, so the two differ only in
bounce and not in distributions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.baseline import baseline_test, build_lags  # noqa: E402
from lead_lag.data.daily_returns import adjusted_daily_returns  # noqa: E402
from lead_lag.data.industry_map import attach_industry, load_siccodes49  # noqa: E402
from lead_lag.data.information_set import attach_lagged  # noqa: E402
from lead_lag.data.leaders import LeaderRule, assign_roles, follower_panel  # noqa: E402
from lead_lag.data.microstructure import (  # noqa: E402
    attach_quote_columns,
    bounce_exposure,
    midpoint_returns,
)
from lead_lag.data.universe import UniverseRules, build_universe  # noqa: E402
from lead_lag.data.wrds_fetch import (  # noqa: E402
    load_or_fetch_crsp_daily,
    load_or_fetch_crsp_delist,
    load_or_fetch_factors_daily,
)

START, END = "1996-01-01", "2024-12-31"
RESULTS = PROJECT_ROOT / "results"
_t0 = time.time()


def step(msg: str) -> None:
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)


def ac1_by_year(df: pd.DataFrame, col: str) -> pd.Series:
    """First-order autocorrelation of `col`, pooled within year.

    Vectorised rather than a per-stock `groupby.apply`: the panel has 16,000
    stocks over 29 years, and a Python-level autocorr per stock-year is
    ~460,000 lambda calls.  Pooling the within-stock lag and correlating once
    per year gives the same quantity for this purpose in seconds.

    The lag is taken WITHIN each stock, so the pooled correlation never pairs
    one stock's return with another's.
    """
    d = df[["date", "permno", col]].dropna().sort_values(["permno", "date"])
    d["lag"] = d.groupby("permno", sort=False)[col].shift(1)
    d = d.dropna(subset=["lag"])
    d["year"] = d["date"].dt.year
    return d.groupby("year").apply(
        lambda g: g[col].corr(g["lag"]), include_groups=False
    )


# ------------------------------------------------------------------ load
step("loading CRSP daily v2 (with quotes and split factors) ...")
daily = load_or_fetch_crsp_daily(None, START, END, verbose=False)
delist = load_or_fetch_crsp_delist(None, START, END)
factors = load_or_fetch_factors_daily(None, START, END)
step(f"{len(daily):,} rows")

returns, _ = adjusted_daily_returns(daily, delist)
del daily
siccodes = load_siccodes49()
panel = attach_industry(returns.frame, siccodes)
del returns

# The delisting merge drops the quote columns (it rebuilds the frame), so
# re-attach them from the raw file before computing midpoints.
step("re-attaching quotes and computing midpoint returns ...")
raw = load_or_fetch_crsp_daily(None, START, END, verbose=False).frame[
    ["date", "permno", "bid", "ask", "cfacpr", "quote_only", "retx"]
]
panel = panel.merge(raw, on=["date", "permno"], how="left")
panel = midpoint_returns(attach_quote_columns(panel))
step(f"midpoint returns computed on {panel['mid_ret'].notna().sum():,} rows "
     f"({panel['mid_ret'].notna().mean():.1%} of the panel)")

# ------------------------------------------------- 1. how big is the artefact
step("bounce exposure by year ...")
exposure = bounce_exposure(panel)
exposure.to_csv(RESULTS / "p1_bounce_exposure.csv")
print(exposure.round(5).to_string())

# ------------------------------------------- 2. does bounce show up as ac1
step("first-order autocorrelation, closes vs midpoints ...")
ac = pd.DataFrame({
    "ac1_close": ac1_by_year(panel, "retx"),
    "ac1_mid": ac1_by_year(panel, "mid_ret"),
})
ac["gap"] = ac["ac1_close"] - ac["ac1_mid"]
ac.to_csv(RESULTS / "p1_autocorr_close_vs_mid.csv")
print(ac.round(4).to_string())

# --------------------------------- 3. the lead-lag test on each return measure
step("building universe and roles (on standard returns, unchanged) ...")
panel = attach_lagged(panel)
universe, _ = build_universe(panel, UniverseRules())
roles = assign_roles(universe, LeaderRule())
step(f"{universe.frame['eligible'].sum():,} eligible stock-months")

rows = []
for ret_col, label in [("ret", "close-to-close (CRSP ret)"),
                       ("retx", "close-to-close, ex-div (retx)"),
                       ("mid_ret", "bid-ask midpoint")]:
    step(f"lead-lag on {label} ...")
    fp = follower_panel(panel, roles, ret_col=ret_col).merge(
        factors.frame[["date", "mktrf"]], on="date", how="left"
    )
    for lag in (1, 2):
        r = baseline_test(fp, lag=lag, ret_col=ret_col, market_col="mktrf")
        rows.append({"return_measure": label, **r.as_row()})
    # And the same split the by-year chart demands.
    for era, lo, hi in [("1996-2006", 1996, 2006), ("2007-2024", 2007, 2024)]:
        sub = fp.loc[(fp["date"].dt.year >= lo) & (fp["date"].dt.year <= hi)]
        r = baseline_test(sub, lag=1, ret_col=ret_col, market_col="mktrf")
        rows.append({"return_measure": f"{label} [{era}]", **r.as_row()})
    del fp

table = pd.DataFrame(rows)
table.to_csv(RESULTS / "p1_bounce_check.csv", index=False)

print()
print("=" * 78)
print("LEAD-LAG COEFFICIENT BY RETURN MEASURE")
print("=" * 78)
print(table[["return_measure", "lag", "n_obs", "beta_fm", "t_fm",
             "beta_pooled", "t_pooled"]].round(5).to_string(index=False))
step("done")
