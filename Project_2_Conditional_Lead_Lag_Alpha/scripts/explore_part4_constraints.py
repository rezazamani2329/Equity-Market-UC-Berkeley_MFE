"""
EXPLORATORY (not pre-registered): which neutrality constraint removes the
lead-lag signals' returns?

The pre-registered industry books neutralize dollar, beta and own-lag together.
This script adds the constraints one at a time, so the report can say whether
the signals' raw predictability is leader information or the followers' own
industry move from the day before (one-day industry momentum).

    PYTHONPATH=src python scripts/explore_part4_constraints.py

Writes results/p4_exploratory_constraints.csv (aggregate, safe to commit).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.data.wrds_fetch import load_or_fetch_factors_daily  # noqa: E402
from lead_lag.portfolio import calendar_lagged_signals, summary_stats  # noqa: E402
from lead_lag.shocks.decomposition import LeaderShocks  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
CACHE = PROJECT_ROOT / "cache.nosync"
SIGNALS = ["conditional_signal", "common_lag", "shock_lag", "leader_ret_lag"]
CONSTRAINT_SETS = {
    "dollar": [],
    "dollar+beta": ["b"],
    "dollar+own_lag": ["o"],
    "dollar+beta+own_lag (pre-registered)": ["b", "o"],
}

fp = pd.read_parquet(RESULTS / "p1_follower_panel.parquet")
shocks = LeaderShocks.from_raw(pd.read_parquet(CACHE / "p2_shocks_rolling.parquet"))
inputs = pd.read_parquet(CACHE / "p4_inputs.parquet",
                         columns=["date", "permno", "ret_A", "ret_C", "beta", "own_lag"])
calendar = pd.DatetimeIndex(
    load_or_fetch_factors_daily(None, "1996-01-01", "2024-12-31").frame["date"]
)
# same calendar-based leader lag as run_part4.py's books
lagged = calendar_lagged_signals(shocks.frame, calendar)
panel = fp[["date", "permno", "ff49"]].merge(lagged, on=["date", "ff49"], how="left")
del fp
panel = panel.merge(inputs, on=["date", "permno"], how="left")
del inputs

# one row per (date, industry); followers are equal-weighted within an industry,
# exactly as in the pre-registered construction
ind = panel.groupby(["date", "ff49"]).agg(
    **{s: (s, "first") for s in SIGNALS},
    b=("beta", "mean"), o=("own_lag", "mean"),
    rA=("ret_A", "mean"), rC=("ret_C", "mean"),
).reset_index().dropna(subset=["b", "o"])
del panel

rows = []
corr = ind.dropna(subset=SIGNALS).groupby("date").apply(
    lambda g: pd.Series({s: g[s].corr(g["o"]) for s in SIGNALS}), include_groups=False
).mean()
for sig in SIGNALS:
    g_all = ind.dropna(subset=[sig])
    dates = g_all["date"].to_numpy()
    cuts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for cname, cons in CONSTRAINT_SETS.items():
        out = {"A": [], "C": []}
        idx = []
        alignment = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            g = g_all.iloc[a:b]
            if len(g) < 10:
                continue
            z = g[sig].rank().to_numpy()
            z = z - z.mean()
            X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in cons])
            w = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
            gross = np.abs(w).sum()
            if gross < 1e-12:
                continue
            w = w / gross
            idx.append(dates[a])
            out["A"].append(np.nansum(w * g["rA"].to_numpy()))
            out["C"].append(np.nansum(w * g["rC"].to_numpy()))
            alignment.append(np.corrcoef(w, g[sig].to_numpy())[0, 1])
        for conv in ("A", "C"):
            st = summary_stats(pd.Series(out[conv], index=pd.to_datetime(idx)))
            rows.append({
                "signal": sig, "constraints": cname, "convention": conv,
                "ann_mean": st["ann_mean"], "t_nw": st["t_nw"], "sharpe": st["sharpe"],
                "corr_weights_signal": float(np.nanmean(alignment)),
                "corr_signal_ownlag": float(corr[sig]),
            })
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "p4_exploratory_constraints.csv", index=False)
show = res.assign(cell=lambda r: r.apply(lambda x: f"{100*x.ann_mean:+.2f}% ({x.t_nw:+.1f})", axis=1))
print(show.pivot_table(index=["signal", "convention"], columns="constraints",
                       values="cell", aggfunc="first")[list(CONSTRAINT_SETS)].to_string())
print("\nmean per-date corr(industry signal, industry avg own-lag):")
print(corr.round(3).to_string())
