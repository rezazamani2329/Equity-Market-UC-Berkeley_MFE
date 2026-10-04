"""
Run Part 5: net-of-cost returns, year-by-year returns and sub-period checks.

    PYTHONPATH=src python scripts/run_part5.py

Requires:
    scripts/run_part4.py -> results/p4_portfolio_daily.csv, results/p4_costs.csv

Both inputs are committed, so this script runs without the WRDS cache.

Aggregate outputs (safe to commit): results/p5_*.csv.

    p5_net_of_cost.csv      return per year after a one-way cost of 0 to 5 bp
    p5_yearly_returns.csv   calendar-year return of each book, open to close
    p5_subperiods.csv       windows x conventions, incl. midpoint and overnight
    p5_decision.csv         the cost rule applied to every book

Every split of 1996-2024 here is pseudo-out-of-sample: the full sample was
examined in Parts 1-4 and no holdout was reserved (docs/STATUS.md).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.portfolio import WINDOWS, summary_stats, window_slice  # noqa: E402
from lead_lag.robustness import ROUND_TRIP, breakeven_bp, net_of_cost  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
BOOKS = ["conditional_signal", "common_lag", "leader_ret_lag", "shock_lag", "reversal"]
COSTS_BP = [0.0, 0.1, 0.5, 1.0, 5.0]    # the median half-spread is added below
COST_WINDOWS = ["full 1996-2024", "1996-2006", "2007-2024"]
RECENT = "2007-2024"                    # the window the cost rule is judged on
DIAGNOSTICS = ["C", "A", "A_mid", "overnight"]

_t0 = time.time()


def step(label: str) -> None:
    print(f"\n[{time.time() - _t0:7.1f}s] {label}", flush=True)


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python {produced_by}` first.")
    return path


# ------------------------------------------------------------------- load
step("loading Part 4 daily book returns ...")
daily = pd.read_csv(_require(RESULTS / "p4_portfolio_daily.csv", "scripts/run_part4.py"),
                    parse_dates=["date"])
wide = daily.pivot(index="date", columns=["book", "convention"], values="ret")
p4_costs = pd.read_csv(_require(RESULTS / "p4_costs.csv", "scripts/run_part4.py")).set_index("book")
half_spread = float(p4_costs["median_half_spread_bp_2007_2024"].iloc[0])
costs_bp = sorted(COSTS_BP + [round(half_spread, 2)])
print(f"{len(wide):,} sessions; median half-spread 2007-2024 {half_spread:.2f} bp")

# ------------------------------------------------------------ net of cost
step("net-of-cost returns ...")
rows = []
for book in BOOKS:
    traded = {"C": ROUND_TRIP, "A": wide[(book, "turnover_carry")]}
    for conv in ("C", "A"):
        for window in COST_WINDOWS:
            gross = window_slice(wide[(book, conv)], *WINDOWS[window])
            for cost in costs_bp:
                s = summary_stats(net_of_cost(gross, traded[conv], cost))
                rows.append({"book": book, "convention": conv, "window": window,
                             "one_way_cost_bp": cost, "ann_mean_net": s["ann_mean"],
                             "t_nw": s["t_nw"], "breakeven_bp": breakeven_bp(gross, traded[conv])})
net = pd.DataFrame(rows)
net.to_csv(RESULTS / "p5_net_of_cost.csv", index=False)
show = net[(net["convention"] == "C") & (net["window"] == RECENT)]
print((show.pivot(index="book", columns="one_way_cost_bp", values="ann_mean_net") * 100).round(2))

# --------------------------------------------------------- yearly returns
step("calendar-year returns, open to close ...")
yearly = pd.DataFrame({b: wide[(b, "C")].groupby(wide.index.year).sum() for b in BOOKS})
yearly.index.name = "year"
yearly.to_csv(RESULTS / "p5_yearly_returns.csv")
print((yearly > 0).sum().rename("positive years").to_frame().assign(of=len(yearly)))

# ------------------------------------------------------------ sub-periods
step("sub-periods, midpoint and overnight legs ...")
rows = []
for book in BOOKS:
    for conv in DIAGNOSTICS:
        if (book, conv) not in wide.columns:
            continue
        for window, (a, b) in WINDOWS.items():
            s = summary_stats(window_slice(wide[(book, conv)], a, b))
            rows.append({"book": book, "convention": conv, "window": window,
                         "status": "pseudo-out-of-sample", "n_days": s["n_days"],
                         "ann_mean": s.get("ann_mean"), "t_nw": s.get("t_nw")})
pd.DataFrame(rows).to_csv(RESULTS / "p5_subperiods.csv", index=False)

# --------------------------------------------------------------- decision
step("decision ...")
at_spread = show[show["one_way_cost_bp"] == round(half_spread, 2)].set_index("book")
decision = pd.DataFrame({
    "book": BOOKS,
    "breakeven_bp_C_2007_2024": [at_spread.loc[b, "breakeven_bp"] for b in BOOKS],
    "median_half_spread_bp_2007_2024": half_spread,
    "ann_mean_net_at_half_spread": [at_spread.loc[b, "ann_mean_net"] for b in BOOKS],
})
decision["passed"] = decision["ann_mean_net_at_half_spread"] > 0
decision.to_csv(RESULTS / "p5_decision.csv", index=False)
print(decision.round(4).to_string(index=False))
print("\nVERDICT:", "OVERTURNED" if decision["passed"].any() else "DO NOT IMPLEMENT")
