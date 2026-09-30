"""
Run Part 4 end to end: portfolio construction and risk, per docs/part4_prereg.md
(changes after the code audit are listed in docs/part4_deviations.md).

    PYTHONPATH=src python scripts/run_part4.py 2>&1 | tee cache.nosync/part4.log

Requires, in order:
    1. scripts/run_part1.py   -> results/p1_follower_panel.parquet
    2. scripts/run_part2.py   -> cache.nosync/p2_shocks_rolling.parquet
    3. scripts/prep_part4.py  -> cache.nosync/p4_inputs.parquet, p4_market_intraday.parquet

Aggregate outputs (safe to commit): results/p4_*.csv, figures/p4_*.png.
Stock-level weights go to cache.nosync/p4_weights.parquet and must not be
committed (the team repo is public; CRSP-derived stock-level data is licensed).
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lead_lag.baseline import quintile_sort  # noqa: E402
from lead_lag.data.wrds_fetch import load_or_fetch_factors_daily  # noqa: E402
from lead_lag.portfolio import (  # noqa: E402
    WINDOWS,
    attribution,
    book_returns,
    calendar_lagged_signals,
    drawdown,
    holm,
    industry_book,
    stock_book,
    stress_table,
    summary_stats,
    turnover,
    var_backtest,
    vol_scaled,
    window_slice,
)
from lead_lag.shocks.decomposition import LeaderShocks  # noqa: E402
from lead_lag.signals import (  # noqa: E402
    conditional_signal_panel,
    forward_returns,
    quantile_portfolios,
)

START, END = "1996-01-01", "2024-12-31"
RESULTS = PROJECT_ROOT / "results"
CACHE = PROJECT_ROOT / "cache.nosync"
FIGURES = PROJECT_ROOT / "figures"
for d in (RESULTS, FIGURES):
    d.mkdir(exist_ok=True)

BOOKS = {  # name -> (construction, signal column)
    "conditional_signal": ("industry", "conditional_signal"),
    "common_lag": ("industry", "common_lag"),
    "leader_ret_lag": ("industry", "leader_ret_lag"),
    "shock_lag": ("industry", "shock_lag"),
    "reversal": ("stock", "reversal_signal"),
}
PRIMARY = ["conditional_signal", "common_lag"]
CONVENTIONS = {"C": "ret_C", "A": "ret_A", "B": "ret_B", "overnight": "ret_O"}
HEADLINE = ("conditional_signal", "C")
RECENT = ("2007-01-01", "2024-12-31")   # the window the cost criterion is judged on
TOL_BP, TOL_T = 0.01, 0.01              # reconciliation tolerance

_t0 = time.time()


def step(label: str) -> None:
    print(f"\n[{time.time() - _t0:7.1f}s] {label}", flush=True)


def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python {produced_by}` first.")
    return path


# ------------------------------------------------------------------- load
step("loading inputs ...")
fp = pd.read_parquet(_require(RESULTS / "p1_follower_panel.parquet", "scripts/run_part1.py"))
shocks = LeaderShocks.from_raw(
    pd.read_parquet(_require(CACHE / "p2_shocks_rolling.parquet", "scripts/run_part2.py"))
)
inputs = pd.read_parquet(_require(CACHE / "p4_inputs.parquet", "scripts/prep_part4.py"))
mkt_c = pd.read_parquet(_require(CACHE / "p4_market_intraday.parquet", "scripts/prep_part4.py"))
factors = load_or_fetch_factors_daily(None, START, END).frame.sort_values("date")
factors["mktrf_lag"] = factors["mktrf"].shift(1)
factors = factors.set_index("date")
factors["mkt_C"] = mkt_c.set_index("date")["mkt_C"]
print(f"follower panel {len(fp):,} rows; p4_inputs {len(inputs):,} rows")

# ------------------------------------------------------- reconciliation
# Uses the team's own functions, so the committed Part 1 / Part 3 numbers must
# come out exactly; only the timing differs between rows.
step("reconciliation with Parts 1 and 3 ...")
sp = conditional_signal_panel(fp, shocks, lag=1, ret_col="ret")
fwd = forward_returns(fp, horizons=(1,), ret_col="ret")
sp_b = sp.merge(fwd[["date", "permno", "fwd_ret_1d"]], on=["date", "permno"], how="left")

q1 = quintile_sort(fp, lag=1, n_bins=5)
q1 = (q1.set_index("bin") if "bin" in q1.columns else q1).loc[-1]  # -1 = top minus bottom
recon = [{
    "check": "Part 1 quintile_sort (industry bins)", "signal": "leader_ret_lag",
    "timing": "A: signal t-1 -> return t", "spread_bp_day": q1["mean_ret"] * 1e4,
    "t_stat": q1["t_stat"], "committed_bp": 3.59, "committed_t": 4.34,
}]
for sig, committed in [("leader_ret_lag", (0.56, 0.67)), ("conditional_signal", (-0.12, -0.16))]:
    for label, frame, rc, com in [
        ("B: signal t-1 -> return t+1 (Part 3 as run)", sp_b, "fwd_ret_1d", committed),
        ("A: signal t-1 -> return t", sp, "ret", (np.nan, np.nan)),
    ]:
        q = quantile_portfolios(frame, signal_col=sig, ret_col=rc, n_quantiles=5)
        recon.append({
            "check": "Part 3 quantile_portfolios (stock quintiles)", "signal": sig,
            "timing": label, "spread_bp_day": q.spread_mean * 1e4,
            "t_stat": q.spread_t_stat, "committed_bp": com[0], "committed_t": com[1],
        })
recon = pd.DataFrame(recon)
has_ref = recon["committed_bp"].notna()
# committed values are published to 2 decimals, so compare at that precision
recon["matches"] = np.where(
    has_ref,
    ((recon["spread_bp_day"].round(2) - recon["committed_bp"]).abs() < TOL_BP)
    & ((recon["t_stat"].round(2) - recon["committed_t"]).abs() < TOL_T),
    np.nan,
)
recon.to_csv(RESULTS / "p4_reconciliation.csv", index=False)
print(recon.round(3).to_string(index=False))
recon_ok = bool(recon.loc[has_ref, "matches"].astype(bool).all())
if not recon_ok:
    print("WARNING: a reconciliation row does not reproduce its committed value")

# Part 4's books lag the leader on the trading calendar instead of by row
# (see calendar_lagged_signals). Count how many follower-days that changes.
calendar = pd.DatetimeIndex(factors.index)
lagged = calendar_lagged_signals(shocks.frame, calendar)
panel = fp[["date", "permno", "ff49"]].merge(lagged, on=["date", "ff49"], how="left")
team = sp[["date", "permno", "conditional_signal"]].rename(
    columns={"conditional_signal": "team_cs"}
)
cmp = panel[["date", "permno", "conditional_signal"]].merge(team, on=["date", "permno"], how="left")
differs = ~np.isclose(cmp["conditional_signal"], cmp["team_cs"], equal_nan=True)
lag_fix = pd.DataFrame([{
    "follower_days": len(cmp),
    "differs": int(differs.sum()),
    "share": float(differs.mean()),
    "only_in_calendar": int((cmp["conditional_signal"].notna() & cmp["team_cs"].isna()).sum()),
    "only_in_team": int((cmp["conditional_signal"].isna() & cmp["team_cs"].notna()).sum()),
}])
lag_fix.to_csv(RESULTS / "p4_calendar_lag_check.csv", index=False)
print(lag_fix.to_string(index=False))
del sp, sp_b, fwd, frame, q1, cmp, team, differs, fp
gc.collect()

# ----------------------------------------------------------------- panel
step("merging full-panel inputs ...")
panel = panel.merge(inputs, on=["date", "permno"], how="left")
del inputs
gc.collect()
panel["reversal_signal"] = -panel["own_lag"]
ok_c = panel["ret_C"].notna()
panel["ret_O"] = ((1 + panel["ret_A"]) / (1 + panel["ret_C"]) - 1).where(ok_c & panel["ret_A"].notna())
# close-to-close return on the same names that have an open print, so that
# A_Cvalid = C + overnight up to a small cross term
panel["ret_A_Cvalid"] = panel["ret_A"].where(ok_c)
panel["ret_mid_fb"] = panel["mid_ret"].where(panel["mid_ret_valid"].fillna(False), panel["retx"])

# ----------------------------------------------------------------- books
step("building books ...")
for name, (kind, col) in BOOKS.items():
    builder = industry_book if kind == "industry" else stock_book
    panel[f"w_{name}"] = builder(panel, col)
    active = panel.loc[panel[f"w_{name}"] != 0, "date"].nunique()
    print(f"  {name:20s} {kind:8s} active on {active:,} days")

panel[["date", "permno", "ff49"] + [f"w_{b}" for b in BOOKS]].to_parquet(
    CACHE / "p4_weights.parquet", index=False
)

# neutrality achieved, measured the way the construction defines it: a missing
# stock beta is imputed at its held industry-day mean (industry books only)
dates_arr = panel["date"].to_numpy()
ff_arr = panel["ff49"].to_numpy()
beta_arr = panel["beta"].to_numpy()
exp_rows = []
for name, (kind, _) in BOOKS.items():
    w = panel[f"w_{name}"].to_numpy()
    held = w != 0
    e = pd.DataFrame({"date": dates_arr[held], "ff49": ff_arr[held],
                      "w": w[held], "beta": beta_arr[held]})
    missing = e["beta"].isna()
    if kind == "industry":
        e["beta"] = e["beta"].fillna(e.groupby(["date", "ff49"])["beta"].transform("mean"))
    e["wb"] = e["w"] * e["beta"]
    daily = e.groupby("date").agg(net=("w", "sum"), gross=("w", lambda s: s.abs().sum()),
                                  beta=("wb", "sum"))
    exp_rows.append({
        "book": name, "mean_abs_net": daily["net"].abs().mean(),
        "mean_gross": daily["gross"].mean(),
        "mean_abs_beta_exposure": daily["beta"].abs().mean(),
        "max_abs_beta_exposure": daily["beta"].abs().max(),
        "share_held_rows_missing_beta": float(missing.mean()),
    })
    del e
pd.DataFrame(exp_rows).to_csv(RESULTS / "p4_exposure_check.csv", index=False)

# --------------------------------------------------------------- returns
step("book returns under each convention ...")
series: dict[tuple[str, str], pd.Series] = {}
status: dict[tuple[str, str], str] = {}
daily_rows = []
turnovers: dict[str, pd.Series] = {}


def _keep(name: str, conv: str, r: pd.Series, how: str, cov: pd.Series | None = None) -> None:
    series[(name, conv)] = r
    status[(name, conv)] = how
    f = pd.DataFrame({"date": r.index, "book": name, "convention": conv, "ret": r.to_numpy()})
    if cov is not None:
        f["coverage"] = cov.reindex(r.index).to_numpy()
    daily_rows.append(f)


for name in BOOKS:
    for conv, rc in CONVENTIONS.items():
        br = book_returns(panel, f"w_{name}", rc)
        _keep(name, conv, br["ret"], "pre-registered", br["coverage"])
    br = book_returns(panel, f"w_{name}", "ret_A_Cvalid")
    _keep(name, "A_Cvalid", br["ret"], "diagnostic", br["coverage"])
    turnovers[name] = turnover(panel, f"w_{name}")
for rc, label in (("retx", "A_retx"), ("ret_mid_fb", "A_mid")):
    br = book_returns(panel, "w_reversal", rc)
    _keep("reversal", label, br["ret"], "pre-registered", br["coverage"])
    br = book_returns(panel, "w_conditional_signal", rc)
    _keep("conditional_signal", label, br["ret"], "diagnostic", br["coverage"])
hb, hc = HEADLINE
_keep(hb, f"{hc}_volscaled", vol_scaled(series[(hb, hc)]), "pre-registered")

daily_out = pd.concat(daily_rows, ignore_index=True)
for name, to in turnovers.items():
    daily_out = pd.concat([daily_out, pd.DataFrame({
        "date": to.index, "book": name, "convention": "turnover_carry", "ret": to.to_numpy(),
    })], ignore_index=True)
daily_out.to_csv(RESULTS / "p4_portfolio_daily.csv", index=False)
del daily_out, daily_rows

# ------------------------------------------------------------ performance
step("performance by book, convention and window ...")
perf = []
for (name, conv), r in series.items():
    for wname, (a, b) in WINDOWS.items():
        perf.append({"book": name, "convention": conv, "window": wname,
                     "status": status[(name, conv)], **summary_stats(window_slice(r, a, b))})
perf = pd.DataFrame(perf)
perf.to_csv(RESULTS / "p4_performance.csv", index=False)
full = perf[perf["window"] == "full 1996-2024"].set_index(["book", "convention"])
print(full[["ann_mean", "ann_vol", "sharpe", "sharpe_se", "t_nw", "max_dd"]].round(3).to_string())

# ------------------------------------------------------- primary tests
step("primary family, Holm-corrected ...")
p_raw = {f"{b} {c}": float(full.loc[(b, c), "p_nw"]) for b in PRIMARY for c in ("C", "A")}
p_adj = holm(p_raw)
prim = pd.DataFrame([{
    "test": k, "ann_mean": full.loc[tuple(k.split(" ")), "ann_mean"],
    "t_nw": full.loc[tuple(k.split(" ")), "t_nw"], "p_raw": p_raw[k], "p_holm": p_adj[k],
} for k in p_raw])
prim.to_csv(RESULTS / "p4_primary_tests.csv", index=False)
print(prim.round(4).to_string(index=False))

# ------------------------------------------------------------ attribution
# Convention C is open-to-close, so its market regressor is the value-weighted
# open-to-close market built from CRSP (mkt_C). The close-to-close version is
# kept as a sensitivity. The other FF factors exist only close-to-close.
step("factor attribution and the Hypothesis 3 spanning test ...")
STYLE = ["smb", "hml", "rmw", "cma", "umd", "mktrf_lag"]
att = []
for name in BOOKS:
    for conv, mkt in (("C", "mkt_C"), ("C_ccmkt", "mktrf"), ("A", "mktrf")):
        base = "C" if conv.startswith("C") else conv
        X = factors[[mkt] + STYLE].copy()
        if name != "reversal":
            X["reversal"] = series[("reversal", base)]
        att.append({"book": name, "convention": conv, "market_regressor": mkt,
                    **attribution(series[(name, base)], X)})
att = pd.DataFrame(att)
att.to_csv(RESULTS / "p4_attribution.csv", index=False)
show = ["book", "convention", "alpha_ann", "t_alpha", "b_hml", "b_umd", "b_reversal",
        "t_reversal", "r2"]
print(att[[c for c in show if c in att.columns]].round(3).to_string(index=False))

# -------------------------------------------------------------- risk
step("VaR/ES backtests and stress periods ...")
var_rows, stress_rows = [], []
for name in PRIMARY + ["reversal"]:
    for conv in ("C", "A"):
        r = series[(name, conv)]
        var_rows.append({"book": name, "convention": conv, **var_backtest(r)})
        st = stress_table(r)
        st.insert(0, "convention", conv)
        st.insert(0, "book", name)
        stress_rows.append(st)
pd.DataFrame(var_rows).to_csv(RESULTS / "p4_var_backtest.csv", index=False)
pd.concat(stress_rows).to_csv(RESULTS / "p4_stress.csv", index=False)

# ----------------------------------------------------------- costs
# A holds positions overnight and rebalances once a day: traded = sum|dw|.
# C must be flat overnight to earn exactly open -> close: it buys the whole book
# at the open and sells it at the close, so traded = 2 * gross = 2.
step("turnover and breakeven cost ...")
held_recent = (panel["w_conditional_signal"] != 0) & (panel["date"] >= RECENT[0])
half_spread_bp = float(panel.loc[held_recent, "rel_spread_lag"].median() / 2 * 1e4)
gross = {}
for n in BOOKS:
    w = panel[f"w_{n}"].to_numpy()
    held = w != 0
    gross[n] = pd.Series(np.abs(w[held])).groupby(dates_arr[held]).sum()
cost_rows = []
for name in BOOKS:
    traded = {"A": turnovers[name], "C": 2.0 * gross[name]}
    row = {"book": name,
           "mean_daily_traded_A_carry": float(traded["A"].mean()),
           "mean_daily_traded_C_roundtrip": float(traded["C"].mean())}
    for conv in ("C", "A"):
        r = series[(name, conv)]
        for tag, (a, b) in (("full", ("1996-01-01", END)), ("2007_2024", RECENT)):
            rr = window_slice(r.dropna(), a, b)
            tt = traded[conv].reindex(rr.index)
            row[f"breakeven_bp_{conv}_{tag}"] = float(rr.mean() / tt.mean() * 1e4)
        cov = book_returns(panel, f"w_{name}", CONVENTIONS[conv])["coverage"]
        row[f"coverage_{conv}"] = float(cov.mean())
    cost_rows.append(row)
costs = pd.DataFrame(cost_rows)
costs["median_half_spread_bp_2007_2024"] = half_spread_bp
costs.to_csv(RESULTS / "p4_costs.csv", index=False)
print(costs.round(3).to_string(index=False))

# ----------------------------------------------------------- decision
step("pre-registered decision rule ...")
key = f"{hb} {hc}"
a_row = att[(att["book"] == hb) & (att["convention"] == hc)].iloc[0]
be = float(costs.loc[costs["book"] == hb, f"breakeven_bp_{hc}_2007_2024"].iloc[0])
criteria = {
    "1. mean return > 0, Holm p < 0.05": (
        bool(full.loc[(hb, hc), "ann_mean"] > 0 and p_adj[key] < 0.05),
        f"ann mean {full.loc[(hb, hc), 'ann_mean']:.4f}, Holm p {p_adj[key]:.4f}"),
    "2. spanning alpha > 0, p < 0.05 (open-to-close market)": (
        bool(a_row["alpha_ann"] > 0 and a_row["p_alpha"] < 0.05),
        f"alpha {a_row['alpha_ann']:.4f}, p {a_row['p_alpha']:.4f}"),
    "3. breakeven cost > median half-spread (both 2007-2024)": (
        bool(be > half_spread_bp),
        f"breakeven {be:.2f} bp vs half-spread {half_spread_bp:.2f} bp"),
}
verdict = "IMPLEMENT" if all(v[0] for v in criteria.values()) else "DO NOT IMPLEMENT"
dec = pd.DataFrame(
    [{"criterion": k, "passed": v[0], "value": v[1]} for k, v in criteria.items()]
    + [{"criterion": "reconciliation reproduces Parts 1 and 3", "passed": recon_ok,
        "value": "all committed rows match" if recon_ok else "MISMATCH, see p4_reconciliation.csv"},
       {"criterion": "VERDICT", "passed": verdict == "IMPLEMENT", "value": verdict}]
)
dec.to_csv(RESULTS / "p4_decision.csv", index=False)
print(dec.to_string(index=False))

# ----------------------------------------------------------- figures
step("figures ...")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = {"conditional_signal": "#1f4e79", "common_lag": "#c0504d", "reversal": "#7f7f7f"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, conv, title in [(axes[0], "C", "Open to close (implementable)"),
                        (axes[1], "A", "Close to close (upper bound)")]:
    for name in PRIMARY + ["reversal"]:
        r = series[(name, conv)].fillna(0)
        ax.plot(r.index, r.cumsum() * 100, color=colors[name],
                ls="--" if name == "reversal" else "-", lw=1.2, label=name)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(pd.Timestamp("2007-01-01"), color="grey", lw=0.6, ls=":")
    ax.set_title(title)
axes[0].set_ylabel("Cumulative return, gross 1 (%)")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES / "p4_cumulative.png", dpi=200)
plt.close(fig)

bars = []
for name in PRIMARY + ["reversal"]:
    for wname in ("1996-2006", "2007-2024"):
        for conv in ("A_Cvalid", "C", "overnight"):
            s = perf[(perf["book"] == name) & (perf["convention"] == conv)
                     & (perf["window"] == wname)]
            bars.append({"book": name, "window": wname, "conv": conv,
                         "ann_mean": float(s["ann_mean"].iloc[0]) * 100})
bars = pd.DataFrame(bars)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
labels = {"A_Cvalid": "close to close", "C": "open to close", "overnight": "overnight"}
for ax, name in zip(axes, PRIMARY + ["reversal"]):
    sub = bars[bars["book"] == name].pivot(index="window", columns="conv", values="ann_mean")
    sub = sub[list(labels)].rename(columns=labels)
    sub.plot.bar(ax=ax, rot=0, width=0.75, color=["#1f4e79", "#9dc3e6", "#bfbfbf"])
    ax.axhline(0, color="black", lw=0.6, label="_nolegend_")
    ax.set_title(name)
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Annualized mean return (%)")
fig.tight_layout()
fig.savefig(FIGURES / "p4_timing_decomposition.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.2))
for conv, ls in (("C", "-"), ("A", "--")):
    ax.plot(drawdown(series[(hb, conv)]) * 100, color="#1f4e79", ls=ls, lw=1,
            label=f"{hb}, {'open to close' if conv == 'C' else 'close to close'}")
ax.set_ylabel("Drawdown (%)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "p4_drawdown.png", dpi=200)
plt.close(fig)

for f in sorted(list(RESULTS.glob("p4_*")) + list(FIGURES.glob("p4_*"))):
    print(f"  {f.relative_to(PROJECT_ROOT)!s:<42} {f.stat().st_size / 1e3:9.1f} KB")
step(f"done. verdict: {verdict}")
