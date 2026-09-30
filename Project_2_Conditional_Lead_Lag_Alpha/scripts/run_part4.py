"""
Run Part 4 end to end: portfolio construction and risk, per docs/part4_prereg.md.

    PYTHONPATH=src python scripts/run_part4.py 2>&1 | tee cache.nosync/part4.log

Requires, in order:
    1. scripts/run_part1.py   -> results/p1_follower_panel.parquet
    2. scripts/run_part2.py   -> cache.nosync/p2_shocks_rolling.parquet
    3. scripts/prep_part4.py  -> cache.nosync/p4_inputs.parquet

Aggregate outputs (safe to commit): results/p4_*.csv, figures/p4_*.png.
Stock-level weights go to cache.nosync/p4_weights.parquet and must not be
committed (the team repo is public; CRSP-derived stock-level data is licensed).
"""

from __future__ import annotations

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
factors = load_or_fetch_factors_daily(None, START, END).frame.sort_values("date")
factors["mktrf_lag"] = factors["mktrf"].shift(1)
factors = factors.set_index("date")
print(f"follower panel {len(fp):,} rows; p4_inputs {len(inputs):,} rows")

# ------------------------------------------------------- reconciliation
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
recon.to_csv(RESULTS / "p4_reconciliation.csv", index=False)
print(recon.round(3).to_string(index=False))
del sp_b, fwd

# ----------------------------------------------------------------- panel
step("merging full-panel inputs onto the signal panel ...")
panel = sp.merge(inputs, on=["date", "permno"], how="left", suffixes=("_fp", ""))
del sp, inputs
panel["reversal_signal"] = -panel["own_lag"]
ok_o = panel["ret_A"].notna() & panel["ret_C"].notna()
panel["ret_O"] = ((1 + panel["ret_A"]) / (1 + panel["ret_C"]) - 1).where(ok_o)
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

# neutrality actually achieved, on raw stock betas (missing betas excluded)
exp_rows = []
for name in BOOKS:
    w = panel[f"w_{name}"]
    held = w != 0
    g = panel.loc[held].assign(wb=(w * panel["beta"])[held], w=w[held])
    daily = g.groupby("date").agg(net=("w", "sum"), gross=("w", lambda s: s.abs().sum()),
                                  beta=("wb", "sum"))
    exp_rows.append({"book": name, "mean_abs_net": daily["net"].abs().mean(),
                     "mean_gross": daily["gross"].mean(),
                     "mean_abs_beta_exposure": daily["beta"].abs().mean(),
                     "max_abs_beta_exposure": daily["beta"].abs().max()})
pd.DataFrame(exp_rows).to_csv(RESULTS / "p4_exposure_check.csv", index=False)

# --------------------------------------------------------------- returns
step("book returns under each convention ...")
series: dict[tuple[str, str], pd.Series] = {}
daily_rows = []
for name in BOOKS:
    for conv, rc in CONVENTIONS.items():
        br = book_returns(panel, f"w_{name}", rc)
        series[(name, conv)] = br["ret"]
        daily_rows.append(br.assign(book=name, convention=conv).reset_index())
    to = turnover(panel, f"w_{name}")
    daily_rows.append(pd.DataFrame({"date": to.index, "book": name,
                                    "convention": "turnover", "ret": to.to_numpy()}))
for name in ("reversal", "conditional_signal"):
    for rc, label in (("retx", "A_retx"), ("ret_mid_fb", "A_mid")):
        br = book_returns(panel, f"w_{name}", rc)
        series[(name, label)] = br["ret"]
        daily_rows.append(br.assign(book=name, convention=label).reset_index())
for name in PRIMARY:
    series[(name, "C_volscaled")] = vol_scaled(series[(name, "C")])
    daily_rows.append(pd.DataFrame({"date": series[(name, "C_volscaled")].index,
                                    "book": name, "convention": "C_volscaled",
                                    "ret": series[(name, "C_volscaled")].to_numpy()}))
pd.concat(daily_rows, ignore_index=True).to_csv(
    RESULTS / "p4_portfolio_daily.csv", index=False
)

# ------------------------------------------------------------ performance
step("performance by book, convention and window ...")
perf = []
for (name, conv), r in series.items():
    for wname, (a, b) in WINDOWS.items():
        perf.append({"book": name, "convention": conv, "window": wname,
                     **summary_stats(window_slice(r, a, b))})
perf = pd.DataFrame(perf)
perf.to_csv(RESULTS / "p4_performance.csv", index=False)
full = perf[perf["window"] == "full 1996-2024"].set_index(["book", "convention"])
print(full[["ann_mean", "ann_vol", "sharpe", "sharpe_se", "t_nw", "max_dd"]].round(3).to_string())

# ------------------------------------------------------- primary tests
step("primary family, Holm-corrected ...")
p_raw = {f"{b} {c}": full.loc[(b, c), "p_nw"] for b in PRIMARY for c in ("C", "A")}
p_adj = holm(p_raw)
prim = pd.DataFrame([{
    "test": k, "ann_mean": full.loc[tuple(k.split(" ")), "ann_mean"],
    "t_nw": full.loc[tuple(k.split(" ")), "t_nw"], "p_raw": p_raw[k], "p_holm": p_adj[k],
} for k in p_raw])
prim.to_csv(RESULTS / "p4_primary_tests.csv", index=False)
print(prim.round(4).to_string(index=False))

# ------------------------------------------------------------ attribution
step("factor attribution and the Hypothesis 3 spanning test ...")
FACS = ["mktrf", "smb", "hml", "rmw", "cma", "umd", "mktrf_lag"]
att = []
for name in BOOKS:
    for conv in ("C", "A"):
        X = factors[FACS].copy()
        if name != "reversal":
            X["reversal"] = series[("reversal", conv)]
        att.append({"book": name, "convention": conv,
                    **attribution(series[(name, conv)], X)})
att = pd.DataFrame(att)
att.to_csv(RESULTS / "p4_attribution.csv", index=False)
show = ["book", "convention", "alpha_ann", "t_alpha", "b_mktrf", "b_hml", "b_umd",
        "b_reversal", "t_reversal", "r2"]
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
step("turnover and breakeven cost ...")
held_0724 = (panel["w_conditional_signal"] != 0) & (panel["date"] >= "2007-01-01")
half_spread_bp = float(panel.loc[held_0724, "rel_spread_lag"].median() / 2 * 1e4)
cost_rows = []
for name in BOOKS:
    to = turnover(panel, f"w_{name}")
    row = {"book": name, "mean_daily_turnover": float(to.mean())}
    for conv in ("C", "A"):
        r = series[(name, conv)]
        row[f"breakeven_bp_{conv}"] = float(r.mean() / to.reindex(r.index).mean() * 1e4)
        cov = book_returns(panel, f"w_{name}", CONVENTIONS[conv])["coverage"]
        row[f"coverage_{conv}"] = float(cov.mean())
    cost_rows.append(row)
costs = pd.DataFrame(cost_rows)
costs["median_half_spread_bp_2007_2024"] = half_spread_bp
costs.to_csv(RESULTS / "p4_costs.csv", index=False)
print(costs.round(3).to_string(index=False))

# ----------------------------------------------------------- decision
step("pre-registered decision rule ...")
hb, hc = HEADLINE
key = f"{hb} {hc}"
a_row = att[(att["book"] == hb) & (att["convention"] == hc)].iloc[0]
be = float(costs.loc[costs["book"] == hb, f"breakeven_bp_{hc}"].iloc[0])
criteria = {
    "1. mean return > 0, Holm p < 0.05": bool(
        full.loc[(hb, hc), "ann_mean"] > 0 and p_adj[key] < 0.05),
    "2. spanning alpha > 0, p < 0.05": bool(
        a_row["alpha_ann"] > 0 and a_row["p_alpha"] < 0.05),
    "3. breakeven cost > median half-spread": bool(be > half_spread_bp),
}
verdict = "IMPLEMENT" if all(criteria.values()) else "DO NOT IMPLEMENT"
dec = pd.DataFrame([{
    "criterion": k, "passed": v,
    "value": {0: f"ann mean {full.loc[(hb, hc), 'ann_mean']:.4f}, Holm p {p_adj[key]:.4f}",
              1: f"alpha {a_row['alpha_ann']:.4f}, p {a_row['p_alpha']:.4f}",
              2: f"breakeven {be:.2f} bp vs half-spread {half_spread_bp:.2f} bp"}[i],
} for i, (k, v) in enumerate(criteria.items())] + [
    {"criterion": "VERDICT", "passed": verdict == "IMPLEMENT", "value": verdict}])
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

dec_rows = []
for name in PRIMARY:
    for wname in ("1996-2006", "2007-2024"):
        for conv in ("A", "C", "overnight"):
            s = perf[(perf["book"] == name) & (perf["convention"] == conv)
                     & (perf["window"] == wname)]
            dec_rows.append({"book": name, "window": wname, "conv": conv,
                             "ann_mean": float(s["ann_mean"].iloc[0]) * 100})
dd = pd.DataFrame(dec_rows)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, name in zip(axes, PRIMARY):
    sub = dd[dd["book"] == name].pivot(index="window", columns="conv", values="ann_mean")
    sub[["A", "C", "overnight"]].plot.bar(ax=ax, rot=0, width=0.75,
                                          color=["#1f4e79", "#9dc3e6", "#bfbfbf"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(name)
    ax.set_xlabel("")
    ax.legend(["close to close (A)", "open to close (C)", "overnight"], frameon=False)
axes[0].set_ylabel("Annualised mean return (%)")
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
    print(f"  {f.relative_to(PROJECT_ROOT)!s:<40} {f.stat().st_size / 1e3:9.1f} KB")
step(f"done. verdict: {verdict}")
