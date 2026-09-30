"""Tests for Part 4 (src/lead_lag/portfolio). Offline, synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_lag.portfolio import (
    book_returns,
    calendar_lagged_signals,
    drawdown,
    dimson_beta,
    full_panel_inputs,
    holm,
    industry_book,
    stock_book,
    summary_stats,
    turnover,
    var_backtest,
    vol_scaled,
)

TOL = 1e-10


# ----------------------------------------------------------------- fixtures
def _follower_panel(n_dates=40, n_ind=15, per_ind=12, seed=0, nan_share=0.1):
    """Follower-day panel with an industry-level signal and stock beta/own-lag."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_dates)
    rows = []
    for d in dates:
        sig = rng.normal(size=n_ind)
        for j in range(n_ind):
            for k in range(per_ind):
                rows.append((d, j + 1, (j + 1) * 1000 + k, sig[j]))
    p = pd.DataFrame(rows, columns=["date", "ff49", "permno", "sig"])
    p["beta"] = rng.uniform(0.5, 1.5, len(p))
    p["own_lag"] = rng.normal(0, 0.02, len(p))
    p["stock_sig"] = rng.normal(size=len(p))
    for c in ("beta", "own_lag"):
        p.loc[rng.random(len(p)) < nan_share, c] = np.nan
    return p


def _full_panel(n=600, seed=1):
    """Two stocks, a market factor, and quote/open data for full_panel_inputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n)
    mkt = rng.normal(0, 0.01, n)
    fac = pd.DataFrame({"date": dates, "mktrf": mkt, "rf": 0.0})
    rets = []
    for permno, (b0, b1) in {10: (0.8, 0.4), 20: (1.0, 0.0)}.items():
        lagged = np.r_[0.0, mkt[:-1]]
        r = b0 * mkt + b1 * lagged + rng.normal(0, 0.005, n)
        prc = 50 * np.cumprod(1 + r)
        rets.append(pd.DataFrame({"date": dates, "permno": permno, "ret": r, "prc": prc,
                                  "mktcap": prc * 1e6}))
    ret = pd.concat(rets, ignore_index=True)
    ret["dlret"] = np.nan
    ext = ret[["date", "permno", "prc"]].copy()
    ext["openprc"] = ext["prc"] / (1 + rng.normal(0, 0.003, len(ext)))
    ext["bid"] = ext["prc"] * 0.999
    ext["ask"] = ext["prc"] * 1.001
    ext["cfacpr"] = 1.0
    ext["quote_only"] = False
    ext["retx"] = ret["ret"]
    return ret, ext.drop(columns="prc"), fac


# ------------------------------------------------------------- construction
def test_industry_book_is_exactly_neutral():
    p = _follower_panel()
    w = industry_book(p, "sig")
    p = p.assign(w=w)
    # constraint values as the construction sees them: missing stock values
    # imputed at their industry-day mean
    for c in ("beta", "own_lag"):
        p[c + "_imp"] = p[c].fillna(p.groupby(["date", "ff49"])[c].transform("mean"))
    for _, g in p.groupby("date"):
        if (g["w"] != 0).sum() == 0:
            continue
        assert abs(g["w"].sum()) < TOL
        assert abs(g["w"].abs().sum() - 1) < TOL
        assert abs((g["w"] * g["beta_imp"]).sum()) < TOL
        assert abs((g["w"] * g["own_lag_imp"]).sum()) < TOL


def test_industry_book_weights_equal_within_industry_and_long_high_signal():
    p = _follower_panel(nan_share=0.0)
    p["w"] = industry_book(p, "sig")
    spread = p.groupby(["date", "ff49"])["w"].agg(lambda s: s.max() - s.min())
    assert spread.max() < TOL
    ind = p.groupby(["date", "ff49"]).agg(w=("w", "first"), s=("sig", "first"))
    assert ind["w"].corr(ind["s"], method="spearman") > 0.5


def test_industry_neutral_demeaning_would_zero_an_industry_signal():
    """The design reason for the industry book: an FF49-demeaned
    industry-level signal is identically zero."""
    p = _follower_panel()
    dm = p["sig"] - p.groupby(["date", "ff49"])["sig"].transform("mean")
    assert dm.abs().max() < TOL


def test_industry_book_skips_thin_dates():
    p = _follower_panel(n_ind=5)
    assert (industry_book(p, "sig", min_industries=10) == 0).all()


def test_stock_book_is_dollar_industry_and_beta_neutral():
    p = _follower_panel(per_ind=20)
    p["w"] = stock_book(p, "stock_sig", min_names=10)
    for _, g in p.groupby("date"):
        g = g[g["w"] != 0]
        assert abs(g["w"].sum()) < TOL
        assert abs(g["w"].abs().sum() - 1) < TOL
        assert abs((g["w"] * g["beta"]).sum()) < TOL
        assert g.groupby("ff49")["w"].sum().abs().max() < TOL


def test_stock_book_excludes_rows_without_beta():
    p = _follower_panel(per_ind=20)
    w = stock_book(p, "stock_sig", min_names=10)
    assert (w[p["beta"].isna()] == 0).all()


def test_book_returns_and_coverage():
    p = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 4),
        "w": [0.25, 0.25, -0.25, -0.25],
        "r": [0.02, np.nan, 0.01, -0.01],
    })
    out = book_returns(p, "w", "r").iloc[0]
    assert out["ret"] == pytest.approx(0.25 * 0.02 - 0.25 * 0.01 + 0.25 * 0.01)
    assert out["coverage"] == pytest.approx(0.75)


def test_turnover_counts_entries_exits_and_changes():
    d1, d2 = pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")
    p = pd.DataFrame({
        "date": [d1, d1, d2, d2],
        "permno": [1, 2, 2, 3],
        "w": [0.5, -0.5, -0.2, 0.2],
    })
    to = turnover(p, "w")
    # day 2: permno 1 exits (0.5), permno 2 changes (0.3), permno 3 enters (0.2)
    assert to.loc[d2] == pytest.approx(1.0)
    assert to.loc[d1] == pytest.approx(1.0)


def test_vol_scaled_uses_only_past_returns():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=300)
    r = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    base = vol_scaled(r)
    shocked = r.copy()
    shocked.iloc[200:] *= 10
    # day 200's scale depends only on days < 200, so the scale itself is unchanged
    assert (vol_scaled(shocked).iloc[:201] / shocked.iloc[:201]).equals(
        base.iloc[:201] / r.iloc[:201]
    )


# ------------------------------------------------------------------ inputs
def test_dimson_beta_recovers_sum_of_slopes():
    ret, ext, fac = _full_panel(n=2000)
    fac = fac.sort_values("date").assign(mktrf_lag=lambda f: f["mktrf"].shift(1))
    df = ret.merge(fac, on="date").sort_values(["permno", "date"]).reset_index(drop=True)
    b = dimson_beta(df, window=1500, min_obs=60, shrink_weight=1.0)
    last = df.assign(b=b).groupby("permno")["b"].last()
    assert last.loc[10] == pytest.approx(1.2, abs=0.05)
    assert last.loc[20] == pytest.approx(1.0, abs=0.05)


def test_dimson_beta_is_point_in_time():
    """Appending data after T must not change any beta dated <= T, and the
    beta dated t must not use day t itself."""
    ret, ext, fac = _full_panel(n=500)
    fac = fac.sort_values("date").assign(mktrf_lag=lambda f: f["mktrf"].shift(1))
    full = ret.merge(fac, on="date").sort_values(["permno", "date"]).reset_index(drop=True)
    T = full["date"].unique()[399]
    trunc = full[full["date"] <= T].reset_index(drop=True)
    b_full = full.assign(b=dimson_beta(full, window=100, min_obs=30))
    b_trunc = trunc.assign(b=dimson_beta(trunc, window=100, min_obs=30))
    m = b_trunc.merge(b_full, on=["date", "permno"], suffixes=("_t", "_f"))
    assert np.allclose(m["b_t"], m["b_f"], equal_nan=True)
    # changing day t's return must not change the beta dated t
    bumped = full.copy()
    i = bumped.index[(bumped["permno"] == 10) & (bumped["date"] == T)][0]
    bumped.loc[i, "ret"] += 0.5
    assert dimson_beta(bumped, window=100, min_obs=30).loc[i] == pytest.approx(
        b_full.loc[i, "b"]
    )


def test_full_panel_inputs_timing():
    ret, ext, fac = _full_panel(n=300)
    out = full_panel_inputs(ret, ext, fac, beta_window=100, beta_min_obs=30)
    s = out[out["permno"] == 10].reset_index(drop=True)
    r = ret[ret["permno"] == 10].sort_values("date").reset_index(drop=True)
    assert np.allclose(s["ret_A"], r["ret"])
    assert np.allclose(s["ret_B"].iloc[:-1], r["ret"].iloc[1:])
    assert np.allclose(s["own_lag"].iloc[1:], r["ret"].iloc[:-1])


def test_follower_gap_does_not_leak_into_next_day_return():
    """A stock absent from the follower set in the middle month must still get
    the TRUE next-session return, which is what computing on the full panel buys."""
    ret, ext, fac = _full_panel(n=80)
    out = full_panel_inputs(ret, ext, fac, beta_window=30, beta_min_obs=10)
    s = out[out["permno"] == 10].reset_index(drop=True)
    followers = s[(s.index < 20) | (s.index >= 40)]      # gap in rows 20-39
    last_before_gap = followers.loc[19]
    true_next = ret[ret["permno"] == 10].sort_values("date")["ret"].iloc[20]
    assert last_before_gap["ret_B"] == pytest.approx(true_next)


def test_open_to_close_and_delisting():
    d = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    ret = pd.DataFrame({
        "date": d, "permno": 7, "mktcap": 1e9,
        "prc": [11.0, 11.0, 11.0, np.nan],
        "ret": [0.0, 0.0, -0.23, -0.5],
        "dlret": [np.nan, np.nan, -0.3, -0.5],
    })
    ext = pd.DataFrame({
        "date": d, "permno": 7, "openprc": [10.0, np.nan, 10.0, np.nan],
        "bid": 10.9, "ask": 11.1, "cfacpr": 1.0, "quote_only": False, "retx": 0.0,
    })
    fac = pd.DataFrame({"date": d, "mktrf": 0.0, "rf": 0.0})
    out = full_panel_inputs(ret, ext, fac, beta_window=5, beta_min_obs=2).set_index("date")
    assert out.loc[d[0], "ret_C"] == pytest.approx(0.1)
    assert np.isnan(out.loc[d[1], "ret_C"]) and not out.loc[d[1], "c_valid"]
    assert out.loc[d[2], "ret_C"] == pytest.approx(1.1 * 0.7 - 1)
    # delisting-only day: no open print to buy at, so no open-to-close return
    assert np.isnan(out.loc[d[3], "ret_C"]) and not out.loc[d[3], "c_valid"]


# ------------------------------------------------------------- performance
def test_planted_lag1_effect_is_seen_by_A_not_B():
    """Mirrors Part 1's lag-2 placebo: a book long on yesterday's leader move
    earns the planted effect under convention A and nothing under B."""
    rng = np.random.default_rng(7)
    n, k = 800, 30
    sig = rng.normal(0, 0.02, n)                 # row t: the leader's move on t-1
    load = rng.choice([-1.0, 1.0], k)            # how each stock responds
    r = 0.5 * np.outer(sig, load) + rng.normal(0, 0.01, (n, k))   # day-t returns
    w = load / np.abs(load).sum()
    pos = np.sign(sig)                           # position formed from row t's signal
    A = pd.Series((r @ w) * pos)                                  # earns day t
    B = pd.Series(np.r_[(r[1:] @ w) * pos[:-1], np.nan])          # earns day t+1
    assert summary_stats(A)["t_nw"] > 10
    assert abs(summary_stats(B)["t_nw"]) < 3


def test_var_backtest_calibrated_on_iid_normal():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0, 0.01, 6000), index=pd.bdate_range("2000-01-03", periods=6000))
    out = var_backtest(r)
    assert out["exceedances"] == pytest.approx(out["expected"], rel=0.35)
    assert out["kupiec_p"] > 0.01
    assert out["avg_es"] > out["avg_var"] * 0.9


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


# ------------------------------------------------- fixes from the code audit
def test_drawdown_counts_a_first_day_loss():
    r = pd.Series([-0.10, 0.0, 0.0, 0.05, 0.0])
    assert drawdown(r).min() == pytest.approx(-0.10)
    assert summary_stats(pd.concat([r] * 5, ignore_index=True))["max_dd"] < 0


def test_kupiec_is_finite_with_zero_exceedances():
    r = pd.Series(np.linspace(0.001, 0.002, 1500),
                  index=pd.bdate_range("2000-01-03", periods=1500))
    out = var_backtest(r)
    assert out["exceedances"] == 0
    assert np.isfinite(out["kupiec_lr"]) and out["kupiec_p"] < 1e-3
    assert out["christoffersen_lr"] == pytest.approx(0.0)


def test_var_backtest_short_series_does_not_crash():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200),
                  index=pd.bdate_range("2000-01-03", periods=200))
    assert var_backtest(r)["n_days"] == 0


def test_book_returns_zero_coverage_is_missing_not_zero():
    p = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 2),
                      "w": [0.5, -0.5], "r": [np.nan, np.nan]})
    assert np.isnan(book_returns(p, "w", "r").iloc[0]["ret"])


def test_holm_rejects_missing_pvalues():
    with pytest.raises(ValueError):
        holm({"a": np.nan, "b": 0.01})


def test_calendar_lag_uses_previous_session_only():
    cal = pd.bdate_range("2021-01-04", periods=6)
    shocks = pd.DataFrame({
        "date": cal[[0, 1, 3, 4]],          # the leader has no row on cal[2]
        "ff49": 1, "common": [1.0, 2.0, 4.0, 5.0], "shock": 0.5,
        "leader_ret": [1.5, 2.5, 4.5, 5.5],
    })
    out = calendar_lagged_signals(shocks, cal).set_index("date")
    assert out.loc[cal[1], "common_lag"] == 1.0      # from cal[0]
    assert out.loc[cal[2], "common_lag"] == 2.0      # from cal[1]
    # cal[2] had no leader row, so cal[3] has NO signal (not a stale cal[1] value)
    assert cal[3] not in out.index
    assert out.loc[cal[4], "common_lag"] == 4.0
    assert out.loc[cal[5], "conditional_signal"] == pytest.approx(5.0 - 0.5)
    # a missing day t leaves day t's own signal intact (it depends on t-1 only)
    assert out.loc[cal[2], "leader_ret_lag"] == 2.5
