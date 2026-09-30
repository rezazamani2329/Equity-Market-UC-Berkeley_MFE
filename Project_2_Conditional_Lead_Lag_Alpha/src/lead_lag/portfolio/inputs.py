"""
Part 4 inputs computed on the FULL daily CRSP panel.

Why the full panel
------------------
`leaders.follower_panel` keeps only the months in which a stock is a follower.
Shifting by row inside it jumps across membership gaps: a stock that is a
follower in January and March but not February gets March 1st's return as the
"next session" after January 31st. Every shifted quantity here (next-day
return, own lag, rolling beta, midpoint return) is therefore computed on the
full panel of all CRSP stocks and merged onto follower rows afterwards.

Timing, anchored on a follower row dated t
------------------------------------------
Part 4 always forms weights from row t's signal, which uses information up to
the close of t-1 (`conditional_signal_panel(lag=1)`). The three conventions in
docs/part4_prereg.md then differ only in which return the position earns:

    ret_A   close t-1 -> close t      the row's own `ret` (delisting-adjusted)
    ret_C   open t    -> close t      prc_t / openprc_t - 1, plus any delisting return on t
    ret_B   close t   -> close t+1    the next session's `ret`

Everything that conditions the weights (own lag, beta, spread) is known by
the close of t-1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lead_lag.data.microstructure import attach_quote_columns, midpoint_returns


def _rolling_sum_by_group(
    values: np.ndarray, group_start: np.ndarray, window: int
) -> np.ndarray:
    """Trailing sum over the last `window` rows within each group, O(n).

    `values` must already be zero wherever the observation is invalid.
    `group_start[i]` is the row position where row i's group begins; the frame
    must be sorted by (group, time). Uses a cumulative sum and subtracts the
    cumulative sum `window` rows back, clamped at the group's first row.
    """
    n = len(values)
    cs = np.concatenate([[0.0], np.cumsum(values)])
    pos = np.arange(n)
    lo = np.maximum(pos - window + 1, group_start)
    return cs[pos + 1] - cs[lo]


def dimson_beta(
    frame: pd.DataFrame,
    window: int = 252,
    min_obs: int = 60,
    shrink_weight: float = 0.5,
    ret_col: str = "ret",
) -> pd.Series:
    """Dimson (1979) market beta, point-in-time as of the previous close.

    Regresses the stock's excess return on the market excess return on the
    same day and the day before, over the `window` sessions ENDING AT t-1, and
    sums the two slopes. Summing the lagged slope corrects the downward bias of
    plain OLS beta for thinly traded stocks, which is most of the follower set.
    The estimate is then shrunk toward 1 (weight `shrink_weight` on the
    estimate).

    `frame` must be sorted by (permno, date) and carry `ret_col`, `rf`,
    `mktrf` and `mktrf_lag`. Rows with fewer than `min_obs` valid
    observations in the window get NaN.
    """
    y = (frame[ret_col] - frame["rf"]).to_numpy(dtype=float)
    x1 = frame["mktrf"].to_numpy(dtype=float)
    x2 = frame["mktrf_lag"].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    y0, a0, b0 = np.where(ok, y, 0.0), np.where(ok, x1, 0.0), np.where(ok, x2, 0.0)

    perm = frame["permno"].to_numpy()
    new_group = np.r_[True, perm[1:] != perm[:-1]]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(len(perm)), 0))

    def rs(v: np.ndarray) -> np.ndarray:
        return _rolling_sum_by_group(v, group_start, window)

    n = rs(ok.astype(float))
    s1, s2, sy = rs(a0), rs(b0), rs(y0)
    s11, s22, s12 = rs(a0 * a0), rs(b0 * b0), rs(a0 * b0)
    s1y, s2y = rs(a0 * y0), rs(b0 * y0)

    with np.errstate(invalid="ignore", divide="ignore"):
        c11 = s11 - s1 * s1 / n
        c22 = s22 - s2 * s2 / n
        c12 = s12 - s1 * s2 / n
        c1y = s1y - s1 * sy / n
        c2y = s2y - s2 * sy / n
        det = c11 * c22 - c12 * c12
        b1 = (c1y * c22 - c2y * c12) / det
        b2 = (c2y * c11 - c1y * c12) / det
    beta = b1 + b2
    beta = np.where((n >= min_obs) & (np.abs(det) > 1e-18), beta, np.nan)
    beta = shrink_weight * beta + (1.0 - shrink_weight) * 1.0

    # window ends at t-1: shift one row within permno
    out = pd.Series(beta, index=frame.index)
    return out.groupby(frame["permno"].to_numpy(), sort=False).shift(1)


def full_panel_inputs(
    returns: pd.DataFrame,
    extras: pd.DataFrame,
    factors: pd.DataFrame,
    beta_window: int = 252,
    beta_min_obs: int = 60,
    shrink_weight: float = 0.5,
) -> pd.DataFrame:
    """Every Part 4 input that needs a shift, computed on the full panel.

    Args:
        returns: `DailyReturns.frame` (all CRSP common stocks, delisting-
            adjusted `ret`, `dlret`, `prc`, including appended delisting rows).
        extras: raw CRSP daily columns the returns frame drops: date, permno,
            openprc, bid, ask, cfacpr, quote_only, retx.
        factors: `FactorsDaily.frame` (date, mktrf, rf, ...).

    Returns:
        One row per (date, permno) of `returns` with ret_A, ret_B, ret_C,
        retx, mid_ret (+ `mid_ret_valid`), own_lag, beta, rel_spread_lag and
        c_valid (True when the open-to-close return is computable).
    """
    keep = ["date", "permno", "prc", "mktcap", "ret", "dlret"]
    df = returns[keep].merge(extras, on=["date", "permno"], how="left")
    fac = factors[["date", "mktrf", "rf"]].sort_values("date").copy()
    fac["mktrf_lag"] = fac["mktrf"].shift(1)
    df = df.merge(fac, on="date", how="left")
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    g = df.groupby("permno", sort=False)
    df["ret_A"] = df["ret"]
    df["ret_B"] = g["ret"].shift(-1)
    df["own_lag"] = g["ret"].shift(1)

    # open -> close on day t. prc and openprc are both absolute levels on the
    # same day, so no split adjustment is needed. A stock that traded at the
    # open and delisted that day still earns its delisting return, so it is
    # compounded in. Without an opening print there is no open to buy at, so
    # ret_C is NaN (the position contributes zero, per the pre-registration);
    # that includes delisting-only rows, whose DLRET runs from the previous
    # close and was never available to an open-to-close holder.
    dl = df["dlret"].fillna(0.0)
    c_ok = (df["openprc"] > 0) & (df["prc"] > 0)
    intraday = (df["prc"] / df["openprc"] - 1.0).where(c_ok)
    df["ret_C"] = ((1.0 + intraday) * (1.0 + dl) - 1.0).where(c_ok)
    df["c_valid"] = c_ok
    # previous close's market cap: weights for the value-weighted open-to-close
    # market used in the convention-C attribution
    df["mktcap_lag"] = g["mktcap"].shift(1)

    df["beta"] = dimson_beta(
        df, window=beta_window, min_obs=beta_min_obs, shrink_weight=shrink_weight
    )

    # delisting-only rows have no quote data; a missing flag must not read as True
    df["quote_only"] = df["quote_only"].astype("boolean").fillna(False).astype(bool)
    df = attach_quote_columns(df)
    df = midpoint_returns(df, out_col="mid_ret")
    df["rel_spread_lag"] = df.groupby("permno", sort=False)["rel_spread"].shift(1)

    cols = [
        "date", "permno", "ret_A", "ret_B", "ret_C", "c_valid", "retx",
        "mid_ret", "mid_ret_valid", "own_lag", "beta", "rel_spread_lag", "mktcap_lag",
    ]
    return df[cols]


def calendar_lagged_signals(
    shocks_frame: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """The leader's decomposed move, lagged one TRADING SESSION on a calendar.

    `shocks.regression.lag_leader_components` shifts by row within each
    industry's shock series and keeps the row's own date. A day-t value
    therefore exists only if a shock row dated t exists, which requires the
    leader's return at the close of t (look-ahead in industry membership), and
    a missing day t-1 makes day t carry day t-2's value.

    Here each shock row dated d is re-dated to the next session after d on
    `calendar`, so the signal on day t exists if and only if the leader's
    decomposition exists for day t-1, and nothing from day t is consulted.

    Returns date, ff49, common_lag, shock_lag, leader_ret_lag,
    conditional_signal (= common_lag - shock_lag).
    """
    cal = pd.DatetimeIndex(np.sort(pd.DatetimeIndex(calendar).unique()))
    src = shocks_frame[["date", "ff49", "common", "shock", "leader_ret"]].copy()
    pos = cal.searchsorted(pd.DatetimeIndex(src["date"]), side="right")
    keep = pos < len(cal)
    src = src[keep].copy()
    src["date"] = cal[pos[keep]]
    out = src.rename(columns={
        "common": "common_lag", "shock": "shock_lag", "leader_ret": "leader_ret_lag",
    })
    out["conditional_signal"] = out["common_lag"] - out["shock_lag"]
    return out.reset_index(drop=True)
