"""
Long/short portfolio construction for Part 4.

Two constructions, because the signals live at two levels
----------------------------------------------------------
`common_lag`, `shock_lag`, `leader_ret_lag` and `conditional_signal` are merged
onto followers on (date, ff49), so every follower in an industry carries the
SAME value. An FF49-industry-neutral book on those signals is identically zero,
and the true breadth is the ~48 industries, not the ~1,750 followers. They get
an industry book:

    1. rank the industries by the signal each day and demean the ranks;
    2. remove, across industries, the exposure to a constant (dollar), the
       industry's average follower beta, and the industry's average follower
       own-lag return. The last one mirrors Part 2's regression, which controls
       for the follower's own lagged return, so the book isolates what the
       LEADER adds beyond the followers' own move;
    3. scale gross exposure to 1 and split each industry's weight equally over
       its followers that day.

`reversal_signal` (-own_lag) varies within industry and gets a stock book that
is dollar-, FF49-industry- and beta-neutral.

Both use closed-form projections (Frisch-Waugh within date), so exposures are
zero to floating-point precision and no numerical optimiser is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _demean(frame: pd.DataFrame, col: str, by: list[str]) -> pd.Series:
    return frame[col] - frame.groupby(by, sort=False)[col].transform("mean")


def industry_book(
    panel: pd.DataFrame,
    signal_col: str,
    beta_col: str = "beta",
    own_col: str = "own_lag",
    date_col: str = "date",
    ind_col: str = "ff49",
    min_industries: int = 10,
) -> pd.Series:
    """Stock weights for an industry-level long/short book.

    Returns a Series aligned to `panel.index`: the weight of each follower-day,
    0 where the row is not in the book. On each date the weights sum to 0, have
    zero exposure to beta and own-lag (with a missing stock value imputed at
    its industry's mean, the same value the construction uses), and sum to 1 in
    absolute value.
    """
    rows = panel[[date_col, ind_col, signal_col, beta_col, own_col]].copy()
    rows = rows[rows[signal_col].notna() & rows[ind_col].notna()]

    ind = rows.groupby([date_col, ind_col], sort=False).agg(
        s=(signal_col, "first"),
        b=(beta_col, "mean"),
        o=(own_col, "mean"),
        n=(signal_col, "size"),
    ).reset_index()
    ind = ind[ind["b"].notna() & ind["o"].notna()]
    ind = ind[ind.groupby(date_col)[ind_col].transform("size") >= min_industries]

    ind["z"] = ind.groupby(date_col)["s"].rank(method="average")
    for c in ("z", "b", "o"):
        ind[c + "d"] = _demean(ind, c, [date_col])

    sums = ind.assign(
        bb=ind["bd"] ** 2, oo=ind["od"] ** 2, bo=ind["bd"] * ind["od"],
        zb=ind["zd"] * ind["bd"], zo=ind["zd"] * ind["od"],
    ).groupby(date_col)[["bb", "oo", "bo", "zb", "zo"]].transform("sum")
    det = sums["bb"] * sums["oo"] - sums["bo"] ** 2
    good = det.abs() > 1e-14
    cb = ((sums["zb"] * sums["oo"] - sums["zo"] * sums["bo"]) / det).where(good)
    co = ((sums["zo"] * sums["bb"] - sums["zb"] * sums["bo"]) / det).where(good)
    w = ind["zd"] - cb * ind["bd"] - co * ind["od"]

    gross = w.abs().groupby(ind[date_col]).transform("sum")
    ind["W"] = (w / gross).where(gross > 1e-14)
    ind = ind[ind["W"].notna()]
    ind["w_stock"] = ind["W"] / ind["n"]

    out = panel[[date_col, ind_col]].merge(
        ind[[date_col, ind_col, "w_stock"]], on=[date_col, ind_col], how="left"
    )["w_stock"]
    out.index = panel.index
    # rows without a signal are not in the book even if their industry is
    out = out.where(panel[signal_col].notna(), 0.0)
    return out.fillna(0.0).rename(f"w_{signal_col}")


def stock_book(
    panel: pd.DataFrame,
    signal_col: str,
    beta_col: str = "beta",
    date_col: str = "date",
    ind_col: str = "ff49",
    min_names: int = 50,
) -> pd.Series:
    """Stock weights for a dollar-, industry- and beta-neutral long/short book.

    Ranks the signal across all names each day, demeans the ranks within
    (date, industry), then removes the within-industry-demeaned beta. Rows
    without a signal or a beta get weight 0. Gross exposure is 1 per date.
    """
    ok = panel[signal_col].notna() & panel[beta_col].notna() & panel[ind_col].notna()
    rows = panel.loc[ok, [date_col, ind_col, signal_col, beta_col]].copy()
    rows = rows[rows.groupby(date_col)[signal_col].transform("size") >= min_names]

    rows["z"] = rows.groupby(date_col)[signal_col].rank(method="average")
    rows["zd"] = _demean(rows, "z", [date_col, ind_col])
    rows["bd"] = _demean(rows, beta_col, [date_col, ind_col])
    sb = rows.assign(zb=rows["zd"] * rows["bd"], bb=rows["bd"] ** 2).groupby(
        date_col
    )[["zb", "bb"]].transform("sum")
    coef = (sb["zb"] / sb["bb"]).where(sb["bb"] > 1e-14, 0.0)
    w = rows["zd"] - coef * rows["bd"]
    gross = w.abs().groupby(rows[date_col]).transform("sum")
    w = (w / gross).where(gross > 1e-14, 0.0)

    out = pd.Series(0.0, index=panel.index, name=f"w_{signal_col}")
    out.loc[w.index] = w.to_numpy()
    return out


def book_returns(
    panel: pd.DataFrame, weight_col: str, ret_col: str, date_col: str = "date"
) -> pd.DataFrame:
    """Daily book return sum(w * r) and the share of gross with a valid return.

    A held stock with a missing return contributes zero (pre-registered); the
    `coverage` column reports how much of the gross that affects.
    """
    w = panel[weight_col]
    r = panel[ret_col]
    held = w != 0
    frame = pd.DataFrame({
        date_col: panel[date_col],
        "pnl": (w * r).where(r.notna(), 0.0),
        "gross": w.abs(),
        "gross_valid": w.abs().where(r.notna(), 0.0),
    })[held]
    agg = frame.groupby(date_col).sum()
    coverage = agg["gross_valid"] / agg["gross"]
    # a day on which no held stock has a return is missing, not a 0% day
    return pd.DataFrame({
        "ret": agg["pnl"].where(coverage > 0),
        "coverage": coverage,
    })


def turnover(
    panel: pd.DataFrame, weight_col: str, date_col: str = "date", id_col: str = "permno"
) -> pd.Series:
    """Daily turnover sum_i |w_{i,t} - w_{i,t-1}| over the union of names.

    Ignores intraday weight drift (pre-trade weights are taken as yesterday's
    targets), the usual simplification for daily-rebalanced books.
    """
    dates = np.sort(panel[date_col].unique())
    pos = pd.Series(np.arange(len(dates)), index=dates)
    cur = panel.loc[panel[weight_col] != 0, [date_col, id_col, weight_col]].copy()
    cur["k"] = pos.reindex(cur[date_col]).to_numpy()
    prev = cur[["k", id_col, weight_col]].rename(columns={weight_col: "w_prev"})
    prev["k"] = prev["k"] + 1
    both = cur[["k", id_col, weight_col]].merge(prev, on=["k", id_col], how="outer")
    both = both[both["k"] < len(dates)]
    both["dw"] = (both[weight_col].fillna(0.0) - both["w_prev"].fillna(0.0)).abs()
    to = both.groupby("k")["dw"].sum()
    to.index = dates[to.index.to_numpy()]
    return to.rename("turnover")


def vol_scaled(
    returns: pd.Series, target: float = 0.10, window: int = 63, cap: float = 3.0
) -> pd.Series:
    """Scale a gross-1 return series to a target volatility using only past data.

    Scale on day t = target / (annualised std of the previous `window` days),
    capped at `cap`. The first `window` days have no scale and are NaN.
    """
    trailing = returns.rolling(window).std().shift(1) * np.sqrt(252)
    scale = (target / trailing).clip(upper=cap)
    return (returns * scale).rename(returns.name)
