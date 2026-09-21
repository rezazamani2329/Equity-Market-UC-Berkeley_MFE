"""
The baseline lead-lag test: does yesterday's leader return predict today's
follower return?

This is the last thing part 1 owes the project, and it is deliberately the
*unconditional* version — no shock decomposition, no signal construction, no
portfolio.  Its job is to establish that the effect the rest of the project
conditions on exists at all in this sample, and to give parts 2-5 a number to
beat.  If the baseline coefficient is indistinguishable from zero, that is
itself the finding, and the report's executive summary should say so.

The regression
--------------
For follower i in industry j, on day t, with lag k:

    r_{i,t} = a + b · r_{L(j),t-k} + c · r_{i,t-k} + d · mkt_t + e_{i,t}

    b   the lead-lag coefficient — the object of interest.
    c   the follower's OWN lagged return.  Without it, b picks up ordinary
        short-horizon reversal: the leader's lagged return is correlated with
        the follower's lagged return (they share an industry), so a pure
        own-reversal effect would show up as a negative b.  Controlling for
        the own lag is what makes b specifically about the LEADER.
    d   the contemporaneous market return.  Followers and leaders share market
        beta; without this control, b partly measures the market's own
        autocorrelation rather than anything industry-specific.

Standard errors
---------------
Two estimators, reported side by side, because they fail in different ways.

*Pooled OLS clustered by date.*  Thousands of followers on the same day share
the market, their industries, and the day's news.  Treating those as
independent observations understates the standard error by roughly the square
root of the average cross-section — a factor of 30 or more.  Clustering by
date is the minimum defensible correction, and it is the one that usually
turns a "t = 40" into something honest.

*Fama-MacBeth.*  Run the cross-section separately each day, then test the mean
of the daily coefficients with Newey-West standard errors.  This handles the
cross-sectional dependence by construction (each day contributes one number)
and the Newey-West lags handle serial correlation in the coefficient series.
It is the convention in this literature, so the report should lead with it.

They should broadly agree.  When they do not, the pooled estimate is usually
being driven by a few enormous cross-sections and the Fama-MacBeth one is the
one to trust.

The horizon profile
-------------------
`horizon_profile` runs the same regression for k = 1 .. K and returns b(k).
This plot is the project's central exhibit in miniature: the hypothesis is
that b(k) > 0 at short k (information diffusing) and turns negative at longer
k for the leader-specific component (reversal).  The unconditional profile
here should show the continuation part; getting the reversal part to appear is
what part 2's decomposition is for.

One overlapping-window warning.  With k > 1 and daily data, consecutive
observations of the same follower share k-1 days of the predictor window when
returns are cumulated.  This module does NOT cumulate — `r_{L,t-k}` is the
leader's single-day return k days ago — so the observations are not
overlapping and the Newey-West lag count only needs to cover genuine serial
correlation, not induced overlap.  If part 3 cumulates the signal over a
window, that changes and the lag count has to grow with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class RegressionResult:
    """One estimate of the lead-lag coefficient, with both standard errors.

    Attributes:
        lag:            k, in trading days.
        n_obs:          follower-days in the regression.
        n_days:         distinct trading days (= Fama-MacBeth sample size).
        beta_pooled:    b from pooled OLS.
        t_pooled:       its t-statistic, clustered by date.
        beta_fm:        mean of the daily cross-sectional coefficients.
        t_fm:           its t-statistic, Newey-West.
        own_lag_pooled: c from the pooled regression — reported because a
                        large negative c is the ordinary reversal effect, and
                        the reader should see how much of it there is.
    """

    lag: int
    n_obs: int
    n_days: int
    beta_pooled: float
    t_pooled: float
    beta_fm: float
    t_fm: float
    own_lag_pooled: float

    def as_row(self) -> dict[str, float | int]:
        """Flat dict, for building a results table across lags."""
        return {
            "lag": self.lag,
            "n_obs": self.n_obs,
            "n_days": self.n_days,
            "beta_pooled": self.beta_pooled,
            "t_pooled": self.t_pooled,
            "beta_fm": self.beta_fm,
            "t_fm": self.t_fm,
            "own_lag_pooled": self.own_lag_pooled,
        }


def build_lags(
    panel: pd.DataFrame, lag: int = 1, ret_col: str = "ret"
) -> pd.DataFrame:
    """Add `leader_lag` and `own_lag` columns, shifted `lag` trading days.

    The shift is WITHIN each follower (`groupby("permno").shift`), so a
    stock's first `lag` observations get NaN rather than borrowing another
    stock's history.

    A subtlety worth stating, because it is the easiest way to get this wrong:
    the shift moves by ROWS, not by calendar days.  That is correct only if
    the panel has one row per trading day per stock with no gaps — which is
    exactly what a missing daily return would break, since `DailyReturns`
    keeps those rows with `ret = NaN` rather than dropping them.  So: shift
    first, drop NaNs second.  Dropping first and shifting second would line up
    day t against a day that is not t-k, silently, for precisely the illiquid
    stocks whose returns are most autocorrelated.
    """
    df = panel.sort_values(["permno", "date"]).copy()
    g = df.groupby("permno", sort=False)
    df["leader_lag"] = g["leader_ret"].shift(lag)
    df["own_lag"] = g[ret_col].shift(lag)
    return df


def pooled_regression(
    panel: pd.DataFrame, ret_col: str = "ret", market_col: str | None = "mktrf"
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Pooled OLS of the follower return on the lagged leader return.

    Requires `leader_lag` and `own_lag` — run `build_lags` first.  Standard
    errors are clustered by `date`.  The market control is included when
    `market_col` is present in the frame; pass None to omit it.
    """
    cols = [ret_col, "leader_lag", "own_lag", "date"]
    if market_col and market_col in panel.columns:
        cols.append(market_col)
    df = panel[cols].dropna()

    exog_cols = ["leader_lag", "own_lag"] + (
        [market_col] if market_col and market_col in df.columns else []
    )
    x = sm.add_constant(df[exog_cols])
    return sm.OLS(df[ret_col], x).fit(
        cov_type="cluster", cov_kwds={"groups": df["date"]}
    )


def fama_macbeth(
    panel: pd.DataFrame,
    ret_col: str = "ret",
    min_cross_section: int = 20,
    nw_lags: int | None = None,
) -> tuple[float, float, pd.Series]:
    """Daily cross-sectional regressions, then a Newey-West test of the mean.

    Args:
        panel: must carry `leader_lag`, `own_lag`, `date`.
        ret_col: the follower return.
        min_cross_section: days with fewer usable followers are skipped — a
            two-stock cross-section produces a coefficient with no information
            and enormous variance, which would then dominate the mean.
        nw_lags: Newey-West lag truncation for the second stage.  Default is
            the usual rule of thumb, ceil(4 · (T/100)^(2/9)).

    Returns:
        (mean coefficient, t-statistic, the daily coefficient series).  The
        series is returned so the report can plot it: a mean that comes
        entirely from 2008 is a different claim from one that is stable.
    """
    df = panel[[ret_col, "leader_lag", "own_lag", "date"]].dropna()

    coefs: dict[pd.Timestamp, float] = {}
    for day, chunk in df.groupby("date", sort=True):
        if len(chunk) < min_cross_section:
            continue
        x = sm.add_constant(chunk[["leader_lag", "own_lag"]])
        # A day on which every follower shares the same leader return has a
        # collinear regressor (leader_lag is constant); `np.linalg.LinAlgError`
        # or a singular fit is the symptom.  Skip rather than crash: it
        # happens when the universe collapses to a single industry.
        if x["leader_lag"].nunique() < 2:
            continue
        try:
            fit = sm.OLS(chunk[ret_col], x).fit()
        except np.linalg.LinAlgError:
            continue
        coefs[day] = float(fit.params["leader_lag"])

    series = pd.Series(coefs).sort_index()
    if series.empty:
        return float("nan"), float("nan"), series

    if nw_lags is None:
        nw_lags = int(np.ceil(4 * (len(series) / 100) ** (2 / 9)))
    second = sm.OLS(series.to_numpy(), np.ones(len(series))).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags}
    )
    return float(second.params[0]), float(second.tvalues[0]), series


def baseline_test(
    panel: pd.DataFrame,
    lag: int = 1,
    ret_col: str = "ret",
    market_col: str | None = "mktrf",
) -> RegressionResult:
    """Run both estimators at one lag and package the numbers.

    `panel` is `leaders.follower_panel` output, optionally merged with the
    daily factor file so `mktrf` is available.
    """
    lagged = build_lags(panel, lag=lag, ret_col=ret_col)
    pooled = pooled_regression(lagged, ret_col=ret_col, market_col=market_col)
    beta_fm, t_fm, _ = fama_macbeth(lagged, ret_col=ret_col)

    used = lagged[[ret_col, "leader_lag", "own_lag", "date"]].dropna()
    return RegressionResult(
        lag=lag,
        n_obs=len(used),
        n_days=int(used["date"].nunique()),
        beta_pooled=float(pooled.params["leader_lag"]),
        t_pooled=float(pooled.tvalues["leader_lag"]),
        beta_fm=beta_fm,
        t_fm=t_fm,
        own_lag_pooled=float(pooled.params["own_lag"]),
    )


def horizon_profile(
    panel: pd.DataFrame,
    lags: range | list[int] = range(1, 11),
    ret_col: str = "ret",
    market_col: str | None = "mktrf",
    verbose: bool = True,
) -> pd.DataFrame:
    """b(k) for each k in `lags` — the project's central diagnostic plot.

    Returns one row per lag with both estimates and both t-statistics.
    """
    rows = []
    for k in lags:
        if verbose:
            print(f"baseline lead-lag: lag {k} ...", flush=True)
        rows.append(baseline_test(panel, lag=k, ret_col=ret_col, market_col=market_col).as_row())
    return pd.DataFrame(rows).set_index("lag")


def quintile_sort(
    panel: pd.DataFrame, lag: int = 1, ret_col: str = "ret", n_bins: int = 5
) -> pd.DataFrame:
    """Non-parametric check: sort follower-days by the lagged leader return.

    Each day, followers are placed into `n_bins` bins by their industry
    leader's return `lag` days ago, and the bins' equal-weighted follower
    returns are averaged over the sample.

    Why bother when the regression already answers the question.  A regression
    coefficient is a single number that assumes linearity and is sensitive to
    the tails; a monotone bin profile is neither.  If the regression says
    b > 0 but the bins are flat except for the extreme one, the effect is a
    handful of large leader moves rather than a general relationship — which
    matters a great deal for whether the eventual strategy is tradeable.

    Returns one row per bin: mean follower return, t-statistic of the mean
    across days, and the observation count.  The top-minus-bottom spread is
    appended as the last row.

    Binning happens at the INDUSTRY level, not the follower level.  Every
    follower in an industry sees the same leader return, so binning
    follower-days directly would split one industry's followers across
    adjacent bins purely by how ties were broken.  Ranking the industries and
    letting each follower inherit its industry's bin is what the signal
    actually says.  The consequence is that `n_bins` cannot exceed the number
    of industries with a leader on a given day — with Fama-French 49 that is
    never binding, but it is on a narrow universe, and the error below says so
    rather than returning an empty table.
    """
    lagged = build_lags(panel, lag=lag, ret_col=ret_col)
    df = lagged[[ret_col, "leader_lag", "date", "ff49"]].dropna()

    # One row per (date, industry): the signal's own cross-section.
    industries = df[["date", "ff49", "leader_lag"]].drop_duplicates()
    # Bin WITHIN each day: a cross-sectional rank is what a tradeable signal
    # would use, and it makes the bins immune to a market-wide day.
    industries["bin"] = industries.groupby("date")["leader_lag"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_bins, labels=False) + 1
        if s.nunique() >= n_bins else np.nan
    )
    industries = industries.dropna(subset=["bin"])
    if industries.empty:
        n_max = int(df.groupby("date")["ff49"].nunique().max()) if len(df) else 0
        raise ValueError(
            f"quintile_sort: no day had {n_bins} distinct leader returns to sort "
            f"on (the busiest day had {n_max} industries with a leader). "
            f"Pass a smaller n_bins, or widen the universe."
        )
    df = df.merge(industries[["date", "ff49", "bin"]], on=["date", "ff49"], how="inner")
    df["bin"] = df["bin"].astype(int)

    # Daily equal-weighted return of each bin, then a t-test across days.
    daily_bins = df.groupby(["date", "bin"])[ret_col].mean().unstack("bin")

    rows = []
    for b in sorted(daily_bins.columns):
        s = daily_bins[b].dropna()
        rows.append(
            {
                "bin": int(b),
                "mean_ret": float(s.mean()),
                "t_stat": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 else np.nan,
                "n_days": int(len(s)),
            }
        )

    lo, hi = min(daily_bins.columns), max(daily_bins.columns)
    spread = (daily_bins[hi] - daily_bins[lo]).dropna()
    rows.append(
        {
            "bin": -1,  # the top-minus-bottom row
            "mean_ret": float(spread.mean()),
            "t_stat": float(spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread))))
            if len(spread) > 1 else np.nan,
            "n_days": int(len(spread)),
        }
    )
    out = pd.DataFrame(rows).set_index("bin")
    out.attrs["note"] = f"bin -1 is bin {hi} minus bin {lo}"
    return out
