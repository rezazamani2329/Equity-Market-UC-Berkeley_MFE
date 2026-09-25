"""
Information Coefficient (IC) and Rank IC: does the signal actually rank
stocks the way subsequent returns will.

Why IC, not just a single pooled correlation
----------------------------------------------
A pooled correlation of signal-today against return-tomorrow, computed
across the whole panel at once, is dominated by whichever CROSS-SECTION
happens to be biggest and by any date with unusually high dispersion in
returns.  It also silently mixes market-wide moves into what is supposed to
be a purely cross-sectional bet: on a day the whole market drops 3%, every
stock's forward return is negative regardless of the signal, and a pooled
correlation partly picks that up as "signal quality" when it is really beta.

The standard fix (Grinold & Kahn) is to compute the correlation WITHIN each
date's cross-section — the signal ranks stocks against each other on that
one day, and today's IC only asks whether that day's ranking then predicted
that day's relative winners — and then treat the resulting time series of
daily ICs as the object of interest: its mean is the average cross-sectional
predictive power, its t-stat tests whether that mean is distinguishable from
zero, and `mean / std` is the Information Ratio (IR) of the signal itself,
independent of any portfolio construction choice.

Rank IC (Spearman) vs raw IC (Pearson)
---------------------------------------
This project's signal is built from returns and a shock decomposition, which
are heavy-tailed — a handful of extreme days would dominate a Pearson
correlation and make single outlier stocks look like most of the story.
Spearman correlates RANKS, so it is immune to the magnitude of an outlier
(only its rank matters) and is the standard choice for signal validation in
the empirical asset-pricing literature this project sits in.  Pearson IC is
still exposed via `method="pearson"` for anyone who wants the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ICResult:
    """One IC time series and its summary statistics.

    Attributes:
        by_period:    Series indexed by date, one IC value per cross-section.
                       Dates with fewer than `min_obs` names, or with zero
                       variance in either the signal or the return that day,
                       are simply absent (not NaN-filled) so `mean()` etc.
                       never need an explicit dropna.
        mean_ic:      mean of `by_period`.
        std_ic:       (sample, ddof=1) std of `by_period`.
        t_stat:       mean_ic / (std_ic / sqrt(n_periods)) — tests whether the
                       average daily IC is distinguishable from zero, treating
                       each day as one independent observation.  That
                       independence assumption is generous for a signal with
                       autocorrelated exposures; treat this as a first pass,
                       not a Newey-West-robust test.
        n_periods:    number of dates actually contributing to `by_period`.
        pct_positive: share of periods with IC > 0 — a distribution-free
                       complement to the t-stat that is easy to sanity check
                       by eye against `by_period`.
        ir:           mean_ic / std_ic (the "Information Ratio" of the signal
                       itself, Grinold & Kahn's IC-IR) — NaN if std_ic is 0
                       or there are fewer than 2 periods.
    """

    by_period: pd.Series
    mean_ic: float
    std_ic: float
    t_stat: float
    n_periods: int
    pct_positive: float
    ir: float


def _period_corr(
    frame: pd.DataFrame, signal_col: str, ret_col: str, method: str, min_obs: int
) -> float:
    """Correlation for one date's cross-section, or NaN if it doesn't qualify."""
    sub = frame[[signal_col, ret_col]].dropna()
    if len(sub) < min_obs:
        return np.nan
    # A signal or return column that is constant that day (e.g. everyone tied)
    # has undefined correlation; pandas returns NaN for this already, but the
    # explicit check keeps the reason legible rather than relying on the
    # library's silent NaN.
    if sub[signal_col].nunique() < 2 or sub[ret_col].nunique() < 2:
        return np.nan
    return float(sub[signal_col].corr(sub[ret_col], method=method))


def compute_ic(
    panel: pd.DataFrame,
    signal_col: str,
    ret_col: str,
    date_col: str = "date",
    method: str = "spearman",
    min_obs: int = 10,
) -> ICResult:
    """Cross-sectional IC per date, then summarised across dates.

    Args:
        panel: long frame with at least `date_col`, `signal_col`, `ret_col`.
            One row per (date, permno); the signal and the return it is
            meant to predict must already be aligned on the SAME row (i.e.
            `ret_col` should already be a forward return relative to the
            information used to build the signal — this function does no
            shifting itself).
        signal_col: the column ranking stocks.
        ret_col: the column of subsequent returns being predicted.
        date_col: the column defining each cross-section.
        method: "spearman" (rank IC, the default) or "pearson" (raw IC).
        min_obs: minimum number of non-missing (signal, return) pairs a date
            needs to contribute an IC.  Below this the correlation is too
            noisy to be worth the day (10 is a conservative floor; industries
            in this project's universe can be genuinely this thin).

    Returns:
        ICResult.  On a panel with zero qualifying dates, `by_period` is an
        empty Series and every scalar is NaN — this is a valid, checkable
        outcome (e.g. `signal_col` and the panel don't actually overlap on
        dates), not an error, so no exception is raised.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"compute_ic: method must be 'spearman' or 'pearson', got {method!r}")

    grouped = panel.groupby(date_col, sort=True)
    by_period = grouped.apply(
        lambda g: _period_corr(g, signal_col, ret_col, method, min_obs),
        include_groups=False,
    )
    by_period = by_period.dropna()
    by_period.name = "ic"

    n = len(by_period)
    if n == 0:
        return ICResult(
            by_period=by_period, mean_ic=np.nan, std_ic=np.nan, t_stat=np.nan,
            n_periods=0, pct_positive=np.nan, ir=np.nan,
        )

    mean_ic = float(by_period.mean())
    std_ic = float(by_period.std(ddof=1)) if n > 1 else np.nan
    t_stat = (
        mean_ic / (std_ic / np.sqrt(n)) if n > 1 and std_ic > 0 else np.nan
    )
    pct_positive = float((by_period > 0).mean())
    ir = mean_ic / std_ic if n > 1 and std_ic > 0 else np.nan

    return ICResult(
        by_period=by_period, mean_ic=mean_ic, std_ic=std_ic, t_stat=t_stat,
        n_periods=n, pct_positive=pct_positive, ir=ir,
    )


def horizon_ic(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    signal_col: str,
    horizons: tuple[int, ...] | list[int] = (1, 5, 10, 20),
    date_col: str = "date",
    permno_col: str = "permno",
    method: str = "spearman",
    min_obs: int = 10,
) -> pd.DataFrame:
    """IC decay: how a signal's cross-sectional predictive power evolves
    across forward horizons.

    Args:
        signal: frame with `date_col`, `permno_col`, `signal_col` — one
            signal value per stock-day, known as of that day's close.
        fwd_returns: output of `forward_returns.forward_returns()` — must
            share `date_col`/`permno_col` and carry one `fwd_ret_{h}d` column
            per horizon in `horizons`.
        signal_col: which column in `signal` to test.
        horizons: which `fwd_ret_{h}d` columns to test against; every value
            here must have a matching column in `fwd_returns` or this raises
            KeyError with the horizon named, rather than silently skipping it.
        method, min_obs: passed through to `compute_ic` for each horizon.

    Returns:
        One row per horizon: horizon, mean_ic, std_ic, t_stat, n_periods,
        pct_positive, ir.  A short IC that decays toward zero by the longest
        horizon is the textbook signature of a signal that is capturing a
        real but transient effect (e.g. price impact reverting) rather than
        an artefact that would show up flat or growing.

    Why a left merge, not a join on the full fwd_returns frame: the signal
    frame is very likely a strict subset of dates (e.g. it needs a formation
    window before it exists), and merging fwd_returns TO signal rather than
    the reverse keeps that restriction explicit rather than silently
    including dates the signal doesn't cover.
    """
    merged = signal[[date_col, permno_col, signal_col]].merge(
        fwd_returns, on=[date_col, permno_col], how="left"
    )

    rows = []
    for h in horizons:
        col = f"fwd_ret_{h}d"
        if col not in merged.columns:
            raise KeyError(
                f"horizon_ic: horizon {h} requested but column {col!r} is not "
                f"in fwd_returns — pass a `fwd_returns` frame built with this "
                f"horizon included."
            )
        result = compute_ic(
            merged, signal_col=signal_col, ret_col=col, date_col=date_col,
            method=method, min_obs=min_obs,
        )
        rows.append(
            {
                "horizon": h,
                "mean_ic": result.mean_ic,
                "std_ic": result.std_ic,
                "t_stat": result.t_stat,
                "n_periods": result.n_periods,
                "pct_positive": result.pct_positive,
                "ir": result.ir,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")
