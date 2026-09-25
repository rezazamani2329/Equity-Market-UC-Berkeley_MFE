"""
Quantile portfolio sorts: turn a signal into a tradeable long-short spread
and test whether that spread is reliably non-zero.

Why this on top of IC
----------------------
IC answers "does the signal rank stocks correctly, on average, across the
whole cross-section."  A quantile sort answers the question a trader (and
the rest of this project's team, when it comes to the transaction-cost and
robustness sections) actually needs: if you bought the top bucket and sold
the bottom bucket every rebalance, using only ex-ante information, what
return would that have earned, and is it distinguishable from zero.  IC can
be modestly positive while the portfolio spread is noisy or economically
tiny (e.g. concentrated in a few illiquid names), so the two checks are
complements, not substitutes — the report should show both.

Per-date sorting, same reasoning as `ic.py`
---------------------------------------------
Buckets are formed WITHIN each date's cross-section, exactly as IC is
computed within each date, and for the same reason: sorting the pooled panel
would let dates with unusually wide return dispersion dominate, and would
let market-wide moves masquerade as bucket separation. `pd.qcut` per date is
used rather than fixed global cutpoints so the buckets stay balanced even as
the number of eligible names varies across the sample (this project's
universe grows from a few hundred names in 1996 to several thousand later).

Equal- vs value-weighting
---------------------------
Equal-weighting is the default because it is what IC (a rank statistic) is
implicitly testing — it treats a $50M name and a $50B name as equally
informative about the SIGN of the relationship. Value-weighting
(`weight_col="mktcap_lag"` or similar) answers the economically different
question of what a capital-weighted implementation would have earned, which
this project's illiquid, small-cap-tilted universe (see `universe.py`'s
NYSE-20th-percentile floor) can move quite a bit relative to equal-weight —
worth running both and comparing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileResult:
    """Per-date bucket returns and the long-short spread built from them.

    Attributes:
        by_period:   long frame, one row per (date, quantile): date,
                     quantile (1 = lowest signal .. n_quantiles = highest),
                     ret (equal- or value-weighted bucket return), n_stocks.
        summary:     one row per quantile, averaged across dates: mean_ret,
                     std_ret, t_stat, n_periods, avg_n_stocks. This is the
                     table that shows whether the sort is MONOTONIC (real
                     signal) or lumpy/non-monotonic (probably noise or a
                     signal that only separates the extremes).
        spread:      Series indexed by date — (top bucket - bottom bucket)
                     return each period, i.e. the long-short portfolio's
                     realised return series.
        spread_mean: mean of `spread`.
        spread_t_stat: spread_mean / (spread.std(ddof=1) / sqrt(n)) — the
                     headline statistic: is the long-short strategy's average
                     return distinguishable from zero. Like `ICResult.t_stat`,
                     this treats periods as independent, which is optimistic
                     for overlapping-horizon returns; a Newey-West correction
                     belongs in part 5's robustness pass, not here.
    """

    by_period: pd.DataFrame
    summary: pd.DataFrame
    spread: pd.Series
    spread_mean: float
    spread_t_stat: float


def _bucket_one_period(
    frame: pd.DataFrame,
    signal_col: str,
    ret_col: str,
    weight_col: str | None,
    n_quantiles: int,
) -> pd.DataFrame | None:
    """Assign quantiles and compute bucket returns for one date's cross-section.

    Returns None (contributing nothing) when the cross-section is too small
    or too tied to form `n_quantiles` distinct, non-empty buckets — silently
    dropping such a date is preferable to raising, since thin early-sample
    industries and thin one-off dates are expected, not a bug.
    """
    sub = frame[[signal_col, ret_col] + ([weight_col] if weight_col else [])].dropna()
    if len(sub) < n_quantiles:
        return None
    if sub[signal_col].nunique() < n_quantiles:
        return None

    try:
        sub = sub.copy()
        sub["_q"] = pd.qcut(sub[signal_col], n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return None
    # `duplicates="drop"` can silently produce fewer than n_quantiles bins
    # when the signal has heavy ties near a cutpoint; that period is then
    # not directly comparable to a full n_quantiles period, so it is dropped
    # rather than mislabelled.
    if sub["_q"].nunique() < n_quantiles:
        return None
    sub["_q"] = sub["_q"] + 1  # 1-indexed: 1 = lowest signal, n = highest

    if weight_col:
        def _wavg(g: pd.DataFrame) -> float:
            w = g[weight_col].to_numpy()
            if w.sum() <= 0:
                return float(np.nan)
            return float(np.average(g[ret_col].to_numpy(), weights=w))
        bucket_ret = sub.groupby("_q").apply(_wavg, include_groups=False)
    else:
        bucket_ret = sub.groupby("_q")[ret_col].mean()

    n_stocks = sub.groupby("_q")[ret_col].size()
    out = pd.DataFrame({"quantile": bucket_ret.index, "ret": bucket_ret.values})
    out["n_stocks"] = n_stocks.reindex(out["quantile"]).to_numpy()
    return out


def quantile_portfolios(
    panel: pd.DataFrame,
    signal_col: str,
    ret_col: str,
    n_quantiles: int = 5,
    weight_col: str | None = None,
    date_col: str = "date",
    long_quantile: int | None = None,
    short_quantile: int | None = None,
) -> QuantileResult:
    """Per-date quantile sort on `signal_col`, and the resulting long-short
    spread's return series and t-stat.

    Args:
        panel: long frame with `date_col`, `signal_col`, `ret_col`, and
            `weight_col` if given. As in `compute_ic`, `ret_col` must already
            be the forward return aligned to this row's signal.
        signal_col, ret_col: as above.
        n_quantiles: number of buckets per date (5 = quintiles by default).
        weight_col: None for equal weighting; a column name (e.g.
            `mktcap_lag`, the previous close's market cap) for value
            weighting within each bucket.
        date_col: the column defining each cross-section.
        long_quantile, short_quantile: which buckets form the spread.
            Default is the top vs bottom quantile (`n_quantiles` vs `1`) —
            the standard construction. Pass explicit values to test, e.g.,
            quantile 4 vs 2 instead.

    Returns:
        QuantileResult. Dates that don't support `n_quantiles` distinct
        buckets (see `_bucket_one_period`) simply don't appear in
        `by_period` — this is expected for early-sample, thin-industry
        dates and is not treated as an error.
    """
    if n_quantiles < 2:
        raise ValueError(f"quantile_portfolios: n_quantiles must be >= 2, got {n_quantiles}")
    long_q = long_quantile if long_quantile is not None else n_quantiles
    short_q = short_quantile if short_quantile is not None else 1
    if long_q == short_q:
        raise ValueError(
            f"quantile_portfolios: long_quantile ({long_q}) and short_quantile "
            f"({short_q}) must differ"
        )

    grouped = panel.groupby(date_col, sort=True)
    period_frames = []
    for date, g in grouped:
        bucketed = _bucket_one_period(g, signal_col, ret_col, weight_col, n_quantiles)
        if bucketed is None:
            continue
        bucketed.insert(0, date_col, date)
        period_frames.append(bucketed)

    if not period_frames:
        empty_by_period = pd.DataFrame(columns=[date_col, "quantile", "ret", "n_stocks"])
        empty_summary = pd.DataFrame(
            columns=["mean_ret", "std_ret", "t_stat", "n_periods", "avg_n_stocks"]
        )
        empty_summary.index.name = "quantile"
        empty_spread = pd.Series(dtype=float, name="spread")
        return QuantileResult(
            by_period=empty_by_period, summary=empty_summary, spread=empty_spread,
            spread_mean=np.nan, spread_t_stat=np.nan,
        )

    by_period = pd.concat(period_frames, ignore_index=True)

    def _summ(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        std = g["ret"].std(ddof=1) if n > 1 else np.nan
        t = g["ret"].mean() / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan
        return pd.Series(
            {
                "mean_ret": g["ret"].mean(),
                "std_ret": std,
                "t_stat": t,
                "n_periods": n,
                "avg_n_stocks": g["n_stocks"].mean(),
            }
        )

    summary = by_period.groupby("quantile").apply(_summ, include_groups=False)
    summary.index.name = "quantile"

    wide = by_period.pivot(index=date_col, columns="quantile", values="ret")
    if long_q not in wide.columns or short_q not in wide.columns:
        # A quantile that never formed on any date (e.g. n_quantiles=10 but
        # some dates only ever supported fewer bins) — surface it plainly
        # rather than silently returning an all-NaN spread.
        raise ValueError(
            f"quantile_portfolios: long_quantile={long_q} or "
            f"short_quantile={short_q} never formed in any period "
            f"(available: {sorted(wide.columns.tolist())})"
        )
    spread = (wide[long_q] - wide[short_q]).dropna()
    spread.name = "spread"

    n = len(spread)
    spread_mean = float(spread.mean()) if n > 0 else np.nan
    spread_std = float(spread.std(ddof=1)) if n > 1 else np.nan
    spread_t_stat = (
        spread_mean / (spread_std / np.sqrt(n)) if n > 1 and spread_std > 0 else np.nan
    )

    return QuantileResult(
        by_period=by_period, summary=summary, spread=spread,
        spread_mean=spread_mean, spread_t_stat=spread_t_stat,
    )
