"""
Bid-ask bounce, stale closes, and the returns that are immune to them.

Why this module exists
----------------------
The strategy is a short-horizon reversal in small, illiquid stocks.  That is
the exact setting in which the measurement apparatus manufactures the result
being looked for, by two distinct mechanisms:

**1. Bid-ask bounce (Roll 1984).**  A close-to-close return is the difference
between two prints, and each print is at the bid or the ask depending on
whether the last trade was a sell or a buy.  If trade direction is serially
random, the price series picks up a spurious component that is *mechanically
negatively autocorrelated*: a print at the ask tends to be followed by one at
the bid, which reads as a reversal.  Roll showed the induced first-order
autocovariance is -(s/2)^2 in the spread s, so the fake reversal scales with
the SQUARE of the spread.

On this sample that is not a footnote.  Median relative spread across CRSP
common stocks:

    1996   3.23%      (pre-decimalisation, tick = 1/8)
    2005   0.29%
    2015   0.11%
    2024   0.18%

and the unconditional lead-lag coefficient is concentrated in 1996-2006.  The
era with the effect is the era with 10-30x the spread.  That correlation is
the single most important thing to rule out before believing any of this.

**2. Non-synchronous trading (Lo & MacKinlay 1990).**  A small stock's
"close" may be a quote struck at 14:30, while the leader's is a print at
16:00.  Day d's follower close then omits information the leader's close
contains, and day d+1's follower return picks it up — which looks exactly
like information diffusing from leader to follower, without any diffusion
happening.  This inflates measured lead-lag in precisely the illiquid names
the hypothesis is about.  `CRSPDaily.quote_only` flags the days on which CRSP
itself says there was no closing trade.

The three defenses
------------------
The project's own research design names them, and each is implemented here:

* **Measure on midpoints, not closes.**  The midpoint (bid+ask)/2 does not
  bounce — it sits between the quotes rather than alternating across them.
  `midpoint_returns` builds a return series from it.
* **Skip a day.**  Bounce is a FIRST-order effect: it contaminates the t to
  t+1 return and dies after that.  An effect measured from t+2 that survives
  is not bounce.  In this codebase that is simply `baseline.build_lags(...,
  lag=2)`, and it is worth stating that the unconditional lag-2 coefficient
  on this sample is already ~0 (beta_FM 0.0012, t = 0.72) against a strongly
  significant lag-1 (0.0093, t = 5.03).
* **Confirm at weekly horizons.**  A five-day return contains one bounce at
  each end rather than one per observation, so the artefact's share of the
  variance falls by roughly a factor of five.  `weekly_returns` builds it.

Nothing here filters anything.  It produces alternative return series and
diagnostic columns, so the robustness section can show the result with and
without each defense.  Deciding which to headline is part 5's call, and it
should be made on the numbers, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Split adjustment
# ---------------------------------------------------------------------------

def split_adjust(price: pd.Series, cfacpr: pd.Series) -> pd.Series:
    """Divide a raw price column by CRSP's cumulative adjustment factor.

    `prc`, `bid` and `ask` in `crsp.dsf` are the RAW prints: a 2-for-1 split
    halves them overnight.  CRSP's own `ret` and `retx` are already adjusted,
    but anything computed from the price columns is not.  Dividing by
    `cfacpr` puts every price on one continuous scale.

    Skipping this step would put a -50% observation in the panel on every
    split date.  In a study of short-horizon reversal those would not be
    noise — a -50% day followed by a normal day is the largest "reversal"
    in the sample, and there are thousands of them.
    """
    factor = pd.to_numeric(cfacpr, errors="coerce")
    # A zero or missing factor would silently produce inf/NaN; treat it as 1
    # (no adjustment) and let the caller see the price unchanged rather than
    # destroyed.  CRSP's factor is 1.0 for the vast majority of rows.
    factor = factor.where(factor > 0, 1.0)
    return pd.to_numeric(price, errors="coerce") / factor


# ---------------------------------------------------------------------------
# Quotes: midpoint, spread, staleness
# ---------------------------------------------------------------------------

def attach_quote_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add midpoint, relative spread, and the bounce-exposure diagnostics.

    Adds:
        mid            split-adjusted bid-ask midpoint, NaN when the quote is
                       unusable
        rel_spread     (ask - bid) / mid, the bounce's scale
        quote_valid    ask > bid > 0 and both present
        roll_bound     Roll's implied one-way bounce, rel_spread / 2 — the
                       magnitude of the spurious return component, for
                       comparison against any measured reversal
        stale_close    True when CRSP reported no closing trade that day
                       (`quote_only`), i.e. the non-synchronous case

    `quote_valid` is not a formality: crossed and locked quotes (ask <= bid)
    appear in CRSP, especially early in the sample, and a midpoint computed
    from them is meaningless rather than merely noisy.
    """
    df = panel.copy()

    has_quote = df["bid"].notna() & df["ask"].notna()
    valid = has_quote & (df["ask"] > df["bid"]) & (df["bid"] > 0)
    df["quote_valid"] = valid

    raw_mid = (df["bid"] + df["ask"]) / 2.0
    cfacpr = df["cfacpr"] if "cfacpr" in df.columns else pd.Series(1.0, index=df.index)
    df["mid"] = split_adjust(raw_mid, cfacpr).where(valid)

    df["rel_spread"] = ((df["ask"] - df["bid"]) / raw_mid).where(valid)
    # Roll's induced autocovariance is -(s/2)^2; the one-way move a bounce
    # produces is s/2.  Reported as a level so it can be compared directly
    # against a measured mean reversal in the same units.
    df["roll_bound"] = df["rel_spread"] / 2.0

    df["stale_close"] = (
        df["quote_only"].astype(bool) if "quote_only" in df.columns
        else (df["vol"].fillna(0) == 0)
    )
    return df


def midpoint_returns(panel: pd.DataFrame, out_col: str = "mid_ret") -> pd.DataFrame:
    """Daily returns computed from the split-adjusted bid-ask midpoint.

    Requires `attach_quote_columns` first.  Adds `<out_col>` and
    `<out_col>_valid` (True when BOTH ends of the return had a usable quote —
    a midpoint return is only bounce-free if neither endpoint is a close).

    What this series is and is not
    ------------------------------
    It is a PRICE return.  It excludes dividends, so the honest comparison is
    against CRSP's `retx`, not `ret`.  `compare_to_close` does that.

    It is not a tradeable return: nobody transacts at the midpoint.  Its
    purpose is diagnostic — if an effect is present in close-to-close returns
    and absent in midpoint returns, the effect was the spread.  A strategy
    that survives on midpoints still has to pay the spread, which is part 5's
    transaction-cost work, and the `rel_spread` column is the input to it.

    The shift is within stock and by row, so a stock's first observation and
    any row whose predecessor lacked a valid quote come back NaN rather than
    silently spanning a gap.
    """
    df = panel.sort_values(["permno", "date"]).copy()
    grouped = df.groupby("permno", sort=False)

    prev_mid = grouped["mid"].shift(1)
    df[out_col] = df["mid"] / prev_mid - 1.0
    df[f"{out_col}_valid"] = df["quote_valid"] & grouped["quote_valid"].shift(1).fillna(False)
    df.loc[~df[f"{out_col}_valid"], out_col] = np.nan
    return df.reset_index(drop=True)


def compare_to_close(
    panel: pd.DataFrame, mid_col: str = "mid_ret", close_col: str = "retx"
) -> pd.DataFrame:
    """How far the midpoint return differs from CRSP's price return, by year.

    The comparison is against `retx` (price return excluding dividends),
    because the midpoint series has no dividends in it.  Comparing against
    `ret` would attribute every dividend to the spread.

    Returns a per-year frame: correlation, the two standard deviations, and
    the first-order autocorrelation of each.  The last pair is the diagnostic
    that matters — **bounce shows up as negative autocorrelation in the close
    series that is absent from the midpoint series.**  If the two
    autocorrelations are similar, bounce is not driving the reversal; if the
    close series is markedly more negative, it is.
    """
    df = panel[["date", "permno", mid_col, close_col]].dropna().copy()
    df["year"] = df["date"].dt.year

    def _ac1(frame: pd.DataFrame, col: str) -> float:
        """Mean over stocks of each stock's first-order autocorrelation."""
        per_stock = frame.groupby("permno")[col].apply(
            lambda s: s.autocorr(lag=1) if len(s) > 20 else np.nan
        )
        return float(per_stock.mean())

    rows = []
    for year, chunk in df.groupby("year"):
        rows.append(
            {
                "year": year,
                "n_obs": len(chunk),
                "corr": float(chunk[mid_col].corr(chunk[close_col])),
                f"sd_{mid_col}": float(chunk[mid_col].std()),
                f"sd_{close_col}": float(chunk[close_col].std()),
                f"ac1_{mid_col}": _ac1(chunk, mid_col),
                f"ac1_{close_col}": _ac1(chunk, close_col),
            }
        )
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Weekly returns
# ---------------------------------------------------------------------------

def weekly_returns(
    panel: pd.DataFrame, ret_col: str = "ret", label: str = "W-FRI"
) -> pd.DataFrame:
    """Compound daily returns into non-overlapping weekly returns per stock.

    Args:
        panel: daily frame with `date`, `permno` and the return column.
        ret_col: which daily return to compound — `ret` for the close-to-close
            series, `mid_ret` for the bounce-free one.
        label: pandas resample rule; `W-FRI` ends weeks on Friday, which is
            the convention that puts a normal holiday-free week in one bucket.

    Returns:
        Frame with `week_end`, `permno`, `<ret_col>_w`, `n_days`.

    Why weekly is a real test rather than a cosmetic one.  Bounce contributes
    one spurious increment per RETURN OBSERVATION, not per day: a five-day
    compounded return contains one bounce at each end, the same as a one-day
    return, while carrying five days of genuine variance.  The artefact's
    share of the variance therefore falls by roughly a factor of five.  An
    effect that is present daily and gone weekly is consistent with bounce; an
    effect that survives weekly is not.

    `n_days` is kept so short weeks (holidays, a stock that traded twice) can
    be dropped — a "weekly" return built from two days is neither.
    """
    df = panel[["date", "permno", ret_col]].dropna().copy()
    gross = 1.0 + df[ret_col]
    df["_g"] = gross

    out = (
        df.set_index("date")
        .groupby("permno")
        .resample(label)
        .agg(_g=("_g", "prod"), n_days=(ret_col, "size"))
        .reset_index()
        .rename(columns={"date": "week_end"})
    )
    out[f"{ret_col}_w"] = out["_g"] - 1.0
    out = out.loc[out["n_days"] > 0]
    return out[["week_end", "permno", f"{ret_col}_w", "n_days"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The headline diagnostic
# ---------------------------------------------------------------------------

def bounce_exposure(panel: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    """Spread, staleness and Roll's bound, aggregated — the table that says
    how much room the artefact has.

    Put beside the measured reversal in the same units.  If the mean
    `roll_bound` is 160 bps and the measured effect is 4 bps, the artefact is
    forty times larger than the thing being measured, and any claim rests
    entirely on the two being uncorrelated.  That is a much weaker position
    than a t-statistic alone suggests, and the report should say so plainly.
    """
    df = panel.copy()
    if by == "year":
        df["_k"] = df["date"].dt.year
    else:
        df["_k"] = df[by]

    out = df.groupby("_k").agg(
        rows=("permno", "size"),
        median_rel_spread=("rel_spread", "median"),
        mean_roll_bound=("roll_bound", "mean"),
        stale_close_share=("stale_close", "mean"),
        quote_valid_share=("quote_valid", "mean"),
    )
    out.index.name = by
    return out
