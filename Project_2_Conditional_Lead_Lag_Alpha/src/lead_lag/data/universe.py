"""
Who is in the sample, month by month, and why each excluded stock was
excluded.

The universe is rebalanced monthly and decided ENTIRELY from data observed
strictly before the month begins.  That is the one non-negotiable property of
this module: a stock is in the December universe because of what it looked
like through November.  Any filter that peeks at December — "stocks that
traded every day of the month", "stocks with a December market cap above X" —
selects on the outcome, and in a study of short-horizon return predictability
that selection alone can manufacture the result.  Every function here takes a
formation window that ends before the eligibility month starts, and the test
suite pins it.

The filters, and what each one is protecting against
----------------------------------------------------
1. **Price floor ($5 at formation).**  Below a few dollars the tick is a large
   fraction of the price, so the bid-ask bounce dominates daily returns.
   Bounce is mechanically negatively autocorrelated, which is indistinguishable
   from the "reversal" this project is trying to measure.  Excluding penny
   stocks is how the literature keeps a microstructure artefact from being
   read as an economic effect — and it is the single filter that most changes
   the answer, so part 5 should re-run without it.

2. **Size floor (NYSE market-cap percentile).**  The breakpoint is computed on
   NYSE stocks only and then applied to all three exchanges, which is the
   Fama-French convention.  Using all stocks would put the breakpoint near the
   bottom of the NASDAQ microcap mass and effectively exclude nothing.

3. **Liquidity floor (median dollar volume).**  A follower that trades $50k a
   day cannot absorb the position the strategy would want, so an alpha that
   lives there is not implementable.  Ranked WITHIN exchange, because CRSP's
   NASDAQ volume double-counts dealer trades before 2004 and a pooled rank
   would call NASDAQ stocks twice as liquid as they are for a purely
   mechanical reason.

4. **Observation count.**  A stock needs enough non-missing daily returns in
   the formation window for its own statistics — and, later, its beta to the
   leader — to mean anything.

5. **A real industry.**  Fama-French industry 49 ("Other") is the residue
   their ranges do not cover.  It is not an industry: its members share no
   common shock, so a "leader" in it leads nothing.  Excluded from the
   lead-lag construction, and the count is reported rather than hidden.

6. **A populated industry.**  A lead-lag test needs one leader and several
   followers.  An industry-month with fewer than `min_industry_size` eligible
   stocks is dropped whole.

Output shape
------------
`Universe` is one row per (eligible_month, permno) with the formation
statistics that produced the decision and a boolean column per filter.  The
per-filter columns are the point: a referee's first question is "how much does
the result depend on the $5 screen?", and the answer is a group-by away rather
than a re-run away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from lead_lag.data.industry_map import OTHER_INDUSTRY
from lead_lag.data.typed_dataset import Dataset


@dataclass(frozen=True)
class UniverseRules:
    """The filter thresholds, in one object so a robustness run is one line.

    Attributes:
        formation_months:   length of the trailing window, in months.
        min_price:          price floor in dollars at the formation month end.
        min_nyse_size_pct:  NYSE market-cap percentile a stock must exceed
                            (0.20 = above the 20th percentile of NYSE stocks).
        min_dollar_vol_pct: within-exchange percentile of median daily dollar
                            volume a stock must exceed.
        min_obs:            minimum non-missing daily returns in the window.
        min_industry_size:  minimum eligible stocks in an industry-month for
                            that industry-month to be used at all.
        exclude_other:      drop Fama-French industry 49.
    """

    formation_months: int = 12
    min_price: float = 5.0
    min_nyse_size_pct: float = 0.20
    min_dollar_vol_pct: float = 0.20
    min_obs: int = 120
    min_industry_size: int = 5
    exclude_other: bool = True


class Universe(Dataset):
    """Monthly eligibility decisions, one row per (month, permno).

    Canonical columns:
        month           first calendar day of the ELIGIBLE month (datetime64)
        permno          int
        ff49            industry number at formation
        exchcd          exchange at formation
        prc             price at the formation window's last observed day
        mktcap          market cap at that day, dollars
        med_dollar_vol  median daily dollar volume over the window
        n_obs           non-missing daily returns in the window
        nyse_size_pct   the stock's market cap as a percentile of the NYSE
                        cross-section at formation (0-1)
        dvol_pct        percentile of median dollar volume within its exchange
        pass_price, pass_size, pass_liquidity, pass_obs, pass_industry
                        one boolean per filter
        eligible        all of the above, and the industry-size rule

    `month` is the month the stock may be TRADED in; every other column is
    measured over the window that ended before it.
    """

    KEY: ClassVar[tuple[str, ...]] = ("month", "permno")
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "month", "permno", "ff49", "exchcd", "prc", "mktcap", "med_dollar_vol",
        "n_obs", "nyse_size_pct", "dvol_pct", "pass_price", "pass_size",
        "pass_liquidity", "pass_obs", "pass_industry", "eligible",
    )
    SYNONYMS: ClassVar[dict[str, str]] = {}

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["month"] = pd.to_datetime(df["month"])
        for col in ("permno", "ff49", "exchcd", "n_obs"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        for col in ("prc", "mktcap", "med_dollar_vol", "nyse_size_pct", "dvol_pct"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("pass_price", "pass_size", "pass_liquidity", "pass_obs",
                    "pass_industry", "eligible"):
            df[col] = df[col].astype(bool)
        return df.sort_values(["month", "permno"]).reset_index(drop=True)


def monthly_stock_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the daily panel to one row per (stock, calendar month).

    This is the step that makes a 30-year monthly rebalance cheap: the
    trailing-window statistics are then rolling sums and medians over ~360
    months per stock instead of 360 passes over 50 million daily rows.

    Returns columns: permno, month, exchcd, ff49, prc, mktcap, med_dollar_vol,
    n_obs — where `prc` and `mktcap` are the month's LAST observed values and
    `n_obs` counts non-missing daily returns.
    """
    df = panel.copy()
    # Month as a timestamp at the first of the month: sortable, joinable, and
    # it prints readably in a table (unlike a pandas Period).
    df["month"] = df["date"].values.astype("datetime64[M]")
    df = df.sort_values(["permno", "date"])

    grouped = df.groupby(["permno", "month"], sort=True)
    stats = grouped.agg(
        exchcd=("exchcd", "last"),
        ff49=("ff49", "last"),
        prc=("prc", "last"),
        mktcap=("mktcap", "last"),
        med_dollar_vol=("dollar_vol", "median"),
        n_obs=("ret", "count"),          # count() skips NaN — that is the point
    ).reset_index()
    return stats


def formation_window_stats(
    monthly: pd.DataFrame, formation_months: int = 12
) -> pd.DataFrame:
    """Trailing-window statistics, shifted so they are known before the month.

    For each (permno, month) row, compute over the `formation_months` months
    ENDING WITH THE PREVIOUS MONTH:
        n_obs           total non-missing daily returns
        med_dollar_vol  median of the monthly medians
        prc, mktcap     the last values of the window's final month
        exchcd, ff49    likewise

    The `shift(1)` after the rolling aggregation is the point-in-time
    guarantee: row `month = 2015-12-01` carries statistics through
    2015-11-30.  Take the shift out and the whole study is lookahead.

    A stock with gaps in its monthly history is handled by the rolling window
    counting MONTHS PRESENT rather than calendar months; a stock that stopped
    trading for half a year therefore carries older data forward rather than
    being dropped, and `n_obs` — which does count actual observations — is the
    filter that catches it.
    """
    df = monthly.sort_values(["permno", "month"]).copy()
    g = df.groupby("permno", sort=False)

    rolled = pd.DataFrame(
        {
            "permno": df["permno"],
            "month": df["month"],
            "n_obs_w": g["n_obs"].transform(
                lambda s: s.rolling(formation_months, min_periods=1).sum()
            ),
            "med_dollar_vol_w": g["med_dollar_vol"].transform(
                lambda s: s.rolling(formation_months, min_periods=1).median()
            ),
        }
    )
    # End-of-window levels: the final month's own values.
    rolled["prc_w"] = df["prc"].to_numpy()
    rolled["mktcap_w"] = df["mktcap"].to_numpy()
    rolled["exchcd_w"] = df["exchcd"].to_numpy()
    rolled["ff49_w"] = df["ff49"].to_numpy()

    # Shift every statistic forward one month: what row `month` knows is what
    # was true at the end of `month - 1`.
    shifted = rolled.groupby("permno", sort=False).shift(1)
    shifted["permno"] = rolled["permno"].to_numpy()
    shifted["month"] = rolled["month"].to_numpy()
    return shifted.reset_index(drop=True)


def _nyse_size_percentile(frame: pd.DataFrame) -> pd.Series:
    """Each stock's market cap as a percentile of the NYSE cross-section.

    The Fama-French convention: the breakpoint is estimated on NYSE stocks
    (exchcd 1) only, then applied to everything.  A stock smaller than every
    NYSE stock gets 0.0; the value is a fraction of NYSE stocks it exceeds,
    which is what `UniverseRules.min_nyse_size_pct` is compared against.

    Called once per month by `build_universe`.
    """
    nyse = frame.loc[frame["exchcd_w"] == 1, "mktcap_w"].dropna().to_numpy()
    caps = frame["mktcap_w"].to_numpy()
    if nyse.size == 0:
        # No NYSE stocks this month (only happens in a truncated test fixture).
        # Fall back to the pooled cross-section rather than passing everyone.
        nyse = frame["mktcap_w"].dropna().to_numpy()
    if nyse.size == 0:
        return pd.Series(np.nan, index=frame.index)
    nyse_sorted = np.sort(nyse)
    # searchsorted gives how many NYSE caps are <= each stock's cap; dividing
    # by the count turns it into a percentile in [0, 1].
    pct = np.searchsorted(nyse_sorted, caps, side="right") / nyse_sorted.size
    return pd.Series(np.where(np.isnan(caps), np.nan, pct), index=frame.index)


def build_universe(
    panel: pd.DataFrame, rules: UniverseRules | None = None
) -> tuple[Universe, pd.DataFrame]:
    """Decide monthly eligibility for every stock in the daily panel.

    Args:
        panel: the daily frame, already carrying `ff49` (run
            `industry_map.attach_industry` first) and `dollar_vol`.
        rules: thresholds; the defaults are the ones the report describes.

    Returns:
        (Universe, attrition).  `attrition` is a month-by-filter table of how
        many stocks each screen removed — the figure that goes in the data
        section, and the honest answer to "how big is your sample and what did
        you throw away?".

    The industry-size rule is applied last and depends on the other five, so
    it is not a column of `formation_window_stats`: a stock can pass every
    individual screen and still be dropped because it ended up alone in its
    industry that month.
    """
    rules = rules or UniverseRules()

    monthly = monthly_stock_stats(panel)
    stats = formation_window_stats(monthly, rules.formation_months)
    stats = stats.dropna(subset=["mktcap_w"])

    # ---- per-month cross-sectional percentiles ---------------------------
    stats["nyse_size_pct"] = (
        stats.groupby("month", group_keys=False).apply(
            _nyse_size_percentile, include_groups=False
        )
    )
    # Dollar-volume rank within (month, exchange) — see filter 3 above.
    stats["dvol_pct"] = stats.groupby(["month", "exchcd_w"])["med_dollar_vol_w"].rank(
        pct=True, na_option="keep"
    )

    # ---- the five independent screens ------------------------------------
    stats["pass_price"] = stats["prc_w"] >= rules.min_price
    stats["pass_size"] = stats["nyse_size_pct"] >= rules.min_nyse_size_pct
    stats["pass_liquidity"] = stats["dvol_pct"] >= rules.min_dollar_vol_pct
    stats["pass_obs"] = stats["n_obs_w"] >= rules.min_obs
    stats["pass_industry"] = (
        (stats["ff49_w"] != OTHER_INDUSTRY) if rules.exclude_other
        else pd.Series(True, index=stats.index)
    )

    passes_all = (
        stats["pass_price"] & stats["pass_size"] & stats["pass_liquidity"]
        & stats["pass_obs"] & stats["pass_industry"]
    )

    # ---- the industry-size rule, applied to the survivors -----------------
    survivors = stats.loc[passes_all]
    sizes = survivors.groupby(["month", "ff49_w"])["permno"].transform("nunique")
    big_enough = pd.Series(False, index=stats.index)
    big_enough.loc[survivors.index] = sizes >= rules.min_industry_size

    stats["eligible"] = passes_all & big_enough

    out = pd.DataFrame(
        {
            "month": stats["month"],
            "permno": stats["permno"],
            "ff49": stats["ff49_w"],
            "exchcd": stats["exchcd_w"],
            "prc": stats["prc_w"],
            "mktcap": stats["mktcap_w"],
            "med_dollar_vol": stats["med_dollar_vol_w"],
            "n_obs": stats["n_obs_w"],
            "nyse_size_pct": stats["nyse_size_pct"],
            "dvol_pct": stats["dvol_pct"],
            "pass_price": stats["pass_price"],
            "pass_size": stats["pass_size"],
            "pass_liquidity": stats["pass_liquidity"],
            "pass_obs": stats["pass_obs"],
            "pass_industry": stats["pass_industry"],
            "eligible": stats["eligible"],
        }
    )
    return Universe.from_raw(out), attrition_table(out)


def attrition_table(universe: pd.DataFrame) -> pd.DataFrame:
    """Per-month counts: candidates, survivors of each screen, final eligible.

    The columns are CUMULATIVE in the order the report describes the filters,
    which is the version a reader can follow: each number is "how many are
    left", not "how many did this screen touch".
    """
    df = universe.copy()
    order = ["pass_obs", "pass_price", "pass_size", "pass_liquidity", "pass_industry"]
    rows = {"candidates": df.groupby("month")["permno"].nunique()}
    running = pd.Series(True, index=df.index)
    for col in order:
        running = running & df[col]
        rows[f"after {col.removeprefix('pass_')}"] = (
            df.loc[running].groupby("month")["permno"].nunique()
        )
    rows["eligible"] = df.loc[df["eligible"]].groupby("month")["permno"].nunique()
    return pd.DataFrame(rows).fillna(0).astype("int64")


def eligible_daily(panel: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    """Restrict the daily panel to eligible (month, permno) pairs.

    The join key is the daily row's own calendar month against the universe's
    `month`, so a stock eligible in December appears on every December day it
    traded and on none of November's.
    """
    df = panel.copy()
    df["month"] = df["date"].values.astype("datetime64[M]")
    keep = universe.frame.loc[universe.frame["eligible"], ["month", "permno", "ff49"]]
    # `ff49` is carried from the universe, not from the daily row: the
    # industry a stock is traded IN this month is the one it was assigned at
    # formation.  Letting it change mid-month would move a stock between
    # leader groups in the middle of a holding period.
    return df.drop(columns=["ff49"], errors="ignore").merge(
        keep, on=["month", "permno"], how="inner"
    )
