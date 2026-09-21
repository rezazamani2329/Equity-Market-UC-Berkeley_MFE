"""
The information set available at the close of day d-1.

The rule this module enforces
-----------------------------
**Every characteristic used to condition a signal for day d must be the value
observed at d-1 or earlier.**  Not the value on day d.  This covers industry
classification, market capitalisation, price, dollar volume and exchange — not
just returns.

The return case is obvious and was never in doubt: you cannot use day d's
return to predict day d's return.  The characteristic cases are the ones that
slip through, because a characteristic *feels* static:

* **Market cap** is `prc x shrout`, and `prc` moves with the day's return.  A
  portfolio weighted by day d's market cap is weighted by day d's winners.
  The bias is not small and it always runs the same way: upward.
* **Industry classification** changes.  CRSP records it on name rows with
  effective dates, and a firm that reclassifies on day d appears in its NEW
  industry from day d.  If the signal for day d groups it by that new
  industry, the grouping used information stamped that morning.  The firm's
  peers on day d should be the peers it had at the close of d-1.
* **Dollar volume and price** enter liquidity screens and position sizing.
  Day d's volume is not known until day d closes.

What "the previous day" means here
----------------------------------
`shift(1)` within a stock, so a row's lagged value is that stock's **last
observed** value, not the calendar day before.  A stock that did not trade for
a week carries its week-old characteristics forward, which is correct: that is
genuinely the most recent information about it.  `days_since_prev` is attached
alongside so a consumer can see the staleness and drop it if it wants — a
characteristic 40 days old is a different object from one 1 day old, and the
robustness section should be able to say so.

A stock's first observed row has no predecessor and its lagged columns are
NaN.  They are left NaN rather than back-filled from the same day: filling
them would be a one-row leak, and the universe filter excludes a stock's first
year anyway (it needs a 12-month formation window).

What this module does NOT cover
-------------------------------
The universe and leader-assignment path is already stricter than d-1: it
decides month m from month m-1's closing statistics, so a characteristic there
is up to a month old, never same-day.  See `universe.formation_window_stats`.
This module is for the DAILY path — industry returns, and any characteristic a
signal touches at daily frequency.
"""

from __future__ import annotations

import pandas as pd

# The characteristics a signal might condition on, all of which move with the
# day's own trading and therefore must be lagged before use.
#
# `ff49` is in the list for the reason in the module docstring: it is not
# static.  `exchcd` is, in practice, nearly static — but it is derived from
# the same CRSP name row as `siccd`, so it changes on the same effective
# dates, and treating it differently would be an inconsistency nobody would
# remember six weeks from now.
LAGGED_COLUMNS: tuple[str, ...] = (
    "ff49", "ff49_name", "siccd", "mktcap", "prc", "dollar_vol", "vol", "exchcd",
)

SUFFIX = "_lag"


def attach_lagged(
    panel: pd.DataFrame,
    columns: tuple[str, ...] | list[str] = LAGGED_COLUMNS,
    suffix: str = SUFFIX,
) -> pd.DataFrame:
    """Add a `<column>_lag` for each characteristic, shifted one row per stock.

    Args:
        panel: the daily frame, with `date` and `permno`.  Columns in
            `columns` that are absent are skipped silently, so the same call
            works on a panel before and after `attach_industry`.
        columns: characteristics to lag.  The default is everything a signal
            could plausibly condition on.
        suffix: appended to make the new column name.

    Returns:
        A copy of `panel`, sorted by `(permno, date)`, with the lagged columns
        and `days_since_prev` added.  `days_since_prev` is NaN on a stock's
        first row and otherwise the calendar gap to its previous observation —
        1 on a normal day, 3 across a weekend, more across a suspension.

    The sort is not optional.  `groupby(...).shift(1)` moves by position
    within each group, so the frame must be in date order per stock or the
    "previous" value is whatever row happened to come before it.
    """
    df = panel.sort_values(["permno", "date"]).copy()
    present = [c for c in columns if c in df.columns]

    grouped = df.groupby("permno", sort=False)
    for col in present:
        df[f"{col}{suffix}"] = grouped[col].shift(1)

    # Staleness, in calendar days.  Kept beside the lagged values so a
    # consumer never has to guess how old they are.
    df["days_since_prev"] = grouped["date"].diff().dt.days

    # Integer-valued characteristics come back as float once shifting
    # introduces a NaN.  Left as float deliberately: casting back would
    # require a fill value, and there is no honest fill for "this stock has no
    # previous day".  Consumers group on them as floats or drop the NaNs.
    return df.reset_index(drop=True)


def require_lagged(panel: pd.DataFrame, column: str) -> str:
    """Return `column` if it already ends in the lag suffix, else its lagged
    name — raising if that is not present.

    Used by functions whose default conditioning column is a lagged one, so
    the error tells the caller what to run instead of silently grouping on
    same-day data:

        >>> require_lagged(df, "ff49_lag")
        'ff49_lag'
    """
    if column.endswith(SUFFIX):
        if column not in panel.columns:
            raise KeyError(
                f"{column!r} is not in the frame. Run "
                f"`information_set.attach_lagged(panel)` first — this function "
                f"conditions on the information set at d-1, not on day d."
            )
        return column
    return column


def same_day_columns(
    panel: pd.DataFrame, exempt: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    """Characteristics present in `panel` that have NOT been lagged.

    A diagnostic for the notebook and for review: hand it the frame a signal
    is about to be built from, and it names every column that still carries
    day-d information.  An empty list is the goal for anything downstream of
    the signal boundary.

    Args:
        panel: the frame to inspect.
        exempt: column names to skip because they are lagged by a route this
            function cannot see.  The one that matters: `follower_panel`'s
            `ff49` comes from the ROLE table, whose industry was fixed at the
            end of the previous month — older than d-1, so it satisfies the
            rule — but it carries no `_lag` suffix, because it is not a
            one-day lag of the daily column.  Pass `exempt={"ff49"}` there.

    The suffix is a naming convention, not a proof.  This check catches the
    common mistake (forgetting `attach_lagged`); it cannot certify provenance,
    which is why `exempt` requires the caller to state what they know.
    """
    return [
        c for c in LAGGED_COLUMNS
        if c in panel.columns
        and c not in exempt
        and f"{c}{SUFFIX}" not in panel.columns
    ]
