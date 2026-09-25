"""
Forward returns: what a signal known at the close of day t is scored against.

The economics
-------------
A signal at (date, permno) is information available by the close of `date`
-- part 2's lagged `common_lag`/`shock_lag`, the follower's own lagged
return, or any other candidate "push signal" built from them. It cannot
predict day t's own return (that would be lookahead); it predicts what
happens AFTER t is known: the compounded return over the next `h` trading
sessions, t+1 .. t+h.

This mirrors `information_set.py`'s d-1 discipline exactly, just pointed the
other way: `information_set.attach_lagged` shifts characteristics BACK one
row so a signal never touches same-day data; `forward_returns` shifts
returns FORWARD so a signal is never scored against a return it could not
yet have seen. Anything downstream of both modules together is safe from
lookahead in both directions.

Trading sessions, not calendar days
------------------------------------
Windows are built by ROW POSITION within each stock (`groupby(permno)`,
shifted), the same convention `information_set.attach_lagged` uses looking
backward and `weekly_returns` uses in `microstructure.py`. A stock's "t+1" is
its next OBSERVED row, not the calendar day after t -- a stock that did not
trade Friday has Monday as "t+1" if Monday was its next session. A five-
session forward return is five sessions of actual exposure, regardless of
the calendar days they span.

Missing days inside a window
-----------------------------
If ANY of the sessions inside a window has a missing return (`ret` is NaN --
CRSP's own gap, not a delisting: those are already folded into `ret` by
`daily_returns.adjusted_daily_returns`), the whole window is NaN, not
silently treated as a 0% day. Getting this wrong would understate volatility
and, worse, would do so more for the illiquid names most likely to have gaps
-- exactly the population this project's hypothesis is about. See
`compounded_window_return`'s docstring for how that is enforced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compounded_window_return(
    panel: pd.DataFrame,
    shift: int,
    ret_col: str = "ret",
    permno_col: str = "permno",
) -> np.ndarray:
    """Compounded return over a window `|shift|` rows away from each row,
    within its stock, with correct all-or-nothing NaN handling.

    Args:
        panel: sorted or not -- this function sorts internally by
            `(permno_col, <original row order preserved via index>)`.
        shift: `shift > 0` looks BACKWARD -- the window is the `shift`
            sessions strictly before this row, i.e. `(i - shift, i]` by
            position, the trailing return ending at this row.  `shift < 0`
            looks FORWARD -- the window is the `abs(shift)` sessions after
            this row, i.e. `(i, i + abs(shift)]`, the forward return starting
            after this row.  There is no `shift = 0`.
        ret_col, permno_col: column names.

    Returns:
        A numpy array aligned to `panel` **in the row order this function
        sorts to** (`(permno_col, original position)`) -- callers that need
        it aligned to their own frame should assign it back onto a frame
        built the same way this function builds its own (see
        `forward_returns`, which does exactly that), not assume positional
        alignment with an arbitrarily-ordered input.

    How the NaN rule is enforced.  Missing return days are treated as a 0%
    contribution for the purpose of a vectorised cumulative sum (so the
    cumsum itself never carries pandas' own NaN-skipping surprises), while a
    SEPARATE cumulative count of missing days is carried alongside. A
    window's result is NaN whenever that count differs between the two ends
    of the window -- i.e. whenever a missing day falls anywhere inside it --
    regardless of whether the missing day is in the interior or right at the
    window's edge. This is the same two-pass trick either way `shift` points;
    only which end subtracts from which flips.

    Not `rolling()`.  `.rolling(k).apply(...)` runs a Python callable once per
    window -- `O(n*k)` callables on a multi-million-row panel -- and does not
    look forward at all. This is two `groupby().cumsum()` passes and one
    `groupby().shift()` per call: `O(n)`, matching the performance discipline
    `industry_map.industry_returns` documents for the same reason.
    """
    if shift == 0:
        raise ValueError("compounded_window_return: shift must not be 0")

    df = panel[[permno_col, ret_col]].copy()
    df["_orig_order"] = np.arange(len(df))
    df = df.sort_values([permno_col, "_orig_order"])

    ret = df[ret_col].to_numpy()
    missing = np.isnan(ret)
    df["_log_gross_filled"] = np.where(missing, 0.0, np.log1p(ret))
    df["_missing"] = missing.astype(float)

    grouped = df.groupby(permno_col, sort=False)
    df["_cumsum"] = grouped["_log_gross_filled"].cumsum()
    df["_miss_cumsum"] = grouped["_missing"].cumsum()
    grouped = df.groupby(permno_col, sort=False)  # re-group: columns just added

    other_cumsum = grouped["_cumsum"].shift(shift)
    other_miss = grouped["_miss_cumsum"].shift(shift)

    if shift > 0:
        # trailing window (i - shift, i]: this row's cumsum minus the one
        # `shift` rows back.
        log_sum = df["_cumsum"] - other_cumsum
        miss_in_window = df["_miss_cumsum"] - other_miss
    else:
        # forward window (i, i + |shift|]: the one `|shift|` rows ahead
        # minus this row's cumsum.
        log_sum = other_cumsum - df["_cumsum"]
        miss_in_window = other_miss - df["_miss_cumsum"]

    valid = miss_in_window.eq(0) & other_cumsum.notna()
    return np.where(valid, np.expm1(log_sum), np.nan)


def forward_returns(
    panel: pd.DataFrame,
    horizons: tuple[int, ...] | list[int] = (1, 5, 10, 20),
    ret_col: str = "ret",
    date_col: str = "date",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Compounded forward return over each horizon, one column per horizon.

    Args:
        panel: a daily frame with `date_col`, `permno_col`, `ret_col` -- one
            row per (date, permno), e.g. `leaders.follower_panel` output.
        horizons: trading-session counts to compound over. `1` is the return
            from t+1 alone; `5` compounds t+1..t+5; etc.
        ret_col, date_col, permno_col: column names.

    Returns:
        A frame with `date_col`, `permno_col`, and one `fwd_ret_{h}d` column
        per horizon: the compounded return over the next `h` sessions AFTER
        `date_col`, or NaN if that stock has fewer than `h` further observed
        rows in `panel` (most often: the sample just ends there) or any of
        those `h` sessions has a missing return.

    A NaN forward return crossing a delisting is not silently "missing the
    worst day": `DailyReturns.ret` (see `daily_returns.py`) already folds the
    delisting return into `ret` via Shumway's correction, so a window that
    compounds through a delisting compounds the REALISED outcome, exactly as
    a holder experienced it. A forward return can only be NaN here when there
    genuinely is no further data -- the panel's own end, or a true CRSP gap.
    """
    df = panel[[date_col, permno_col]].copy()
    df["_orig_order"] = np.arange(len(df))
    df = df.sort_values([permno_col, "_orig_order"]).reset_index(drop=True)

    for h in horizons:
        df[f"fwd_ret_{h}d"] = compounded_window_return(
            panel, shift=-h, ret_col=ret_col, permno_col=permno_col
        )

    return df.drop(columns=["_orig_order"]).reset_index(drop=True)
