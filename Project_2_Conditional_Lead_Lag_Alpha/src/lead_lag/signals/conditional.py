"""
Part 3's conditional push signal: turning part 2's leader-shock decomposition
into a single, rankable per-follower-day number, ready for
`ic.horizon_ic` / `portfolios.quantile_portfolios`.

Why this reuses `shocks.regression.build_conditional_panel` unchanged
-----------------------------------------------------------------------
Part 2 already solved "merge the leader's lagged common/specific components
and the follower's own lag onto every follower-day" -- that is exactly what
`build_conditional_panel` returns (`common_lag`, `shock_lag`, `own_lag`,
`leader_ret_lag`, one row per follower-day). Part 3's IC-based validation
needs the same merge part 2's Fama-MacBeth regression uses; it just SCORES
the result differently -- rank IC and a quantile long-short spread instead
of a cross-sectional regression coefficient. Rebuilding the merge here would
risk it drifting out of sync with part 2's own regression, which is exactly
the trap `decomposition.py`'s module docstring warns about (two places doing
the same join, one of them eventually wrong).

The composite signal
---------------------
    conditional_signal = common_lag - shock_lag

The sign is not arbitrary; it is hypotheses 1 and 2 (README) rendered as one
sortable number:

* Hypothesis 1: a follower should CONTINUE on the leader's common component
  -- b_c > 0, so a bigger `common_lag` should predict a bigger forward
  return.  `common_lag` enters with a PLUS.
* Hypothesis 2: a follower should REVERT on the leader's leader-specific
  shock -- b_u < 0, so a bigger `shock_lag` should predict a SMALLER forward
  return.  `shock_lag` enters with a MINUS.

A follower-day where the leader's move was large, common, and positive, or
large, specific, and negative, gets the highest `conditional_signal`; a day
that was the opposite of both gets the lowest.  The two components only pull
the composite toward agreement when the theory in hypotheses 1 and 2 is
actually right -- which is what hypothesis 3 claims (conditioning beats
undifferentiated reversal), now testable with `ic.horizon_ic` and
`portfolios.quantile_portfolios` exactly as the unconditional baselines are.

`common_lag` and `shock_lag` are also kept as their own columns so each can
be IC/quantile-tested on its own -- the report should show all three, not
just the composite, since a composite that "works" while one leg is flat or
wrong-signed is a different (weaker) finding than both legs pulling their
own weight.
"""

from __future__ import annotations

import pandas as pd

from lead_lag.shocks.decomposition import LeaderShocks
from lead_lag.shocks.regression import build_conditional_panel


def conditional_signal_panel(
    follower_panel: pd.DataFrame,
    shocks: LeaderShocks,
    lag: int = 1,
    ret_col: str = "ret",
) -> pd.DataFrame:
    """Follower-days carrying part 2's lagged decomposed leader components
    plus the composite `conditional_signal`.

    Args:
        follower_panel: `leaders.follower_panel` output (optionally already
            merged with the daily factor file -- extra columns pass through
            untouched).
        shocks: a `LeaderShocks` from `shocks.decompose_rolling` for any
            signal that will actually be traded/back-tested (point-in-time,
            no look-ahead) -- or from `shocks.decompose_full_sample` only for
            a descriptive/in-sample comparison, never for a headline result.
        lag: how many trading days back the leader's decomposed move is
            drawn from, mirroring `baseline.build_lags`'s `lag`.
        ret_col: the follower's own return column.

    Returns:
        `build_conditional_panel`'s frame (`date, permno, ff49, ret,
        leader_ret, size_pct, mktcap_lag, ..., own_lag, common_lag,
        shock_lag, leader_ret_lag`) with `conditional_signal` appended.
        Rows where either component is missing (e.g. a follower whose
        industry had no leader-shock estimate that day -- a thin industry
        under `decompose_rolling`'s `min_obs`) get a NaN `conditional_signal`
        and are dropped by `ic.compute_ic` / `portfolios.quantile_portfolios`
        the same way any other NaN signal row is.
    """
    panel = build_conditional_panel(follower_panel, shocks, lag=lag, ret_col=ret_col)
    panel["conditional_signal"] = panel["common_lag"] - panel["shock_lag"]
    return panel
