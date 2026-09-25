"""
Part 3 — Alpha Signal & Validation (not part 1; owned by teammate 3).

Builds the conditional push signal from part 2's decomposition, and validates
it: information coefficient and rank IC, horizon analysis, quintile
portfolios, and the comparison against a plain short-term reversal baseline.

Part 1's `baseline.horizon_profile` and `baseline.quintile_sort` are the
unconditional versions of two of those exhibits — reuse them so the
conditional and unconditional numbers are computed the same way and the
comparison in the report is apples to apples.

What this part hands over:
    conditional.conditional_signal_panel(follower_panel, shocks, lag=1)
        -> follower-days with `common_lag`, `shock_lag`, `own_lag`,
           `leader_ret_lag`, and the composite `conditional_signal =
           common_lag - shock_lag` (hypotheses 1 & 2, README, as one
           sortable number).

    forward_returns.forward_returns(panel, horizons=(1,5,10,20))
        -> compounded return over the next h sessions, per stock-day.
           Built independently of any one signal so the same forward-return
           frame scores every candidate signal identically.

    ic.compute_ic / ic.horizon_ic
        -> daily cross-sectional Spearman IC (Grinold & Kahn), summarised
           across dates: mean, t-stat (naive, periods-independent), IR,
           pct positive.  `horizon_ic` runs this across every horizon in one
           call and returns the decay table.

    portfolios.quantile_portfolios
        -> per-date quantile sort and the resulting long-short spread's
           return series, mean, and t-stat.  The complement to IC: a signal
           can rank correctly on average (positive IC) while the tradeable
           spread it implies is noisy or concentrated in a few names.

Public API
----------
    conditional      conditional_signal_panel
    forward_returns  forward_returns, compounded_window_return
    ic               compute_ic, horizon_ic, ICResult
    portfolios       quantile_portfolios, QuantileResult

Usage, end to end::

    from lead_lag.shocks import decompose_rolling
    from lead_lag.signals import (
        conditional_signal_panel, forward_returns, horizon_ic,
        quantile_portfolios,
    )

    shocks = decompose_rolling(frame, spec)          # part 2, point-in-time
    signal = conditional_signal_panel(fp, shocks)    # part 3
    fwd = forward_returns(fp)                        # part 3

    ic_table = horizon_ic(signal, fwd, signal_col="conditional_signal")
    q = quantile_portfolios(
        signal.merge(fwd[["date", "permno", "fwd_ret_1d"]], on=["date", "permno"]),
        signal_col="conditional_signal", ret_col="fwd_ret_1d",
    )

Every exhibit here should be run three times per horizon -- once for
`conditional_signal`, once for `common_lag` alone, once for `shock_lag`
alone -- plus once for the naive baselines (`leader_ret_lag`, unconditional;
`-own_lag`, naive reversal).  A composite that "beats" the baselines while
one of its own legs is flat or wrong-signed is a materially weaker result
than both legs pulling their own weight, and the report should show which
case obtains.
"""

from lead_lag.signals.conditional import conditional_signal_panel
from lead_lag.signals.forward_returns import (
    compounded_window_return,
    forward_returns,
)
from lead_lag.signals.ic import ICResult, compute_ic, horizon_ic
from lead_lag.signals.portfolios import QuantileResult, quantile_portfolios

__all__ = [
    "conditional_signal_panel",
    "compounded_window_return",
    "forward_returns",
    "ICResult",
    "compute_ic",
    "horizon_ic",
    "QuantileResult",
    "quantile_portfolios",
]
