"""
Net-of-cost returns for Part 5.

Works on the daily return series Part 4 writes to
results/p4_portfolio_daily.csv, so it needs no licensed data.

Cost convention (docs/part4_deviations.md, change 1)
----------------------------------------------------
A convention-C book (open -> close) is flat overnight, so it buys its whole
gross at the open and sells it at the close: traded notional is 2 x gross every
day. A convention-A book (close -> close) is carried, and is charged its
realised turnover, sum |dw|, which Part 4 stores as convention
`turnover_carry`.

Only the spread is charged. Market impact, fees and borrow costs are left out,
which makes this the most favourable cost case for the strategy.
"""

from __future__ import annotations

import pandas as pd

ROUND_TRIP = 2.0  # traded notional per day of a gross-1 book that is flat overnight


def net_of_cost(gross: pd.Series, traded: pd.Series | float, cost_bp: float) -> pd.Series:
    """Daily return after a one-way cost of `cost_bp` basis points.

    `traded` is the notional traded that day as a multiple of gross exposure: a
    Series aligned to `gross`, or a constant (`ROUND_TRIP` for convention C).
    A day with no gross return stays missing.
    """
    return gross - traded * cost_bp / 1e4


def breakeven_bp(gross: pd.Series, traded: pd.Series | float) -> float:
    """One-way cost, in basis points, at which the mean net return is zero."""
    if isinstance(traded, pd.Series):
        mean_traded = float(traded.reindex(gross.dropna().index).mean())
    else:
        mean_traded = float(traded)
    return float(gross.mean() / mean_traded * 1e4)
