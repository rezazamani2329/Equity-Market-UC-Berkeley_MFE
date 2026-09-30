"""
Part 4 — Portfolio Optimization & Risk (owner: Thomas Claudel, part 4).

Long/short construction, market and sector neutrality, factor exposures,
Sharpe, VaR / expected shortfall, drawdown. Design choices are pre-registered
in docs/part4_prereg.md.

    inputs        returns under each execution convention, Dimson beta and
                  midpoint returns, computed on the full CRSP panel
    construction  industry-level book (leader signals), stock-level
                  reversal book, book returns, turnover, volatility overlay
    performance   summary statistics, drawdown, VaR/ES backtest, stress
                  table, Newey-West factor attribution, Holm correction

Part 1 supplies the inputs: `wrds_fetch` (CRSP daily, delistings, FF5 + UMD
factors), `daily_returns.adjusted_daily_returns` and `leaders.follower_panel`.
"""

from lead_lag.portfolio.construction import (
    book_returns,
    industry_book,
    stock_book,
    turnover,
    vol_scaled,
)
from lead_lag.portfolio.inputs import (
    calendar_lagged_signals,
    dimson_beta,
    full_panel_inputs,
)
from lead_lag.portfolio.performance import (
    STRESS,
    WINDOWS,
    attribution,
    drawdown,
    holm,
    summary_stats,
    stress_table,
    var_backtest,
    window_slice,
)

__all__ = [
    "STRESS", "WINDOWS", "attribution", "book_returns", "calendar_lagged_signals", "dimson_beta", "drawdown",
    "full_panel_inputs", "holm", "industry_book", "stock_book", "stress_table",
    "summary_stats", "turnover", "var_backtest", "vol_scaled", "window_slice",
]
