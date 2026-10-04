"""
Part 5 — Robustness, Final Report & Submission (owner: Paraj Goyal, part 5).

Transaction costs, microstructure checks, out-of-sample tests, and the final
integration of report, figures, tables and code.

    costs         net-of-cost returns and breakeven cost from Part 4's daily
                  book returns

The decision rule was fixed before the numbers were run (appendix/chatgpt_p5.md):
"do not implement" is overturned only if a book keeps a positive net return at
the 2007-2024 median quoted half-spread with zero market impact.

The robustness handles part 1 deliberately left in place, each a one-line
change rather than a re-implementation. They need the stock-level panel:

    UniverseRules(min_price=...)          the $5 screen is the single filter
                                          most likely to be driving a
                                          short-horizon reversal result
    UniverseRules(min_dollar_vol_pct=...) does the alpha survive in liquid names
    LeaderRule(by="med_dollar_vol")       is "leader" a size artefact
    LeaderRule(n_leaders=3)               one noisy stock vs a leader portfolio
    adjusted_daily_returns(repair=False)  does the Shumway substitute matter
    DailyReturns.repaired                 drop the repaired rows entirely
    ExtremeReturnCheck                    the winsorisation decision, with the
                                          affected row count already measured
"""

from lead_lag.robustness.costs import ROUND_TRIP, breakeven_bp, net_of_cost

__all__ = ["ROUND_TRIP", "breakeven_bp", "net_of_cost"]
