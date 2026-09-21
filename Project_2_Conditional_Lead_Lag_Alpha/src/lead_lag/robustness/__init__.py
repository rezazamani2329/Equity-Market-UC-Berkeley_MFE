"""
Part 5 — Robustness, Final Report & Submission (not part 1; owned by teammate 5).

Transaction costs, microstructure checks, out-of-sample tests, and the final
integration of report, figures, tables and code.

The robustness handles part 1 deliberately left in place, each a one-line
change rather than a re-implementation:

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
