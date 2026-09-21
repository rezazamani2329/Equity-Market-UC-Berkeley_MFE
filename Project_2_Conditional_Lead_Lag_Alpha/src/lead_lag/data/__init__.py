"""
Part 1 — Data & Universe.

Pipeline order, which is also the order to read the modules in:

    wrds_connection   is WRDS reachable, with whose credentials
    wrds_fetch        raw pulls + the per-year parquet cache
    daily_returns     delisting events folded into the daily return series
    industry_map      SIC -> Fama-French 49, point in time
    universe          monthly eligibility: price, size, liquidity, industry
    leaders           leader / follower assignment per industry-month
    quality_checks    the audit vocabulary and the daily-panel checks
    typed_dataset     the schema contract every dataset above is built on

`../baseline.py` then runs the unconditional lead-lag test on the result.
"""
