# Part 1 → the rest of the team

What part 1 produces, what it guarantees, and the three ways to misuse it.

Everything below is written by `notebooks/part1_data_universe.ipynb` §10 into
`results/`. Read the parquet; do not re-derive the universe.

## The artefacts

| File | For | Key | Contents |
|---|---|---|---|
| `p1_universe.parquet` | 4 | `(month, permno)` | monthly eligibility + the formation statistics behind it |
| `p1_roles.parquet` | 2, 3 | `(month, permno)` | `role` ∈ {leader, follower}, `rank_in_industry`, `size_pct` |
| `p1_industry_returns.parquet` | 2 | `(date, ff49)` | value-weighted industry return, lagged weights |
| `p1_follower_panel.parquet` | 2, 3 | `(date, permno)` | follower return + its industry's same-day leader return |
| `p1_horizon_profile.csv` | 3, 5 | `lag` | the unconditional baseline — the numbers to beat |
| `p1_quality_report.csv` | 5 | — | every data-quality finding, for the cleaning appendix |

## What part 1 guarantees

1. **Point-in-time.** Every universe and role decision for month *m* is made
   from data observed through the end of month *m−1*. `formation_window_stats`
   shifts each statistic forward one month, and a test pins it. A stock is
   December's leader because it was largest through November.

2. **Point-in-time industries too.** `siccd` comes from the CRSP name row valid
   on the observation date, not the header code. A firm that became a software
   company in 2010 is a manufacturer in its 2005 rows.

3. **No survivorship bias.** Delisting returns are merged in, including events
   on days the stock did not trade, which are appended as their own rows.
   Missing delisting returns on performance-related codes get Shumway's
   substitute, flagged in `repaired` and counted in the report.

4. **The `(date, permno)` key is unique.** Enforced by `DuplicateKeyCheck`,
   which the notebook asserts clean. Every join downstream can rely on it.

5. **Schemas are checked at the boundary.** A `Dataset` that exists is
   schema-valid; a frame missing a column fails where it is built, not three
   functions later.

## Three ways to misuse it

**1. Decomposing the leader against an industry return that contains the leader.**
`p1_industry_returns.parquet` is built from *every* eligible stock, leader
included. The leader is a large share of a value-weighted industry index by
construction — it was picked for being the largest. Regressing the leader's
return on that index puts it on both sides, and the leader-specific residual
`u` shrinks toward zero exactly where part 2 needs it to be informative.

Rebuild it ex-leader:

```python
ex_leader = panel.merge(
    roles.frame.loc[roles.frame["role"] == "leader", ["month", "permno"]],
    on=["month", "permno"], how="left", indicator=True,
)
ex_leader = ex_leader.loc[ex_leader["_merge"] == "left_only"]
ind_ret_ex = industry_returns(ex_leader)
```

**2. Dropping missing returns before lagging.** `DailyReturns` keeps rows where
`ret` is NaN on purpose. `build_lags` shifts by *rows*, which is only correct
when the panel has one row per trading day per stock. Drop the NaNs first and
day *t* gets lined up against a day that is not *t−k* — silently, and worst for
exactly the illiquid stocks whose returns are most autocorrelated. Shift first,
drop second.

**3. Cumulating the signal without raising the Newey-West lag count.** The
baseline uses single-day leader returns, so observations do not overlap and the
standard lag rule is fine. A signal cumulated over a *w*-day window makes
consecutive observations share *w−1* days, and the lag truncation has to grow
with *w* or the t-statistics are overstated.

## Robustness handles already wired

Each is a one-line change, not a re-implementation:

| Change | Question it answers |
|---|---|
| `UniverseRules(min_price=...)` | is the effect bid-ask bounce in cheap stocks |
| `UniverseRules(min_dollar_vol_pct=...)` | does it survive in tradeable names |
| `UniverseRules(min_nyse_size_pct=...)` | is it a microcap effect |
| `LeaderRule(by="med_dollar_vol")` | is "leader" a size artefact |
| `LeaderRule(n_leaders=3)` | one noisy stock vs a leader portfolio |
| `LeaderRule(follower_max_size_pct=0.5)` | is it concentrated in the smallest followers |
| `adjusted_daily_returns(repair=False)` | does the Shumway substitute matter |
| filter on `DailyReturns.repaired` | drop the substituted rows entirely |

## Diagnostics to check before trusting a result

- **Leader turnover** (`leaders.leader_turnover`). Real leaders rarely change.
  A high rate means the size ranking is decided by noise — two firms of nearly
  equal size trading places — and the coefficient partly measures that.
- **The quintile profile** (`baseline.quintile_sort`). A regression coefficient
  assumes linearity. If the bins are flat except the extreme one, the effect is
  a handful of large leader moves, not a general relationship.
- **The by-year coefficient chart.** A mean that comes entirely from 2008 is a
  different claim from one that is stable.
- **Both standard errors.** Pooled-clustered and Fama-MacBeth should broadly
  agree. When they do not, trust Fama-MacBeth.
