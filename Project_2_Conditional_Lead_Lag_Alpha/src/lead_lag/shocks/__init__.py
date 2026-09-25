"""
Part 2 — Econometrics & Shock Identification (not part 1; owned by teammate 2).

What part 1 hands over:
    leaders.leader_returns(daily, roles)   -> date, ff49, leader_ret
    industry_map.industry_returns(panel)   -> date, ff49, ind_ret, n_stocks
    wrds_fetch.load_or_fetch_factors_daily -> date, mktrf, smb, hml, rmw, cma, umd, rf

The decomposition this part owns, in the project's own terms: split the
leader's daily return into the piece explained by common industry and factor
movement and the piece specific to the leader,

    r_{L,t} = alpha + beta_m · mkt_t + beta_j · ind_ret_{j,t}^{ex-leader} + u_{L,t}

with `ind_ret^{ex-leader}` excluding the leader itself — otherwise the leader
is on both sides and `u` is mechanically shrunk toward zero.  Part 1's
`industry_returns` weights by lagged market cap and includes every eligible
stock, so building the ex-leader version means re-running it on the panel with
the leader's rows removed.  Flagged here because it is the easiest thing in
the project to get wrong and the hardest to notice afterwards.

Public API
----------
    decomposition   ex_leader_industry_returns, assemble_leader_frame,
                    decompose_full_sample, decompose_rolling,
                    classify_shock_day, follower_merge_table,
                    DecompositionSpec, LeaderShocks
    regression      build_conditional_panel, conditional_response,
                    conditional_horizon_profile, nw_lags_overlap
    leader_robustness  run_leader_robustness, default_variants, LeaderVariant

`follower_merge_table` is the hand-off point for whoever runs the conditional
subsample test: it takes either estimator's `LeaderShocks` and returns the
`(date, ff49)`-keyed `common_shock` / `leader_specific_shock` / `shock_type`
table, ready for `follower_panel.merge(..., on=["date", "ff49"])` and then a
split on `shock_type` before re-running the IC / horizon-IC / quantile-
portfolio pipeline on each subsample.
"""

from lead_lag.shocks.decomposition import (
    DecompositionSpec,
    LeaderShocks,
    assemble_leader_frame,
    classify_shock_day,
    decompose_full_sample,
    decompose_rolling,
    ex_leader_industry_returns,
    follower_merge_table,
)
from lead_lag.shocks.leader_robustness import (
    LeaderVariant,
    default_variants,
    run_leader_robustness,
    run_variant,
)
from lead_lag.shocks.regression import (
    ConditionalResult,
    build_conditional_panel,
    conditional_horizon_profile,
    conditional_response,
    lag_leader_components,
    nw_lags_overlap,
    nw_lags_rule,
)

__all__ = [
    "DecompositionSpec",
    "LeaderShocks",
    "ex_leader_industry_returns",
    "assemble_leader_frame",
    "decompose_full_sample",
    "decompose_rolling",
    "classify_shock_day",
    "follower_merge_table",
    "ConditionalResult",
    "build_conditional_panel",
    "conditional_response",
    "conditional_horizon_profile",
    "lag_leader_components",
    "nw_lags_rule",
    "nw_lags_overlap",
    "LeaderVariant",
    "default_variants",
    "run_variant",
    "run_leader_robustness",
]
