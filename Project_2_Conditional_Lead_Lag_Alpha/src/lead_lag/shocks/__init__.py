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
"""
