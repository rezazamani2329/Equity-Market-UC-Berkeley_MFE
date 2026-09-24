"""
Is the common/specific split a fact about the market, or about one arbitrary
choice of "leader"?

The decomposition rests on three judgement calls part 1 deliberately left as
parameters (``leaders.LeaderRule``): what makes a stock the leader, how many
leaders, and who counts as a follower.  If the leader-specific reversion only
appears when the leader is *the single largest by market cap*, it is a property
of that definition and a referee is right to be unconvinced.  If it survives
ranking by dollar volume, using a three-stock leader portfolio, and restricting
to the smaller half of followers, it is a property of the lead-lag relationship.

This module runs the whole part-2 pipeline - assemble, decompose, conditional
response - once per definition and lays the answers side by side.  It also
sweeps the two *decomposition-design* choices that are mine rather than part
1's: whether the market is partialled out, and the Dimson ``sync_lags`` that
addresses non-synchronous trading.  A result that flips when the market control
is added, or when the index is desynchronised, is telling you which artefact it
was.

Each row reports, for one variant:

    leader_turnover     month-to-month leader change rate.  A high rate means
                        the "leader" is two similar-sized firms trading places
                        and the coefficient partly measures ranking noise -
                        read this before trusting the rest of the row.
    median_specific     median across industries of the leader-specific
                        variance share (1 - R^2).  How idiosyncratic the
                        leader's move is under this definition.
    b_common / b_shock  the Fama-MacBeth conditional responses at lag 1.
    beta_diff / t_diff  the added predictability and its t-stat - the number
                        the whole section is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from lead_lag.data.leaders import (
    LeaderRule,
    assign_roles,
    follower_panel,
    leader_turnover,
)
from lead_lag.data.universe import Universe, eligible_daily
from lead_lag.shocks.decomposition import (
    DecompositionSpec,
    assemble_leader_frame,
    decompose_full_sample,
)
from lead_lag.shocks.regression import build_conditional_panel, conditional_response


@dataclass(frozen=True)
class LeaderVariant:
    """One named cell of the robustness grid.

    A variant is a (LeaderRule, DecompositionSpec) pair with a label.  Bundling
    both lets the same grid vary the leader definition (part 1's choices) and
    the decomposition design (mine) without two separate loops.
    """

    label: str
    rule: LeaderRule = field(default_factory=LeaderRule)
    spec: DecompositionSpec = field(default_factory=DecompositionSpec)


def default_variants() -> list[LeaderVariant]:
    """The grid the report should lead with.

    Row 1 is the base case.  Rows 2-4 vary the leader definition (part 1's three
    knobs); rows 5-6 vary the decomposition design (mine).  Each isolates one
    choice from the base case so a change in the result is attributable.
    """
    return [
        LeaderVariant("base: mktcap, 1 leader"),
        LeaderVariant(
            "leader by dollar volume",
            rule=LeaderRule(by="med_dollar_vol"),
        ),
        LeaderVariant(
            "3-stock leader portfolio",
            rule=LeaderRule(n_leaders=3),
        ),
        LeaderVariant(
            "smaller-half followers only",
            rule=LeaderRule(follower_max_size_pct=0.5),
        ),
        LeaderVariant(
            "no market control",
            spec=DecompositionSpec(include_market=False),
        ),
        LeaderVariant(
            "Dimson sync (+/-1 day)",
            spec=DecompositionSpec(sync_lags=1),
        ),
    ]


def run_variant(
    panel: pd.DataFrame,
    universe: Universe,
    factors: pd.DataFrame,
    variant: LeaderVariant,
    lag: int = 1,
    ret_col: str = "ret",
) -> dict[str, float | str | int]:
    """Full part-2 pipeline for one variant; returns one summary row.

    ``panel`` is the lagged daily panel (``information_set.attach_lagged``
    output with ``ff49``); ``universe`` and ``factors`` are part 1's.  The
    decomposition is the full-sample one - the robustness question is about the
    *existence and sign* of the split, for which the descriptive estimator is
    the right, and much cheaper, choice.
    """
    roles = assign_roles(universe, variant.rule)
    turn = leader_turnover(roles).attrs.get("overall_rate", float("nan"))

    eligible = eligible_daily(panel, universe)
    frame = assemble_leader_frame(
        eligible, roles, factors,
        ret_col=ret_col, market_col=variant.spec.market_col,
    )
    shocks = decompose_full_sample(frame, variant.spec)
    shares = shocks.variance_shares()

    fp = follower_panel(panel, roles, ret_col=ret_col).merge(
        factors[["date", variant.spec.market_col]], on="date", how="left"
    )
    cpanel = build_conditional_panel(fp, shocks, lag=lag, ret_col=ret_col)
    resp = conditional_response(
        cpanel, lag=lag, ret_col=ret_col, market_col=variant.spec.market_col
    )

    return {
        "variant": variant.label,
        "regressors": variant.spec.regressor_note(),
        "leader_turnover": float(turn),
        "n_industries": int(shocks.frame["ff49"].nunique()),
        "median_specific_share": float(shares["specific_share"].median()),
        "b_common": resp.b_common_fm,
        "t_common": resp.t_common_fm,
        "b_shock": resp.b_shock_fm,
        "t_shock": resp.t_shock_fm,
        "beta_diff": resp.beta_diff_fm,
        "t_diff": resp.t_diff_fm,
        "n_days": resp.n_days,
    }


def run_leader_robustness(
    panel: pd.DataFrame,
    universe: Universe,
    factors: pd.DataFrame,
    variants: list[LeaderVariant] | None = None,
    lag: int = 1,
    ret_col: str = "ret",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every variant and return the comparison table.

    One row per variant, indexed by label, in the order given.  The base case
    is first; read every other row as a deviation from it.  The columns to scan
    are ``t_diff`` (does the split still add predictability) and its sign
    agreement with the base case - a variant that keeps ``b_common > 0`` and
    ``b_shock < 0`` has reproduced the effect, one that collapses them has not.
    """
    variants = variants or default_variants()
    rows = []
    for v in variants:
        if verbose:
            print(f"leader robustness: {v.label} ...", flush=True)
        try:
            row = run_variant(panel, universe, factors, v, lag=lag, ret_col=ret_col)
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            # A variant that cannot be estimated (e.g. an alternative LeaderRule
            # that thins the frame below min_obs) must not take the whole grid
            # down with it — the table exists for the *comparison*, and a blank,
            # labelled row is itself the finding. The error is kept in `status`
            # so the failing variant names itself.
            if verbose:
                print(f"  -> skipped ({type(exc).__name__}: {exc})", flush=True)
            row = {"variant": v.label, "status": f"{type(exc).__name__}: {exc}"}
        rows.append(row)

    out = pd.DataFrame(rows).set_index("variant")
    cols = [c for c in out.columns if c != "status"] + ["status"]  # status last
    return out[cols]
