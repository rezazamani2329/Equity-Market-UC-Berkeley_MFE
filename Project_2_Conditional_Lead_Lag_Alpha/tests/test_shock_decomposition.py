"""
Part 2 against a planted answer.

``fake_shocks.make_shock_panel`` builds a panel whose leader move splits into a
known common component and a known specific shock, and whose followers continue
on the lagged common piece (``+gamma_cont``) and revert on the lagged specific
piece (``-gamma_rev``).  These tests assert that the decomposition recovers that
structure and, just as importantly, that it does NOT find an effect where none
was planted (the lag-2 placebo) and cannot peek ahead (the rolling guards).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fake_shocks import make_shock_panel

from lead_lag.baseline import baseline_test
from lead_lag.data.daily_returns import adjusted_daily_returns
from lead_lag.data.industry_map import attach_industry, industry_returns
from lead_lag.data.information_set import attach_lagged
from lead_lag.data.leaders import LeaderRule, assign_roles, follower_panel
from lead_lag.data.universe import UniverseRules, build_universe, eligible_daily
from lead_lag.data.wrds_fetch import CRSPDaily, CRSPDelist
from lead_lag.shocks.decomposition import (
    DecompositionSpec,
    LeaderShocks,
    assemble_leader_frame,
    decompose_full_sample,
    decompose_rolling,
    ex_leader_industry_returns,
)
from lead_lag.shocks.leader_robustness import default_variants, run_leader_robustness
from lead_lag.shocks.regression import (
    build_conditional_panel,
    conditional_response,
    lag_leader_components,
    nw_lags_overlap,
    nw_lags_rule,
)
from conftest import MINI_SICCODES


# --------------------------------------------------------------------------- #
# Build the part-2 pipeline once for the whole module                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pipeline():
    raw, factors, truth, dgp = make_shock_panel()
    daily = CRSPDaily.from_raw(raw)
    empty = CRSPDelist.from_raw(
        pd.DataFrame({"permno": [], "dlstdt": [], "dlstcd": [], "dlret": [], "dlretx": []})
    )
    returns, _ = adjusted_daily_returns(daily, empty)
    panel = attach_lagged(attach_industry(returns.frame, MINI_SICCODES))
    universe, _ = build_universe(panel, UniverseRules())
    roles = assign_roles(universe, LeaderRule())
    eligible = eligible_daily(panel, universe)
    frame = assemble_leader_frame(eligible, roles, factors)
    fp = follower_panel(panel, roles).merge(
        factors[["date", "mktrf"]], on="date", how="left"
    )
    return {
        "panel": panel, "universe": universe, "roles": roles,
        "eligible": eligible, "factors": factors, "frame": frame,
        "fp": fp, "truth": truth, "dgp": dgp,
    }


# --------------------------------------------------------------------------- #
# The ex-leader index                                                         #
# --------------------------------------------------------------------------- #
def test_ex_leader_index_removes_the_leader(pipeline):
    """The ex-leader index must contain strictly fewer stocks than the full
    index, and must decorrelate the leader from its own industry benchmark."""
    elig, roles = pipeline["eligible"], pipeline["roles"]
    full = industry_returns(elig, industry_col="ff49", weight="mktcap_lag")
    ex = ex_leader_industry_returns(elig, roles)

    merged = full.merge(ex, on=["date", "ff49"], how="inner")
    # every industry-day loses at least the one leader
    assert (merged["n_stocks_exl"] < merged["n_stocks"]).all()

    # the leader inclusive index mechanically tracks the leader more than the
    # ex-leader one does — that gap is the whole reason the function exists.
    from lead_lag.data.leaders import leader_returns
    lr = leader_returns(elig, roles)
    a = lr.merge(full, on=["date", "ff49"]).merge(ex, on=["date", "ff49"])
    corr_full = a["leader_ret"].corr(a["ind_ret"])
    corr_ex = a["leader_ret"].corr(a["ind_ret_exl"])
    assert corr_ex < corr_full


# --------------------------------------------------------------------------- #
# The residual identity                                                       #
# --------------------------------------------------------------------------- #
def test_common_plus_shock_equals_leader_return(pipeline):
    shocks = decompose_full_sample(pipeline["frame"], DecompositionSpec())
    err = (shocks.frame["common"] + shocks.frame["shock"] - shocks.frame["leader_ret"]).abs()
    assert err.max() < 1e-10


def test_lagged_components_preserve_identity(pipeline):
    shocks = decompose_full_sample(pipeline["frame"], DecompositionSpec())
    lagged = lag_leader_components(shocks, lag=1).dropna()
    err = (lagged["common_lag"] + lagged["shock_lag"] - lagged["leader_ret_lag"]).abs()
    assert err.max() < 1e-10


# --------------------------------------------------------------------------- #
# Recovery of the planted decomposition                                       #
# --------------------------------------------------------------------------- #
def test_recovers_planted_shock(pipeline):
    """Estimated shock should track the planted latent shock (imperfectly — the
    ex-leader index is a noisy proxy for the industry factor)."""
    shocks = decompose_full_sample(pipeline["frame"], DecompositionSpec())
    roles, truth = pipeline["roles"], pipeline["truth"]

    lead_ids = roles.frame.loc[roles.frame["role"] == "leader", ["month", "permno", "ff49"]]
    tr = truth.copy()
    tr["month"] = tr["date"].values.astype("datetime64[M]")
    tr = tr.merge(lead_ids, on=["month", "permno"], how="inner")
    cmp = shocks.frame.merge(
        tr[["date", "ff49", "common_true", "shock_true"]], on=["date", "ff49"]
    )
    assert cmp["shock"].corr(cmp["shock_true"]) > 0.7
    assert cmp["common"].corr(cmp["common_true"]) > 0.7


def test_specific_share_is_sane(pipeline):
    shocks = decompose_full_sample(pipeline["frame"], DecompositionSpec())
    shares = shocks.variance_shares()["specific_share"]
    # a genuine mix of common and specific variance: neither near 0 nor near 1
    assert (shares.between(0.2, 0.8)).all()


# --------------------------------------------------------------------------- #
# The sign flip — the whole point                                             #
# --------------------------------------------------------------------------- #
def test_sign_flip_common_positive_shock_negative(pipeline):
    shocks = decompose_full_sample(pipeline["frame"], DecompositionSpec())
    cp = build_conditional_panel(pipeline["fp"], shocks, lag=1)
    r = conditional_response(cp, lag=1, market_col="mktrf")

    # continuation on the common component, reversion on the specific shock
    assert r.b_common_fm > 0 and r.t_common_fm > 3
    assert r.b_shock_fm < 0 and r.t_shock_fm < -3
    # the split adds predictability over the undifferentiated leader move
    assert r.beta_diff_fm > 0 and r.t_diff_fm > 3


def test_decomposition_beats_undifferentiated_baseline(pipeline):
    """The baseline sees one coefficient on the leader's total move; because the
    planted reversion is larger than the planted continuation, that single
    number is negative and hides the positive common response the split
    recovers."""
    fp = pipeline["fp"]
    b = baseline_test(fp, lag=1, market_col="mktrf")
    shocks = decompose_full_sample(pipeline["frame"], DecompositionSpec())
    r = conditional_response(
        build_conditional_panel(fp, shocks, lag=1), lag=1, market_col="mktrf"
    )
    # the common response is on the opposite side of zero from the pooled
    # baseline — the thing the baseline could not have told you.
    assert np.sign(r.b_common_fm) != np.sign(b.beta_fm)


def test_lag2_placebo_is_weak(pipeline):
    """Only a one-day effect was planted, so at lag 2 both responses collapse
    toward zero — the part-2 analogue of the baseline's off-by-one placebo."""
    shocks = decompose_full_sample(pipeline["frame"], DecompositionSpec())
    r1 = conditional_response(
        build_conditional_panel(pipeline["fp"], shocks, lag=1), lag=1, market_col="mktrf"
    )
    r2 = conditional_response(
        build_conditional_panel(pipeline["fp"], shocks, lag=2), lag=2, market_col="mktrf"
    )
    assert abs(r2.b_common_fm) < abs(r1.b_common_fm)
    assert abs(r2.b_shock_fm) < abs(r1.b_shock_fm)


# --------------------------------------------------------------------------- #
# Rolling / point-in-time guards                                              #
# --------------------------------------------------------------------------- #
def test_rolling_matches_full_sample_share(pipeline):
    full = decompose_full_sample(pipeline["frame"], DecompositionSpec())
    roll = decompose_rolling(pipeline["frame"], DecompositionSpec(window=252, min_obs=60))
    mf = full.variance_shares()["specific_share"].median()
    mr = roll.variance_shares()["specific_share"].median()
    assert abs(mf - mr) < 0.15  # the trailing estimator drifts a little, not a lot


def test_rolling_refuses_future_leads(pipeline):
    with pytest.raises(ValueError, match="future"):
        decompose_rolling(pipeline["frame"], DecompositionSpec(sync_lags=1))


# --------------------------------------------------------------------------- #
# Regression-design helpers                                                    #
# --------------------------------------------------------------------------- #
def test_overlap_lag_count_covers_the_window():
    # non-overlapping: the automatic rule
    assert nw_lags_overlap(5000, window=1) == nw_lags_rule(5000)
    # overlapping by w-1: must be at least w-1
    assert nw_lags_overlap(5000, window=10) >= 9


# --------------------------------------------------------------------------- #
# Robustness of leader definitions                                            #
# --------------------------------------------------------------------------- #
def test_leader_robustness_reproduces_the_split(pipeline):
    table = run_leader_robustness(
        pipeline["panel"], pipeline["universe"], pipeline["factors"],
        variants=default_variants()[:4],  # the leader-definition rows
        verbose=False,
    )
    assert len(table) == 4
    # every leader definition keeps the sign split
    assert (table["b_common"] > 0).all()
    assert (table["b_shock"] < 0).all()
