"""
The baseline lead-lag test, checked against a planted effect.

`fake_crsp.make_panel` builds followers whose return is exactly
`beta * leader_return[t-1]` plus independent noise.  So the estimator has a
right answer, and these tests assert it finds it — at lag 1, and not at lag 2.
That pair is what catches an off-by-one in the shifting, which is the single
most likely bug in this module and the one that would otherwise show up as a
plausible-looking result.
"""

from __future__ import annotations

import numpy as np
import pytest

from lead_lag.baseline import (
    baseline_test,
    build_lags,
    fama_macbeth,
    horizon_profile,
    pooled_regression,
    quintile_sort,
)
from lead_lag.data.leaders import LeaderRule, assign_roles, follower_panel
from lead_lag.data.universe import UniverseRules, build_universe

PLANTED_BETA = 0.30


@pytest.fixture(scope="module")
def panel(lagged_panel):
    """Follower-days with their leader's same-day return attached.

    Built from `lagged_panel`, not `clean_panel`: every characteristic the
    regression panel carries is a d-1 value, and `follower_panel` refuses a
    frame that has not been through `attach_lagged`.
    """
    universe, _ = build_universe(lagged_panel, UniverseRules(min_obs=20))
    roles = assign_roles(universe, LeaderRule(min_followers=2))
    return follower_panel(lagged_panel, roles)


def test_build_lags_shifts_within_stock(panel):
    """A stock's first observation has no lag, and the lag comes from its own
    history rather than the previous stock's last row."""
    lagged = build_lags(panel, lag=1)
    first_rows = lagged.groupby("permno").head(1)
    assert first_rows["leader_lag"].isna().all()
    assert first_rows["own_lag"].isna().all()

    # And the second row's lag equals the first row's leader return.
    one = lagged.loc[lagged["permno"] == lagged["permno"].iloc[0]].head(2)
    assert one["leader_lag"].iloc[1] == pytest.approx(one["leader_ret"].iloc[0])


def test_pooled_recovers_the_planted_beta(panel):
    lagged = build_lags(panel, lag=1)
    fit = pooled_regression(lagged, market_col=None)
    assert fit.params["leader_lag"] == pytest.approx(PLANTED_BETA, abs=0.03)
    assert fit.tvalues["leader_lag"] > 5


def test_fama_macbeth_agrees_with_pooled(panel):
    lagged = build_lags(panel, lag=1)
    beta, t_stat, series = fama_macbeth(lagged)
    assert beta == pytest.approx(PLANTED_BETA, abs=0.05)
    assert t_stat > 5
    # The daily coefficient series is returned for the report's stability plot.
    assert len(series) > 100


def test_effect_is_absent_at_lag_two(panel):
    """The planted effect is one day long.  A non-zero estimate at lag 2 would
    mean the shifting is wrong, not that the market is interesting."""
    result = baseline_test(panel, lag=2, market_col=None)
    assert abs(result.beta_pooled) < 0.05
    assert abs(result.t_fm) < 4


def test_market_control_does_not_move_the_estimate(panel, factors):
    """The fake factors are independent noise, so adding `mktrf` must leave
    the coefficient essentially unchanged.  If it moves, the merge is wrong."""
    merged = panel.merge(factors[["date", "mktrf"]], on="date", how="left")
    without = baseline_test(panel, lag=1, market_col=None)
    with_mkt = baseline_test(merged, lag=1, market_col="mktrf")
    assert with_mkt.beta_pooled == pytest.approx(without.beta_pooled, abs=0.01)


def test_horizon_profile_shape(panel):
    profile = horizon_profile(panel, lags=[1, 2, 3], market_col=None, verbose=False)
    assert list(profile.index) == [1, 2, 3]
    assert profile.loc[1, "beta_pooled"] > profile.loc[3, "beta_pooled"]
    assert {"beta_pooled", "t_pooled", "beta_fm", "t_fm", "n_obs"} <= set(profile.columns)


def test_quintile_sort_is_monotone(panel):
    """With a linear planted effect the bin means must increase in the bin."""
    bins = quintile_sort(panel, lag=1)
    ordered = bins.loc[bins.index > 0].sort_index()["mean_ret"].to_numpy()
    assert np.all(np.diff(ordered) > -1e-4)   # monotone up to sampling noise
    # The top-minus-bottom row is positive and significant.
    spread = bins.loc[-1]
    assert spread["mean_ret"] > 0
    assert spread["t_stat"] > 5
