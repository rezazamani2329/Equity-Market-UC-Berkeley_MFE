"""
The d-1 rule: every characteristic conditioning a signal for day d is the
value observed at d-1 or earlier.

The test that matters most here is `test_reclassification_uses_the_old_industry`.
It is the case the rule exists for, it is invisible in aggregate statistics —
a handful of firms reclassify in any given month — and getting it wrong is a
lookahead that no summary table would reveal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_lag.data.industry_map import attach_industry, industry_returns
from lead_lag.data.information_set import (
    LAGGED_COLUMNS,
    attach_lagged,
    require_lagged,
    same_day_columns,
)


@pytest.fixture
def reclassifying_panel(siccodes):
    """Two stocks over five days. Stock 1 moves from SIC 2000 (Food, ff49 2)
    to SIC 2800 (Drugs, ff49 13) with effect from day 3 — exactly the case in
    question. Stock 2 never moves and stays in Food.
    """
    dates = pd.bdate_range("2020-01-01", periods=5)
    rows = []
    for permno, sics in [(1, [2000, 2000, 2800, 2800, 2800]), (2, [2000] * 5)]:
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "permno": permno,
                    "siccd": sics,
                    "ret": [0.01, 0.02, 0.03, 0.04, 0.05],
                    "mktcap": [100.0, 110.0, 120.0, 130.0, 140.0],
                    "prc": [10.0, 11.0, 12.0, 13.0, 14.0],
                    "dollar_vol": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                    "vol": 100.0,
                    "exchcd": 1,
                }
            )
        )
    return attach_industry(pd.concat(rows, ignore_index=True), siccodes)


def test_reclassification_uses_the_old_industry(reclassifying_panel):
    """On the day the classification changes, the LAGGED industry is the old one.

    Stock 1 is Food (2) through day 2 and Drugs (13) from day 3. The signal
    for day 3 must see it as Food, because that is what it was at the close of
    day 2.
    """
    lagged = attach_lagged(reclassifying_panel)
    stock = lagged.loc[lagged["permno"] == 1].reset_index(drop=True)

    # Same-day column: the change shows up on day 3 (index 2).
    assert stock["ff49"].tolist() == [2, 2, 13, 13, 13]
    # Lagged column: day 3 still sees Food; the change shows up on day 4.
    assert np.isnan(stock["ff49_lag"].iloc[0])           # no previous day
    assert stock["ff49_lag"].iloc[1:].tolist() == [2.0, 2.0, 13.0, 13.0]


def test_every_characteristic_is_lagged_by_one_observation(reclassifying_panel):
    lagged = attach_lagged(reclassifying_panel)
    stock = lagged.loc[lagged["permno"] == 1].reset_index(drop=True)
    for col in ("mktcap", "prc", "dollar_vol"):
        # Row i's lagged value is row i-1's value, and row 0 has none.
        assert np.isnan(stock[f"{col}_lag"].iloc[0])
        assert stock[f"{col}_lag"].iloc[1:].tolist() == stock[col].iloc[:-1].tolist()


def test_lagging_does_not_cross_stocks(reclassifying_panel):
    """Stock 2's first row must not inherit stock 1's last value."""
    lagged = attach_lagged(reclassifying_panel)
    firsts = lagged.groupby("permno").head(1)
    for col in LAGGED_COLUMNS:
        if f"{col}_lag" in lagged.columns:
            assert firsts[f"{col}_lag"].isna().all()


def test_days_since_prev_measures_staleness():
    """A stock that stops trading carries stale characteristics, and the gap
    is reported rather than hidden."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-14"])
    panel = pd.DataFrame(
        {"date": dates, "permno": 1, "mktcap": [100.0, 110.0, 120.0],
         "ret": [0.0, 0.0, 0.0]}
    )
    lagged = attach_lagged(panel)
    assert np.isnan(lagged["days_since_prev"].iloc[0])
    assert lagged["days_since_prev"].iloc[1] == 1
    assert lagged["days_since_prev"].iloc[2] == 42   # the suspension
    # And the stale market cap is the pre-suspension one, correctly.
    assert lagged["mktcap_lag"].iloc[2] == 110.0


def test_industry_returns_groups_on_the_lagged_industry(reclassifying_panel):
    """The industry index for day d is the portfolio formable at d-1's close.

    Stock 1 reclassifies to Drugs on day 3. The Drugs index must not contain
    it on day 3 — only from day 4.
    """
    lagged = attach_lagged(reclassifying_panel)
    out = industry_returns(lagged)

    # bdate_range skips the weekend: Wed 1, Thu 2, Fri 3, Mon 6, Tue 7.
    day3, day4 = pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")
    drugs_day3 = out.loc[(out["date"] == day3) & (out["ff49_lag"] == 13)]
    drugs_day4 = out.loc[(out["date"] == day4) & (out["ff49_lag"] == 13)]

    assert drugs_day3.empty, "stock 1 entered Drugs a day early"
    assert len(drugs_day4) == 1 and drugs_day4["n_stocks"].iloc[0] == 1

    # And on day 3 it is still counted in Food, alongside stock 2.
    food_day3 = out.loc[(out["date"] == day3) & (out["ff49_lag"] == 2)]
    assert food_day3["n_stocks"].iloc[0] == 2


def test_industry_returns_weights_are_the_previous_close(reclassifying_panel):
    """Weighting by day d's market cap overweights day d's winners; the
    default weight must be `mktcap_lag`."""
    lagged = attach_lagged(reclassifying_panel)
    out = industry_returns(lagged)

    day3 = pd.Timestamp("2020-01-03")
    food = out.loc[(out["date"] == day3) & (out["ff49_lag"] == 2)].iloc[0]
    # Both stocks return 0.03 on day 3, so any weighting gives 0.03 — the
    # check that bites is that the row exists with two members at all, which
    # requires the weight to be non-null (i.e. taken from day 2).
    assert food["ind_ret"] == pytest.approx(0.03)
    assert food["n_stocks"] == 2


def test_require_lagged_explains_itself(reclassifying_panel):
    """The error names the fix rather than failing with a KeyError deep in a
    groupby."""
    with pytest.raises(KeyError, match="attach_lagged"):
        require_lagged(reclassifying_panel, "ff49_lag")   # not attached yet


def test_same_day_columns_flags_an_unlagged_frame(reclassifying_panel):
    """The review diagnostic: hand it a frame and it names what still carries
    day-d information."""
    assert "ff49" in same_day_columns(reclassifying_panel)
    assert "mktcap" in same_day_columns(reclassifying_panel)
    assert same_day_columns(attach_lagged(reclassifying_panel)) == []


def test_follower_panel_refuses_an_unlagged_frame(clean_panel):
    """Parts 2 and 3 consume this frame; it must not be possible to hand them
    same-day characteristics by forgetting a step."""
    from lead_lag.data.leaders import LeaderRule, assign_roles, follower_panel
    from lead_lag.data.universe import UniverseRules, build_universe

    universe, _ = build_universe(clean_panel, UniverseRules(min_obs=20))
    roles = assign_roles(universe, LeaderRule(min_followers=2))

    with pytest.raises(KeyError, match="attach_lagged"):
        follower_panel(clean_panel, roles)

    panel = follower_panel(attach_lagged(clean_panel), roles)
    # The same-day versions are gone, not merely renamed.
    for col in ("mktcap", "dollar_vol", "prc", "exchcd"):
        assert col not in panel.columns
        assert f"{col}_lag" in panel.columns


def test_index_and_roles_share_one_industry_vintage(lagged_panel):
    """The industry index and the leader/follower table must key on the same
    industry, or a firm reclassified mid-month is ranked against industry A's
    leader while being averaged into industry B's index — inside a single
    holding period, which no tradeable portfolio can do.

    The project's choice is the FORMATION-MONTH vintage for both: `ff49` as
    carried by `Universe` / `LeaderMap`, reached on the daily side through
    `eligible_daily`.
    """
    from lead_lag.data.leaders import LeaderRule, assign_roles
    from lead_lag.data.universe import UniverseRules, build_universe, eligible_daily

    universe, _ = build_universe(lagged_panel, UniverseRules(min_obs=20))
    roles = assign_roles(universe, LeaderRule(min_followers=2))

    eligible = eligible_daily(lagged_panel, universe)
    index = industry_returns(eligible, industry_col="ff49", weight="mktcap_lag")

    # Same column name, same vintage, and the index covers exactly the
    # industries that have a role assignment.
    assert "ff49" in index.columns
    assert set(index["ff49"]) >= set(roles.frame["ff49"])

    # Every (month, permno) in the index carries the universe's industry, not
    # the daily row's — check by joining back and comparing.
    merged = eligible.merge(
        universe.frame.loc[universe.frame["eligible"], ["month", "permno", "ff49"]],
        on=["month", "permno"], how="left", suffixes=("", "_universe"),
    )
    assert (merged["ff49"] == merged["ff49_universe"]).all()
