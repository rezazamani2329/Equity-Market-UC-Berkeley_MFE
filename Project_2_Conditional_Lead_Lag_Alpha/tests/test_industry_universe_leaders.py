"""
Industry mapping, the universe filters, and the leader/follower assignment.

The tests that matter most here are the point-in-time ones. A lookahead bug
produces beautiful results and no error message, so it has to be pinned by a
test rather than by care.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_lag.data.industry_map import (
    OTHER_INDUSTRY,
    assign_ff49,
    attach_industry,
    industry_returns,
    parse_siccodes,
)
from lead_lag.data.information_set import attach_lagged
from lead_lag.data.leaders import (
    LeaderRule,
    assign_roles,
    follower_panel,
    leader_returns,
    leader_turnover,
)
from lead_lag.data.quality_checks import DuplicateKeyCheck, Severity, daily_panel_audit
from lead_lag.data.universe import (
    UniverseRules,
    build_universe,
    eligible_daily,
    formation_window_stats,
    monthly_stock_stats,
)
from lead_lag.data.wrds_fetch import CRSPDaily


# --------------------------------------------------------------------- SIC

def test_assign_ff49_maps_ranges_and_gaps(siccodes):
    codes = pd.Series([100, 199, 2000, 2800, 9999, 0])
    out = assign_ff49(codes, siccodes)
    assert out.tolist() == [1, 1, 2, 13, OTHER_INDUSTRY, OTHER_INDUSTRY]


def test_assign_ff49_is_inclusive_at_both_ends(siccodes):
    """A code equal to `sic_lo` or `sic_hi` belongs to the range."""
    assert assign_ff49(pd.Series([2800]), siccodes).iloc[0] == 13
    assert assign_ff49(pd.Series([2899]), siccodes).iloc[0] == 13
    assert assign_ff49(pd.Series([2900]), siccodes).iloc[0] == OTHER_INDUSTRY


def test_parse_siccodes_reads_the_french_format(tmp_path):
    """The parser handles the real file's indentation convention."""
    text = (
        " 1 Agric  Agriculture\n"
        "          0100-0199 Agric production - crops\n"
        "          0200-0299 Agric production - livestock\n"
        " 2 Food   Food Products\n"
        "          2000-2009 Food and kindred products\n"
    )
    path = tmp_path / "Siccodes49.txt"
    path.write_text(text)
    table = parse_siccodes(path)
    assert len(table) == 3
    assert set(table["ff49"]) == {1, 2}
    assert table.loc[table["sic_lo"] == 200, "ff49"].iloc[0] == 1


def test_industry_returns_lags_the_weights(siccodes):
    """Weighting day t by day t's own market cap is a same-day information
    leak; the default weight must be the previous close's market cap."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "permno": [1, 1, 1, 2, 2, 2],
            "ff49": 1,
            "ret": [0.10, 0.10, 0.10, -0.10, -0.10, -0.10],
            # Stock 1's cap jumps on the last day; with same-day weights that
            # would drag the index up, with lagged weights it must not.
            "mktcap": [100.0, 100.0, 1000.0, 100.0, 100.0, 100.0],
        }
    ).sort_values(["permno", "date"]).reset_index(drop=True)
    panel = attach_lagged(panel)

    lagged = industry_returns(panel)                        # mktcap_lag, ff49_lag
    same_day = industry_returns(panel, weight="mktcap", industry_col="ff49")

    last = dates[-1]
    last_lagged = lagged.loc[lagged["date"] == last, "ind_ret"].iloc[0]
    last_same = same_day.loc[same_day["date"] == last, "ind_ret"].iloc[0]
    assert last_lagged == pytest.approx(0.0)       # 50/50 weights from t-1
    assert last_same > last_lagged                 # the leak, made visible


# ---------------------------------------------------------------- universe

def test_formation_stats_are_strictly_backward_looking(clean_panel):
    """Row `month` must carry statistics computed through `month - 1`.

    Built by hand rather than on the fixture, so the expected number is
    arithmetic rather than a re-implementation of the function.
    """
    monthly = pd.DataFrame(
        {
            "permno": [1] * 4,
            "month": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "exchcd": 1,
            "ff49": 1,
            "prc": [10.0, 20.0, 30.0, 40.0],
            "mktcap": [100.0, 200.0, 300.0, 400.0],
            "med_dollar_vol": [1.0, 2.0, 3.0, 4.0],
            "n_obs": [20, 20, 20, 20],
        }
    )
    stats = formation_window_stats(monthly, formation_months=12)
    march = stats.loc[stats["month"] == pd.Timestamp("2020-03-01")].iloc[0]
    # March knows February's level, not March's.
    assert march["prc_w"] == 20.0
    assert march["mktcap_w"] == 200.0
    # And the first month knows nothing.
    january = stats.loc[stats["month"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert np.isnan(january["prc_w"])


def test_universe_filters_bite_and_attrition_is_monotone(clean_panel):
    universe, attrition = build_universe(clean_panel, UniverseRules(min_obs=20))
    df = universe.frame
    assert len(df) > 0
    # Every eligible row passed every screen — the composite is not looser
    # than its parts.
    eligible = df.loc[df["eligible"]]
    for col in ("pass_price", "pass_size", "pass_liquidity", "pass_obs", "pass_industry"):
        assert eligible[col].all()
    # Attrition is cumulative, so each column is <= the one before it.
    counts = attrition.to_numpy()
    assert (np.diff(counts, axis=1) <= 0).all()


def test_price_floor_removes_penny_stocks(clean_panel):
    """The $5 screen is the one most likely to drive a reversal result, so it
    had better actually be applied."""
    panel = clean_panel.copy()
    victim = panel["permno"].unique()[-1]
    panel.loc[panel["permno"] == victim, "prc"] = 2.0
    panel.loc[panel["permno"] == victim, "mktcap"] = 2.0 * 1e6

    universe, _ = build_universe(panel, UniverseRules(min_obs=20))
    rows = universe.frame.loc[universe.frame["permno"] == victim]
    assert not rows["pass_price"].any()
    assert not rows["eligible"].any()


def test_eligible_daily_restricts_to_the_right_month(clean_panel):
    universe, _ = build_universe(clean_panel, UniverseRules(min_obs=20))
    restricted = eligible_daily(clean_panel, universe)
    keys = set(
        map(tuple, universe.frame.loc[universe.frame["eligible"], ["month", "permno"]].to_numpy())
    )
    got = set(map(tuple, restricted[["month", "permno"]].drop_duplicates().to_numpy()))
    assert got <= keys


# ----------------------------------------------------------------- leaders

def test_largest_stock_is_the_leader(clean_panel):
    universe, _ = build_universe(clean_panel, UniverseRules(min_obs=20))
    roles = assign_roles(universe, LeaderRule(min_followers=2))
    df = roles.frame

    assert set(df["role"].unique()) == {"leader", "follower"}
    # Exactly one leader per industry-month, and it is the largest.
    per_group = df.loc[df["role"] == "leader"].groupby(["month", "ff49"]).size()
    assert (per_group == 1).all()
    assert (df.loc[df["role"] == "leader", "rank_in_industry"] == 1).all()


def test_follower_size_cap_narrows_the_set(clean_panel):
    universe, _ = build_universe(clean_panel, UniverseRules(min_obs=20))
    wide = assign_roles(universe, LeaderRule(min_followers=2))
    narrow = assign_roles(
        universe, LeaderRule(min_followers=2, follower_max_size_pct=0.5)
    )
    n_wide = (wide.frame["role"] == "follower").sum()
    n_narrow = (narrow.frame["role"] == "follower").sum()
    assert n_narrow < n_wide


def test_a_follower_is_never_its_own_leader(lagged_panel):
    universe, _ = build_universe(lagged_panel, UniverseRules(min_obs=20))
    roles = assign_roles(universe, LeaderRule(min_followers=2))
    panel = follower_panel(lagged_panel, roles)
    leaders = set(
        map(tuple, roles.frame.loc[roles.frame["role"] == "leader", ["month", "permno"]].to_numpy())
    )
    present = set(map(tuple, panel[["month", "permno"]].drop_duplicates().to_numpy()))
    assert not (present & leaders)


def test_leader_turnover_is_low_when_sizes_are_fixed(clean_panel):
    """The fake panel has constant market caps, so the leader must never
    change.  On real data this number is the diagnostic; here it is a check
    that the diagnostic works."""
    universe, _ = build_universe(clean_panel, UniverseRules(min_obs=20))
    roles = assign_roles(universe, LeaderRule(min_followers=2))
    turnover = leader_turnover(roles)
    assert turnover.attrs["overall_rate"] == pytest.approx(0.0)


def test_leader_returns_match_the_leader_stock(clean_panel):
    universe, _ = build_universe(clean_panel, UniverseRules(min_obs=20))
    roles = assign_roles(universe, LeaderRule(min_followers=2))
    lead = leader_returns(clean_panel, roles)
    assert (lead["n_leaders_obs"] == 1).all()
    assert not lead.duplicated(subset=["date", "ff49"]).any()


# ---------------------------------------------------------- quality checks

def test_duplicate_key_check_fires(clean_panel):
    from lead_lag.data.daily_returns import DailyReturns

    doubled = pd.concat([clean_panel, clean_panel.iloc[[0]]], ignore_index=True)
    dataset = DailyReturns(doubled)
    issues = DuplicateKeyCheck().run(dataset)
    assert len(issues) == 1
    assert issues[0].severity is Severity.HIGH


def test_audit_reports_all_severities(clean_panel):
    from lead_lag.data.daily_returns import DailyReturns

    report = daily_panel_audit().run(DailyReturns(clean_panel))
    summary = report.summary()
    assert set(summary.index) == {"low", "medium", "high"}
    assert summary["high"] == 0          # the clean fixture has no duplicates
