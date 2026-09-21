"""
Bid-ask bounce and the defenses against it.

The centrepiece is `test_planted_bounce_is_visible_in_closes_and_absent_from_midpoints`.
It builds a stock whose TRUE price is a random walk — zero autocorrelation by
construction — and then prints each close at the bid or the ask at random, the
way a real tape does. Close-to-close returns then show Roll's spurious
negative autocorrelation; midpoint returns do not. That is the whole argument
for measuring on midpoints, demonstrated rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_lag.data.microstructure import (
    attach_quote_columns,
    bounce_exposure,
    compare_to_close,
    midpoint_returns,
    split_adjust,
    weekly_returns,
)


def _bouncing_stock(
    n_days: int = 2000, rel_spread: float = 0.03, seed: int = 7
) -> pd.DataFrame:
    """A random-walk stock whose closes bounce between bid and ask.

    True (unobservable) price follows a random walk with 1.5% daily vol. The
    quotes straddle it at `rel_spread`. The observed close is the bid or the
    ask with equal probability — Roll's model exactly.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)

    true_px = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n_days)))
    half = true_px * rel_spread / 2.0
    bid, ask = true_px - half, true_px + half
    at_ask = rng.random(n_days) < 0.5
    close = np.where(at_ask, ask, bid)

    df = pd.DataFrame(
        {
            "date": dates,
            "permno": 1,
            "prc": close,
            "bid": bid,
            "ask": ask,
            "vol": 1e6,
            "cfacpr": 1.0,
            "quote_only": False,
        }
    )
    # CRSP's `retx` is the price return off the (bouncing) closes.
    df["retx"] = df["prc"].pct_change()
    df["ret"] = df["retx"]
    return df


def test_split_adjust_makes_price_continuous():
    """A 2-for-1 split halves the raw print; cfacpr is 2 before and 1 after.

    CRSP anchors the cumulative factor at the END of the security's history,
    so it counts DOWN through splits. Getting the direction backwards turns a
    2-for-1 into a -75% return instead of a 0% one.
    """
    price = pd.Series([100.0, 100.0, 50.0, 50.0])
    cfacpr = pd.Series([2.0, 2.0, 1.0, 1.0])
    adj = split_adjust(price, cfacpr)
    assert adj.tolist() == [50.0, 50.0, 50.0, 50.0]
    # The whole point: the return across the split is zero.
    assert adj.pct_change().iloc[2] == pytest.approx(0.0)
    # Without the adjustment it would have been -50%.
    assert price.pct_change().iloc[2] == pytest.approx(-0.5)


def test_split_adjust_survives_a_zero_factor():
    """A zero or missing cfacpr must not produce inf."""
    out = split_adjust(pd.Series([10.0, 10.0]), pd.Series([0.0, np.nan]))
    assert out.tolist() == [10.0, 10.0]


def test_quote_columns_reject_crossed_quotes():
    """ask <= bid is meaningless, not merely noisy."""
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=3),
            "permno": 1,
            "bid": [10.0, 11.0, 12.0],
            "ask": [10.1, 10.9, 12.1],    # row 1 is crossed
            "vol": 1.0, "cfacpr": 1.0, "quote_only": False,
        }
    )
    out = attach_quote_columns(df)
    assert out["quote_valid"].tolist() == [True, False, True]
    assert np.isnan(out.loc[1, "mid"])
    assert np.isnan(out.loc[1, "rel_spread"])


def test_midpoint_return_needs_both_endpoints_valid():
    """A midpoint return is only bounce-free if NEITHER end is a close."""
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=4),
            "permno": 1,
            "bid": [10.0, np.nan, 12.0, 13.0],
            "ask": [10.1, np.nan, 12.1, 13.1],
            "vol": 1.0, "cfacpr": 1.0, "quote_only": False,
        }
    )
    out = midpoint_returns(attach_quote_columns(df))
    # Row 0 has no predecessor; rows 1 and 2 touch the missing quote.
    assert out["mid_ret"].isna().tolist() == [True, True, True, False]


def test_midpoint_return_is_split_adjusted():
    """Without cfacpr a 2-for-1 split reads as a -50% midpoint return.

    cfacpr counts DOWN through the split (2 before, 1 after), matching CRSP's
    end-of-history anchoring.
    """
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=3),
            "permno": 1,
            "bid": [100.0, 100.0, 50.0],
            "ask": [100.2, 100.2, 50.1],
            "vol": 1.0,
            "cfacpr": [2.0, 2.0, 1.0],
            "quote_only": False,
        }
    )
    out = midpoint_returns(attach_quote_columns(df))
    assert out["mid_ret"].iloc[2] == pytest.approx(0.0, abs=1e-3)

    # And the unadjusted version is the -50% the adjustment exists to prevent.
    raw = df.assign(cfacpr=1.0)
    raw_out = midpoint_returns(attach_quote_columns(raw))
    assert raw_out["mid_ret"].iloc[2] == pytest.approx(-0.5, abs=1e-3)


def test_planted_bounce_is_visible_in_closes_and_absent_from_midpoints():
    """The argument for midpoints, demonstrated.

    The true price is a random walk, so any autocorrelation in the observed
    return series is an artefact. Close-to-close returns must show Roll's
    negative first-order autocorrelation; midpoint returns must not.
    """
    df = _bouncing_stock(rel_spread=0.03)
    out = midpoint_returns(attach_quote_columns(df))

    ac1_close = out["retx"].autocorr(lag=1)
    ac1_mid = out["mid_ret"].autocorr(lag=1)

    assert ac1_close < -0.25, f"planted bounce did not show up (ac1={ac1_close:.3f})"
    assert abs(ac1_mid) < 0.10, f"midpoint series is contaminated (ac1={ac1_mid:.3f})"
    # And the midpoint series is far less volatile, because the bounce is gone.
    assert out["mid_ret"].std() < out["retx"].std()


def test_roll_bound_tracks_the_spread():
    """Roll's one-way bounce is half the relative spread, and it is reported
    in the same units as a measured reversal so the two can be compared."""
    df = _bouncing_stock(rel_spread=0.03)
    out = attach_quote_columns(df)
    assert out["rel_spread"].median() == pytest.approx(0.03, rel=0.02)
    assert out["roll_bound"].median() == pytest.approx(0.015, rel=0.02)


def test_compare_to_close_separates_the_two_autocorrelations():
    """The per-year diagnostic must surface the gap the previous test asserts."""
    df = _bouncing_stock(rel_spread=0.03)
    out = midpoint_returns(attach_quote_columns(df))
    table = compare_to_close(out)
    assert (table["ac1_retx"] < table["ac1_mid_ret"]).all()


def test_weekly_compounding_is_exact():
    """Five daily returns compound to the week's gross return."""
    dates = pd.bdate_range("2020-01-06", periods=10)   # two clean Mon-Fri weeks
    rets = np.full(10, 0.01)
    df = pd.DataFrame({"date": dates, "permno": 1, "ret": rets})
    out = weekly_returns(df)
    assert len(out) == 2
    assert (out["n_days"] == 5).all()
    assert out["ret_w"].iloc[0] == pytest.approx(1.01 ** 5 - 1)


def test_weekly_keeps_the_day_count_so_short_weeks_are_droppable():
    """A 'weekly' return built from two days is neither; the caller needs to
    be able to see that."""
    dates = pd.to_datetime(["2020-01-09", "2020-01-10", "2020-01-13"])
    df = pd.DataFrame({"date": dates, "permno": 1, "ret": [0.01, 0.01, 0.02]})
    out = weekly_returns(df)
    assert out["n_days"].tolist() == [2, 1]


def test_bounce_exposure_reports_the_scale_of_the_artefact():
    """The table that goes beside the measured effect."""
    df = _bouncing_stock(rel_spread=0.03)
    out = attach_quote_columns(df)
    table = bounce_exposure(out)
    assert table["median_rel_spread"].iloc[0] == pytest.approx(0.03, rel=0.05)
    assert (table["quote_valid_share"] == 1.0).all()
    assert (table["stale_close_share"] == 0.0).all()
