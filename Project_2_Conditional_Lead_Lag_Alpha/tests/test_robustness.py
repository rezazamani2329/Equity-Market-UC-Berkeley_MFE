"""Tests for Part 5 (src/lead_lag/robustness). Offline, synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_lag.robustness import ROUND_TRIP, breakeven_bp, net_of_cost

TOL = 1e-12


def _series(n=250, mean=0.0002, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.002, n), index=pd.bdate_range("2010-01-04", periods=n))


def test_zero_cost_is_gross():
    gross = _series()
    assert (net_of_cost(gross, ROUND_TRIP, 0.0) - gross).abs().max() < TOL


def test_round_trip_cost_is_two_legs():
    gross = _series()
    net = net_of_cost(gross, ROUND_TRIP, 1.0)
    assert (gross - net).sub(2.0e-4).abs().max() < TOL


def test_breakeven_sets_mean_net_to_zero():
    gross = _series()
    traded = pd.Series(np.linspace(1.0, 1.6, len(gross)), index=gross.index)
    for t in (ROUND_TRIP, traded):
        assert abs(net_of_cost(gross, t, breakeven_bp(gross, t)).mean()) < TOL


def test_missing_gross_stays_missing():
    gross = _series()
    gross.iloc[3] = np.nan
    assert np.isnan(net_of_cost(gross, ROUND_TRIP, 1.0).iloc[3])


def test_losing_book_has_negative_breakeven():
    assert breakeven_bp(_series(mean=-0.0005), ROUND_TRIP) < 0
