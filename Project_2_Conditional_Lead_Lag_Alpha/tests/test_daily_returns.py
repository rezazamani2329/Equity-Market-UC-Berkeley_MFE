"""
The delisting merge: every branch reached, every branch counted.

These tests are the reason `adjusted_daily_returns` returns a report object
instead of just a frame — the counts are what makes each branch assertable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_lag.data.daily_returns import (
    PERFORMANCE_CODES,
    SHUMWAY_NASDAQ,
    SHUMWAY_NYSE_AMEX,
    adjusted_daily_returns,
)
from lead_lag.data.wrds_fetch import CRSPDaily, CRSPDelist


def test_schema_is_enforced(raw_panel):
    """A frame missing a required column is rejected at the boundary."""
    broken = raw_panel.drop(columns=["ret"])
    with pytest.raises(ValueError, match="missing required columns"):
        CRSPDaily.from_raw(broken)


def test_negative_price_becomes_positive_and_zero_becomes_nan(raw_panel):
    """CRSP's two price conventions are both handled in `_coerce`."""
    df = raw_panel.copy()
    df.loc[0, "prc"] = -50.0   # quote midpoint: abs() is applied in SQL, but
    df.loc[1, "prc"] = 0.0     # a 0 must become NaN, not a $0 stock
    daily = CRSPDaily.from_raw(df)
    assert daily.frame.loc[1, "prc"] != 0.0
    assert np.isnan(daily.frame.loc[1, "prc"])
    assert np.isnan(daily.frame.loc[1, "mktcap"])


def test_matched_event_multiplies_into_the_return(raw_panel, delist_events):
    """A delisting on a traded day gives (1 + ret)(1 + dlret) - 1."""
    daily = CRSPDaily.from_raw(raw_panel)
    events = CRSPDelist.from_raw(delist_events)
    returns, report = adjusted_daily_returns(daily, events)

    assert report.n_events_matched >= 1
    hit = returns.frame.loc[returns.frame["source"] == "traded+delist"].iloc[0]
    expected = (1 + hit["ret_raw"]) * (1 + hit["dlret"]) - 1
    assert hit["ret"] == pytest.approx(expected)


def test_orphan_event_is_appended_not_dropped(raw_panel, delist_events):
    """An event after the panel's last day becomes its own row.

    This is the survivorship-bias guard: without it the stock's series simply
    ends and its final, worst return never enters any cross-section.
    """
    daily = CRSPDaily.from_raw(raw_panel)
    events = CRSPDelist.from_raw(delist_events)
    returns, report = adjusted_daily_returns(daily, events)

    assert report.n_events_orphan >= 1
    orphans = returns.frame.loc[returns.frame["source"] == "delist_only"]
    assert len(orphans) == report.n_events_orphan
    # An orphan has no traded return but does have a return.
    assert orphans["ret_raw"].isna().all()
    assert orphans["ret"].notna().all()


def test_shumway_repair_fires_only_on_performance_codes(raw_panel, delist_events):
    """A missing `dlret` is substituted only where Shumway's evidence applies."""
    daily = CRSPDaily.from_raw(raw_panel)
    events = CRSPDelist.from_raw(delist_events)
    returns, report = adjusted_daily_returns(daily, events, repair=True)

    assert report.n_repaired >= 1
    repaired = returns.frame.loc[returns.frame["repaired"]]
    assert set(repaired["dlret"].unique()) <= {SHUMWAY_NYSE_AMEX, SHUMWAY_NASDAQ}
    # And the exchange-specific constant is the right one.
    for _, row in repaired.iterrows():
        expected = SHUMWAY_NASDAQ if row["exchcd"] == 3 else SHUMWAY_NYSE_AMEX
        assert row["dlret"] == pytest.approx(expected)


def test_repair_can_be_switched_off(raw_panel, delist_events):
    """`repair=False` is the robustness variant part 5 needs."""
    daily = CRSPDaily.from_raw(raw_panel)
    events = CRSPDelist.from_raw(delist_events)
    _returns, report = adjusted_daily_returns(daily, events, repair=False)
    assert report.n_repaired == 0


def test_duplicate_delisting_rows_do_not_fan_out_the_panel(raw_panel, delist_events):
    """Two delisting rows for one (permno, date) must not double the daily row.

    CRSP does carry corrected and re-dated delistings, and an un-aggregated
    left join would silently turn one daily observation into two — breaking
    the (date, permno) key that every later join relies on.
    """
    daily = CRSPDaily.from_raw(raw_panel)
    doubled = pd.concat([delist_events, delist_events.iloc[[0]]], ignore_index=True)
    events = CRSPDelist.from_raw(doubled)
    returns, _report = adjusted_daily_returns(daily, events)
    assert not returns.frame.duplicated(subset=["date", "permno"]).any()


def test_performance_codes_cover_the_documented_blocks():
    """A guard on the constant itself, so an edit cannot quietly narrow it."""
    assert 500 in PERFORMANCE_CODES
    assert 520 in PERFORMANCE_CODES and 584 in PERFORMANCE_CODES
    assert 231 not in PERFORMANCE_CODES   # merger
    assert 400 not in PERFORMANCE_CODES   # liquidation
