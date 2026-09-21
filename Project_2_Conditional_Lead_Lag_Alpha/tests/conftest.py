"""
Shared fixtures.  `conftest.py` in `tests/` also puts that directory on
`sys.path`, which is what lets the test modules do `from fake_crsp import ...`
without a package.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fake_crsp import make_delist, make_factors, make_panel

from lead_lag.data.daily_returns import adjusted_daily_returns
from lead_lag.data.industry_map import attach_industry
from lead_lag.data.information_set import attach_lagged
from lead_lag.data.wrds_fetch import CRSPDaily, CRSPDelist

# A small Fama-French-style range table covering the SIC codes `fake_crsp`
# uses, so the mapping can be tested without downloading the real file.
# Numbers and names follow the real scheme where they overlap, but nothing in
# the tests depends on that — only on the ranges being disjoint.
MINI_SICCODES = pd.DataFrame(
    [
        {"ff49": 1, "ff49_name": "Agric", "sic_lo": 100, "sic_hi": 199},
        {"ff49": 2, "ff49_name": "Food", "sic_lo": 2000, "sic_hi": 2099},
        {"ff49": 13, "ff49_name": "Drugs", "sic_lo": 2800, "sic_hi": 2899},
        {"ff49": 22, "ff49_name": "ElcEq", "sic_lo": 3570, "sic_hi": 3579},
        {"ff49": 31, "ff49_name": "Util", "sic_lo": 4900, "sic_hi": 4949},
        {"ff49": 45, "ff49_name": "Banks", "sic_lo": 6000, "sic_hi": 6099},
        {"ff49": 36, "ff49_name": "Softw", "sic_lo": 7370, "sic_hi": 7379},
        {"ff49": 30, "ff49_name": "Oil", "sic_lo": 1300, "sic_hi": 1399},
    ]
)


@pytest.fixture(scope="session")
def siccodes() -> pd.DataFrame:
    return MINI_SICCODES


@pytest.fixture(scope="session")
def raw_panel() -> pd.DataFrame:
    """500 business days, 4 industries, 8 followers each, beta = 0.30."""
    return make_panel()


@pytest.fixture(scope="session")
def delist_events(raw_panel: pd.DataFrame) -> pd.DataFrame:
    return make_delist(raw_panel)


@pytest.fixture(scope="session")
def factors(raw_panel: pd.DataFrame) -> pd.DataFrame:
    return make_factors(raw_panel)


@pytest.fixture(scope="session")
def clean_panel(
    raw_panel: pd.DataFrame, delist_events: pd.DataFrame, siccodes: pd.DataFrame
) -> pd.DataFrame:
    """The panel as the rest of part 1 sees it: delisting-adjusted, with
    `ff49` attached.  Session-scoped because building it is the slow part."""
    daily = CRSPDaily.from_raw(raw_panel)
    events = CRSPDelist.from_raw(delist_events)
    returns, _report = adjusted_daily_returns(daily, events)
    return attach_industry(returns.frame, siccodes)


@pytest.fixture(scope="session")
def lagged_panel(clean_panel: pd.DataFrame) -> pd.DataFrame:
    """`clean_panel` with the d-1 characteristics attached.

    Most downstream code needs this rather than `clean_panel`: the regression
    panel and the industry index both condition on the previous close.
    `clean_panel` is kept unlagged so the tests that assert the pipeline
    REFUSES a same-day frame have one to hand it.
    """
    return attach_lagged(clean_panel)
