"""
Synthetic CRSP frames with a KNOWN answer, so the pipeline can be tested
without WRDS.

Why bother building fakes when real data is one query away.  Three reasons,
and they are the reasons this file exists rather than a `@pytest.mark.live`
on everything:

  * Real data has no ground truth.  If `baseline_test` returns b = 0.04 on
    CRSP, there is no way to tell a correct implementation from one that is
    off by a lag.  Here the effect is planted, so the test can assert that the
    estimate is close to what was planted, and a lag error fails loudly.
  * The tests run offline, in a second, on a teammate's laptop with no WRDS
    account.  That matters for a five-person project.
  * Edge cases can be constructed.  A delisting on a day the stock did not
    trade happens in maybe one in ten thousand real rows; here it is row 3.

What `make_panel` plants
------------------------
    - `n_industries` industries, each with one large stock (the leader) and
      `n_followers` smaller ones.
    - The leader's return is i.i.d. noise.
    - Each follower's return is `beta * leader_return[t-1]` plus its own
      noise, i.e. a pure one-day lead-lag effect of known size and no
      contemporaneous correlation.
    - Market caps are constant per stock and ordered so the intended leader
      is unambiguously the largest.
    - Followers are split across NYSE and NASDAQ with caps spread over more
      than an order of magnitude.  That spread is not decoration: the size
      screen takes its breakpoint from the NYSE cross-section only, so a
      fixture whose NYSE members are just the four leaders would put every
      follower below the 20th percentile and empty the universe.  A test
      fixture has to be rich enough for the filters to have something to bite
      on besides everything.

So `baseline_test(..., lag=1).beta_pooled` should recover `beta`, and the
same test at `lag=2` should return roughly zero.  Those two assertions
together catch every off-by-one in the shifting logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# SIC codes that the Fama-French 49 ranges definitely assign to distinct real
# industries, so the fake panel survives `attach_industry` with sensible
# labels instead of falling into "Other".
SIC_BY_INDUSTRY = [100, 2000, 2800, 3570, 4900, 6020, 7372, 1311]


def make_panel(
    # Eight, not four, and the reason is econometric rather than cosmetic.
    # Every follower in an industry sees the SAME leader return, so a daily
    # cross-sectional regression — the first stage of Fama-MacBeth — has an
    # effective sample size equal to the number of INDUSTRIES, however many
    # followers there are.  With four industries and three parameters that is
    # one degree of freedom, and the daily coefficients are so noisy that
    # their mean is visibly attenuated (0.23 against a planted 0.30).  The
    # same arithmetic applies to the real thing: what powers this design is
    # the industry count, which is why Fama-French 49 rather than a coarser
    # scheme.
    n_industries: int = 8,
    n_followers: int = 8,
    n_days: int = 500,
    beta: float = 0.30,
    seed: int = 0,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    """Build a fake daily panel in `CRSPDaily`'s canonical schema.

    Returns a plain DataFrame (not a `CRSPDaily`) so a test can corrupt it —
    duplicate a key, blank a price — before handing it to the loader and
    checking that the loader complains.

    The dates are business days, which is close enough to a trading calendar
    for tests that never look at holidays.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)

    rows = []
    permno = 10000
    for j in range(n_industries):
        siccd = SIC_BY_INDUSTRY[j % len(SIC_BY_INDUSTRY)]

        # ---- the leader: i.i.d. returns, the largest market cap -----------
        leader_ret = rng.normal(0.0, 0.02, size=n_days)
        rows.append(
            _stock_frame(
                dates, permno, leader_ret, siccd,
                mktcap=1e11, exchcd=1, prc=100.0,
            )
        )
        leader_permno = permno
        permno += 1

        # ---- followers: beta on the leader's PREVIOUS day -----------------
        for i in range(n_followers):
            own_noise = rng.normal(0.0, 0.02, size=n_days)
            follower_ret = np.empty(n_days)
            follower_ret[0] = own_noise[0]
            follower_ret[1:] = beta * leader_ret[:-1] + own_noise[1:]
            rows.append(
                _stock_frame(
                    dates, permno, follower_ret, siccd,
                    # Geometric spread from $2bn to $50bn: well below the
                    # leader, well above a penny stock, and distinct enough
                    # that within-industry size ranks never tie.
                    mktcap=2e9 * (25 ** (i / max(n_followers - 1, 1))),
                    # Alternate exchanges so the NYSE cross-section that sets
                    # the size breakpoint is more than just the leaders.
                    exchcd=1 if i % 2 == 0 else 3,
                    prc=40.0,
                )
            )
            permno += 1

        del leader_permno  # named only for readability above

    return pd.concat(rows, ignore_index=True).sort_values(["date", "permno"]).reset_index(
        drop=True
    )


def _stock_frame(
    dates: pd.DatetimeIndex,
    permno: int,
    ret: np.ndarray,
    siccd: int,
    mktcap: float,
    exchcd: int,
    prc: float,
    rel_spread: float = 0.002,
    split_at: int | None = None,
) -> pd.DataFrame:
    """One stock's rows, in `CRSPDaily`'s canonical column set (schema v2).

    Quotes and adjustment factors are populated rather than left NaN, because
    the microstructure module is built on them and a fixture of NaNs would let
    it pass by doing nothing:

    * `bid` / `ask` straddle the price at `rel_spread`, so the midpoint is
      exactly `prc` and a midpoint return equals the price return.  A test
      that plants bounce does so by perturbing these, not by accident.
    * `cfacpr` is 1.0 unless `split_at` is given.  CRSP anchors the
      cumulative factor at the END of a security's history, so for a single
      2-for-1 split it is 2 BEFORE the event and 1 after, while the raw price
      halves — which is what makes `prc / cfacpr` continuous across it.  (The
      inverse convention, 1 then 2, turns a split into a -75% return; that
      was a real bug here, caught by `test_midpoint_return_is_split_adjusted`.)
    * `quote_only` is False: these are traded days.  Staleness tests
      construct their own rows.
    """
    n = len(dates)
    shrout = mktcap / prc

    half = prc * rel_spread / 2.0
    cfacpr = np.ones(n)
    price = np.full(n, prc, dtype=float)
    if split_at is not None:
        # After a 2-for-1 split CRSP halves the raw prints; the cumulative
        # factor is 2 before and 1 after, so price/cfacpr is continuous.
        cfacpr[:split_at] = 2.0
        price[split_at:] = prc / 2.0

    return pd.DataFrame(
        {
            "date": dates,
            "permno": permno,
            "permco": permno,
            "ncusip": f"{permno:08d}",
            "shrcd": 11,
            "exchcd": exchcd,
            "siccd": siccd,
            "prc": price,
            "shrout": shrout,
            "mktcap": mktcap,
            "ret": ret,
            "retx": ret,
            # Dollar volume comfortably above any plausible liquidity floor,
            # and proportional to size so the within-exchange rank is stable.
            "vol": mktcap / prc * 0.01,
            "numtrd": np.nan,
            "bid": price - half,
            "ask": price + half,
            "openprc": price,
            "cfacpr": cfacpr,
            "cfacshr": cfacpr,
            "quote_only": False,
        }
    )


def make_delist(panel: pd.DataFrame, n_events: int = 3, seed: int = 1) -> pd.DataFrame:
    """Delisting events for the fake panel, deliberately mixing the branches.

    Builds `n_events` events: the first on a day the stock HAS a row (the
    matched branch), the second on a day after the panel ends (the orphan
    branch), the third a performance-related code with a missing `dlret` (the
    Shumway-repair branch).  With `n_events < 3` the later branches are
    dropped in that order.
    """
    rng = np.random.default_rng(seed)
    permnos = sorted(panel["permno"].unique())
    chosen = rng.choice(permnos, size=min(n_events, len(permnos)), replace=False)
    last_day = panel["date"].max()

    events = []
    if len(chosen) > 0:
        # 1. matched: a day the stock traded, in the middle of the sample.
        mid = panel.loc[panel["permno"] == chosen[0], "date"].iloc[len(panel["date"].unique()) // 2]
        events.append({"permno": int(chosen[0]), "dlstdt": mid, "dlstcd": 231, "dlret": -0.05, "dlretx": -0.05})
    if len(chosen) > 1:
        # 2. orphan: after the panel's last day, so no daily row exists.
        events.append({"permno": int(chosen[1]), "dlstdt": last_day + pd.Timedelta(days=5),
                       "dlstcd": 241, "dlret": -0.10, "dlretx": -0.10})
    if len(chosen) > 2:
        # 3. performance code with no dlret — the repair branch.
        mid = panel.loc[panel["permno"] == chosen[2], "date"].iloc[10]
        events.append({"permno": int(chosen[2]), "dlstdt": mid, "dlstcd": 574,
                       "dlret": np.nan, "dlretx": np.nan})

    return pd.DataFrame(events)


def make_factors(panel: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    """A factor file covering the panel's dates, in `FactorsDaily`'s schema.

    The factors are independent noise: the fake panel's returns do not load on
    them, so including `mktrf` as a control should leave the lead-lag estimate
    essentially unchanged.  A test that finds otherwise has found a bug in the
    merge, not an economic effect.
    """
    rng = np.random.default_rng(seed)
    dates = pd.Index(sorted(panel["date"].unique()), name="date")
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "mktrf": rng.normal(0, 0.01, n),
            "smb": rng.normal(0, 0.005, n),
            "hml": rng.normal(0, 0.005, n),
            "rmw": rng.normal(0, 0.005, n),
            "cma": rng.normal(0, 0.005, n),
            "umd": rng.normal(0, 0.005, n),
            "rf": 0.0001,
        }
    )
