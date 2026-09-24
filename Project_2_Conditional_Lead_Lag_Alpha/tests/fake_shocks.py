"""
A synthetic panel with a KNOWN decomposition, so part 2 can be validated
without WRDS - the same discipline ``fake_crsp`` applies to part 1.

``fake_crsp.make_panel`` plants a pure one-day lead-lag effect but gives the
leader i.i.d. returns with no common industry factor, so its ex-leader index is
uncorrelated with the leader and *everything* looks leader-specific.  That is
the right fixture for the baseline and the wrong one for the decomposition,
which needs a leader move that genuinely splits into a common piece and a
specific piece.  This module plants exactly that.

The data-generating process
---------------------------
For industry ``j`` on day ``t``:

    m_t          ~ N(0, sigma_m)          one market factor, shared by all
    f_{j,t}      ~ N(0, sigma_f)          the industry common factor
    u_{L,t}      ~ N(0, sigma_u)          the leader-specific shock

    leader:   r_{L,t} = bm_L*m_t + bf_L*f_{j,t} + u_{L,t}
              common_true = bm_L*m_t + bf_L*f_{j,t}   ;   shock_true = u_{L,t}

    follower: r_{i,t} = bm_i*m_t + bf_i*f_{j,t}                 (common exposure)
                      + gamma_cont * common_true_{L,t-1}        (diffusion -> continue)
                      - gamma_rev  * u_{L,t-1}                  (spurious   -> revert)
                      + e_{i,t}                                 e ~ N(0, sigma_e)

So the ground truth for the whole project is built in:

  * the leader's move divides cleanly into a common component (spanned by the
    market and the industry factor, which the ex-leader index proxies) and an
    orthogonal specific shock - so ``decompose_*`` should recover a specific
    variance share near ``sigma_u^2 / (bf_L^2 sigma_f^2 + bm_L^2 sigma_m^2 +
    sigma_u^2)``;
  * the follower CONTINUES on the leader's lagged common component
    (``b_common > 0``) and REVERTS on the leader's lagged specific shock
    (``b_shock < 0``), with planted magnitudes ``+gamma_cont`` and
    ``-gamma_rev``.

``make_shock_panel`` returns the raw CRSP-schema frame, a matching factor frame
whose ``mktrf`` IS ``m_t`` (so the market control is the real market), and a
``truth`` frame carrying ``common_true`` and ``shock_true`` per (date, permno)
for the leaders - the thing an estimate is scored against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fake_crsp import SIC_BY_INDUSTRY, _stock_frame


@dataclass(frozen=True)
class ShockDGP:
    """Planted parameters, returned alongside the panel so tests assert against
    the numbers that were actually used rather than hard-coded literals."""

    gamma_cont: float
    gamma_rev: float
    sigma_m: float
    sigma_f: float
    sigma_u: float
    sigma_e: float
    bm_leader: float
    bf_leader: float


def make_shock_panel(
    n_industries: int = 8,
    # 14, not 8: the size and liquidity screens remove the smallest few
    # followers per industry, and the ``follower_max_size_pct=0.5`` robustness
    # variant then halves what is left.  Eight followers survives neither, and
    # the point of a fixture is to have something for every screen to bite on.
    n_followers: int = 14,
    n_days: int = 900,
    gamma_cont: float = 0.25,
    gamma_rev: float = 0.30,
    sigma_m: float = 0.010,
    sigma_f: float = 0.012,
    sigma_u: float = 0.015,
    sigma_e: float = 0.012,
    bm_leader: float = 1.1,
    bf_leader: float = 1.0,
    seed: int = 7,
    start: str = "2013-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ShockDGP]:
    """Build the panel, its factor file, and the leader ground-truth frame.

    ``n_days`` defaults to 900 so the 252-day rolling estimator has several
    hundred point-in-time fits per industry to work with, not a handful.

    Returns:
        (raw_panel, factors, truth, dgp)
        raw_panel: CRSPDaily-schema DataFrame, ready for ``CRSPDaily.from_raw``.
        factors:   FactorsDaily-schema DataFrame; ``mktrf`` == the m_t used.
        truth:     date, permno, ff49, common_true, shock_true  (leaders only).
        dgp:       the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)

    m = rng.normal(0.0, sigma_m, size=n_days)  # market factor, shared

    rows: list[pd.DataFrame] = []
    truth_rows: list[pd.DataFrame] = []
    permno = 10000

    for j in range(n_industries):
        siccd = SIC_BY_INDUSTRY[j % len(SIC_BY_INDUSTRY)]
        f = rng.normal(0.0, sigma_f, size=n_days)          # industry factor
        u = rng.normal(0.0, sigma_u, size=n_days)          # leader-specific

        common_true = bm_leader * m + bf_leader * f
        leader_ret = common_true + u

        # ---- leader: largest market cap, unambiguously ----
        rows.append(
            _stock_frame(dates, permno, leader_ret, siccd,
                         mktcap=1e11, exchcd=1, prc=100.0)
        )
        truth_rows.append(
            pd.DataFrame({
                "date": dates, "permno": permno, "ff49_sic": siccd,
                "common_true": common_true, "shock_true": u,
            })
        )
        permno += 1

        # ---- followers: common exposure + lagged continuation/reversion ----
        for i in range(n_followers):
            bm_i = rng.uniform(0.6, 1.2)
            bf_i = rng.uniform(0.6, 1.2)
            e = rng.normal(0.0, sigma_e, size=n_days)

            r = bm_i * m + bf_i * f + e
            # lagged terms: continue on the leader's common move, revert on its
            # specific shock.  Day 0 has no t-1, so start the lagged part at 1.
            r[1:] += gamma_cont * common_true[:-1] - gamma_rev * u[:-1]

            rows.append(
                _stock_frame(dates, permno, r, siccd,
                             mktcap=2e9 * (25 ** (i / max(n_followers - 1, 1))),
                             exchcd=1 if i % 2 == 0 else 3, prc=40.0)
            )
            permno += 1

    raw = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["date", "permno"])
        .reset_index(drop=True)
    )

    factors = pd.DataFrame({
        "date": dates,
        "mktrf": m,
        "smb": rng.normal(0, 0.004, n_days),
        "hml": rng.normal(0, 0.004, n_days),
        "rmw": rng.normal(0, 0.004, n_days),
        "cma": rng.normal(0, 0.004, n_days),
        "umd": rng.normal(0, 0.004, n_days),
        "rf": 0.0001,
    })

    truth = pd.concat(truth_rows, ignore_index=True)
    dgp = ShockDGP(
        gamma_cont=gamma_cont, gamma_rev=gamma_rev,
        sigma_m=sigma_m, sigma_f=sigma_f, sigma_u=sigma_u, sigma_e=sigma_e,
        bm_leader=bm_leader, bf_leader=bf_leader,
    )
    return raw, factors, truth, dgp
