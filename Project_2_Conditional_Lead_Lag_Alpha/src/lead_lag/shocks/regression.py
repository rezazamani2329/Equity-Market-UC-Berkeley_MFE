"""
The regression that turns the decomposition into a finding.

Part 1's ``baseline`` regresses the follower's return on the leader's *total*
lagged return and gets one coefficient ``b`` - which in this sample is positive
(continuation / diffusion).  Part 2's claim is that this single ``b`` is an
average of two opposite responses, and that splitting the leader's move into its
common and leader-specific pieces separates them:

    r_{i,t} = a + b_c * c_{L,t-k} + b_u * u_{L,t-k}
                + phi * r_{i,t-k} + delta * mkt_t + e_{i,t}

    b_c   the follower's response to the COMMON component of the leader's move
          k days ago.  Hypothesis: b_c > 0 - real industry news diffusing to
          the smaller firm (Hou 2007), which should continue.
    b_u   the response to the LEADER-SPECIFIC shock k days ago.  Hypothesis:
          b_u < 0 - the follower was dragged along by a move that had nothing
          to do with its own fundamentals, and it reverts.
    phi   the follower's own lagged return, exactly as in the baseline: without
          it, b_u would pick up ordinary short-horizon own-reversal, because a
          follower's own lag is correlated with its industry leader's lag.
    delta the contemporaneous market control, again as in the baseline.

Because ``c + u = r_L`` identically, restricting ``b_c = b_u`` collapses this to
the baseline regression on the leader's total return.  So the whole project
reduces to one testable inequality, ``b_u < 0 < b_c``, and the cleanest single
number is the *difference* ``b_c - b_u`` with a standard error - the amount of
predictability the decomposition adds over treating the leader's move as one
undifferentiated shock.  ``conditional_response`` returns all three.

Standard errors are computed the same two ways as the baseline, on purpose, so
the comparison in the report is apples to apples: pooled OLS clustered by date,
and Fama-MacBeth with Newey-West.  This module reuses ``baseline.fama_macbeth``'s
philosophy but runs a two-regressor cross-section, and it owns one thing the
baseline flagged and deferred - the overlap correction to the Newey-West lag
count when a signal is cumulated over a window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lead_lag.shocks.decomposition import LeaderShocks


# --------------------------------------------------------------------------- #
# Newey-West lag rules                                                         #
# --------------------------------------------------------------------------- #
def nw_lags_rule(n: int) -> int:
    """The textbook automatic truncation, ceil(4 * (n/100)^(2/9)).

    Matches ``baseline.fama_macbeth`` so the two parts' t-statistics are built
    with the same rule.
    """
    return int(np.ceil(4 * (n / 100) ** (2 / 9)))


def nw_lags_overlap(n: int, window: int) -> int:
    """Newey-West truncation when observations overlap by ``window - 1`` days.

    The baseline's warning, made concrete.  The baseline uses single-day leader
    returns, so consecutive observations do not overlap and ``nw_lags_rule`` is
    enough.  The moment a signal cumulates the leader shock over a ``window``-day
    lookback, consecutive observations share ``window - 1`` days of that window;
    their errors are mechanically autocorrelated out to lag ``window - 1``, and
    a truncation shorter than that leaves the induced autocorrelation in the
    residual and overstates the t-statistic.  The lag count must therefore be at
    least ``window - 1``; we take the larger of that and the automatic rule.
    """
    return max(nw_lags_rule(n), window - 1)


# --------------------------------------------------------------------------- #
# Bringing lagged leader components onto the follower panel                    #
# --------------------------------------------------------------------------- #
def lag_leader_components(shocks: LeaderShocks, lag: int = 1) -> pd.DataFrame:
    """Shift the leader's common and specific components ``lag`` days forward.

    The shift is WITHIN each industry's own ordered series, mirroring
    ``baseline.build_lags`` exactly: by rows, which is correct because the
    industry-level shock frame has one row per (industry, trading day) with no
    gaps.  Shift first; a consumer drops the resulting NaNs after the merge, not
    before, for the same reason the baseline gives.

    Returns ``date, ff49, common_lag, shock_lag, leader_ret_lag`` - the last a
    cross-check, since ``common_lag + shock_lag`` must equal it.
    """
    df = shocks.frame.sort_values(["ff49", "date"]).copy()
    g = df.groupby("ff49", sort=False)
    df["common_lag"] = g["common"].shift(lag)
    df["shock_lag"] = g["shock"].shift(lag)
    df["leader_ret_lag"] = g["leader_ret"].shift(lag)
    return df[["date", "ff49", "common_lag", "shock_lag", "leader_ret_lag"]]


def build_conditional_panel(
    follower_panel: pd.DataFrame,
    shocks: LeaderShocks,
    lag: int = 1,
    ret_col: str = "ret",
) -> pd.DataFrame:
    """Follower-days with the lagged leader components and the follower's own lag.

    ``follower_panel`` is part 1's ``leaders.follower_panel`` output (optionally
    merged with the factor file for ``mktrf``).  This attaches:

    * ``common_lag``, ``shock_lag`` - the leader's decomposed move ``lag`` days
      ago, merged on (date, ff49);
    * ``own_lag`` - the follower's own return ``lag`` days ago, shifted within
      ``permno``.

    Everything is shifted before anything is dropped, so an illiquid follower's
    day t is never lined up against a day that is not its own t-k.
    """
    df = follower_panel.sort_values(["permno", "date"]).copy()
    df["own_lag"] = df.groupby("permno", sort=False)[ret_col].shift(lag)

    lagged = lag_leader_components(shocks, lag=lag)
    df = df.merge(lagged, on=["date", "ff49"], how="left")
    return df


# --------------------------------------------------------------------------- #
# The conditional-response regression                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConditionalResult:
    """One horizon's estimate of the two responses, with both standard errors.

    Attributes mirror ``baseline.RegressionResult`` where they can, so the two
    tables stack.  ``beta_diff`` and its Fama-MacBeth t-statistic are the
    headline: the predictability the split adds over the undifferentiated
    leader move.
    """

    lag: int
    n_obs: int
    n_days: int
    # response to the common component
    b_common_pooled: float
    t_common_pooled: float
    b_common_fm: float
    t_common_fm: float
    # response to the leader-specific shock
    b_shock_pooled: float
    t_shock_pooled: float
    b_shock_fm: float
    t_shock_fm: float
    # the difference, Fama-MacBeth
    beta_diff_fm: float
    t_diff_fm: float

    def as_row(self) -> dict[str, float | int]:
        return {
            "lag": self.lag,
            "n_obs": self.n_obs,
            "n_days": self.n_days,
            "b_common_pooled": self.b_common_pooled,
            "t_common_pooled": self.t_common_pooled,
            "b_common_fm": self.b_common_fm,
            "t_common_fm": self.t_common_fm,
            "b_shock_pooled": self.b_shock_pooled,
            "t_shock_pooled": self.t_shock_pooled,
            "b_shock_fm": self.b_shock_fm,
            "t_shock_fm": self.t_shock_fm,
            "beta_diff_fm": self.beta_diff_fm,
            "t_diff_fm": self.t_diff_fm,
        }


def _fama_macbeth_multi(
    df: pd.DataFrame,
    regressors: list[str],
    ret_col: str,
    min_cross_section: int,
    nw_lags: int | None,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    """Daily cross-sectional OLS of ``ret_col`` on ``regressors``; NW test of
    each mean coefficient.  Also returns the daily coefficient frame so the
    difference of two columns can itself be tested.
    """
    cols = [ret_col, "date", *regressors]
    d = df[cols].dropna()

    daily: dict[pd.Timestamp, dict[str, float]] = {}
    for day, chunk in d.groupby("date", sort=True):
        if len(chunk) < min_cross_section:
            continue
        x = sm.add_constant(chunk[regressors], has_constant="add")
        # Skip days where any regressor is constant across followers (the whole
        # industry cross-section collapsed onto one leader): the design is
        # singular and the coefficient is meaningless, exactly as the baseline
        # guards its single-regressor version.
        if any(x[r].nunique() < 2 for r in regressors):
            continue
        try:
            fit = sm.OLS(chunk[ret_col], x).fit()
        except np.linalg.LinAlgError:
            continue
        daily[day] = {r: float(fit.params[r]) for r in regressors}

    coefs = pd.DataFrame(daily).T.sort_index()
    means: dict[str, float] = {}
    tstats: dict[str, float] = {}
    if coefs.empty:
        return ({r: float("nan") for r in regressors},
                {r: float("nan") for r in regressors}, coefs)

    lags = nw_lags if nw_lags is not None else nw_lags_rule(len(coefs))
    for r in regressors:
        s = coefs[r].dropna()
        fit = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        means[r] = float(fit.params[0])
        tstats[r] = float(fit.tvalues[0])
    return means, tstats, coefs


def conditional_response(
    conditional_panel: pd.DataFrame,
    lag: int = 1,
    ret_col: str = "ret",
    market_col: str | None = "mktrf",
    min_cross_section: int = 20,
    nw_lags: int | None = None,
) -> ConditionalResult:
    """Estimate ``b_c`` and ``b_u`` at one horizon, both standard-error ways.

    ``conditional_panel`` is ``build_conditional_panel`` output.  The pooled fit
    is OLS clustered by date; the Fama-MacBeth fit runs the two-regressor
    cross-section each day and tests the mean coefficients (and their
    difference) with Newey-West.  The market control and the follower own-lag
    enter both, matching the baseline so ``b_c``/``b_u`` are the decomposed
    counterpart of the baseline's single ``b``.
    """
    regressors = ["common_lag", "shock_lag", "own_lag"]
    ctrl = [market_col] if market_col and market_col in conditional_panel.columns else []
    used = conditional_panel[[ret_col, "date", *regressors, *ctrl]].dropna()

    # ---- pooled OLS, clustered by date ----
    x = sm.add_constant(used[[*regressors, *ctrl]], has_constant="add")
    pooled = sm.OLS(used[ret_col], x).fit(
        cov_type="cluster", cov_kwds={"groups": used["date"]}
    )

    # ---- Fama-MacBeth on common_lag and shock_lag (own_lag as control) ----
    means, tstats, coefs = _fama_macbeth_multi(
        used, ["common_lag", "shock_lag", "own_lag"], ret_col,
        min_cross_section, nw_lags,
    )

    # ---- difference b_c - b_u, tested on the daily series ----
    if not coefs.empty and {"common_lag", "shock_lag"}.issubset(coefs.columns):
        diff = (coefs["common_lag"] - coefs["shock_lag"]).dropna()
        lags = nw_lags if nw_lags is not None else nw_lags_rule(len(diff))
        dfit = sm.OLS(diff.to_numpy(), np.ones(len(diff))).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        beta_diff_fm, t_diff_fm = float(dfit.params[0]), float(dfit.tvalues[0])
    else:
        beta_diff_fm = t_diff_fm = float("nan")

    return ConditionalResult(
        lag=lag,
        n_obs=len(used),
        n_days=int(used["date"].nunique()),
        b_common_pooled=float(pooled.params["common_lag"]),
        t_common_pooled=float(pooled.tvalues["common_lag"]),
        b_common_fm=means["common_lag"],
        t_common_fm=tstats["common_lag"],
        b_shock_pooled=float(pooled.params["shock_lag"]),
        t_shock_pooled=float(pooled.tvalues["shock_lag"]),
        b_shock_fm=means["shock_lag"],
        t_shock_fm=tstats["shock_lag"],
        beta_diff_fm=beta_diff_fm,
        t_diff_fm=t_diff_fm,
    )


def conditional_horizon_profile(
    follower_panel: pd.DataFrame,
    shocks: LeaderShocks,
    lags: range | list[int] = range(1, 11),
    ret_col: str = "ret",
    market_col: str | None = "mktrf",
    verbose: bool = True,
) -> pd.DataFrame:
    """``b_c(k)`` and ``b_u(k)`` across horizons - the decomposed twin of
    ``baseline.horizon_profile``.

    The project's central picture: the common response ``b_c(k)`` should look
    like the baseline continuation profile, while the leader-specific response
    ``b_u(k)`` is where the reversal the baseline never showed is expected to
    appear.  Plot the two side by side against the baseline's single ``b(k)``.
    """
    rows = []
    for k in lags:
        if verbose:
            print(f"conditional response: lag {k} ...", flush=True)
        panel = build_conditional_panel(follower_panel, shocks, lag=k, ret_col=ret_col)
        rows.append(
            conditional_response(
                panel, lag=k, ret_col=ret_col, market_col=market_col
            ).as_row()
        )
    return pd.DataFrame(rows).set_index("lag")
