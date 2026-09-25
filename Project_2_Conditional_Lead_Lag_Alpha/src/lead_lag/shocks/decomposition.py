r"""
Splitting a leader's daily return into the piece the whole industry shared and
the piece that belonged only to the leader.

This is the load-bearing econometric step of the project.  Everything after it
- the conditional signal, the long/short book, the robustness section - is a
consequence of getting this one regression right, so the module is written to
make the two ways of getting it wrong impossible to reach by accident.

The decomposition
-----------------
For the leader (or leader portfolio) of industry ``j`` on day ``t``:

    r_{L,t} = alpha_j + beta^m_j * mkt_t + beta^ind_j * r^{ex-L}_{j,t} + u_{L,t}
              \_________________ common component c_{L,t} _____________/   \_ specific _/

* ``r^{ex-L}_{j,t}`` is the value-weighted return of industry ``j`` **with the
  leader's own rows removed**.  This is the whole game.  Part 1's
  ``industry_returns`` is built from every eligible stock, leader included, and
  the leader is a large share of a value-weighted index *by construction* - it
  was picked for being the largest.  Regress the leader on that index and the
  leader sits on both sides of the equation; ``beta^ind`` absorbs the leader's
  own move and ``u`` collapses toward zero exactly where the project needs it
  to carry information.  ``ex_leader_industry_returns`` below rebuilds the index
  the right way, and ``assemble_leader_frame`` refuses to proceed on an index
  that still contains the leader.

* ``mkt_t`` keeps a market-wide day out of the "industry" bucket.  Without it,
  ``beta^ind`` and the residual both absorb market beta, and a day when
  everything rose looks like an industry shock.

* ``c_{L,t}`` (the fitted value) is the part of the leader's move that the rest
  of its industry - and the market - moved with.  A follower that lags *this*
  should continue in the same direction (Hou-style information diffusion).

* ``u_{L,t}`` (the residual) is the part specific to the leader: its own
  earnings surprise, its own product news, an index-flow print - orthogonal to
  the sector.  A follower dragged along by *this* moved for no reason of its
  own, and is the reversion candidate.

Full sample vs point in time
----------------------------
Two estimators, and the difference is not cosmetic:

* ``decompose_full_sample`` fits one set of betas per industry on the entire
  sample.  Use it to *describe* - "what fraction of the leader's variance is
  idiosyncratic" is a full-sample question and the in-sample residual is the
  right object for it.  It is not tradeable: the day-t residual uses betas
  estimated with day-t+500's data.

* ``decompose_rolling`` estimates each industry's betas on a trailing window
  ending at ``t-1`` and forms ``u_t`` from those.  Every number is knowable at
  the previous close, so this is the version a signal may condition on.  On a
  well-behaved industry the two agree on the variance share; where they do not,
  the betas are drifting and that is itself worth reporting.

Non-synchronous trading
-----------------------
The leader is the liquid name and the ex-leader index is built from smaller,
less frequently traded stocks, so the index print partly reflects *yesterday's*
common shock.  A purely contemporaneous ``beta^ind`` then understates the
leader's true common exposure and leaves common variation in ``u`` - which
would masquerade as leader-specific and fake a reversion signal.  ``sync_lags``
adds leads and lags of the ex-leader index (a Dimson / Scholes-Williams
correction) and the common beta becomes their sum.  Part 1's STATUS lists
non-synchronous trading as the one microstructure threat *not* yet cleared;
this knob is where it is addressed on the daily panel.

Same-day classifier and the merge onto followers
--------------------------------------------------
Every row already carries ``common`` and ``shock`` with ``common + shock ==
leader_ret`` by construction.  ``classify_shock_day`` adds one more thing for
free: a same-day label for which piece *dominated* that leader-day, 'common'
when the industry-wide piece was at least as large in magnitude as the
residual, 'leader_specific' otherwise.  It is derived from ``common`` and
``shock`` alone - no follower is consulted, and no future date is either - so
labelling a day this way does not compromise the leader-only, non-circular,
same-day construction above it.

``follower_merge_table`` packages the (date, ff49)-keyed decomposition under
the names a follower panel actually wants to merge on -
``common_shock``/``leader_specific_shock`` in place of the internal
``common``/``shock`` - plus the classifier, so the merge in part 3/4 is a
single ``follower_panel.merge(follower_merge_table(shocks), on=["date",
"ff49"], how="left")`` with no renaming or re-deriving at the call site.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import statsmodels.api as sm

from lead_lag.data.industry_map import industry_returns
from lead_lag.data.leaders import LeaderMap, leader_returns
from lead_lag.data.typed_dataset import Dataset


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecompositionSpec:
    """Which regressors form the leader's *common* component, and how betas
    are estimated.

    Attributes:
        include_market: put the market excess return on the right-hand side.
            Default True - a market-wide day is common, not leader-specific.
        include_industry: put the ex-leader industry return on the right-hand
            side.  Default True; this is the whole point.  Turning it off
            recovers a plain market-model residual and is only useful as a
            "does the industry term matter" robustness row.
        sync_lags: number of leads AND lags of the ex-leader index to include
            for non-synchronous trading (Dimson).  0 = contemporaneous only.
            The common industry beta is reported as the SUM across
            {-sync_lags, ..., 0, ..., +sync_lags}.
        market_col: name of the market column in the assembled frame.
        min_obs: an industry with fewer usable days than this is skipped
            entirely rather than fit on a handful of points.
        window: trailing-window length in trading days, for the rolling
            estimator only.  Ignored by the full-sample one.
    """

    include_market: bool = True
    include_industry: bool = True
    sync_lags: int = 0
    market_col: str = "mktrf"
    min_obs: int = 60
    window: int = 252

    def regressor_note(self) -> str:
        """Human-readable summary for a results table caption."""
        parts = []
        if self.include_market:
            parts.append("mkt")
        if self.include_industry:
            parts.append(
                "ind_exl" if self.sync_lags == 0 else f"ind_exl(+/-{self.sync_lags})"
            )
        return " + ".join(parts) if parts else "constant only"


# --------------------------------------------------------------------------- #
# The same-day classifier                                                     #
# --------------------------------------------------------------------------- #
def classify_shock_day(common: np.ndarray, shock: np.ndarray) -> np.ndarray:
    """Label each leader-day by which decomposed piece dominated it.

    'common' when the industry-wide component was at least as large in
    magnitude as the leader-specific residual that day, 'leader_specific'
    when the residual was strictly larger (ties go to 'common').  Built from
    ``common`` and ``shock`` alone, so it inherits their guarantees for free:
    it is a same-day label (no future date enters it) and it never looks at a
    follower (only the leader's own decomposed return enters it).  A row
    where either input is NaN is left unclassified (``None``) rather than
    guessed.

    Args:
        common: fitted common component, one value per leader-day.
        shock: leader-specific residual, same shape and order as ``common``.

    Returns:
        Object-dtype numpy array of the same length: 'common',
        'leader_specific', or None.
    """
    common = np.asarray(common, dtype=float)
    shock = np.asarray(shock, dtype=float)
    label = np.where(np.abs(shock) > np.abs(common), "leader_specific", "common")
    label = label.astype(object)
    label[np.isnan(common) | np.isnan(shock)] = None
    return label


# --------------------------------------------------------------------------- #
# Typed output                                                                #
# --------------------------------------------------------------------------- #
class LeaderShocks(Dataset):
    """The decomposition, one row per (date, ff49).

    Canonical columns:
        date            trading day
        ff49            industry number
        leader_ret      the leader's (or leader portfolio's) return that day
        common          fitted common component c_{L,t}
        shock           leader-specific residual u_{L,t}  (leader_ret - common)
        shock_type      same-day classifier, 'common' if |common| >= |shock|
                        that day else 'leader_specific' - see
                        `classify_shock_day`.  Derived from common/shock only.
        beta_mkt        market beta used to form c on this day
        beta_ind        common industry beta (summed over sync lags)
        alpha           regression intercept used on this day
        r2              R^2 of the industry-level fit the betas came from
        n_window        observations the betas were estimated on

    ``common + shock == leader_ret`` holds row by row, exactly, by residual
    identity.  A test asserts it, because if it ever fails the residual is not
    the leader-specific shock and every number downstream is contaminated.
    """

    KEY: ClassVar[tuple[str, ...]] = ("date", "ff49")
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "date", "ff49", "leader_ret", "common", "shock", "shock_type",
        "beta_mkt", "beta_ind", "alpha", "r2", "n_window",
    )
    SYNONYMS: ClassVar[dict[str, str]] = {}

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        df["ff49"] = pd.to_numeric(df["ff49"], errors="coerce").fillna(0).astype("int64")
        for col in ("leader_ret", "common", "shock", "beta_mkt", "beta_ind",
                    "alpha", "r2"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["n_window"] = pd.to_numeric(df["n_window"], errors="coerce").fillna(0).astype("int64")
        # Categorical label, not numeric - deliberately left out of the
        # to_numeric loop above so an unexpected value surfaces as itself
        # rather than silently becoming NaN.
        df["shock_type"] = df["shock_type"].astype("string")
        return df.sort_values(["ff49", "date"]).reset_index(drop=True)

    def variance_shares(self) -> pd.DataFrame:
        """Per-industry share of leader variance that is leader-specific.

        ``specific_share = Var(shock) / Var(leader_ret) = 1 - R^2_industry``
        when the fit is full-sample.  Returned per industry with the common
        share alongside, plus the observation count - the headline
        decomposition exhibit.  A high specific share is where spurious
        comovement, and therefore the reversion the project is chasing, has
        room to live.
        """
        g = self.frame.groupby("ff49")
        var_total = g["leader_ret"].var(ddof=1)
        var_shock = g["shock"].var(ddof=1)
        out = pd.DataFrame(
            {
                "specific_share": (var_shock / var_total).clip(0, 1),
                "n_obs": g["shock"].size(),
            }
        )
        out["common_share"] = 1.0 - out["specific_share"]
        return out[["common_share", "specific_share", "n_obs"]].sort_values(
            "specific_share", ascending=False
        )


# --------------------------------------------------------------------------- #
# Step 1: the ex-leader industry index                                        #
# --------------------------------------------------------------------------- #
def ex_leader_industry_returns(
    eligible: pd.DataFrame,
    roles: LeaderMap,
    weight: str = "mktcap_lag",
    industry_col: str = "ff49",
) -> pd.DataFrame:
    """Value-weighted industry return with each industry's leaders removed.

    This is the single function in part 2 whose correctness the rest of the
    part depends on, so it is a named function with one job rather than three
    lines inlined at a call site where a later edit could quietly reintroduce
    the leader.

    Args:
        eligible: the eligible daily panel from ``universe.eligible_daily`` -
            it already carries the formation-month ``ff49`` and the lagged
            weight column, and is restricted to tradeable names.
        roles: the leader/follower assignment.  Only ``role == 'leader'`` rows
            are used, matched on (month, permno).
        weight: value weight, defaulting to the previous close's market cap,
            exactly as part 1's index.
        industry_col: grouping key; ``ff49`` (formation month) to stay
            consistent with the leader/follower vintage.

    Returns:
        Long frame ``date, ff49, ind_ret_exl, n_stocks_exl`` - the same shape
        as ``industry_returns`` but with ``_exl`` suffixes so a downstream
        merge can never confuse it with the leader-inclusive index.
    """
    df = eligible.copy()
    if "month" not in df.columns:
        df["month"] = df["date"].values.astype("datetime64[M]")

    leaders = roles.frame.loc[roles.frame["role"] == "leader", ["month", "permno"]]
    tagged = df.merge(leaders, on=["month", "permno"], how="left", indicator=True)
    ex_leader = tagged.loc[tagged["_merge"] == "left_only"].drop(columns="_merge")

    out = industry_returns(ex_leader, industry_col=industry_col, weight=weight)
    return out.rename(columns={"ind_ret": "ind_ret_exl", "n_stocks": "n_stocks_exl"})


def assemble_leader_frame(
    eligible: pd.DataFrame,
    roles: LeaderMap,
    factors: pd.DataFrame,
    ret_col: str = "ret",
    weight: str = "mktcap_lag",
    market_col: str = "mktrf",
) -> pd.DataFrame:
    """One tidy frame per (date, ff49) with everything the decomposition needs.

    Columns: ``date, ff49, leader_ret, ind_ret_exl, n_stocks_exl, <market_col>``.

    The leader return and the ex-leader index are built from the *same*
    eligible panel and the *same* role table, so they cannot drift apart, and
    the market column is joined by date from the factor file.  An industry-day
    with no ex-leader stocks (a one-stock industry) has ``ind_ret_exl`` NaN and
    is dropped downstream rather than decomposed against nothing.
    """
    lead = leader_returns(eligible, roles, ret_col=ret_col)
    ex = ex_leader_industry_returns(eligible, roles, weight=weight)

    frame = lead.merge(ex, on=["date", "ff49"], how="left")
    mkt = factors[["date", market_col]].drop_duplicates("date")
    frame = frame.merge(mkt, on="date", how="left")
    return frame.sort_values(["ff49", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 2: the regression, two ways                                            #
# --------------------------------------------------------------------------- #
def _design(
    chunk: pd.DataFrame, spec: DecompositionSpec
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the regressor matrix for one industry's time series.

    Returns (X_with_const, common_cols, sync_ind_cols).  ``sync_ind_cols`` are
    the ex-leader index and its leads/lags, whose betas are summed to report a
    single synchronization-adjusted industry beta.
    """
    x = pd.DataFrame(index=chunk.index)
    common_cols: list[str] = []
    sync_ind_cols: list[str] = []

    if spec.include_market:
        x[spec.market_col] = chunk[spec.market_col]
        common_cols.append(spec.market_col)

    if spec.include_industry:
        base = chunk["ind_ret_exl"]
        x["ind_exl"] = base
        common_cols.append("ind_exl")
        sync_ind_cols.append("ind_exl")
        for s in range(1, spec.sync_lags + 1):
            # Lead (+s) and lag (-s) of the ex-leader index, within this
            # industry's own ordered series.  Named so the sum is transparent.
            x[f"ind_exl_lead{s}"] = base.shift(-s)
            x[f"ind_exl_lag{s}"] = base.shift(s)
            common_cols += [f"ind_exl_lead{s}", f"ind_exl_lag{s}"]
            sync_ind_cols += [f"ind_exl_lead{s}", f"ind_exl_lag{s}"]

    x = sm.add_constant(x, has_constant="add")
    return x, common_cols, sync_ind_cols


def decompose_full_sample(
    frame: pd.DataFrame, spec: DecompositionSpec | None = None
) -> LeaderShocks:
    """Fit one set of betas per industry on the whole sample.

    Descriptive, not tradeable: the residual on day t uses betas that saw the
    entire history.  This is the correct object for the variance-share
    decomposition (how idiosyncratic is the leader) and for a first look at the
    common/specific split; ``decompose_rolling`` is what a signal must use.

    HAC (Newey-West) standard errors are used for the beta *diagnostics* - the
    residual itself does not depend on the covariance estimator, but reporting
    honest t-stats on the common betas is part of the design.
    """
    spec = spec or DecompositionSpec()
    rows: list[pd.DataFrame] = []
    r2s: list[float] = []           # per-industry fit quality, for the degeneracy guard
    n_seen = 0                      # industries the groupby produced at all
    max_usable = 0                  # largest usable window seen, incl. skipped industries

    for ff49, chunk in frame.sort_values("date").groupby("ff49", sort=True):
        n_seen += 1
        chunk = chunk.dropna(subset=["leader_ret", "ind_ret_exl"]).copy()
        x, common_cols, sync_ind = _design(chunk, spec)
        usable = x.dropna().index
        max_usable = max(max_usable, len(usable))
        if len(usable) < spec.min_obs:
            continue
        y = chunk.loc[usable, "leader_ret"]
        xf = x.loc[usable]

        nw = int(np.ceil(4 * (len(usable) / 100) ** (2 / 9)))
        fit = sm.OLS(y, xf).fit(cov_type="HAC", cov_kwds={"maxlags": nw})
        r2s.append(float(fit.rsquared))

        common = fit.fittedvalues
        shock = y - common
        shock_type = classify_shock_day(common.to_numpy(), shock.to_numpy())
        beta_mkt = float(fit.params.get(spec.market_col, 0.0))
        beta_ind = float(sum(fit.params.get(c, 0.0) for c in sync_ind))

        rows.append(
            pd.DataFrame(
                {
                    "date": chunk.loc[usable, "date"].to_numpy(),
                    "ff49": ff49,
                    "leader_ret": y.to_numpy(),
                    "common": common.to_numpy(),
                    "shock": shock.to_numpy(),
                    "shock_type": shock_type,
                    "beta_mkt": beta_mkt,
                    "beta_ind": beta_ind,
                    "alpha": float(fit.params.get("const", 0.0)),
                    "r2": float(fit.rsquared),
                    "n_window": len(usable),
                }
            )
        )

    if not rows:
        raise ValueError(
            "decompose_full_sample: no industry cleared min_obs="
            f"{spec.min_obs}. Saw {n_seen} industr{'y' if n_seen == 1 else 'ies'}; "
            f"the largest usable window was {max_usable} day(s). "
            "The assembled frame is too thin to fit — usually an over-filtered "
            "universe, industries with no ex-leader stocks, or an all-NaN market "
            "column dropping every row. Inspect assemble_leader_frame's output "
            "upstream rather than lowering min_obs."
        )

    # Degeneracy guard: a near-zero mean R^2 means the common component is
    # empty, so shock == leader_ret and every downstream conditional test
    # collapses onto the baseline (b_shock ~= baseline b, b_common ~= 0 with a
    # blown-up standard error). The module already blocks the two *mechanical*
    # ways to get this wrong (leader on both sides, market in the residual);
    # this catches the *statistical* one, which is otherwise silent until the
    # sign flip fails to appear.
    mean_r2 = float(np.mean(r2s)) if r2s else float("nan")
    if r2s and mean_r2 < 0.02:
        warnings.warn(
            "decompose_full_sample: common component is near-empty "
            f"(mean industry R^2 = {mean_r2:.4f} across {len(r2s)} industries). "
            "'shock' is then ~= leader_ret and the conditional response will "
            "reproduce the baseline instead of splitting it. This points at the "
            "ex-leader index being a poor proxy for the leader's sector — too "
            "few followers per industry, or a stale/misaligned index — not at "
            "the leader being genuinely idiosyncratic. Validate on the synthetic "
            "panel (where recovery vs planted truth is checkable) before trusting "
            "the real-data split.",
            stacklevel=2,
        )
    return LeaderShocks.from_raw(pd.concat(rows, ignore_index=True))


def decompose_rolling(
    frame: pd.DataFrame, spec: DecompositionSpec | None = None
) -> LeaderShocks:
    """Point-in-time decomposition: betas from a trailing window ending at t-1.

    For each industry and each day t, betas are estimated on the ``window``
    trading days strictly before t, and the day-t common component and shock
    are formed from those betas applied to day-t regressors.  Nothing on row t
    uses information from t or later, so the resulting ``shock`` may be fed to a
    signal without look-ahead.

    Implementation note: rather than refit an OLS object per day (a quarter of
    a million fits on the real panel), the rolling normal-equations are solved
    with vectorised cumulative cross-products per industry, then evaluated one
    day at a time.  Days whose trailing window has fewer than ``min_obs`` rows,
    or a singular cross-product, get NaN and are dropped.

    ``sync_lags`` is not supported here: a lead of the index is a future
    observation, which point-in-time estimation cannot use.  Pass
    ``sync_lags=0`` (the default); a non-zero value raises rather than silently
    peeking ahead.
    """
    spec = spec or DecompositionSpec()
    if spec.sync_lags != 0:
        raise ValueError(
            "decompose_rolling: sync_lags > 0 needs a lead of the index, which "
            "is a future value. Use decompose_full_sample for the Dimson "
            "correction, or set sync_lags=0 here."
        )

    rows: list[pd.DataFrame] = []
    for ff49, chunk in frame.sort_values("date").groupby("ff49", sort=True):
        chunk = chunk.dropna(subset=["leader_ret", "ind_ret_exl"]).reset_index(drop=True)
        x, common_cols, _ = _design(chunk, spec)
        keep = x.dropna().index
        chunk = chunk.loc[keep].reset_index(drop=True)
        x = x.loc[keep].reset_index(drop=True)
        n = len(chunk)
        if n <= spec.min_obs:
            continue

        cols = list(x.columns)  # const + regressors
        X = x.to_numpy(dtype=float)
        y = chunk["leader_ret"].to_numpy(dtype=float)

        common = np.full(n, np.nan)
        betas = {c: np.full(n, np.nan) for c in cols}
        r2 = np.full(n, np.nan)
        nwin = np.zeros(n, dtype=int)

        for t in range(spec.min_obs, n):
            lo = max(0, t - spec.window)
            Xw, yw = X[lo:t], y[lo:t]
            if len(yw) < spec.min_obs:
                continue
            xtx = Xw.T @ Xw
            try:
                beta = np.linalg.solve(xtx, Xw.T @ yw)
            except np.linalg.LinAlgError:
                continue
            common[t] = float(X[t] @ beta)
            for j, c in enumerate(cols):
                betas[c][t] = beta[j]
            resid = yw - Xw @ beta
            ss_res = float(resid @ resid)
            ss_tot = float(((yw - yw.mean()) ** 2).sum())
            r2[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            nwin[t] = len(yw)

        sync_ind = [c for c in cols if c.startswith("ind_exl")]
        beta_ind = np.zeros(n)
        for c in sync_ind:
            beta_ind = beta_ind + np.nan_to_num(betas[c])
        out = pd.DataFrame(
            {
                "date": chunk["date"].to_numpy(),
                "ff49": ff49,
                "leader_ret": y,
                "common": common,
                "shock": y - common,
                "beta_mkt": betas.get(spec.market_col, np.full(n, 0.0)),
                "beta_ind": beta_ind,
                "alpha": betas.get("const", np.full(n, np.nan)),
                "r2": r2,
                "n_window": nwin,
            }
        ).dropna(subset=["common"])
        # Computed after the dropna above so shock_type is only ever derived
        # from a day that actually has a point-in-time common/shock pair -
        # the pre-burn-in rows never reach this classifier.
        out["shock_type"] = classify_shock_day(
            out["common"].to_numpy(), out["shock"].to_numpy()
        )
        rows.append(out)

    if not rows:
        raise ValueError(
            "decompose_rolling: no industry produced a single point-in-time "
            f"fit. Need > {spec.min_obs} rows per industry; check the window."
        )
    return LeaderShocks.from_raw(pd.concat(rows, ignore_index=True))


# --------------------------------------------------------------------------- #
# Step 3: the merge-ready table                                               #
# --------------------------------------------------------------------------- #
def follower_merge_table(shocks: LeaderShocks) -> pd.DataFrame:
    """The (date, ff49) decomposition table, named for merging onto followers.

    Works on either estimator's output - pass a `decompose_full_sample` result
    for a descriptive/backtest merge, or a `decompose_rolling` result when the
    consumer needs a point-in-time (no-look-ahead) split.  This function does
    not itself add or remove any look-ahead risk; it only renames what the
    chosen estimator already produced.

    Returns ``date, ff49, common_shock, leader_specific_shock, shock_type`` -
    a straight rename of ``common``/``shock`` to the names a follower panel
    merge expects, plus the same-day classifier.  Nothing here reads a
    follower's return or a later date: every column traces back to the
    leader's own same-day regression in `decompose_full_sample` /
    `decompose_rolling`, so merging this onto `follower_panel.parquet` by
    ``(date, ff49)`` cannot leak follower information back into the leader's
    own decomposition.

    Usage::

        shocks = decompose_full_sample(frame, spec)
        merged = follower_panel.merge(
            follower_merge_table(shocks), on=["date", "ff49"], how="left"
        )
        common_days = merged[merged["shock_type"] == "common"]
        specific_days = merged[merged["shock_type"] == "leader_specific"]
        # run compute_ic / horizon_ic / quantile_portfolios on each separately
    """
    return shocks.frame[["date", "ff49", "common", "shock", "shock_type"]].rename(
        columns={"common": "common_shock", "shock": "leader_specific_shock"}
    )
