"""
Who leads and who follows, industry by industry, month by month.

The economic claim the project rests on is that information arrives at large,
visible firms first and reaches smaller firms in the same industry with a
delay (Lo and MacKinlay 1990 on cross-autocorrelation; Hou 2007 on the
intra-industry version).  Turning that into data means committing to three
things, and each one is an assumption a referee will push on, so each is a
parameter here rather than a hard-coded choice:

1. **What makes a stock the leader.**  Default: the largest formation-window
   market cap in its industry.  Size is the standard proxy for visibility and
   analyst coverage, it is observable for every stock without extra data, and
   it is what `LeaderRule.by` can be swapped away from — `dollar_vol` and
   `turnover` are the two alternatives the robustness section should report.
   If the leader relationship only exists under one definition, it is a
   property of that definition, not of the market.

2. **How many leaders.**  Default: one per industry.  `n_leaders > 1` builds
   an equal-weighted leader portfolio instead, which is less noisy — a single
   stock's idiosyncratic day is a large part of its return — at the cost of
   blurring exactly the leader-specific shock part 2 needs to isolate.  Both
   are worth reporting.

3. **Who counts as a follower.**  Default: every other eligible stock in the
   industry.  `follower_max_size_pct` narrows it to the smaller firms the
   hypothesis is actually about (e.g. 0.5 keeps the smaller half), which
   sharpens the test but shrinks the cross-section.

Point-in-time, again
--------------------
Everything here is decided from `Universe`, whose statistics were already
shifted to the end of the previous month.  A stock is December's leader
because it was the biggest through November — not because it was the biggest
in December, which would mean the leader was chosen partly by December's own
returns.  That is a subtle lookahead and it biases in the direction of the
hypothesis: a stock that is largest *at the end of* December is
disproportionately one that went up during December.

Stability
---------
A monthly rebalance lets the leader change from month to month.  Real leaders
do not, mostly — `leader_turnover` reports how often the assignment actually
changes, and a sensible number is low single-digit percent per month.  A high
number means the size ranking is being decided by noise, which usually means
the industry has two firms of nearly equal size and the choice between them is
arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from lead_lag.data.typed_dataset import Dataset
from lead_lag.data.universe import Universe


@dataclass(frozen=True)
class LeaderRule:
    """How leaders and followers are picked.

    Attributes:
        by:                    formation statistic to rank on — "mktcap"
                               (default), "med_dollar_vol", or any other
                               numeric column of `Universe`.
        n_leaders:             how many top-ranked stocks form the leader;
                               >1 gives an equal-weighted leader portfolio.
        follower_max_size_pct: keep only followers below this within-industry
                               size percentile (1.0 = all non-leaders).
        min_followers:         industry-months with fewer eligible followers
                               than this are dropped — a lead-lag regression
                               on two followers is noise.
    """

    by: str = "mktcap"
    n_leaders: int = 1
    follower_max_size_pct: float = 1.0
    min_followers: int = 4


class LeaderMap(Dataset):
    """Role assignment, one row per (month, permno).

    Canonical columns:
        month            eligible month (datetime64)
        permno           int
        ff49             industry number
        role             'leader' | 'follower'
        rank_in_industry 1 = largest on the ranking statistic
        size_pct         within-industry percentile of formation market cap
        n_in_industry    eligible stocks in this industry-month

    Only stocks with a role are present: an eligible stock in an industry that
    failed `min_followers`, or a non-leader above `follower_max_size_pct`, is
    absent rather than flagged, because every consumer of this table wants
    "the stocks with a role" and would otherwise have to filter it again.
    """

    KEY: ClassVar[tuple[str, ...]] = ("month", "permno")
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "month", "permno", "ff49", "role", "rank_in_industry", "size_pct",
        "n_in_industry",
    )
    SYNONYMS: ClassVar[dict[str, str]] = {}

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["month"] = pd.to_datetime(df["month"])
        for col in ("permno", "ff49", "rank_in_industry", "n_in_industry"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        df["size_pct"] = pd.to_numeric(df["size_pct"], errors="coerce")
        df["role"] = df["role"].astype(str)
        return df.sort_values(["month", "ff49", "rank_in_industry"]).reset_index(drop=True)


def assign_roles(universe: Universe, rule: LeaderRule | None = None) -> LeaderMap:
    """Label every eligible stock a leader or a follower of its industry.

    Args:
        universe: output of `universe.build_universe`; only rows with
            `eligible = True` are considered.
        rule: the three choices described in the module docstring.

    Returns:
        LeaderMap.  Industries that end up with fewer than `min_followers`
        followers are dropped entirely, leader included — a leader with no
        followers contributes nothing to the regression and would otherwise
        inflate the industry count in the summary table.
    """
    rule = rule or LeaderRule()
    df = universe.frame.loc[universe.frame["eligible"]].copy()
    if rule.by not in df.columns:
        raise ValueError(
            f"assign_roles: ranking column {rule.by!r} is not in Universe "
            f"({sorted(df.columns)})"
        )

    grp = df.groupby(["month", "ff49"], sort=False)
    # rank 1 = largest.  `method="first"` breaks ties by row order rather than
    # averaging, so ranks stay integers and exactly `n_leaders` stocks are
    # labelled leader even when two firms report identical market caps.
    df["rank_in_industry"] = grp[rule.by].rank(ascending=False, method="first").astype("int64")
    df["size_pct"] = grp["mktcap"].rank(pct=True)
    df["n_in_industry"] = grp["permno"].transform("nunique")

    df["role"] = "follower"
    df.loc[df["rank_in_industry"] <= rule.n_leaders, "role"] = "leader"

    # Narrow the follower set to the smaller firms, if asked.
    too_big = (df["role"] == "follower") & (df["size_pct"] > rule.follower_max_size_pct)
    df = df.loc[~too_big]

    # Drop industry-months without enough followers left.
    n_followers = (
        df.assign(is_f=df["role"] == "follower")
        .groupby(["month", "ff49"])["is_f"].transform("sum")
    )
    df = df.loc[n_followers >= rule.min_followers]

    return LeaderMap.from_raw(
        df[["month", "permno", "ff49", "role", "rank_in_industry", "size_pct",
            "n_in_industry"]]
    )


def leader_returns(
    daily: pd.DataFrame, roles: LeaderMap, ret_col: str = "ret"
) -> pd.DataFrame:
    """Daily leader return per industry: one series per (date, ff49).

    With `n_leaders = 1` this is just that stock's return.  With more, it is
    the equal-weighted average of the leader portfolio — equal-weighted rather
    than value-weighted on purpose, so the largest firm does not dominate a
    portfolio whose whole point was to average away one firm's noise.

    Returns columns: date, ff49, leader_ret, n_leaders_obs.
    """
    df = daily.copy()
    df["month"] = df["date"].values.astype("datetime64[M]")
    leaders = roles.frame.loc[roles.frame["role"] == "leader", ["month", "permno", "ff49"]]
    merged = df.merge(leaders, on=["month", "permno"], how="inner", suffixes=("", "_role"))
    # `ff49` may exist on both sides; the role table's is authoritative.
    ind_col = "ff49_role" if "ff49_role" in merged.columns else "ff49"

    out = (
        merged.dropna(subset=[ret_col])
        .groupby(["date", ind_col])
        .agg(leader_ret=(ret_col, "mean"), n_leaders_obs=("permno", "nunique"))
        .reset_index()
        .rename(columns={ind_col: "ff49"})
    )
    return out


def leader_turnover(roles: LeaderMap) -> pd.DataFrame:
    """How often each industry's leader set changes from one month to the next.

    Returns one row per (month, ff49) with `changed` — True when the set of
    leader permnos differs from the previous month's — plus the overall rate
    as the frame's `.attrs["overall_rate"]`.

    A diagnostic, not a filter.  Read it before trusting any leader-based
    result: if the leader changes every other month, the "leader" is a
    statistical artefact of two similarly sized firms trading places, and the
    lead-lag coefficient is measuring the ranking noise as much as the
    information flow.
    """
    leaders = roles.frame.loc[roles.frame["role"] == "leader"]
    sets = (
        leaders.groupby(["ff49", "month"])["permno"]
        .apply(lambda s: frozenset(s.tolist()))
        .reset_index(name="leader_set")
        .sort_values(["ff49", "month"])
    )
    sets["prev"] = sets.groupby("ff49")["leader_set"].shift(1)
    sets["changed"] = (sets["prev"].notna()) & (sets["leader_set"] != sets["prev"])

    out = sets[["month", "ff49", "changed"]].reset_index(drop=True)
    comparable = sets["prev"].notna()
    out.attrs["overall_rate"] = (
        float(sets.loc[comparable, "changed"].mean()) if comparable.any() else float("nan")
    )
    return out


def follower_panel(
    daily: pd.DataFrame, roles: LeaderMap, ret_col: str = "ret"
) -> pd.DataFrame:
    """The regression panel: every follower-day with its industry's leader return.

    Columns: date, permno, ff49, month, ret (the follower's), leader_ret
    (same day), `size_pct`, and the **lagged** characteristics
    `mktcap_lag`, `dollar_vol_lag`, `prc_lag`, `exchcd_lag`.

    Everything here that is not a day-d return is a d-1 object
    -----------------------------------------------------------
    Two returns are day d's by construction: the follower's `ret` (the
    left-hand side) and `leader_ret` (the thing `baseline.build_lags` will
    lag).  Every other column is information available at the previous close:

    * `ff49` comes from the role table, which was built from the universe,
      which was built from the month's formation window — so it is the
      industry the stock belonged to at the end of the PREVIOUS month, not
      the one CRSP may have moved it to this morning.
    * `mktcap_lag`, `dollar_vol_lag`, `prc_lag`, `exchcd_lag` are the previous
      observed day's values, attached by
      `information_set.attach_lagged`.  The same-day versions are dropped
      rather than renamed: a signal that wants market cap must take the
      lagged one, and leaving `mktcap` in the frame beside `mktcap_lag` is an
      invitation to reach for the wrong one at 2am.

    `daily` must therefore have been through `attach_lagged` first; the
    function raises a clear error if it has not.

    The LAGGING OF RETURNS is not done here — `baseline.py` owns the choice of
    how many days to lag and whether to use overlapping windows, because that
    choice is the experiment rather than the data.  What this function
    guarantees is that a follower is never its own leader (the leader's rows
    are excluded) and that both returns come from the same trading day, so a
    lag of k means exactly k sessions.
    """
    lagged_extras = ("mktcap_lag", "dollar_vol_lag", "prc_lag", "exchcd_lag")
    missing = [c for c in lagged_extras if c not in daily.columns]
    if missing:
        raise KeyError(
            f"follower_panel: {missing} not in the daily frame. Run "
            f"`information_set.attach_lagged(panel)` first — the regression "
            f"panel carries d-1 characteristics, never day-d ones."
        )

    df = daily.copy()
    df["month"] = df["date"].values.astype("datetime64[M]")

    followers = roles.frame.loc[
        roles.frame["role"] == "follower", ["month", "permno", "ff49", "size_pct"]
    ]
    panel = df.drop(columns=["ff49"], errors="ignore").merge(
        followers, on=["month", "permno"], how="inner"
    )

    lead = leader_returns(daily, roles, ret_col=ret_col)
    panel = panel.merge(lead, on=["date", "ff49"], how="left")

    cols = ["date", "permno", "ff49", "month", ret_col, "leader_ret", "size_pct"]
    extra = [c for c in lagged_extras if c in panel.columns]
    if "days_since_prev" in panel.columns:
        # Staleness of the lagged characteristics, so a consumer can drop rows
        # whose "previous day" is three weeks old.
        extra = [*extra, "days_since_prev"]
    return panel[cols + extra].sort_values(["permno", "date"]).reset_index(drop=True)
