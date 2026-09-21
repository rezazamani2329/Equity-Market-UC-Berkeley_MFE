"""
From the CRSP design to a Databento request: which tickers, on which venue,
over which windows.

The intraday leg tests the same leaders and followers as the daily leg — that
is the point of it.  The daily panel says the leader-follower effect was
strong through 2006 and absent afterwards; the intraday question is whether it
disappeared or merely moved to a horizon a daily panel cannot resolve.  That
comparison only means anything if both legs study the same stocks, so the
symbol list is derived from `LeaderMap`, not chosen afresh.

Three translation problems, all of which bite silently
-------------------------------------------------------
**1. permno -> ticker is not a function.**  CRSP keys on permno; Databento
keys on the trading symbol.  A stock's ticker changes (and 4,439 tickers in
this sample have been used by more than one permno — symbols get recycled
aggressively).  So the mapping is per-interval, and this module emits one row
per `(permno, ticker, valid-from, valid-to)` span rather than a dictionary.
Requesting a whole date range under today's symbol would silently pull some
other company's data for the earlier part of it.

**2. Venue.**  The 2018-2026 Databento datasets chosen for this project are
per-venue, not consolidated.  A stock's quotes live in its primary listing
venue's feed:

    exchcd 1  NYSE            -> XNYS.PILLAR
    exchcd 2  NYSE American   -> XASE.PILLAR
    exchcd 3  Nasdaq          -> XNAS.ITCH

The BBO from a stock's primary venue is a close approximation to the
consolidated NBBO for that stock, but it is an approximation: during the
session other venues can be inside the primary's quote.  This is a documented
limitation of the long-history option, and the validation overlay for it is a
consolidated re-run on the 2023+ DBEQ.BASIC sample.

**3. Membership changes monthly.**  Leader and follower assignment is
rebalanced monthly, so the symbol set for 2018-2026 is the UNION of the
monthly sets, with each symbol's active months recorded.  Pulling the union
over the whole window and filtering afterwards is far cheaper than 100
separate monthly requests, but the `active_months` column has to come along
or the intraday panel will contain stock-days on which the stock was not in
the universe.
"""

from __future__ import annotations

import pandas as pd

from lead_lag.data.wrds_fetch import CRSPNames

# CRSP exchange code -> the Databento dataset carrying that venue's book.
VENUE_BY_EXCHCD: dict[int, str] = {
    1: "XNYS.PILLAR",   # NYSE
    2: "XASE.PILLAR",   # NYSE American (formerly AMEX)
    3: "XNAS.ITCH",     # Nasdaq
}

# Databento's US equity history begins here for the per-venue datasets.
DATABENTO_EQUITY_START = pd.Timestamp("2018-05-01")


def select_intraday_symbols(
    roles: pd.DataFrame,
    universe: pd.DataFrame,
    names: CRSPNames,
    start: str = "2018-05-01",
    end: str = "2026-09-01",
    n_followers: int = 20,
) -> pd.DataFrame:
    """Every leader plus the `n_followers` largest followers per industry-month.

    Args:
        roles: `LeaderMap.frame` — month, permno, ff49, role, size_pct.
        universe: `Universe.frame` — supplies the formation market cap that
            ranks followers.  Note this is the d-1-safe one: followers are
            ranked on the previous month's close, like everything else.
        names: the CRSP name history, for point-in-time tickers.
        start, end: the intraday window.  `start` is clamped to Databento's
            own history start, so asking for 2015 quietly gets 2018-05-01
            rather than an empty pull.
        n_followers: how many followers per industry-month.

    Returns:
        One row per `(permno, ticker, venue, valid_from, valid_to)` with:
            permno, ticker, comnam, venue, ff49, role,
            valid_from, valid_to       the ticker span, clipped to the window
            active_months              months this permno held the role
            first_month, last_month

    Why the largest followers and not a random sample.  The strategy has to be
    tradeable; the smallest followers in an industry are the ones whose spread
    would eat the entire effect.  Taking the top 20 by size biases toward
    finding LESS reversion (bigger names are more efficiently priced), which
    is the conservative direction — a result that survives here is not an
    artefact of picking illiquid names.  That asymmetry is worth stating in
    the report rather than leaving for a referee to notice.
    """
    window_start = max(pd.Timestamp(start), DATABENTO_EQUITY_START)
    window_end = pd.Timestamp(end)

    df = roles.loc[roles["month"] >= window_start].copy()
    if df.empty:
        raise ValueError(
            f"select_intraday_symbols: no role assignments at or after "
            f"{window_start.date()} — the daily pipeline may not have been run "
            f"over a period Databento covers."
        )

    # Rank followers by the formation market cap (previous month's close).
    caps = universe.loc[universe["eligible"], ["month", "permno", "mktcap", "exchcd"]]
    df = df.merge(caps, on=["month", "permno"], how="left")

    followers = df.loc[df["role"] == "follower"].copy()
    followers["cap_rank"] = followers.groupby(["month", "ff49"])["mktcap"].rank(
        ascending=False, method="first"
    )
    keep = pd.concat(
        [df.loc[df["role"] == "leader"], followers.loc[followers["cap_rank"] <= n_followers]],
        ignore_index=True,
    )

    # Collapse the monthly rows to one record per permno.
    #
    # A stock is frequently BOTH: leader turnover is ~5% a month, so over eight
    # years an industry has several leaders, and a stock that led for ten
    # months and followed for forty is genuinely both.  An earlier version
    # resolved this by keeping whichever role the stock held more often, which
    # silently deleted leaders — an industry whose leader spent most of the
    # window as a follower ended up with no leader in the symbol list at all,
    # and the list covered 38 industries instead of 48.
    #
    # The fix is to stop treating role as a partition.  The symbol list needs
    # the UNION of stocks to pull; which role each held, and for how long, is
    # metadata that both flags can carry.
    keep["is_leader"] = keep["role"] == "leader"
    per_permno = (
        keep.groupby("permno")
        .agg(
            ff49=("ff49", "last"),
            exchcd=("exchcd", "last"),
            months_leader=("is_leader", "sum"),
            active_months=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
        )
        .reset_index()
    )
    per_permno["months_follower"] = (
        per_permno["active_months"] - per_permno["months_leader"]
    )
    per_permno["ever_leader"] = per_permno["months_leader"] > 0
    per_permno["ever_follower"] = per_permno["months_follower"] > 0
    per_permno["dual_role"] = per_permno["ever_leader"] & per_permno["ever_follower"]
    # `role` is kept as the PREDOMINANT role, for readable tables only — every
    # join downstream should use `ever_leader` / `ever_follower`.
    per_permno["role"] = pd.Series(
        ["leader" if x else "follower" for x in
         per_permno["months_leader"] >= per_permno["months_follower"]],
        index=per_permno.index,
    )

    # ---- point-in-time tickers ------------------------------------------
    # One output row per (permno, name span) overlapping the window, so the
    # fetch layer requests each symbol only over the dates it was that stock's.
    spans = names.frame[["permno", "namedt", "nameendt", "ticker", "comnam"]]
    out = per_permno.merge(spans, on="permno", how="left")
    out = out.loc[(out["nameendt"] >= window_start) & (out["namedt"] <= window_end)]
    out["valid_from"] = out["namedt"].clip(lower=window_start)
    out["valid_to"] = out["nameendt"].clip(upper=window_end)

    out["venue"] = out["exchcd"].map(VENUE_BY_EXCHCD)

    missing_venue = out["venue"].isna().sum()
    if missing_venue:
        # exchcd 0 or negative means the stock was not on one of the three
        # exchanges at formation — it should have been filtered by the
        # universe, so this is worth surfacing rather than dropping quietly.
        out = out.loc[out["venue"].notna()]

    out = out.loc[out["ticker"].str.len() > 0]

    # CRSP opens a new name row whenever ANY of its fields changes — share
    # code, exchange, SIC — so one stock can have several consecutive spans
    # under the SAME ticker.  For a Databento request those are one span;
    # leaving them split produced duplicate rows (KO three times, NFLX three
    # times) and would have requested the same symbol-window repeatedly.
    # Collapse to the outer envelope per (permno, ticker).
    out = (
        out.groupby(["permno", "ticker"], as_index=False)
        .agg(
            comnam=("comnam", "last"),
            venue=("venue", "last"),
            ff49=("ff49", "last"),
            role=("role", "last"),
            ever_leader=("ever_leader", "last"),
            ever_follower=("ever_follower", "last"),
            dual_role=("dual_role", "last"),
            months_leader=("months_leader", "last"),
            months_follower=("months_follower", "last"),
            valid_from=("valid_from", "min"),
            valid_to=("valid_to", "max"),
            active_months=("active_months", "last"),
            first_month=("first_month", "last"),
            last_month=("last_month", "last"),
        )
    )

    cols = [
        "permno", "ticker", "comnam", "venue", "ff49", "role", "ever_leader",
        "ever_follower", "dual_role", "months_leader", "months_follower",
        "valid_from", "valid_to", "active_months", "first_month", "last_month",
    ]
    out = out[cols].sort_values(["ff49", "role", "ticker"]).reset_index(drop=True)
    out.attrs["dropped_no_venue"] = int(missing_venue)
    return out


def requests_by_venue(symbols: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Group the symbol table into one Databento request per venue.

    Returns `{venue: {"symbols": [...], "start": Timestamp, "end": Timestamp}}`.

    One request per venue rather than per symbol: Databento bills and streams
    by volume, and a single multi-symbol request over the full window is far
    cheaper in round trips than 550 individual ones.  The per-symbol
    `valid_from` / `valid_to` spans are applied when the returned data is
    filtered, not when it is requested — Databento resolves `raw_symbol` per
    day against its own reference data, so a symbol that changed hands mid-
    window returns the right instrument for each day, but the ROWS for the
    days it belonged to another company still have to be dropped against this
    table.  That filtering is not optional; see the module docstring.
    """
    out: dict[str, dict[str, object]] = {}
    for venue, chunk in symbols.groupby("venue"):
        out[str(venue)] = {
            "symbols": sorted(chunk["ticker"].unique().tolist()),
            "start": chunk["valid_from"].min(),
            "end": chunk["valid_to"].max(),
            "n_permnos": int(chunk["permno"].nunique()),
        }
    return out


def summarise(symbols: pd.DataFrame) -> pd.DataFrame:
    """Counts by venue and role — the table that goes in the data section."""
    rows = []
    for venue, chunk in symbols.groupby("venue"):
        rows.append({
            "venue": venue,
            "symbols": chunk["ticker"].nunique(),
            "permnos": chunk["permno"].nunique(),
            "industries": chunk["ff49"].nunique(),
            "ever_leader": int(chunk["ever_leader"].sum()),
            "ever_follower": int(chunk["ever_follower"].sum()),
            "dual_role": int(chunk["dual_role"].sum()),
        })
    return pd.DataFrame(rows)
