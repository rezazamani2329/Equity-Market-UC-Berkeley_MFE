"""
Raw daily rows plus delisting events -> one daily return series per stock that
does not stop at the last traded day.

Pure pandas and numpy; nothing here touches WRDS.  `wrds_fetch` deliberately
left both halves raw — `CRSPDaily.ret` exactly as the vendor serves it and
`CRSPDelist` as a separate event table — so that the joining, the repair of
missing delisting returns and the counting of each branch happen in one place
that can be tested.  This is that place.

    raw daily rows + delisting events
      1. attach_delisting       event -> the stock's row for that day
      2. append_orphan_events   an event whose day has NO daily row
      3. repair_missing_dlret   Shumway's substitute, flagged and counted
      4. combine                (1 + ret)(1 + dlret) - 1
                                                  ->  DailyReturns + report

Why step 2 is not optional.  CRSP's daily file usually stops the day BEFORE a
stock delists, and the universe filter (share code, exchange, name row valid
at the date) removes many of the remaining cases, so a plain left join reaches
only a minority of the events.  Those orphans are not missing data, they are
the days on which the worst outcomes were realised.  Dropping them is the
classic survivorship bias — the return series of a stock that went to zero
would simply end, and every cross-sectional moment computed from it would be
too high.

Why it matters *here* specifically.  This project's signal is a conditional
reversal in small followers.  Small followers are exactly the population that
delists, and a delisting is the largest negative return many of them ever
print.  A long-short book that is long small followers would look better than
it is if those days were missing; the short leg would look worse.  The bias
runs directly through the result the project is trying to establish, so the
correction is not hygiene, it is part of the answer.

Vocabulary
----------
* **delisting return (`dlret`)** — what a holder actually realised when the
  stock stopped trading: the final distribution or acquisition consideration
  against the previous close.  It replaces the missing part of the last day,
  so it MULTIPLIES with whatever partial return CRSP reports:
  1 + r = (1 + ret)(1 + dlret).
* **performance-related delisting** — codes 500 and 520-584 (dropped by the
  exchange, insufficient capital, bankruptcy, price too low, ...).  Codes in
  the 200s are mergers and acquisitions, 300s exchanges, 400s liquidations.
* **the Shumway repair** — Shumway (1997) shows CRSP's missing delisting
  returns are not missing at random: they cluster on performance-related
  delistings, where the realised return was very negative.  He estimates -30%
  for NYSE and AMEX; Shumway and Warther (1999) estimate -55% for NASDAQ,
  where the delisting process differs.  The substitute is applied only to
  performance codes with no reported `dlret`, is flagged in the `repaired`
  column, and is counted in the report — never silent.
* **source** — where a row's return comes from: `traded` (a normal day),
  `traded+delist` (the stock traded and then delisted that day),
  `delist_only` (an appended orphan: the delisting day with no daily row).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from lead_lag.data.typed_dataset import Dataset
from lead_lag.data.wrds_fetch import CRSPDaily, CRSPDelist

# Delisting codes that mean the stock failed rather than was bought: 500
# ("stopped trading, reason unavailable") and the 520-584 block (dropped by
# the exchange for price, capital, filing or bankruptcy reasons).
PERFORMANCE_CODES: frozenset[int] = frozenset({500} | set(range(520, 585)))

# Shumway (1997) for NYSE/AMEX, Shumway & Warther (1999) for NASDAQ.
SHUMWAY_NYSE_AMEX: float = -0.30
SHUMWAY_NASDAQ: float = -0.55


class DailyReturns(Dataset):
    """Daily returns with the delisting days folded in, one row per
    `(date, permno)`.

    Canonical columns:
        date        trading day (datetime64)
        permno      int
        exchcd      exchange code of the stock's last observed day
        siccd       SIC code valid at `date` — the industry map's input
        prc         close price, $ per share, NaN on an orphan row
        mktcap      dollars, NaN where prc is NaN
        vol         share volume as CRSP reports it
        dollar_vol  prc * vol, the liquidity filter's input
        ret_raw     CRSP's own daily return; NaN on an appended orphan row
        dlret       delisting return actually used, repair included; else NaN
        ret         the combined return, (1 + ret_raw)(1 + dlret) - 1
        source      'traded' | 'traded+delist' | 'delist_only'
        repaired    True where `dlret` is Shumway's substitute, not CRSP's

    `ret_raw` and `dlret` are kept beside `ret` so any application can undo the
    correction and show what it did or did not change; `repaired` makes the
    substituted rows droppable in one line for a robustness check — which is
    part 5's job and is easier if part 1 leaves the handle in place.
    """

    KEY: ClassVar[tuple[str, ...]] = ("date", "permno")
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "date", "permno", "exchcd", "siccd", "prc", "mktcap", "vol", "dollar_vol",
        "ret_raw", "dlret", "ret", "source", "repaired",
    )
    SYNONYMS: ClassVar[dict[str, str]] = {}

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        df["permno"] = df["permno"].astype("int64")
        for col in ("exchcd", "siccd"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        for col in ("prc", "mktcap", "vol", "dollar_vol", "ret_raw", "dlret", "ret"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source"] = df["source"].astype(str)
        df["repaired"] = df["repaired"].astype(bool)
        return df.sort_values(["date", "permno"]).reset_index(drop=True)


@dataclass(frozen=True)
class ReturnsReport:
    """What the delisting merge did, in numbers, for the cleaning appendix.

    Attributes:
        n_events_total:    delisting events in the requested window.
        n_events_matched:  events that landed on an existing daily row.
        n_events_orphan:   events appended as a new `delist_only` row.
        n_events_unused:   events for a permno outside the universe entirely.
        n_repaired:        rows where Shumway's substitute replaced a missing
                           `dlret` on a performance-related code.
        n_perf_events:     performance-related events among the used ones.
    """

    n_events_total: int
    n_events_matched: int
    n_events_orphan: int
    n_events_unused: int
    n_repaired: int
    n_perf_events: int

    def to_frame(self) -> pd.DataFrame:
        """One-column table, for printing under the cleaning section."""
        return pd.DataFrame(
            {"count": [
                self.n_events_total, self.n_events_matched, self.n_events_orphan,
                self.n_events_unused, self.n_perf_events, self.n_repaired,
            ]},
            index=[
                "delisting events in window", "matched to a traded day",
                "appended as delist-only row", "permno not in universe",
                "performance-related (used)", "dlret repaired (Shumway)",
            ],
        )


def _shumway_substitute(exchcd: pd.Series) -> pd.Series:
    """Shumway's delisting-return constant per exchange, as a Series.

    NASDAQ (exchcd 3) gets -55%; NYSE and AMEX (1, 2) get -30%.  Anything else
    — there should be nothing else after the universe filter — gets -30%, the
    more conservative of the two.
    """
    return pd.Series(
        np.where(exchcd == 3, SHUMWAY_NASDAQ, SHUMWAY_NYSE_AMEX),
        index=exchcd.index,
        dtype="float64",
    )


def adjusted_daily_returns(
    daily: CRSPDaily, delist: CRSPDelist, repair: bool = True
) -> tuple[DailyReturns, ReturnsReport]:
    """Merge delisting events into the daily panel and combine the returns.

    Args:
        daily: the raw daily panel from `wrds_fetch.load_or_fetch_crsp_daily`.
        delist: the delisting events from `load_or_fetch_crsp_delist`, covering
            at least the same window.
        repair: apply Shumway's substitute to performance-related events with
            no reported `dlret`.  False leaves those `dlret` as NaN, which
            makes `ret` fall back to `ret_raw` — the robustness variant.

    Returns:
        (DailyReturns, ReturnsReport).  The report is what goes in the
        appendix; do not throw it away.

    The four steps in the module docstring, in order.  Each one is a few lines
    of pandas, and each one is counted.
    """
    panel = daily.frame.copy()
    events = delist.frame.copy()

    # ---- step 1: attach the event to the stock's row for that day ---------
    # A delisting has a DATE, and the daily file has a row for that date only
    # if the stock traded.  The join is therefore on (permno, date) exactly —
    # no date_trunc as in the monthly case, where the event date is not the
    # month end.
    universe_permnos = set(panel["permno"].unique())
    used = events[events["permno"].isin(universe_permnos)].copy()
    n_unused = len(events) - len(used)

    # A permno can carry more than one delisting row (a delisting later
    # corrected or re-dated).  Collapse to one row per (permno, date) before
    # the join, or the left join FANS OUT the daily row into two and breaks the
    # (date, permno) key.  `max` over dlret keeps the non-null value when one
    # of the duplicates is null.
    used = (
        used.groupby(["permno", "dlstdt"], as_index=False)
        .agg(dlret=("dlret", "max"), dlstcd=("dlstcd", "max"))
        .rename(columns={"dlstdt": "date"})
    )

    merged = panel.merge(used, on=["permno", "date"], how="left")
    n_matched = int(merged["dlstcd"].notna().sum())

    # ---- step 2: append the orphans --------------------------------------
    # An event whose (permno, date) found no daily row: the stock's last
    # traded day was earlier.  It becomes a row of its own with `ret_raw` NaN,
    # carrying the stock's last known exchange and industry so downstream
    # group-bys still place it.
    matched_keys = set(
        map(tuple, merged.loc[merged["dlstcd"].notna(), ["permno", "date"]].to_numpy())
    )
    orphan_mask = ~used.apply(lambda r: (r["permno"], r["date"]) in matched_keys, axis=1) \
        if len(used) else pd.Series(dtype=bool)
    orphans = used.loc[orphan_mask].copy() if len(used) else used.copy()
    n_orphan = len(orphans)

    if n_orphan:
        # Last observed attributes per permno — the stock's identity at exit.
        last = (
            panel.sort_values("date")
            .groupby("permno")
            .agg(exchcd=("exchcd", "last"), siccd=("siccd", "last"))
            .reset_index()
        )
        orphans = orphans.merge(last, on="permno", how="left")
        for col in ("prc", "mktcap", "vol", "ret", "retx"):
            orphans[col] = float("nan")
        merged = pd.concat([merged, orphans], ignore_index=True)

    # ---- step 3: repair missing delisting returns ------------------------
    is_perf = merged["dlstcd"].isin(PERFORMANCE_CODES)
    n_perf = int(is_perf.sum())
    needs_repair = is_perf & merged["dlret"].isna()
    merged["repaired"] = False
    if repair and needs_repair.any():
        merged.loc[needs_repair, "dlret"] = _shumway_substitute(
            merged.loc[needs_repair, "exchcd"]
        )
        merged.loc[needs_repair, "repaired"] = True
    n_repaired = int(merged["repaired"].sum())

    # ---- step 4: combine -------------------------------------------------
    # (1 + ret)(1 + dlret) - 1, with each factor defaulting to 1 when its
    # return is missing.  `fillna(0)` on the return is the same thing as
    # `fillna(1)` on the gross factor and reads closer to the formula.
    gross_traded = 1.0 + merged["ret"].fillna(0.0)
    gross_delist = 1.0 + merged["dlret"].fillna(0.0)
    combined = gross_traded * gross_delist - 1.0
    # A row with neither a traded return nor a delisting return has no return
    # at all; the arithmetic above would call it 0.0, which is a made-up
    # observation.  Put the NaN back.
    both_missing = merged["ret"].isna() & merged["dlret"].isna()
    combined = combined.mask(both_missing)

    merged["source"] = np.select(
        [
            merged["ret"].isna() & merged["dlstcd"].notna(),   # orphan
            merged["ret"].notna() & merged["dlstcd"].notna(),  # traded then delisted
        ],
        ["delist_only", "traded+delist"],
        default="traded",
    )

    out = pd.DataFrame(
        {
            "date": merged["date"],
            "permno": merged["permno"],
            "exchcd": merged["exchcd"],
            "siccd": merged["siccd"],
            "prc": merged["prc"],
            "mktcap": merged["mktcap"],
            "vol": merged["vol"],
            # Dollar volume, the liquidity filter's input.  Computed here
            # rather than in universe.py so that every consumer of
            # DailyReturns sees the same definition.
            "dollar_vol": merged["prc"] * merged["vol"],
            "ret_raw": merged["ret"],
            "dlret": merged["dlret"],
            "ret": combined,
            "source": merged["source"],
            "repaired": merged["repaired"],
        }
    )

    report = ReturnsReport(
        n_events_total=len(events),
        n_events_matched=n_matched,
        n_events_orphan=n_orphan,
        n_events_unused=n_unused,
        n_repaired=n_repaired,
        n_perf_events=n_perf,
    )
    return DailyReturns.from_raw(out), report
