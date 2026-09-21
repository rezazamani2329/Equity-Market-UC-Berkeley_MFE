"""
Every raw pull this project makes, and the parquet cache in front of it.

Scope of this module: talk to WRDS, hand back schema-checked datasets, write
them to `cache.nosync/` so nobody pulls the same year twice.  It does no cleaning and
no economics — the delisting merge lives in `daily_returns.py`, the industry
assignment in `industry_map.py`, the filters in `universe.py`.  That split is
deliberate: fetching is the slow, non-deterministic, credentialed step, and
keeping it separate means the rest of part 1 can be developed and tested
offline against a cached parquet.

What gets pulled
----------------
    crsp.dsf + crsp.dsenames   ->  CRSPDaily       daily prices and returns
    crsp.dsedelist             ->  CRSPDelist      delisting events
    ff.fivefactors_daily       ->  FactorsDaily    FF5 + momentum + rf, daily

Why daily and not monthly.  The project's hypothesis is about SHORT-horizon
continuation and reversal in followers after a leader moves.  At monthly
frequency the distinction the whole thesis rests on — continuation over a few
days versus reversal over a few weeks — is invisible: both live inside one
observation.  The cost is size, which is what the per-year cache is for.

Size, so the first run is not a surprise
----------------------------------------
`crsp.dsf` is roughly 250 trading days x 4,000-7,000 stocks per year, so about
1.0-1.7 million rows a year.  A 1990-2024 pull is ~50 million rows: far too
much for one query and one frame, comfortable one year at a time (~120 MB as
parquet per year).  Hence `load_or_fetch_crsp_daily`, which loops over years
and concatenates only the years asked for.

Vocabulary
----------
* **permno** — CRSP's permanent stock identifier.  Survives ticker and name
  changes, which is why every join in this project keys on it rather than on
  a ticker.
* **name row** — `crsp.dsenames` records a stock's share code, exchange, SIC
  code and CUSIP over a date RANGE (`namedt` .. `nameendt`).  Joining on the
  row valid at the observation date is what makes the industry assignment
  point-in-time rather than as-of-today; getting this wrong is a lookahead
  bug that quietly inflates every result.
* **share code (shrcd)** — 10 and 11 are ordinary common stock of US firms.
  Excluding everything else drops ADRs, REITs, closed-end funds and units,
  which have their own return dynamics and would pollute an industry leader.
* **exchange code (exchcd)** — 1 NYSE, 2 AMEX, 3 NASDAQ.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from lead_lag.data.typed_dataset import Dataset
from lead_lag.data.wrds_connection import PROJECT_ROOT  # also loads .env into os.environ

# Default cache location: <project2>/cache.nosync/ (created on first write).
#
# The `.nosync` suffix is not decoration.  This repo lives in iCloud Drive, and
# a full-sample daily CRSP pull is several gigabytes of parquet; without the
# suffix macOS would upload every year file to iCloud and back down to every
# other device.  The same convention is used for `.venv.nosync`.
CACHE_DIR = PROJECT_ROOT / "cache.nosync"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class WRDSSession:
    """An open WRDS session: the `wrds.Connection` plus its raw DBAPI connection.

    Why both.  The `wrds` package handles login (username + `~/.pgpass`) and
    owns the SQLAlchemy engine; but its `raw_sql` helper breaks under pandas 3
    (pandas asks the SQLAlchemy connection for a `.cursor()` it does not
    have).  `engine.raw_connection()` hands back the underlying psycopg2
    connection, which pandas reads from happily.  Keeping the `wrds.Connection`
    alive on the dataclass keeps the engine (and its pool) alive too.
    """

    db: Any          # wrds.Connection — typed Any: the package ships no stubs
    raw: Any         # DBAPI connection from db.engine.raw_connection()

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run one parameterised SELECT and return the result as a DataFrame.

        Parameters use psycopg2's `%(name)s` placeholders; values are sent
        separately from the SQL text, so dates and strings are never formatted
        into the query string.

        The warning filter suppresses pandas' "only supports SQLAlchemy
        connectable" notice.  Handing pandas the raw psycopg2 connection is
        deliberate (see the class docstring — its SQLAlchemy path is broken
        under pandas 3), the warning fires once per query, and in a notebook it
        buries the progress lines that say how the pull is going.  Scoped to
        this call so it cannot hide a warning from anywhere else.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable",
                category=UserWarning,
            )
            return pd.read_sql_query(sql, self.raw, params=params)

    def close(self) -> None:
        """Release the server-side session.  WRDS caps concurrent sessions per
        user, so a leaked one eventually locks you out."""
        self.raw.close()
        self.db.close()


def open_wrds(username: str | None = None) -> WRDSSession:
    """Log in to WRDS and return a session.

    The username comes from the argument, else `$WRDS_USERNAME` (loaded from
    `.env` by `wrds_connection` at import).  The password is never handled
    here: the `wrds` package reads it from `~/.pgpass` for that username.
    Raises if no username is available — better than the package's interactive
    prompt, which hangs a non-interactive run (and a notebook cell).
    """
    import wrds  # imported lazily: the module stays importable without the package

    username = username or os.environ.get("WRDS_USERNAME")
    if not username:
        raise RuntimeError("open_wrds: no username given and $WRDS_USERNAME is not set")
    db: Any = wrds.Connection(wrds_username=username)
    return WRDSSession(db=db, raw=db.engine.raw_connection())


# ---------------------------------------------------------------------------
# Small date helpers
# ---------------------------------------------------------------------------

def years_in_range(start: str, end: str) -> list[int]:
    """Every calendar year touched by [start, end], inclusive.

    The daily cache is one file per calendar year, so a request for
    2015-06-01 .. 2017-03-31 has to read three files and then trim.  This is
    the list it loops over.
    """
    return list(range(date.fromisoformat(start).year, date.fromisoformat(end).year + 1))


def _trim(df: pd.DataFrame, start: str, end: str, column: str = "date") -> pd.DataFrame:
    """Keep rows with `start <= column <= end`.  Used after reading whole-year
    cache files for a request that starts or ends mid-year."""
    mask = (df[column] >= pd.Timestamp(start)) & (df[column] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CRSP daily stock file
# ---------------------------------------------------------------------------

class CRSPDaily(Dataset):
    """Daily common-stock observations, one row per (date, permno).

    Canonical columns:
        date     trading day (datetime64)                    — time key
        permno   int                                         — asset id
        permco   int, company id: several permnos can share one (dual-class)
        ncusip   8-char CUSIP valid at `date`                 — provenance
        shrcd    10 or 11                                     — kept for audit
        exchcd   1 NYSE / 2 AMEX / 3 NASDAQ                   — NYSE breakpoints
        siccd    4-digit SIC valid at `date`                  — industry input
        prc      close price, > 0 or NaN, $ per share
        shrout   shares outstanding, in SHARES (not thousands)
        mktcap   prc * shrout, in dollars (NaN when prc is NaN)
        ret      CRSP daily holding-period return, RAW        — may be NaN
        retx     the same return excluding dividends          — may be NaN
        vol      share volume as CRSP reports it              — see caveat
        numtrd   number of trades (NASDAQ only, NaN elsewhere)
        bid      closing bid, NaN when unavailable
        ask      closing ask, NaN when unavailable
        openprc  opening price, NaN when unavailable
        quote_only  True when CRSP reported no closing TRADE and `prc` is
                 (minus) the bid-ask midpoint — the non-synchronous-trading
                 marker; see below
        cfacpr   cumulative price-adjustment factor  — REQUIRED for any
                 return computed from raw prices (see below)
        cfacshr  cumulative share-adjustment factor

    Why `cfacpr` is not optional.  `prc`, `bid` and `ask` are the RAW prints:
    a 2-for-1 split halves them overnight.  CRSP's own `ret` is already
    adjusted, but anything computed from the price columns — notably the
    bid-ask midpoint return, which this project needs to defuse bounce — must
    divide by `cfacpr` first, or every split becomes a -50% observation.  In a
    study of short-horizon reversal those observations would not be noise,
    they would be the result.

    Three deliberate choices.

    *Rows with no price are KEPT.*  A day with a missing return is itself
    information: a lead-lag design lines day t-1 up against day t, and it has
    to know an observation is absent rather than silently treat the next
    available day as adjacent.  A non-positive `prc` (CRSP writes 0 for "no
    price this day") becomes NaN, and `mktcap` with it, so no cross-sectional
    mean is dragged toward zero.

    *A negative price is a quote midpoint, not a negative price.*  CRSP marks
    a day with no trade by storing the NEGATIVE of the bid-ask midpoint.  The
    SQL takes `abs(prc)` and records the sign separately as `quote_only`,
    because that sign is the cleanest available marker of NON-SYNCHRONOUS
    TRADING: the stock's "close" is a quote struck at some unknown earlier
    moment, not a print at 16:00.  A lead-lag design is acutely sensitive to
    this — if the follower's close is stale relative to the leader's, part of
    the measured lead-lag is that staleness rather than information diffusion,
    and it is concentrated in exactly the small illiquid names the hypothesis
    is about.  Keeping only `abs(prc)` would have made that untestable.

    *The delisting return is NOT applied.*  `ret` stays exactly what CRSP
    serves.  Combining it with the delisting return is a transform, and lives
    in `daily_returns.py`.

    Caveat on `vol`.  CRSP's volume units differ between the daily and monthly
    files, and pre-2004 NASDAQ volume double-counts dealer trades (a trade
    through a market maker is recorded on both legs), so NASDAQ turnover is
    roughly twice NYSE turnover for mechanical reasons.  `universe.py` takes
    dollar volume ranks WITHIN exchange rather than pooled, which is immune to
    both problems; anything that needs the level should verify the units
    against CRSP's current documentation first.
    """

    KEY: ClassVar[tuple[str, ...]] = ("date", "permno")
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "date", "permno", "permco", "ncusip", "shrcd", "exchcd", "siccd",
        "prc", "shrout", "mktcap", "ret", "retx", "vol", "numtrd", "bid", "ask",
        "openprc", "cfacpr", "cfacshr", "quote_only",
    )
    SYNONYMS: ClassVar[dict[str, str]] = {"cusip": "ncusip"}

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        for col in ("permno", "permco"):
            df[col] = df[col].astype("int64")
        df["ncusip"] = df["ncusip"].astype(str).str.strip()
        for col in ("shrcd", "exchcd", "siccd"):
            # siccd is occasionally NULL on old name rows; 0 is CRSP's own
            # "unknown" convention and industry_map maps it to the residual
            # industry, so the column can stay integer.
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        for col in ("prc", "shrout", "mktcap", "ret", "retx", "vol", "numtrd",
                    "bid", "ask", "openprc", "cfacpr", "cfacshr"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # CRSP writes prc = 0 when there is no price at all for the day; abs()
        # in SQL leaves it as 0, which would be a $0 stock rather than a
        # missing one.  Both the price and the market cap it implies go NaN.
        no_price = df["prc"] <= 0
        df.loc[no_price, ["prc", "mktcap"]] = float("nan")
        # CRSP writes vol = -99 for "missing" and 0 for a genuine no-trade day.
        df.loc[df["vol"] < 0, "vol"] = float("nan")
        df["quote_only"] = df["quote_only"].fillna(False).astype(bool)
        return df.reset_index(drop=True)


# One calendar year of daily common-stock rows.
#
#   dsenames — the name row VALID AT d.date.  Ranges are non-overlapping per
#              permno, so the join stays 1:1 and `siccd` is the industry CRSP
#              recorded at the time, not today's.  Filtering on the name row
#              (rather than on dsf's own `hsiccd`, which is the *header*, i.e.
#              most recent, value) is what keeps the industry assignment
#              point-in-time.
#   prc      — negative means "no trade; this is minus the bid-ask midpoint".
#              abs() recovers the level and `bid`/`ask` flag the case.
#   shrout   — CRSP stores thousands of shares; x1000 puts it in shares so
#              mktcap comes out in dollars.
CRSP_DAILY_SQL = """
select d.date,
       d.permno,
       d.permco,
       n.ncusip,
       n.shrcd,
       n.exchcd,
       n.siccd,
       abs(d.prc)                     as prc,
       d.shrout * 1000.0              as shrout,
       abs(d.prc) * d.shrout * 1000.0 as mktcap,
       d.ret,
       d.retx,
       d.vol,
       d.numtrd,
       d.bid,
       d.ask,
       abs(d.openprc)                 as openprc,
       (d.prc < 0)                    as quote_only,  -- no closing trade
       d.cfacpr,
       d.cfacshr
from crsp.dsf as d
join crsp.dsenames as n
  on n.permno = d.permno
 and n.namedt <= d.date
 and n.nameendt >= d.date
where d.date between %(start)s and %(end)s
  and n.shrcd in (10, 11)            -- ordinary common stock only
  and n.exchcd in (1, 2, 3)          -- NYSE, AMEX, NASDAQ
order by d.date, d.permno
"""


def fetch_crsp_daily(session: WRDSSession, start: str, end: str) -> CRSPDaily:
    """Pull daily common-stock rows for [start, end] in one round trip.

    Call this for at most a year or two at a time; `load_or_fetch_crsp_daily`
    is the function to use from a notebook, since it chunks and caches.
    """
    df = session.query(CRSP_DAILY_SQL, {"start": start, "end": end})
    return CRSPDaily.from_raw(df)


# Bump when CRSP_DAILY_SQL's column set changes.  The cache file name carries
# it, so a schema change re-pulls into NEW files instead of failing against
# old ones — and the previous version stays on disk until the new pull is
# known good, which matters when a full pull is 45 minutes.
DAILY_SCHEMA_VERSION = 2


def daily_cache_path(year: int, cache_dir: Path = CACHE_DIR) -> Path:
    """`cache.nosync/crsp_daily_v<V>_<year>.parquet`.

    One file per calendar year, not per request range.  A range-named file
    (the convention the 230ZA project uses for its monthly pulls) would make
    2015-2017 and 2016-2018 two overlapping downloads of the same rows; a
    year-named file means the second request re-reads three files it already
    has and pulls only 2018.
    """
    return cache_dir / f"crsp_daily_v{DAILY_SCHEMA_VERSION}_{year}.parquet"


def load_or_fetch_crsp_daily(
    session: WRDSSession | None,
    start: str,
    end: str,
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
) -> CRSPDaily:
    """Daily CRSP rows for [start, end], reading the year cache where possible.

    For each calendar year the range touches: read `cache.nosync/crsp_daily_<y>.parquet`
    if it exists, else pull that whole year from WRDS and write it.  Whole
    years are cached even when the request is a partial year, so the next
    request that needs the rest of the year costs nothing.

    Args:
        session: an open WRDS session, or None to run cache-only (raises if a
            year is missing — useful in tests and on a plane).
        start, end: ISO dates, inclusive.
        cache_dir: override for tests.
        verbose: print one line per year, since a cold full-sample pull takes
            minutes and silence looks like a hang.

    Returns:
        CRSPDaily covering exactly [start, end], sorted by (date, permno).
    """
    frames: list[pd.DataFrame] = []
    for year in years_in_range(start, end):
        path = daily_cache_path(year, cache_dir)
        if path.exists():
            if verbose:
                print(f"crsp daily {year}: cache hit  ({path.name})")
            frames.append(pd.read_parquet(path))
            continue
        if session is None:
            raise FileNotFoundError(
                f"no cached CRSP daily file at {path} and no WRDS session given"
            )
        if verbose:
            print(f"crsp daily {year}: fetching from WRDS ...", flush=True)
        year_data = fetch_crsp_daily(session, f"{year}-01-01", f"{year}-12-31")
        path.parent.mkdir(parents=True, exist_ok=True)
        year_data.frame.to_parquet(path, index=False)
        if verbose:
            print(f"crsp daily {year}: {len(year_data):,} rows cached")
        frames.append(year_data.frame)

    combined = pd.concat(frames, ignore_index=True)
    return CRSPDaily.from_raw(_trim(combined, start, end))


# ---------------------------------------------------------------------------
# CRSP delisting events
# ---------------------------------------------------------------------------

class CRSPDelist(Dataset):
    """Delisting events, one row per (permno, dlstdt).

    Canonical columns:
        permno   int
        dlstdt   delisting date (datetime64)
        dlstcd   delisting code: 100s still trading, 200s merger,
                 300s exchange, 400s liquidation, 500s dropped
        dlret    delisting return, NaN when CRSP could not compute one
        dlretx   the same excluding dividends

    Kept as a separate event table rather than joined into `CRSPDaily` at
    fetch time, because the join has judgement in it (what to do when `dlret`
    is missing, what to do when the delisting day has no daily row at all) and
    judgement belongs somewhere testable.  See `daily_returns.py`.
    """

    KEY: ClassVar[tuple[str, ...]] = ("permno", "dlstdt")
    REQUIRED: ClassVar[tuple[str, ...]] = ("permno", "dlstdt", "dlstcd", "dlret", "dlretx")
    SYNONYMS: ClassVar[dict[str, str]] = {}

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["dlstdt"] = pd.to_datetime(df["dlstdt"])
        df["permno"] = df["permno"].astype("int64")
        df["dlstcd"] = pd.to_numeric(df["dlstcd"], errors="coerce").fillna(0).astype("int64")
        for col in ("dlret", "dlretx"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.reset_index(drop=True)


CRSP_DELIST_SQL = """
select permno, dlstdt, dlstcd, dlret, dlretx
from crsp.dsedelist
where dlstdt between %(start)s and %(end)s
order by permno, dlstdt
"""


def fetch_crsp_delist(session: WRDSSession, start: str, end: str) -> CRSPDelist:
    """Pull delisting events in [start, end].  Small — tens of thousands of
    rows over three decades — so one query, one cache file."""
    df = session.query(CRSP_DELIST_SQL, {"start": start, "end": end})
    return CRSPDelist.from_raw(df)


def delist_cache_path(start: str, end: str, cache_dir: Path = CACHE_DIR) -> Path:
    """`cache.nosync/crsp_delist_<start>_<end>.parquet` — range-named, because the
    table is small enough that re-pulling a different range is cheap and a
    year-split would add files for nothing."""
    return cache_dir / f"crsp_delist_{start}_{end}.parquet"


def load_or_fetch_crsp_delist(
    session: WRDSSession | None, start: str, end: str, cache_dir: Path = CACHE_DIR
) -> CRSPDelist:
    """Return the cached delisting events if present, else fetch and cache."""
    path = delist_cache_path(start, end, cache_dir)
    if path.exists():
        return CRSPDelist.from_raw(pd.read_parquet(path))
    if session is None:
        raise FileNotFoundError(f"no cached delist file at {path} and no WRDS session given")
    events = fetch_crsp_delist(session, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    events.frame.to_parquet(path, index=False)
    return events


# ---------------------------------------------------------------------------
# Fama-French factors, daily
# ---------------------------------------------------------------------------

class FactorsDaily(Dataset):
    """Daily Fama-French 5 factors + momentum + the risk-free rate.

    Canonical columns (all decimal returns, not percent):
        date    trading day (datetime64)
        mktrf   market excess return, Rm - Rf
        smb     size
        hml     value
        rmw     profitability
        cma     investment
        umd     momentum
        rf      one-day risk-free rate

    Used by part 1 for the baseline lead-lag regression's market control, and
    by part 4 for the factor-exposure attribution.  The assignment's suggested
    data list names the Ken French library directly; `ff.fivefactors_daily` on
    WRDS is the same series, served as a table instead of a zipped CSV, so it
    is the one this module pulls.  `french_library.py` downloads from Dartmouth
    for the pieces WRDS does not serve.
    """

    KEY: ClassVar[tuple[str, ...]] = ("date",)
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "date", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf",
    )
    SYNONYMS: ClassVar[dict[str, str]] = {}

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        for col in ("mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)


# `ff.fivefactors_daily` carries the five factors plus umd and rf on WRDS.  If
# a run hits a missing-column error here, the fallback is `ff.factors_daily`
# (three factors + umd + rf), which has no rmw/cma — the baseline regression
# only needs mktrf, so part 1 survives that; part 4's attribution would not.
FACTORS_DAILY_SQL = """
select date, mktrf, smb, hml, rmw, cma, umd, rf
from ff.fivefactors_daily
where date between %(start)s and %(end)s
order by date
"""


def fetch_factors_daily(session: WRDSSession, start: str, end: str) -> FactorsDaily:
    """Pull daily factor returns for [start, end].  A few thousand rows."""
    df = session.query(FACTORS_DAILY_SQL, {"start": start, "end": end})
    return FactorsDaily.from_raw(df)


def factors_cache_path(start: str, end: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"ff_factors_daily_{start}_{end}.parquet"


def load_or_fetch_factors_daily(
    session: WRDSSession | None, start: str, end: str, cache_dir: Path = CACHE_DIR
) -> FactorsDaily:
    """Return the cached daily factors if present, else fetch and cache."""
    path = factors_cache_path(start, end, cache_dir)
    if path.exists():
        return FactorsDaily.from_raw(pd.read_parquet(path))
    if session is None:
        raise FileNotFoundError(f"no cached factor file at {path} and no WRDS session given")
    factors = fetch_factors_daily(session, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    factors.frame.to_parquet(path, index=False)
    return factors


# ---------------------------------------------------------------------------
# CRSP name history (permno <-> ticker, point in time)
# ---------------------------------------------------------------------------

class CRSPNames(Dataset):
    """`crsp.dsenames` as its own table: one row per (permno, name period).

    Canonical columns:
        permno    int
        namedt    first date this row is valid (datetime64)
        nameendt  last date this row is valid (datetime64)
        ticker    trading symbol during that period
        comnam    company name during that period
        ncusip    8-char CUSIP during that period
        shrcd, exchcd, siccd   as of that period

    Why a separate table rather than more columns on `CRSPDaily`.  A ticker is
    a property of a DATE RANGE, not of a day: carrying it on 34 million daily
    rows would repeat the same string ~250 times per stock-year for no gain.
    Kept normalised, it is ~100k rows and joins on demand.

    Why it exists at all: **tickers are how every non-CRSP data source is
    keyed.**  Databento, Compustat's newer tables, ETF constituent files and
    anything scraped all speak ticker; CRSP speaks permno.  This table is the
    bridge, and it is point-in-time, which matters more than it sounds —
    tickers are RECYCLED.  Matching on today's ticker would map a 2005 row to
    whichever company holds that symbol now, silently and wrongly.  Always
    join with `namedt <= date <= nameendt`.
    """

    KEY: ClassVar[tuple[str, ...]] = ("permno", "namedt")
    REQUIRED: ClassVar[tuple[str, ...]] = (
        "permno", "namedt", "nameendt", "ticker", "comnam", "ncusip",
        "shrcd", "exchcd", "siccd",
    )
    SYNONYMS: ClassVar[dict[str, str]] = {}

    @classmethod
    def _coerce(cls, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("namedt", "nameendt"):
            df[col] = pd.to_datetime(df[col])
        df["permno"] = df["permno"].astype("int64")
        for col in ("ticker", "comnam", "ncusip"):
            df[col] = df[col].astype(str).str.strip()
        for col in ("shrcd", "exchcd", "siccd"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        return df.sort_values(["permno", "namedt"]).reset_index(drop=True)


CRSP_NAMES_SQL = """
select permno, namedt, nameendt, ticker, comnam, ncusip, shrcd, exchcd, siccd
from crsp.dsenames
where nameendt >= %(start)s and namedt <= %(end)s
order by permno, namedt
"""


def fetch_crsp_names(session: WRDSSession, start: str, end: str) -> CRSPNames:
    """Pull the name history overlapping [start, end].  Small: one query."""
    df = session.query(CRSP_NAMES_SQL, {"start": start, "end": end})
    return CRSPNames.from_raw(df)


def names_cache_path(start: str, end: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"crsp_names_{start}_{end}.parquet"


def load_or_fetch_crsp_names(
    session: WRDSSession | None, start: str, end: str, cache_dir: Path = CACHE_DIR
) -> CRSPNames:
    """Return the cached name history if present, else fetch and cache."""
    path = names_cache_path(start, end, cache_dir)
    if path.exists():
        return CRSPNames.from_raw(pd.read_parquet(path))
    if session is None:
        raise FileNotFoundError(f"no cached names file at {path} and no WRDS session given")
    names = fetch_crsp_names(session, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    names.frame.to_parquet(path, index=False)
    return names


def ticker_at(names: CRSPNames, permno: int, on: str | pd.Timestamp) -> str | None:
    """The ticker a permno traded under on a given date, or None.

    The point-in-time lookup the module docstring warns about, as one call.
    """
    when = pd.Timestamp(on)
    df = names.frame
    hit = df.loc[
        (df["permno"] == permno) & (df["namedt"] <= when) & (df["nameendt"] >= when),
        "ticker",
    ]
    return None if hit.empty else str(hit.iloc[0])
