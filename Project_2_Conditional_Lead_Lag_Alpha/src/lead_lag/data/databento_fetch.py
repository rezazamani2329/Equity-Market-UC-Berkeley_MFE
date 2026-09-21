"""
Intraday quotes from Databento, and the synchronised snapshots the intraday
lead-lag test runs on.

Why there is an intraday leg at all
-----------------------------------
The daily panel says the leader-follower effect is strong through 2006 and
absent from 2007 (`results/p1_coefficient_by_year.csv`).  Two readings:

  (a) it was always microstructure — median relative spread was 3.23% in 1996
      and 0.11% in 2015, so the "effect" decayed as the spread collapsed; or
  (b) it did not die, it got FASTER — information that took a day to reach
      small peers in 1998 takes minutes now, and a daily panel cannot resolve
      a twenty-minute diffusion because it happens inside one observation.

Databento's US equity history starts 2018-05, which is exactly the window
where the daily effect is dead.  That makes it the instrument for
discriminating (a) from (b), which is the interesting question this project
can actually answer.

It also dissolves the non-synchronous-trading problem rather than controlling
for it.  With daily closes, a small stock's "close" may be a quote struck at
14:30 against the leader's 16:00 print, and the resulting stale-price effect
is indistinguishable from diffusion.  With intraday data, both legs are
sampled at the same wall-clock instant by construction.

Schema choice
-------------
`bbo-1m` — the best bid and offer sampled once a minute.  Three reasons over
the alternatives:

* against `ohlcv-1m`: bars are built from TRADES, so a bar close is a print at
  the bid or the ask and bounces exactly like a daily close.  The whole point
  of coming here is to measure on quotes.
* against `trades` / `tbbo`: those are event-time, so two stocks' observations
  do not line up, which re-creates the synchronisation problem at higher
  frequency.  `bbo-1m` is clock-sampled: every symbol has a row at the same
  minute.
* against `mbp-1`: full quote updates are ~4x the volume for information this
  design does not use.

`tbbo` still matters for part 5's transaction costs — effective spread needs
trades matched to the quote prevailing at the trade — but that is a separate,
smaller pull over a subset.

Venue caveat
------------
`XNAS.ITCH` and `XNYS.PILLAR` are per-venue books, not the consolidated NBBO.
A stock's primary-venue BBO is a close approximation to the NBBO for that
stock, but other venues can be inside it intraday.  The long history (2018 vs
2023) is worth the approximation; the validation overlay is a re-run of the
2023+ period against the consolidated `DBEQ.BASIC`, which part 5 should do.

Cost and volume
---------------
On this account `metadata.get_cost` returns zero for every configuration, so
the constraint is bytes and time, not money.  Measured for the project's
1,209-symbol list over 2018-05 to 2024-12:

    XNAS.ITCH    bbo-1m   18.6 GB
    XNYS.PILLAR  bbo-1m   35.9 GB
    XASE.PILLAR  bbo-1m    0.1 GB

`estimate_size` re-checks before any pull, because that number moves with the
symbol list and nobody should start a 55 GB download on a stale estimate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lead_lag.data.wrds_connection import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "cache.nosync" / "databento"

# Databento's US equity history begins here for the per-venue datasets.
EQUITY_HISTORY_START = "2018-05-01"

# Regular US equity session, Eastern time.  Quotes exist outside it, but the
# opening and closing auctions behave differently enough that mixing them into
# a minute-bar panel would put two different price-formation processes in one
# regression.  The first and last minutes are trimmed for the same reason.
SESSION_OPEN = "09:30"
SESSION_CLOSE = "16:00"
TRIM_OPEN_MINUTES = 5
TRIM_CLOSE_MINUTES = 5


def _api_key(key: str | None = None) -> str:
    """The Databento key, from the argument or `$DATABENTO_API_KEY`.

    Never defaulted into a file and never logged.  `.env` is git-ignored and
    `wrds_connection` has already loaded it into the environment by the time
    anything here runs.
    """
    resolved = key or os.environ.get("DATABENTO_API_KEY")
    if not resolved:
        raise RuntimeError(
            "databento_fetch: no key given and $DATABENTO_API_KEY is not set. "
            "Add it to .env (git-ignored) or export it."
        )
    return resolved


def _client(key: str | None = None):
    """A Databento historical client.  Imported lazily so this module stays
    importable — and the rest of part 1 stays testable — without the package."""
    import databento as db  # noqa: PLC0415

    return db.Historical(_api_key(key))


@dataclass(frozen=True)
class PullSpec:
    """One Databento request: a venue, a schema, a symbol list, a window.

    Kept as an object rather than loose arguments so a pull can be estimated,
    logged and executed from the same description — the estimate and the
    download must not be able to drift apart.
    """

    dataset: str
    schema: str
    symbols: tuple[str, ...]
    start: str
    end: str

    def cache_path(self, cache_dir: Path = CACHE_DIR) -> Path:
        """`cache.nosync/databento/<dataset>_<schema>_<start>_<end>_<n>syms.dbn.zst`.

        The symbol count is in the name because two pulls of the same venue,
        schema and window but different symbol lists are different data, and
        silently reading one for the other is the kind of error that produces
        a plausible-looking panel with stocks missing from it.
        """
        return cache_dir / (
            f"{self.dataset}_{self.schema}_{self.start}_{self.end}"
            f"_{len(self.symbols)}syms.dbn.zst"
        )

    def describe(self) -> str:
        return (
            f"{self.dataset} {self.schema} {len(self.symbols)} symbols "
            f"{self.start}..{self.end}"
        )


def estimate_size(spec: PullSpec, key: str | None = None) -> dict[str, float]:
    """Billable bytes and cost for `spec`, without downloading anything.

    Always call this before a large pull.  Returns
    `{"bytes": float, "gb": float, "cost_usd": float}`.
    """
    client = _client(key)
    size = client.metadata.get_billable_size(
        dataset=spec.dataset,
        schema=spec.schema,
        symbols=list(spec.symbols),
        stype_in="raw_symbol",
        start=spec.start,
        end=spec.end,
    )
    cost = client.metadata.get_cost(
        dataset=spec.dataset,
        schema=spec.schema,
        symbols=list(spec.symbols),
        stype_in="raw_symbol",
        start=spec.start,
        end=spec.end,
    )
    return {"bytes": float(size), "gb": float(size) / 1e9, "cost_usd": float(cost)}


def fetch(spec: PullSpec, cache_dir: Path = CACHE_DIR, key: str | None = None) -> Path:
    """Download `spec` to the cache, or return the cached file if present.

    Streams straight to disk rather than into memory: even a pilot month of
    one-minute quotes is hundreds of megabytes, and the full pull is tens of
    gigabytes.  The file is DBN (Databento's binary format), zstd-compressed;
    `load` turns it into a DataFrame.
    """
    path = spec.cache_path(cache_dir)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    client = _client(key)
    # A partial file from an interrupted download must never be mistaken for a
    # complete one, so write to a temporary name and rename on success —
    # rename is atomic within a filesystem.
    tmp = path.with_suffix(path.suffix + ".partial")
    client.timeseries.get_range(
        dataset=spec.dataset,
        schema=spec.schema,
        symbols=list(spec.symbols),
        stype_in="raw_symbol",
        start=spec.start,
        end=spec.end,
        path=str(tmp),
    )
    tmp.rename(path)
    return path


def load(path: Path, tz: str = "America/New_York") -> pd.DataFrame:
    """Read a cached DBN file into a DataFrame with Eastern timestamps.

    Databento timestamps are UTC nanoseconds.  They are converted to Eastern
    here and nowhere else, because a US equity session is defined in Eastern
    and doing the conversion at the point of use invites one place to forget.
    """
    import databento as db  # noqa: PLC0415

    store = db.DBNStore.from_file(str(path))
    df = store.to_df()
    if df.index.name in ("ts_recv", "ts_event"):
        df = df.reset_index()
    for col in ("ts_recv", "ts_event"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(tz)
    return df


# ---------------------------------------------------------------------------
# From raw quotes to a synchronised minute panel
# ---------------------------------------------------------------------------

def minute_panel(
    df: pd.DataFrame,
    trim_open: int = TRIM_OPEN_MINUTES,
    trim_close: int = TRIM_CLOSE_MINUTES,
) -> pd.DataFrame:
    """Turn raw `bbo-1m` records into one clean row per (minute, symbol).

    Returns columns: `ts` (Eastern), `symbol`, `bid`, `ask`, `mid`,
    `rel_spread`, `date`, `minute`.

    Three things happen here, each of which would otherwise corrupt the
    intraday regression:

    * **Session trimming.**  The opening and closing auctions are a different
      price-formation process from continuous trading, and the first minutes
      after the open carry enormous spreads.  Rows outside
      09:30+trim .. 16:00-trim are dropped.
    * **Crossed and locked quotes removed.**  `ask <= bid` happens, and a
      midpoint computed from it is meaningless rather than noisy.
    * **Deduplication.**  A venue can emit more than one record for the same
      minute and symbol; the last one is the state at the minute boundary.

    No forward-filling.  A symbol with no quote in a minute is ABSENT, not
    carried forward — carrying a stale quote forward is precisely the
    non-synchronous-trading artefact the intraday leg exists to avoid, and it
    would silently manufacture lead-lag in illiquid followers.
    """
    out = df.copy()
    ts_col = "ts_recv" if "ts_recv" in out.columns else "ts_event"
    out = out.rename(columns={ts_col: "ts"})

    # Databento names the top-of-book levels with a 00 suffix.
    rename = {}
    if "bid_px_00" in out.columns:
        rename["bid_px_00"] = "bid"
    if "ask_px_00" in out.columns:
        rename["ask_px_00"] = "ask"
    out = out.rename(columns=rename)

    keep = ["ts", "symbol", "bid", "ask"]
    missing = [c for c in keep if c not in out.columns]
    if missing:
        raise KeyError(
            f"minute_panel: {missing} not in the frame. Columns present: "
            f"{sorted(out.columns)[:25]}"
        )
    out = out[keep]

    valid = out["bid"].notna() & out["ask"].notna() & (out["ask"] > out["bid"]) & (out["bid"] > 0)
    out = out.loc[valid]

    out["date"] = out["ts"].dt.date
    out["minute"] = out["ts"].dt.floor("min")

    tod = out["ts"].dt.time
    lo = (pd.Timestamp(SESSION_OPEN) + pd.Timedelta(minutes=trim_open)).time()
    hi = (pd.Timestamp(SESSION_CLOSE) - pd.Timedelta(minutes=trim_close)).time()
    out = out.loc[(tod >= lo) & (tod <= hi)]

    # Last record wins within a (minute, symbol).
    out = out.sort_values("ts").drop_duplicates(["minute", "symbol"], keep="last")

    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["rel_spread"] = (out["ask"] - out["bid"]) / out["mid"]
    return out.reset_index(drop=True)


def minute_returns(panel: pd.DataFrame, freq_minutes: int = 1) -> pd.DataFrame:
    """Midpoint returns at a chosen minute frequency, per symbol.

    Args:
        panel: `minute_panel` output.
        freq_minutes: sampling interval.  1 uses every minute; 5 samples every
            fifth minute, which is the usual choice for a lead-lag design —
            one-minute quote changes in a small stock are dominated by quote
            flicker rather than information.

    Returns `ts`, `symbol`, `mid`, `ret`, `rel_spread`, with `ret` the
    midpoint return over one sampling interval.

    Returns are NOT computed across the overnight gap: the first observation
    of each session is NaN.  An overnight "return" spans sixteen hours and a
    market open, and putting it in the same series as a five-minute return
    would let one observation per day dominate the variance.
    """
    df = panel.copy()
    if freq_minutes > 1:
        # Keep the last quote in each bucket, so the sampling grid is regular
        # and every symbol is observed at the same instants.
        df["bucket"] = df["minute"].dt.floor(f"{freq_minutes}min")
        df = df.sort_values("minute").drop_duplicates(["bucket", "symbol"], keep="last")
        df["ts"] = df["bucket"]
    else:
        df["ts"] = df["minute"]

    df = df.sort_values(["symbol", "ts"])
    grouped = df.groupby(["symbol", "date"], sort=False)
    df["ret"] = grouped["mid"].pct_change()
    return df[["ts", "date", "symbol", "mid", "ret", "rel_spread"]].reset_index(drop=True)
