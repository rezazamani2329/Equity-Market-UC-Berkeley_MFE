"""
SIC code -> Fama-French 49 industry, point in time, plus value-weighted
industry returns.

Why 49 and not GICS or SIC itself
---------------------------------
The project needs a partition of stocks into groups that share a common
shock — an "industry" in the economic sense, not a filing category.  Raw
4-digit SIC has ~1,000 codes, far too fine: a leader and its followers would
often be alone in their code.  The 2-digit roll-up is too coarse in the other
direction and mixes unrelated businesses.  Fama and French's 49-industry
scheme is the standard compromise in this literature, it is defined as an
explicit list of SIC ranges (so it is reproducible, unlike a vendor
classification behind a licence), and it is exactly what the assignment's
suggested data list points at — the Ken French Data Library.

Point-in-time, and why it matters here
--------------------------------------
The SIC code used is `siccd` from the CRSP name row valid on the observation
date (see `wrds_fetch.CRSP_DAILY_SQL`), not the header code `hsiccd`, which is
the stock's MOST RECENT industry applied backwards over its whole history.
Using the header code would, for instance, place a firm that reinvented itself
as a software company in "Software" for its years as a manufacturer — and it
would do so using information from the future.  In a lead-lag design that
lookahead goes straight into the leader/follower assignment, which is the
mechanism the project is testing.  So: name-row code, always.

The residual industry
---------------------
Fama and French's definition leaves some SIC codes unassigned on purpose;
those, and CRSP's `siccd = 0` (unknown), fall into industry 49, "Other".  This
module keeps them rather than dropping them, and `universe.py` is where the
decision to exclude "Other" from the leader/follower construction is made and
documented — an industry that is a grab bag has no common shock, so it has no
business in a lead-lag test.

Source file
-----------
`Siccodes49.zip` from the Ken French Data Library.  It is downloaded once,
unzipped into `cache.nosync/Siccodes49.txt`, and parsed.  The file is small and
stable, and committing it would be redistributing someone else's data, so the
cache is git-ignored and the download is reproducible.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from lead_lag.data.information_set import require_lagged
from lead_lag.data.wrds_connection import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "cache.nosync"

FRENCH_SICCODES49_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes49.zip"
)

# The industry number Fama and French reserve for everything their ranges do
# not cover.  CRSP's "unknown" (siccd 0) lands here too.
OTHER_INDUSTRY: int = 49


def download_siccodes49(
    cache_dir: Path = CACHE_DIR, url: str = FRENCH_SICCODES49_URL
) -> Path:
    """Download and unzip Ken French's 49-industry definition file.

    Returns the path to the extracted `.txt`.  If the file is already in the
    cache the download is skipped, so this is safe to call from a notebook
    cell that gets re-run.

    Network access happens here and nowhere else in part 1 except WRDS, which
    is why it is its own function: a test can point `cache_dir` at a fixture
    directory containing the file and never touch the network.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "Siccodes49.txt"
    if target.exists():
        return target

    import requests  # lazy import: the module stays importable offline

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        # The archive holds a single .txt whose exact name has changed across
        # versions of the library, so pick the first .txt rather than hardcode.
        name = next(n for n in archive.namelist() if n.lower().endswith(".txt"))
        target.write_bytes(archive.read(name))
    return target


def parse_siccodes(path: Path) -> pd.DataFrame:
    """Parse a Ken French `SiccodesNN.txt` into a range table.

    The file's shape:

        1 Agric  Agriculture
              0100-0199 Agric production - crops
              0200-0299 Agric production - livestock
        2 Food   Food Products
              2000-2009 Food and kindred products
        ...

    A line whose first token is a bare integer opens an industry and gives its
    number and short name; a line whose first token is `dddd-dddd` is a SIC
    range belonging to the industry most recently opened.

    The rule is about the TOKEN, not the indentation.  Ken French's files
    indent the industry header by a space or two and the ranges by ten, but
    the exact amounts have varied between versions of the library and between
    the 5-, 10-, 12-, 17-, 30-, 38-, 48- and 49-industry files.  Keying on
    "first token is an integer" vs "first token is a range" parses all of them
    and does not break when the whitespace changes.

    Returns a frame with one row per RANGE:
        ff49       int    industry number, 1-49
        ff49_name  str    short name, e.g. "Agric"
        sic_lo     int    first SIC code in the range, inclusive
        sic_hi     int    last SIC code in the range, inclusive

    Ranges are inclusive at both ends and, within the file, non-overlapping —
    `assign_ff49` relies on that to do a single interval lookup.
    """
    rows: list[dict[str, object]] = []
    current_num: int | None = None
    current_name: str = ""

    for raw_line in path.read_text(errors="replace").splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        token = parts[0]

        # Industry header: a bare integer, e.g. "1" in " 1 Agric  Agriculture".
        if token.isdigit():
            current_num = int(token)
            current_name = parts[1] if len(parts) > 1 else str(current_num)
            continue

        # Range line: "0100-0199".  Anything else (a stray comment, a blank
        # continuation of a long description) is skipped.
        lo_str, sep, hi_str = token.partition("-")
        if not sep or current_num is None:
            continue
        if not (lo_str.isdigit() and hi_str.isdigit()):
            continue
        rows.append(
            {
                "ff49": current_num,
                "ff49_name": current_name,
                "sic_lo": int(lo_str),
                "sic_hi": int(hi_str),
            }
        )

    table = pd.DataFrame(rows, columns=["ff49", "ff49_name", "sic_lo", "sic_hi"])
    if table.empty:
        raise ValueError(f"parse_siccodes: no SIC ranges found in {path}")
    return table


def load_siccodes49(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Download (if needed) and parse the 49-industry range table."""
    return parse_siccodes(download_siccodes49(cache_dir))


def assign_ff49(siccd: pd.Series, table: pd.DataFrame) -> pd.Series:
    """Map a Series of 4-digit SIC codes to Fama-French 49 industry numbers.

    Args:
        siccd: integer SIC codes, one per row of the panel.
        table: the range table from `load_siccodes49`.

    Returns:
        An int Series aligned to `siccd`, with `OTHER_INDUSTRY` (49) wherever
        no range matched — including `siccd = 0`, CRSP's "unknown".

    Implementation note.  A row-by-row lookup over 50 million rows would take
    minutes; `np.searchsorted` over the sorted range starts turns it into one
    vectorised pass.  The ranges are non-overlapping, so the candidate for a
    code is the last range whose `sic_lo` is <= it; the row is assigned only
    if that range's `sic_hi` also covers it, which is what rejects codes
    sitting in the gaps Fama and French left unassigned.
    """
    ordered = table.sort_values("sic_lo").reset_index(drop=True)
    lo = ordered["sic_lo"].to_numpy()
    hi = ordered["sic_hi"].to_numpy()
    ind = ordered["ff49"].to_numpy()

    codes = pd.to_numeric(siccd, errors="coerce").fillna(0).astype("int64").to_numpy()
    # "right" so that a code equal to a range's lower bound picks that range
    # (index of the first lo strictly greater than the code, minus one).
    pos = np.searchsorted(lo, codes, side="right") - 1
    valid = (pos >= 0) & (codes <= hi[np.clip(pos, 0, len(hi) - 1)])
    out = np.where(valid, ind[np.clip(pos, 0, len(ind) - 1)], OTHER_INDUSTRY)
    return pd.Series(out.astype("int64"), index=siccd.index, name="ff49")


def attach_industry(panel: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Return `panel` with `ff49` and `ff49_name` columns added.

    Works on any frame carrying a `siccd` column — the daily panel, or a
    stock-level summary.  The name is attached for readability of tables and
    figures; every join and group-by in the project uses the NUMBER, because
    names have changed spelling between versions of the file.
    """
    out = panel.copy()
    out["ff49"] = assign_ff49(out["siccd"], table)
    names = (
        table.drop_duplicates("ff49").set_index("ff49")["ff49_name"].to_dict()
    )
    names.setdefault(OTHER_INDUSTRY, "Other")
    out["ff49_name"] = out["ff49"].map(names).fillna("Other")
    return out


def industry_returns(
    panel: pd.DataFrame,
    weight: str = "mktcap_lag",
    ret_col: str = "ret",
    industry_col: str = "ff49_lag",
    lag_weights: bool = False,
) -> pd.DataFrame:
    """Value-weighted daily return per industry, on the d-1 information set.

    Args:
        panel: a daily frame with `date`, `permno`, the return column, and the
            lagged weight and industry columns — i.e. the output of
            `information_set.attach_lagged`.
        weight: column to weight by.  Defaults to `mktcap_lag`: the market cap
            at the previous close.  An equal-weighted industry is a different
            object and should be requested explicitly by passing a column of
            ones.
        ret_col: the return column — day d's, since that is what the index
            measures.
        industry_col: the grouping key.  Defaults to `ff49_lag`, the industry
            the stock belonged to at the previous close.
        lag_weights: apply a further `shift(1)` to `weight` inside this
            function.  Defaults to False because the default `weight` is
            already lagged; set it True only when passing a same-day column
            deliberately.

    Returns:
        Long frame with `date`, `<industry_col>`, `ind_ret`, `n_stocks`.

    Why both defaults are lagged
    ----------------------------
    The industry index for day d is the portfolio you could have *formed at
    the close of d-1* and held through d.  That means both the membership and
    the weights are d-1 objects:

    * **Weights.** Market cap is `prc x shrout`, and the price moves with the
      day's return.  Weighting day d by day d's market cap overweights the
      day's winners mechanically and biases the index upward.
    * **Membership.** A firm that CRSP reclassifies with effect from day d is
      in its new industry from day d.  Grouping day d's return by that new
      industry uses a fact stamped that morning — the firm's peers for day d
      are the peers it had at the previous close.

    The second one is the easy one to miss, because an industry code feels
    static. It is not: it is a dated field on a CRSP name row.

    Which vintage this project actually uses
    ----------------------------------------
    Two backward-looking vintages are available and both satisfy the d-1 rule:

    * **previous trading day** — `ff49_lag`, this function's default, correct
      for any caller holding only a daily panel;
    * **formation month** — the industry fixed at the end of month m-1 by
      `universe.build_universe`, and carried on `Universe` / `LeaderMap` as
      plain `ff49`.

    **The project uses the formation-month vintage**, and the pipeline calls
    this function as

        industry_returns(eligible_daily(panel, universe), industry_col="ff49")

    The reason is internal consistency rather than freshness.  Leader and
    follower membership is fixed monthly; if the index used a daily vintage, a
    firm reclassified mid-month would be ranked against industry A's leader
    while being averaged into industry B's index, inside a single holding
    period.  A tradeable portfolio cannot do that.  Passing `ff49_lag`
    instead is the robustness variant — on the 1996-2024 sample the two
    disagree on 4,290 stock-days out of 34.4 million, so it should change
    nothing, and if it does, that is worth knowing.

    Running it on `eligible_daily(...)` rather than the raw panel matters for
    the same reason: the index part 2 decomposes against should contain the
    names the strategy could actually hold, not the ones the liquidity and
    price screens removed.

    This is where part 1 hands off to part 2: the common-industry component of
    a leader's move is measured against this series.
    """
    industry_col = require_lagged(panel, industry_col)
    weight = require_lagged(panel, weight)

    df = panel[["date", "permno", industry_col, ret_col, weight]].copy()

    if lag_weights:
        # One trading-day lag WITHIN each stock.  `groupby(...).shift(1)`
        # respects gaps in a stock's history (a suspended stock's next
        # observed day gets the weight from its last observed day, not from a
        # calendar day it did not trade) — which is the intended behaviour
        # here: the weight is "what the position was worth at the last close".
        df["w"] = df.groupby("permno")[weight].shift(1)
    else:
        df["w"] = df[weight]

    df = df.dropna(subset=[ret_col, "w"])
    df = df[df["w"] > 0]

    # Computed as sum(w * r) / sum(w) rather than with `np.average` inside a
    # `groupby.apply`.  They give the same number, but the full sample has
    # ~350,000 (date, industry) groups and `apply` runs Python once per group:
    # that turns a two-second aggregation into a twenty-minute one, and it is
    # the slowest thing in part 1 by an order of magnitude if left alone.
    df = df.assign(_wr=df[ret_col] * df["w"])
    grouped = df.groupby(["date", industry_col], sort=True)
    out = grouped.agg(
        _wr_sum=("_wr", "sum"),
        _w_sum=("w", "sum"),
        n_stocks=("permno", "nunique"),
    ).reset_index()
    out["ind_ret"] = out["_wr_sum"] / out["_w_sum"]
    out["n_stocks"] = out["n_stocks"].astype("int64")
    return out[["date", industry_col, "ind_ret", "n_stocks"]]
