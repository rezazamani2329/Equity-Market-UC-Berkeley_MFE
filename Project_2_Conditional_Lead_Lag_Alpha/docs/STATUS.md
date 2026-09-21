# Project status — data & universe (Part 1)

**Last updated:** 2026-09-20 · **Owner:** Elias Roubache (part 1)

Read this first if you are picking up parts 2–5. It says what exists, what it
means, what you can start on now, and what is not settled.

---

## TL;DR for parts 2–5

**You can start now.** Parts 2, 3 and 4 are unblocked — the panel, the roles,
the industry index and the factor file all exist and are stable. Part 5 can
start on the robustness *design* but not on final numbers.

**Two findings you need to know before you model anything**, because they
change what a good result looks like:

1. **The daily effect is positive (continuation), not reversal.** Lagged
   leader return predicts follower return with β = +0.0093 (t = 5.03) at one
   day. That is Hou-style information diffusion — momentum, not reversion. Our
   hypothesis needs a *conditioning variable* to flip the sign; it does not get
   the sign for free.
2. **The daily effect is confined to 1996–2006.** From 2007 it oscillates
   around zero. Over the same period median relative spread fell from 3.23% to
   0.11%. Those two facts together are the biggest threat to the whole thesis
   and every result should be reported split at 2007.

---

## Data pipeline — done

| Stage | Module | State |
|---|---|---|
| WRDS connection | `data/wrds_connection.py` | ✅ |
| CRSP daily, delistings, FF5 daily, name history | `data/wrds_fetch.py` | ✅ 34.4M rows, 1996–2024 |
| Delisting merge + survivorship repair | `data/daily_returns.py` | ✅ |
| Data-quality audit | `data/quality_checks.py` | ✅ 0 blocking issues |
| SIC → Fama-French 49, point-in-time | `data/industry_map.py` | ✅ 8.98% in "Other" |
| d−1 information set | `data/information_set.py` | ✅ |
| Monthly universe, 5 screens | `data/universe.py` | ✅ 606,228 eligible stock-months |
| Leader / follower assignment | `data/leaders.py` | ✅ 5.4% monthly leader turnover |
| Baseline lead-lag test | `baseline.py` | ✅ |
| Microstructure defenses | `data/microstructure.py` | ✅ |
| Databento symbol bridge | `data/symbols.py` | ✅ 1,209 tickers |
| Databento intraday fetch | `data/databento_fetch.py` | ✅ piloted |

**51 tests passing**, all offline (`PYTHONPATH=src:tests pytest -q`). They run
against a synthetic panel with a planted lead-lag effect of 0.30, and the
lag-2 placebo is what catches an off-by-one in the shifting.

---

## Results you can consume

In `results/`. Small tables are committed; the two large parquets are not —
see **Getting the data** below.

| File | What it is | In git |
|---|---|---|
| `p1_roles.parquet` | leader/follower per (month, permno, ff49) | ✅ |
| `p1_intraday_symbols.parquet` | 1,209 tickers + venue for Databento | ✅ |
| `p1_horizon_profile.csv` | β(k) for k = 1..10, both estimators | ✅ |
| `p1_coefficient_by_year.csv` | **read this one** — the 1996–2006 story | ✅ |
| `p1_quintile_sort.csv` | non-parametric check | ✅ |
| `p1_universe_attrition.csv` | how many stocks each screen removed | ✅ |
| `p1_delisting_report.csv` | survivorship corrections applied | ✅ |
| `p1_quality_report.csv` | cleaning appendix | ✅ |
| `p1_coverage_by_year.csv` | rows/stocks/days per year | ✅ |
| `p1_intraday_pilot.csv` | pilot result (see below) | ✅ |
| `p1_universe.parquet` | 46 MB — monthly eligibility + formation stats | ❌ regenerate |
| `p1_follower_panel.parquet` | 289 MB — the regression panel | ❌ regenerate |
| `p1_industry_returns.parquet` | value-weighted industry index | ❌ regenerate |

### Headline numbers

**Daily, 1996–2024, 12.4M follower-days, 7,173 sessions**

| Lag | β (Fama-MacBeth) | t | β (pooled, clustered by date) | t |
|---|---|---|---|---|
| 1 | +0.00928 | **+5.03** | +0.02202 | +6.84 |
| 2 | +0.00122 | +0.72 | +0.01199 | +3.84 |
| 3 | −0.00050 | −0.30 | +0.00207 | +0.64 |

Quintile sort at lag 1 is monotone; top-minus-bottom = 3.6 bps/day, t = 4.3.

**Universe attrition** (mean stocks per month): 4,709 candidates → 4,464 after
observation count → 3,381 after the $5 price floor → 1,946 after the NYSE size
floor → 1,885 after liquidity → 1,764 after dropping industry "Other" →
**1,747 eligible**. The size screen is the most expensive (56% fail it
standalone); the price screen is next (24%).

**Intraday pilot** — June 2023, 3 industries (Computers/Banking/Utilities),
22 symbols, 5-minute midpoint returns from Databento `bbo-1m`:
β = −0.0436, **t = −1.67, not significant**. The sign is negative (reversal),
opposite to the daily result, but one month of 22 symbols is not evidence.
The pilot's purpose was to prove the pipeline, and it did: the minute grid is
genuinely synchronised (median 22 symbols per minute, min 20, max 22).

---

## Guarantees the data layer makes

Rely on these; do not re-derive them.

1. **Point-in-time, everywhere.** Universe and leader decisions for month *m*
   use data through the end of *m−1*. Industry codes come from CRSP name rows
   valid on the observation date, never header codes.
2. **The d−1 rule.** Every characteristic that could condition a signal —
   industry, market cap, price, dollar volume, exchange — is available as a
   `_lag` column holding its value at the previous close. `follower_panel`
   *refuses* a frame that has not been through `attach_lagged`, and drops the
   same-day versions rather than renaming them.
3. **One industry vintage.** `p1_roles.parquet` and `p1_industry_returns.parquet`
   both key on `ff49` = the formation-month industry. A firm reclassified
   mid-month cannot be ranked against industry A's leader while being averaged
   into industry B's index.
4. **No survivorship bias.** Delisting returns merged in, including 121 events
   on days the stock did not trade (appended as their own rows). 417 missing
   delisting returns on performance codes got Shumway's substitute, flagged in
   `repaired` so you can drop them.
5. **`(date, permno)` is unique.** Enforced and asserted.

---

## Three ways to break it

Documented at length in `docs/part1_handoff.md`. The short version:

1. **Do not decompose the leader against an industry index that contains the
   leader.** `p1_industry_returns.parquet` includes it. The leader is a large
   share of a value-weighted index *by construction* — it was picked for being
   largest — so regressing it on that index shrinks the leader-specific
   residual toward zero exactly where part 2 needs it informative. Rebuild
   ex-leader; the recipe is in the handoff doc.
2. **Do not drop missing returns before lagging.** `build_lags` shifts by
   *rows*. NaN return rows are kept on purpose. Shift first, drop second, or
   day *t* gets lined up against a day that is not *t−k* — silently, and worst
   for the illiquid stocks whose returns are most autocorrelated.
3. **Do not raise the horizon without raising the Newey-West lag count.** The
   baseline uses single-day leader returns, so observations do not overlap. A
   signal cumulated over a *w*-day window makes consecutive observations share
   *w−1* days.

---

## What each part can start on now

**Part 2 — shock identification.** Fully unblocked. You have
`p1_roles.parquet` (who leads whom, monthly), `p1_industry_returns.parquet`
(the common component, lagged weights, formation-month membership) and the FF5
daily factors. Build the ex-leader industry index first — see trap 1 — then
the decomposition `r_L = α + β_m·mkt + β_j·ind_ex_leader + u_L`. Skeleton and
notes in `src/lead_lag/shocks/__init__.py`.

**Part 3 — signal.** Unblocked. `p1_follower_panel.parquet` is the regression
panel (regenerate it, 4 min). `baseline.horizon_profile` and
`baseline.quintile_sort` are the *unconditional* versions of two exhibits you
need — reuse them so the conditional and unconditional numbers are computed
identically and the comparison is apples to apples. **You must beat
unconditional short-term reversal**, or the conditioning adds nothing.

**Part 4 — portfolio.** Unblocked on inputs: `p1_universe.parquet` says which
names are tradeable each month and carries `med_dollar_vol` to cap position
size; `FactorsDaily` gives the exposure regressions their right-hand side.
You can build and test the optimiser against the *unconditional* signal while
part 3 is still working.

**Part 5 — robustness.** Design now, numbers later. The handles are already
wired as one-line changes (`UniverseRules(min_price=...)`,
`LeaderRule(by="med_dollar_vol")`, `adjusted_daily_returns(repair=False)`,
etc.) — full list in `docs/part1_handoff.md`. The **bounce question is the
one that matters most**: `data/microstructure.py` gives midpoint returns,
relative spread, Roll's bound and weekly compounding, and the fact that the
effect lives where spreads were 3.2% needs an answer.

---

## Getting the data

Raw CRSP and Databento extracts are **not in git** — they are subscriber-
licensed (committing them breaches the WRDS agreement) and the follower panel
alone is 289 MB against GitHub's 100 MB per-file limit. Everything is
reproducible from code:

```bash
cd 230GA/project2
cp .env.example .env          # add your own WRDS_USERNAME / WRDS_PASSWORD
uv venv .venv.nosync --python 3.12
uv pip install --python .venv.nosync/bin/python -r requirements.txt
.venv.nosync/bin/python -m lead_lag.data.wrds_connection   # should print True
.venv.nosync/bin/python scripts/run_part1.py               # ~45 min cold, 4 min warm
```

The first run pulls CRSP daily one calendar year at a time into
`cache.nosync/` (1.1 GB, ~45 min). Every run after that reads the cache and
takes about four minutes. `cache.nosync` is excluded from iCloud sync as well
as git — do not rename it.

If you would rather not pull it yourself, ask me for the cache directory over
a shared drive.

---

## Open / not settled

- **No git remote yet.** The repo is local only. Someone needs to create the
  shared repo and I will push.
- **No out-of-sample holdout is reserved.** Everything so far is full-sample.
  Given the 1996–2006 concentration this needs deciding before parts 3–4 start
  tuning anything.
- **No pre-registered parameter grid**, and the horizon profile scans 10 lags
  × 2 estimators = 20 looks. Lag 7 shows t_pooled = 3.43 with t_FM = −0.99;
  that is noise, and it is the shape of mistake this setup invites.
- **ETF ownership is not sourced.** It is the sharpest conditioning variable
  in the research design and Databento cannot supply it — it needs 13F
  holdings or ETF constituent files. Next on my list.
- **Full intraday pull not started.** Sized at 54.6 GB for `bbo-1m` over
  2018-05→2024-12 (XNAS.ITCH 18.6, XNYS.PILLAR 35.9, XASE.PILLAR 0.1); cost is
  zero on this account's plan. Pilot passed; awaiting the go-ahead.
- **Venue caveat.** The 2018+ datasets are per-venue books, not consolidated
  NBBO. Validation overlay against `DBEQ.BASIC` (2023+) is part 5's.
- **The assignment PDF in `assignment/` is the 2025 edition** and states a
  10 September 2025 deadline. Someone should confirm this year's date.

---

## Log

**2026-09-20** — Part 1 built end to end and run on the full 1996–2024 sample.
Baseline established (Phase 0 of the research design): the diffusion effect
reproduces at +0.0093 / t 5.03 at one day, and is confined to 1996–2006.
Enforced the d−1 rule across all conditioning characteristics after finding
`industry_returns` grouped on same-day classification. Added microstructure
defenses (midpoint returns, Roll's bound, stale-close flag, weekly
compounding) and re-pulled CRSP with `cfacpr`/`quote_only`. Built the
Databento bridge and piloted the intraday leg on 3 industries × June 2023:
pipeline validated, result not significant. Fixed a `cfacpr` direction error
that would have turned every stock split into a −75% midpoint return.
