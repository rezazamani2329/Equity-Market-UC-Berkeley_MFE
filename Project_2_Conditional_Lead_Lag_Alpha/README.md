# Project 2 — Conditional Lead-Lag Alpha

**A risk-constrained long-short equity strategy built on industry lead-lag effects**
UC Berkeley MFE · Equity Markets · 1996–2024 CRSP daily data

> **Bottom line.** Large-cap industry leaders do predict their smaller followers' next-day returns, and that predictability comes from the **common (industry/market) part** of the leader's move, not the leader-specific shock. But the effect is concentrated in 1996–2006, the proposed conditional signal does not beat simpler baselines, and once the portfolio is made dollar-, beta- and own-lag-neutral, traded from the open, and charged realistic costs, nothing is left. **Verdict: do not implement.**

---

## Contents

1. [Research question and hypotheses](#1-research-question-and-hypotheses)
2. [Data and universe](#2-data-and-universe)
3. [Part 1 — Baseline lead-lag](#3-part-1--baseline-lead-lag)
4. [Part 2 — Decomposing the leader's move](#4-part-2--decomposing-the-leaders-move)
5. [Part 3 — Conditional signal and validation](#5-part-3--conditional-signal-and-validation)
6. [Part 4 — Portfolio construction, risk and costs](#6-part-4--portfolio-construction-risk-and-costs)
7. [Summary of findings](#7-summary-of-findings)
8. [Reproducing the results](#8-reproducing-the-results)
9. [Repository structure](#9-repository-structure)

---

## 1. Research question and hypotheses

When a dominant large-cap stock moves, should smaller firms in the same industry **continue** in the same direction (slow information diffusion), or **reverse** when the leader's move was specific to the leader rather than news about the industry?

| # | Hypothesis | Tested in | Outcome |
|---|---|---|---|
| H1 | Followers **continue** on the common (industry/market) component of the leader's move | Parts 2–3 | ✅ Supported |
| H2 | Followers **revert** on the leader-specific shock | Parts 2–3 | ❌ Not supported |
| H3 | Conditioning on the source of the move beats unconditional signals and short-term reversal | Parts 3–4 | ❌ Rejected |
| H4 | The signal stays economically meaningful after risk controls, turnover and costs | Part 4 | ❌ Rejected |

## 2. Data and universe

| Item | Detail |
|---|---|
| Source | CRSP daily stock file, delistings, name history, Fama-French 5 factors + momentum (via WRDS); Databento intraday quotes (pilot) |
| Sample | 1996–2024; 34.4M CRSP rows; 7,173 trading sessions |
| Industries | Fama-French 49, assigned **point-in-time** from CRSP name history (not header SIC) |
| Universe | Monthly, 5 screens (observation count, $5 price floor, NYSE size floor, liquidity, excluding industry "Other"): **~1,747 eligible stocks per month** on average |
| Leaders / followers | Leader = largest firm in each industry, chosen with data through month *m−1*; all other eligible firms are followers (5.4% monthly leader turnover) |
| Regression panel | **12.4M follower-days** |
| Survivorship | Delisting returns merged in (incl. non-trading days); Shumway substitute for 417 missing performance delistings, flagged |
| Look-ahead controls | Every conditioning characteristic is taken at the previous close (the "d−1 rule"); `(date, permno)` uniqueness enforced |

Licensed data (CRSP extracts, stock-level panels) is **not** in this repository; everything is regenerated from code (see [§8](#8-reproducing-the-results)). Only aggregate result tables are committed.

<p align="center">
  <img src="figures/p1_universe_attrition.png" width="48%">
  <img src="figures/p1_stocks_per_year.png" width="48%">
</p>

## 3. Part 1 — Baseline lead-lag

**Model.** Daily Fama-MacBeth cross-sectional regressions of follower returns on the industry leader's lagged return, controlling for the follower's own lagged return, with Newey-West standard errors (pooled, date-clustered estimates as a check):

$$r_{i,t} = a_t + b_t\, r_{L(i),\,t-k} + c_t\, r_{i,\,t-k} + \varepsilon_{i,t}$$

**Results.**

| Lag *k* | β (Fama-MacBeth) | t | β (pooled, clustered) | t |
|---|---|---|---|---|
| 1 | +0.00928 | **5.03** | +0.02202 | 6.84 |
| 2 | +0.00122 | 0.72 | +0.01199 | 3.84 |
| 3 | −0.00050 | −0.30 | +0.00207 | 0.64 |

- **The effect is continuation, not reversal**, and lasts one day. A quintile sort on the leader's lagged return is monotone, with a top-minus-bottom spread of **3.59 bps/day (t = 4.34)**.
- **It is concentrated in 1996–2006:** β = 0.0206 (t = 7.97) then, versus 0.0025 (t = 1.02) in 2007–2024, as median relative spreads fell from 3.23% to 0.11%.
- **It is not bid-ask bounce:** rebuilt on bid-ask midpoint returns, β = 0.00963 (t = 5.19), essentially unchanged.

| Return measure (lag 1) | β_FM | t | 1996–2006 | 2007–2024 |
|---|---|---|---|---|
| Close-to-close | 0.00928 | 5.03 | 0.0206 (t 7.97) | 0.0025 (t 1.02) |
| Close, ex-dividend | 0.00912 | 4.96 | 0.0206 (t 7.97) | 0.0022 (t 0.92) |
| **Bid-ask midpoint** | **0.00963** | **5.19** | **0.0220 (t 8.12)** | 0.0022 (t 0.93) |

<p align="center">
  <img src="figures/p1_coefficient_by_year.png" width="48%">
  <img src="figures/p1_quintile_sort.png" width="48%">
</p>
<p align="center">
  <img src="figures/p1_horizon_profile.png" width="60%">
</p>

An intraday pilot (Databento 5-minute midpoints, June 2023, 22 stocks) validated the pipeline but was not significant (β = −0.044, t = −1.67). Details: `docs/STATUS.md`, `docs/part1_handoff.md`.

## 4. Part 2 — Decomposing the leader's move

**Model.** Each day, split the leader's return into a **common** part explained by the market and an **ex-leader** industry index, and a **leader-specific** residual. The regression is rolling and point-in-time, and the industry index excludes the leader so the residual is not shrunk toward zero:

$$r_{L,t} = \alpha + \beta_m\, \text{MKT}_t + \beta_j\, \text{IND}^{\text{ex-leader}}_{j,t} + u_{L,t}, \qquad \text{common}_t = r_{L,t} - u_{L,t}, \quad \text{shock}_t = u_{L,t}$$

Followers' responses to each component are then estimated with Fama-MacBeth regressions:

$$r_{i,t} = a_t + b_c\, \text{common}_{t-k} + b_u\, \text{shock}_{t-k} + c\, r_{i,t-k} + \varepsilon_{i,t}$$

**Results (lag 1, Fama-MacBeth).**

| Coefficient | Estimate | t |
|---|---|---|
| b_common | **0.0396** | **7.70** |
| b_shock | 0.0034 | 1.86 |
| b_common − b_shock | 0.0362 | 6.90 |

- Followers respond strongly to the **common** component and barely to the leader-specific shock, which supports **H1**.
- The shock coefficient is **positive**, the wrong sign for the reversal in **H2**.
- On average the common component explains only about **40%** of a leader's daily variance (median across 48 industries: 38%). So the predictability comes from the smaller share of the move.

<p align="center">
  <img src="figures/p2_common_vs_shock.png" width="48%">
  <img src="figures/part2_variance_shares.png" width="48%">
</p>

Additional diagnostics in `figures/`: horizon profile (`part2_horizon_profile.png`), rolling betas (`part2_rolling_betas.png`), sign flip (`part2_sign_flip.png`), recovery (`part2_recovery.png`), ex-leader scatter (`part2_exleader_scatter.png`), leader robustness (`part2_leader_robustness.png`).

## 5. Part 3 — Conditional signal and validation

**Signal.** One rankable number per follower-day, combining H1 and H2:

$$\text{conditional\_signal} = \text{common\_lag} - \text{shock\_lag}$$

Each leg is also tested alone, against two baselines: the unconditional Part 1 leader signal (`leader_ret_lag`) and naive own-return reversal (`−own_lag`).

**Validation.** Daily cross-sectional Spearman rank IC at horizons of 1, 5, 10 and 20 days, plus equal-weighted quintile long-short portfolios at h = 1. The signal is known at day *t*'s close and tested on the return from *t* to *t+1*.

**Results (h = 1).**

| Signal | Mean IC | IC t-stat | Long-short (bps/day) | Long-short t-stat |
|---|---|---|---|---|
| Composite (common − shock) | 0.0010 | 0.93 | 3.0 | 3.72 |
| **Common leg** | **0.0049** | **3.52** | **7.1** | **6.76** |
| Shock leg (raw sign) | 0.0007 | 0.72 | 0.5 | 0.76 |
| Part 1 baseline (leader return) | 0.0021 | 1.87 | 3.4 | 3.91 |
| Naive own-return reversal | 0.0166 | 10.91 | 3.8 | 3.23 |

- The **common leg** is the strongest leader-based signal: it doubles the Part 1 baseline's spread.
- The **shock leg** is flat, so subtracting it **dilutes** the composite, which falls below both baselines. H3 is rejected.
- Naive reversal leads on IC, but its spread is about half the common leg's, and its top quintile is the most volatile (consistent with small, illiquid stocks).

<p align="center">
  <img src="figures/p3_ic_by_horizon.png" width="90%">
</p>
<p align="center">
  <img src="figures/p3_quintile_monotonicity.png" width="95%">
</p>

> **Timing correction.** An earlier version tested the signal with a two-day gap (the leader's move on *t−1* against the return on *t+1*), which made it look like noise (composite −0.12 bps/day, t −0.16). Commit `1c1544b` fixed this (`lag=0`); all Part 3 numbers above are corrected. Parts 1, 2 and 4 were not affected. Report: `Conditional_Lead_Lag_Alpha_Report.pdf`.

## 6. Part 4 — Portfolio construction, risk and costs

Everything below follows a **pre-registration** written before any Part 4 result (`docs/part4_prereg.md`). Changes after an independent code audit are listed in `docs/part4_deviations.md`; none changes the verdict.

**Construction.**
- **Industry books** (`conditional_signal`, `common_lag`; appendix: `leader_ret_lag`, `shock_lag`): each day, rank industries by the signal and demean the ranks. Then project out dollar, market-beta and own-lag exposure, split each industry's weight equally across its followers, and scale gross exposure to 1.
- **Reversal benchmark:** stock-level ranks of `−own_lag`, made dollar-, industry- and beta-neutral.
- **Beta:** Dimson (1979) over 252 sessions, shrunk halfway to 1.
- **Signal timing:** the leader's move is lagged one session on the trading calendar, using only information to the close of *t−1*.

**Holding conventions.**

| | Holding period | Role |
|---|---|---|
| **C** | open *t* → close *t* | **Headline** (implementable with market-on-open orders) |
| A | close *t−1* → close *t* | Upper bound (requires trading at the signal's own close) |
| Overnight | close *t−1* → open *t* | Decomposition |

**Primary tests** (annualized mean, Newey-West t, Holm-corrected):

| Book | Convention | Ann. mean | t | Holm p |
|---|---|---|---|---|
| conditional_signal | C | −0.46% | −1.08 | 0.28 |
| conditional_signal | A | −0.70% | −1.48 | 0.28 |
| common_lag | C | +0.85% | 1.90 | 0.17 |
| common_lag | A | +1.19% | 2.39 | 0.07 |

**Decision rule** (all three needed for the headline book):

| Criterion | Result | Passed |
|---|---|---|
| 1. Mean return > 0, Holm p < 0.05 | −0.46%/yr, Holm p = 0.28 | ❌ |
| 2. Spanning alpha > 0 vs. reversal book (open-to-close market) | −0.46%/yr, p = 0.29 | ❌ |
| 3. Breakeven cost > median half-spread (2007–2024) | −0.09 bp vs. 2.58 bp | ❌ |
| **Verdict** | | **DO NOT IMPLEMENT** |

**Costs.** Every leader-based book breaks even at **0.17 bp or less** one-way, against a median half-spread of **2.58 bp**: more than an order of magnitude short. The reversal book reaches 0.50 bp under C.

**What removes the return?** In an exploratory, non-pre-registered analysis that adds the constraints one at a time, `common_lag` earns 4.6%/yr (t 5.6) open-to-close when only dollar-neutral, but just 0.9%/yr (t 1.9) once it is also beta- and own-lag-neutral. Much of the raw signal is market-beta and own-return exposure, not new information.

**Timing.**
- Convention C cannot capture the overnight leg. For the common leg in 2007–2024, close-to-close and open-to-close are similar (about 0.75% vs 0.7%/yr), and the overnight part has almost vanished.
- The reversal book's large close-to-close return (10.1%/yr) shrinks to 1.9%/yr on midpoint returns, so most of it is bid-ask bounce.

**Risk.**
- The headline book runs at only **2.3%** annual volatility. A 10% volatility target would need 4.3× leverage, so the 3× cap binds and the scaled book realizes 6.7%.
- 99% historical VaR is **rejected** by the Kupiec test (99 exceedances vs. 69 expected): the tails are fatter than a trailing window captures.
- Stress periods: Aug 2007 +0.6%, Sep–Oct 2008 +0.5%, Mar 2020 −2.4% (headline book, C).

<p align="center">
  <img src="figures/p4_cumulative.png" width="48%">
  <img src="figures/p4_drawdown.png" width="48%">
</p>
<p align="center">
  <img src="figures/p4_timing_decomposition.png" width="95%">
</p>

Report: `report/Part4_Portfolio_Construction_and_Risk.pdf`.

## 7. Summary of findings

1. **Lead-lag exists:** leaders' returns predict followers' next-day returns (β = 0.0093, t = 5.03). It is continuation, not reversal, and not bid-ask bounce.
2. **It is fading:** almost all of it is in 1996–2006 and statistically absent after 2007.
3. **The information is in the common component:** followers respond to the industry/market part of the leader's move (t = 7.70), not to the leader-specific shock.
4. **The proposed conditional signal fails:** the shock leg adds noise, so the composite underperforms the plain common leg and both baselines.
5. **It is not tradable:** after neutralizing beta and own-lag, trading from the open, and charging realistic costs, every leader-based book earns less than a tenth of its trading cost.

## 8. Reproducing the results

Use a dedicated **Python 3.12** environment. Do not install into a shared base environment: `wrds` pins an older pandas, which can break other projects.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
[ -f .env ] || cp .env.example .env    # only if you have no .env yet; then add WRDS_USERNAME / WRDS_PASSWORD
.venv/bin/python scripts/run_part1.py  # ~45 min first run (pulls CRSP into cache.nosync/), ~4 min after
.venv/bin/python scripts/run_part2.py
.venv/bin/python scripts/run_part3.py
.venv/bin/python scripts/make_p3_figures.py
.venv/bin/python scripts/prep_part4.py
.venv/bin/python scripts/run_part4.py
PYTHONPATH=src:tests .venv/bin/python -m pytest -q   # offline unit tests on a synthetic panel
```

WRDS access is required. `cache.nosync/` and `results/*.parquet` hold licensed data and must never be committed. `scripts/sync_to_team_repo.sh` publishes from a private working copy with an explicit allow-list.

## 9. Repository structure

```
Project_2_Conditional_Lead_Lag_Alpha/
├── README.md
├── Conditional_Lead_Lag_Alpha_Report.pdf      # Part 3 report
├── report/Part4_Portfolio_Construction_and_Risk.pdf
├── docs/
│   ├── STATUS.md                # Part 1 status, data guarantees, open issues
│   ├── part1_handoff.md         # artefacts, guarantees, pitfalls
│   ├── part4_prereg.md          # Part 4 pre-registration
│   └── part4_deviations.md      # post-audit changes
├── src/lead_lag/                # data, baseline, shocks, signals, portfolio
├── scripts/                     # run_part1-4, figure and data-prep scripts
├── tests/                       # unit tests (synthetic panel)
├── notebooks/                   # part1_data_universe, part2_shock_decomposition
├── results/                     # aggregate CSV tables (p1_*, p2_*, p3_*, p4_*)
├── figures/                     # all figures used above
└── appendix/                    # supporting material and ChatGPT interactions
```
