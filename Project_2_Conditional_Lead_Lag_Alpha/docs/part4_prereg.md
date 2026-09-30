# Part 4 pre-registration: portfolio construction and risk

**Owner:** Thomas Claudel (part 4) · **Written:** 2026-09-29, before any Part 4 result was computed.

Everything below is fixed in advance. Anything run afterwards that is not listed here is
labelled exploratory in the report.

## Signals

| Role | Signal | Level | Sign |
|---|---|---|---|
| Primary (thesis) | `conditional_signal` = `common_lag - shock_lag` | industry | long high |
| Primary (Part 2's supported leg) | `common_lag` | industry | long high |
| Benchmark | `reversal_signal` = `-own_lag` | stock | long high |
| Appendix | `leader_ret_lag`, `shock_lag` | industry | long high |

Signs come from the hypotheses. No estimated IC enters the weights: under volatility scaling
the IC magnitude cancels (h = V⁻¹α/2λ with α = IC·ω·z, as in HW2), and the ICs available
from Part 3 were measured one session late.

## Timing

Every signal on row *t* uses information up to the close of *t−1* (`conditional_signal_panel`,
`lag=1`). Three holding conventions:

| | Holding period | Status |
|---|---|---|
| **C** | open *t* → close *t* | **Headline.** Implementable with market-on-open orders. Excludes the overnight catch-up of stale closes. |
| A | close *t−1* → close *t* | Upper bound. Same timing as Parts 1–2; requires trading at the close used to form the signal. |
| B | close *t* → close *t+1* | Reconciliation with Part 3's `fwd_ret_1d` only. |

The overnight leg (1+A)/(1+C) − 1 is reported as a decomposition.

All returns, lags, betas and residual volatilities are computed on the full CRSP daily panel
and then merged onto follower rows, so a row shift never jumps across a gap in follower
membership.

## Construction

**Industry books** (primary and appendix signals). On each date, rank the industries by the
signal, demean the ranks, and project out three exposures across industries: dollar (a
constant), market beta (industry average of follower betas) and own-lag (industry average of
followers' previous-day return). The own-lag constraint mirrors Part 2's regression, which
controls for the follower's own lagged return. Each industry's weight is split equally across
its followers that day. Gross exposure is scaled to 1.

**Reversal book** (benchmark). Stock-level ranks of `reversal_signal`, demeaned within FF49
industry, then projected on within-industry-demeaned beta. That makes the book dollar-,
industry- and beta-neutral. Gross exposure is scaled to 1.

**Beta.** Dimson (1979): sum of slopes on the market excess return at *t* and *t−1*, regressed
on excess stock returns over 252 sessions ending at *t−1*, minimum 60 observations, shrunk
halfway toward 1.

**Volatility overlay** (headline only, as a second version). Scale factor = 10% ÷ annualised
realised volatility of the gross-1 book over the previous 63 sessions, capped at 3.

## Reconciliation rows

1. Part 1 `baseline.quintile_sort` at lag 1 must reproduce 3.59 bp/day, t 4.34.
2. Part 3 `quantile_portfolios` of `leader_ret_lag` on `fwd_ret_1d` must reproduce 0.56 bp/day, t 0.67.
3. The same Part 3 function on the same-day return isolates the effect of the timing alone.

## Metrics

Annualised mean, volatility and Sharpe ratio (Lo 2002 standard error); Newey-West (5 lags)
t-statistic of the mean; maximum drawdown and its duration; one-day-ahead 99% VaR and 97.5%
ES from a trailing 250-session historical window, backtested with Kupiec and Christoffersen
tests; stress periods Aug 2007, Sep–Oct 2008, Mar 2020; daily turnover Σ|Δw|; breakeven
one-way cost = mean daily return ÷ mean daily Σ|Δw|.

Factor attribution, Newey-West (5 lags): mktrf, smb, hml, rmw, cma, umd, lagged mktrf, and
the reversal book under the same convention. The reversal coefficient and the intercept form
the spanning test of Hypothesis 3.

Midpoint returns (price-only, compared against `retx`) are a diagnostic for the reversal book
under convention A. When a held stock has no valid midpoint return, `retx` is used instead.

## Windows

Full 1996–2024; 1996–2006 and 2007–2024 (team convention); post-2010 (from 2010-01-04);
last 18 months (from 2023-07-03); last 12 months (2024). The two recent windows are
descriptive and reported with Sharpe standard errors.

## Primary tests and multiple testing

Primary family (Holm-corrected, two-sided 5%): mean return of the `conditional_signal` and
`common_lag` books under conventions C and A. Four tests. Everything else is exploratory.

## Decision rule

**Implement** only if all three hold for the `conditional_signal` book under convention C:
1. Holm-adjusted mean return significantly positive;
2. spanning-test intercept significantly positive after controlling for the reversal book;
3. breakeven one-way cost above the median half-spread of traded stocks over 2007–2024.

Otherwise **do not implement**, citing these three numbers.
