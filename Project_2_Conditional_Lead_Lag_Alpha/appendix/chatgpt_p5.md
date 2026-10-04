# Part 5: ChatGPT interaction

One interaction, run on ChatGPT on 2026-10-03, after Part 4's results were final and
before any Part 5 result was computed. ChatGPT was therefore used to
design the Part 5 tests and to fix their pass/fail rules in advance, rather than to review
results already in hand. The section below gives the exact prompt, a summary of the output,
the critical evaluation, and what is adopted. The full output is reproduced at the end of
this file.

---

## Interaction 3: robustness design for costs, microstructure and out-of-sample tests

### Prompt

> I am finishing the robustness section of a daily long/short US equity backtest,
> 1996–2024, on CRSP daily data. About 1,750 "follower" stocks a month sit in 48
> Fama-French industries. The signal is the previous day's return of each industry's
> largest stock (the "leader"), split into a market-and-industry component and a
> leader-specific component; every follower in an industry gets the same value. The
> books are built at the industry level, neutral to dollar, market beta and the
> industry's own lagged return, with gross exposure 1 (50% long, 50% short). They are
> formed from information at the previous close, bought at the open and sold at that
> day's close.
>
> Results so far: before costs the books earn between −0.5% and +0.9% a year. Because
> they are flat overnight they trade twice their gross every day. Over 2007–2024 the
> one-way cost at which each book's return is zero is under 0.2 bp, against a median
> quoted half-spread of 2.6 bp for the stocks held. The return is concentrated in
> 1996–2006. The whole sample has already been examined and no holdout was reserved.
> Our current conclusion is "do not implement".
>
> Data I have: CRSP daily open, close, closing bid and ask, volume, shares outstanding
> and delisting returns; Fama-French daily factors; and one-minute best bid and offer
> by exchange (not consolidated) for 1,209 of the stocks, May 2018 to December 2024.
>
> 1. Propose a transaction cost model with the exact formula for cost per stock per
>    day, covering spread, market impact as a function of trade size relative to
>    volume, and trading in the opening and closing auctions. Say which components I
>    can calibrate from this data, which I cannot, and what to assume for those.
> 2. List the microstructure effects that could still create or hide a return in a
>    book formed at the previous close and held open to close. For each, give a test
>    I can run with this data, including how to use the one-minute quotes to test
>    non-synchronous trading. Rank them by how likely they are to matter for this
>    design, and say which do not apply.
> 3. Given that the full sample has been seen, say which out-of-sample tests still
>    carry evidential weight, which are only pseudo-out-of-sample, and how each
>    should be labelled in the report.
> 4. For every test in 1 to 3, state before I run it what result would overturn "do
>    not implement" and what result would confirm it.
> 5. Give Python for the cost model in 1, given a DataFrame with columns date, permno,
>    weight, open, close, bid, ask, volume.

The prompt was written to avoid the two weaknesses Part 4 found in its own prompts. It
gives the turnover, breakeven and spread numbers, so costs can be judged rather than only
flagged, and it asks for a ranking and for what does not apply, so the answer is not a flat
checklist.

### Summary

- **Costs.** A per-stock, per-leg cost made of the quoted half-spread on the part of the
  order traded in the continuous market, a square-root impact term (coefficient ×
  daily volatility × √(shares traded ÷ volume available)) for continuous and auction
  trading separately, and an auction "dislocation" allowance. Three scenarios fixed in
  advance: spread only with zero impact (most favourable), central, and conservative, with
  an instruction not to tune the impact coefficient.
- **What the data identify.** Closing quoted spreads are measurable; the opening spread
  only by proxy (the previous close's quote); impact coefficients, auction volumes, fees
  and effective spreads are not measurable from this data and must be assumed.
- **A historical warning.** Nasdaq's opening and closing crosses began in 2004, so modern
  auction assumptions do not carry back to 1996–2003, the period where the return is
  concentrated.
- **Microstructure.** Ten effects ranked: bounce in the leader's signal, stale leader
  prices, the opening print, the closing print, bounce in the follower's return, price
  discreteness, intraday seasonality, non-consolidated quotes, index-rebalance dates, and
  overnight information. Each has a test and an overturn/confirm rule. Most tests use the
  one-minute quotes.
- **Out-of-sample.** Nothing inside 1996–2024 is genuine out-of-sample: sub-period splits
  and held-out stocks or industries are pseudo-out-of-sample, the 2018–2024 quote data are
  an independent data source but not out-of-sample, and only data after 2024, tested with
  the specification locked, would be genuine.
- **Decision rules.** An 18-row table of overturn/confirm outcomes, and six conditions that
  must all hold to overturn "do not implement".
- **Code.** A Python function implementing the cost model, with capital, impact
  coefficients and auction shares as explicit inputs.

### Critical evaluation

**Where it was right, checked against the repository**

- **The cost arithmetic.** A 0.2 bp breakeven on turnover of 2 a day is about 1.0% a year,
  and crossing a 2.6 bp half-spread on both legs is about 13.1% a year. Both are correct
  and both match Part 4's convention that an open-to-close book trades twice its gross
  (`docs/part4_deviations.md`, item 1).
- **Spread-only as the deciding case.** If a book fails with zero impact, the impact
  exponent and coefficient do not matter. This makes the cheapest test the decisive one
  and keeps Part 5 from resting on parameters nobody can estimate.
- **The 2004 auction change.** This was not on our list. It matters because Part 1 found
  the effect confined to 1996–2006, so most of the period that produces the return
  cannot be costed with auction assumptions at all.
- **The out-of-sample labels.** It refused to call any split of 1996–2024 out-of-sample and
  made the same distinction for the 2018–2024 quote data. This agrees with the open item in
  `docs/STATUS.md` ("no out-of-sample holdout is reserved").
- **It says what cannot be measured.** It states that an Amihud measure is not an impact
  coefficient and that a gap between the open and a minute quote is not causal impact.
- **The data fit.** Every column its cost function needs (open, close, bid, ask, volume) is
  already pulled by `data/wrds_fetch.py`.

**Where it was wrong or would have misled**

- **Bounce is ranked first, but Part 1 has already cleared it.** On midpoint returns the
  lead-lag coefficient is 0.00963 (t = 5.19) against 0.00928 (t = 5.03) on closes. Our
  prompt did not say so, which is our omission; but its mechanism is also weaker than it
  suggests, since bounce contaminates a stock's own autocorrelation and the follower's own
  lagged return is already controlled for.
- **Most of its tests can only be run where there is nothing to explain.** The one-minute
  quotes cover May 2018 to December 2024, and the effect is about zero after 2007. The
  quote-based tests (ranks 1 to 5, 7 and 8) can show the effect is absent on clean prices
  recently, but cannot say whether the 1996–2006 return was stale pricing. It noted that
  the quote sample is a "diagnostic sample" but did not draw this conclusion. Only its
  close-within-spread test, its price and spread sorts and its calendar exclusions use
  daily data and so reach the early period.
- **Its impact numbers are assumptions.** The coefficient of 0.75 and the auction volume
  shares of 1.5% and 10% come from outside this data, as it says itself. They can only
  be used as a stress test.
- **The volatility in its code looks ahead by one day.** The rolling standard deviation
  includes the same day's return. The effect on a cost estimate is small, but the fix is
  one `shift(1)`.
- **Some exclusions do not apply.** It proposes dropping stocks below $2 and below $5; the
  universe already has a $5 price floor.
- **One of its tests is already done.** Its rank 10, splitting the follower's return into
  overnight and open-to-close parts, is Part 4's timing decomposition.
- **Its overturn rule cannot apply to the headline book.** The conditional signal loses
  money before costs (−0.46% a year), so no cost result can rescue it. The cost tests are
  informative only for `common_lag`, `leader_ret_lag` and `shock_lag`.
- **Length.** Ten ranked effects and an 18-row decision table are more than this result
  needs; three or four of the tests decide it.

**Adopted.** Part 5 takes three things from this answer. (1) The spread-only cost test is
the primary cost result (`scripts/run_part5.py`, `results/p5_net_of_cost.csv`): at the
2007–2024 median half-spread of 2.58 bp with zero impact, every book loses between 10.8%
and 13.5% a year open to close, so the test confirms "do not implement". (2) Its
out-of-sample labels are used as written: every split of 1996–2024 in Part 5 is reported
as pseudo-out-of-sample. (3) Its overturn/confirm rule was fixed before any Part 5 number
was computed. Its 2004 auction point is added to the limitations. Not carried out: the
square-root impact stress and its cost function, which need stock-level weights and
volume and cannot change a result that already fails on spread alone; the one-minute tests
of stale leader prices and of the opening and closing prints; and a locked-specification
run on data after 2024, which remains the only genuine out-of-sample test.

---

## Reflection (report section 3d)

Used before the tests were run, ChatGPT was most useful for discipline rather than ideas.
Given the turnover and breakeven numbers, it went straight to the point that decides the
section: a book that needs a one-way cost under 0.2 bp cannot survive a 2.6 bp half-spread,
so the spread-only test settles the cost question and impact modelling is secondary. It
labelled out-of-sample evidence more strictly than we had, and it supplied pass/fail rules
we could commit to in advance. It also raised one fact we had missed, that auction trading
on Nasdaq began only in 2004. Its weakness was the same as in Part 4: it could not judge
which checks matter for this dataset. It ranked bid-ask bounce first although Part 1 had
already ruled it out, and it built most of its tests on minute data that cover only the
years in which the effect is already gone. Its code, checked by reading and not yet run, is
structurally sound but has a one-day look-ahead in the volatility estimate. Giving it our numbers fixed the "cannot
quantify costs" problem from Part 4; not giving it our earlier findings produced a test
list that partly repeats work already done.

---

## Full output

### Interaction 3 output

The main conclusion is stronger than “the backtest has low returns.” Given the turnover you describe, **the strategy’s economic hurdle is extraordinarily low**: a one-way cost below 0.2 bp is needed to erase the gross return. A median quoted half-spread of 2.6 bp is already more than an order of magnitude above that hurdle, before impact. I would therefore treat the remaining work primarily as a test of whether the apparent return is a **measurement/microstructure artifact**, not as a search for a cost specification that might rescue it.

CRSP provides daily open, close, bid/ask and volume, but its daily bid/ask are closing quotes and are not an execution record; for example, CRSP describes the NASDAQ closing bid as an inside quote but the NYSE series as the last representative quote.

##### 1\. Transaction-cost model

###### 1.1 The model I would preregister

Let:

- $A$ = portfolio NAV.
- $w_{it}$ = dollar portfolio weight of stock $i$, before the intraday round trip.
- $P^O_{it},P^C_{it}$ = CRSP open and close.
- $Q^O_{it}=|w_{it}|A/P^O_{it}$ = shares bought/sold at the open.
- $Q^C_{it}=|w_{it}|A/P^C_{it}$ = shares bought/sold at the close.
- $V_{it}$ = CRSP daily volume in shares.
- $h^O,h^C$ = quoted half-spread as a fraction of price.
- $\sigma_{it}$ = daily volatility in decimal units.
- $a^O,a^C$ = fraction of each order routed to the opening/closing auction.
- $\phi^O,\phi^C$ = fraction of daily volume occurring in those auctions.
- $d^O,d^C$ = non-impact auction-dislocation allowance.
- $Y_C,Y_O,Y_C^{auc}$ = impact coefficients.

For an execution mechanism $m$, use the square-root impact specification

$$
I_{mit}
=
Y_m\sigma_{it}
\sqrt{\frac{Q_{mit}}{V_{mit}}}.
$$

The square-root form is preferable to a linear model: recent US evidence finds it fits continuous-market and auction impact substantially better, including closing-auction impact.

For each **opening leg**, define

$$
\begin{aligned}
c^O_{it}
={}&
(1-a^O_{it})h^O_{it}
+a^O_{it}d^O_{it}\\
&+
Y_C\sigma_{it}
\sqrt{
\frac{(1-a^O_{it})Q^O_{it}}
{(1-\phi^O_{it})V_{it}}
}\\
&+
Y_O\sigma_{it}
\sqrt{
\frac{a^O_{it}Q^O_{it}}
{\phi^O_{it}V_{it}}
}.
\end{aligned}
$$

For the **closing leg**,

$$
\begin{aligned}
c^C_{it}
={}&
(1-a^C_{it})h^C_{it}
+a^C_{it}d^C_{it}\\
&+
Y_C\sigma_{it}
\sqrt{
\frac{(1-a^C_{it})Q^C_{it}}
{(1-\phi^C_{it})V_{it}}
}\\
&+
Y_{C}^{auc}\sigma_{it}
\sqrt{
\frac{a^C_{it}Q^C_{it}}
{\phi^C_{it}V_{it}}
}.
\end{aligned}
$$

Then the **exact modeled dollar cost for stock $i$ on day $t$** is

$$
\boxed{
C_{it}
=
|w_{it}|A
\left(c^O_{it}+c^C_{it}\right)
}
$$

and the portfolio cost is

$$
\boxed{
TC_t=\sum_i C_{it}/A.
}
$$

Equivalently, $TC_t$ is the NAV-weighted sum of the two one-way cost fractions.

For a pure continuous-market round trip, $a^O=a^C=0$, and this reduces to

$$
C_{it}
=
|w_{it}|A
\left[
h^O+h^C
+
Y_C\sigma
\left(
\sqrt{\frac{Q^O}{(1-\phi^O)V}}
+
\sqrt{\frac{Q^C}{(1-\phi^C)V}}
\right)
\right].
$$

For a pure auction execution, $a^O=a^C=1$, and the quoted-spread terms disappear; you instead pay the auction-dislocation allowance plus auction impact.

###### 1.2 What your data actually identify

| Component | Your data | Calibration status |
| --- | --- | --- |
| Closing quoted spread | CRSP bid/ask | **Directly calibratable**, subject to CRSP quote-quality issues |
| Previous-close spread used as open proxy | CRSP bid/ask | **Calibratable proxy**, not actual opening spread |
| Opening spread | No intraday CRSP quote | **Not directly calibratable** |
| Continuous impact | Daily volume + prices | **Not causally calibratable** |
| Amihud-style illiquidity | Return / dollar volume | **Calibratable proxy**, not impact |
| Auction volume | Not supplied | **Not calibratable** |
| Auction impact | Not supplied | **Not calibratable** |
| Auction price dislocation | One-minute BBO + CRSP open/close for 1,209 stocks | **Partially calibratable**, 2018–24 only |
| Effective spread | No trade direction/trade data in the minute file | **Not directly calibratable** |
| Realized spread | No signed trades | **Not calibratable** |
| Permanent vs temporary impact | No order-level data | **Not calibratable** |
| Exchange routing | Exchange-specific BBO exists for 1,209 names | **Partially testable** |
| Fees/rebates | None | **Not calibratable** |

This distinction matters. You should **not call an Amihud coefficient a calibrated market-impact coefficient**. Nor should you infer auction impact from the difference between the open/close and a minute quote and describe it as causal impact.

The recent Goyal–Jegadeesh–Wu evidence is useful as an external benchmark: closing auctions tend to have lower impact than continuous trading, while opening auctions are substantially more expensive; their estimated closing-auction impact is square-root in order size.

###### 1.3 What I would assume

I would report three prespecified scenarios rather than one allegedly “correct” estimate.

**Scenario A — optimistic lower bound**

    - continuous execution;
- quoted half-spread only;
- zero market impact;
- no fees;
- no auction penalty.

This is deliberately favorable to the strategy.

**Scenario B — central**

- quoted half-spread;
- square-root impact;
- $Y$ fixed ex ante, e.g. 0.5–1.0;
- auction impact using external auction-volume assumptions;
- auction dislocation allowance estimated from the 2018–24 minute data.

**Scenario C — conservative**

- $Y$ at the upper end of the prespecified range;
- opening-auction impact larger than closing-auction impact;
- use the 75th/90th percentile rather than mean auction dislocation;
- include reasonable commissions/fees;
- no assumption that you receive midpoint or price improvement.

I would **not tune $Y$** until the strategy becomes profitable.

A useful reality check is that recent evidence puts closing-auction impact for a 1%-ADV order at a mean of 17.7 bp and median of 8.4 bp, with large cross-sectional heterogeneity; opening-auction impact is higher. Those are not estimates for your stocks or period, but they demonstrate how implausible a sub-0.2-bp all-in one-way cost can be for a strategy that trades meaningful fractions of volume.

###### 1.4 Important historical auction issue

Do **not** mechanically apply today's auction assumptions back to 1996.

NASDAQ's formal Opening and Closing Crosses were introduced in 2004; its Closing Cross launched in April 2004 and the Opening Cross rollout was completed in December 2004.

Thus:

- 1996–2003 needs a different opening/closing execution model.
- 2004 onward needs exchange/security-specific treatment.
- This is especially important because your apparent return is concentrated in **1996–2006**, exactly where today's auction assumptions are least portable.

---

#### 2\. Microstructure effects that could create or hide the return

I would rank them as follows.

##### Rank 1 — Bid/ask bounce in the leader signal

**Very likely.**

Your signal uses yesterday's leader return. If the leader's previous close happens to be at the bid one day and the ask the next, the measured return contains a mechanical component even when fundamental value does not move. This is a well-known issue with CRSP daily transaction prices.

###### Test

For the 1,209-stock minute sample:

1. Construct the leader signal from CRSP transaction prices.
2. Reconstruct it using:\

  $$
  r^{mid}_{L,t-1}
     =
     \log(M_{L,t-1}^{C})-\log(M_{L,t-2}^{C}),
  $$
  \
   where $M=(Bid+Ask)/2$.
3. Re-form exactly the same portfolio using the midpoint signal.
4. Compare:
  - mean return;
  - t-statistic;
  - leader-component coefficient;
  - contribution of 1996–2006 versus 2007–24.

Also use the CRSP close relative to its closing midpoint:

$$
z_{it}
=
\frac{P^C_{it}-M^C_{it}}
{(Ask^C_{it}-Bid^C_{it})/2}.
$$

Test whether the next-day follower return is related to $z_{L,t-1}$.

###### Overturn

The return remains economically similar using midpoint signals, and the signal-return relation is not concentrated in extreme closing-quote positions.

###### Confirm

The return falls materially or disappears using midpoint signals, particularly for stocks with large spreads or extreme close-within-spread positions.

---

##### Rank 2 — Non-synchronous trading / stale leader prices

**Very likely for the small stocks; probably less important for the largest leaders.**

Classic nonsynchronous-trading results show that infrequent trading can generate serial and cross-serial correlations that look like predictability.

Your one-minute quotes make this unusually testable.

###### The best test

For every stock-day in the 1,209-name sample:

1. Consolidate exchange-specific quotes at each minute:\

  $$
  NBBO\ Bid_t=\max_e Bid_{et},
     \qquad
     NBBO\ Ask_t=\min_e Ask_{et}.
  $$
2. Construct $M_t$, the consolidated midpoint.
3. Measure **quote staleness** at the close:
  - minutes since last midpoint change;
  - minutes since last bid change;
  - minutes since last ask change;
  - fraction of the final 30 minutes with no quote update.
4. Construct several leader returns:
  - close-to-close midpoint;
  - 15:59-to-15:59 midpoint;
  - 15:55-to-15:59 midpoint;
  - last 5-minute midpoint return;
  - last 30-minute midpoint return.
5. Re-run the strategy using each.
6. More importantly, estimate:

$$
R^{F}_{i,t}
=
\alpha+
\beta r^{L}_{g,t-1}
+
\gamma
\left(r^{L}_{g,t-1}\times Stale_{g,t-1}\right)
+\text{controls}.
$$

If $\gamma$ is large, the apparent leader effect is disproportionately coming from stale leader prices.

###### A stronger lead/lag test

At one-minute frequency estimate

$$
r^F_{i,t,\tau}
=
\alpha+
\sum_{k=-K}^{K}\beta_k r^L_{g,t,\tau+k}
+\epsilon.
$$

Do this separately for:

- first 30 minutes;
- middle of day;
- last 30 minutes.

A genuine previous-close leader signal should not be reducible to a mechanical **follower lag of a leader price update**.

###### Overturn

The signal remains after midpoint construction, remains when conditioning on quote freshness, and the one-minute cross-correlation does not show that the apparent daily effect is simply delayed incorporation of the leader's return.

###### Confirm

The effect is strongest when the leader's final quote is stale, disappears when using midpoint/last-active-minute prices, or appears as a clear leader-return → follower-midpoint-return lag pattern.

---

##### Rank 3 — Opening-price microstructure

**Very likely.**

You enter exactly at the open. The open is not an ordinary observation from the continuous market. Opening auctions incorporate overnight information and can have substantial price impact. Recent evidence finds opening-auction impact larger than closing-auction impact.

###### Tests

On the 1,209 stocks:

- calculate the return from the previous close to:
 - 9:30 midpoint;
 - first minute midpoint;
 - 5-minute midpoint;
 - 15-minute midpoint;
 - CRSP open.

Then compare your strategy's return over:

1. CRSP open → CRSP close;
2. 9:31 midpoint → 15:59 midpoint;
3. 9:35 midpoint → 15:55 midpoint;
4. 10:00 midpoint → 15:30 midpoint.

###### Diagnostic

If almost all of the strategy's return occurs between the open and 9:35, that is a major warning.

###### Overturn

The strategy's alpha survives removal of the first 5–30 minutes.

###### Confirm

The return disappears when the first 5–30 minutes are excluded.

---

##### Rank 4 — Closing-auction effects

**Very likely.**

You exit exactly at the close. Closing auctions are a special price-discovery mechanism, and closing prices can contain temporary price pressure. Recent work finds significant auction effects and substantial heterogeneity across stocks.

###### Tests

On the minute sample, replace the exit price by:

- 15:55 midpoint;
- 15:58 midpoint;
- 15:59 midpoint;
- CRSP close.

Compare the strategy return for each.

Also calculate

$$
D^C_{it}
=
\frac{P^C_{it}-M_{it,15:59}}
{M_{it,15:59}}.
$$

Regress $D^C$ on the strategy's desired trade direction.

If longs systematically have positive close dislocations and shorts negative ones, your strategy is interacting with closing-price formation.

###### Overturn

The signal remains after exiting at 15:55–15:59 midpoint prices.

###### Confirm

Most of the return is generated by the final minute/closing price.

---

##### Rank 5 — Bid/ask bounce in the follower return

**Likely.**

The open and close are both transaction prices. The round trip can mechanically produce a return even if the efficient price barely changes.

###### Test

Compare:

$$
R^{trade}_{it}
=
\log(P^C/P^O)
$$

with

$$
R^{mid}_{it}
=
\log(M_{15:59}/M_{9:31}).
$$

Also estimate the strategy on:

- transaction prices;
- midpoint prices;
- bid/ask conservative execution prices.

###### Overturn

Midpoint returns retain the economic magnitude.

###### Confirm

The return largely vanishes using midpoint returns.

---

##### Rank 6 — Price discreteness / tick effects

**Moderately likely**, especially because you have \~1,750 followers and the return is tiny.

This is particularly relevant to low-priced stocks. Historical CRSP work finds bid/ask effects become more important as spreads widen and prices fall.

###### Tests

Sort stocks into quintiles by:

- price;
- relative spread;
- absolute spread;
- market cap.

Run the strategy separately in each.

Also exclude:

  - stocks below $2;
- below $5;
- top 10% relative spreads.

###### Overturn

The return survives removing low-price/high-spread stocks.

###### Confirm

The return is concentrated in the lowest-price/highest-spread buckets.

---

##### Rank 7 — Intraday seasonality

**Moderately likely.**

There is substantial evidence that the first and last parts of the trading day behave differently from the middle of the day, including predictable cross-sectional patterns and short-term reversal associated with liquidity imbalances.

###### Test

With one-minute midpoints, calculate strategy returns in:

- 9:30–10:00;
- 10:00–11:00;
- 11:00–14:00;
- 14:00–15:00;
- 15:00–15:30;
- 15:30–16:00.

###### Overturn

The effect is reasonably distributed across the day.

###### Confirm

It is essentially a first/last-half-hour phenomenon.

---

##### Rank 8 — Exchange fragmentation / non-consolidated quotes

**Moderately likely in the 2018–24 diagnostic sample.**

Because your one-minute BBO is exchange-specific, using one exchange can make a quote appear stale when another exchange has already moved.

###### Test

Construct:

1. primary-exchange midpoint;
2. consolidated NBBO midpoint;
3. cross-exchange dispersion:\

  $$
  Disp_t = \max_e M_{et}-\min_e M_{et}.
  $$

Repeat the strategy using each.

###### Overturn

Results are unchanged using consolidated quotes.

###### Confirm

The apparent effect is substantially weaker using NBBO.

---

##### Rank 9 — Closing/opening index and rebalance effects

**Moderately likely but easy to test.**

Closing auctions can be particularly large on index-rebalance dates. The NASDAQ Closing Cross, for example, was explicitly used for Russell reconstitutions from 2004 onward.

###### Tests

Exclude:

- month-end;
- quarter-end;
- year-end;
- known index-reconstitution dates;
- triple-witching/expiration dates if available.

###### Overturn

The return survives all exclusions.

###### Confirm

The return is concentrated on these dates.

---

##### Rank 10 — Overnight information

**Not itself an explanation of your intraday return.**

You are flat overnight, so you do **not** earn the overnight return.

But overnight information can affect the opening auction and therefore the open-to-close return.

###### Test

Split the follower return into:

$$
R_{CO}=\log(P^O/P^C_{t-1})
$$

and

$$
R_{OC}=\log(P^C/P^O).
$$

Your strategy earns only $R_{OC}$. Check whether the leader signal predicts the overnight component or only the intraday component.

###### Overturn

The strategy continues to earn positive $R_{OC}$ after controlling for the opening gap.

###### Confirm

The apparent relation is mostly an artifact of the opening price incorporating information that was not available at the previous close.

---

##### Effects that essentially do **not** apply

- **Overnight drift as a directly earned strategy return:** no; you are flat overnight.
- **Delisting returns as an explanation for open-to-close microstructure:** no, except as a separate backtest/data-integrity issue.
- **Market beta itself:** largely addressed by your construction, though nonsynchronous beta estimation can still contaminate the hedge.
- **Industry lagged return:** explicitly neutralized, though it does not eliminate stock-level bid/ask or timing effects.
- **Survivorship bias:** not a microstructure effect; CRSP's security history/delisting information addresses it if used correctly. CRSP explicitly supplies delisting returns and permanent identifiers.

---

#### 3\. What "out of sample" still means after you've seen the whole sample

This distinction is important.

###### Genuine out-of-sample

A test is genuinely OOS only if **the observations used to evaluate it were unavailable to you when the specification, signal, sample rules and decision criteria were fixed**.

For your existing 1996–2024 data, that condition is no longer satisfied.

So:

| Exercise | Correct label |
| --- | --- |
| 1996–2006 vs 2007–24 split | **Pseudo-OOS / temporal subsample** |
| 2007–24 cost analysis | **Pseudo-OOS robustness** |
| 2018–24 one-minute quotes | **Independent-data-source robustness, but not temporal OOS** |
| Holding out random stocks now | **Pseudo-OOS cross-sectional validation** |
| Leave-one-industry-out now | **Pseudo-OOS cross-sectional validation** |
| New specification using same 1996–24 data | **In-sample robustness** |
| New test on 2025–26 data never previously inspected | **Genuine temporal OOS**, if specification was locked beforehand |
| A future live paper trade | **Prospective/live validation** |

The 2018–24 quote data are particularly valuable, but do **not** call them “out of sample” simply because they are a different data vendor/file. The dates overlap your already-seen backtest.

###### What I would put in the paper

Something like:

> “Because the complete 1996–2024 sample was examined during strategy development, all subsequent sample splits and microstructure tests using that period are treated as robustness analyses rather than genuine out-of-sample tests. The 2018–2024 quote-based analyses constitute an independent-data-source validation but not a temporal out-of-sample test.”

That is much more defensible.

---

#### 4\. Precommit the decision rules

I would make this explicit **before running the tests**.

Your existing economics imply:

$$
c_{\rm BE}<0.2\text{ bp per one-way dollar traded}.
$$

Because turnover is approximately 2 per day, the maximum annual gross return attributable to a 0.2-bp one-way cost is approximately

$$
252\times 2\times0.2{\rm bp}
\approx1.01\%.
$$

That is already larger than the gross returns you report for most specifications.

And a 2.6-bp half-spread implies approximately

$$
252\times2\times2.6{\rm bp}
\approx13.1\%
$$

of annualized round-trip spread cost **if you crossed that spread every day**. That is not a realistic cost estimate for every stock, but it illustrates how extreme the economic hurdle is.

##### Decision matrix

| Test | Result that would overturn “do not implement” | Result confirming it |
| --- | --- | --- |
| Quoted spread only | Strategy remains profitable under actual conservative execution | Net return negative |
| Square-root impact | Net return comfortably positive under central and conservative $Y$ | Negative under even modest impact |
| Auction scenario | Opening/closing auction execution leaves positive net return | Auction costs eliminate it |
| Midpoint leader signal | Signal survives with similar magnitude | Signal collapses |
| Midpoint follower return | Alpha survives | Alpha collapses |
| Stale-quote interaction | No concentration in stale leaders | Alpha concentrated in stale leaders |
| One-minute lead/lag | No delayed-price explanation | Clear leader → follower lag explains return |
| Remove first 5–30 minutes | Return survives | Return disappears |
| Remove final 5–30 minutes | Return survives | Return disappears |
| Consolidated NBBO | Similar result | Result materially weaker |
| Low-price exclusion | Similar result | Effect concentrated in low-priced names |
| High-spread exclusion | Similar result | Effect concentrated in wide spreads |
| Month/quarter/index-event exclusions | Similar result | Effect concentrated in event days |
| 1996–2006 exclusion | Positive economically significant return remains | Return disappears |
| 2007–24 alone | Positive net return with plausible costs | Negative |
| Quote-data 2018–24 | Same directional effect with economically meaningful size | Effect absent |
| New genuine OOS period | Positive net return after locked costs | Zero/negative |
| All robustness tests jointly | Positive return survives _and_ plausible execution cost is below the break-even hurdle | Any realistic cost/measurement correction eliminates it |

The key is **“economically meaningful”**, not merely $p<0.05$. With a strategy whose gross return is 10–90 bp/year, a statistically significant 5-bp/year effect is not enough to implement.

I would require, for an overturn:

1. positive net return under the **central** cost model;
2. positive net return under a reasonably conservative model;
3. survival using quote midpoints;
4. survival after removing open/close minutes;
5. survival in 2007–24;
6. ideally, positive genuine post-2024 OOS performance.

If those conditions aren't met, “do not implement” survives.

---

#### 5\. Python cost model

The code below deliberately makes the controversial assumptions explicit rather than hiding them.

It produces:

- opening/closing quoted-spread cost;
- continuous-market square-root impact;
- opening/closing auction impact;
- auction dislocation allowance;
- total stock-day cost in dollars and bps of NAV.

It uses the **previous day's CRSP closing quote as the opening-spread proxy**, because your supplied DataFrame has no opening quote. That should be labeled as a proxy, not a measured opening spread.

```
import numpy as np
import pandas as pd

def transaction_cost_model(
    df,
    nav=1_000_000.0,
    vol_window=60,
    # Square-root impact coefficients.
    # These are assumptions, not estimates from the supplied daily data.
    Y_cont=0.75,
    Y_open_auction=1.00,
    Y_close_auction=0.75,

    # Fraction of OUR order routed to the auction.
    # Run multiple scenarios rather than treating these as known.
    auction_frac_open=1.0,
    auction_frac_close=1.0,

    # Assumed fraction of total daily volume occurring in each auction.
    # Recent-market benchmarks are roughly 1-2% at the open and ~10% at
    # the close, but these are NOT appropriate as historical 1996-2024
    # estimates without further calibration.
    auction_volume_frac_open=0.015,
    auction_volume_frac_close=0.10,

    # Auction dislocation allowance, in basis points.
    # Set from the minute quote analysis once that has been done.
    auction_dislocation_open_bps=0.0,
    auction_dislocation_close_bps=0.0,

    # Optional hard cap to avoid absurd impact from tiny volumes.
    max_participation=None,
):
    """
    Estimate stock-day transaction costs for an intraday round trip.

    Required columns:
        date, permno, weight, open, close, bid, ask, volume

    Interpretation:
        weight = dollar portfolio weight, e.g. +0.001 or -0.001.
        nav    = portfolio NAV in dollars.

    Returns:
        Original rows plus:
            half_spread_open
            half_spread_close
            sigma_daily
            q_open
            q_close
            spread_cost_open
            spread_cost_close
            impact_cont_open
            impact_cont_close
            impact_auction_open
            impact_auction_close
            auction_cost_open
            auction_cost_close
            cost_frac_open
            cost_frac_close
            cost_dollars
            cost_bps_nav
    """

    required = {
        "date", "permno", "weight", "open",
        "close", "bid", "ask", "volume"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["permno", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # 1. Basic validity checks
    # ------------------------------------------------------------

    for col in ["open", "close", "bid", "ask", "volume", "weight"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    valid_quote = (
        x["bid"].notna()
        & x["ask"].notna()
        & (x["bid"] > 0)
        & (x["ask"] >= x["bid"])
    )

    x["mid_close"] = (x["bid"] + x["ask"]) / 2.0

    # ------------------------------------------------------------
    # 2. Daily volatility for square-root impact
    #
    # This is DAILY volatility, not annualized volatility.
    # ------------------------------------------------------------

    x["log_close"] = np.log(x["close"].where(x["close"] > 0))
    x["log_ret"] = x.groupby("permno")["log_close"].diff()

    x["sigma_daily"] = (
        x.groupby("permno")["log_ret"]
         .transform(
             lambda s: s.rolling(
                 vol_window,
                 min_periods=max(20, vol_window // 2)
             ).std()
         )
    )

    # ------------------------------------------------------------
    # 3. Opening quote proxy
    #
    # CRSP bid/ask are closing quotes, so the previous day's
    # closing quote is the only quote-based proxy available from
    # the supplied daily data for liquidity at the next open.
    # ------------------------------------------------------------

    x["bid_prev"] = x.groupby("permno")["bid"].shift(1)
    x["ask_prev"] = x.groupby("permno")["ask"].shift(1)

    prev_valid_quote = (
        x["bid_prev"].notna()
        & x["ask_prev"].notna()
        & (x["bid_prev"] > 0)
        & (x["ask_prev"] >= x["bid_prev"])
    )

    prev_mid = (x["bid_prev"] + x["ask_prev"]) / 2.0

    x["half_spread_open"] = (
        0.5 * (x["ask_prev"] - x["bid_prev"]) / prev_mid
    )

    # Current closing quoted half-spread.
    x["half_spread_close"] = (
        0.5 * (x["ask"] - x["bid"]) / x["mid_close"]
    )

    # Invalid spreads -> NaN rather than silently treating them as zero.
    x.loc[~prev_valid_quote, "half_spread_open"] = np.nan
    x.loc[~valid_quote, "half_spread_close"] = np.nan

    # ------------------------------------------------------------
    # 4. Dollar notional and shares traded on each leg
    # ------------------------------------------------------------

    x["notional"] = x["weight"].abs() * nav

    x["q_open"] = (
        x["notional"] / x["open"].where(x["open"] > 0)
    )

    x["q_close"] = (
        x["notional"] / x["close"].where(x["close"] > 0)
    )

    # ------------------------------------------------------------
    # 5. Auction/continuous order decomposition
    # ------------------------------------------------------------

    ao = float(auction_frac_open)
    ac = float(auction_frac_close)

    if not (0.0 <= ao <= 1.0 and 0.0 <= ac <= 1.0):
        raise ValueError("auction_frac_* must lie between 0 and 1.")

    # Shares routed to each mechanism.
    x["q_open_auc"] = ao * x["q_open"]
    x["q_open_cont"] = (1.0 - ao) * x["q_open"]

    x["q_close_auc"] = ac * x["q_close"]
    x["q_close_cont"] = (1.0 - ac) * x["q_close"]

    # Approximate volume available to each mechanism.
    #
    # If the assumed auction share is zero, there is no auction
    # execution and its impact is zero.
    x["v_open_auc"] = auction_volume_frac_open * x["volume"]
    x["v_close_auc"] = auction_volume_frac_close * x["volume"]

    x["v_open_cont"] = (
        (1.0 - auction_volume_frac_open) * x["volume"]
    )
    x["v_close_cont"] = (
        (1.0 - auction_volume_frac_close) * x["volume"]
    )

    # ------------------------------------------------------------
    # 6. Optional participation caps
    # ------------------------------------------------------------

    def safe_participation(q, v):
        p = q / v.replace(0, np.nan)

        if max_participation is not None:
            p = p.clip(upper=max_participation)

        return p

    p_open_cont = safe_participation(
        x["q_open_cont"], x["v_open_cont"]
    )
    p_close_cont = safe_participation(
        x["q_close_cont"], x["v_close_cont"]
    )
    p_open_auc = safe_participation(
        x["q_open_auc"], x["v_open_auc"]
    )
    p_close_auc = safe_participation(
        x["q_close_auc"], x["v_close_auc"]
    )

    # ------------------------------------------------------------
    # 7. Spread cost
    #
    # Only the continuous fraction pays the quoted half-spread.
    # ------------------------------------------------------------

    x["spread_cost_open"] = (
        (1.0 - ao) * x["half_spread_open"]
    )

    x["spread_cost_close"] = (
        (1.0 - ac) * x["half_spread_close"]
    )

    # ------------------------------------------------------------
    # 8. Continuous-market square-root impact
    # ------------------------------------------------------------

    x["impact_cont_open"] = (
        Y_cont
        * x["sigma_daily"]
        * np.sqrt(p_open_cont)
    )

    x["impact_cont_close"] = (
        Y_cont
        * x["sigma_daily"]
        * np.sqrt(p_close_cont)
    )

    # ------------------------------------------------------------
    # 9. Auction square-root impact
    # ------------------------------------------------------------

    x["impact_auction_open"] = (
        Y_open_auction
        * x["sigma_daily"]
        * np.sqrt(p_open_auc)
    )

    x["impact_auction_close"] = (
        Y_close_auction
        * x["sigma_daily"]
        * np.sqrt(p_close_auc)
    )

    # No auction impact if auction fraction of our order is zero.
    x.loc[x["q_open_auc"] == 0, "impact_auction_open"] = 0.0
    x.loc[x["q_close_auc"] == 0, "impact_auction_close"] = 0.0

    # ------------------------------------------------------------
    # 10. Auction dislocation allowance
    # ------------------------------------------------------------

    d_open = auction_dislocation_open_bps / 1e4
    d_close = auction_dislocation_close_bps / 1e4

    x["auction_cost_open"] = ao * d_open
    x["auction_cost_close"] = ac * d_close

    # ------------------------------------------------------------
    # 11. Total one-way cost fractions
    # ------------------------------------------------------------

    x["cost_frac_open"] = (
        x["spread_cost_open"]
        + x["impact_cont_open"]
        + x["auction_cost_open"]
        + x["impact_auction_open"]
    )

    x["cost_frac_close"] = (
        x["spread_cost_close"]
        + x["impact_cont_close"]
        + x["auction_cost_close"]
        + x["impact_auction_close"]
    )

    # ------------------------------------------------------------
    # 12. Total dollar cost for the stock-day
    #
    # Both legs have approximately the same NAV notional:
    # |weight| * NAV.
    # ------------------------------------------------------------

    x["cost_dollars"] = (
        x["notional"]
        * (x["cost_frac_open"] + x["cost_frac_close"])
    )

    # Cost as bps of total portfolio NAV.
    x["cost_bps_nav"] = (
        1e4 * x["cost_dollars"] / nav
    )

    return x
```

###### How I would run the scenarios

**1\. Extremely favorable continuous-market lower bound**

```
cost_lb = transaction_cost_model(
    df,
    Y_cont=0.0,
    auction_frac_open=0.0,
    auction_frac_close=0.0,
)
```

This essentially asks: _does the strategy survive quoted spread alone?_

**2\. Continuous-market central case**

```
cost_cont = transaction_cost_model(
    df,
    Y_cont=0.75,
    auction_frac_open=0.0,
    auction_frac_close=0.0,
)
```

**3\. Pure auction execution**

```
cost_auc = transaction_cost_model(
    df,
    Y_cont=0.75,
    auction_frac_open=1.0,
    auction_frac_close=1.0,
    auction_volume_frac_open=0.015,
    auction_volume_frac_close=0.10,
    auction_dislocation_open_bps=2.0,
    auction_dislocation_close_bps=1.0,
)
```

I would **not treat those 2 bp/1 bp auction-dislocation values as estimates**. Replace them with the values obtained from your minute-quote analysis.

---

#### Bottom line

Given what you have already found, I would structure the robustness section around one very clear proposition:

> **Can anything plausibly make a daily open-to-close strategy with roughly 2× gross turnover and less than 0.2 bp of one-way break-even cost economically implementable?**

On the information given, the answer looks like **no**.

The most important remaining analyses are therefore:

1. **Replace leader transaction-price returns with consolidated quote-midpoint returns.**
2. **Use the one-minute data to quantify leader quote staleness and leader→follower lead/lag.**
3. **Replace the open and close with interior intraday midpoint benchmarks.**
4. **Run the strategy excluding the first and last 5/15/30 minutes.**
5. **Run conservative spread \+ square-root impact costs.**
6. **Show that 1996–2006 concentration is not simply a pre-decimalization/auction/microstructure phenomenon.**
7. **Label everything using 1996–2024 as robustness or pseudo-OOS, not OOS.**
8. If possible, **lock the entire specification and evaluate 2025–26 data as the only genuine temporal OOS test.**

One particularly strong result would be a table showing **CRSP transaction-price return → quote-midpoint return → interior-day midpoint return → net return after conservative costs**. If the return dies along that sequence, you have an unusually clean robustness argument for “do not implement.” Conversely, if it survives all four, _then_ the strategy deserves a serious challenge rather than dismissal.
