# Part 4: ChatGPT interactions

Two interactions, run on GPT-6.1 (sol, medium) on 2026-09-30, after the Part 4 design had
been pre-registered and the results computed. ChatGPT was therefore used as an
independent review of the design rather than to generate it. Each section gives the
exact prompt, a summary of the output, the critical evaluation, and what was adopted.
The full outputs are reproduced at the end of this file.

---

## Interaction 1: robustness design

### Prompt

> I am backtesting a daily-rebalanced long/short equity strategy on US stocks, 1996–2024,
> using CRSP daily data. The universe is about 1,750 "follower" stocks a month, grouped
> into 48 Fama-French industries. The signal is the previous day's return of each
> industry's largest stock (the "leader"), decomposed into a market-and-industry
> component and a leader-specific component. Every follower in an industry gets the
> same signal value. I trade at the close and hold one day.
>
> List every way this backtest could show a profit that would not survive in real
> trading, or that is not the effect I think I am measuring. For each one, say how I
> would detect it in the data I have (CRSP daily prices, bid and ask, opening price,
> volume, delisting returns, Fama-French daily factors) and what the fix is. Be
> specific to this design rather than generic.

### Summary

- About 70 failure modes in six groups (information timing, CRSP price artifacts,
  execution and shorting, delistings and accounting, alternative explanations,
  statistical inference), each with a detection method and a fix, ending with a
  seven-step priority order.
- A fixed timing convention: with the leader's return on t−1, enter at the close of t,
  exit at the close of t+1, and earn the follower's return on t+1.
- Stale follower prices and bid-ask bounce as the main mechanical explanations, to be
  checked by comparing close and midpoint returns and by splitting the holding-day
  return into overnight and daytime parts, with open entry as a test.
- Because the signal is shared within an industry, the strategy is industry allocation:
  1,750 followers are not 1,750 independent forecasts, and industry neutrality would
  remove the payoff.
- Market continuation, industry momentum and the follower being included in the
  industry index used to decompose the leader can all masquerade as leader information.

### Critical evaluation

**Where it was right, and matched what Part 4 found in the data**

- **Industry-level signal.** It said an industry-neutral book removes the payoff of a
  signal shared within an industry, and treated breadth as industry-days. That is the
  design fact Part 4's construction is built on.
- **Market-beta timing.** Its "generic market continuation" item is what removes the
  conditional signal in Part 4: beta neutrality alone takes it from t = 3.0 to t = −0.6.
- **Industry momentum and self-inclusion.** Its point that the industry factor contains
  the follower being predicted is the mechanism behind Part 4's finding that
  `common_lag` is 0.63 correlated with the followers' own previous-day return.
- **Calendar lag.** "Lag on the common market calendar" rather than the previous
  available observation is the defect the code audit found in the team's
  `lag_leader_components` (758 follower-days).
- **Bounce and staleness.** The close-versus-midpoint comparison and the
  overnight/daytime split are exactly what show that about 80% of the close-to-close
  reversal benchmark is bounce (10.1% a year on closes against 1.9% on midpoints) and that
  most of its close-to-close return arrives overnight (7.5 of 10.1 points a year; 15 of 18
  in 1996–2006).

**Where it was wrong or would have misled**

- **Its timing would have hidden the effect.** Our prompt said "I trade at the close".
  ChatGPT took that literally, correctly ruled out trading at the same close that produced
  the signal, and moved the trade to the close of t, a full session after the signal. That
  is the two-session gap that hid the effect in Part 3: Part 1 shows the effect is gone
  after one session, and in Part 4 every lead-lag book earns roughly nothing under this
  timing (`leader_ret_lag` t = 0.02). It did suggest testing entry at the next open and
  splitting the return into overnight and daytime parts, but it did not say that the next
  open is the earliest feasible entry and should be the headline. Part of the blame lies
  with our prompt: by stating the execution time, we steered it toward the wrong default.
- **Too long and partly generic.** Many items do not apply here, were already handled
  by Part 1 (delisting repair, negative prices, split adjustment, point-in-time
  industry codes), or cannot be tested with this data (borrow availability, recalls,
  auction fills, margin ledgers, the Fama-French data vintage, which matters little when
  the factors are used only for attribution). The ten or so items that decide the result
  are buried among the rest; the closing priority list helps.
- **No numbers on costs.** It cannot know the turnover, so it could only say costs might
  be decisive, not that the strategy is an order of magnitude below breakeven.

**Adopted.** Part 4 had already used open-to-close as its headline, the calendar lag and
the midpoint diagnostic, so the answer served as an independent check of those choices.
Its default close-to-close-of-t+1 timing was rejected for the reason above. Three checks
it suggests are not in Part 4 and are passed to Part 5: long and short legs reported
separately, P&L concentration by industry and leader, and leave-one-industry-out tests.

---

## Interaction 2: portfolio construction and risk model

### Prompt

> I have a daily cross-sectional signal that takes one value per industry (48
> industries) and is shared by every stock in that industry. I want a long/short book
> that is dollar-neutral and market-beta-neutral, and that is also neutral to each
> industry's average lagged stock return, so I isolate information that is not just
> industry momentum. My only optimizer is scipy.
>
> 1. Propose a construction, with the exact formula for the weights.
> 2. Propose a risk model suitable for this book and explain whether a diagonal
>    (stock-specific) risk model is adequate.
> 3. Say how to set the position size: gross exposure versus a volatility target.
> 4. Give Python for the weights given a DataFrame with columns date, industry,
>    signal, beta, own_lag.

### Summary

- A covariance-weighted projection of the signal on the constraints (a constant, beta,
  and one industry-average own-lag variable):
  q = Σ⁻¹s − Σ⁻¹A(AᵀΣ⁻¹A)⁺AᵀΣ⁻¹s, scaled to the desired gross exposure.
- Why 48 separate own-lag constraints, one per industry, or full industry-dollar
  neutrality would wipe out a signal shared within an industry.
- A diagonal risk model is inadequate because net industry positions carry common
  industry risk; it recommends Σ = BFBᵀ + D with market, industry and style factors,
  shrunk and estimated on past data.
- A volatility target with a gross-exposure cap, noting the target is missed when the
  cap binds, and holding zero when the projection removes nearly all of the signal.
- Python implementing the projection with whitening and SVD rank handling.

### Critical evaluation

- **Agrees with Part 4 on the essentials.** One industry-level own-lag constraint rather
  than 48, a closed-form projection rather than an optimizer, and a diagonal model being
  inadequate.
- **But it builds a different book, with a flaw it had itself warned about.** Its
  projection runs at the stock level on the raw signal, with each stock's own beta. With
  Σ = I, weights therefore vary within an industry and each industry's net weight grows with
  its number of followers, which is the "follower count accidentally determines industry
  exposure" problem it listed in its first answer. Part 4 instead ranks the industries,
  projects with each industry weighted equally, and splits each industry's weight equally
  across its followers.
- **Its risk-model advice points to a real gap in Part 4.** It recommends a factor model
  (Σ = BFBᵀ + D) and calls a diagonal model inadequate. Part 4 has no full factor risk
  model; it sizes books by gross exposure and realized volatility instead. That does not change the
  verdict, since the books either lose money or earn far less than their costs, but it
  belongs in the limitations.
- **One warning does not apply here.** It advises holding zero when the projection removes
  nearly all of the signal, while Part 4 always rescales to gross 1. On the real data the
  final weights of the pre-registered books keep a correlation of 0.61 (`common_lag`) to
  0.88 (`shock_lag`) with the signal, so the books are not trading a near-empty residual.
- **It anticipated the volatility-target result.** Part 4's headline hits its 3× cap and
  realizes 6.7% volatility rather than the 10% target.
- **Missed.** It did not mention that ordinary least-squares betas are biased down for
  thinly traded stocks, which is why Part 4 uses Dimson betas, and it projects the raw
  signal rather than ranks, which is more sensitive to outliers.
- **Convention.** Its gross exposure of 2 means 100% long and 100% short; Part 4 reports
  books at gross 1 (50% long, 50% short), so its returns would be twice Part 4's for the
  same book.
- **Code.** As copied, the code block lost its line breaks and the start of the function
  (signature, imports, date loop), so it cannot be run as is, and its statement that it
  verified the code cannot be checked from the transcript. Part 4 did not rely on it.

**Adopted.** No change to the pre-registered construction. The missing factor risk model
is added to the Part 4 limitations.

---

## Reflection (report section 3d)

Used as an independent review after the design was fixed, ChatGPT was most useful as a
checklist. Its first answer worked out, without being told, the three issues that decide
Part 4: that a signal shared within an industry makes the strategy industry allocation (so
industry neutrality removes it), market-beta timing, and industry momentum leaking into
the "common" component through the industry index. It also flagged the calendar-lag defect
the code audit found. Its weakest point was timing: following our prompt's "I trade at the
close", it moved the trade a full session after the signal, the same two-session gap that
hid the effect in Part 3, and offered the next-open entry only as a test. That is partly a
lesson about our prompt, which should have asked for the earliest feasible execution rather
than stating one. It was long and partly generic, and it could not quantify costs. Its construction
answer shared Part 4's constraints but weighted industries by follower count, a problem it
had itself flagged, and its risk-model advice pointed to one genuine gap, the lack of a
full factor risk model. The pattern matches
HW1 and HW2: it is strong at listing what could go wrong and weak at judging which items
matter for a specific dataset.

---

## Full outputs

### Interaction 1 output

The biggest risks in this design are **earning returns before your entry, trading stale follower prices, mistaking common-factor continuation for leader information, and treating 1,750 followers as independent signals**. Daily rebalancing also makes modest execution costs potentially decisive.

Below is a comprehensive audit of the material failure modes. Some can be diagnosed with your daily data; others can only be stress-tested. Historical borrow availability, auction fills, intraday quote timing, and the cause of a news shock cannot be established from these fields alone.

First, fix the time convention. Let \(r_{i,t}\) be the return from close \(t-1\) to close \(t\). **If you use the leader's return \(r_{L,t-1}\), enter at close \(t\), and exit at close \(t+1\), your earned follower return is \(r_{i,t+1}\).** The follower's \(r_{i,t}\) has already happened.

**1. Information timing and selection**

| Failure mode specific to your design | How to detect it | Fix |
|---|---|---|
| **One-day return alignment error.** You use yesterday's leader return but credit today's follower close-to-close return despite entering today at the close. | For several trades, print signal start/end dates, order date, entry price, exit price, and the CRSP return date. Reconcile individual dollar profits manually. | Use the timeline above. Keep signal date, execution date, and return date as separate fields. |
| **Using the completed close to trade at that same close.** This can enter through leader returns, factor returns, volatility scaling, or follower weights—even if the headline signal is lagged. | Record the latest observation used by *every* input. Compare against the order submission time, not merely the calendar date. | Freeze orders using information available before submission. Completed daily inputs generally require execution at a later opportunity. |
| **Fama–French factors treated as available immediately.** Yesterday's economic factor return and yesterday's published factor value are different objects. | Identify which factor files enter live signal construction and their assumed availability. Recompute with additional lags or internally constructed factors. Daily factor files do not establish historical publication times. | Construct contemporaneously available market/industry proxies yourself, or apply a defensible publication delay. Ex-post factors remain usable for attribution. |
| **Full-sample or forward-looking decomposition.** Leader betas, means, residual volatility, winsorization thresholds, or signal coefficients use future observations. | Recreate every day using a dataset truncated at the decision date. Compare signals with the original series. | Fit rolling or expanding models using past data only. A conservative specification estimates coefficients through the day before the return being decomposed. |
| **Month-end leader selection applied backward within the month.** The eventual largest stock can become leader partly because it rose during that month. | Compare the recorded leader with ranks at the previous month-end. Inspect leader changes and profit around them. | Select leaders using information available before the new month; hold that definition fixed for the stated selection period. |
| **Future-dependent follower eligibility.** Requiring complete monthly returns, positive volume throughout the month, survival, or a future exit quote selects favorable outcomes. | Compare your universe with eligibility reconstructed at each prior month-end. Examine excluded names' subsequent returns and delistings. | Set eligibility prospectively. Handle subsequent missing observations and trading failures inside the position ledger. |
| **Survivorship or endpoint selection.** Your historical 1,750-stock universe comes from stocks surviving to 2024 or a current list. | Examine entrants, exits, IPOs, and delisted stocks over time. Reconstruct membership from historical CRSP records. | Use point-in-time membership, retaining failed companies and securities that disappear. |
| **Historical industry assignment overwritten by current SIC.** Followers and leaders acquire relationships that did not exist then. | Compare assignments against effective-dated SIC histories; inspect reclassifications and mergers. Prices alone cannot validate this. | Use historical classification records. Distinguish "FF48 SIC mapping" from reproducing French's actual annual portfolio memberships. |
| **"Largest stock" is misdefined.** Share price substitutes for market capitalization, shares have incorrect units, or one share class substitutes for the company. | Audit leader ranks using \(|PRC|\times SHROUT\), units, and company identifiers. Compare security-level and company-level ranks. | Define whether leadership is by security or company capitalization. For company leadership, aggregate eligible classes and specify which return represents the leader. You need historical shares and identifiers. |
| **Leader contamination among followers.** Another share class of the same company, or the leader itself, is traded as a follower. | Match leader and follower company identifiers; separately report affiliated-class profits. | Exclude the leader and, for an intercompany transmission claim, its other share classes. |
| **Identity/join errors.** Ticker reuse, many-to-many name-history joins, or merger links create duplicate trades or artificial continuity. | Require unique security–date rows; check row counts before and after joins and inspect abrupt identity changes. | Use permanent security identifiers and effective-dated joins; handle corporate events explicitly. |
| **"Previous day" means previous available observation.** A suspended follower's next return spans several sessions but is labeled one day. | Measure trading-calendar gaps between observations and entry/exit records. | Lag on the common market calendar. Preserve gaps and actual elapsed holding periods. |

French's published 48-industry portfolios use June assignments held from July through the following June, with a specified Compustat/CRSP SIC hierarchy. A monthly CRSP-SIC classification can be legitimate, but it is a different construction. [Detail for 48 Industry Portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_48_ind_port.html?utm_source=chatgpt.com)

**2. CRSP prices and apparent lead–lag profits**

CRSP's legacy `PRC` can represent a trade price or a bid/ask average. Also, `BIDLO` and `ASKHI` are conditional bid/low and ask/high fields; they must not automatically be treated as executable quotes. Separate `BID` and `ASK` fields exist. [9.3](https://support.sas.com/documentation/cdl/en/etsug/63939/HTML/default/etsug_sasecrsp_sect017.htm?utm_source=chatgpt.com)

| Failure mode | How to detect it | Fix |
|---|---|---|
| **Negative `PRC` mishandled.** In legacy data the negative sign denotes a quote-based price, rather than a negative economic price. Sign changes can create enormous false returns. | Flag negative prices; inspect every large return against price signs and CRSP returns. | Use absolute price magnitudes for valuation while retaining the quote-price indicator. Prefer correctly interpreted CRSP return fields for total returns. |
| **Stale follower closes manufacture delayed reaction.** The liquid leader incorporates news while the follower's recorded last trade predates it; the next trade appears to "follow." | Stratify profits by zero volume, repeated prices, zero-return runs, negative-price observations, lagged dollar volume, and spreads. Compare trade-close and genuine midpoint-return results. | Require prospectively adequate liquidity; model executable entry prices. Obtain timestamped quotes/trades for a definitive synchronization test. Positive daily volume does **not** prove a fresh closing trade. |
| **Bid–ask bounce masquerades as predictability.** Industry-wide buying/selling can affect which side supplies the follower close. | Compare close-return predictions with midpoint-return predictions on the same observations; measure close-minus-midpoint changes and next-day reversals. | Treat midpoints as a diagnostic valuation series, then price actual buys and sells appropriately. Do not claim midpoint profitability is executable profitability. |
| **Alternating trade and quote marks create returns without equivalent trades.** | Split profits into trade→trade, trade→quote, quote→trade, and quote→quote intervals. | Keep valuation and execution separate. A quote-derived mark is not evidence of an obtainable fill. |
| **Daily high/low fields used as spreads.** Your cost model or midpoint series actually uses the intraday range. | Inspect field names and definitions; compare candidate "spreads" with daily ranges and available true quotes. | Use genuine bid/ask fields. If unavailable, report cost scenarios rather than interpreting high/low ranges as quoted spreads. |
| **Bad, stale, or noncomparable quotes.** Crossed quotes, different adjustment bases, or noncontemporaneous observations distort midpoints and costs. | Flag nonpositive/crossed quotes, extreme relative spreads, repeated quotes, and large close–midpoint discrepancies; report coverage by exchange and year. | Apply documented quote-validity rules. Keep a separate matched-sample comparison so quote coverage does not silently change the universe. |
| **A split or distribution creates the leader signal.** Raw price changes are mistaken for economic returns, broadcasting a false shock to an entire industry. | Inspect large leader signals; compare raw-price returns with CRSP total and ex-distribution returns, where available. Check opening/closing adjustment consistency. | Use corporate-action-aware returns. Price, shares, volume, quotes, and opening prices must have consistent adjustment bases. |
| **Dividend price drops generate short profits.** The short earns the ex-dividend price decline without owing the dividend. | Compare price-return and total-return P&L, especially around large price drops unexplained by market moves. Exact diagnosis needs return/distribution fields. | Include dividend receipts on longs and payments on shorts. Do not add distributions again if using total returns. |
| **Signal and traded P&L use incompatible return definitions.** A dividend-inclusive leader return is interpreted as price information, or price-only follower returns are compared with total-return factors. | Re-run leader signals using price/ex-distribution and total returns; inspect distribution-sensitive dates. | State the economic return definition for each component; separate cash distributions from price news when testing transmission. |
| **Missing codes, percentage units, or exceptional observations become tradable signals.** One malformed leader observation affects every follower in its industry. | Check ranges and vendor missing-value codes; confirm FF percentage units versus CRSP decimal returns. Trace the largest industry-day P&L contributions. | Parse missing values explicitly, validate units, and correct identifiable errors. Use any clipping threshold prospectively; do not delete genuine crashes merely because they are extreme. |

Nonsynchronous observations can generate return autocorrelation and cross-autocorrelation. Thus, your largest-to-smaller-stock structure is particularly exposed to this alternative explanation; it is a hypothesis to test, not proof that all lead–lag profits are artificial. [NBER](https://www.nber.org/papers/w2960?utm_source=chatgpt.com)

**3. Execution, costs, and the short book**

| Failure mode | How to detect or bound it with your data | Fix |
|---|---|---|
| **CRSP's close is assumed to be your closing-auction fill.** A last reported trade can be a different execution opportunity. | Identify reliance on quote-based/no-volume closes; compare close-entry results with delayed open-entry results. Daily data cannot establish auction accessibility or depth. | Use the close as an explicitly conditional benchmark; validate auction fills with auction/intraday data or actual executions. |
| **Profitable adjustment happens before your delayed fill.** The follower catches up overnight or immediately at the opening. | For the correctly aligned holding day, split adjusted price returns into close→open and open→close. Also test next-open entry through the following close. | Report exactly which interval earns the profit and whether your order could precede it. An open-entry strategy cannot collect the preceding gap. |
| **Free crossing of the spread.** Frequent industry sign changes cause followers to trade in the same direction together. | Calculate costs on actual net dollar trades using valid relative spreads. Stress multiple spread assumptions and report gross versus net results. | Charge buys at ask/sells at bid under a marketable-order model, or use a defensible auction model. Spread crossing is not automatically the right model for genuine auction fills. |
| **Passive fills assumed whenever favorable.** Trading at the midpoint or bid/ask is treated as guaranteed. | Compare optimistic midpoint/passive P&L with aggressive-fill scenarios. Daily prices cannot reveal fill probability or adverse selection. | Do not award guaranteed spread capture. Validate a passive strategy with order-level data. |
| **Commissions, fees, and trading frictions omitted.** Low gross daily returns can disappear even without much impact. | Report net traded dollars, ticket counts, and break-even total cost per traded dollar, by period. | Apply historically relevant schedules or transparent cost ranges, including minimum-ticket effects where material. |
| **Turnover calculated from yesterday's targets rather than today's actual holdings.** Returns, cash flows, and membership changes alter the portfolio before rebalancing. | Reconcile positions immediately before each trade with target positions. Include monthly additions/removals and sign flips. | Trade from drifted holdings. Net retained positions rather than automatically closing/reopening them; charge a long-to-short flip for the full change in shares. |
| **Unlimited liquidity and capacity.** Simultaneous follower orders overwhelm small stocks even if aggregate capital appears modest. | For several AUM levels, compute order dollars/lagged average dollar volume, largest participation, and contribution from constrained names. Inspect industry-wide order concentration. | Cap positions and participation using prior information; model nonlinear impact. Total daily volume is only a loose capacity bound for close execution. |
| **Same-day volume determines achievable size.** Realized full-day volume or the best eventual liquidity day enters sizing. | Recompute sizes using trailing volume available before orders. | Use lagged liquidity forecasts; model partial fills without knowledge of subsequent volume. |
| **Free and universal stock borrowing.** The short followers may be precisely the small/distressed names with expensive or unavailable borrow. | Separate long and short contributions; stratify shorts by lagged price, volume, spreads, and distress proxies. Apply borrow-fee and unavailable-name scenarios. | Obtain historical lending data for validation. Otherwise report conditional net returns and transparent borrow scenarios; liquidity is not proof of borrow availability. |
| **Recalls, buy-ins, short-sale restrictions, or emergency bans ignored.** | Identify short profits concentrated in distress/crisis episodes and stress forced covering or short exclusions. Daily CRSP cannot identify actual locates or restricted execution opportunities. | Use historical restriction/borrow records and operational rules. Do not infer unrestricted short execution from a recorded price. |
| **Halts and disappearing stocks conveniently liquidate at their last mark.** | Inspect positions with missing exit prices, zero volume, large observation gaps, and delistings. Check whether they vanish from P&L. | Keep positions open until a feasible exit or terminal settlement; model capital tied up and delayed liquidation. |
| **Cash, financing, collateral, and margin create fictitious extra return.** Short proceeds are treated as freely investable cash, or financing is ignored. | Reconcile equity, gross exposure, long funding, short collateral, cash balances, interest, and losses each day. | Use an explicit broker cash/margin ledger. RF alone is not the actual funding or short-rebate rate. Include liquidation when equity/margin constraints bind. |
| **Weekend and holiday costs understated.** "One day" actually includes several calendar days. | Compare funding/borrow accrual with calendar time between entry and exit. | Accrue applicable financing and borrow costs over elapsed calendar days. |

**4. Delistings and portfolio accounting**

| Failure mode | Detection | Fix |
|---|---|---|
| **Delisting losses omitted, or short gains overstated.** Terminal economic returns disappear with the security. | Match every held disappearing security to delisting information; reconcile its last ordinary return, terminal return, and position. | Include terminal economics on both sides. For legacy, nonoverlapping ordinary and delisting return segments compound as \((1+RET)(1+DLRET)-1\); they are not simply added. |
| **Delisting return counted twice or booked at an impossible cash date.** | Audit your CRSP format and return coverage; inspect every terminal position and settlement date. | Check whether terminal effects are already incorporated before applying another adjustment. Separate economic valuation from actual receipt/payment of cash. |
| **Missing delisting returns replaced by zero, or unfavorable terminal rows discarded.** | Tabulate missing terminal returns and show P&L under alternative terminal-payoff assumptions, separately for longs/shorts. | Use documented, event-sensitive assumptions and sensitivity ranges. Exact recovery requires additional terminal-event data. |
| **Held stocks with missing returns are dropped and weights renormalized after the fact.** | Compare the position list before the holding period with the names included in its return denominator. | Preserve all positions and cash. A missing observation is a valuation/execution problem, not permission to redistribute capital retrospectively. |
| **Returns use ending weights or realized winners receive more weight.** | Reconstruct profits from entry shares, actual prices, and distributions; compare with the weighted-return calculation. | Use beginning-of-period exposures. Lag capitalization and volatility inputs appropriately. |
| **Return denominator exaggerates performance.** \(R_L-R_S\) is advertised as the return on a 100% gross portfolio. | Report long dollars, short dollars, gross/net exposure, equity, and P&L together. | For 50% long and 50% short relative to equity, contribution is \(0.5R_L-0.5R_S\). A 100% long/100% short book has 200% gross exposure. State the convention. |
| **Short returns compounded incorrectly.** Inverting a stock wealth index is substituted for daily short P&L; negative equity or >100% short losses are ignored. | Reconcile individual shorts in large rallies and verify portfolio equity never receives impossible resets. | Use signed shares and dollar P&L. Apply actual collateral and solvency constraints. |
| **Daily arithmetic profits, overlapping sleeves, or leverage resets misstate investable wealth.** | Compound the daily equity ledger and compare with reported annual returns; check that capital is not reused across overlapping holdings. | Report a funded, self-financing wealth path with costs and cash flows. Geometric wealth growth differs from annualized mean return. |

Factor-file vintage also deserves an audit: French switched the production source from legacy FIZ to CIZ beginning with the January 2025 release. Freeze and identify your CRSP and factor vintages rather than silently mixing revised histories. [Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html?utm_source=chatgpt.com)

**5. Profits that may be real but measure something else**

Suppose you write the leader's excess return as

\[
r^e_{L,g,d}
=\widehat\alpha_{g,d}
+\widehat\beta_{M,g,d}M_d
+\widehat\beta_{I,g,d}I_{g,d}
+\widehat\varepsilon_{L,g,d}.
\]

The residual is **specific to this statistical model**, not established leader-specific news. Orthogonality in an estimation sample does not establish causal independence or remove all future factor exposure.

| Alternative explanation | Detection using your data | Fix or appropriate interpretation |
|---|---|---|
| **Generic market continuation or market timing.** Leaders proxy for yesterday's market move, while long and short industries have different betas. | Compare with a market-only signal; estimate rolling portfolio market exposure and regress realized strategy returns on contemporaneous FF factors. Test lagged market predictors of followers. | Report the market-driven component separately. Construct ex-ante beta-controlled portfolios and assess the remaining incremental leader signal. |
| **Industry continuation rather than leader transmission.** | Compare against lagged industry returns constructed without the leader; include both industry and leader residual predictors. | Require incremental out-of-sample prediction beyond the industry benchmark. Otherwise describe industry timing. |
| **The industry factor contains the leader being decomposed.** Its large weight mechanically determines how much of its own return is assigned to the common component. | Measure leader weight in the industry portfolio; rebuild the factor excluding the leader company and compare decompositions. | Use a leave-leader-out industry benchmark for identification. Report sensitivity to equal/value weighting. Inclusion is not automatically look-ahead, but it changes the meaning of the decomposition. |
| **The factor contains the follower being predicted.** Follower own-return information enters the purported leader signal through residualization. | For each follower, remove that follower from the industry factor; compare results. Alternatively use a disjoint reference set of stocks. | Use leave-follower-out or disjoint benchmark portfolios where feasible; show that self-inclusion does not explain results. |
| **Industry weights/membership use contemporaneous or future information.** | Rebuild industry returns with beginning-of-period weights and prospective membership; compare signals. | Lag portfolio weights and define membership prospectively. |
| **Omitted common factors are called leader-specific.** Size, value, profitability, investment, momentum, liquidity, or subindustry news remain in the residual. | Add available FF factors and momentum if separately available; add returns of other large peers and CRSP-built liquidity/size portfolios. Compare residuals and incremental forecasts. | Label the residual conditional on the model used. Test richer models prospectively; recognize that your data cannot exhaust all economic shocks. |
| **Market and industry are collinear; attribution is unstable.** | Track condition numbers, coefficient instability, residual variance, and component correlations. Compare joint regression with alternative sequential decompositions. | Define the projection convention explicitly. Orthogonalizing industry to market gives an ordered interpretation, not a uniquely true allocation. |
| **Leader coefficients change through time.** Stale betas leave common shocks in the "specific" term, especially in crises or after business changes. | Check rolling residual–factor correlations, breakpoints, and forecast performance around leader switches. | Use past-only rolling/shrunk estimates and sensible minimum-history requirements; report window sensitivity. |
| **Follower own momentum/reversal explains the forecast.** A shared shock links yesterday's leader return to follower returns already observed. | Control for follower returns over the signal day and intervening day, prior momentum, lagged liquidity, and volatility. Compare with a follower-only predictor. | Demonstrate incremental prediction beyond what followers already reveal before entry. Keep these controls available at decision time. |
| **A large-stock/liquidity effect is mistaken for an industry relationship.** | Replace the leader with size/liquidity-matched stocks from other industries; compare the largest, second-largest, and a large-peer portfolio. | Require stronger prediction within the assigned industry than from comparable outsiders and broader large-stock benchmarks. |
| **Common news timing creates apparent leadership without transmission.** Both companies react to the same event at different times; broad SIC groups hide subindustry shocks. | Test industry specificity, peer-return controls, lead/reverse-lead patterns, and opening versus daytime responses. Inspect extreme event dates. | Describe predictive ordering unless news timestamps or another identification design supports causal transmission. Daily returns cannot establish who learned from whom. |
| **Mechanical same-industry positions create an industry allocation strategy.** Because every follower gets the same signal, trading more followers does not add distinct stock-selection information. | Aggregate weights and P&L to industry-day observations. Compare with a strategy trading follower industry baskets directly. | Report industry-level exposures and breadth. Describe the strategy as industry allocation based on leaders, with stock-level implementation. |
| **Follower count accidentally determines industry exposure.** An industry with 150 names receives much more capital than one with 10 under equal stock weights. | Compare industry weights with follower counts; run equal-industry, equal-stock, and capitalization/liquidity-weighted variants. | Choose and state the intended exposure rule. Separate signal quality from the incidental number of listed followers. |
| **Long/short or residual signals are mistaken for industry neutrality.** | Inspect each industry's net dollar and beta exposure. | Do not claim industry neutrality. With identical signals and equal factor sensitivities within an industry, imposing zero net industry exposure removes the common-signal payoff; any remaining payoff comes from differing weights/exposures. |
| **Stock-level ranking ties produce unintended selection.** Every follower in an industry has the same score, but stock quantiles cut through ties using ticker, row order, or another hidden rule. | Shuffle row order and re-sort; examine industries split across long/short cutoffs. | Rank industries or use explicit, deterministic allocation across tied stocks. Report any secondary selection variable. |
| **Separate component portfolios give misleading attribution.** Ranking, volatility scaling, and constraints are nonlinear, so common-component and residual-component portfolio profits need not add to total-signal profit. | Compare score additivity with actual position additivity; fit both predictors jointly and compare nested forecasts. | Separate predictive attribution from implementable portfolio comparisons. Do not present nonlinear portfolio P&Ls as an additive decomposition. |
| **Contemporaneous controls either manufacture or remove the result.** Realized holding-period factors are used to form positions, or industry returns containing followers absorb the outcome being explained. | Distinguish executable P&L, ex-post risk regression, and predictive regression; audit dates and factor constituents. | Use future returns only as outcomes or clearly labeled attribution controls. Use independent industry benchmarks where outcome overlap matters. |

**6. Statistical confidence and historical relevance**

| Failure mode | Detection | Fix |
|---|---|---|
| **Pseudoreplication: 1,750 identical industry signals appear to be 1,750 independent forecasts.** | Compare stock-level standard errors with industry-day aggregation; examine residual correlation within industries and across the same date. | For prediction panels, account for dependence across dates and industries using appropriate clustering/block inference. For strategy returns, use the daily portfolio series with serial-dependence-aware inference. |
| **Generated residual uncertainty is ignored.** Estimated leader betas are treated as known observations. | Re-estimate the decomposition inside resampling and compare confidence intervals. | Resample time blocks and rerun estimation, signal construction, and portfolio formation—not just the final residual regression. |
| **Multiple testing selects the profitable version.** Two components, signs, windows, filters, weighting rules, industries, and start dates create many opportunities. | Keep an inventory of all tried specifications; compare the selected version with untouched periods. | Use chronological training/validation/test partitions and nested walk-forward selection; adjust inference for the actual search. |
| **Random train/test splits leak regimes and repeated signals.** Followers sharing the same industry-day shock land in both sets. | Inspect whether the same dates or industry-day signals occur in training and test sets; check overlapping labels/windows. | Split chronologically, with an appropriate gap for overlapping outcomes. Fit all transformations within training windows. |
| **A few leaders, industries, or days explain the result.** | Report P&L by leader, industry, year, long/short side, and the largest contributing days. Run leave-one-industry/leader-out tests and contiguous-block exclusions. | Report concentration and robust uncertainty. Do not retrospectively delete economically valid events to obtain a preferred result. |
| **Older market structure drives the pooled average.** A profitable 1990s lag disappears in modern liquid trading. | Show gross/net performance, spreads, staleness, and turnover by era; test later periods under a frozen specification. | Base a current tradability claim on recent, genuinely held-out results and relevant costs. Historical existence is a separate claim. |
| **Calendar seasonality or crisis exposure supplies the premium.** Industry signals coincide with month-end moves, holiday intervals, or crisis risk. | Break out month-end, weekday, holiday-gap, and high-volatility results; examine factor exposures and downside conditional on market states. | Report conditional performance and risk; test incremental predictive value beyond calendar/state controls available before trading. |
| **Sharpe and t-statistics ignore serial dependence, tails, or changing leverage.** | Examine autocorrelation, drawdowns, skew, tail losses, gross exposure, and rolling performance; compare naive and block/HAC inference. | Use net funded returns, serial-dependence-aware uncertainty, and tail/drawdown reporting. A high annualized daily Sharpe alone is insufficient. |
| **Placebos accidentally preserve or destroy the wrong structure.** Stock-level randomization breaks the industry shock dependence and makes your result look exceptional. | Compare industry-label permutations, matched outsider leaders, reverse direction, and blockwise date shifts. | Preserve the dependence relevant to each null. Treat placebos as complementary tests, not proof of causality. |

I would prioritize these checks in this order:

1. **Manually reconcile the signal/entry/exit timeline and several trades**, including a short, a dividend date, a gap, and a delisting.
2. **Rebuild all signals and monthly selections prospectively**, with past-only decomposition estimates.
3. **Compare close and genuine-midpoint results on an identical sample**, then stratify by staleness, spreads, and liquidity.
4. **Split the earned holding-day return into overnight and daytime components**, and test delayed open execution.
5. **Calculate net P&L from drifted holdings**, including realistic spread/auction assumptions, capacity, borrow, and financing scenarios.
6. **Test incremental leader-residual prediction against market, leave-leader-out industry, follower-own-return, and matched-outsider benchmarks.**
7. **Evaluate at industry-day and daily-portfolio levels**, with chronological out-of-sample testing and dependence-aware inference.

The defensible outcome is three separate findings: **predictive ordering in recorded daily returns**, **incremental prediction beyond available common and follower information**, and **profit under feasible execution and funding**. Passing one does not establish the other two.

### Interaction 2 output

Use a covariance-weighted projection of the signal onto your neutrality constraints. But first distinguish one industry-momentum exposure from 48 separate constraints: the latter generally destroy your industry-level signal.

1. Construction and exact weights

For a given date, define:
\[ s_i=s_{g(i)},\qquad m_g=\frac{1}{n_g}\sum_{i:g(i)=g}\mathrm{own\_lag}_i, \qquad m_i=m_{g(i)}. \]
Here, \(m_i\) is the industry-average lagged return broadcast back to each stock. I assume an equal-weighted average over your eligible universe.
The useful neutrality conditions are:
\[ \mathbf1^\top w=0,\qquad \beta^\top w=0,\qquad m^\top w=0. \]
The last condition neutralizes one cross-sectional industry-momentum variable. It does not require each industry to have zero net investment.
Set \(A=[\mathbf1,\beta,m]\), and let \(\Sigma\) be a positive-definite daily stock-return covariance estimate. Solve
\[ \max_q\left\{s^\top q-\frac12q^\top\Sigma q\right\} \quad\text{subject to }A^\top q=0. \]
The exact solution is
\[ \boxed{ q=\Sigma^{-1}s- \Sigma^{-1}A \left(A^\top\Sigma^{-1}A\right)^+ A^\top\Sigma^{-1}s } \]
where \(+\) denotes the Moore–Penrose pseudoinverse, allowing redundant constraints.
For gross exposure \(G\),
\[ \boxed{w=G\,\frac{q}{\sum_i|q_i|}}. \]
Weights are fractions of NAV. Thus \(G=2\) means 100% long and 100% short. Scaling preserves all three neutralities. This is a risk-adjusted direction normalized to gross; it is not generally the solution to maximizing raw signal under an \(L^1\) constraint.
Why 48 separate momentum constraints fail: write \(W_g=\sum_{i\in g}w_i\). If you impose
\[ m_gW_g=0\quad\text{separately for every }g, \]
then every industry with \(m_g\ne0\) must have \(W_g=0\). Consequently,
\[ s^\top w=\sum_g s_gW_g=0. \]
Within-industry long/short positions may remain feasible, but your shared industry signal cannot predict their return. Full industry-dollar neutrality has the same problem.
Also, \(m^\top w=0\) removes linear exposure to this particular momentum control; it does not establish that the remaining signal represents distinct information. If you also want stock-level reversal neutrality, append individual `own_lag` to \(A\).

2. Risk model

Use a factor model containing market, industry, and relevant style risks:
\[ r=Bf+\epsilon,\qquad \boxed{\Sigma=BFB^\top+D}. \]
Include your market betas, industry indicators, and estimated style exposures such as size, value, and momentum. Estimate factor covariance \(F\) with shrinkage and specific variances \(D\) with positive floors, using only history available before trading. Specify industry factors consistently with the market factor—for example, use market-residualized industry returns. Factor models provide both a parsimonious covariance estimate and a decomposition into common and specific risk. [MOSEK Portfolio Optimization Cookbook 1.6.0](https://docs.mosek.com/portfolio-cookbook/factormodels.html?utm_source=chatgpt.com)
A stock-specific diagonal model alone is generally inadequate for this book. Its industry net positions carry common industry risk. For industry covariance \(F_{\rm ind}\), that contribution is
\[ W^\top F_{\rm ind}W. \]
Adding more stocks within an industry diversifies specific risk, but does not eliminate this term. A diagonal model falsely treats those stocks as independent and can encourage excessive industry exposure.
Diagonal risk can be adequate if the omitted common-factor exposures are zero or negligible. Your momentum constraint does not ensure that. Market-beta neutrality cancels the modeled market exposure, while industry and style risks remain.

3. Position size

Use a volatility target with a gross-exposure cap when managing risk. Fixed gross is useful for comparing signals under consistent leverage, but does not give consistent volatility.
For annual target \(\sigma_*\), with daily covariance \(\Sigma\),
\[ \boxed{ w=cq,\qquad c=\min\left( \frac{\sigma_*}{\sqrt{252\,q^\top\Sigma q}}, \frac{G_{\max}}{\sum_i|q_i|} \right). } \]
The volatility target will be missed when the gross cap binds. The \(\sqrt{252}\) annualization assumes negligible serial covariance; check realized strategy volatility.
Do not leverage a numerically tiny residual after projection: hold zero when the constraints remove essentially all signal. Use position, industry, liquidity, and turnover limits in a production optimizer; clipping weights afterward breaks neutrality.

4. Python

This implements both interpretations, handles redundant constraints through SVD, and preserves row order. With only your five columns, it can construct weights using an identity risk penalty. Those columns are insufficient to estimate a realistic covariance or support volatility targeting, so supply a covariance callback for that.

_(The code block below lost its line breaks and its opening lines when copied from ChatGPT; it is reproduced as copied.)_

```
import numpy as np            codes, labels = g["industry"].factorize(sort=False)            H = np.eye(len(labels))[codes]            controls = H * m[:, None]        A = np.column_stack([np.ones(n), beta, controls])        C = (            np.ones(n)            if covariance is None            else np.asarray(covariance(date, g), dtype=float)        )        if not np.isfinite(C).all():            raise ValueError(f"Nonfinite covariance on {date}.")        # Sigma = L L.T. Project in whitened coordinates.        if C.shape == (n,):            if np.any(C <= 0):                raise ValueError("Diagonal variances must be positive.")            root = np.sqrt(C)            y = s / root            X = A / root[:, None]            def unwhiten(v):                return v / root            def variance(v):                return np.dot(C, v * v)        elif C.shape == (n, n):            if not np.allclose(C, C.T, rtol=1e-10, atol=1e-12):                raise ValueError("Covariance must be symmetric.")            L = cholesky((C + C.T) / 2, lower=True)            y = solve_triangular(L, s, lower=True)            X = solve_triangular(L, A, lower=True)            def unwhiten(v):                return solve_triangular(L.T, v, lower=False)            def variance(v):                return v @ C @ v        else:            raise ValueError("Covariance has an incorrect shape.")        # Normalize control columns before rank determination.        norms = np.linalg.norm(X, axis=0)        keep = norms > 0        X = X[:, keep] / norms[keep]        U, singular, _ = svd(X, full_matrices=False)        U = U[:, singular > rcond * singular[0]]        residual = y - U @ (U.T @ y)        if np.linalg.norm(residual) <= rcond * np.linalg.norm(y):            continue  # No surviving signal; leave weights at zero.        q = unwhiten(residual)        scale = gross / np.abs(q).sum()        if target_vol is not None:            annual_vol = np.sqrt(252 * variance(q))            scale = min(scale, target_vol / annual_vol)        weights[pos] = scale * q    out["weight"] = weights    return out
```

Usage:

```
# Weight construction from the supplied columns; identity penalty.book = make_weights(df, gross=2.0)# Recommended: realistic daily covariance, 10% annual vol, 200% gross cap.book = make_weights(    df,    covariance=lambda date, g: daily_covariance(date, g),    target_vol=0.10,    gross=2.0,)
```

I verified the projection against the closed-form formula, including neutrality, gross scaling, redundant constraints, and signal annihilation under separate industry constraints.
