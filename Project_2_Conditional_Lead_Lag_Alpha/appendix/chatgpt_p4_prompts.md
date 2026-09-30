# Part 4: ChatGPT interactions

The rubric requires at least three substantial ChatGPT interactions across the project,
each documented as the prompt, a summary of the output, and a critical evaluation. These
are Part 4's two. Run each prompt in ChatGPT as written, paste the reply under "Output",
and write the evaluation using the checklist under each prompt. The checklist is what
Part 4 actually found, so it tells you what a good answer should contain and where
ChatGPT went wrong.

---

## Interaction 1: robustness design

**Prompt**

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

**Output**

_(paste ChatGPT's reply here)_

**Summary** _(3–5 bullets)_

**Critical evaluation**: check its answer against what Part 4 found:

- [ ] **Execution timing.** Did it say that trading at the close used to compute the
  signal is impossible, and suggest open-to-close execution? Part 4's implementable
  headline is open-to-close. It also found Part 3 had scored the signal one session late,
  which a reply is unlikely to anticipate.
- [ ] **Bid-ask bounce.** Did it flag bounce? Part 4 found about 80% of the close-to-close
  reversal benchmark is bounce: 10.1%/yr on closes vs 1.9%/yr on midpoints, and it is
  negative on midpoints in 1996–2006.
- [ ] **Signal constant within industry.** Did it notice that an industry-neutral book
  on an industry-level signal is identically zero? This is the design point most likely
  to be missed.
- [ ] **Market-beta timing.** Did it say that ranking on a component containing
  yesterday's market move is a beta bet? Beta-neutralization alone takes the
  conditional signal from t = 3.0 to t = −0.6.
- [ ] **Industry momentum.** Did it separate leader information from the followers'
  own industry move the day before?
- [ ] **Look-ahead in lagging.** Did it warn about lagging by row rather than by
  calendar? The team's code had this.
- [ ] **Transaction costs.** Did it put numbers on turnover and costs? Turnover is about
  130% of gross per day, and breakeven is below 1 bp against a 2.6 bp half-spread.
- [ ] **Regime change.** Did it mention decimalization in 2001? The reversal benchmark
  lost money every year from 1996 to 2001 when traded open-to-close.
- [ ] **Generic items.** List anything it suggested that does not apply or cannot be
  tested with this data.

---

## Interaction 2: portfolio construction and risk model

**Prompt**

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

**Output**

_(paste ChatGPT's reply here)_

**Summary** _(3–5 bullets)_

**Critical evaluation**: what Part 4 did, for comparison:

- [ ] **Level of the book.** Did it rank the 48 industries and spread weight within
  each? Stock-level construction does not work for a signal that is constant within
  an industry.
- [ ] **Neutrality method.** Did it neutralize by projection, i.e. the regression
  residual of the ranks on [1, beta, own_lag] each day? That is exact and needs no
  optimizer. Or did it reach for a numerical optimizer?
- [ ] **Risk model.** Did it see that a diagonal model understates risk here, because
  the real breadth is about 48 correlated industry bets, not 1,750 stocks?
- [ ] **Beta estimation.** Did it warn that plain OLS beta is biased down for thinly
  traded stocks (Dimson correction)?
- [ ] **Code.** Does its code run? Does it use only past data? Does it handle missing
  betas? Test it on a few rows before trusting it.
- [ ] **What we adopted or rejected, and why.**

---

## Reflection for the report (section 3d)

A short paragraph: where ChatGPT helped (checklist coverage, standard items), where it
was generic or wrong, and what it missed that the data showed. The pattern from HW1
and HW2 was that it assumed richer data than we had. Here the likely pattern is that
it treats the signal as stock-level and so misses the design issues that come from the
signal being industry-level.
