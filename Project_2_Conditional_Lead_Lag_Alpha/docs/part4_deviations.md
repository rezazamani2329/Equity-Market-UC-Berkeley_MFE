# Part 4: changes after the pre-registration

`docs/part4_prereg.md` was committed before any Part 4 result existed. After a first
run on real data, an independent code audit (three reviewers, each serious finding
re-checked by a second reviewer) found the defects below. Each was fixed before the
results reported in the group report were produced. None of the fixes was chosen by
looking at which result it gave, and none changes the verdict. Effects are measured on
the final run.

| # | Change | Why | Effect |
|---|---|---|---|
| 1 | Convention C is charged a full round trip each day: traded notional = 2 × gross. | To earn exactly open → close the book must be flat overnight, so it buys the whole book at the open and sells it at the close. The first run charged Σ\|Δw\|, which is the cost of a book held overnight. | Traded notional is 2.0 per day under C, against 1.30–1.36 for a carried book, so every open-to-close breakeven falls by about a third. Headline: −0.14 bp to −0.09 bp; `common_lag`: 0.26 bp to 0.17 bp. |
| 2 | The cost criterion compares breakeven cost and half-spread over the same window, 2007–2024. | The pre-registration named 2007–2024 for the half-spread but did not say which window the breakeven used; the first run mixed a full-sample breakeven with a 2007–2024 spread. Full-sample breakeven is still reported. | Negligible for the headline (−0.09 bp either way). `common_lag`: 0.17 bp full sample, 0.14 bp over 2007–2024. |
| 3 | Part 4's books lag the leader's move by one session on the trading calendar (`calendar_lagged_signals`) instead of by row. | The team's `lag_leader_components` shifts by row and keeps the row's own date, so an industry entered day t's book only if its leader had a return at the close of t (look-ahead), and a missing day t−1 gave day t a stale t−2 value. The reconciliation rows still use the team's function so the committed numbers reproduce. | 758 of 12,388,705 follower-days (0.006%) change. Headline open-to-close mean is −0.46%/yr before and after (t −1.07 to −1.08). |
| 4 | Open-to-close return is missing on delisting-only days. | A stock with no opening print cannot be bought at the open; its delisting return runs from the previous close. | No change at three decimals for any book. |
| 5 | The convention-C spanning test uses a value-weighted open-to-close market built from CRSP (`mkt_C`) as its market factor. The close-to-close version is kept as a sensitivity. | Regressing an open-to-close book on a close-to-close market puts the overnight market premium into the intercept. The other Fama-French factors exist only close-to-close. | Headline spanning alpha −0.46%/yr, p = 0.29, against −0.44%/yr, p = 0.32, with the close-to-close market. |
| 6 | Drawdown is measured from initial capital of 1. | A first-day loss in a series or window was invisible. | Correctness only. |
| 7 | A day with no valid return among held stocks is missing, not 0%. | It is not the pre-registered "missing return contributes zero" case: the book has no return at all. | Correctness only. |
| 8 | The volatility overlay is applied to the headline book only, as pre-registered. | The first run also scaled `common_lag`. The 3× cap binds, so the scaled headline runs below the 10% target. | The scaled headline realises 6.7% volatility, not 10%: its unscaled volatility is 2.3%, so the required scale of 4.3 exceeds the cap of 3. |
| 9 | The timing decomposition compares open-to-close and overnight with close-to-close on the same stocks (those with an opening print). | Otherwise close-to-close includes stale-price days that neither leg covers. | Presentation only. |
| 10 | The neutrality check imputes a missing stock beta at its industry-day mean, as the construction does. | The first check counted a missing beta as 0. Beta coverage is 100%, so it made no difference here. | None on this data. |

## Exploratory analysis (not pre-registered)

`scripts/explore_part4_constraints.py` adds the industry-book constraints one at a time
(dollar only; + beta; + own-lag; all three) to see which one removes each signal's return.
It was run after the pre-registered results and is reported as exploratory.
