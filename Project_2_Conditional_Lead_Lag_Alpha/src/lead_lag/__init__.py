"""
Conditional lead-lag alpha — Berkeley MFE 230GA, Project 2.

The package is laid out along the team's five parts, so that "who owns this
file" never has to be asked:

    data/        part 1  Data & Universe        WRDS pulls, cleaning, industry
                                                mapping, liquidity filters,
                                                leader/follower assignment
    baseline.py  part 1  the unconditional lead-lag test
    shocks/      part 2  Econometrics & Shock Identification
    signals/     part 3  Alpha Signal & Validation
    portfolio/   part 4  Portfolio Optimization & Risk
    robustness/  part 5  Robustness, Final Report & Submission

Dependencies run one way only: `shocks` imports from `data`, `signals` from
`shocks`, and so on.  Nothing in `data/` imports from a later part.  That is
what lets part 1 be finished, tested and frozen while the rest is still moving.
"""
