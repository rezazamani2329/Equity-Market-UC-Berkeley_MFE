# Project 2 — Conditional Lead-Lag Alpha

## A Risk-Constrained Long-Short Equity Strategy

This project investigates whether smaller firms exhibit different return dynamics following common industry shocks versus leader-specific residual shocks.

## Research Question

When a dominant large-cap stock moves, should smaller firms in the same industry continue in the same direction because of information diffusion, or reverse when their movement reflects exposure to a leader-specific shock?

## Main Hypotheses

1. Common industry shocks generate short-horizon continuation in smaller firms.
2. Small-stock movements associated with leader-specific residual shocks may subsequently reverse.
3. Conditioning on the source of the leader's move may produce stronger alpha than unconditional short-term reversal.
4. The signal should remain economically meaningful after risk controls, turnover, and transaction costs.

## Research Pipeline

1. Data preparation and universe construction
2. Industry leader and follower identification
3. Baseline lead-lag analysis
4. Leader shock decomposition
5. Conditional alpha signal construction
6. Signal validation and horizon analysis
7. Long-short portfolio construction
8. Portfolio optimization and risk management
9. Transaction-cost and robustness analysis
10. Out-of-sample evaluation

## Repository Structure

- `data/` — project data and data documentation
- `notebooks/` — empirical analysis notebooks
- `src/` — reusable Python functions
- `results/` — numerical outputs and tables
- `figures/` — final charts and figures
- `report/` — final project report
- `appendix/` — supporting material and ChatGPT interactions
