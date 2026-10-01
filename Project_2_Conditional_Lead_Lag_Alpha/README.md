# notebooks/

Every notebook here is paired with a companion `.py` file of the same name via [jupytext](https://jupytext.readthedocs.io/) (`py:percent` format, configured in the root `pyproject.toml`).

**The `.py` file is the source of truth.** Write and edit code there; the `.ipynb` is a synced, runnable view of it. See `CLAUDE.md` at the repo root for the exact workflow — that file governs how notebooks get created and updated in this project, so it isn't repeated here.

## Layout
Plain descriptive `snake_case` names, no numeric prefix — e.g. `data_exploration.py`, `dtw_baseline.py`. Name each notebook after what it does, not its position in a sequence.

- `data_cleaning.py` — writes the cleaned ES OHLCV parquet to `data/processed/` (raw -> processed).
- `macro_event_calendar.py` — a second, clearly-named cleaning notebook; writes the macro event calendar CSV to `data/processed/`. It builds an independent derived dataset (event timestamps, not price data), so it doesn't belong in `data_cleaning.py`.
- Every other notebook only reads from `data/processed/`; it never writes/derives data files itself. See `CLAUDE.md` for the full rule.

## DTW 30min game experiments

### Experiment A&B Design

Let me explain how we constructed the two DTW-based factors.

Both factors start from the same game definition.
For each macro-event timestamp $T$, we create anchor times $T_0 \in \{T-30, T, T+30, T+60\}$ minutes.

At each anchor $T_0$, we build an input path from $T_0$ to $T_0+60$ minutes, sampled every 5 minutes.
The path is transformed into z-scored log returns, so DTW focuses on shape rather than price level.
The prediction target is the 30-minute forward return:

$y = \ln P(T_0+90) - \ln P(T_0+60)$

For similarity, we compute DTW distance with a **Sakoe-Chiba** constraint window, using `dtaidistance`.
Then for each current game, we find nearest historical neighbors and average their future returns with distance weights:
$\text{weight}_i$ is proportional to $1/(\text{distance}_i + \epsilon) \times \exp(-\text{age}_i / \text{halflife})$.

Now the two factors differ in how the historical pool is defined.

**Factor 1**

Factor 1 is the mixed major-event DTW factor.
The event universe is four major releases: CPI, NFP, PCE, and FOMC.

When matching neighbors, we keep the same clock time and same offset, but we allow cross-event matching across those four event types.

This gives a larger pool and usually more stable estimates.

**Factor 2**

Factor 2 is the single-event expanded DTW factor.
The event universe is the full expanded macro calendar.

When matching neighbors, we require same event type, same clock time, and same offset.

So CPI only matches CPI, FOMC only matches FOMC, and so on.

Because each pool is smaller, we relax the minimum-history requirement and tune neighbor settings accordingly.

**In short**
1. Factor 1 prioritizes sample size by mixing major event types.
2. Factor 2 prioritizes economic purity by enforcing event-type-specific matching.
Both output one scalar expected return signal per game, and we evaluate them with IC, hit rate, and quantile spread.

### Shared setup (both experiments)

- **Price data**: `data/processed/es_1min_bars_2010_2026.parquet` (`close`, UTC 1-minute bars).
- **Event data**: `data/processed/macro_event_calendar_expanded_2010_2026.parquet`.
- **Game construction**:
  - For each event timestamp `T`, create anchors `T0 ∈ {T-30, T, T+30, T+60}` minutes.
  - DTW input path: `[T0, T0+60]`, sampled every 5 minutes, transformed to z-scored log returns.
  - Label return: `log P(T0+90) - log P(T0+60)` (a 30-minute forward "game").
- **DTW factor definition**:
  - Compute DTW distance with `dtaidistance` + Sakoe-Chiba window.
  - Select top-k nearest historical neighbors (no significance threshold in baseline).
  - Factor = distance-weighted + recency-weighted average of neighbors' future 30-minute returns.
- **Evaluation**:
  - Pearson IC, Spearman IC, sign hit-rate, quintile spread (Q5-Q1),
  - and per-event IC breakdown.

### Experiment A: Mixed major-event pool

- **Notebook**: `dtw_event_game_simple.py` / `dtw_event_game_simple.ipynb`
- **Research idea**:
  - Keep only major events (`CPI`, `NFP`, `PCE`, `FOMC`),
  - but allow matching across event types (mixed pool), conditioned on same `offset_min` and same ET clock.
- **Design parameters**:
  - `K_NEIGHBORS=15`, `LOOKBACK_YEARS=5`, `MIN_HISTORY=30`, `DTW_WINDOW_STEPS=3`.
- **Latest run summary**:
  - Event rows: `764` (CPI 216, NFP 208, PCE 205, FOMC 135).
  - Valid games: `2355`; scored games: `2105`.
  - Pearson IC: `0.0699`; Spearman IC: `0.0355`; hit-rate: `50.69%`; Q5-Q1: `0.000477`.
  - Per-event Pearson IC: CPI `0.0234`, FOMC `0.1486`, NFP `-0.0419`, PCE `0.0889`.
- **Conclusion**:
  - Mixed pooling gives a stronger aggregate IC in this run, but may blend heterogeneous event mechanisms.
  - Signal seems concentrated in FOMC/PCE; NFP remains weak/negative.

### Experiment B: Single-event pool on expanded calendar

- **Notebook**: `dtw_event_game_single_event_factors.py` / `dtw_event_game_single_event_factors.ipynb`
- **Research idea**:
  - Use all event types from expanded calendar,
  - and force matching strictly within the same event type (plus same `offset_min` and ET clock).
- **Design parameters**:
  - `K_NEIGHBORS=20`, `MIN_HISTORY=12`, `MAX_HISTORY_PER_GAME=250`, `DTW_WINDOW_STEPS=3`.
  - Smaller history threshold is used because per-event pools are thinner.
- **Latest run summary**:
  - Event rows: `3167` across `13` event types.
  - Valid games: `9926`; scored games: `9292`.
  - Pearson IC: `0.0111`; Spearman IC: `0.0006`; hit-rate: `49.19%`; Q5-Q1: `0.000092`.
  - Best per-event IC is still FOMC (`0.1229`), while most other event types are near zero.
- **Conclusion**:
  - Removing cross-event mixing improves economic purity but weakens aggregate predictive power in this baseline.
  - The result suggests event-specific structure exists (notably FOMC), but broad cross-event generalization is limited.

### Practical takeaway and next steps

- DTW-as-factor is currently **promising but not robust** across all event buckets.
- Next high-priority upgrades:
  - add significance gating / permutation null threshold before neighbor selection;
  - compare against simple momentum baseline on identical games;
  - add rolling monthly retrain and out-of-sample tracking by event type;
  - report uncertainty bands (e.g., bootstrap IC confidence intervals).

### Visual result dashboard

The following figures summarize the latest baseline outputs and help identify
where the signal is strongest/weakest:

1. **Overall metric comparison** (mixed major-event vs single-event expanded):

![DTW overall metrics comparison](figures/dtw_overall_metrics_comparison.png)

2. **Event-level IC decomposition**:

![DTW event-level IC decomposition](figures/dtw_event_level_ic.png)

3. **T0-specific factor IC decomposition** (for mixed major-event setup):

![DTW T0 IC decomposition](figures/dtw_t0_ic_decomposition.png)