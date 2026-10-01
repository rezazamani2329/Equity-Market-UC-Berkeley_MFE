# Project 1 — Hull–White Calibration and REMIC Bond Pricing

## MFE 230M — Asset Securitization

This project calibrates a one-factor Hull–White interest-rate model to market prices of SOFR caps and then applies the calibrated model to price mortgage-backed REMIC securities from **Freddie Mac Multiclass Certificate Series 5310**.

The project connects four important topics in securitized fixed income:

- term-structure modeling,
- mortgage prepayment behavior,
- Monte Carlo valuation,
- option-adjusted spread (OAS) analysis.

---

# Project Objective

The project has two main parts.

### Part 1 — Hull–White Calibration

Calibrate the one-factor Hull–White short-rate model

\[
dr(t) = [\theta(t)-\kappa r(t)]dt+\sigma dW(t)
\]

to **15 at-the-money backward-looking SOFR caps** with maturities from 1 to 30 years.

The calibration requires:

- SOFR cap strikes,
- normal/Bachelier implied volatilities,
- caplet accrual and payment dates,
- discount factors,
- Bloomberg cap prices.

The goal is to estimate:

\[
\kappa,\qquad \sigma,\qquad \theta(t)
\]

and reproduce the market term structure as closely as possible.

### Part 2 — Freddie Mac REMIC Pricing

Use the calibrated Hull–White model to price the two Group 3 securities from **Freddie Mac REMIC Series 5310**:

- **FB — Floater**
- **SB — Inverse Floater**

The valuation incorporates stochastic interest rates, refinancing incentives, mortgage prepayments, principal amortization, security cash flows, Monte Carlo simulation, antithetic variance reduction, and OAS estimation.

---

# Repository Structure

The project folder is organized as follows:

```text
project 1/
│
├── Data/
│   ├── cap_sofr_atm_..._20230413.csv
│   ├── cap_sofr_atm_..._20230413.csv
│   ├── P_caplet_accr..._20230413.csv
│   ├── P_caplet_pay..._20230413.csv
│   ├── REMIC_5310_Template.xlsx
│   ├── sofr_cap_accr..._20230413.csv
│   └── sofr_cap_bloo..._20230413.csv
│
├── Figures/
│   ├── fig_1d_bachelier_vs_bbg.png
│   ├── fig_1e_hw_vs_bachelier.png
│   └── fig_1f_theta.png
│
├── notebooks/
│   └── MFE230M_HW1_Group3.py
│
├── others/
│   ├── MFE230M_Homework_Set_1_2026.pdf
│   ├── MFE230M_HW1_Group3_writeup.docx
│   ├── REMIC_5310_Prospectus.pdf
│   └── sample_output.txt
│
└── README.md
```

> The exact CSV filenames may differ slightly from the abbreviated names displayed by Finder.

---

# Project Workflow

The overall workflow is:

```text
Market Data
    │
    ▼
SOFR Curve + Cap Volatilities + Bloomberg Cap Prices
    │
    ▼
Bachelier Cap Pricing
    │
    ▼
Hull–White Calibration
    │
    ├── κ
    ├── σ
    └── θ(t)
    │
    ▼
Monte Carlo Short-Rate Simulation
    │
    ▼
Simulated 10-Year Refinancing Rate
    │
    ▼
Mortgage Prepayment Model
    │
    ▼
REMIC Collateral Cash Flows
    │
    ▼
Floater / Inverse Floater Cash Flows
    │
    ▼
Bond Prices + Standard Errors
    │
    ▼
Option-Adjusted Spread Analysis
```

---

# 1. Hull–White Calibration

## 1.1 Accrual Periods

SOFR caplet accrual periods follow the ACT/360 convention:

\[
\tau_i =
\frac{D_i^{Acc}-D_{i-1}^{Acc}}{360}.
\]

The resulting quarterly accrual periods are approximately 0.25 years.

---

## 1.2 Forward SOFR Rates

Quarterly forward SOFR rates are calculated from the initial discount curve:

\[
F_i^{SOFR}
=
\frac{1}{\tau_i}
\left(
\frac{P_{i-1}^{Acc}}{P_i^{Acc}}-1
\right).
\]

A total of **120 forward SOFR rates** are generated over the 30-year horizon.

The forward curve begins near 4.98% and declines toward approximately 2.6% at longer maturities.

---

## 1.3 Proxy Option Expiry

For each backward-looking caplet, the volatility is scaled using the proxy time

\[
T_i^{1/3}
=
\hat T_{i-1}^{Acc}
+
\frac{
\hat T_i^{Acc}-\hat T_{i-1}^{Acc}
}{3}.
\]

This proxy accounts for the timing structure of the backward-looking SOFR caplets.

---

# 2. Bachelier Cap Pricing

Each caplet is first priced with the Bachelier normal-volatility model:

\[
Caplet_i =
\tau_i P_i^{Pmt}
\left[
(F_i-K)N(d_i)
+
\sigma_N \sqrt{T_i}n(d_i)
\right],
\]

where

\[
d_i =
\frac{F_i-K}
{\sigma_N\sqrt{T_i}}.
\]

Each cap price is obtained by summing its component caplets.

## Validation Result

The implementation reproduces Bloomberg cap prices very closely:

- pricing differences are **below 0.10% across all 15 maturities**,
- the maximum dollar difference is approximately **$2,170** for the 30-year cap on a $10 million notional.

This provides a useful validation of the market inputs and Bachelier implementation before the Hull–White calibration.

The comparison is shown in:

```text
Figures/fig_1d_bachelier_vs_bbg.png
```

---

# 3. Hull–White Parameter Estimation

The Hull–White parameters are estimated using nonlinear least squares:

\[
\min_{\kappa,\sigma}
\sqrt{
\sum_{i=1}^{15}
\left(
Cap_i^{Bachelier}
-
Cap_i^{HW}(\kappa,\sigma)
\right)^2
}.
\]

## Calibrated Parameters

| Parameter | Result |
|---|---:|
| Mean-reversion speed \(\kappa\) | **0.072395** |
| Short-rate volatility \(\sigma\) | **0.012861** |
| Volatility | **128.61 bp** |
| Initial short rate \(r_0\) | **4.9861%** |

The cap-price calibration RMSE is approximately:

\[
\boxed{\$33,554}
\]

for a $10 million cap notional.

The calibration was stable across different optimizer starting values.

The model fit is shown in:

```text
Figures/fig_1e_hw_vs_bachelier.png
```

---

# 4. Estimating the Hull–White Drift

The monthly time-dependent drift is estimated using

\[
\theta(t)
=
\frac{\partial f(0,t)}{\partial t}
+
\kappa f(0,t)
+
\frac{\sigma^2}{2\kappa}
\left(
1-e^{-2\kappa t}
\right).
\]

The instantaneous forward curve \(f(0,t)\) is obtained from the fitted initial discount curve.

The estimated \(\theta(t)\) curve is shown in:

```text
Figures/fig_1f_theta.png
```

---

# 5. Freddie Mac REMIC Series 5310

The second part of the project applies the calibrated Hull–White model to Group 3 of Freddie Mac REMIC Series 5310.

## Group 3 Collateral

| Characteristic | Value |
|---|---:|
| Initial principal | **$66,534,768** |
| Weighted-average maturity | **357 months** |
| Weighted-average coupon | **7.42%** |
| Servicing fee | **0.92%** |
| Net collateral coupon | **6.50%** |

The underlying deal information is contained in:

```text
Data/REMIC_5310_Template.xlsx
others/REMIC_5310_Prospectus.pdf
```

---

# 6. Securities Priced

## Floater — FB

The floater:

- receives all Group 3 collateral principal,
- pays floating interest based on approximately

\[
SOFR + 0.95\%.
\]

Its coupon is subject to the available collateral interest.

## Inverse Floater — SB

The inverse floater is a **notional interest-only security**.

Its coupon is approximately

\[
5.55\%-SOFR.
\]

Because the floater and inverse floater divide the available Group 3 interest, their values react very differently to interest-rate movements.

---

# 7. Monte Carlo Simulation

Monthly short-rate paths are simulated from the calibrated Hull–White model:

\[
r_{t+\Delta t}
=
r_t
+
[\theta(t)-\kappa r_t]\Delta t
+
\sigma\sqrt{\Delta t}Z_t.
\]

Simulation setup:

- **10,000 total interest-rate paths**
- **5,000 antithetic pairs**
- **357 monthly periods**
- Hull–White calibrated \(\kappa\), \(\sigma\), and \(\theta(t)\)

For each path, the model generates a simulated 10-year rate that serves as the refinancing rate used in the mortgage prepayment model.

---

# 8. Mortgage Prepayment Model

The model begins with a **250% PSA** baseline.

Monthly prepayment probability is modeled as

\[
q_t =
\lambda_t e^{\beta x_t},
\]

where

\[
\lambda_t
=
1-(1-CPR_t)^{1/12},
\]

and

\[
CPR_t =
2.5\times0.2\%
\times
\min(\text{Pool Age},30).
\]

The refinancing incentive is

\[
x_t =
\text{Mortgage WAC}
-
\text{Simulated 10Y Rate}_t.
\]

The sensitivity parameter is

\[
\beta=0.38089.
\]

Therefore:

\[
\text{Interest Rates}\downarrow
\quad\Rightarrow\quad
\text{Refinancing Incentive}\uparrow
\quad\Rightarrow\quad
\text{Prepayments}\uparrow.
\]

This creates the path dependency that is central to mortgage-backed-security valuation.

---

# 9. Cash-Flow Process

For every simulated interest-rate path, the program:

1. simulates the short rate,
2. calculates the corresponding 10-year refinancing rate,
3. determines the refinancing incentive,
4. calculates mortgage prepayments,
5. re-amortizes the mortgage collateral,
6. distributes collateral principal to the floater,
7. computes floater interest,
8. computes inverse-floater interest,
9. discounts all security cash flows,
10. averages discounted cash flows across Monte Carlo paths.

This converts simulated interest-rate scenarios into security-level valuations.

---

# 10. Main REMIC Results

| Security | Coupon | Price | Standard Error |
|---|---|---:|---:|
| **Floater (FB)** | SOFR + 0.95% | **101.5037** | **0.006743** |
| **Inverse Floater (SB)** | 5.55% − SOFR | **2.9138** | **0.004608** |

Combined value:

\[
\boxed{FB+SB=104.4175}
\]

The average principal returned to the floater is equal to the initial Group 3 collateral principal:

\[
\boxed{\$66,534,768}.
\]

---

# 11. Antithetic Variance Reduction

To improve Monte Carlo efficiency, every vector of random shocks \(Z\) is paired with \(-Z\).

The pair estimator is

\[
Y_i =
\frac12
\left[
PV(Z_i)+PV(-Z_i)
\right].
\]

Results:

| Security | Antithetic SE | Benchmark SE | Pair Correlation | Variance Reduction |
|---|---:|---:|---:|---:|
| **Floater** | 0.006743 | 0.006445 | +0.095 | ~0% |
| **Inverse Floater** | 0.004608 | 0.010381 | −0.803 | ~80% |

### Interpretation

Antithetic variates work extremely well for the inverse floater because its value is approximately monotonic in interest rates.

For the floater, however, coupon collars, changing mortgage prepayments, and principal timing make the payoff less monotonic. As a result, antithetic sampling provides little variance reduction.

---

# 12. Option-Adjusted Spread

A constant spread \(s\) is added to simulated discounting:

\[
DF_s(t)
=
DF(t)e^{-st}.
\]

The OAS is obtained by solving

\[
Price(s)=Market\ Price.
\]

Assuming a market price of 100:

| Security | Implied OAS |
|---|---:|
| **Floater (FB)** | **+76 bp** |
| **Inverse Floater (SB)** | **≈ −6,271 bp** |

### Interpretation

The floater has a zero-OAS model value above par, so a positive discount spread is required to reduce the model value to 100.

The inverse floater is a notional IO security with a zero-OAS value of approximately 2.91. Treating 100 as its market price therefore produces an economically extreme OAS and demonstrates why OAS must be interpreted carefully for non-standard securities.

---

# 13. Key Findings

The project highlights several important principles of asset securitization and fixed-income modeling:

### Interest-rate models must be calibrated to market instruments

SOFR cap prices provide market information for estimating the parameters of the Hull–White model.

### Mortgage securities are path dependent

Interest-rate movements affect refinancing incentives, which affect mortgage prepayments and therefore the timing of principal and interest.

### Security structure changes risk exposure

The floater and inverse floater receive cash flows from the same mortgage collateral but have dramatically different interest-rate sensitivities.

### Variance reduction is security dependent

Antithetic variates reduced the inverse-floater simulation variance by roughly **80%**, but provided little benefit for the floater.

### OAS requires economic interpretation

A mathematically computed spread is not automatically an economically meaningful relative-value measure, especially for notional IO securities.

---

# 14. How to Run

The main Python program is:

```text
notebooks/MFE230M_HW1_Group3.py
```

From the project directory, run:

```bash
python notebooks/MFE230M_HW1_Group3.py
```

The script reads the required market and REMIC inputs from the `Data/` directory and produces the numerical results and figures used in the analysis.

> If your current Python script expects the data files in the same directory rather than `Data/`, update the relative paths before publishing the repository.

---

# 15. Output

The project generates:

```text
Figures/fig_1d_bachelier_vs_bbg.png
Figures/fig_1e_hw_vs_bachelier.png
Figures/fig_1f_theta.png
```

A representative text output is stored in:

```text
others/sample_output.txt
```

---

# 16. Implementation Tools

The project is implemented in Python and uses numerical techniques including:

- NumPy / pandas data processing
- Bachelier normal cap pricing
- Hull–White zero-coupon bond option pricing
- nonlinear least-squares optimization
- term-structure interpolation
- Monte Carlo simulation
- antithetic variance reduction
- mortgage prepayment modeling
- REMIC cash-flow modeling
- numerical OAS solving

---

# 17. Course Context

**Course:** MFE 230M — Asset Securitization  
**Program:** Master of Financial Engineering  
**Project:** Project 1 — Hull–White Calibration and REMIC Bond Pricing

This project demonstrates the link between **interest-rate derivatives, term-structure modeling, mortgage behavior, structured-product cash flows, Monte Carlo valuation, and fixed-income relative-value analysis**.

---

## Notes on Data

Some files used in this project may contain instructor-provided or Bloomberg-derived data. Before making the repository public, verify that redistribution of these files is permitted.

If redistribution is restricted, the recommended public repository structure is:

```text
project 1/
├── README.md
├── notebooks/
│   └── MFE230M_HW1_Group3.py
├── Figures/
└── others/
    └── sample_output.txt
```

with proprietary or restricted raw data excluded through `.gitignore`.
