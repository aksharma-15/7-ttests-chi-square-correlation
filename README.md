# 7-ttests-chi-square-correlation
statistical tests foundations for Machine Learning and Data Science

---

# Uber ETA Reliability & Operational Hypothesis Testing

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

## Executive Summary

Before a rider confirms a trip, Uber surfaces an Estimated Time of Arrival (ETA). If these estimates are systematically optimistic, rider trust erodes, cancellation rates climb, and marketplace liquidity suffers. 

This repository contains a rigorous statistical audit of **4,328 simulated ride records** spanning 30 days (720 hourly slots) to evaluate ETA calibration, peak-hour operational bottlenecks, cancellation dependencies, fraud detection efficacy, and fare-duration linearity. Using parametric t-tests, non-parametric chi-square tests of independence, and bivariate correlation analysis, we quantify where operational assumptions hold and where systematic bias exists.

---

## Dataset Architecture & Simulation Mechanics

The synthetic dataset (`load_uber`) models realistic marketplace dynamics across a 30-day window (`hour_slot` 0 to 719) using a Poisson arrival process (lambda = 6 rides per hour).

| Feature | Data Type | Operational Definition & Generative Logic |
| :--- | :--- | :--- |
| `hour_slot` | `int` | Sequential hourly index across 30 days (`0` to `719`). |
| `hour` | `int` | Hour of the day (`0` to `23`), derived via modulo 24 arithmetic. |
| `is_peak` | `binary` | Rush-hour indicator (`1` for hours 8–10 and 17–19; `0` otherwise). |
| `trip_minutes` | `float` | Actual ground-truth trip duration drawn from a normal distribution (mean = 24.0 min, SD = 6.0 min, floor-clipped at 4.0 min). |
| `fare` | `float` | Base fare plus linear time rate (`45 + 11 * trip_minutes + Gaussian noise`), scaled by a `2.0x–2.8x` multiplier during surge events. |
| `surge` | `binary` | Dynamic pricing indicator occurring at a 2% baseline probability. |
| `cancelled` | `binary` | Rider/driver cancellation flag (20% probability during peak hours vs. 8% off-peak). |
| `is_fraud` | `binary` | Ground-truth fraudulent trip indicator (1% baseline prevalence). |
| `flagged` | `binary` | Automated anomaly detector alert (90% True Positive Rate / Recall; 5% False Positive Rate). |
| `eta_minutes` | `float` | Pre-trip promised duration (`trip_minutes - optimism_bias + Gaussian noise`). |
| `eta_error` | `float` | Prediction residual defined as `actual trip_minutes - promised eta_minutes`. Positive values indicate the ride ran longer than promised. |

---

## Statistical Methodology & Theoretical Foundations

### 1. Hypothesis Testing Framework
Every operational question is formulated under the **Neyman-Pearson frequentist framework**:
* **Null Hypothesis (H0):** The baseline assumption of no effect, no difference, or independence between variables.
* **Alternative Hypothesis (H1):** The directional or non-directional operational claim we seek evidence to support.
* **P-Value & Significance Level (alpha = 0.05):** The probability of observing a test statistic at least as extreme as the sample result, assuming H0 is true. When `p < alpha`, we reject H0 in favor of H1.
* **One-Tailed vs. Two-Tailed Tests:** Directional hypotheses (`alternative='less'` or `alternative='greater'`) concentrate the entire Type I error rate (alpha) into a single tail of the sampling distribution, increasing statistical power when domain logic dictates a specific direction of bias.

---

### 2. Parametric T-Tests (`scipy.stats`)

T-tests evaluate continuous numerical variables by comparing sample means relative to sample variance and degrees of freedom (`df`).

#### A. One-Sample T-Test (`stats.ttest_1samp`)
* **Objective:** Determine whether actual average trip durations are significantly shorter than a published benchmark of 25 minutes using a random sample of `n = 200` rides.
* **Hypotheses:**
  * **H0:** True population mean trip duration is greater than or equal to 25 minutes (`mu >= 25`).
  * **H1:** True population mean trip duration is strictly less than 25 minutes (`mu < 25`).
* **Findings:** With a sample mean of **24.05 minutes**, `t = -2.2924` (`df = 199`), and `p = 0.0115`, we reject H0 at the 5% significance level. Trips are statistically faster than the 25-minute benchmark.

#### B. Two-Sample Independent T-Test / Welch's T-Test (`stats.ttest_ind(..., equal_var=False)`)
* **Objective 1 (ETA Optimism Bias):** Test whether the ETA prediction error (`eta_error`) is significantly worse during rush hour (`n = 1,064`) compared to off-peak hours (`n = 3,264`).
  * **H0:** Mean ETA error during peak hours equals mean ETA error during off-peak hours (`mu_peak = mu_off`).
  * **H1:** Mean ETA error is strictly greater during peak hours (`mu_peak > mu_off`).
  * **Why Welch's T-Test (`equal_var=False`)?** Sample sizes (`1,064` vs. `3,264`) and group variances differ. Welch's adaptation adjusts the degrees of freedom using the Welch-Satterthwaite equation, preventing Type I error inflation without requiring homoscedasticity (equal variance).
  * **Findings:** Peak rides underestimate duration by **3.06 minutes** on average versus **0.84 minutes** off-peak (`t = 25.37`, `p = 6.47e-122`). The operations team's ETA model exhibits severe, statistically undeniable optimism bias during rush hour.
* **Objective 2 (Actual Trip Duration):** Test whether rush-hour trips actually take longer (`mu_peak > mu_off`) rather than merely suffering from worse ETA predictions.
  * **Findings:** Peak trips average **24.51 minutes** compared to **23.99 minutes** off-peak (`t = 2.41`, `p = 0.0079`). While statistically significant due to large sample size, the practical effect size (~31 seconds) is minor compared to the 2.22-minute gap in ETA error.

#### C. Paired-Sample T-Test (`stats.ttest_rel`)
* **Objective:** Audit a sample of `n = 40` individual rides by comparing each ride's promised `eta_minutes` directly against its realized `trip_minutes`.
* **Mathematical Intuition:** Because both measurements belong to the exact same trip, taking the row-wise difference (`d_i = trip_minutes_i - eta_minutes_i`) eliminates trip-to-trip baseline variance (e.g., short 10-minute trips vs. long 40-minute trips). This effectively reduces the problem to a one-sample t-test on the paired differences against a null mean of zero.
* **Hypotheses:**
  * **H0:** Mean paired difference is zero (`mu_diff = 0`).
  * **H1:** Actual trip duration exceeds promised ETA (`mu_diff > 0`).
* **Findings:** Promised ETA averaged **23.13 minutes** against an actual duration of **24.51 minutes** (`t = 2.92`, `p = 0.0029`), confirming systematic underestimation even on a small 40-ride operational audit.

---

### 3. Non-Parametric Chi-Square Test of Independence (`stats.chi2_contingency`)

When evaluating categorical variables, comparing means is invalid. Instead, we construct a **contingency table** (`pd.crosstab`) of observed frequencies and compare them against expected frequencies under the assumption of statistical independence.

* **Core Formula Logic:** For every cell in the contingency table, the expected count is `(Row Total * Column Total) / Grand Total`. The chi-square statistic sums the squared deviations between observed and expected counts, normalized by the expected count.

#### A. Peak Hours vs. Ride Cancellations
* **Hypotheses:**
  * **H0:** `is_peak` and `cancelled` are statistically independent.
  * **H1:** Cancellation rates depend on whether the ride occurs during peak hours.
* **Findings:** Off-peak cancellations sit at **8.6%** (`280 / 3,264`), whereas rush-hour cancellations more than double to **17.8%** (`189 / 1,064`). With `chi-square = 69.11` (`df = 1`), we reject H0 and confirm strong operational coupling between peak congestion and cancellations.

#### B. Fraud Ground Truth vs. Automated Detector Flags
* **Hypotheses:**
  * **H0:** `is_fraud` and `flagged` are independent (the classifier performs no better than random noise).
  * **H1:** `is_fraud` and `flagged` are significantly associated.
* **Findings:** The confusion matrix yields `chi-square = 530.30` (`p = 2.43e-117`), validating that the rule/model-based detector captures genuine fraudulent signal.

---

### 4. Bivariate Correlation & Outlier Sensitivity (`stats.pearsonr`)

* **Objective:** Quantify the linear relationship between `trip_minutes` and `fare`.
* **Hypotheses:**
  * **H0:** Linear correlation coefficient is zero (`rho = 0`).
  * **H1:** A non-zero linear relationship exists (`rho != 0`).
* **Findings & Domain Nuance:**
  * Across **all rides**, Pearson's `r = 0.728` (`p < 1e-16`), indicating a strong positive linear relationship.
  * However, visual inspection via `show_scatter` reveals a distinct upper-tier regime driven by `surge == 1` rides (2% of volume). When filtering out surge multipliers (`normal_rides`, `n = 4,246`), linearity tightens dramatically to **`r = 0.938`**. This demonstrates why senior practitioners always pair summary statistics with stratified visualizations to uncover regime-specific variance.

---

## Code Annotations & Engineering Best Practices

Several implementation details in the notebook reflect deliberate data science engineering choices:

1. **Modern PRNG Seeding (`np.random.default_rng(seed)`):** Uses NumPy's PCG64 Generator interface rather than legacy `np.random.seed()`, ensuring thread-safe, statistically sound reproducibility across simulation runs.
2. **Modular Visualization Helpers (`_axes`, `show_groups`, `show_paired`, `show_scatter`, `show_line`):** Encapsulates Matplotlib boilerplate (removing top/right chart spines, standardizing grid z-order, and applying colorblind-friendly hex palettes) so analytical cells remain declarative and readable.
3. **Dumbbell / Slope Plots (`show_paired`):** Visualizes paired t-test mechanics by sorting rides by promised ETA and plotting directional error segments—distinguishing over-runs (amber) from early arrivals (teal) at a glance.
4. **Defensive `.shape` vs. `len()` Checks:** Explicitly verifies group sample sizes (`(3264,)` vs `(1064,)`) prior to running Welch's t-test to validate sufficient sample size for the Central Limit Theorem (CLT) to hold.

---

## Repository Structure & Quickstart

```text
├── uber_eta_hypothesis_testing.ipynb   # Main analytical notebook & visualizations
└── README.md                           # Statistical documentation & operational findings
