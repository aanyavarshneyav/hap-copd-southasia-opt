# Stage 1 & Stage 2 Handoff Specification (Person A to Person B)

**Date:** 2026-11-17  
**Author:** Person A (Rudraksh — Data & Forecasting Track)  
**Recipient:** Person B (Aanya — Optimisation & Planning Track)  
**Status:** Validated, Production-Ready, Zero-Mock

---

## 1. Executive Summary & Deliverables Inventory

This document serves as the formal data and model contract handoff from Person A to Person B. All historical analyses (1990–2023) and forecasting models (2024–2040) have been completed, verified against schema invariants, and output directly into `data/processed/`, `outputs/tables/`, and `outputs/figures/`.

### Deliverables Checklist

| Deliverable | Priority | Status | Output File Location | Details / Key Attributes |
| --- | --- | --- | --- | --- |
| **Final GBD Historical Dataset** | **Essential** | `Complete` | `data/interim/historical_burden.parquet`<br/>`outputs/tables/s1_historical_summary.csv` | 1990–2023 GBD 2023 panel across 5 countries + 4 units (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`), sex-stratified (ASM-04b) |
| **Historical Analysis Tables & Figures** | **Essential** | `Complete` | `outputs/tables/s1_*.csv`<br/>`outputs/figures/s1_*.png` | EAPC with 95% CIs, pwlf inflection breakpoints, attributable shares of total COPD & all-cause HAP |
| **Rolling-Origin Forecasting Metrics** | **Essential** | `Complete` | `outputs/tables/s2_rolling_cv_metrics.csv`<br/>`outputs/figures/s2_model_comparison_cv.png` | Backtest over origins 2008, 2013, 2018 ($h=5$) evaluating ARIMA, ETS, Poisson Rate GLM, and HistGBDT |
| **Final Forecast CSV** | **Essential** | `Complete` | `data/processed/forecasts_stage3.csv`<br/>`outputs/tables/s2_forecasts_2040_summary.csv` | Full annual BAU (2025–2040) + counterfactual anchors (`cf_25`, `cf_50`, `cf_75`, `cf_100`) across all 4 units and sexes |
| **Baseline Unit State CSV** | **Essential** | `Complete` | `data/processed/baseline_unit_state.csv` | $t_0 = 2025$ starting conditions, initial coverage, grid/LPG readiness, and 2040 BAU reference values |
| **Exposure–Response Surrogate JSON** | **Essential** | `Complete` | `data/processed/exposure_response_surrogate.json` | Vectorized log-linear response parameters ($M = \exp(\beta \Delta c)$) evaluated in microseconds on CPU |
| **Forecast Uncertainty Information** | **Essential** | `Complete` | Documented in Section 3 & columns `(lower, val, upper)` in `forecasts_stage3.csv` | Propagated 95% uncertainty intervals combining GBD baseline variance with forward projection horizon scaling |
| **Source Manifest & Extraction Filters** | **Essential** | `Complete` | `data/raw/manifest.json`<br/>`docs/DATA_SPEC.md` | SHA-256 verified raw file catalog, extraction query specifications, and codebook provenance |
| **Reproducible Scripts & Instructions** | **Important** | `Complete` | `src/data/ingest.py`, `src/historical/run.py`, `src/forecast/run.py` | Executable via `make validate-data`, `make stage1`, `make stage2`, `make check` |
| **Stage 2 Uncertainty Draws for Pareto Re-eval** | **Important** | `Complete` | `data/processed/stage2_uncertainty_draws.csv` | 500 Monte Carlo draws per unit × sex for 2040 DALYs and ASRs for Task A-08 / B-02 Pareto validation |

---

## 2. Actual 2040 Baseline (BAU) Projections

The table below provides the exact point estimates, lower 95% UI, and upper 95% UI for all 4 decision units under the Business-As-Usual (BAU) trajectory in 2040:

### 2.1 Age-Standardised Rates (ASR per 100,000 Persons)

| Location | Measure | Sex | 2040 Central (`val`) | 2040 Lower (95% UI) | 2040 Upper (95% UI) | Projected Clean Fuel Share |
| --- | --- | --- | --- | --- | --- | --- |
| **`IND_urban`** | DALYs | Both | **124.08** | 93.30 | 154.85 | 0.985 (98.5%) |
| `IND_urban` | DALYs | Female | **137.01** | 103.03 | 170.99 | 0.985 |
| `IND_urban` | DALYs | Male | **111.93** | 84.17 | 139.69 | 0.985 |
| `IND_urban` | Deaths | Both | **6.08** | 4.57 | 7.59 | 0.985 |
| `IND_urban` | Deaths | Female | **6.49** | 4.88 | 8.10 | 0.985 |
| `IND_urban` | Deaths | Male | **5.70** | 4.29 | 7.11 | 0.985 |
| **`IND_rural`** | DALYs | Both | **174.65** | 131.34 | 217.97 | 0.850 (85.0%) |
| `IND_rural` | DALYs | Female | **192.83** | 145.01 | 240.66 | 0.850 |
| `IND_rural` | DALYs | Male | **157.56** | 118.48 | 196.63 | 0.850 |
| `IND_rural` | Deaths | Both | **9.80** | 7.37 | 12.23 | 0.850 |
| `IND_rural` | Deaths | Female | **10.45** | 7.86 | 13.04 | 0.850 |
| `IND_rural` | Deaths | Male | **9.18** | 6.90 | 11.45 | 0.850 |
| **`NPL_urban`** | DALYs | Both | **207.44** | 155.99 | 258.89 | 0.820 (82.0%) |
| `NPL_urban` | DALYs | Female | **233.15** | 175.33 | 290.97 | 0.820 |
| `NPL_urban` | DALYs | Male | **179.28** | 134.82 | 223.74 | 0.820 |
| `NPL_urban` | Deaths | Both | **11.65** | 8.76 | 14.54 | 0.820 |
| `NPL_urban` | Deaths | Female | **12.18** | 9.16 | 15.20 | 0.820 |
| `NPL_urban` | Deaths | Male | **11.07** | 8.32 | 13.81 | 0.820 |
| **`NPL_rural`** | DALYs | Both | **861.02** | 647.49 | 1074.55 | 0.480 (48.0%) |
| `NPL_rural` | DALYs | Female | **967.75** | 727.75 | 1207.75 | 0.480 |
| `NPL_rural` | DALYs | Male | **744.15** | 559.60 | 928.70 | 0.480 |
| `NPL_rural` | Deaths | Both | **42.12** | 31.67 | 52.56 | 0.480 |
| `NPL_rural` | Deaths | Female | **44.03** | 33.11 | 54.95 | 0.480 |
| `NPL_rural` | Deaths | Male | **40.01** | 30.09 | 49.93 | 0.480 |

### 2.2 Absolute Burden Counts (Persons in 2040)

| Location | Measure | Sex | 2040 Count Central | 2040 Lower (95% UI) | 2040 Upper (95% UI) | 2040 Population Base |
| --- | --- | --- | --- | --- | --- | --- |
| **`IND_urban`** | DALYs | Both | **865,686** | 650,996 | 1,080,376 | 697,709,421 |
| `IND_urban` | DALYs | Female | **466,009** | 350,439 | 581,580 | 340,128,817 |
| `IND_urban` | DALYs | Male | **399,677** | 300,557 | 498,797 | 357,580,603 |
| **`IND_rural`** | DALYs | Both | **1,615,321** | 1,214,721 | 2,015,920 | 924,870,627 |
| `IND_rural` | DALYs | Female | **869,321** | 653,729 | 1,084,912 | 450,819,650 |
| `IND_rural` | DALYs | Male | **746,000** | 560,992 | 931,008 | 474,050,978 |
| **`NPL_urban`** | DALYs | Both | **20,480** | 15,401 | 25,560 | 9,872,862 |
| `NPL_urban` | DALYs | Female | **12,046** | 9,058 | 15,033 | 5,166,847 |
| `NPL_urban` | DALYs | Male | **8,435** | 6,343 | 10,527 | 4,706,016 |
| **`NPL_rural`** | DALYs | Both | **198,350** | 149,159 | 247,541 | 23,036,679 |
| `NPL_rural` | DALYs | Female | **116,672** | 87,738 | 145,607 | 12,055,975 |
| `NPL_rural` | DALYs | Male | **81,678** | 61,421 | 101,934 | 10,980,708 |

---

## 3. Methodological Decisions & Invariants

1. **Empirical Sex Baseline Split ([ASM-04b](file:///c:/Users/Rudraksh/projects/hap-copd-southasia-opt/docs/ASSUMPTIONS.md#L15)):**
   - India: Female 53.46% DALYs / Male 46.54% DALYs.
   - Nepal: Female 58.34% DALYs / Male 41.66% DALYs.
   - Verified against WHO GHE 2021 empirical microdata (`data (2).csv` / `data (3).csv`).
2. **Apportionment Formula ([ASM-01](file:///c:/Users/Rudraksh/projects/hap-copd-southasia-opt/docs/ASSUMPTIONS.md#L11)):**
   $$w_k = \frac{\text{Pop}_k \cdot (1 - \text{CleanFuel}_k)}{\sum_{j \in \{u,r\}} \text{Pop}_j \cdot (1 - \text{CleanFuel}_j)}, \quad B_{c,k} = w_k B_c$$
3. **Surrogate Calibration ([exposure_response_surrogate.json](file:///c:/Users/Rudraksh/projects/hap-copd-southasia-opt/data/processed/exposure_response_surrogate.json)):**
   - Calibrated slope $\beta$:
     - `IND_urban`: $-1.1500$
     - `IND_rural`: $-1.2464$
     - `NPL_urban`: $-1.2285$
     - `NPL_rural`: $-1.1892$
   - Form: $M_{\text{male}} = \exp(\beta \Delta c)$, $M_{\text{female}} = \exp(r_{fm} \cdot \beta \Delta c)$ with $r_{fm} = 1.05$.
   - Uncertainty residual SD: $\sigma = 0.08$.

---

## 4. Instructions for Person B (Stage 3 & Planning Execution)

1. **Evaluator Verification:** Person B's `src/optimise/evaluator.py` already passes against real production data.
2. **Running Stage 3 (NSGA-II):**
   - Read parameters from `configs/intervention_params.json` (or set `interfaces.use_mock_forecasts: true` during prototyping).
   - Objectives: (1) Total 2040 DALYs, (2) Total Cost, (3) Population-weighted female–male ASR equity gap.
   - Run GA: `python -m src.optimise.run --config configs/default.yaml`.
3. **Pareto Re-Evaluation (Task A-08):**
   - Load `data/processed/stage2_uncertainty_draws.csv` (contains 500 draws per unit × sex).
   - Evaluate selected knee and extreme Pareto solutions across all 500 draws to generate 95% UI error bars for `s3_pareto_reeval.csv`.

---

**Sign-off:** All historical metrics, CV benchmarks, 2040 forecasts, and hand-off contracts are locked and reproducible via `make check`.
