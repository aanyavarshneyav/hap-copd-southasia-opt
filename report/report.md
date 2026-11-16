# Forecasting and Optimal Intervention Planning for COPD Burden Attributable to Household Air Pollution in South Asia

**Authors:** Rudraksh (Person A), Aanya (Person B)  
**Course / Lab:** University AI Lab Project  
**Submission Format Note:** Drafted in Markdown during project execution; convert to final required submission template (`TODO(owner): LaTeX / Word / PDF`) at project freeze.

---

## Abstract
`TODO(shared): 150–250 word summary after final Stage 2–3 and planning results are locked.`

## 1. Introduction & Problem Statement
`TODO(shared): Context on HAP-COPD burden in South Asia, India vs. Nepal contrasts, gap between descriptive GBD studies and prescriptive resource allocation, and Research Questions 1–4.`

## 2. Data & Preprocessing
`TODO(Person A): Summarize GBD 2023 exports, UN WPP 2024, WHO SDG 7.1.2 clean-fuel access, urban/rural apportionment rule (reference docs/ASSUMPTIONS.md), and data validation checks.`

## 3. Methodology
### 3.1 Stage 1: Historical Burden & Trend Analysis (1990–2023)
`TODO(Person A): ASRs, EAPC, piecewise/joinpoint regression (pwlf), attributable share of total COPD.`

### 3.2 Stage 2: Forecasting (2024–2040) & Surrogate Calibration
`TODO(Person A): Statistical baselines (ARIMA, ETS, Poisson GLM APC), ML panel models (RF/HistGBDT), rolling-origin validation (MAE, RMSE, MASE), uncertainty propagation, and fast response surrogate.`

### 3.3 Stage 3: Multi-Objective Intervention Optimisation (NSGA-II)
`TODO(Person B): Decision space (4 units × 4 interventions × 3 periods), 3 objectives (2040 DALYs, cost, female–male ASR gap), constraints, and surrogate-assisted evaluation.`

### 3.4 Sequential & Symbolic Planning (MDP, HTN, STRIPS/PDDL)
`TODO(Person B): Value-iteration MDP formulation, HTN decomposition structure, STRIPS/PDDL encoding in pyperplan, and equal-footing comparison protocol against GA.`

## 4. Results
### 4.1 Historical Trends & Age–Sex Patterns (RQ1)
`TODO(Person A): Embed outputs/figures/s1_* and outputs/tables/s1_*.`

### 4.2 Forecasting Accuracy & 2040 Projections (RQ2)
`TODO(Person A): Embed outputs/tables/s2_rolling_cv_metrics.csv and outputs/figures/s2_*.`

### 4.3 Pareto-Optimal Clean-Cooking Allocations (RQ3)
`TODO(Person B): Embed outputs/tables/s3_pareto_front.csv and outputs/figures/s3_*.`

### 4.4 Adaptive MDP vs. Static GA & Symbolic Plans (RQ4)
`TODO(Person B): Embed outputs/tables/plan_mdp_vs_ga_comparison.csv, outputs/figures/plan_*, and PDDL/HTN plan summaries.`

## 5. Discussion, Limitations & Sensitivity Analysis
`TODO(shared): Modelled GBD limitations, scenario-analysis framing (not policy recommendations), sensitivity to cost/effectiveness/apportionment assumptions, and key risks.`

## 6. Conclusion
`TODO(shared): Concise synthesis of findings per RQ1–RQ4.`

## References
`TODO(shared): Verified literature citations for GBD 2023, UN WPP 2024, WHO SDG 7.1.2, and all sourced parameters in docs/ASSUMPTIONS.md.`
