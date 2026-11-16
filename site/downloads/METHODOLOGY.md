# Methodology Specification

All stages execute in **100% Python (3.12)** on CPU-only laptops using parameters from `configs/default.yaml`. Every stage outputs metadata (`source`, `generated_at`, `data_version`, `seed`).

---

## Stage Summary Matrix

| Stage | Owner | Primary Inputs | Primary Outputs | Core Metrics |
| --- | --- | --- | --- | --- |
| **0. Ingestion & Apportionment** | Person A | `data/raw/*` (see `DATA_SPEC.md`) | `data/interim/*.parquet`, `data/processed/baseline_unit_state.csv` | Schema, completeness, uncertainty validity (`lower <= val <= upper`) |
| **1. Historical Analysis (1990–2023)** | Person A | `data/interim/gbd_*.parquet` | `outputs/tables/s1_*.csv`, `outputs/figures/s1_*.png` | Deaths, DALYs, ASRs, EAPC (95% CI), joinpoint APCs, HAP share of total COPD |
| **2. Forecasting (2024–2040)** | Person A | `data/interim/*.parquet`, `who_sdg712_clean.parquet` | `outputs/tables/s2_*.csv`, `outputs/figures/s2_*.png`, `data/processed/forecasts_stage3.csv`, `data/processed/exposure_response_surrogate.json` | Rolling-origin CV: MAE, RMSE, MASE; 95% UI propagation |
| **3. Optimisation (NSGA-II)** | Person B | `data/processed/forecasts_stage3.csv`, `baseline_unit_state.csv`, `exposure_response_surrogate.json`, `docs/ASSUMPTIONS.md` | `outputs/tables/s3_pareto_front.csv`, `outputs/tables/s3_pareto_reeval.csv`, `outputs/figures/s3_*.png` | Projected 2040 DALYs, total cost, female–male ASR gap, constraint feasibility |
| **4. Planning (MDP, HTN, PDDL)** | Person B | Same state/surrogate inputs as Stage 3 + Stage 3 Pareto reference | `outputs/tables/plan_mdp_vs_ga_comparison.csv`, `outputs/tables/plan_htn_trace.json`, `pddl/*.pddl`, `outputs/tables/plan_pddl_solution.txt`, `outputs/figures/plan_*.png` | Expected DALYs averted, discounted/undiscounted reward, policy parity vs GA, plan validity |

---

## Stage 1: Historical Analysis (1990–2023)
**Scope:** 5 GBD South Asia countries (`India`, `Nepal`, `Pakistan`, `Bangladesh`, `Bhutan`), highlighting **India vs. Nepal**.  
**Entrypoint:** `src/historical/run.py` (`make stage1`).

1. **Burden & Share Metrics:**
   - Extract HAP-attributable COPD deaths and DALYs (`Number`, `Rate` per 100k, `Percent` of COPD) by `location × year × sex × age_group`, preserving GBD `(val, lower, upper)`.
   - Compute HAP-attributable share of total COPD (`HAP_COPD / Total_COPD`) and COPD share of all HAP-linked causes (`IND` and `NPL`).
2. **EAPC (Minimum Deliverable):**
   - Fit $\ln(\text{ASR}_t) = \alpha + \beta t + \varepsilon_t$ via OLS over `1990–2023` (and sub-periods if needed); $\text{EAPC} = 100 \times (e^\beta - 1)$ with 95% CI from $\text{SE}(\beta)$.
3. **Piecewise / Joinpoint Trends:**
   - Fit piecewise linear regression on $\ln(\text{ASR}_t)$ using `pwlf` (selecting $0 \le k \le 3$ breakpoints via BIC/AIC) to estimate segment Annual Percentage Changes (APCs) and inflection years.
4. **Outputs:**
   - `outputs/tables/s1_historical_summary.csv` — 1990 vs. 2023 deaths, DALYs, ASRs, and attributable shares by country and sex.
   - `outputs/tables/s1_eapc_joinpoint.csv` — EAPC + `pwlf` segment breakpoints and segment APCs.
   - `outputs/figures/s1_trends_southasia.png` — National ASR trajectories (1990–2023) with 95% UI ribbons.
   - `outputs/figures/s1_age_sex_heatmap_ind_npl.png` — Age × sex DALY rate heatmaps comparing India and Nepal.
   - `outputs/figures/s1_hap_copd_share.png` — Attributable fraction of total COPD and all-cause HAP share.

---

## Stage 2: Forecasting (2024–2040) & Surrogate Calibration
**Scope:** 5-country South Asia panel (`1990–2023` train/CV → `2024–2040` forecast), apportioned to `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural` for Stage 3.  
**Entrypoint:** `src/forecast/run.py` (`make stage2`).

1. **Models Evaluated (100% Python):**
   - **ARIMA:** `statsmodels` SARIMAX/ARIMA per `location × sex` ASR series (order selected by AICc on training window).
   - **ETS:** `statsmodels` exponential smoothing (`ExponentialSmoothing` with damped/linear trend).
   - **Age-Period-Cohort (APC):** `statsmodels` Poisson GLM (`log(deaths or DALYs) ~ age + period + cohort + offset(log(population))`) with identifiability constraint (zero-sum / reference cohort) projected using UN WPP 2024 population; optional simple Bayesian Poisson APC (`pymc` extra) if CPU budget allows (<2 min).
   - **Panel ML Models:** `scikit-learn` `RandomForestRegressor` and `HistGradientBoostingRegressor` trained across the 5-country panel (`5 × 34 = 170` country-years per sex) using clean-fuel access (`WHO SDG 7.1.2`), urban population share, UN demographic dependency ratios (65+ share), and constrained time/lag features.
2. **Rolling-Origin Validation:**
   - Expanding-window panel cross-validation respecting temporal ordering across all 5 countries simultaneously (no future leakage):
     - Fold 1: train `1990–2008`, test `2009–2013` ($h=5$)
     - Fold 2: train `1990–2013`, test `2014–2018` ($h=5$)
     - Fold 3: train `1990–2018`, test `2019–2023` ($h=5$; also report 2019–2022 pre/post-2020 sensitivity)
   - Metrics: **MAE**, **RMSE**, **MASE** (scaled against in-sample naive-1 random walk on each country series).
3. **Uncertainty Propagation & Counts:**
   - Propagate GBD 2023 baseline `(lower, upper)` bounds + model residual uncertainty (500 Monte Carlo draws or quantile scaling) through 2040.
   - Derive projected counts (`Number`) from projected age-sex rates multiplied by **UN WPP 2024** medium-variant population projections.
4. **Urban/Rural Apportionment & Stage 3 Surrogate:**
   - Apportion India and Nepal national forecasts into `{IND_urban, IND_rural, NPL_urban, NPL_rural}` using the documented population- and exposure-weighted formula in `docs/ASSUMPTIONS.md`.
   - Export a fast, closed-form/tabular surrogate (`data/processed/exposure_response_surrogate.json`) mapping clean-fuel coverage $c_{u,t} \in [0, 1]$ to projected DALY rate multiplier $f(c_{u,t})$ so NSGA-II and MDP evaluate in milliseconds.
5. **Outputs:**
   - `outputs/tables/s2_rolling_cv_metrics.csv`
   - `outputs/tables/s2_forecasts_2040_summary.csv`
   - `outputs/figures/s2_model_comparison_cv.png`
   - `outputs/figures/s2_forecasts_2040_ind_npl.png`
   - Hand-off contracts: `data/processed/forecasts_stage3.csv`, `data/processed/baseline_unit_state.csv`, `data/processed/exposure_response_surrogate.json` (see `docs/INTERFACES.md`).

---

## Stage 3: Multi-Objective Optimisation (NSGA-II)
**Scope:** 4 allocation units (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`) × 4 interventions (`lpg_subsidy`, `improved_biomass`, `electric_induction`, `infra_lpg_grid`) × 3 five-year periods (`P1: 2026-2030`, `P2: 2031-2035`, `P3: 2036-2040`).  
**Entrypoint:** `src/optimise/run.py` (`make stage3`).

1. **Decision Vector ($4 \times 4 \times 3 = 48$ variables):**
   - For each `(unit u, intervention i, period p)`: continuous or discrete coverage-increment / budget-share allocation $x_{u,i,p} \ge 0$ (representing new clean/improved cooking adoption share or infrastructure buildout fraction in unit $u$ during period $p$).
2. **Objectives (Minimise all 3):**
   - $f_1(x)$: Projected total HAP-attributable COPD **DALYs in 2040** summed across all 4 units (evaluated via the calibrated surrogate + UN WPP 2040 population).
   - $f_2(x)$: **Total discounted/undiscounted intervention cost** (USD millions) summed across units, interventions, and periods.
   - $f_3(x)$: **Female–male burden gap** in 2040. Primary metric: population-weighted absolute age-standardised DALY rate gap $\sum_u w_u \left|\text{ASR}^{\text{female}}_{u,2040}(x) - \text{ASR}^{\text{male}}_{u,2040}(x)\right|$ (with absolute DALY difference $|\text{DALYs}^F - \text{DALYs}^M|$ logged as sensitivity check; see `docs/RISKS.md`).
3. **Constraints:**
   - **Total budget:** $\text{Cost}(x) \le B_{\text{total}}$ (`docs/ASSUMPTIONS.md`).
   - **Adoption & coverage ceiling:** Total clean-cooking coverage per unit cannot exceed $100\%$ (or feasible ceiling $c^{\max}_u$), and period-on-period adoption increase cannot exceed $\Delta c^{\max}_{u,i,p}$.
   - **Infrastructure prerequisites:** `lpg_subsidy` and `electric_induction` expansion in rural units above baseline capacity require prior or concurrent `infra_lpg_grid` investment in period $\le p$.
4. **Two-Stage Verification (CPU-Fast):**
   - Run `pymoo` `NSGA2` (`pop=80, gen=100`) using the fast vectorized surrogate (`< 10s` on CPU).
   - Re-evaluate representative knee, min-DALY, min-cost, and min-equity Pareto solutions using full Stage 2 uncertainty draws `(lower, val, upper)` to attach 95% UI error bars.
5. **Outputs:**
   - `outputs/tables/s3_pareto_front.csv`
   - `outputs/tables/s3_pareto_reeval.csv`
   - `outputs/figures/s3_pareto_3d_tradeoffs.png`
   - `outputs/figures/s3_allocation_breakdown.png`

---

## Stage 4: Sequential & Symbolic Planning (MDP, HTN, STRIPS/PDDL)
See **[docs/PLANNING_SPEC.md](PLANNING_SPEC.md)** for complete formal specifications.  
**Entrypoint:** `src/planning/run.py` (`make planning`).

- **MDP (Custom NumPy Value Iteration / Finite-Horizon DP):** Adaptive policy solved over discretized coverage, infrastructure, and budget states across `P1–P3`; compared directly on equal footing against GA solutions using identical cost parameters, budget cap, and surrogate response function.
- **HTN (Plain Python Decomposition Tree):** Translates high-level national/unit targets into feasible sub-programmes and primitive actions (`src/planning/htn.py`).
- **STRIPS / PDDL (`pyperplan`):** Validates precondition/effect feasibility of prerequisite infrastructure + rollout sequences (`pddl/domain.pddl`, `pddl/problem_ind_npl.pddl`).
- **Outputs:** `outputs/tables/plan_mdp_vs_ga_comparison.csv`, `outputs/tables/plan_htn_trace.json`, `outputs/tables/plan_pddl_solution.txt`, `outputs/figures/plan_mdp_vs_ga.png`.
