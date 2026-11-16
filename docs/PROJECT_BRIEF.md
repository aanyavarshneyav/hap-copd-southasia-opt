# Project Brief

**Title:** Forecasting and Optimal Intervention Planning for COPD Burden Attributable to Household Air Pollution in South Asia: A hybrid predictive–optimisation approach using Global Burden of Disease 2023 data  
**Group:** Rudraksh (Person A), Aanya (Person B) · University AI Lab project

---

## 1. Problem Statement
Household air pollution (HAP) from solid cooking fuels is a major environmental risk factor for chronic obstructive pulmonary disease (COPD), and South Asia carries a large share of the global burden. India and Nepal differ widely in population size, demographic structure, and progress towards clean cooking. Existing GBD analyses mostly describe past burden and extrapolate trends. They rarely answer the decision question: with a limited budget, which clean-cooking interventions should be prioritised, where, and for whom, to avert the most disease burden by 2040?

This project forecasts HAP-attributable COPD burden to 2040 using statistical and machine-learning models, then uses a genetic algorithm and formal planning to determine how a limited clean-cooking budget should be allocated across locations, interventions, and time periods.

## 2. Research Questions
1. How have HAP-attributable COPD deaths and DALYs changed in India, Nepal, and wider South Asia from 1990 to 2023, by age and sex?
2. Which forecasting approach (statistical or ML) projects attributable burden to 2040 most reliably on held-out data?
3. Under a fixed budget, which intervention allocation minimises projected 2040 burden, and how does it trade off against cost and sex-based equity?
4. How does an adaptive planning policy (MDP) compare with a one-off optimised allocation (GA)?

## 3. Data
- **GBD 2023 (IHME Results Tool):** HAP-attributable COPD deaths and DALYs (number, rate, percent), all age groups, both sexes, 1990–2023, for India, Nepal, and South Asian peers (GBD South Asia region: India, Nepal, Pakistan, Bangladesh, Bhutan); total COPD for the same locations; HAP-attributable burden across all linked causes.
- **UN World Population Prospects 2024:** Population by age and sex, with projections through 2040.
- **WHO clean cooking fuel access (SDG 7.1.2):** National, urban, and rural, including by fuel type.
- **Costs and intervention effectiveness:** Taken from published literature and programme reports, treated as scenario assumptions with sensitivity ranges (tracked in `docs/ASSUMPTIONS.md`).

## 4. Methodology
- **Stage 1 — Historical analysis (1990–2023):** Attributable deaths, DALYs, and age-standardised rates; estimated annual percentage change (EAPC) as minimum deliverable and piecewise/joinpoint trends (`pwlf`); attributable share of total COPD; age–sex heatmaps; India versus Nepal comparison across the 5-country South Asia panel.
- **Stage 2 — Forecasting (2024–2040):** ARIMA, ETS, and Poisson GLM age-period-cohort (APC) baselines (or simple Bayesian APC in Python), plus a gradient-boosting or random-forest model trained on the 5-country South Asia panel using clean-fuel access as a driver. Models are compared with rolling-origin validation (MAE, RMSE, MASE). GBD uncertainty intervals are propagated into the forecasts, and counts are derived with UN population projections.
- **Stage 3 — Optimisation (Genetic Algorithm):** NSGA-II (`pymoo`) allocates a budget across `location (India/Nepal × urban/rural = 4 units) × intervention (LPG subsidy, improved biomass stoves, electric/induction cooking, LPG/grid infrastructure prerequisite) × period (2026–2030, 2031–2035, 2036–2040)`.
  - **Objectives:** Minimise projected attributable DALYs in 2040, minimise total cost, and minimise the female–male burden gap.
  - **Constraints:** Total budget, maximum feasible adoption rate, and infrastructure prerequisites.
  - **Fitness evaluation:** Evaluated using a CPU-fast surrogate/response function calibrated from the Stage 2 forecasting model and exposure–response assumptions, with top Pareto solutions cross-checked on the full model.

## 5. Planning Component
| Technique | Implementation | Role in the Project |
| --- | --- | --- |
| **Markov Decision Process (MDP)** | Custom NumPy value iteration | Adaptive investment policy. States: discretized clean-fuel coverage per unit, infrastructure status, and remaining budget tier. Actions: investment choices per 5-year period. Reward: DALYs averted minus cost penalty. Compared with the GA plan on equal footing. |
| **Hierarchical Task Network (HTN)** | Documented decomposition tree in plain Python | Decomposes the national/regional clean-cooking goal into urban/rural programmes, prerequisite infrastructure tasks, and concrete intervention rollouts. |
| **STRIPS / PDDL** | Classical STRIPS encoded for `pyperplan` | Encodes intervention actions with discrete preconditions and effects (e.g., LPG subsidy requires distribution infrastructure and available budget tier). Generates valid sequential action plans. |

## 6. Expected Outputs
- Historical trend and age–sex visualisations (`outputs/figures/s1_*`).
- Forecasting model comparison table and 2040 projections with uncertainty intervals (`outputs/tables/s2_*`, `outputs/figures/s2_*`).
- Pareto front of allocation plans (`outputs/tables/s3_*`, `outputs/figures/s3_*`).
- MDP versus GA equal-footing comparison (`outputs/tables/plan_mdp_vs_ga_comparison.csv`, `outputs/figures/plan_*`).
- PDDL domain and problem files (`pddl/*.pddl`) and planner output (`outputs/tables/plan_pddl_solution.txt`).
- Reproducible 100% Python codebase and final report (`report/`).

## 7. Novelty and Limitations
- **Contribution:** This is not the first study of HAP-attributable COPD. The contribution is the decision layer: moving from describing burden to prescribing allocation by combining forecasting, multi-objective genetic optimisation, MDP planning, and classical planning, with uncertainty carried through.
- **Limitations:**
  - GBD values are modelled estimates, not observed counts.
  - Intervention costs and effects are scenario assumptions, so results are scenario analyses rather than policy recommendations.
  - Forecasts assume historical structural patterns continue (with clean-fuel access as primary policy lever).
  - GBD 2023 exposure data may not be directly downloadable; WHO SDG 7.1.2 clean-fuel access serves as the exposure driver fallback.
  - Allocation is performed at national urban/rural granularity (4 units across India and Nepal) rather than sub-national states/provinces, requiring documented burden apportionment.
