# Executive Summary: Forecasting and Optimal Intervention Planning for HAP-Attributable COPD Burden in South Asia (1990–2040)

**Authors:** Rudraksh (Stage 0–2 Data, Historical Epidemiology & Forecasting) and Aanya Varshney (Stage 3–4 Multi-Objective Optimisation, Sequential/Symbolic AI Planning, Parameter Sourcing, Sex Reconciliation & Interactive Website)  
**Repository Branch:** `b/b10-final-integration` | **Epidemiological Baseline:** `GBD_2023_WPP_2024` (Sex-Reconciled Derived Contracts) | **Parameter Baseline:** `REAL_V1_2024USD`  

---

## Core Project Findings

1. **Historical Epidemiology (`1990–2023`, Stage 1):**
   Across all five South Asian nations in GBD 2023, age-standardized HAP–COPD DALY rates fell substantially over 1990–2023: Bhutan **−94.15%** (`1,431.89 → 83.79` per 100k; EAPC `−7.32%/yr`), India **−56.35%** (`1,360.47 → 593.89`; EAPC `−2.54%/yr`), Pakistan **−43.96%** (`1,023.64 → 573.69`; EAPC `−1.64%/yr`), Nepal **−39.96%** (`1,721.87 → 1,033.86`; EAPC `−1.56%/yr`), and Bangladesh **−32.17%** (`1,964.80 → 1,332.71`; EAPC `−1.35%/yr`). However, demographic growth and ageing caused **absolute DALY counts** to rise by **+24.82%** in India (`5.58M → 6.96M`) and **+75.67%** in Nepal (`144.6k → 254.0k`). Structural inflection (`pwlf`) accelerated declines around **2008.4** in `IND_urban` (`−1.70% → −12.27%/yr`), **2010.7** in `IND_rural` (`−0.35% → −4.53%/yr`), and **2014.6** in `NPL_rural` (`−0.79% → −3.10%/yr`).

2. **2040 Projections, Forecasting Validation (B-12) & Replacement of HistGBDT Surrogate with Smooth Log-OLS (B-13):**
   Fixed ETS (`47.73`/`48.78` MAE; `+53.9%`/`+52.8%` skill) and ARIMA (`49.33`; `+52.3%`) dominate short-term historical CV (`0` fit failures across `891` evaluations), while reconciled 2040 BAU burden (`forecasts_stage3_reconciled.csv`) stays unchanged at **`2,699,837.26` DALYs** (exact `float64` sex reconciliation via WHO GHE 2021; `31.41/100k` equity gap). Across identical 4-unit counterfactual validation folds (`h = 1, 3, 5, 10`), `HistGradientBoosting` showed inferior error (`151.00` `h=5` MAE vs. **`102.34`** for `log_ols_smooth`), coarse leaf cliffs collapsing all units to `124.08/100k` at `cf = 1.00`, and an artificial `0.0` `IND_urban` plateau. Production Contract 3 (`exposure_response_surrogate.json`) therefore replaces `HistGBDT` (preserved at `exposure_response_surrogate_histgbdt.json`) with **pooled Smooth Log-OLS** (`slope = −2.1933 ± 0.2557 SE`, `R^2 = 0.8938`, `p = 2.51e-14`) across all four intervention units; see [`B13_SURROGATE_COMPARISON_REPORT.md`](../B13_SURROGATE_COMPARISON_REPORT.md).

3. **Multi-Objective Intervention Optimisation (`NSGA-II`, Stage 3, Scenario A Base — Regenerated on Log-OLS Surrogate):**
   Under the real 50% fiscal clean-cooking budget cap (**\$5,985.73M**) and **Scenario A** (`infra_lpg_grid: null`), 5-seed `NSGA-II` finds **251 Pareto-optimal feasible plans** (combined HV **`2.7965e+10`**, HistGBDT baseline `265` pts preserved), beating all three baselines:
   - **Cost-Minimal (`#0`):** **\$70.75M** → **59,359 DALYs averted** (`2,640,478` DALYs; equity gap `30.42/100k`).
   - **Knee Compromise (`#156`):** **\$1,695.91M** → **311,707 DALYs averted** (`2,388,130` DALYs; joint 500-draw mean `2,401,031` `[1,974,879, 2,799,565]`; equity gap `26.85/100k`).
   - **DALY-Minimal = Equity-Minimal (`#250`):** **\$3,097.29M** → **453,474 DALYs averted** (`16.8%` reduction to `2,246,364` DALYs; joint 500-draw mean `2,258,315` `[1,868,489, 2,631,221]`; equity gap `24.91/100k`), saturating below cap due to 2040-BAU headroom limits.

4. **Sequential & Symbolic Planning (`MDP`, `HTN`, `PDDL`, Stage 4 — Regenerated):**
   Against the updated Log-OLS Pareto front (median slope `125.5` DALYs/\$1M), Finite-Horizon MDP (`λ = 116.0`, `22,136` states) achieves **0.0% post-clamp constraint violations** across 1,000 CRN shock rollouts (**127,463** mean averted DALYs at **\$600.35M** mean spend; scalarized value `46,750`). Pre-clamp auditing (`comp_mdp_clamp_audit_b14.csv`) shows MDP attempts `1.25×` cap/ceiling overshoots on **40.0%** of rollouts before `_apply_action_branch` repairs them (`132,017 → 127,463` averted DALYs); under symmetric execution clamping, open-loop GA achieves **100.0% feasibility** and higher averted DALYs (`237.5k` matched; `294.2k` Knee; `426.0k` DALY-Min) vs. `36.5%–40.0%` unclamped violations (`pddl_classical_plan`: `410,256` averted at **\$3,262.52M**, **26.9%** unclamped violations; HTN: **9 leaves**, **\$807.36M**).

5. **Infrastructure & Operating Cost Sensitivities (18 Real Scenarios + Alt B):**
   Enabling regional infra builds in **Scenario B1 / B2** unlocks **+59,562 / +59,383 additional averted DALYs** (`513,035` / `512,856` vs. `453,474` in Scenario A). Switching to recurring P1–P3 LPG refill opex (**Alt B**) increases Knee cost by **+49.3%** (`\$1,695.91M → \$2,531.41M`) and DALY-Minimal cost by **+67.7%** (`\$3,097.29M → \$5,192.82M`), fitting within the \$5,985.73M cap.

**Framing Notice:** All GBD 2023 figures are modelled estimates; sex-specific counts and rates are WHO GHE 2021 proxy-reconciled allocations; sex-specific intervention responsiveness uses a uniform proxy elasticity (`r_fm = 1.05`); and intervention allocations represent academic scenario analyses rather than prescriptive government budgets.

See the full technical report in [`report/report.md`](report.md) and verification dossier in [`FINAL_QA_REPORT.md`](../FINAL_QA_REPORT.md).
