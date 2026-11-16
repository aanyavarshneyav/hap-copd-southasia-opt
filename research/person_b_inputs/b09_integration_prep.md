# B-09: Final Baseline Harmonisation & Decision-Track Integration Preparation

**Branch:** `b/param-sourcing`  
**Status:** Preparation report & evaluator hook ready (`use_mock_forecasts: true` preserved; shared configs, docs, mock fixtures, and Person A files untouched; no commit made).

---

## 1. Separate Population vs. Household Baseline Table

`baseline_clean_fuel_population` (WHO GHO / WB SDG 7.1.2, `people`) and `baseline_clean_fuel_households` (NFHS-5 / Nepal NPHC 2021, `households`) are maintained as separate quantities:
- `baseline_clean_fuel_population` governs exposure-response / PAF calculations and compatibility with Person A's forecasting outputs (Contracts 1–3).
- `baseline_clean_fuel_households` governs additional household adoption (`x_{u,i,p}`), coverage ceiling headroom, and household-denominated fiscal costs.
- Explicit bridge implemented in `src/optimise/evaluator.py` (`Stage3Evaluator`):
  $$\Delta \text{population}_{u} = \text{bridge\_factor}_{u} \times \Delta \text{households}_{u}$$
  with provisional central `bridge_factor = 1.0`, configurable via `coverage_bridge_factor` (tested at `0.8`, `1.0`, `1.2`), and effective population gain capped at remaining population headroom `1.0 - baseline_clean_fuel_population[u]`.

| Unit | `households_2025_derived` | `baseline_clean_fuel_population` (2024, WHO SDG 7.1.2) | `baseline_clean_fuel_households_obs` (Survey/Census) | Obs. HH Source & Year | Observed DHS Spread (HH - Pop) | `household_2024_sensitivity_derived` | `coverage_ceiling_households` | `gap_households_from_obs` | `gap_population_from_2024` | `bau_2040_population_status` |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `IND_urban` | 124,390,773 | 0.96 | 0.8970 | India NFHS-5 (2019–21, DHS `HC_CKFL_H_CLN`, 2020) | +0.007 | 0.967 | 0.99 | 0.0930 | 0.03 | `blocked_person_a (Stage 2 pending)` |
| `IND_rural` | 209,205,395 | 0.69 | 0.4320 | India NFHS-5 (2019–21, DHS `HC_CKFL_H_CLN`, 2020) | +0.023 | 0.713 | 0.95 | 0.5180 | 0.26 | `blocked_person_a (Stage 2 pending)` |
| `NPL_urban` | 4,606,688 | 0.64 | 0.5846 | Nepal NPHC 2021 Table 07 (conventional HH, 2021) | +0.042 | 0.682 | 0.98 | 0.3954 | 0.34 | `blocked_person_a (Stage 2 pending)` |
| `NPL_rural` | 2,166,519 | 0.25 | 0.2036 | Nepal NPHC 2021 Table 07 (conventional HH, 2021) | +0.019 | 0.269 | 0.95 | 0.7464 | 0.70 | `blocked_person_a (Stage 2 pending)` |

*Source:* [`b09_baseline_comparison.csv`](file:///Users/aanyavarshney/Documents/GitHub/hap-copd-southasia-opt/research/person_b_inputs/b09_baseline_comparison.csv).

---

## 2. Reconciled Budget Scenario Comparison (Alternative A Preserved)

All scenarios below follow **Alternative A** (one-time 5-year package per adopting cohort, `apply_cost_discounting: false`, recurring subsidies excluded from base cost and treated as separate sensitivity). Infrastructure (`infra_lpg_grid`) costs remain **excluded as an unresolved gap (`null`)** and are never invented.

| `scenario_id` | Denominator Basis | Intervention Basis (`cost_multiplier`) | `IND_urban` ($M) | `IND_rural` ($M) | `NPL_urban` ($M) | `NPL_rural` ($M) | Full 100% Gap ($M) | 50% Budget ($M) | 25% Budget ($M) | Infra Status | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `H_observed_hh_lpg_fiscal` | `households` | `lpg_subsidy` (1.0) | 1,112.61 | 10,422.53 | 218.30 | 218.03 | **11,971.47** | 5,985.73 | 2,992.87 | `excluded_unresolved_gap (null)` | Primary unadjusted HH observed gap (IND 2020 predates PMUY 2.0). |
| `H_2024_derived_hh_lpg_fiscal` | `households_sensitivity` | `lpg_subsidy` (1.0) | 275.16 | 4,768.61 | 164.53 | 198.93 | **5,407.22** | 2,703.61 | 1,351.81 | `excluded_unresolved_gap (null)` | Sensitivity: WHO 2024 + observed DHS (HH - Pop) spread. |
| `P_who2024_pop_lpg_fiscal` | `people` | `lpg_subsidy` (1.0) | 358.91 | 5,231.38 | 187.72 | 204.48 | **5,982.48** | 2,991.24 | 1,495.62 | `excluded_unresolved_gap (null)` | Population-denominated reference only; do not mix with HH denominator. |
| `H_observed_hh_bio_50pct_cfa` | `households` | `improved_biomass` (0.5) | 260.29 | 2,438.29 | 40.98 | 36.38 | **2,775.94** | 1,387.97 | 693.99 | `excluded_unresolved_gap (null)` | **Scenario assumption:** 50% MNRE/AEPC CFA on $45/HH 5-yr societal ICS cost ($22.50/HH). |
| `H_observed_hh_elec_50pct_hw` | `households` | `electric_induction` (0.5) | 307.20 | 2,877.72 | 57.99 | 51.48 | **3,294.39** | 1,647.19 | 823.60 | `excluded_unresolved_gap (null)` | **Scenario assumption:** 50% public hardware co-finance (India lacks central induction subsidy). |

*Source:* [`b09_budget_scenarios.csv`](file:///Users/aanyavarshney/Documents/GitHub/hap-copd-southasia-opt/research/person_b_inputs/b09_budget_scenarios.csv).

---

## 3. Bridge-Factor Sensitivity Results (`0.8`, `1.0`, `1.2`) & Coverage Feasibility

### 3.1 Dual-Representation Sensitivity (`b09_bridge_sensitivity.csv`)

Run over the mock pipeline via [`run_b09_analysis.py`](file:///Users/aanyavarshney/Documents/GitHub/hap-copd-southasia-opt/research/person_b_inputs/run_b09_analysis.py) comparing `mock_aligned_baselines` and `mock_separated_hh_baseline`:

| `baseline_mode` | `bridge_factor` | `method` | `dalys_2040_mean` | `dalys_averted_vs_bau` | `total_cost_usd_millions` | `equity_gap_asr` | `feasible_hh_rules` | `pop_100pct_overshoot` | `feasible_both_hh_and_pop100` | `n_pareto_solutions` | `hypervolume` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: |
| `mock_aligned_baselines` | 0.8 | `GA_knee` | 4,608,188.34 | 960,359.07 | 120.99 | 42.91 | True | 0.000000 | True | 225 | 25,425,111,795.55 |
| `mock_aligned_baselines` | 0.8 | `GA_cost_minimal` | 5,542,432.17 | 26,115.25 | 5.49 | 60.92 | True | 0.000000 | True | 225 | 25,425,111,795.55 |
| `mock_aligned_baselines` | 0.8 | `GA_daly_minimal` | 3,869,810.14 | 1,698,737.28 | 404.53 | 29.46 | True | 0.000000 | True | 225 | 25,425,111,795.55 |
| `mock_aligned_baselines` | 0.8 | `GA_equity_minimal` | 3,873,376.25 | 1,695,171.17 | 387.95 | 29.43 | True | 0.000000 | True | 225 | 25,425,111,795.55 |
| `mock_aligned_baselines` | 0.8 | `MDP_b6_mean_rollout` | 5,023,526.91 | 545,020.51 | 58.03 | 50.50 | True | 0.000000 | True | 225 | 25,425,111,795.55 |
| `mock_aligned_baselines` | 1.0 | `GA_knee` | 4,399,390.38 | 1,169,157.04 | 120.14 | 39.14 | True | 0.000000 | True | 219 | 33,979,952,627.15 |
| `mock_aligned_baselines` | 1.0 | `GA_cost_minimal` | 5,531,598.72 | 36,948.69 | 5.95 | 60.78 | True | 0.000000 | True | 219 | 33,979,952,627.15 |
| `mock_aligned_baselines` | 1.0 | `GA_daly_minimal` | 3,532,136.77 | 2,036,410.64 | 394.91 | 23.95 | True | 0.000000 | True | 219 | 33,979,952,627.15 |
| `mock_aligned_baselines` | 1.0 | `GA_equity_minimal` | 3,532,136.77 | 2,036,410.64 | 394.91 | 23.95 | True | 0.000000 | True | 219 | 33,979,952,627.15 |
| `mock_aligned_baselines` | 1.0 | `MDP_b6_mean_rollout` | 4,877,901.87 | 690,645.55 | 63.65 | 47.83 | True | 0.000000 | True | 219 | 33,979,952,627.15 |
| `mock_aligned_baselines` | 1.2 | `GA_knee` | 4,244,091.20 | 1,324,456.22 | 117.81 | 36.49 | True | 0.000000 | True | 219 | 40,393,233,000.27 |
| `mock_aligned_baselines` | 1.2 | `GA_cost_minimal` | 5,519,671.59 | 48,875.83 | 6.29 | 60.54 | True | 0.000000 | True | 219 | 40,393,233,000.27 |
| `mock_aligned_baselines` | 1.2 | `GA_daly_minimal` | 3,342,062.77 | 2,226,484.64 | 387.51 | 21.00 | True | 0.000000 | True | 219 | 40,393,233,000.27 |
| `mock_aligned_baselines` | 1.2 | `GA_equity_minimal` | 3,342,062.77 | 2,226,484.64 | 387.51 | 21.00 | True | 0.000000 | True | 219 | 40,393,233,000.27 |
| `mock_aligned_baselines` | 1.2 | `MDP_b6_mean_rollout` | 4,759,196.76 | 809,350.65 | 68.29 | 45.86 | True | 0.000000 | True | 219 | 40,393,233,000.27 |
| `mock_separated_hh_baseline` | 0.8 | `GA_knee` | 4,665,144.46 | 903,402.95 | 109.74 | 43.96 | True | 0.000000 | True | 230 | 24,223,960,035.92 |
| `mock_separated_hh_baseline` | 0.8 | `GA_cost_minimal` | 5,532,678.74 | 35,868.67 | 7.73 | 60.80 | True | 0.000000 | True | 230 | 24,223,960,035.92 |
| `mock_separated_hh_baseline` | 0.8 | `GA_daly_minimal` | 3,944,102.74 | 1,624,444.68 | 356.76 | 30.73 | True | 0.000000 | True | 230 | 24,223,960,035.92 |
| `mock_separated_hh_baseline` | 0.8 | `GA_equity_minimal` | 3,944,102.74 | 1,624,444.68 | 356.76 | 30.73 | True | 0.000000 | True | 230 | 24,223,960,035.92 |
| `mock_separated_hh_baseline` | 0.8 | `MDP_b6_mean_rollout` | 5,051,892.81 | 516,654.61 | 54.70 | 51.04 | True | 0.000000 | True | 230 | 24,223,960,035.92 |
| `mock_separated_hh_baseline` | 1.0 | `GA_knee` | 4,448,190.18 | 1,120,357.23 | 116.34 | 40.02 | True | 0.000000 | True | 216 | 32,232,045,492.55 |
| `mock_separated_hh_baseline` | 1.0 | `GA_cost_minimal` | 5,529,272.38 | 39,275.04 | 5.59 | 60.67 | True | 0.000000 | True | 216 | 32,232,045,492.55 |
| `mock_separated_hh_baseline` | 1.0 | `GA_daly_minimal` | 3,637,337.88 | 1,931,209.54 | 375.45 | 25.56 | True | 0.000000 | True | 216 | 32,232,045,492.55 |
| `mock_separated_hh_baseline` | 1.0 | `GA_equity_minimal` | 3,637,337.88 | 1,931,209.54 | 375.45 | 25.56 | True | 0.000000 | True | 216 | 32,232,045,492.55 |
| `mock_separated_hh_baseline` | 1.0 | `MDP_b6_mean_rollout` | 4,911,629.13 | 656,918.29 | 60.50 | 48.47 | True | 0.000000 | True | 216 | 32,232,045,492.55 |
| `mock_separated_hh_baseline` | 1.2 | `GA_knee` | 4,267,365.76 | 1,301,181.66 | 111.34 | 36.75 | True | 0.000000 | True | 212 | 40,458,908,570.27 |
| `mock_separated_hh_baseline` | 1.2 | `GA_cost_minimal` | 5,515,415.30 | 53,132.11 | 6.45 | 60.41 | True | 0.000000 | True | 212 | 40,458,908,570.27 |
| `mock_separated_hh_baseline` | 1.2 | `GA_daly_minimal` | 3,343,539.49 | 2,225,007.93 | 365.00 | 21.01 | True | 0.000000 | True | 212 | 40,458,908,570.27 |
| `mock_separated_hh_baseline` | 1.2 | `GA_equity_minimal` | 3,343,539.49 | 2,225,007.93 | 365.00 | 21.01 | True | 0.000000 | True | 212 | 40,458,908,570.27 |
| `mock_separated_hh_baseline` | 1.2 | `MDP_b6_mean_rollout` | 4,768,411.03 | 800,136.39 | 66.34 | 45.97 | True | 0.000000 | True | 212 | 40,458,908,570.27 |

**Key takeaway:** With joint `effective_adoption_headroom()` (`min(coverage_ceiling - hh_base, (1.0 - pop_base) / bridge_factor)`) enforced identically in `Stage3Evaluator`, `Stage3Repair`, and `build_layered_mdp`, every inspected GA and MDP policy across `bridge_factor` in `{0.8, 1.0, 1.2}` satisfies both household ceiling and population saturation bounds (`pop_100pct_overshoot = 0.000000`, `feasible_both_hh_and_pop100 = True`), preventing expenditure past population saturation.

### 3.2 Coverage & Structural Feasibility Audit (`b09_feasibility_audit.csv`)

| Unit | `infra_ready_init` | `hh_obs` | `pop_2024` | `ceiling` | `gap_hh_obs` | `gap_pop_2024` | `lpg_unlocked_cap_sum` | `elec_unlocked_cap_sum` | `bio_cap_sum` | `combined_unlocked_cap_sum` | LPG closes HH / Pop gap | Combined closes HH / Pop gap | Key Constraint / Risk |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | --- |
| `IND_urban` | True | 0.8970 | 0.96 | 0.99 | 0.0930 | 0.03 | 0.15 | 0.40 | 0.60 | 1.15 | True / True | True / True | **CRITICAL:** 2024 pop=0.96 leaves only `0.03` margin under 0.99 ceiling; 2040 BAU may reach ~0.98–0.99, so verify `ceiling >= cf_bau_2040`. |
| `IND_rural` | False | 0.4320 | 0.69 | 0.95 | 0.5180 | 0.26 | 0.44 | 0.30 | 0.60 | 1.34 | False / True | True / True | Unlocked P2+P3 (or P1–P3) caps close 2024 pop gap; strictly earlier `q < p` blocks P1 LPG/Elec without initial readiness. |
| `NPL_urban` | True | 0.5846 | 0.64 | 0.98 | 0.3954 | 0.34 | 0.45 | 0.40 | 0.60 | 1.45 | True / True | True / True | Unlocked P2+P3 / P1–P3 caps close both HH and 2024 pop gaps. |
| `NPL_rural` | False | 0.2036 | 0.25 | 0.95 | 0.7464 | 0.70 | 0.30 | 0.30 | 0.60 | 1.20 | False / False | True / True | **CRITICAL:** `q < p` blocks P1 LPG/Elec; unlocked P2+P3 LPG+Elec = `0.60 < 0.70` (pop) & `< 0.7464` (HH), requiring P1 `improved_biomass`. |

*Source:* [`b09_feasibility_audit.csv`](file:///Users/aanyavarshney/Documents/GitHub/hap-copd-southasia-opt/research/person_b_inputs/b09_feasibility_audit.csv).

---

## 4. Proposed Interface, Config & Register Diffs (Draft — Not Applied)

> None of the shared files below were modified on disk. Changes require Person A / team approval before landing.

### 4.1 Proposed Addition to `docs/INTERFACES.md` (Contract 4 & Cross-Contract Clarification)
```diff
--- a/docs/INTERFACES.md
+++ b/docs/INTERFACES.md
@@ Contract 2 / Contract 3 vs Contract 4 Denomination
+Population vs. Household Coverage Separation:
+- Contract 2 (`data/processed/baseline_unit_state.csv`) and Contract 1 (`cf_bau_2040`)
+  are population-based clean-fuel access rates (`WHO SDG 7.1.2`, denominator: people),
+  used in Contract 3 (`exposure_response_surrogate.json`) PAF/exposure-response calculations.
+- Contract 4 (`configs/intervention_params.json`) MAY supply:
+  - `baseline_clean_fuel_households`: dict[unit, float] (optional; defaults to Contract 2
+    population baseline if omitted) — household-based clean-fuel access (NFHS-5 / Nepal NPHC 2021),
+    used for household adoption headroom, ceiling feasibility, and household-denominated costing.
+  - `coverage_bridge_factor`: dict[unit, float] (optional; default 1.0 per unit) — explicit bridge
+    `delta_population = bridge_factor[unit] * delta_households`, capped at `1.0 - baseline_clean_fuel_population`.
```

### 4.2 Proposed Addition to `configs/default.yaml` (Sensitivity Configuration Only)
```diff
--- a/configs/default.yaml
+++ b/configs/default.yaml
@@ stage3
   apply_cost_discounting: false
+  coverage_bridge_factor_default: 1.0
+  coverage_bridge_sensitivity: [0.8, 1.0, 1.2]
```

### 4.3 Proposed Updates to `docs/ASSUMPTIONS.md`
```diff
--- a/docs/ASSUMPTIONS.md
+++ b/docs/ASSUMPTIONS.md
+| ASM-04..ASM-09 (Unit Costs & Caps) | Sourced/derived USD-per-HH-to-USD-M tables staged in `research/person_b_inputs/proposed_contract4_table.csv` (fiscal public expenditure perspective; `infra_lpg_grid` remains `null` / `placeholder: needs literature`; biomass 50% CFA and electric induction 50% HW subsidy rates marked `scenario assumption`). |
+| ASM-17 (new) | Constant average household size 2025–2040 per unit (from `demographic_inputs.csv`), held fixed unless population/household-size sensitivity is activated. |
+| ASM-18 (new) | Separated baselines & provisional `coverage_bridge_factor = 1.0` linking additional household adoption to population clean-fuel access (`0.8, 1.0, 1.2` sensitivity). |
```

### 4.4 Proposed Decision Log Entries for `docs/DECISIONS.md`
- **DEC-022:** Separate `baseline_clean_fuel_population` (exposure-response / Contract 1–3 compatibility) from `baseline_clean_fuel_households` (adoption headroom, ceiling feasibility, household costing), linked by configurable `coverage_bridge_factor` (provisional central `1.0`).
- **DEC-023:** Preserve Alternative A one-time 5-year package fiscal costing (`apply_cost_discounting: false` per `DEC-014`), discounting MDP/comparator trajectories at most once, and isolate recurring LPG/electricity operating subsidies plus unverified biomass/induction subsidy splits as separate sensitivity runs.
- **DEC-024:** Exclude `infra_lpg_grid` cost gaps as unresolved (`null`) rather than inventing numbers until primary infrastructure expenditure estimates are sourced or approved as scenario bounds.

---

## 5. Checklist of Exact Files Expected from Person A

1. **Contract 1:** `data/processed/forecasts_stage3.csv` (including projected `cf_bau_2040` in population units `WHO SDG 7.1.2` alongside mean/lower/upper COPD burden trajectories).
2. **Contract 2:** `data/processed/baseline_unit_state.csv` (2024/2025 unit baselines, `pop_2025`, ASR/exposure baseline).
3. **Contract 3:** `data/processed/exposure_response_surrogate.json` (surrogate coefficients/tables keyed on population clean-fuel coverage).
4. **Integration Check from Stage 1–2:** Confirmation that `IND_urban` `cf_bau_2040 <= coverage_ceiling (0.99)` (since 2024 population access is already `0.96`) and alignment on whether a 2025→2040 BAU household baseline projection accompanies `cf_bau_2040`.

---

## 6. Remaining Decisions Requiring Human Approval

1. **`infra_lpg_grid` Unresolved Gap:** Approve either sourcing an institutional grid/bottling expansion cost per rural unit or bounding `infra_lpg_grid` explicitly as a scenario parameter (currently `null`).
2. **Primary Costing & Subsidy Scenario Approval:** Sign off on fiscal public-expenditure costing with `50%` biomass CFA (`$2,775.94M` full gap) and `50%` electric induction hardware co-finance (`$3,294.39M` full gap) flagged as **scenario assumptions**.
3. **India Rural Household Baseline Choice:** Decide whether production runs use observed `NFHS-5 2020` (`0.4320`, `$10,422.53M` IND_rural full LPG gap / `$11,971.47M` total) or the `2024` sensitivity-adjusted household baseline (`0.7130`, `$4,768.61M` IND_rural / `$5,407.22M` total) or a 2040 BAU household series from Person A.
4. **`coverage_bridge_factor` & Dual-Feasibility Rule:** Approve central `bridge_factor = 1.0` and whether `bridge_factor > 1.0` runs should enforce `delta_households * bridge_factor <= 1.0 - baseline_clean_fuel_population` as a hard constraint or rely on evaluator headroom capping (`pop_100pct_overshoot` audit).
5. **Production Switch & Contract Diffs:** Approve applying the `docs/INTERFACES.md`, `configs/default.yaml`, `docs/ASSUMPTIONS.md`, and `docs/DECISIONS.md` updates and flipping `use_mock_forecasts: false` once Person A lands Contracts 1–3.
