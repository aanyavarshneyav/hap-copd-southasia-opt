# B-08: Real-Input Consistency and Cost-Model Readiness Report

> **Scope & Isolation:** Read-only audit of `research/person_b_inputs/`, `src/optimise/evaluator.py`, `src/planning/mdp.py`, `src/planning/compare.py`, `docs/`, and `configs/default.yaml` on branch `b/param-sourcing`. No mock fixtures, shared configs, Person A files, or shared contracts were modified, and nothing was committed.

---

## 1. Household vs. Population Coverage (Task 1)

### 1.1 Four-Unit Coverage Comparison

| Unit | WHO / WB SDG 7.1.2 (2024, `people`) | DHS / NFHS Household (`households`) | DHS / NFHS Population (`people`) | Nepal Census 2021 Table 07 (`households`) | Gap (`HH - Pop` in DHS) |
| --- | --- | --- | --- | --- | --- |
| `IND_urban` | **96.0%** `[88–100]` | **89.7%** (NFHS-5, 2020) | 89.0% (NFHS-5, 2020) | — | `+0.7` pct pts |
| `IND_rural` | **69.0%** `[53–82]` | **43.2%** (NFHS-5, 2020) | 40.9% (NFHS-5, 2020) | — | `+2.3` pct pts |
| `NPL_urban` | **64.0%** `[34–93]` | **56.7%** (NDHS, 2022) | 52.5% (NDHS, 2022) | **58.46%** (`4,474,699` conv. HH; LPG `56.69%`, Elec `0.55%`, Biogas `1.22%`) | `+4.2` pct pts |
| `NPL_rural` | **25.0%** `[14–40]` | **21.2%** (NDHS, 2022) | 19.3% (NDHS, 2022) | **20.36%** (`2,186,142` conv. HH; LPG `18.91%`, Elec `0.36%`, Biogas `1.10%`) | `+1.9` pct pts |

*Notes:*
1. WHO GHO (`PHE_HHAIR_PROP_POP_CLEAN_FUELS`) and World Bank WDI (`EG.CFT.ACCS.*`) match identically (`0.0` pct pt diff, `2000–2024`, denominator = total/urban/rural population).
2. In every unit, household-based clean-fuel coverage exceeds population-based coverage by `+0.7` to `+4.2` pct pts because larger households disproportionately rely on solid fuels.
3. For `IND_rural`, WHO 2024 (`69.0%`) is much higher than NFHS-5 2020 (`43.2%` HH / `40.9%` pop) because NFHS-5 fieldwork predates 2021–2024 PMUY 2.0 growth (WHO 2020 was `51.0%` `[39–63]`).

### 1.2 Baseline & Contract 3 Coordination Recommendation
- **Do NOT silently substitute population access for household access.**
- **Contract 1–3 Exposure Series (`forecasts_stage3.csv`, `exposure_response_surrogate.json`):** Keep **WHO SDG 7.1.2 (`people`-denominated, 2000–2024)** as the Stage 2 covariate and `cf_bau_2040` baseline (`ASM-02`, `DEC-007`), because GBD 2023 COPD DALYs and `slope_clean_fuel` are person-level epidemiological quantities and WHO provides the only continuous annual panel (`2000–2024`) across all 5 South Asia countries.
- **Stage 3 Intervention Baseline & Bridge:** Because `Stage3Evaluator` evaluates `delta_c = eff_gain_u` directly above Contract 1's `cf_bau_2040` (`evaluator.py` L318–323) while costing interventions per household $H_u$, keep `clean_fuel_share` in Contracts 1–3 `people`-denominated and adopt an explicit **1:1 marginal household-to-population share bridge** ($\Delta c_{\text{pop}} \approx \Delta c_{\text{HH}}$) for Stage 3 increments, while logging Nepal NPHC 2021 (`58.46%` / `20.36%`) and India NFHS-5 (`89.7%` / `43.2%`) as observed household-denominated reference benchmarks and testing `±2–4` pct pt denominator sensitivity.

---

## 2. Household Count Methodology (Task 2)

| Unit | Observed Census Pop & HH | Observed Survey HH Size | WUP 2025 NatDef Pop | Operational 2025 Derived HH | 2040 Derived HH (Const. Size) | Sensitivity (`NDHS 2022` / `2033 Midpoint`) |
| --- | --- | --- | --- | --- | --- | --- |
| `IND_urban` | 2011 A-1: `377,106,125` / `80,888,766` (`4.662`) | NFHS-5 (2020): **4.2** | `522,441,248` (`35.69%`) | **124,390,773** | `155,344,653` (`+24.9%`) | 2033 mid: `~141.4M` (`+13.7%`) |
| `IND_rural` | 2011 A-1: `833,748,852` / `168,612,897` (`4.945`) | NFHS-5 (2020): **4.5** | `941,424,277` (`64.31%`) | **209,205,395** | `215,584,999` (`+3.1%`) | 2033 mid: `~214.1M` (`+2.3%`) |
| `NPL_urban` | 2021 T15: `19,296,788` / `4,479,662` (`4.3076` / `4.31`) | NDHS (2022): `3.9` *(de jure)* | `19,843,969` (`67.00%`) | **4,606,688** | `5,349,639` (`+16.1%`) | NDHS 3.9: `5,088,197` (`+10.5%`) |
| `NPL_rural` | 2021 T15: `9,867,790` / `2,187,275` (`4.5115` / `4.51`) | NDHS (2022): `4.0` *(de jure)* | `9,774,149` (`33.00%`) | **2,166,519** | `2,186,701` (`+0.9%`) | NDHS 4.0: `2,443,537` (`+12.8%`) |

- **Urban/Rural Classification Consistency:** All operational estimates pair **WUP 2025 National Definitions** (not DEGURBA Level 1) with national Census/DHS strata. Nepal's 2021 municipal structure (`66.17%` urban in NPHC 2021 vs `67.00%` in WUP 2025 NatDef) and India's statutory/census-town frame (`31.16%` urban in 2011 vs `35.69%` in WUP 2025 NatDef) align cleanly.
- **Constant Household Size Assumption:** Holding $\bar{s}_u$ constant at `4.2 / 4.5 / 4.3076 / 4.5115` through 2040 is a **conservative lower bound** on future household counts (`IND` fell `4.6/4.9 -> 4.2/4.5` from 2006 to 2020; `NPL` fell `4.88 -> 4.37` from 2011 to 2021).
- **Operational Recommendation:** Use the **2025 Derived Households** (`124,390,773`, `209,205,395`, `4,606,688`, `2,166,519`) as the operational `t_0 = 2025` Contract 4 denominators (matching Contract 2's `t_0 = 2025` anchor), and test `+12%` `IND_urban` midpoint growth sensitivity.

---

## 3. Cost Accounting Consistency (Task 3)

### 3.1 How Current Code Charges Costs
- **`Stage3Evaluator` (`src/optimise/evaluator.py` L389–402):** Charges `sum(x_cook * unit_costs * period_discount) + sum(I(x_infra >= threshold) * infra_cost * period_discount)`. Every adopting cohort `x[u, i, p]` is charged **once** in period `p`.
- **`LayeredMDP` (`src/planning/mdp.py` L263–296, L586–587, L829–831):** Uses the exact same one-time cost formula (`steps * step_u * non_infra_costs * evaluator._period_discount[t]`).
- **`compare.py` (`src/planning/compare.py` L86–102, L123, L142–143):** Charges `planned_cost_t` (committed open-loop spend) once per period, matching `Stage3Evaluator` on nominal branches (`realized_cost_if_pay_as_go` logged separately).
- **Double-Discounting Code Guard:** When `apply_cost_discounting: false` (`DEC-014`), `evaluator._period_discount[t] = 1.0`, so `mdp.py` and `compare.py` discount period-`t` cost **once** by `gamma**t` (`0.774**t`). If `apply_cost_discounting` were set to `true`, `mdp.py`/`compare.py` would discount `cost` by `gamma**(2t)` (`0.774**(2t)`). Keep `apply_cost_discounting: false`.

### 3.2 Alternative A vs. Alternative B Evaluation

| Dimension | Alternative A: One-Time 5-Yr Cohort Package (Current Code) | Alternative B: Recurring Opex Charged for Every Active Period $q \ge p$ |
| --- | --- | --- |
| Budget math (`P1` adopter) | `1 × C_5yr` (`$96.18/HH` IND LPG) | `C_capex + 3 × C_opex_5yr` (`$239.55/HH` IND LPG = **2.49×**; `$337.08/HH` NPL rural = **2.50×**) |
| Timing sensitivity (`P1` vs `P3`) | Cost-neutral (`apply_cost_discounting: false`); `P1` and `P3` differ only by infra prerequisites | `P1` adoption costs **~2.5× more** than `P3` adoption for **identical** 2040 terminal DALYs |
| Interaction with `f_1(x)` (2040 DALYs) | **Consistent** — `f_1` measures 2040 terminal state, not cumulative 2026–2040 DALYs | **Severe distortion** — NSGA-II/MDP will defer all cooking to `P3` unless `f_1` is rewritten to cumulative 2026–2040 DALYs |

- **Recommendation:** **Retain Alternative A** across `Stage3Evaluator`, `MDP`, and `compare.py` (do not modify evaluator). Define all unit costs as **one-time 5-year transitional cohort packages**:
  - `lpg_subsidy`: Upfront connection + 5-year transitional refill subsidy (1 cohort package, assuming household graduation after 5 yrs; report multi-period fiscal continuation sensitivity `+2× opex` on `P1` cohorts separately).
  - `improved_biomass`: 5-year hardware + 1.5× replacement + 5%/yr maintenance bundled per cohort.
  - `electric_induction`: Hardware + 15A internal wiring (+ optional 5-yr tariff support) bundled per cohort.

---

## 4. Harmonised Intervention Cost Perspective (Task 4)

| Intervention | Gov Capex | Gov Recurring (5-yr) | HH / Societal Capex | HH / Societal Recurring (5-yr) | Missing / Unresolved Components |
| --- | --- | --- | --- | --- | --- |
| `lpg_subsidy` (IND) | `$24.49/HH` (`₹2,050`) | `$71.69/HH` (`4 refills × ₹300 × 5`) | Included in PMUY / OMC | Unsubsidized refills (`~₹503–603/cyl`, not sourced) | Household unsubsidized refill outlay for full societal view |
| `lpg_subsidy` (NPL) | `$0` (no PMUY grant) | `$101.12/HH` (`6 cyl × NPR 450 NOC × 5`) | `$18.73–$33.71/HH` (assumed deposit/kit) | Retail LPG cylinder outlay | Formal upfront connection grant does not exist in Nepal |
| `improved_biomass` | 50% CFA (`~$22.50/HH`; MNRE `₹300–800`/stove hist.) | `$0` | `$25–$85/HH` societal (`$45` central w/ repl. + 5% O&M) | Included in `$45/HH` package | None for hardware; biomass collection time not monetized |
| `electric_induction` (IND) | **`$0` central cash subsidy** (EESL NECP bulk retail) | State tariff cross-subsidy (unquantified) | `$22.70–$101.43/HH` (`$53.11` central hw+wiring) | Domestic electricity tariff (not sourced) | **No Indian central cash subsidy scheme**; 5-yr kWh cost missing |
| `electric_induction` (NPL) | 20–50% AEPC/GCF co-finance (`~$22–$45/HH`) | Optional NEA 20% discount (`$52–$112/HH`) | `$37.45–$89.89/HH` (`$63.67` central stove + 15A) | Electricity tariff (not sourced) | Full societal kWh generation/tariff cost |
| `infra_lpg_grid` | **Unresolved gap (`null`)** | — | — | — | **Regional 2 kW feeder/transformer upgrades & rural LPG depot/bottling capex** (`Saubhagya` `$68/HH`, `RDSS` `$571/HH`, `NEA 15A` `$15–$38/HH` are HH-connection proxies only) |

- **Recommended Perspective:** **Public-Sector Programme / Fiscal Budget Perspective** (aligned with `total_budget_usd_millions`, PMUY `₹12,000 cr/yr`, and AEPC/GCF `$49.2M`).
- **Harmonisation Rule:** Apply **50% public co-financing** to `improved_biomass` (`$22.50/HH`, matching MNRE/AEPC 50% CFA) and model `electric_induction` either under a **harmonized 50% hardware + 100% wiring programme subsidy** (`$26.55–$44.95/HH` IND, `$37.45–$56.18/HH` NPL) OR as a **100% programme-procured appliance package** (`$53.11/HH` IND, `$63.67/HH` NPL), rather than mixing 100% societal costs with partial fiscal subsidies.

---

## 5. Budget & Feasibility Audit (Task 5)

### 5.1 Full-Coverage (`+1.0`) vs. Full-Gap Costs (`USD Millions`, 2024 USD, Excl. `infra_lpg_grid`)

| Unit | 2025 Derived HH | Full `+1.0` LPG Cost (`$M`) | Full `+1.0` Bio (50% CFA / Soc) | Full `+1.0` Elec (50% / 100% HW) | 2024 WHO Base | Ceiling | Gap | LPG Gap Cost (`$M`, 2024 WHO -> Ceil) | LPG Gap Cost (`$M`, Obs HH -> Ceil) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `IND_urban` | `124,390,773` | `$11,963.51M` | `$2,798.79M` / `$5,597.58M` | `$3,303.20M` / `$6,606.39M` | `0.96` | `0.99` | `0.03` | **`$358.91M`** | `$1,112.61M` (NFHS-5 `0.897`) |
| `IND_rural` | `209,205,395` | `$20,120.71M` | `$4,707.12M` / `$9,414.24M` | `$5,555.45M` / `$11,110.90M` | `0.69` | `0.95` | `0.26` | **`$5,231.38M`** | `$10,422.53M` (NFHS-5 `0.432`) |
| `NPL_urban` | `4,606,688` | `$552.11M` | `$103.65M` / `$207.30M` | `$146.66M` / `$293.31M` | `0.64` | `0.98` | `0.34` | **`$187.72M`** | `$218.30M` (NPHC `0.5846`) |
| `NPL_rural` | `2,166,519` | `$292.11M` | `$48.74M` / `$97.49M` | `$68.97M` / `$137.94M` | `0.25` | `0.95` | `0.70` | **`$204.48M`** | `$218.03M` (NPHC `0.2036`) |
| **Total** | **`340,369,375`** | **`$32,928.44M`** | **`$7,658.30M`** / `$15,316.61M` | **`$9,074.28M`** / `$18,148.54M` | — | — | — | **`$5,982.48M`** | **`$11,971.47M`** |

### 5.2 Revised Budget Scenarios (`2024 WHO -> Ceiling`, Excl. Unresolved `infra_lpg_grid` Fixed Capex)
- **LPG Full-Gap Basis (`100%`):** **`$5,982.48M`** (`50%` = **`$2,991.24M`**, `25%` = **`$1,495.62M`**).
- **Lowest-Cost Harmonized Clean-Cooking Mix (50% Electric Hardware, `2024 WHO -> Ceiling`):** `$1,641.65M` (`50%` = `$820.83M`, `25%` = `$410.41M`).
- **Note:** Once Person A supplies `cf_bau_2040`, actual Stage 3 gaps (`ceiling - cf_bau_2040`) will be smaller than `ceiling - 2024_baseline`, and `infra_lpg_grid` fixed capex (`B_infra`) must be added on top for unready rural units.

### 5.3 Adoption-Cap, Infrastructure-Prerequisite, and Ceiling Checks
1. **`IND_urban` Ceiling Risk:** 2024 WHO access is already `0.96`; by 2040 BAU it will likely hit `0.98–0.99`. Mock ceiling `0.98` would likely violate `coverage_ceiling >= cf_bau_2040`; raise `coverage_ceiling.IND_urban` to `0.99` (or `max(0.99, min(1.0, cf_bau_2040 + 0.01))`).
2. **`NPL_rural` Single-Fuel Cap Deficit (`DEC-021`):** With `lpg_infra_ready_init = False` and `grid_ready_init = False`, `q < p` blocks LPG and induction in `P1`. Unlocked `P2+P3` LPG (`0.15 + 0.15 = 0.30`) and induction (`0.15 + 0.15 = 0.30`) alone sum to `0.60 < 0.70` gap from 2024 (`0.25 -> 0.95`), forcing `P1` `improved_biomass` (`0.10–0.20`) unless `cf_bau_2040 >= 0.35` or `P2/P3` caps rise.
3. **`IND_rural`:** Unlocked `P2+P3` LPG (`0.24 + 0.20 = 0.44 >= 0.26`) cleanly closes the 2024 gap even if `lpg_infra_ready_init = False`.

---

## 6. Proposed Shared-File Diffs (Unapplied — For Approval Only)

### 6.1 `docs/INTERFACES.md`
```diff
--- a/docs/INTERFACES.md
+++ b/docs/INTERFACES.md
@@ -138,7 +138,8 @@
   - Global Stage 3 settings: `total_budget_usd_millions` (`float > 0`), `infra_build_threshold` (`float in (0, 1]`, e.g. `0.5`), `apply_cost_discounting` (`bool`, default `false`), `discount_factor` (`float in (0, 1]`, e.g. `0.774`), `period_order` (`["P1", "P2", "P3"]`).
   - Per-unit/intervention maps over `["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]`:
     - `coverage_ceiling`: `{unit: float in (0, 1]}` ($c^{\max}_u$).
-    - `effectiveness`: `{unit: {lpg_subsidy: 1.0, improved_biomass: float in (0, 1), electric_induction: 1.0, infra_lpg_grid: 0.0}}`.
+    - `effectiveness`: `{unit: {lpg_subsidy: float in (0, 1], improved_biomass: float in (0, 1), electric_induction: float in (0, 1], infra_lpg_grid: 0.0}}` (values `< 1.0` reflect real-world fuel stacking).
     - `unit_costs_usd_millions`: `{unit: {intervention: float >= 0}}` (for cooking interventions: USD millions per `1.0` full-unit coverage increment; for `infra_lpg_grid`: fixed USD millions per build indicator `x >= infra_build_threshold`).
     - `max_adoption_per_period`: `{unit: {intervention: {P1: float, P2: float, P3: float}}}` in `[0, 1]`.
```

### 6.2 `docs/DECISIONS.md`
```diff
--- a/docs/DECISIONS.md
+++ b/docs/DECISIONS.md
@@ -25,3 +25,4 @@
 | 2026-11-16 | `DEC-019` | Discretize MDP state as `(clean_bins, bio_bins, infra_flags, budget_bin)` over `coverage_bins * (headroom - 1e-12)`, reusing `Stage3Evaluator.evaluate_effective_gains` directly, with 85 systematic multi-unit portfolio actions and an absorbing zero-reward terminal sink for `no_op` in infinite-horizon VI. | Tracks raw coverage ceilings separately from `improved_biomass` (`eta=0.45`) effective gains, eliminates `1e-16` float ceiling overshoots, solves in <4s, and avoids `gamma=1.0` self-loop ties in VI. | Person B |
 | 2026-11-16 | `DEC-020` | Enforce strictly-earlier-period infrastructure (`q < p`) in MDP transitions, HTN, and STRIPS PDDL (`(strictly-before ?p_infra ?p)` + sequential `(budget-level ?b)`), while keeping `Stage3Evaluator` at `q <= p` (`DEC-012`) so all MDP/HTN/PDDL plans validate with zero violations. | Preserves existing `test_evaluator.py` contract, eliminates `8!` unordered budget-token symmetry in `pyperplan`, and documents the `q < p` vs `q <= p` timing difference for `B-05` equal-footing alignment. | Person B |
 | 2026-11-16 | `DEC-021` | **Supersedes `DEC-012` & aligns `DEC-020`:** Unify `Stage3Evaluator`, `Stage3Repair`, baselines, MDP, HTN, and PDDL on the single **strictly-earlier-period** infrastructure rule (`q < p` unlocks `lpg_subsidy` and `electric_induction` in periods `p > q` when init flags are `false`). | Eliminates the `P1` same-period timing advantage between open-loop GA and sequential MDP/PDDL so `B-05` compares all paradigms on strictly identical physical feasibility rules. | Person B |
+| 2026-11-17 | `DEC-022` | Lock B-08 real-data integration conventions: (a) WUP 2025 National Definitions + 2025 derived households (`124.39M`, `209.21M`, `4.61M`, `2.17M`); (b) WHO SDG 7.1.2 (`people`-denominated) as Contracts 1–3 driver with explicit 1:1 marginal bridge and DHS/NPHC household benchmarks; (c) Alternative A one-time 5-year cohort cost accounting with `apply_cost_discounting: false` (preventing double-discounting in MDP/compare); (d) Public-sector programme costing perspective. | Preserves Contract 1–3 compatibility without `P3`-deferral distortion under terminal-2040 DALYs and harmonizes intervention cost boundaries. | Shared |
```

### 6.3 `configs/default.yaml` *(Only after Person A delivers Contracts 1–3 & Contract 4 is finalized)*
```diff
--- a/configs/default.yaml
+++ b/configs/default.yaml
@@ -24,7 +24,7 @@
   raw_manifest: "data/raw/manifest.json"
 
 interfaces:
-  use_mock_forecasts: true  # Flip to false once Stage 2 exports validated forecasts_stage3.csv
+  use_mock_forecasts: false  # Enabled only after Person A Contracts 1-3 & Contract 4 pass validation
   forecast_contract_file: "data/processed/forecasts_stage3.csv"
   forecast_mock_file: "data/processed/forecasts_stage3_mock.csv"
   baseline_state_file: "data/processed/baseline_unit_state.csv"
@@ -104,7 +104,7 @@
   mdp:
     coverage_bins: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
     budget_bins: 6
     discount_factor: 0.774  # ~0.95^5 per 5-year period; sensitivity tested with 1.0
-    cost_weight_lambda: 3500.0  # Calibrated from Stage 3 Pareto slope (~3,431 DALYs/$1M upper-quantile tangent; DEC-018)
+    cost_weight_lambda: null  # TODO(B-06): recalibrate via calibrate_lambda_from_pareto() on real Stage 3 front
     transition_probs: [0.15, 0.70, 0.15]  # [low, nominal, high] uptake (ASM-15)
```
