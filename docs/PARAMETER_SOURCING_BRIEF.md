# Parameter Sourcing Brief (Stage 3 Optimisation & Stage 4 MDP Planning)

> **Purpose:** Reference specification for researching real-world literature and policy values to replace synthetic placeholders across Stage 3 (`src/optimise/evaluator.py`, `src/optimise/nsga2.py`) and Stage 4 (`src/planning/mdp.py`, `src/planning/compare.py`).
>
> **Scientific Integrity Rules (`AGENTS.md` §6 & `docs/ASSUMPTIONS.md`):**
> - **Never invent numbers.** Leave unverified parameters as `null` in `configs/intervention_params.json` and `placeholder: needs literature` in `docs/ASSUMPTIONS.md`.
> - **Scenario framing:** All intervention costs, effectiveness factors, adoption caps, and budgets are scenario assumptions—never real-world policy prescriptions.
> - **Mock file path note:** `configs/default.yaml` points `interfaces.intervention_params_mock_file` to `data/processed/intervention_params_mock.json` (there is no `configs/intervention_params_mock.json`), and `interfaces.intervention_params_file` to `configs/intervention_params.json`.

---

## 1. Scope & File Ownership

This brief covers **every non-forecast parameter** read by `Stage3Evaluator` (`src/optimise/evaluator.py`), `Stage3Repair` (`src/optimise/nsga2.py`), the custom NumPy MDP (`src/planning/mdp.py`), and the equal-footing stochastic comparator (`src/planning/compare.py`):

| Source File | Role | What Comes From Here |
| --- | --- | --- |
| `configs/intervention_params.json` *(mock: `data/processed/intervention_params_mock.json`)* | **Contract 4** (Person B owned) | Total budget, infra threshold, cost discounting toggle/factor, coverage ceilings, effectiveness, unit costs, per-period adoption caps |
| `data/processed/exposure_response_surrogate.json` *(mock: `*_mock.json`)* | **Contract 3** (Person A exports from Stage 2; cross-checked with `ASM-04`, `ASM-14`) | Surrogate clean-fuel slope (`slope_clean_fuel`), female-to-male elasticity ratio (`sex_elasticity_ratio_f_to_m`), surrogate uncertainty SD (`uncertainty_sd`) |
| `data/processed/baseline_unit_state.csv` *(mock: `*_mock.csv`)* | **Contract 2** (Person A owned; non-forecast infra flags) | Initial infrastructure readiness booleans (`lpg_infra_ready_init`, `grid_ready_init`) at `t_0 = 2025` |
| `configs/default.yaml` | Shared config (`planning.mdp.*`, `planning.compare.*`) | MDP stochastic uptake probabilities, shock multipliers, MDP discount factor, scalarization weights (`lambda_cost`, `lambda_eq`), discretization grids |

*(Forecast-derived inputs—2040 BAU DALYs, 2040 BAU ASRs, 2040 population weights, 2040 BAU `clean_fuel_share`, and Contract 3 `baseline_clean_fuel`—are produced by Person A's Stage 2 pipeline and are excluded from sourcing.)*

**Structural axes (fixed in `configs/default.yaml`):**
- **4 allocation units (`u`):** `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`
- **4 interventions (`i`):** `lpg_subsidy`, `improved_biomass`, `electric_induction`, `infra_lpg_grid`
- **3 five-year periods (`p`):** `P1` (`2026–2030`), `P2` (`2031–2035`), `P3` (`2036–2040`)

---

## 2. Complete Parameter Reference

### 2.1 Contract 4 — `configs/intervention_params.json`

| ASM ID | JSON Key Path | Plain-English Meaning | Exact Unit Code Expects | Mock Value (`*_mock.json`) | Allowed Range (`contracts.py`) | How the Code Uses It | Most Sensitive Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-13` | `total_budget_usd_millions` | Cumulative 15-year (`2026–2040`) programme budget cap across all 4 units, 4 interventions, and 3 periods. | **USD millions** (`$M`, same base-year USD as `unit_costs_usd_millions`) | `500.0` *(template: `null`)* | Finite float `> 0.0` | `evaluator.py`: `violation_budget = max(0, total_cost - total_budget_usd_millions)`. `mdp.py`: sets initial budget state and `budget_step = total_budget / (budget_bins - 1)`; rejects actions exceeding remaining budget. `nsga2.py`: HV reference point uses `1.05 * total_budget`. | Feasibility frontier (`is_feasible`), minimum achievable `dalys_2040_mean`, and whether rural `infra_lpg_grid` + clean fuels are affordable vs. `improved_biomass` bridge. |
| — *(modeling switch)* | `infra_build_threshold` | Minimum continuous decision variable `x[u, infra_lpg_grid, p]` that triggers a binary regional infrastructure build. | **Dimensionless** threshold on `x in [0, 1]` | `0.5` *(template: `0.5`)* | Finite float in `(0.0, 1.0]` | `evaluator.py`: `infra_built = (x_infra >= infra_build_threshold)`. Values `>= threshold` incur full fixed `infra_lpg_grid` cost and unlock `lpg_subsidy`/`electric_induction` in strictly later periods (`q < p`); values `< threshold` cost `$0` and build nothing. `mdp.py`/`baselines.py` write this value into `x_infra` when building. | NSGA-II binary switching behavior around `0.5` (no direct empirical impact once fixed at `0.5`). |
| `ASM-16` / `DEC-014` | `apply_cost_discounting` | Whether future-period expenditures (`P2`, `P3`) are discounted to present value (`P1` = `1.0`). | **Boolean** (`true` / `false`) | `false` *(template: `false`)* | `bool` | `evaluator.py` & `mdp.py`: `_period_discount[p_idx] = (discount_factor ** p_idx) if apply_cost_discounting else 1.0` for `p_idx in {0, 1, 2}`. Multiplies both `cook_cost` and `infra_cost`. *(Does not discount 2040 DALYs.)* | `total_cost_usd_millions`, `violation_budget`, and `P1` vs. `P3` expenditure timing. |
| `ASM-16` | `discount_factor` | Per-5-year-period cost discount factor ($\gamma = (1 + r)^{-5}$) applied to `P2` ($\gamma^1$) and `P3` ($\gamma^2$) when `apply_cost_discounting` is `true`. | **Dimensionless** factor per 5-year period | `0.774` ($\approx 0.95^5$; *template: `0.774`*) | Finite float in `(0.0, 1.0]` | `evaluator.py` & `mdp.py`: sets `_period_discount = [1.0, discount_factor, discount_factor**2]` when `apply_cost_discounting == true`. | `total_cost_usd_millions` and `P1` vs. `P2`/`P3` pacing (only active if `apply_cost_discounting: true`). |
| `ASM-12` / `DEC-011` | `coverage_ceiling.<unit>` *(4 keys: `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`)* | Maximum total clean/improved cooking coverage fraction reachable in unit `u` by 2040 (BAU 2040 share + new `P1–P3` adoption). | **Total unit coverage fraction** in `(0, 1]` (`0.95` = 95% of households) | `IND_urban: 0.98`<br>`IND_rural: 0.95`<br>`NPL_urban: 0.98`<br>`NPL_rural: 0.95` *(template: `null`)* | Finite float in `(0.0, 1.0]`; must be `>= cf_bau_2040[u]` for positive MDP headroom | `evaluator.py`: `violation_coverage_ceiling = sum_u max(0, cf_bau_2040[u] + raw_gain_u[u] - coverage_ceiling[u])`. `mdp.py`: defines `headroom_u = max(0, coverage_ceiling[u] - cf_bau_2040[u] - 1e-12)` and grid step `step_u = headroom_u / max_bin`. | Maximum achievable DALYs averted (`dalys_2040_mean`), especially in high-BAU urban units (`IND_urban`, `NPL_urban`), and MDP discretization step `step_u`. |
| `ASM-09` | `effectiveness.<unit>.lpg_subsidy` *(4 units)* | Fractional HAP exposure / COPD PAF reduction per `+1.0` coverage of `lpg_subsidy`, relative to full clean-cooking counterfactual (`1.0`). | **Dimensionless** multiplier $\eta \in [0, 1]$ | `1.0` for all 4 units *(template: `1.0` per `INTERFACES.md` §5; see note below)* | Finite float in `[0.0, 1.0]` in `contracts.py` (`INTERFACES.md` §5 locks `1.0` unless schema revised) | `evaluator.py`: multiplies `x[u, lpg_subsidy, p]` inside `eff_gain_u`, which feeds `exp(slope * delta_c)`. | `dalys_2040_mean`, `equity_gap_asr`, and `lpg_subsidy` vs. `electric_induction` / `improved_biomass` ranking. |
| `ASM-10` | `effectiveness.<unit>.improved_biomass` *(4 units)* | Fractional exposure / COPD PAF reduction per `+1.0` coverage of improved biomass cookstoves (`ICS`) relative to full clean fuel (`1.0`), accounting for field vs. lab efficiency and fuel stacking. | **Dimensionless** multiplier $\eta \in [0, 1]$ | `0.45` for all 4 units *(template: `null`)* | Finite float in `[0.0, 1.0]` in `contracts.py`; `(0.0, 1.0)` in `INTERFACES.md` §5 | `evaluator.py`: contributes `x * eta_bio` to `eff_gain_u` (DALYs) while counting full `1.0 * x` against `coverage_ceiling` (`raw_gain_u`). `mdp.py`: tracks `bio_bins` separately and weights by `bio_eta`. | Rural `dalys_2040_mean` under tight budgets and lock-in trade-off vs. `infra_lpg_grid` + `lpg_subsidy`. |
| `ASM-11` | `effectiveness.<unit>.electric_induction` *(4 units)* | Fractional HAP exposure / COPD PAF reduction per `+1.0` coverage of `electric_induction`, relative to full clean-cooking counterfactual (`1.0`). | **Dimensionless** multiplier $\eta \in [0, 1]$ | `1.0` for all 4 units *(template: `1.0` per `INTERFACES.md` §5; see note below)* | Finite float in `[0.0, 1.0]` in `contracts.py` (`INTERFACES.md` §5 locks `1.0` unless schema revised) | `evaluator.py`: multiplies `x[u, electric_induction, p]` inside `eff_gain_u`, gated by `grid_ready`. | `dalys_2040_mean` and `electric_induction` vs. `lpg_subsidy` choice. |
| — *(structural)* | `effectiveness.<unit>.infra_lpg_grid` *(4 units)* | Direct HAP exposure reduction from building infrastructure alone (without household stove/fuel adoption). | **Dimensionless** (`0.0`) | `0.0` for all 4 units *(template: `0.0`)* | Strictly `0.0` (`contracts.py` rejects `!= 0.0`) | Excluded from `eff_gain_u` (`NON_INFRA_INTERVENTIONS`); only unlocks clean-cooking interventions in subsequent periods. | N/A (fixed at `0.0`). |
| `ASM-05` | `unit_costs_usd_millions.<unit>.lpg_subsidy` *(4 units)* | Total 5-year programme/subsidy cost to transition `100%` of unit `u` (`+1.0` coverage) onto sustained LPG cooking during one 5-year period. | **USD millions per `1.0` unit coverage increment** (`$M / 1.0 cov`) | `IND_u: 320.0`, `IND_r: 620.0`, `NPL_u: 18.0`, `NPL_r: 48.0` *(template: `null`)* | Finite float `>= 0.0` | `evaluator.py` & `mdp.py`: `cost += x[u, lpg, p] * unit_cost * period_discount[p]`. | `total_cost_usd_millions`, `violation_budget`, and `lpg_subsidy` vs. `electric_induction` vs. `improved_biomass` share. |
| `ASM-06` | `unit_costs_usd_millions.<unit>.improved_biomass` *(4 units)* | Total 5-year programme/hardware cost to transition `100%` of unit `u` (`+1.0` coverage) onto improved biomass cookstoves during one 5-year period. | **USD millions per `1.0` unit coverage increment** (`$M / 1.0 cov`) | `IND_u: 120.0`, `IND_r: 210.0`, `NPL_u: 7.0`, `NPL_r: 18.0` *(template: `null`)* | Finite float `>= 0.0` | `evaluator.py` & `mdp.py`: `cost += x[u, bio, p] * unit_cost * period_discount[p]`. | `total_cost_usd_millions` and low-budget Pareto/MDP adoption of `improved_biomass`. |
| `ASM-07` | `unit_costs_usd_millions.<unit>.electric_induction` *(4 units)* | Total 5-year programme/appliance/tariff cost to transition `100%` of unit `u` (`+1.0` coverage) onto electric induction cooking during one 5-year period. | **USD millions per `1.0` unit coverage increment** (`$M / 1.0 cov`) | `IND_u: 410.0`, `IND_r: 780.0`, `NPL_u: 22.0`, `NPL_r: 60.0` *(template: `null`)* | Finite float `>= 0.0` | `evaluator.py` & `mdp.py`: `cost += x[u, elec, p] * unit_cost * period_discount[p]`. | `total_cost_usd_millions` and whether `electric_induction` beats `lpg_subsidy` (when both have `effectiveness = 1.0`, solver prefers whichever has lower unit cost). |
| `ASM-08` | `unit_costs_usd_millions.<unit>.infra_lpg_grid` *(4 units)* | Fixed one-time regional capital expenditure to build LPG distribution and electric cooking grid capacity in unit `u` (`x >= infra_build_threshold`). | **USD millions per binary regional build event** (`$M`, not scaled by `x`) | `IND_u: 40.0`, `IND_r: 95.0`, `NPL_u: 5.0`, `NPL_r: 16.0` *(template: `null`)* | Finite float `>= 0.0` | `evaluator.py` & `mdp.py`: adds full `unit_cost * period_discount[p]` whenever `x_infra >= threshold`. Not charged for units where both `lpg_infra_ready_init` and `grid_ready_init` are `true` (though JSON schema still requires a `>= 0` number for all 4 units). | Whether rural LPG/electric scale-up is affordable in `P2–P3` under `total_budget_usd_millions`, `P1` spend, and MDP value cliff. |
| `ASM-12` | `max_adoption_per_period.<unit>.<cook_interv>.<period>` *(4 units × 3 cooking interventions × `P1, P2, P3` = 36 keys)* | Maximum feasible additional coverage fraction ($\Delta c^{\max}_{u,i,p}$) addable by cooking intervention `i` in unit `u` during one 5-year period `p`. | **Fractional coverage increment per 5-year period** in `[0, 1]` (`0.20` = +20 pct pts / 5 yrs) | `lpg_subsidy: 0.20`<br>`improved_biomass: 0.25`<br>`electric_induction: 0.15` *(all units/periods; template: `null`)* | Finite float in `[0.0, 1.0]` | `evaluator.py`: `violation_adoption_cap = sum max(0, x_cook - max_adoption)`. `mdp.py`: blocks any discrete step `steps * step_u > cap + 1e-12` and caps high-uptake shocks at `floor(cap / step_u)`. | `violation_adoption_cap`, multi-period pacing (`P1–P3`), max `dalys_averted_mean` in rural units, and MDP action feasibility (`step_u <= cap`). |
| — *(structural)* | `max_adoption_per_period.<unit>.infra_lpg_grid.<period>` *(4 units × 3 periods = 12 keys)* | Schema placeholder for `infra_lpg_grid` adoption cap. | **Dimensionless** in `[0, 1]` | `1.0` for all units/periods *(template: `1.0`)* | Finite float in `[0.0, 1.0]` | Validated by `contracts.py`, **ignored** by `evaluator.py` and `mdp.py` (which slice only `_non_infra_max_adoption`). Keep at `1.0`. | None (unused by evaluator/MDP). |

> **Schema alignment note on `ASM-09` & `ASM-11` (`lpg_subsidy` and `electric_induction` effectiveness):**
> `docs/INTERFACES.md` §5 and `configs/intervention_params.json` pre-fill `effectiveness = 1.0` for `lpg_subsidy` and `electric_induction` (`DEC-011`: full clean cooking = `1.0`, transitional `improved_biomass` < `1.0`). However, `docs/ASSUMPTIONS.md` marks `ASM-09` and `ASM-11` as `placeholder: needs literature` for **fuel-stacking adjustment**, and `src/data/contracts.py` already allows any value in `[0.0, 1.0]`. Research fuel-stacking-adjusted effectiveness for both anyway—if real values are `< 1.0`, the team can update `docs/INTERFACES.md` §5 and `configs/intervention_params.json` cleanly without changing any Python code.

---

### 2.2 Contract 3 — `data/processed/exposure_response_surrogate.json`

Although Person A's Stage 2 calibration script (`src/forecast/export_contracts.py`) exports this file, `Stage3Evaluator` and `MDP` read three parameters per unit directly from it, and `docs/ASSUMPTIONS.md` assigns `ASM-04` and `ASM-14` for literature grounding / cross-check:

| ASM ID | JSON Key Path | Plain-English Meaning | Exact Unit Code Expects | Mock Value (`*_mock.json`) | Allowed Range (`contracts.py`) | How the Code Uses It | Most Sensitive Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-14` | `units.<unit>.slope_clean_fuel` *(4 units)* | Male (reference sex) log-linear exposure–response slope $\beta_u$: log-multiplier on 2040 HAP-COPD DALYs and ASR per `+1.0` effective clean-cooking coverage above 2040 BAU. | **Log burden ratio per `1.0` effective coverage increment** (dimensionless, $\le 0$) | `IND_u: -1.30`, `IND_r: -1.45`, `NPL_u: -1.25`, `NPL_r: -1.50` | Finite float `<= 0.0` | `evaluator.py` (`evaluate_effective_gains`): `M_{u,male} = exp(slope_clean_fuel * delta_c_u)` where `delta_c_u = min(max(0, eff_gain_u), 1 - cf_bau_2040)`. Multiplies `bau_dalys_2040` and `bau_asr_2040`. Shared by `mdp.py`. | `dalys_2040_mean`, `dalys_averted_mean` (exponentially sensitive), `cost_weight_lambda`, and rural vs. urban marginal DALYs/$M. |
| `ASM-04` | `units.<unit>.sex_elasticity_ratio_f_to_m` *(4 units)* | Female-to-male relative risk response ratio $r_{f/m, u}$ per unit of clean-cooking coverage gain (`> 1.0` means larger percentage HAP-COPD reduction in women). | **Dimensionless ratio** ($\beta_{u,\text{female}} / \beta_{u,\text{male}}$) | `IND_u: 1.10`, `IND_r: 1.15`, `NPL_u: 1.10`, `NPL_r: 1.18` | Finite float `> 0.0` | `evaluator.py`: `_slope_female = _slope_male * sex_elasticity_ratio_f_to_m`; `M_{u,female} = exp(r_{f/m,u} * beta_u * delta_c_u)`. | `equity_gap_asr` (population-weighted `|ASR_F - ASR_M|`), `equity_gap_dalys_abs`, and female-heavy unit prioritization. |
| `ASM-14` / `DEC-013` | `units.<unit>.uncertainty_sd` *(4 units)* | Relative SD of the surrogate's log-multiplier per `+1.0` effective coverage increment (captures Stage 2 / exposure–response residual uncertainty). | **Dimensionless fractional SD per `1.0` effective coverage increment** | `IND_u: 0.09`, `IND_r: 0.11`, `NPL_u: 0.09`, `NPL_r: 0.12` | Finite float `>= 0.0` | `evaluator.py`: `surr_hw = dalys_mean_us * (1.96 * uncertainty_sd * delta_c)`, combined via `np.hypot` with scaled BAU half-widths for `dalys_2040_lower` and `dalys_2040_upper`. Does not affect mean or optimisation fitness. | `dalys_2040_lower` and `dalys_2040_upper` (95% UI width at large `delta_c`). |

*(Note: `units.<unit>.baseline_clean_fuel` in Contract 3 is validated in `[0, 1]` by `contracts.py`, while `evaluator.py` reads `_cf_bau_2040` from Contract 1 `forecasts_stage3.csv`; both come from Person A's Stage 2 forecast.)*

---

### 2.3 Contract 2 — `data/processed/baseline_unit_state.csv` (Non-Forecast Infrastructure Readiness Flags)

While `baseline_unit_state.csv` is produced by Person A (`Contract 2`), the two initial infrastructure readiness flags are **energy/infrastructure status assumptions** rather than GBD/UN WPP forecasts (`docs/PLANNING_SPEC.md` §1.1):

| ASM ID | Column Name (`baseline_unit_state.csv`) | Plain-English Meaning | Exact Unit Code Expects | Mock Value (`*_mock.csv`) | Allowed Range (`contracts.py`) | How the Code Uses It | Most Sensitive Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — *(needs check; `PLANNING_SPEC.md` §1.1)* | `lpg_infra_ready_init` *(4 units)* | Whether unit `u` already has sufficient LPG supply/bottling/distribution infrastructure at `t_0 = 2025` to scale `lpg_subsidy` in `P1` without building `infra_lpg_grid` first. | **Boolean** (`true` / `false`) | `IND_urban: True`, `IND_rural: False`, `NPL_urban: True`, `NPL_rural: False` | `bool` | `evaluator.py` & `mdp.py`: if `false`, `lpg_subsidy` is blocked until strictly after a period where `infra_lpg_grid` is built (`q < p`, i.e., `P1` build unlocks `P2–P3`). | `P1` action feasibility (`violation_infra_prereq`), `P1` capital allocation, and whether rural LPG can start before `P2`. |
| — *(needs check; `PLANNING_SPEC.md` §1.1)* | `grid_ready_init` *(4 units)* | Whether unit `u` already has sufficient electric distribution/feeder capacity at `t_0 = 2025` to scale `electric_induction` in `P1` without building `infra_lpg_grid` first. | **Boolean** (`true` / `false`) | `IND_urban: True`, `IND_rural: False`, `NPL_urban: True`, `NPL_rural: False` | `bool` | `evaluator.py` & `mdp.py`: if `false`, `electric_induction` is blocked until strictly after a period where `infra_lpg_grid` is built (`q < p`). *(Note: `infra_lpg_grid` unlocks both LPG and grid together.)* | `P1` electric induction feasibility and `infra_lpg_grid` prerequisite trigger. |

---

### 2.4 `configs/default.yaml` — MDP & Equal-Footing Comparison Parameters

| ASM ID | YAML Key Path (`configs/default.yaml`) | Plain-English Meaning | Exact Unit Code Expects | Mock / Default Value | Allowed Range | How the Code Uses It | Most Sensitive Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-15` | `planning.mdp.transition_probs` | Probability triplet `[p_low, p_nom, p_high]` for stochastic uptake realization (low, nominal, high) in each 5-year period. | **Probability triplet** summing to `1.0` | `[0.15, 0.70, 0.15]` | 3 finite floats `>= 0.0`, `sum == 1.0` | `mdp.py`: weights the 3 stochastic branches in Bellman backup `sum(probs * (immediate + gamma * V[next]))` and rollout simulation. `compare.py`: drives Common Random Numbers shocks across MDP, GA, and PDDL. | MDP `expected_value_s0`, Value of Adaptive Recourse (MDP vs. open-loop GA), and `p05`/`p95` DALY range. |
| `ASM-15` / `PLANNING_SPEC` | `planning.compare.shock_multipliers` *(and hardcoded in `mdp.py`)* | Realized coverage multipliers `[m_low, m_nom, m_high]` applied to planned period cooking actions under low/nominal/high branches. | **Dimensionless multiplier** on planned period step | `[0.5, 1.0, 1.25]` | 3 floats (typically `m_low <= 1.0 <= m_high`) | `compare.py`: scales open-loop GA & PDDL cooking tensors (`x_cook * m`). **Code caveat:** `mdp.py` (`_apply_action_branch` L336–341) currently hardcodes `floor(0.5 * steps)`, `steps`, `min(ceil(1.25 * steps), cap_bins)` on discrete bins rather than reading `shock_multipliers` from YAML. | Open-loop GA `constraint_violation_rate` (when `m_high > 1.0` exceeds `max_adoption` or `coverage_ceiling`), rollout variance, and MDP vs. GA gap. |
| `ASM-16` | `planning.mdp.discount_factor` *(sensitivity: `planning.mdp.discount_sensitivity`)* | 5-year Bellman discount factor $\gamma$ applied to future-period rewards (`DALYs_averted - lambda * cost + lambda_eq * eq_red`) in MDP DP/VI and discounted returns. | **Dimensionless** factor per 5-year period | `0.774` *(sensitivity: `[0.774, 1.0]`)* | Finite float in `(0.0, 1.0]` | `mdp.py`: `immediate + gamma * V[next_state]` in finite-horizon backward induction and infinite-horizon VI; also discounts `discounted_return` in `mdp.py` and `compare.py`. | MDP `expected_value_s0`, early (`P1`) vs. deferred (`P2`/`P3`) action timing, and `infra_lpg_grid` vs. `improved_biomass` threshold. |
| `DEC-018` *(calibrated, not literature)* | `planning.mdp.cost_weight_lambda` *(sensitivity: `planning.mdp.lambda_sensitivity`)* | Scalarization penalty converting `$1M` expenditure into DALY-equivalents inside MDP reward (`DALYs_averted - lambda * cost`). | **DALYs per USD million** (`DALYs / $M`) | `3500.0` *(sensitivity: `[0.0, 1000.0, ..., 10000.0]`)* | Finite float `>= 0.0` | `mdp.py`: calibrated from Stage 3 Pareto front slope via `calibrate_lambda_from_pareto()` once real Stage 3 runs finish (`DEC-018`). Optionally cross-check against WHO/DCP3 willingness-to-pay thresholds (`$/DALY`). | MDP policy aggressiveness (`no_op` vs. `improved_biomass` vs. `infra + lpg`), `mean_cost_usd_millions`, `mean_dalys_averted`. |
| — *(sensitivity sweep)* | `planning.compare.lambda_eq_sensitivity` | Equity scalarization weight rewarding reductions in population-weighted female–male ASR gap inside MDP vs. GA sensitivity sweeps. | **DALYs per (DALY/100k ASR gap)** | `[0.0, 2500.0, 10000.0, 25000.0, 50000.0]` | Finite floats `>= 0.0` | `compare.py`: passed as `equity_weight_lambda` to `solve_finite_horizon_backward_induction` and `simulate_open_loop_schedule`. | `mean_equity_gap_asr` and shift toward units with wider female–male ASR gaps. |
| — *(algorithmic grid)* | `planning.mdp.coverage_bins`<br>`planning.mdp.budget_bins`<br>`planning.compare.budget_bins_sensitivity` | Discrete MDP grid resolution: `max_bin = len(coverage_bins) - 1` coverage steps from BAU to `coverage_ceiling`, and `budget_bins` remaining-budget tranches. | **Dimensionless** grid points / integer bin count | `coverage_bins: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (`max_bin = 5`)<br>`budget_bins: 6` *(sens: `[6, 11]`)* | Monotonic list in `[0, 1]`; integer `>= 2` | `mdp.py`: `step_u = (coverage_ceiling[u] - cf_bau_2040[u]) / max_bin`. **Constraint interaction:** MDP only allows a cooking step if `step_u <= max_adoption_per_period[u, i, p]`! | MDP state count, budget rounding loss (`budget_step`), and whether `1`-step actions fit under `max_adoption_per_period`. |

*(Note: `configs/default.yaml` also contains `stage3.total_budget_usd_millions: null`, which is **not** read by `Stage3Evaluator`—Contract 4 `intervention_params.json` is the single source of truth—and solver hyperparameters `n_rollouts`, `vi_tol`, `vi_max_iter`, `population_size`, `n_generations`, `seeds`.)*

---

## 3. Literature-to-Code Unit Conversions & Required Auxiliary Inputs

### 3.1 Cooking Intervention Unit Costs (`ASM-05`, `ASM-06`, `ASM-07`)
- **Literature's usual unit:** Stove/connection upfront hardware cost ($\text{USD / HH}$) plus annual fuel, subsidy, maintenance, or programme delivery cost ($\text{USD / HH / yr}$), often reported in `INR`, `NPR`, or `PPP USD` from an earlier study year.
- **Code's expected unit:** `USD millions per 1.0 full-unit coverage increment over one 5-year period` (`$M / 1.0 coverage`).
- **Step-by-step conversion:**
  1. **Currency & price-year normalization** to constant base-year USD (e.g., 2024 or 2025 USD):
     $$C^{\text{USD}}_{\text{HH}} = C^{\text{local}}_{\text{HH}} \times \frac{1}{\text{FX}_{\text{local/USD},\,\text{study\_yr}}} \times \frac{\text{CPI}_{\text{base}}}{\text{CPI}_{\text{study\_yr}}}$$
  2. **5-year cohort cost per household** (matching the evaluator's one-time period charge `x[u, i, p] * unit_cost` when a household cohort adopts in period `p`):
     $$C^{\text{5yr}}_{\text{HH}, u, i} = n_{\text{rep}, u, i} \cdot C^{\text{hardware}}_{\text{HH}, u, i} + 5 \cdot C^{\text{annual\_op\_or\_subsidy}}_{\text{HH}, u, i}$$
     where $n_{\text{rep}, u, i} \ge 1$ accounts for stove replacement if useful life $< 5\text{ years}$ (e.g., biomass ICS vs. LPG/induction). Decide and document **one consistent cost perspective** across all interventions and `total_budget_usd_millions` (either *public-sector programme/subsidy cost* or *full societal cost including household stove & fuel*).
  3. **Scale from per-household to `1.0` (100%) coverage of unit $u$ in USD millions:**
     $$\text{unit\_costs\_usd\_millions}_{u, i} = \frac{C^{\text{5yr}}_{\text{HH}, u, i} \times H_u}{10^6}, \qquad H_u = \frac{\text{Pop}_u}{\bar{s}_u}$$
- **Extra inputs needed:**
  1. $\text{Pop}_u$: Unit population in the decision horizon (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`) from UN WPP 2024 + UN WUP urban/rural apportionment (`baseline_unit_state.csv` `population_total` or `forecasts_stage3.csv` `population` over 2026–2040).
  2. $\bar{s}_u$: Average household size (persons/household) for **urban India**, **rural India**, **urban Nepal**, and **rural Nepal** (from Census of India / NFHS-5, Nepal Population & Housing Census 2021 / NDHS, or UN DESA Household Size and Composition).
  3. $\text{FX}_{\text{INR/USD}}$, $\text{FX}_{\text{NPR/USD}}$, and $\text{CPI}$ / GDP deflator series (World Bank / IMF).
  4. Stove lifespan (years) if hardware replacement within 5 years is required.

### 3.2 Infrastructure Fixed Cost (`ASM-08`: `infra_lpg_grid`)
- **Literature's usual unit:** Capital cost per bottling plant / distributor / km of feeder line / transformer upgrade ($\text{USD / facility}$ or $\text{USD / unserved HH}$), or total national/state grid-strengthening capex (`INR crore` / `NPR billion`).
- **Code's expected unit:** Fixed `USD millions` per binary regional build event (`x >= infra_build_threshold`).
- **Conversion:**
  $$\text{unit\_costs\_usd\_millions}_{u, \text{infra}} = \frac{C^{\text{USD}}_{\text{capex\_per\_unserved\_HH}, u} \times H_{u, \text{unserved}}}{10^6} \quad \text{or} \quad \frac{N_{\text{facilities}, u} \times C^{\text{USD}}_{\text{facility}, u}}{10^6}$$
- **Extra inputs needed:** Number of unserved / infrastructure-constrained households $H_{u, \text{unserved}}$ (or required facilities/feeder upgrades) in `IND_rural` and `NPL_rural`, $\text{Pop}_u$, $\bar{s}_u$, $\text{FX}$, and $\text{CPI}$. *(For units where `lpg_infra_ready_init = true` and `grid_ready_init = true`, `evaluator.py` never requires an infra build, but `contracts.py` still requires a non-null number `>= 0.0`.)*

### 3.3 Total Programme Budget Envelope (`ASM-13`: `total_budget_usd_millions`)
- **Literature's usual unit:** Annual government ministry allocation (e.g., `INR crore/yr` or `NPR billion/yr`), 5-year national plan budget, or donor/multilateral clean-cooking commitment.
- **Code's expected unit:** Cumulative 15-year (`2026–2040`, 3 × 5-year periods) combined India + Nepal budget in `USD millions`.
- **Conversion:**
  $$B_{\text{total}} = \left( B^{\text{annual}}_{\text{IND, USD\_M}} + B^{\text{annual}}_{\text{NPL, USD\_M}} \right) \times 15\text{ yrs}$$
  *(normalized to the exact same base-year USD and cost boundary—public fiscal vs. societal—as `unit_costs_usd_millions`)*.
- **Extra inputs needed:** Share of ministry/utility budgets specifically dedicated to household clean cooking (excluding industrial/commercial LPG or bulk generation capex), $\text{FX}$, and $\text{CPI}$.

### 3.4 Effectiveness Factors (`ASM-09`, `ASM-10`, `ASM-11`)
- **Literature's usual unit:** Percentage or absolute reduction in kitchen/personal $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$), COPD Relative Risk ($\text{RR}$) change, or exclusive-use / fuel-stacking displacement fraction (`%` of cooking energy shifted).
- **Code's expected unit:** Dimensionless multiplier $\eta_{u,i} \in [0, 1]$ relative to full clean cooking (`1.0`).
- **Conversion:**
  - Using GBD Integrated Exposure–Response (IER) / MR-BRT COPD relative risk $\text{RR}(\text{PM}_{2.5})$ at baseline solid-fuel exposure $E_0$, intervention field exposure $E_i$ (stacking-adjusted), and clean-fuel benchmark $E_{\text{clean}}$:
    $$\eta_{u, i} = \frac{\text{PAF}(E_0) - \text{PAF}(E_i)}{\text{PAF}(E_0) - \text{PAF}(E_{\text{clean}})}, \qquad \text{where } \text{PAF}(E) = \frac{\text{RR}(E) - 1}{\text{RR}(E)}$$
  - Or, when using displacement shares from field fuel-stacking studies: $\eta_{u,i} = f_{\text{displaced}, u, i} \times \eta^{\text{lab\_or\_exclusive}}_{u, i}$.
- **Extra inputs needed:** Field personal/kitchen $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) for traditional biomass, improved biomass (forced-draft/rocket/gasifier), LPG (with stacking), and electric induction in South Asia; GBD 2023 COPD $\text{RR}(\text{PM}_{2.5})$ curve; exclusive-use vs. stacking fractions.

### 3.5 Max Adoption Caps (`ASM-12`) & Coverage Ceilings
- **Literature's usual unit:** Annual clean-fuel share growth ($\text{pct pts / yr}$ from WHO SDG 7.1.2 / NFHS / DHS) or annual households reached ($\text{HH / yr}$, e.g., connections/stoves distributed per year) and maximum achievable `%` ceiling.
- **Code's expected unit:** Fractional coverage increment per 5-year period in `[0, 1]` (`max_adoption_per_period`) and total fraction in `(0, 1]` (`coverage_ceiling`).
- **Conversion:**
  $$\Delta c^{\max}_{u,i,p} = \min\!\left(1.0,\; 5 \times \frac{r_{\text{pct\_pt/yr}}}{100} \times f_{\text{sustained}}\right) \quad \text{or} \quad \min\!\left(1.0,\; \frac{5 \times N_{\text{HH/yr}} \times f_{\text{sustained}}}{H_u}\right)$$
  $$c^{\max}_u = \frac{\text{Max\_Reachable\_Pct}_u}{100}$$
- **Extra inputs needed:** $H_u = \text{Pop}_u / \bar{s}_u$, historical annual adoption rates, sustained-adoption/refill retention rate $f_{\text{sustained}}$, and Person A's `cf_bau_2040[u]` (to confirm `coverage_ceiling[u] >= cf_bau_2040[u]` and check MDP step size `step_u = (coverage_ceiling[u] - cf_bau_2040[u]) / max_bin <= max_adoption_per_period[u, i, p]`).

### 3.6 Surrogate Slope (`ASM-14`) & Sex Elasticity Ratio (`ASM-04`)
- **Literature's usual unit:** COPD Relative Risk ($\text{RR}$ or $\text{OR}$) for solid-fuel exposure vs. clean fuel, sex-stratified $\text{RR}_F, \text{RR}_M$, or female-to-male personal $\text{PM}_{2.5}$ exposure ratios.
- **Code's expected unit:** Log-linear slope $\beta_u \le 0$ (`slope_clean_fuel`) and ratio $r_{f/m,u} > 0$ (`sex_elasticity_ratio_f_to_m`) applied to **already HAP-attributable** 2040 COPD DALYs (`bau_dalys_2040`).
- **Conversion:**
  - **Primary source (Stage 2 calibration):** Person A fits $\ln\!\left(\frac{\text{DALYs}_{u,\text{male},2040}(\text{cf}_k)}{\text{DALYs}_{u,\text{male},2040}(\text{bau})}\right) = \beta_u (c_k - c_{\text{bau}})$ across counterfactual anchors `cf_25..cf_100`.
  - **Literature cross-check for $\beta_u$:** Because `bau_dalys_2040` is **HAP-attributable** COPD burden (not total COPD), eliminating $\Delta c$ of remaining solid-fuel share $s_{\text{bau}} = 1 - c_{\text{bau},2040}$ under linear PAF reduces HAP-attributable burden by $\frac{\Delta c}{s_{\text{bau}}}$, implying local slope $\beta_u \approx \frac{\ln(1 - \Delta c_{\text{ref}} / s_{\text{bau}})}{\Delta c_{\text{ref}}}$ (adjusted for GBD IER/MR-BRT concavity at high solid-fuel exposure).
  - **Sex elasticity ratio $r_{f/m,u}$:** From sex-specific burden multipliers over the same exposure shift $\Delta c$:
    $$r_{f/m, u} = \frac{\ln(M_{u, \text{female}}(\Delta c))}{\ln(M_{u, \text{male}}(\Delta c))}$$
    or approximated from sex-specific excess-risk/personal-exposure ratios in South Asian kitchen exposure studies.
- **Extra inputs needed:** Stage 2 counterfactuals (`forecasts_stage3.csv`), $c_{\text{bau}, 2040}$, GBD 2023 COPD IER/MR-BRT risk curves, and sex-stratified COPD RRs / personal $\text{PM}_{2.5}$ exposure ratios.

### 3.7 Discount Factor (`ASM-16`), MDP Shocks (`ASM-15`), & Initial Infra Flags
- **Discount factor (`discount_factor`, `planning.mdp.discount_factor`):** Convert annual real discount rate $r$ (e.g., from WHO-CHOICE / iDSI / national HTA guidelines) to 5-year factor via $\gamma = (1 + r)^{-5}$.
- **MDP transition probabilities (`transition_probs`) & shock multipliers (`shock_multipliers`):** Derive `[p_low, p_nom, p_high]` and `[m_low, m_nom, m_high]` from empirical programme realization ratios ($\text{Achieved adoption} / \text{Planned adoption}$) in Indian/Nepalese clean-cooking evaluations via 3-bin quantile or moment-matching. *(Note: update `mdp.py` `_apply_action_branch` if `shock_multipliers` are moved away from `[0.5, 1.0, 1.25]`.)*
- **Initial infrastructure flags (`lpg_infra_ready_init`, `grid_ready_init`):** Define a transparent boolean threshold (e.g., `%` of unit households within LPG distributor delivery coverage / reliable grid feeder capacity adequate for induction cooking) using national energy-access reports (MoPNG/PPAC/CEA for India; NOC/NEA for Nepal).

---

## 4. Master Sourcing Worksheet (Blank Template to Fill In)

Fill in `Low`, `Central`, `High`, `Source` (full citation, DOI, or URL + access date), and `Status` (`placeholder: needs literature` $\to$ `sourced`). Keep all cost figures in a single documented constant base-year USD before scaling to `USD millions`.

### 4.1 Global & Planning Parameters

| ASM ID | Parameter / Key Path | Scope | Code Unit | Low | Central | High | Source (Citation / DOI / URL) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-13` | `total_budget_usd_millions` | All 4 units, `2026–2040` (15 yrs) | USD millions (`$M`) | | | | | |
| `ASM-16` | `discount_factor` & `planning.mdp.discount_factor` | `P1–P3` (5-yr period) | Dimensionless `(0, 1]` | | | | | |
| `ASM-16` | `apply_cost_discounting` | Contract 4 cost toggle | Boolean (`true`/`false`) | | | | | |
| `ASM-15` | `planning.mdp.transition_probs` | `[p_low, p_nom, p_high]` | Probability triplet (`sum=1`) | | | | | |
| `ASM-15` | `planning.compare.shock_multipliers` | `[m_low, m_nom, m_high]` | Dimensionless multipliers | | | | | |

### 4.2 Auxiliary Conversion Inputs (Needed Before Computing Unit Costs & Adoption Caps)

| Aux ID | Input Needed | Scope | Unit | Low | Central | High | Source (Citation / DOI / URL) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `AUX-01` | Average household size ($\bar{s}_u$) | `IND_urban` | Persons / household | | | | | |
| `AUX-01` | Average household size ($\bar{s}_u$) | `IND_rural` | Persons / household | | | | | |
| `AUX-01` | Average household size ($\bar{s}_u$) | `NPL_urban` | Persons / household | | | | | |
| `AUX-01` | Average household size ($\bar{s}_u$) | `NPL_rural` | Persons / household | | | | | |
| `AUX-02` | Base-year USD, FX rates (`INR/USD`, `NPR/USD`), & CPI deflator | `IND`, `NPL` | Rate / Index | | | | | |
| `AUX-03` | Cost boundary definition (fiscal subsidy vs. societal) | All interventions & budget | Documentation rule | | | | | |

### 4.3 Per-Unit Parameters (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`)

| ASM ID | Parameter / Key Path | Scope (Unit / Intervention) | Code Unit | Low | Central | High | Source (Citation / DOI / URL) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-05` | `unit_costs_usd_millions.IND_urban.lpg_subsidy` | `IND_urban` / `lpg_subsidy` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-05` | `unit_costs_usd_millions.IND_rural.lpg_subsidy` | `IND_rural` / `lpg_subsidy` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-05` | `unit_costs_usd_millions.NPL_urban.lpg_subsidy` | `NPL_urban` / `lpg_subsidy` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-05` | `unit_costs_usd_millions.NPL_rural.lpg_subsidy` | `NPL_rural` / `lpg_subsidy` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-06` | `unit_costs_usd_millions.IND_urban.improved_biomass` | `IND_urban` / `improved_biomass` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-06` | `unit_costs_usd_millions.IND_rural.improved_biomass` | `IND_rural` / `improved_biomass` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-06` | `unit_costs_usd_millions.NPL_urban.improved_biomass` | `NPL_urban` / `improved_biomass` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-06` | `unit_costs_usd_millions.NPL_rural.improved_biomass` | `NPL_rural` / `improved_biomass` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-07` | `unit_costs_usd_millions.IND_urban.electric_induction` | `IND_urban` / `electric_induction` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-07` | `unit_costs_usd_millions.IND_rural.electric_induction` | `IND_rural` / `electric_induction` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-07` | `unit_costs_usd_millions.NPL_urban.electric_induction` | `NPL_urban` / `electric_induction` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-07` | `unit_costs_usd_millions.NPL_rural.electric_induction` | `NPL_rural` / `electric_induction` | `$M` per `1.0` cov (5-yr) | | | | | |
| `ASM-08` | `unit_costs_usd_millions.IND_urban.infra_lpg_grid` | `IND_urban` / `infra_lpg_grid` | `$M` fixed per build | | | | | |
| `ASM-08` | `unit_costs_usd_millions.IND_rural.infra_lpg_grid` | `IND_rural` / `infra_lpg_grid` | `$M` fixed per build | | | | | |
| `ASM-08` | `unit_costs_usd_millions.NPL_urban.infra_lpg_grid` | `NPL_urban` / `infra_lpg_grid` | `$M` fixed per build | | | | | |
| `ASM-08` | `unit_costs_usd_millions.NPL_rural.infra_lpg_grid` | `NPL_rural` / `infra_lpg_grid` | `$M` fixed per build | | | | | |
| `ASM-09` | `effectiveness.IND_urban.lpg_subsidy` | `IND_urban` / `lpg_subsidy` | Dimensionless `[0, 1]` | | | | | |
| `ASM-09` | `effectiveness.IND_rural.lpg_subsidy` | `IND_rural` / `lpg_subsidy` | Dimensionless `[0, 1]` | | | | | |
| `ASM-09` | `effectiveness.NPL_urban.lpg_subsidy` | `NPL_urban` / `lpg_subsidy` | Dimensionless `[0, 1]` | | | | | |
| `ASM-09` | `effectiveness.NPL_rural.lpg_subsidy` | `NPL_rural` / `lpg_subsidy` | Dimensionless `[0, 1]` | | | | | |
| `ASM-10` | `effectiveness.IND_urban.improved_biomass` | `IND_urban` / `improved_biomass` | Dimensionless `[0, 1]` | | | | | |
| `ASM-10` | `effectiveness.IND_rural.improved_biomass` | `IND_rural` / `improved_biomass` | Dimensionless `[0, 1]` | | | | | |
| `ASM-10` | `effectiveness.NPL_urban.improved_biomass` | `NPL_urban` / `improved_biomass` | Dimensionless `[0, 1]` | | | | | |
| `ASM-10` | `effectiveness.NPL_rural.improved_biomass` | `NPL_rural` / `improved_biomass` | Dimensionless `[0, 1]` | | | | | |
| `ASM-11` | `effectiveness.IND_urban.electric_induction` | `IND_urban` / `electric_induction` | Dimensionless `[0, 1]` | | | | | |
| `ASM-11` | `effectiveness.IND_rural.electric_induction` | `IND_rural` / `electric_induction` | Dimensionless `[0, 1]` | | | | | |
| `ASM-11` | `effectiveness.NPL_urban.electric_induction` | `NPL_urban` / `electric_induction` | Dimensionless `[0, 1]` | | | | | |
| `ASM-11` | `effectiveness.NPL_rural.electric_induction` | `NPL_rural` / `electric_induction` | Dimensionless `[0, 1]` | | | | | |
| `ASM-12` | `coverage_ceiling.IND_urban` | `IND_urban` (2040 max total cov) | Fraction `(0, 1]` | | | | | |
| `ASM-12` | `coverage_ceiling.IND_rural` | `IND_rural` (2040 max total cov) | Fraction `(0, 1]` | | | | | |
| `ASM-12` | `coverage_ceiling.NPL_urban` | `NPL_urban` (2040 max total cov) | Fraction `(0, 1]` | | | | | |
| `ASM-12` | `coverage_ceiling.NPL_rural` | `NPL_rural` (2040 max total cov) | Fraction `(0, 1]` | | | | | |
| `ASM-14` | `units.IND_urban.slope_clean_fuel` | `IND_urban` (Contract 3 cross-check) | Log ratio / `1.0` cov (`<= 0`) | | | | | |
| `ASM-14` | `units.IND_rural.slope_clean_fuel` | `IND_rural` (Contract 3 cross-check) | Log ratio / `1.0` cov (`<= 0`) | | | | | |
| `ASM-14` | `units.NPL_urban.slope_clean_fuel` | `NPL_urban` (Contract 3 cross-check) | Log ratio / `1.0` cov (`<= 0`) | | | | | |
| `ASM-14` | `units.NPL_rural.slope_clean_fuel` | `NPL_rural` (Contract 3 cross-check) | Log ratio / `1.0` cov (`<= 0`) | | | | | |
| `ASM-04` | `units.IND_urban.sex_elasticity_ratio_f_to_m` | `IND_urban` (Contract 3) | Ratio `> 0` | | | | | |
| `ASM-04` | `units.IND_rural.sex_elasticity_ratio_f_to_m` | `IND_rural` (Contract 3) | Ratio `> 0` | | | | | |
| `ASM-04` | `units.NPL_urban.sex_elasticity_ratio_f_to_m` | `NPL_urban` (Contract 3) | Ratio `> 0` | | | | | |
| `ASM-04` | `units.NPL_rural.sex_elasticity_ratio_f_to_m` | `NPL_rural` (Contract 3) | Ratio `> 0` | | | | | |
| `ASM-14` | `units.IND_urban.uncertainty_sd` | `IND_urban` (Contract 3) | Fractional SD `>= 0` | | | | | |
| `ASM-14` | `units.IND_rural.uncertainty_sd` | `IND_rural` (Contract 3) | Fractional SD `>= 0` | | | | | |
| `ASM-14` | `units.NPL_urban.uncertainty_sd` | `NPL_urban` (Contract 3) | Fractional SD `>= 0` | | | | | |
| `ASM-14` | `units.NPL_rural.uncertainty_sd` | `NPL_rural` (Contract 3) | Fractional SD `>= 0` | | | | | |
| — | `lpg_infra_ready_init` / `grid_ready_init` | `IND_urban` (Contract 2) | Boolean (`true`/`false`) | | | | | |
| — | `lpg_infra_ready_init` / `grid_ready_init` | `IND_rural` (Contract 2) | Boolean (`true`/`false`) | | | | | |
| — | `lpg_infra_ready_init` / `grid_ready_init` | `NPL_urban` (Contract 2) | Boolean (`true`/`false`) | | | | | |
| — | `lpg_infra_ready_init` / `grid_ready_init` | `NPL_rural` (Contract 2) | Boolean (`true`/`false`) | | | | | |

### 4.4 Per-Period Adoption Caps (`ASM-12`: `max_adoption_per_period.<unit>.<intervention>.<period>`)

*(Repeat or differentiate across `P1`, `P2`, `P3` where ramp-up accelerates or saturates over time; `infra_lpg_grid` remains `1.0`.)*

| ASM ID | JSON Key Path | Scope (Unit × Intervention × Period) | Code Unit | Low | Central | High | Source (Citation / DOI / URL) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-12` | `max_adoption_per_period.IND_urban.lpg_subsidy.{P1,P2,P3}` | `IND_urban` / `lpg_subsidy` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.IND_rural.lpg_subsidy.{P1,P2,P3}` | `IND_rural` / `lpg_subsidy` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.NPL_urban.lpg_subsidy.{P1,P2,P3}` | `NPL_urban` / `lpg_subsidy` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.NPL_rural.lpg_subsidy.{P1,P2,P3}` | `NPL_rural` / `lpg_subsidy` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.IND_urban.improved_biomass.{P1,P2,P3}` | `IND_urban` / `improved_biomass` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.IND_rural.improved_biomass.{P1,P2,P3}` | `IND_rural` / `improved_biomass` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.NPL_urban.improved_biomass.{P1,P2,P3}` | `NPL_urban` / `improved_biomass` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.NPL_rural.improved_biomass.{P1,P2,P3}` | `NPL_rural` / `improved_biomass` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.IND_urban.electric_induction.{P1,P2,P3}` | `IND_urban` / `electric_induction` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.IND_rural.electric_induction.{P1,P2,P3}` | `IND_rural` / `electric_induction` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.NPL_urban.electric_induction.{P1,P2,P3}` | `NPL_urban` / `electric_induction` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |
| `ASM-12` | `max_adoption_per_period.NPL_rural.electric_induction.{P1,P2,P3}` | `NPL_rural` / `electric_induction` / `P1–P3` | Fraction / 5-yr `[0, 1]` | | | | | |

---

## 5. Hand-Off Checklist After Sourcing

1. Update `docs/ASSUMPTIONS.md` (`ASM-04` through `ASM-16`) with exact values, low/central/high sensitivity ranges, DOIs/URLs, and change `Status` from `placeholder: needs literature` to `sourced`.
2. Populate `configs/intervention_params.json` (`Central` values), set `data_version` and `generated_at` (ISO-8601 UTC), and keep `is_mock: false`.
3. Coordinate with Person A before overriding `ASM-04`/`ASM-14` in `data/processed/exposure_response_surrogate.json` or `lpg_infra_ready_init`/`grid_ready_init` in `data/processed/baseline_unit_state.csv`.
4. Update `planning.mdp.*` / `planning.compare.*` in `configs/default.yaml` if `ASM-15`, `ASM-16`, or `cost_weight_lambda` (re-calibrated from the real Stage 3 Pareto front) change, and verify `make check` passes.
