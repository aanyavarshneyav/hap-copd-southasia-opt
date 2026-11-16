# Track Interfaces & Data Contracts

Person A (Rudraksh — Data & Forecasting) and Person B (Aanya — Optimisation & Planning) work in parallel from **Day 1**. This document locks the file paths, schemas, units, and mock-to-real hand-off rules so Track B never blocks on Stage 1–2 completion.

> **Change Rule:** Neither track may rename columns, change units, or alter file paths listed below without updating this document and notifying the other owner first.

---

## 1. Dependency Timeline

```mermaid
flowchart LR
    D1["Day 1: Lock INTERFACES.md + Generate *_mock.* fixtures"] --> PA1["Person A: Data ingestion & validation"]
    D1 --> PB1["Person B: Build NSGA-II, MDP, HTN, PDDL against *_mock.*"]
    PA1 --> PA2["Person A: Stage 1 Historical Analysis"]
    PA2 --> PA3["Person A: Stage 2 Forecasting & Surrogate Calibration"]
    PA3 --> H["Handoff Checkpoint: Validate real contracts & flip use_mock_forecasts: false"]
    PB1 --> H
    H --> FINAL["Both Tracks: Final equal-footing Stage 3 + Planning runs & Report"]
```

| Phase | Rudraksh (Person A) Deliverable | Aanya (Person B) Deliverable | Interface State |
| --- | --- | --- | --- |
| **Hour 1 / M0** | Pair on `*_mock.*` schema & build `src/data/contracts.py` validator | Build `*_mock.*` fixtures tailored to evaluator (`B-00`), then start `B-01` | `interfaces.use_mock_forecasts: true` |
| **Stage 1–2 Dev** | Real `data/interim/` + Stage 1 + Stage 2 CV & forecasts | Develop & unit-test NSGA-II, MDP, HTN, STRIPS/PDDL using mocks | Mock contracts validated by tests |
| **Handoff** | Write real `forecasts_stage3.csv`, `baseline_unit_state.csv`, `exposure_response_surrogate.json` | Run contract validation + switch `use_mock_forecasts: false` | Schema parity verified |
| **Final Runs** | Final Stage 1–2 figures + Stage 2 re-evaluation of top Pareto points | Final Stage 3 Pareto front, MDP vs GA table, PDDL/HTN traces | Real data locked |

---

## 2. Contract 1: Stage 3 & MDP Forecast Table
- **Production file:** `data/processed/forecasts_stage3.csv` (+ `.meta.json`)
- **Day-1 Mock file:** `data/processed/forecasts_stage3_mock.csv` (+ `.meta.json`)
- **Granularity:** One row per `location × year × sex × scenario × measure × metric`.
  - `location` (4 units): `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`
  - `year`: `bau` covers all 16 annual years `2025–2040`; counterfactual anchors (`cf_25`, `cf_50`, `cf_75`, `cf_100`) are required at milestone years `{2025, 2030, 2035, 2040}` at minimum (validators also accept full annual `2025–2040` supersets).
  - `sex`: `female`, `male`, `both`
  - `scenario`: `bau` (business-as-usual clean-fuel trajectory) + reference counterfactual anchors (`cf_25`, `cf_50`, `cf_75`, `cf_100` representing clean-fuel share targets, used to cross-check the surrogate)
  - `measure`: `DALYs`, `Deaths`
  - `metric`: `Number`, `Rate_per_100k` (Age-standardised rate for `Rate_per_100k`)

### Schema
| Column | Dtype | Units / Allowed Values | Notes |
| --- | --- | --- | --- |
| `location` | `str` | `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural` | Exact decision unit ID |
| `year` | `int` | `2025` .. `2040` | Calendar year |
| `sex` | `str` | `female`, `male`, `both` | Required for equity objective |
| `scenario` | `str` | `bau`, `cf_25`, `cf_50`, `cf_75`, `cf_100` | Driver trajectory |
| `measure` | `str` | `DALYs`, `Deaths` | Burden measure |
| `metric` | `str` | `Number`, `Rate_per_100k` | `Rate_per_100k` = ASR |
| `val` | `float` | `>= 0.0` | Point / mean forecast |
| `lower` | `float` | `>= 0.0, <= val` | 95% UI lower bound |
| `upper` | `float` | `>= val` | 95% UI upper bound |
| `population` | `float` | Persons (`> 0`) | Apportioned UN WPP 2024 pop (sex-specific on `female`/`male` rows; total on `both` rows) |
| `clean_fuel_share` | `float` | `[0.0, 1.0]` | Assumed/projected clean-fuel coverage |
| `model_name` | `str` | e.g., `hist_gbdt_panel`, `poisson_apc`, `mock_generator` | Best validated Stage 2 model |
| `is_mock` | `bool` | `true` or `false` | Must be `false` for final results |
| `data_version` | `str` | e.g., `GBD_2023_WPP_2024` or `MOCK_V1` | Provenance tag |

### Example Rows *(Synthetic Schema Illustration Only — Do Not Use as Data)*
```csv
location,year,sex,scenario,measure,metric,val,lower,upper,population,clean_fuel_share,model_name,is_mock,data_version
IND_rural,2030,female,bau,DALYs,Rate_per_100k,0.0,0.0,0.0,1.0,0.50,mock_generator,true,MOCK_V1
NPL_rural,2040,both,bau,DALYs,Number,0.0,0.0,0.0,1.0,0.40,mock_generator,true,MOCK_V1
```
*(Zeroed placeholder values above illustrate column types only. Day-1 mock generator fills deterministic synthetic baselines clearly flagged `is_mock=true`.)*

---

## 3. Contract 2: Baseline Unit State (`t_0 = 2025`)
- **Production file:** `data/processed/baseline_unit_state.csv`
- **Day-1 Mock file:** `data/processed/baseline_unit_state_mock.csv`
- **Granularity:** One row per `location` (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`).

| Column | Dtype | Units / Allowed Values | Notes |
| --- | --- | --- | --- |
| `location` | `str` | `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural` | Primary key |
| `country` | `str` | `India`, `Nepal` | Parent country |
| `residence` | `str` | `urban`, `rural` | Residence stratum |
| `baseline_year` | `int` | `2025` (or latest observed + short-term step) | Pre-intervention anchor |
| `population_total` | `float` | Persons | Apportioned WPP 2024 population |
| `population_female_share` | `float` | `(0.0, 1.0)` | Female share |
| `clean_fuel_share_init` | `float` | `[0.0, 1.0]` | Starting clean-cooking access (SDG 7.1.2) |
| `lpg_infra_ready_init` | `bool` | `true` / `false` | Starting LPG supply readiness |
| `grid_ready_init` | `bool` | `true` / `false` | Starting electric cooking grid readiness |
| `bau_dalys_2040_female` | `float` | DALYs (`>= 0`) | Baseline 2040 projected DALYs (female) |
| `bau_dalys_2040_male` | `float` | DALYs (`>= 0`) | Baseline 2040 projected DALYs (male) |
| `bau_asr_2040_female` | `float` | Rate per 100k (`>= 0`) | Baseline 2040 projected ASR (female) |
| `bau_asr_2040_male` | `float` | Rate per 100k (`>= 0`) | Baseline 2040 projected ASR (male) |
| `is_mock` | `bool` | `true` / `false` | Flag |
| `data_version` | `str` | Version string | Provenance |

---

## 4. Contract 3: Fast Exposure–Response Surrogate
- **Production file:** `data/processed/exposure_response_surrogate.json`
- **Day-1 Mock file:** `data/processed/exposure_response_surrogate_mock.json`
- **Purpose:** Lets NSGA-II and MDP compute DALYs averted in microseconds on CPU instead of re-running ML/APC inside every fitness evaluation.
- **Required JSON Keys:**
```json
{
  "schema_version": "1.0",
  "is_mock": true,
  "data_version": "MOCK_V1",
  "functional_form": "log_linear_or_piecewise_paf",
  "generated_at": "YYYY-MM-DDTHH:MM:SSZ",
  "source_model": "stage2_best_model_or_placeholder",
  "units": {
    "IND_urban": {"slope_clean_fuel": null, "baseline_clean_fuel": null, "sex_elasticity_ratio_f_to_m": null, "uncertainty_sd": null},
    "IND_rural": {"slope_clean_fuel": null, "baseline_clean_fuel": null, "sex_elasticity_ratio_f_to_m": null, "uncertainty_sd": null},
    "NPL_urban": {"slope_clean_fuel": null, "baseline_clean_fuel": null, "sex_elasticity_ratio_f_to_m": null, "uncertainty_sd": null},
    "NPL_rural": {"slope_clean_fuel": null, "baseline_clean_fuel": null, "sex_elasticity_ratio_f_to_m": null, "uncertainty_sd": null}
  }
}
```
*(In `*_mock.json`, Aanya populates clearly labelled placeholder numerical parameters derived from `docs/ASSUMPTIONS.md`—paired with `src/data/contracts.py`—so evaluator math executes cleanly before Stage 2 finishes.)*

### Surrogate Mathematical Specification (`log_linear`)
Let $\Delta c_u \ge 0$ be the effective additional clean-cooking coverage above unit $u$'s 2040 BAU clean-fuel baseline (`baseline_clean_fuel`, matching `clean_fuel_share` at `year = 2040, scenario = bau`), capped so $c_{u,2040} \le 1.0$. With $\beta_u = \text{slope\_clean\_fuel}_u \le 0$ and $r_{f/m,u} = \text{sex\_elasticity\_ratio\_f\_to\_m}_u > 0$:
- **Sex-specific burden multipliers:**
  $$M_{u,\text{male}} = \exp(\beta_u \cdot \Delta c_u), \qquad M_{u,\text{female}} = \exp(r_{f/m,u} \cdot \beta_u \cdot \Delta c_u)$$
- **Mean 2040 projections:**
  $$\text{DALYs}_{u,s,2040}(x) = M_{u,s} \cdot \text{bau\_dalys\_2040}_{u,s}, \qquad \text{ASR}_{u,s,2040}(x) = M_{u,s} \cdot \text{bau\_asr\_2040}_{u,s}$$
  for $s \in \{\text{female}, \text{male}\}$, so $x = 0$ ($\Delta c_u = 0$) recovers exact 2040 BAU DALYs and ASRs.
- **Simplified 95% UI propagation:** Scale the 2040 BAU half-widths by $M_{u,s}$ and combine in quadrature with the surrogate uncertainty contribution $\delta_{u,s} = \text{mean}_{u,s} \cdot (1.96 \cdot \text{uncertainty\_sd}_u \cdot \Delta c_u)$:
  $$\text{lower}_{u,s} = \max\Big(0,\; \text{mean}_{u,s} - \sqrt{\big(M_{u,s}(\text{bau\_val}_{u,s} - \text{bau\_lower}_{u,s})\big)^2 + \delta_{u,s}^2}\Big)$$
  $$\text{upper}_{u,s} = \text{mean}_{u,s} + \sqrt{\big(M_{u,s}(\text{bau\_upper}_{u,s} - \text{bau\_val}_{u,s})\big)^2 + \delta_{u,s}^2}$$
  At $\Delta c_u = 0$, this preserves the exact 2040 BAU `(lower, val, upper)` bounds.

---

## 5. Contract 4: Intervention & Scenario Parameters (Person B Owned)
- **Production file (template until sourced):** `configs/intervention_params.json` (`is_mock: false`; parameter values remain `null` / `"TODO(owner)"` until literature values are logged in `docs/ASSUMPTIONS.md`).
- **Day-1 Mock file:** `data/processed/intervention_params_mock.json` (`is_mock: true`; synthetic placeholder values only).
- **Owner:** Person B (does not alter Person A's Contracts 1–3).
- **Required JSON Keys:**
  - Metadata: `schema_version` (`"1.0"`), `is_mock` (`bool`), `data_version` (`str`), `generated_at` (`ISO-8601 UTC`), `note` (`str`).
  - Global Stage 3 settings: `total_budget_usd_millions` (`float > 0`), `infra_build_threshold` (`float in (0, 1]`, e.g. `0.5`), `apply_cost_discounting` (`bool`, default `false`), `discount_factor` (`float in (0, 1]`, e.g. `0.774`), `period_order` (`["P1", "P2", "P3"]`).
  - Per-unit/intervention maps over `["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]`:
    - `coverage_ceiling`: `{unit: float in (0, 1]}` ($c^{\max}_u$).
    - `effectiveness`: `{unit: {lpg_subsidy: 1.0, improved_biomass: float in (0, 1), electric_induction: 1.0, infra_lpg_grid: 0.0}}`.
    - `unit_costs_usd_millions`: `{unit: {intervention: float >= 0}}` (for cooking interventions: USD millions per `1.0` full-unit coverage increment; for `infra_lpg_grid`: fixed USD millions per build indicator `x >= infra_build_threshold`).
    - `max_adoption_per_period`: `{unit: {intervention: {P1: float, P2: float, P3: float}}}` in `[0, 1]`.

---

## 6. Temporary Mock Strategy Rules
1. `configs/default.yaml` controls `interfaces.use_mock_forecasts`. When `true`, Stage 3 and Planning automatically read `*_mock.*` files and stamp all outputs with `is_mock: true` / `WARNING_MOCK_INPUTS: true` so mock runs can never be confused with final results.
2. Day-1 mock files are deterministic (seeded via `seeds.global`), lightweight (`<500 KB`, covering ~960 rows / ~80 KB for milestone-anchor counterfactuals + annual BAU, or ~3,840 rows / ~300 KB if all years are expanded), and committed under the explicit `!data/processed/*_mock.*` `.gitignore` exception.
3. At the Handoff Checkpoint, Person A runs `src/forecast/export_contracts.py` to generate real Contracts 1–3 and verifies schema parity using the shared contract test suite (`tests/`).
