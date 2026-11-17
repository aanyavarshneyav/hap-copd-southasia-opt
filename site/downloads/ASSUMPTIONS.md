# Scenario Assumptions Register

> **Honesty Rule:** Never invent numbers, costs, effect sizes, or citations. Every unverified parameter must remain marked `placeholder: needs literature` until a verified source is logged here. All Stage 3 and Planning outputs are **scenario analyses**, not real-world policy recommendations.

---

## 1. Methodological & Apportionment Assumptions

| ID | Category | Assumption / Formula | Scope | Value / Rule | Units | Source | Status | Sensitivity Range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-01` | Apportionment | **Urban/rural burden apportionment:** National GBD HAP-COPD burden $B_c$ is apportioned to urban ($u$) and rural ($r$) strata by population-weighted solid-fuel exposure: $w_k = \frac{\text{Pop}_{k} \cdot (1 - \text{CleanFuel}_k)}{\sum_{j \in \{u,r\}} \text{Pop}_j \cdot (1 - \text{CleanFuel}_j)}$, $B_{c,k} = w_k B_c$ | `IND`, `NPL` | Exposure-weighted share (assuming equal conditional COPD susceptibility given solid-fuel exposure as baseline) | Fraction | Methodological design; uses UN WPP/WUP + WHO SDG 7.1.2 | `sourced_formula` (inputs pending ingestion) | Test $\pm 20\%$ rural relative susceptibility ratio |
| `ASM-02` | Exposure driver | **GBD exposure fallback:** Use WHO SDG 7.1.2 `clean_fuel_share` ($c \in [0, 1]$) and implied `1 - c` solid-fuel exposure if GBD 2023 SEV is non-downloadable | 5-country panel + 4 units | `solid_share = 1 - clean_fuel_share` | Proportion | WHO GHO SDG 7.1.2 | `sourced_fallback` | Compare linear vs. concave exposure–response |
| `ASM-03` | Equity metric | **Female–male burden gap:** Population-weighted absolute age-standardised DALY rate difference $\sum_u p_u \left|\text{ASR}^F_u - \text{ASR}^M_u\right|$ | Stage 3 & MDP | `abs_asr_gap_female_male` | DALYs per 100k | Project design (`docs/METHODOLOGY.md`) | `sourced_formula` | Sensitivity against total absolute DALY gap $\|\text{DALYs}^F - \text{DALYs}^M\|$ |
| `ASM-04` | Sex exposure split | **Differential kitchen exposure elasticity:** Ratio of female-to-male relative risk response to clean-cooking shift | `IND`, `NPL` | `1.05` (`sex_elasticity_ratio_f_to_m`) | Ratio | Proxy scenario multiplier in `exposure_response_surrogate.json` | `proxy_assumption` | Disclosed as proxy (`1.0` sex-neutral alt) |
| `ASM-04b` | Sex baseline split | **Empirical HAP-COPD sex split:** Country-specific female vs male DALY shares calibrated from WHO GHE 2021 (`data (2).csv`): IND 53.46% F / 46.54% M; NPL 58.34% F / 41.66% M | 5-country panel | `female_share` from WHO GHE 2021 | Fraction | WHO GHE 2021 HAP-COPD dataset | `sourced_empirical` | Test $\pm 10\%$ shift in female share |
| `ASM-17` | Age modelling | **Aggregate ASR & Population Modelling:** Due to GBD single-risk query restrictions to aggregate ages, historical analysis and forecasting use ASRs and all-age counts, with Poisson GLM Rate modelling replacing discrete cohort APC | 5-country panel | ASR & All-ages counts | Rates per 100k | GBD 2023 Results Tool & WHO SDG 7.1.2 | `sourced_adaptation` | Compare against log-linear rate trends |

---

## 2. Intervention Costs, Effectiveness & Constraints (`REAL_V1_2024USD`)

| ID | Category | Parameter | Scope | Value / Range | Units | Source | Status | Sensitivity Range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-05` | Cost | **LPG subsidy** (`lpg_subsidy`) targeted public fiscal package | `IND_u`, `IND_r`, `NPL_u`, `NPL_r` | `IND: $84.33–$90.30/HH` (`$11,963.5M` / `$20,120.7M` full); `NPL: $99–$106/HH` (`$552.1M` / `$292.1M` full) | 2024 USD / HH (`USD M` for +1.0 cov) | MoPNG / PPAC / PRS India PMUY; Nepal NOC (`intervention_source_register.csv`) | `sourced` *(Alt A 1-time vs. Alt B opex)* | `Alt B` recurring refill opex (`+$1,308.3M` Knee) |
| `ASM-06` | Cost | **Improved biomass stoves** (`improved_biomass`) at 50% public co-finance | All units (focus `IND_r`, `NPL_r`) | `$22.50/HH` public (`$45.00/HH` full societal); `NPL_r $48.74M`, `IND_r $4,707.12M` full | 2024 USD / HH (`USD M`) | MNRE Unnat Chulha / South Asia ICS (`intervention_source_register.csv`) | `sourced_with_50pct_cofinance_assumption` | `100%` societal (`$45/HH`) tested in Factor 8 |
| `ASM-07` | Cost | **Electric / induction cooking** (`electric_induction`) at 50% public co-finance | `IND_u`, `IND_r`, `NPL_u`, `NPL_r` | `IND: $26.555/HH` (`$53.11` full); `NPL: $31.835/HH` (`$63.67` full) | 2024 USD / HH (`USD M`) | EESL NECP / AEPC Nepal / ESMAP (`intervention_source_register.csv`) | `sourced_with_50pct_cofinance_assumption` | `100%` societal (`$53.11/$63.67`) tested in Factor 8 |
| `ASM-08` | Cost | **LPG / grid infrastructure prerequisite** (`infra_lpg_grid`) | `IND_r`, `NPL_r` | **`null` (disabled in Scenario A)**; explicit proxies in Scenario B1 (`$150/$250/HH`) & B2 (`$300/$400/HH`) | USD Millions | MoP/RDSS & NEA benchmarks inconclusive (`parameter_gap_matrix.csv`) | `unresolved_null_disabled_scen_a` | Scenario B1/B2 tested (`+80.3k` DALYs) |
| `ASM-09` | Effectiveness | **Exposure reduction factor** (`lpg_subsidy`) | All units | `0.78` | Fractional HAP PM2.5 / PAF reduction | HAPIN India trial / CEEW PMUY (`doi:10.1016/j.envpol.2021.118198`) | `sourced` | Combined stacking sensitivity |
| `ASM-10` | Effectiveness | **Exposure reduction factor** (`improved_biomass`) | All units | `0.25` (sensitivity `[0.15, 0.30, 0.32, 0.50]`) | Fractional HAP PM2.5 reduction | Sarlahi Nepal RCT & Pope et al. 2017 (`doi:10.1016/S2214-109X(16)30024-9`) | `sourced_conservative` | `0.15–0.50` (`Factor 3` tornado top driver) |
| `ASM-11` | Effectiveness | **Exposure reduction factor** (`electric_induction`) | All units | `0.85` | Fractional HAP PM2.5 reduction | WHO IAQ Guidelines / ESMAP Nepal (`intervention_source_register.csv`) | `sourced` | Grid-prereq gated |
| `ASM-12` | Constraint | **Max feasible adoption rate & coverage ceilings** ($\Delta c^{\max}_{u,i,p}$, $c^{\max}_u$) | All units × P1–P3 | Caps `0.05–0.24` per 5-yr period; ceilings `[0.99, 0.95, 0.98, 0.95]`; infra caps `0.0` (Scen A) | % pts per 5 years | WHO GHO SDG 7.1.2 empirical max 5-yr gains (`2000–2024`) | `sourced_empirical` | Halved induction caps (`Factor 4`) |
| `ASM-13` | Constraint | **Total programme budget envelope** ($B_{\text{total}}$, 2026–2040) | India + Nepal | **`$5,985.73M`** (`50% Base` of `$11,971.47M` 2025 HH LPG gap; not a government appropriation) | USD Millions | B-09 fiscally derived (`b09_budget_scenarios.csv`) | `scenario_assumption` | `25% ($2,992.9M)`, `100% ($11,971.5M)`, derived-2024 (`$2,703.6M`), WHO (`$2,991.2M`) |
| `ASM-14` | Exposure–Response | **Clean-fuel coverage to COPD DALY elasticity** ($\beta_u$) | 4 units | Uniform pooled slope `-2.1933`, analytical SE `0.2557` (`HistGBDT` baseline `0.0 / -2.2794 / -2.8553 / -3.7254`, `sd=0.08` preserved as snapshot) | Elasticities | Stage 2 `log_ols_smooth` pooled counterfactual fit (`exposure_response_surrogate.json`, `DEC-029`) | `calibrated_stage2` | 500 Stage 2 + surrogate MC draws (`s3_pareto_reeval.csv`) |
| `ASM-15` | Planning | **MDP stochastic uptake probabilities** $(p_{\text{low}}, p_{\text{nom}}, p_{\text{high}})$ | MDP transitions | `(0.15, 0.70, 0.15)` on `[0.5×, 1.0×, 1.25×]` shocks; `lambda_cost = 116.0` | Probability / DALYs per `$1M` | Scenario design (`DEC-023` recalibrated `λ` from real Pareto slope) | `provisional_scenario` | `λ in [0..3500]`, `budget_bins in {6, 11}` |
| `ASM-16` | Planning | **Discount factor** ($\gamma$) per 5-year epoch | Stage 3 & MDP | `0.774` ($\approx 0.95^5$, ~5%/yr) for MDP value; Stage 3 costs undiscounted by default | Dimensionless | WHO-CHOICE / iDSI convention (`DEC-014`) | `provisional_default` | `gamma = 1.0`; discounted GA cost (`Factor 5`) |

---

## 3. Literature Sourcing Status
All non-infrastructure parameters (`ASM-05`–`ASM-17`) are populated in `configs/intervention_params.json` (`REAL_V1_2024USD`) and logged with source URLs/DOIs in `research/person_b_inputs/intervention_source_register.csv`. Only `ASM-08` (`infra_lpg_grid`) remains `null` (disabled in Scenario A and bracketed via explicit Scenario B1/B2 proxies).

