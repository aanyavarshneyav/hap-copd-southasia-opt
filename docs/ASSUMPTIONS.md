# Scenario Assumptions Register

> **Honesty Rule:** Never invent numbers, costs, effect sizes, or citations. Every unverified parameter must remain marked `placeholder: needs literature` until a verified source is logged here. All Stage 3 and Planning outputs are **scenario analyses**, not real-world policy recommendations.

---

## 1. Methodological & Apportionment Assumptions

| ID | Category | Assumption / Formula | Scope | Value / Rule | Units | Source | Status | Sensitivity Range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-01` | Apportionment | **Urban/rural burden apportionment:** National GBD HAP-COPD burden $B_c$ is apportioned to urban ($u$) and rural ($r$) strata by population-weighted solid-fuel exposure: $w_k = \frac{\text{Pop}_{k} \cdot (1 - \text{CleanFuel}_k)}{\sum_{j \in \{u,r\}} \text{Pop}_j \cdot (1 - \text{CleanFuel}_j)}$, $B_{c,k} = w_k B_c$ | `IND`, `NPL` | Exposure-weighted share (assuming equal conditional COPD susceptibility given solid-fuel exposure as baseline) | Fraction | Methodological design; uses UN WPP/WUP + WHO SDG 7.1.2 | `sourced_formula` (inputs pending ingestion) | Test $\pm 20\%$ rural relative susceptibility ratio |
| `ASM-02` | Exposure driver | **GBD exposure fallback:** Use WHO SDG 7.1.2 `clean_fuel_share` ($c \in [0, 1]$) and implied `1 - c` solid-fuel exposure if GBD 2023 SEV is non-downloadable | 5-country panel + 4 units | `solid_share = 1 - clean_fuel_share` | Proportion | WHO GHO SDG 7.1.2 | `sourced_fallback` | Compare linear vs. concave exposure–response |
| `ASM-03` | Equity metric | **Female–male burden gap:** Population-weighted absolute age-standardised DALY rate difference $\sum_u p_u \left|\text{ASR}^F_u - \text{ASR}^M_u\right|$ | Stage 3 & MDP | `abs_asr_gap_female_male` | DALYs per 100k | Project design (`docs/METHODOLOGY.md`) | `sourced_formula` | Sensitivity against total absolute DALY gap $\|\text{DALYs}^F - \text{DALYs}^M\|$ |
| `ASM-04` | Sex exposure split | **Differential kitchen exposure elasticity:** Ratio of female-to-male relative risk response to clean-cooking shift | `IND`, `NPL` | `TODO(owner) — placeholder: needs literature` | Ratio | `placeholder: needs literature` | `placeholder: needs literature` | Test `1.0` (sex-neutral) vs. literature differential |
| `ASM-04b` | Sex baseline split | **Empirical HAP-COPD sex split:** Country-specific female vs male DALY shares calibrated from WHO GHE 2021 (`data (2).csv`): IND 53.46% F / 46.54% M; NPL 58.34% F / 41.66% M | 5-country panel | `female_share` from WHO GHE 2021 | Fraction | WHO GHE 2021 HAP-COPD dataset | `sourced_empirical` | Test $\pm 10\%$ shift in female share |
| `ASM-17` | Age modelling | **Aggregate ASR & Population Modelling:** Due to GBD single-risk query restrictions to aggregate ages, historical analysis and forecasting use ASRs and all-age counts, with Poisson GLM Rate modelling replacing discrete cohort APC | 5-country panel | ASR & All-ages counts | Rates per 100k | GBD 2023 Results Tool & WHO SDG 7.1.2 | `sourced_adaptation` | Compare against log-linear rate trends |

---

## 2. Intervention Costs, Effectiveness & Constraints (Placeholders Pending Literature)

| ID | Category | Parameter | Scope | Value / Range | Units | Source | Status | Sensitivity Range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ASM-05` | Cost | **LPG subsidy** (`lpg_subsidy`) unit cost per household or +1% coverage | `IND_u`, `IND_r`, `NPL_u`, `NPL_r` | `TODO(owner) — placeholder: needs literature` | USD / HH / 5-yr (or USD M per +1% cov) | `placeholder: needs literature` | `placeholder: needs literature` | $\pm 30\%$ |
| `ASM-06` | Cost | **Improved biomass stoves** (`improved_biomass`) unit cost | `IND_r`, `NPL_r` (and peri-urban) | `TODO(owner) — placeholder: needs literature` | USD / HH / 5-yr | `placeholder: needs literature` | `placeholder: needs literature` | $\pm 30\%$ |
| `ASM-07` | Cost | **Electric / induction cooking** (`electric_induction`) unit cost | `IND_u`, `IND_r`, `NPL_u`, `NPL_r` | `TODO(owner) — placeholder: needs literature` | USD / HH / 5-yr | `placeholder: needs literature` | `placeholder: needs literature` | $\pm 30\%$ |
| `ASM-08` | Cost | **LPG / grid infrastructure prerequisite** (`infra_lpg_grid`) fixed/tiered cost | `IND_r`, `NPL_r` | `TODO(owner) — placeholder: needs literature` | USD Millions per region | `placeholder: needs literature` | `placeholder: needs literature` | $\pm 40\%$ |
| `ASM-09` | Effectiveness | **Exposure reduction factor** of `lpg_subsidy` (fuel stacking adjusted) | All units | `TODO(owner) — placeholder: needs literature` | Fractional HAP PM2.5 / PAF reduction | `placeholder: needs literature` | `placeholder: needs literature` | Low / Central / High uptake |
| `ASM-10` | Effectiveness | **Exposure reduction factor** of `improved_biomass` (field vs. lab efficiency) | Rural units | `TODO(owner) — placeholder: needs literature` | Fractional HAP PAF reduction | `placeholder: needs literature` | `placeholder: needs literature` | Low / Central / High |
| `ASM-11` | Effectiveness | **Exposure reduction factor** of `electric_induction` | All units | `TODO(owner) — placeholder: needs literature` | Fractional HAP PAF reduction | `placeholder: needs literature` | `placeholder: needs literature` | Low / Central / High |
| `ASM-12` | Constraint | **Max feasible adoption rate** per 5-year period ($\Delta c^{\max}_{u,i,p}$) | All units × P1–P3 | `TODO(owner) — placeholder: needs literature` | % pts per 5 years | `placeholder: needs literature` | `placeholder: needs literature` | $\pm 5\text{–}10$ pct pts |
| `ASM-13` | Constraint | **Total programme budget envelope** ($B_{\text{total}}$ across 2026–2040) | India + Nepal | `TODO(owner) — placeholder: needs literature` | USD Millions | `placeholder: needs literature` | `placeholder: needs literature` | Tight / Base / Expanded budget scenarios |
| `ASM-14` | Exposure–Response | **Clean-fuel coverage to COPD DALY elasticity** (calibrated from Stage 2 or GBD CRA literature) | 4 units | `TODO(Person A/B) — placeholder: needs literature` | Elasticities / IER curve | `placeholder: needs literature` (cross-checked with Stage 2 panel) | `placeholder: needs literature` | $\pm 1$ SE from Stage 2 / literature |
| `ASM-15` | Planning | **MDP stochastic uptake probabilities** $(p_{\text{low}}, p_{\text{nom}}, p_{\text{high}})$ | MDP transitions | `TODO(Person B) — provisional scenario: (0.15, 0.70, 0.15)` | Probability | Scenario design (`docs/PLANNING_SPEC.md`) | `placeholder: needs literature` | Deterministic `(0,1,0)` vs. high-variance `(0.25,0.50,0.25)` |
| `ASM-16` | Planning | **Discount factor** ($\gamma$) per 5-year epoch | Stage 3 & MDP | `0.774` ($\approx 0.95^5$, ~5%/yr) | Dimensionless | Standard health-economic discounting convention | `provisional_default` | `1.0` (undiscounted) vs. `0.774` |

---

## 3. Literature Sourcing Queue
`TODO(owner)` — Before Stage 3 final runs, fill `ASM-04` through `ASM-14` from peer-reviewed clean-cooking / PMUY (India) / Nepal clean-cooking studies and WHO/World Bank ESMAP reports, recording exact DOIs/URLs in the `Source` column and updating `Status` to `sourced`.
