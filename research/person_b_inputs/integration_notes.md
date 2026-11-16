# Integration Notes: Person B Parallel Auxiliary Data & Parameter Sourcing (`B-07`)

> **Scope & Isolation:** All files under `research/person_b_inputs/` were downloaded, derived, and validated independently by Person B on branch `b/param-sourcing`. No Person A files (`data/`, `src/data/`, `src/historical/`, `src/forecast/`), mock fixtures (`data/processed/*_mock.*`), or shared configs (`configs/default.yaml`, `configs/intervention_params.json`, `docs/*`) were modified. `interfaces.use_mock_forecasts` remains `true`.

---

## 1. Demographic Data & Urban/Rural Definition Reconciliation (`demographic_inputs.csv`)

### 1.1 National Definitions vs. UN Harmonized DEGURBA
UN World Urbanization Prospects (WUP) 2025 publishes two distinct urban/rural series in `research/person_b_inputs/raw/`:
1. **National Definitions (`WUP2025-DB-National-Definitions-Population-Data.csv.gz`):**
   - **India:** Uses Census of India statutory towns and Census towns ($\ge 5{,}000$ inhabitants, $\ge 75\%$ male non-agricultural workers, density $\ge 400/\text{km}^2$), giving **35.69% urban** (`522.44M`) / **64.31% rural** (`941.42M`) in 2025, rising to **40.21% urban** (`652.45M`) / **59.79% rural** (`970.13M`) in 2040.
   - **Nepal:** Uses Nepal NPHC 2021 federal municipal classifications (293 Urban Municipalities vs. 460 Rural Municipalities / Gaunpalikas), giving **67.00% urban** (`19.84M`) / **33.00% rural** (`9.77M`) in 2025, rising to **70.02% urban** (`23.04M`) / **29.98% rural** (`9.87M`) in 2040.
2. **UN Harmonized Degree of Urbanization Level 1 (`WUP2025-DB-DEGURBA-Level1-Population-Surface-Data.csv.gz`):**
   - Classifies $1\,\text{km}^2$ grid cells into **Cities** (urban centres), **Towns** (urban clusters/semi-dense areas), and **Rural** grid cells.
   - Under DEGURBA Level 1 (Cities + Towns = Urban), India is **84.06% urban** (`1,230.59M`) / **15.94% rural** (`233.28M`) in 2025, and Nepal is **73.28% urban** (`21.70M`) / **26.72% rural** (`7.91M`) in 2025.

**Integration Rule:** Both Person A's `ASM-01` apportionment and Stage 3 unit costing MUST use **WUP 2025 National Definitions**, NOT DEGURBA, because WHO/World Bank SDG 7.1.2 and DHS/NFHS clean-cooking surveys stratify respondents by **national** urban/rural frames. Cross-check: pairing 2024 WHO SDG 7.1.2 (`IND_urban=96%`, `IND_rural=69%`) with WUP 2025 National Definitions (`35.69%` urban / `64.31%` rural) yields a population-weighted national clean-cooking share of $0.3569(96) + 0.6431(69) = 78.6\%$ (closely tracking WHO's reported `81%`), whereas DEGURBA (`84.06%` urban) would yield $91.7\%$ and severely mis-apportion India's rural burden.

### 1.2 Mock vs. Real Structural Flips
- **Nepal urban/rural reversal:** `src/data/mock.py` assumed `NPL_urban = 7.0M` (23%) and `NPL_rural = 23.0M` (77%). Under real Nepal NPHC 2021 (`19.30M` urban / `9.87M` rural) and WUP 2025 National Definitions (`19.84M` urban / `9.77M` rural), **Nepal is ~67% urban by municipal classification**, although clean-cooking access in `NPL_urban` remains only `64%` (WHO 2024) because newly reclassified municipalities retain substantial semi-rural biomass reliance (`39.32%` firewood in NPHC 2021 Table 07).
- **2025 Household Count Derivation (Explicitly Flagged `derived`):**
  - India's 2021 Census was delayed, so 2011 Census Table A-1 (`80.89M` urban HH, `168.61M` rural HH) cannot be treated as 2025 actuals. Using WUP 2025 National-Definition populations divided by official **NFHS-5 (2019–21)** mean household sizes from the DHS Program API (`4.2` urban, `4.5` rural) gives **124,390,773 `IND_urban`** and **209,205,395 `IND_rural`** households in 2025 (total `333.60M`, consistent with MoPNG's `328.3M` active domestic LPG connections as of Nov 2024).
  - Nepal's 2025 households divide WUP 2025 National-Definition populations by **NPHC 2021 Table 15** exact observed household sizes (`4.3076` urban, `4.5115` rural), giving **4,606,688 `NPL_urban`** and **2,166,519 `NPL_rural`** households (vs. `4,479,662` and `2,187,275` observed in 2021). Note that Nepal NDHS 2022 reports smaller *de jure* usual-resident sizes (`3.9` / `4.0`) because absent migrants abroad are excluded.

---

## 2. Clean-Fuel Coverage & Denominator Integrity (`clean_fuel_access.csv`)

- **Latest Common Baseline Year:** **2024** for both WHO GHO (`PHE_HHAIR_PROP_POP_CLEAN_FUELS`) and World Bank WDI (`EG.CFT.ACCS.*`) across Total, Urban, and Rural strata (`0.0` percentage point discrepancy across all overlapping 2000–2024 records). Fuel-category disaggregation (`PHE_HHAIR_PROP_POP_CATEGORY_FUELS`) runs `1990–2023` (latest year **2023**).
- **2024 WHO/WB SDG 7.1.2 Baseline (% of population, people-denominated):**
  - `IND_urban`: **96%** (95% UI: `88%–100%`) vs. mock `78%`
  - `IND_rural`: **69%** (95% UI: `53%–82%`) vs. mock `42%`
  - `NPL_urban`: **64%** (95% UI: `34%–93%`) vs. mock `64%`
  - `NPL_rural`: **25%** (95% UI: `14%–40%`) vs. mock `24%`
- **People vs. Households Denominator Warning:** SDG 7.1.2 measures **percentage of population**, NOT percentage of households. Direct DHS API comparison (`dhs_ind_npl_hh_size_and_clean_fuel.json`) confirms a consistent **household > population** gap because larger households rely more heavily on solid fuels:
  - India NFHS-5 (2019–21): Urban `89.7%` HH vs `89.0%` pop; Rural `43.2%` HH vs `40.9%` pop (`-2.3` pct pts).
  - Nepal NDHS 2022: Urban `56.7%` HH vs `52.5%` pop (`-4.2` pct pts); Rural `21.2%` HH vs `19.3%` pop (`-1.9` pct pts).
  - Therefore, `clean_fuel_access.csv` retains explicit `unit_basis` (`people` vs `households`) without automatic conversion.

---

## 3. Contract 4 Integration Implications (`proposed_contract4_table.csv` & `proposed_intervention_params.json`)

1. **Order-of-Magnitude Scale Shift (`IND_rural` vs `NPL_rural`):**
   - Scaling real 2024 USD/HH costs by real 2025 derived households produces `+1.0` full-coverage 5-year LPG subsidy costs of **`$11,963.51M` (`IND_urban`)**, **`$20,120.71M` (`IND_rural`)**, **`$552.11M` (`NPL_urban`)**, and **`$292.11M` (`NPL_rural`)**—roughly **30× larger** for India than the synthetic mock (`$320M` / `$620M`), while `IND_rural` is now **~69× larger** in full-coverage cost than `NPL_rural` (`$20.12B` vs `$292.1M`).
   - Consequently, once real Stage 3 data is wired, `planning.mdp.cost_weight_lambda` (currently `3500.0` DALYs/\$1M from mock Pareto slope) will need recalibration via `calibrate_lambda_from_pareto()`, because real `DALYs / $M` slopes will be roughly an order of magnitude smaller unless DALYs scale proportionally!
2. **Cost Boundary Harmonization Needed Before Production:**
   - **LPG (`lpg_subsidy`)** is sourced on a **public-fiscal subsidy basis** (India PMUY ₹2,050 connection + ₹300 × 4 refills/yr; Nepal NOC loss NPR 450 × 6 cyl/yr + assumed deposit).
   - **Improved biomass (`improved_biomass`)** and **electric induction (`electric_induction`)** currently reflect **societal hardware/O&M** (`$25–$85/HH/5yr`) and **appliance+wiring retail/procurement** (`$22.70–$101.43/HH/5yr`, excluding ongoing electricity tariffs), because India has no central cash subsidy scheme for induction equivalent to PMUY. Before locking `configs/intervention_params.json`, the team must choose **one consistent boundary** (either *full societal cost including unsubsidized fuel* for all 3 fuels, or *public-fiscal programme subsidy* with an explicit scenario subsidy rate for induction/ICS).
3. **Unresolved Gap (`infra_lpg_grid` Fixed `$M` Capex & `total_budget_usd_millions`):**
   - No primary public dataset isolates regional distribution transformer/feeder reinforcement + LPG depot capex specifically for rural cooking transitions (`Saubhagya` `$68/HH` and `RDSS` `$571/HH` cover basic electrification; `NEA` `$15–$38/HH` covers 15A household meters only). `infra_lpg_grid` unit costs and pooled `total_budget_usd_millions` remain **`null`** (`unresolved_gap` / `assumption`) until the team defines a cost proxy or scenario range.
4. **Tight `IND_urban` Headroom vs. MDP Discretization:**
   - With `IND_urban` already at **96%** in 2024, 2040 BAU will likely sit at `~0.97–0.98`. If `coverage_ceiling.IND_urban = 0.99`, remaining headroom is only `0.01–0.02`, which is well below `max_adoption_per_period` (`0.05`) and causes MDP `step_u = headroom / 5 = 0.002–0.004`. Check `coverage_ceiling >= cf_bau_2040` immediately upon receiving Person A's Stage 2 forecasts.
5. **Improved Biomass Effectiveness (`eta_biomass`):**
   - Field trials (Tielsch et al. 2016 in Sarlahi Nepal: `~32%` 20-hr PM2.5 reduction; Pope et al. 2017 meta-analysis: `41%` [29–50%]) leave kitchen PM2.5 near `900 ug/m^3`. Under GBD's concave COPD exposure–response curve, the implied COPD PAF reduction is **`~0.15–0.30` (proposed central `0.25`)**, materially lower than the `0.45` mock baseline.

---

## 4. Validation Summary & Inaccessible / Limited Sources

- **Automated checks passed (`generate_outputs.py`):**
  - UN WPP 2024 5-year age/sex sums match total population within `1.94e-07` relative error across all 51 years (`1990–2040`) and 21 age bins for India and Nepal.
  - UN WUP 2025 National Definitions `Urban + Rural == Total` within `7.56e-16` relative error.
  - WHO GHO and World Bank WDI SDG 7.1.2 match to `0.0` percentage points across 2000–2024.
  - `src.optimise.param_audit.validate_household_counts` and `convert_usd_per_hh_to_usd_millions` validated cleanly on the 2025 derived households.
- **Reported limitations / inaccessible items:**
  1. **India Census 2011 `HH-01`:** Published on `censusindia.gov.in` only as 35 separate state/UT legacy binary `.XLS` files without a national `0000` `.xlsx` file (and `xlrd` is absent from `.venv`). Sample file `IND_Census2011_HH01_01_JK.XLS` downloaded and logged; national totals/households were obtained cleanly from Census 2011 Table A-1 `.xlsx`, paired with NFHS-5 (`dhs_ind_npl_hh_size_and_clean_fuel.json`) for 2019–21 household sizes.
  2. **World Bank FX/CPI JSON API (`PA.NUS.FCRF`, `FP.CPI.TOTL`):** Timed out (`curl` exit 28 after 30–45s); 2023–2024 annual average FX (`83.70 INR/USD`, `133.50 NPR/USD`) and US CPI-U (`313.689` in 2024) were verified via web search of World Bank/IMF/BLS series and documented alongside `param_audit.py`'s `83.56 / 133.70` rates.
  3. **IHME GBD 2023 Results Tool:** Requires interactive browser session/GHDx login; covered by `gbd_extraction_checklist.md` for Person A.
