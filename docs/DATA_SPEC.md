# Data Specification (`data/raw/` & Validation Rules)

> **Immutable Raw Data Rule:** Files in `data/raw/` are **never** edited after download. Every file placed in `data/raw/` must be logged in `data/raw/manifest.json` with `source`, `url_or_tool`, `download_date`, `version`, `sha256`, and `row_count`.

---

## 1. Dataset Inventory & Target Filenames

| # | Dataset | Target Filename in `data/raw/` | Version | Role |
| --- | --- | --- | --- | --- |
| 1 | GBD 2023 HAP-attributable COPD | `gbd2023_hap_copd_southasia.csv` | GBD 2023 | Primary historical burden outcome (Stages 1–2) |
| 2 | GBD 2023 Total COPD | `gbd2023_total_copd_southasia.csv` | GBD 2023 | Denominator for HAP-attributable share of COPD (Stage 1) |
| 3 | GBD 2023 HAP All Linked Causes | `gbd2023_hap_all_causes_ind_npl.csv` | GBD 2023 | Context: COPD share of total HAP burden in India & Nepal (Stage 1) |
| 4 | UN World Population Prospects 2024 | `un_wpp2024_pop_by_age_sex.csv` | WPP 2024 | Population offsets & 2024–2040 count projections (Stage 2) |
| 5 | WHO Clean Cooking Fuel Access | `who_sdg712_clean_cooking.csv` | Latest GHO / SDG 7.1.2 | Exposure driver fallback, ML feature, urban/rural split (Stages 0–3) |
| 6 | Intervention Cost & Effect Literature | `docs/ASSUMPTIONS.md` (register) | Scenario specs | Unit costs, effectiveness, adoption caps (Stages 3–4) |

---

## 2. Exact Download Filters & Schemas

### 2.1 `gbd2023_hap_copd_southasia.csv` (IHME GBD Results Tool)
- **Source:** IHME Global Health Data Exchange — GBD Results Tool (`https://vizhub.healthdata.org/gbd-results/`). Export format: **Both IDs and Names**, `.csv`.
- **Exact Filters:**
  - `GBD Estimate`: `Risk factor`
  - `Measure`: `Deaths`, `DALYs (Disability-Adjusted Life Years)`
  - `Metric`: `Number`, `Rate`, `Percent`
  - `Risk`: `Household air pollution from solid fuels`
  - `Cause`: `Chronic obstructive pulmonary disease`
  - `Location`: `India`, `Nepal`, `Pakistan`, `Bangladesh`, `Bhutan` (5 countries)
  - `Age`: `All ages`, `Age-standardized`, plus standard 5-year GBD COPD age bins (`TODO(Person A): verify exact available GBD 2023 age groups on export; COPD GBD bins typically start at 15–19 or <5 through 95+`)
  - `Sex`: `Both`, `Male`, `Female`
  - `Year`: `1990` to `2023` (all 34 annual years)
- **Expected Columns:**
  `measure_id, measure_name, location_id, location_name, sex_id, sex_name, age_id, age_name, cause_id, cause_name, rei_id, rei_name, metric_id, metric_name, year, val, upper, lower`
- **Expected Row Count:**
  $5 \text{ locations} \times 34 \text{ years} \times 2 \text{ measures} \times 3 \text{ metrics} \times 3 \text{ sexes} \times N_{\text{ages}}$.  
  For $N_{\text{ages}} \approx 20\text{–}22$ (All ages + Age-standardized + 5-year bins): **~61,200 to ~67,320 rows** (excluding any age-metric combinations IHME omits, e.g., `Age-standardized` only has `Rate` and `Percent`, not `Number`, which reduces actual row count slightly — verify exact count on download).

### 2.2 `gbd2023_total_copd_southasia.csv` (IHME GBD Results Tool)
- **Source:** IHME GBD Results Tool (Both IDs and Names, `.csv`).
- **Exact Filters:**
  - `GBD Estimate`: `Cause of death or injury`
  - `Measure`: `Deaths`, `DALYs (Disability-Adjusted Life Years)`
  - `Metric`: `Number`, `Rate`
  - `Cause`: `Chronic obstructive pulmonary disease`
  - `Location`: `India`, `Nepal`, `Pakistan`, `Bangladesh`, `Bhutan`
  - `Age`: `All ages`, `Age-standardized` (plus 5-year bins if age-specific share needed)
  - `Sex`: `Both`, `Male`, `Female`
  - `Year`: `1990` to `2023`
- **Expected Columns:** Same as 2.1 without `rei_id, rei_name`.
- **Expected Row Count:** $5 \times 34 \times 2 \times 2 \times 3 \times N_{\text{ages}}$ (**~4,080 rows** for `All ages` + `Age-standardized`; **~40,000+ rows** if all 5-year age bins included).

### 2.3 `gbd2023_hap_all_causes_ind_npl.csv` (IHME GBD Results Tool)
- **Source:** IHME GBD Results Tool (Both IDs and Names, `.csv`).
- **Exact Filters:**
  - `GBD Estimate`: `Risk factor`
  - `Measure`: `Deaths`, `DALYs (Disability-Adjusted Life Years)`
  - `Metric`: `Number`, `Rate`, `Percent`
  - `Risk`: `Household air pollution from solid fuels`
  - `Cause`: `All causes` plus top HAP-linked Level 3 causes (e.g., `Lower respiratory infections`, `Chronic obstructive pulmonary disease`, `Ischemic heart disease`, `Stroke`, `Tracheal, bronchus, and lung cancer`, `Diabetes mellitus`, `Cataract` — verify GBD 2023 linked cause list in tool)
  - `Location`: `India`, `Nepal`
  - `Age`: `All ages`, `Age-standardized`
  - `Sex`: `Both`, `Male`, `Female`
  - `Year`: `1990` to `2023`
- **Expected Columns:** Same as 2.1.
- **Expected Row Count:** $2 \text{ locations} \times 34 \times 2 \times 3 \times 2 \text{ ages} \times 3 \text{ sexes} \times N_{\text{causes}}$ (~8,000–12,000 rows for ~8 causes).

### 2.4 `un_wpp2024_pop_by_age_sex.csv` (UN World Population Prospects 2024)
- **Source:** UN DESA Population Division — WPP 2024 Data Portal (`https://population.un.org/wpp/Download/Standard/CSV/`), 5-year age groups by sex (Estimates `1950–2023` + Medium Variant Projections `2024–2100`, filtered/saved for `1990–2040`).
- **Exact Filters:**
  - `Location` / `ISO3_code`: `IND`, `NPL`, `PAK`, `BGD`, `BTN`
  - `Variant`: `Medium`
  - `Time`: `1990` to `2040`
  - `AgeGrp`: 5-year age groups (`0-4`, `5-9`, ..., `95-99`, `100+`; 21 groups)
  - `Sex`: Male, Female, Total
- **Expected Columns (WPP 2024 standard):**
  `LocID, ISO3_code, Location, Variant, Time, AgeGrp, AgeGrpStart, AgeGrpSpan, PopMale, PopFemale, PopTotal` (units: thousands of persons — convert explicitly to persons $\times 1000$ during ingestion!). `TODO(Person A): verify whether WPP 2024 urban/rural split requires UN World Urbanization Prospects (WUP) or WHO population weights.`
- **Expected Row Count:** $5 \text{ countries} \times 51 \text{ years (1990–2040)} \times 21 \text{ age groups} = \mathbf{5,355 \text{ rows}}$.

### 2.5 `who_sdg712_clean_cooking.csv` (WHO SDG 7.1.2 / Household Energy Database)
- **Source:** WHO Global Health Observatory (`https://www.who.int/data/gho/data/themes/air-pollution/household-air-pollution`) / SDG 7.1.2 indicator exports.
- **Exact Filters:**
  - `Countries`: `IND`, `NPL`, `PAK`, `BGD`, `BTN`
  - `Years`: `1990–2023` (or earliest available through latest WHO release, typically `1990–2022/2023`)
  - `Disaggregation / Residence`: `Total` (National), `Urban`, `Rural`
  - `Fuel Type` (if from WHO Household Energy Database): clean fuels total (`SDG 7.1.2` primary clean fuels/technologies) plus fuel-specific shares (LPG, electricity, biomass, charcoal, coal, kerosene) where available.
- **Expected Columns:**
  `SpatialDimValueCode` (ISO3), `Location`, `Period` (Year), `Dim1` (`Total` / `Urban` / `Rural`), `FuelType` (if applicable), `FactValueNumeric` (or `Value`), `FactValueNumericLow`, `FactValueNumericHigh` (95% CI, if provided). `TODO(Person A): confirm exact WHO API/CSV export column names.`
- **Expected Row Count:** ~5 countries × ~25–34 years × 3 residence categories ≈ **375–510 rows** (more if disaggregated by fuel type).

---

## 3. Fallback Rule: Unavailable GBD 2023 Exposure Data
- **Risk:** IHME GBD Results Tool may restrict or omit raw GBD Summary Exposure Value (SEV) / solid-fuel proportion exports for GBD 2023, or require separate GBD Compare scraping.
- **Approved Fallback:** Do **not** scrape or guess GBD exposure values. Use **WHO SDG 7.1.2 clean-cooking fuel access** (`clean_fuel_share` $\in [0, 1]$, and implied `solid_fuel_share = 1 - clean_fuel_share` adjusted for kerosene where reported) as the primary exposure driver across Stage 0 apportionment, Stage 2 ML panel covariates, and Stage 3 / MDP exposure–response functions. Record activation of this fallback in `docs/DECISIONS.md`.

---

## 4. Automated Validation Checklist (`make validate-data`)
`src/data/validate.py` must assert all checks below before writing `data/interim/*.parquet`:

- [ ] **Manifest match:** Every file in `data/raw/` matches an entry in `data/raw/manifest.json` (source, download date, version, SHA-256).
- [ ] **Schema check:** Required columns present; `year` is integer in `[1990, 2040]`; numeric values parse cleanly without NaNs in primary series.
- [ ] **Completeness check:** All 5 countries present for GBD/UN/WHO datasets; all 34 years (`1990–2023`) present in GBD; all `1990–2040` present in UN WPP.
- [ ] **Uncertainty & sign invariants:** For every GBD row, `0 <= lower <= val <= upper` (flag and inspect any `Percent` edge cases if negative PAFs ever occur — HAP-COPD should be non-negative).
- [ ] **Bounds check:** WHO SDG 7.1.2 proportions satisfy `0 <= clean_fuel_share <= 100` (normalize to `[0, 1]` in `data/interim/`).
- [ ] **Unit verification:** UN WPP `PopTotal` converted explicitly from thousands to persons (`PopMale + PopFemale ≈ PopTotal` within rounding tolerance).
- [ ] **Immutability check:** Raw files remain untouched; outputs written strictly to `data/interim/` and `data/processed/`.
