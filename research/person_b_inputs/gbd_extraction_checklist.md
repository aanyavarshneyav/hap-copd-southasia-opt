# GBD 2023 Extraction Checklist for Person A (Rudraksh)

> **Purpose:** Exact specification for extracting GBD 2023 CSVs from the [IHME GBD Results Tool](https://vizhub.healthdata.org/gbd-results/) into `data/raw/`, plus known schema mismatches against `docs/INTERFACES.md` and `src/data/contracts.py`.
>
> **Strict Rule:** Do **not** substitute WHO GHO COPD mortality/DALY estimates for GBD estimates without explicit team approval. For exposure (`ASM-02`) and urban/rural population apportionment (`ASM-01`), you can immediately use Person B's downloaded files in `research/person_b_inputs/`.

---

## 1. Exact GBD Results Tool Queries (3 Exports)

In all 3 exports, select **Download -> CSV** and set **Format = Both IDs and Names**. Log exact download date, URL/tool, and SHA-256 in `data/raw/manifest.json`.

### Query 1: `data/raw/gbd2023_hap_copd_southasia.csv` *(Primary outcome for Stages 1–2)*
- **GBD Estimate:** `Risk factor`
- **Risk:** `Household air pollution from solid fuels`
- **Cause:** `Chronic obstructive pulmonary disease`
- **Locations (5):** `India`, `Nepal`, `Pakistan`, `Bangladesh`, `Bhutan` *(national level only; do NOT include Indian sub-national states per `DEC-003`)*
- **Measures (2):** `Deaths`, `DALYs (Disability-Adjusted Life Years)`
- **Metrics (3):** `Number`, `Rate`, `Percent`
- **Sexes (3):** `Both`, `Male`, `Female`
- **Years (34):** `1990` through `2023` (all annual years)
- **Age groups:** `All ages`, `Age-standardized`, **plus** all standard 5-year GBD COPD age bins (e.g., `15-19` through `95+`, or `<5` through `95+` if populated — needed for Stage 2 Poisson APC!)
- **Required value/UI fields:** `val`, `lower`, `upper` (95% uncertainty interval)

### Query 2: `data/raw/gbd2023_total_copd_southasia.csv` *(Denominator for HAP share of COPD in Stage 1)*
- **GBD Estimate:** `Cause of death or injury`
- **Risk:** *(none / N/A)*
- **Cause:** `Chronic obstructive pulmonary disease`
- **Locations (5):** `India`, `Nepal`, `Pakistan`, `Bangladesh`, `Bhutan`
- **Measures (2):** `Deaths`, `DALYs (Disability-Adjusted Life Years)`
- **Metrics (2):** `Number`, `Rate`
- **Sexes (3):** `Both`, `Male`, `Female`
- **Years (34):** `1990` through `2023`
- **Age groups:** `All ages`, `Age-standardized` *(add 5-year bins if computing age-specific HAP shares)*
- **Required value/UI fields:** `val`, `lower`, `upper`

### Query 3: `data/raw/gbd2023_hap_all_causes_ind_npl.csv` *(Context: COPD share of total HAP burden in India & Nepal)*
- **GBD Estimate:** `Risk factor`
- **Risk:** `Household air pollution from solid fuels`
- **Cause:** `All causes` **plus** Level 3 HAP-linked causes (`Chronic obstructive pulmonary disease`, `Lower respiratory infections`, `Ischemic heart disease`, `Stroke`, `Tracheal, bronchus, and lung cancer`, `Diabetes mellitus`, `Cataract`, plus neonatal outcomes if linked in GBD 2023)
- **Locations (2):** `India`, `Nepal`
- **Measures (2):** `Deaths`, `DALYs (Disability-Adjusted Life Years)`
- **Metrics (3):** `Number`, `Rate`, `Percent`
- **Sexes (3):** `Both`, `Male`, `Female`
- **Years (34):** `1990` through `2023`
- **Age groups:** `All ages`, `Age-standardized`
- **Required value/UI fields:** `val`, `lower`, `upper`

*(Note on GBD 2023 Exposure/SEV: Per `DEC-007` and `docs/DATA_SPEC.md` §3, if GBD 2023 Summary Exposure Value (SEV) for HAP is restricted or awkward to export, skip it and use WHO SDG 7.1.2 from `research/person_b_inputs/clean_fuel_access.csv` as the approved fallback driver.)*

---

## 2. Potential Mismatches Between GBD Results Tool CSV & Existing Contracts

Check these 6 items explicitly in `src/data/ingest.py` and `src/forecast/export_contracts.py` so `src/data/contracts.py` validates cleanly:

1. **Crude `Rate` vs. `Rate_per_100k` (Age-Standardized Rate):**
   - GBD exports `metric_name = "Rate"` for **both** `age_name = "All ages"` (crude rate) and `age_name = "Age-standardized"` (ASR).
   - Contract 1 (`docs/INTERFACES.md` §2) requires `metric = "Rate_per_100k"` to be **Age-standardized rate (ASR)**! Never pass `All ages` crude `Rate` into `Rate_per_100k`.
   - Exact mapping: `(age_name="All ages", metric_name="Number") -> metric="Number"`, and `(age_name="Age-standardized", metric_name="Rate") -> metric="Rate_per_100k"`.
2. **Age-Standardized `Number` Does Not Exist in GBD:**
   - GBD omits `Number` for `Age-standardized` rows (returns only `Rate` and `Percent`). Do not expect square cross-products across `age × metric`.
3. **Measure & Sex String Casing / Naming:**
   - GBD outputs `measure_name = "DALYs (Disability-Adjusted Life Years)"` -> Contract 1 requires `"DALYs"`.
   - GBD outputs `sex_name = "Both"`, `"Male"`, `"Female"` -> Contract 1 requires lowercase `"both"`, `"male"`, `"female"`.
4. **National vs. 4-Unit Granularity (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`):**
   - GBD provides national `India` and `Nepal` only (no urban/rural). Apportion national `Number` via `ASM-01` using **WUP 2025 National Definitions** (from `research/person_b_inputs/demographic_inputs.csv`) + **WHO SDG 7.1.2** (`clean_fuel_access.csv`), then recompute unit ASRs consistently so `urban + rural` weighted burden reconciles with national GBD totals.
5. **Non-Negativity Invariant (`0 <= lower <= val <= upper`):**
   - `src/data/contracts.py` asserts `0 <= lower <= val <= upper` on all Contract 1 rows. Verify no GBD `Percent` (PAF) rows have small negative `lower` bounds before transforming; HAP-COPD `Number` and `Rate` should always be `>= 0`.
6. **GBD Release Consistency & Export Row Cap:**
   - Verify all 3 files show the exact same `GBD 2023` release tag (do not mix GBD 2021 and GBD 2023). Keeping the 3 queries separate keeps each under ~68,000 rows, well below GBD Results Tool's per-download limit.

---

## 3. Ready-to-Use Auxiliary Inputs Person B Can Hand Off Immediately

All verified raw files are already in `research/person_b_inputs/raw/` (with SHA-256 hashes in `source_manifest.json`):
- **UN WPP 2024 (`1990–2040`, 21 five-year age groups × sex, India & Nepal):** `WPP2024_PopulationByAge5GroupSex_Medium.csv.gz` and `WPP2024_TotalPopulationBySex.csv.gz` *(note: WPP 2024 `CSV_FILES` also contains PAK, BGD, BTN if you copy/extract those 5 countries for Stage 2!)*.
- **UN WUP 2025 National Definitions (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`, `1990–2040`):** `WUP2025-DB-National-Definitions-Population-Data.csv.gz` + processed `demographic_inputs.csv`.
- **WHO GHO & World Bank SDG 7.1.2 (`2000–2024` clean cooking + `1990–2023` fuel categories):** `clean_fuel_access.csv` (and raw JSONs).
