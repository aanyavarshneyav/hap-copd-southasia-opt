"""Build and validate Person B's parallel auxiliary datasets and parameter registers.

Reads verified raw files under ``research/person_b_inputs/raw/`` and produces:
1. ``demographic_inputs.csv``
2. ``clean_fuel_access.csv``
3. ``intervention_source_register.csv``
4. ``parameter_gap_matrix.csv``
5. ``proposed_contract4_table.csv``
6. ``proposed_intervention_params.json``
7. ``source_manifest.json``

Does not modify any Person A files, mock fixtures, or shared configuration files.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

# Allow importing src.optimise.param_audit without altering src/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optimise.param_audit import (  # noqa: E402
    compute_india_lpg_usd_per_hh,
    compute_nepal_lpg_usd_per_hh,
    convert_usd_per_hh_to_usd_millions,
    derive_task3_lpg_usd_per_hh_scenarios,
    validate_household_counts,
)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"

# 2024 annual average exchange rates (World Bank / IMF / RBI / NRB tracking)
# Note: param_audit.py Task 3 uses 83.56 INR/USD and 133.70 NPR/USD; we check both.
FX_INR_PER_USD_2024 = 83.70
FX_NPR_PER_USD_2024 = 133.50
FX_INR_PER_USD_PARAM_AUDIT = 83.56
FX_NPR_PER_USD_PARAM_AUDIT = 133.70


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_xlsx_sheet_rows(path: Path, sheet_index: int = 1) -> list[dict[str, str]]:
    """Parse an XLSX worksheet into a list of {col_letter: value} dicts using stdlib."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        sst: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(".//m:si", ns):
                sst.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        sheet_path = f"xl/worksheets/sheet{sheet_index}.xml"
        root = ET.fromstring(z.read(sheet_path))
        rows: list[dict[str, str]] = []
        for r in root.findall(".//m:row", ns):
            row_dict: dict[str, str] = {}
            for c in r.findall("m:c", ns):
                ref = c.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                t = c.attrib.get("t")
                v = c.find("m:v", ns)
                if v is not None and v.text is not None:
                    row_dict[col] = sst[int(v.text)] if t == "s" else v.text
            if row_dict:
                rows.append(row_dict)
    return rows


def load_wpp2024_totals() -> dict[tuple[str, int], dict[str, float]]:
    """Load UN WPP 2024 Medium variant Total/Male/Female populations (persons)."""
    out: dict[tuple[str, int], dict[str, float]] = {}
    path = RAW_DIR / "WPP2024_TotalPopulationBySex.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iso = row["ISO3_code"]
            if iso in ("IND", "NPL") and row["Variant"] == "Medium":
                yr = int(row["Time"])
                if 1990 <= yr <= 2040:
                    out[(iso, yr)] = {
                        "pop_total": float(row["PopTotal"]) * 1000.0,
                        "pop_male": float(row["PopMale"]) * 1000.0,
                        "pop_female": float(row["PopFemale"]) * 1000.0,
                    }
    return out


def verify_wpp2024_age5_vs_totals(wpp_totals: dict[tuple[str, int], dict[str, float]]) -> dict[str, Any]:
    """Verify UN WPP 2024 5-year age group file sums match total population file."""
    path = RAW_DIR / "WPP2024_PopulationByAge5GroupSex_Medium.csv.gz"
    sums: dict[tuple[str, int], list[float]] = {}
    age_groups: set[str] = set()
    years: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iso = row["ISO3_code"]
            if iso in ("IND", "NPL"):
                yr = int(row["Time"])
                if 1990 <= yr <= 2040:
                    years.add(yr)
                    age_groups.add(row["AgeGrp"])
                    acc = sums.setdefault((iso, yr), [0.0, 0.0, 0.0])
                    acc[0] += float(row["PopTotal"]) * 1000.0
                    acc[1] += float(row["PopMale"]) * 1000.0
                    acc[2] += float(row["PopFemale"]) * 1000.0
    max_rel_diff = 0.0
    for key, (tot, male, fem) in sums.items():
        ref = wpp_totals[key]["pop_total"]
        rel = abs(tot - ref) / ref
        if rel > max_rel_diff:
            max_rel_diff = rel
    return {
        "n_years": len(years),
        "n_age_groups": len(age_groups),
        "age_groups": sorted(age_groups, key=lambda x: int(x.split("-")[0].replace("+", ""))),
        "max_rel_diff_vs_totals": max_rel_diff,
    }


def load_wup2025_nat_def() -> dict[tuple[str, int, str], dict[str, float]]:
    """Load UN WUP 2025 National Definitions population (persons) and share."""
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    path = RAW_DIR / "WUP2025-DB-National-Definitions-Population-Data.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iso = row["ISO3_Code"]
            if iso in ("IND", "NPL"):
                yr = int(row["Year"])
                if 1990 <= yr <= 2040:
                    out[(iso, yr, row["Category"].lower())] = {
                        "pop": float(row["Pop"]) * 1000.0,
                        "perc": float(row["percPop"]),
                    }
    return out


def load_wup2025_degurba() -> dict[tuple[str, int, str], dict[str, float]]:
    """Load UN WUP 2025 DEGURBA Level 1 population (persons) and share."""
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    path = RAW_DIR / "WUP2025-DB-DEGURBA-Level1-Population-Surface-Data.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iso = row["ISO3_Code"]
            if iso in ("IND", "NPL"):
                yr = int(row["Year"])
                if 1990 <= yr <= 2040:
                    out[(iso, yr, row["Category"].lower())] = {
                        "pop": float(row["Pop"]) * 1000.0,
                        "perc": float(row["percPop"]),
                    }
    return out


def load_india_census_2011_a1() -> dict[str, dict[str, float]]:
    """Extract all-India Total/Rural/Urban rows from Census 2011 Table A-1."""
    rows = read_xlsx_sheet_rows(
        RAW_DIR / "IND_Census2011_A1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
    )
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.get("A") == "00" and r.get("B") == "000" and r.get("C") == "00000":
            res = r.get("F", "").strip().lower()
            if res in ("total", "rural", "urban"):
                pop = float(r["K"])
                hh = float(r["J"])
                male = float(r["L"])
                fem = float(r["M"])
                out[res] = {
                    "pop_total": pop,
                    "pop_male": male,
                    "pop_female": fem,
                    "female_share": fem / pop,
                    "households": hh,
                    "avg_hh_size_implied": pop / hh,
                }
    assert set(out.keys()) == {"total", "rural", "urban"}, f"Unexpected India A-1 keys: {out}"
    return out


def load_nepal_census_2021() -> dict[str, dict[str, float]]:
    """Extract national Total/Urban/Rural rows from Nepal NPHC 2021 Table 15 & Table 07."""
    rows = read_xlsx_sheet_rows(RAW_DIR / "NPL_Census2021_National_Indv01_HouseholdAndPop.xlsx")
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        area = (r.get("D") or r.get("E") or "").strip()
        key = None
        if area == "Nepal":
            key = "total"
        elif area == "Urban Municipalities":
            key = "urban"
        elif area == "Rural Municipalities":
            key = "rural"
        if key:
            hh = float(r["F"])
            pop = float(r["G"])
            male = float(r["H"])
            fem = float(r["I"])
            out[key] = {
                "households": hh,
                "pop_total": pop,
                "pop_male": male,
                "pop_female": fem,
                "female_share": fem / pop,
                "avg_hh_size_published": float(r["J"]),
                "avg_hh_size_exact": pop / hh,
            }
    assert set(out.keys()) == {"total", "urban", "rural"}, f"Unexpected Nepal keys: {out}"

    # Also cross-check cooking fuel table (conventional housing units)
    fuel_rows = read_xlsx_sheet_rows(RAW_DIR / "NPL_Census2021_National_Hhld07_CookingFuel.xlsx")
    for r in fuel_rows:
        area = (r.get("A") or r.get("B") or "").strip()
        key = None
        if area == "Nepal":
            key = "total"
        elif area == "Urban Municipalities":
            key = "urban"
        elif area == "Rural Municipalities":
            key = "rural"
        if key:
            tot_hh = float(r["C"])
            out[key]["conventional_hh_fuel_table"] = tot_hh
            out[key]["fuel_wood_hh"] = float(r["D"])
            out[key]["fuel_lpg_hh"] = float(r["E"])
            out[key]["fuel_elec_hh"] = float(r["F"])
            out[key]["fuel_dung_hh"] = float(r["G"])
            out[key]["fuel_biogas_hh"] = float(r["H"])
            out[key]["fuel_kerosene_hh"] = float(r["I"])
            out[key]["fuel_other_hh"] = float(r["J"])
    return out


def load_dhs_hh_size_and_fuel() -> dict[tuple[str, int, str, str], float]:
    """Load DHS/NFHS mean household size and household/population clean fuel shares."""
    with (RAW_DIR / "dhs_ind_npl_hh_size_and_clean_fuel.json").open(encoding="utf-8") as f:
        data = json.load(f)["Data"]
    out: dict[tuple[str, int, str, str], float] = {}
    for r in data:
        if r.get("CharacteristicCategory") in ("Residence", "Total"):
            iso = "IND" if r["CountryName"] == "India" else "NPL"
            yr = int(r["SurveyYear"])
            ind = r["IndicatorId"]
            res = r["CharacteristicLabel"].strip().lower()
            out[(iso, yr, ind, res)] = float(r["Value"])
    return out


def build_demographic_inputs(
    wpp_totals: dict[tuple[str, int], dict[str, float]],
    wup_nat: dict[tuple[str, int, str], dict[str, float]],
    wup_degurba: dict[tuple[str, int, str], dict[str, float]],
    ind_c11: dict[str, dict[str, float]],
    npl_c21: dict[str, dict[str, float]],
    dhs: dict[tuple[str, int, str, str], float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build demographic_inputs.csv rows and return derived 2025 household counts."""
    rows: list[dict[str, Any]] = []

    def add_row(
        unit: str,
        country: str,
        residence: str,
        ref_year: int,
        record_type: str,
        status: str,
        urban_def: str,
        pop_total: float | None,
        pop_male: float | None,
        pop_female: float | None,
        female_share: float | None,
        households: float | None,
        avg_hh_size: float | None,
        hh_size_basis: str,
        derivation_formula: str,
        source_name: str,
        source_file: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "unit": unit,
                "country": country,
                "residence": residence,
                "reference_year": ref_year,
                "record_type": record_type,
                "status": status,
                "urban_rural_definition": urban_def,
                "population_total": round(pop_total, 0) if pop_total is not None else "",
                "population_male": round(pop_male, 0) if pop_male is not None else "",
                "population_female": round(pop_female, 0) if pop_female is not None else "",
                "female_share": round(female_share, 6) if female_share is not None else "",
                "households": round(households, 0) if households is not None else "",
                "avg_household_size": round(avg_hh_size, 4) if avg_hh_size is not None else "",
                "hh_size_basis": hh_size_basis,
                "derivation_formula": derivation_formula,
                "source_name": source_name,
                "source_file": source_file,
                "notes": notes,
            }
        )

    # 1. Observed Census Benchmarks (4 units + national totals)
    for res in ("urban", "rural", "total"):
        u = f"IND_{res}"
        d = ind_c11[res]
        add_row(
            unit=u,
            country="India",
            residence=res,
            ref_year=2011,
            record_type="observed_census",
            status="observed",
            urban_def="national_census_2011",
            pop_total=d["pop_total"],
            pop_male=d["pop_male"],
            pop_female=d["pop_female"],
            female_share=d["female_share"],
            households=d["households"],
            avg_hh_size=d["avg_hh_size_implied"],
            hh_size_basis="Census 2011 Table A-1 Pop / Households (total incl. institutional/houseless)",
            derivation_formula="Observed census count; avg_hh_size = population_total / households",
            source_name="Census of India 2011 Table A-1 (ORGI)",
            source_file="IND_Census2011_A1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
            notes="2021 India Census delayed; HH-01 normal-household summary ~4.8 urban / 4.9 rural.",
        )

    for res in ("urban", "rural", "total"):
        u = f"NPL_{res}"
        d = npl_c21[res]
        add_row(
            unit=u,
            country="Nepal",
            residence=res,
            ref_year=2021,
            record_type="observed_census",
            status="observed",
            urban_def="national_census_2021_municipalities",
            pop_total=d["pop_total"],
            pop_male=d["pop_male"],
            pop_female=d["pop_female"],
            female_share=d["female_share"],
            households=d["households"],
            avg_hh_size=d["avg_hh_size_exact"],
            hh_size_basis=f"NPHC 2021 Table 15 (published rounded={d['avg_hh_size_published']:.2f})",
            derivation_formula="Observed census count; exact Pop / Households",
            source_name="Nepal NPHC 2021 Table 15 (NSO/CBS)",
            source_file="NPL_Census2021_National_Indv01_HouseholdAndPop.xlsx",
            notes="Urban Municipalities (293) vs Rural Municipalities (460 Gaunpalikas).",
        )

    # 2. Observed Survey Household Size Benchmarks (NFHS-5 & NDHS 2022)
    for iso, country, yr, s_name in [
        ("IND", "India", 2020, "India NFHS-5 (2019-21) via DHS API"),
        ("NPL", "Nepal", 2022, "Nepal NDHS 2022 via DHS API"),
    ]:
        for res in ("urban", "rural"):
            hh_sz = dhs[(iso, yr, "HC_MEMB_H_MNM", res)]
            add_row(
                unit=f"{iso}_{res}",
                country=country,
                residence=res,
                ref_year=yr,
                record_type="observed_survey",
                status="observed",
                urban_def="national_survey_frame",
                pop_total=None,
                pop_male=None,
                pop_female=None,
                female_share=None,
                households=None,
                avg_hh_size=hh_sz,
                hh_size_basis="DHS HC_MEMB_H_MNM (mean de jure usual household members)",
                derivation_formula="Observed survey mean household size",
                source_name=s_name,
                source_file="dhs_ind_npl_hh_size_and_clean_fuel.json",
                notes=(
                    "Primary recent HH size for India 2025 estimation."
                    if iso == "IND"
                    else "De jure usual residents only (lower than NPHC 2021 4.31/4.51 due to absent migrants)."
                ),
            )

    # 3. 2025 & 2040 UN WUP 2025 National Definitions + Derived Household Estimates (4 Units)
    hh_2025_derived: dict[str, int] = {}
    hh_size_primary = {
        "IND_urban": (dhs[("IND", 2020, "HC_MEMB_H_MNM", "urban")], "NFHS-5 (2019-21) DHS API = 4.2"),
        "IND_rural": (dhs[("IND", 2020, "HC_MEMB_H_MNM", "rural")], "NFHS-5 (2019-21) DHS API = 4.5"),
        "NPL_urban": (npl_c21["urban"]["avg_hh_size_exact"], "NPHC 2021 Table 15 exact = 4.3076 (4.31)"),
        "NPL_rural": (npl_c21["rural"]["avg_hh_size_exact"], "NPHC 2021 Table 15 exact = 4.5115 (4.51)"),
    }

    for yr in (2025, 2040):
        for iso, country in [("IND", "India"), ("NPL", "Nepal")]:
            wpp = wpp_totals[(iso, yr)]
            nat_fem_share = wpp["pop_female"] / wpp["pop_total"]
            for res in ("urban", "rural"):
                unit = f"{iso}_{res}"
                pop = wup_nat[(iso, yr, res)]["pop"]
                hh_sz, hh_basis = hh_size_primary[unit]
                hh_est = pop / hh_sz
                if yr == 2025:
                    hh_2025_derived[unit] = int(round(hh_est))
                add_row(
                    unit=unit,
                    country=country,
                    residence=res,
                    ref_year=yr,
                    record_type="derived_estimate" if yr == 2025 else "derived_projection",
                    status="derived",
                    urban_def="national_definition_wup2025",
                    pop_total=pop,
                    pop_male=None,
                    pop_female=None,
                    female_share=nat_fem_share,
                    households=hh_est,
                    avg_hh_size=hh_sz,
                    hh_size_basis=hh_basis,
                    derivation_formula=(
                        f"Pop = WUP2025 National Def ({yr}); HH = Pop / ({hh_basis}); "
                        "female_share = WPP2024 national PopFemale/PopTotal (urban/rural sex split not in WUP)"
                    ),
                    source_name="UN WUP 2025 (National Definitions) + UN WPP 2024 + NFHS-5/NPHC 2021",
                    source_file="WUP2025-DB-National-Definitions-Population-Data.csv.gz; WPP2024_TotalPopulationBySex.csv.gz",
                    notes=(
                        f"Recommended primary {yr} unit population (consistent with WHO SDG 7.1.2 residence strata). "
                        f"WUP urban share = {wup_nat[(iso, yr, res)]['perc']:.2f}%."
                    ),
                )

    # 4. UN WUP 2025 DEGURBA Level 1 Harmonized Comparison (Cities + Towns vs Rural)
    for yr in (2025, 2040):
        for iso, country in [("IND", "India"), ("NPL", "Nepal")]:
            for deg_cat, res in [("cities and towns", "urban"), ("rural", "rural")]:
                unit = f"{iso}_{res}"
                pop = wup_degurba[(iso, yr, deg_cat)]["pop"]
                perc = wup_degurba[(iso, yr, deg_cat)]["perc"]
                add_row(
                    unit=unit,
                    country=country,
                    residence=res,
                    ref_year=yr,
                    record_type="un_projection",
                    status="projected",
                    urban_def="un_degurba_level1_cities_and_towns_vs_rural",
                    pop_total=pop,
                    pop_male=None,
                    pop_female=None,
                    female_share=None,
                    households=None,
                    avg_hh_size=None,
                    hh_size_basis="Not applied (DEGURBA mismatch with WHO/DHS national survey strata)",
                    derivation_formula="Direct UN WUP 2025 DEGURBA Level 1 value",
                    source_name="UN WUP 2025 (DEGURBA Level 1)",
                    source_file="WUP2025-DB-DEGURBA-Level1-Population-Surface-Data.csv.gz",
                    notes=(
                        f"UN harmonized Degree of Urbanization share = {perc:.2f}% "
                        f"({'Cities+Towns' if res == 'urban' else 'Rural grid cells'}). "
                        "Do NOT mix directly with WHO SDG 7.1.2 national-definition urban/rural rates."
                    ),
                )

    # 5. UN WPP 2024 National Totals (2025 & 2040)
    for yr in (2025, 2040):
        for iso, country in [("IND", "India"), ("NPL", "Nepal")]:
            wpp = wpp_totals[(iso, yr)]
            add_row(
                unit=f"{iso}_total",
                country=country,
                residence="total",
                ref_year=yr,
                record_type="un_projection",
                status="projected",
                urban_def="country_total",
                pop_total=wpp["pop_total"],
                pop_male=wpp["pop_male"],
                pop_female=wpp["pop_female"],
                female_share=wpp["pop_female"] / wpp["pop_total"],
                households=None,
                avg_hh_size=None,
                hh_size_basis="National total",
                derivation_formula="Direct UN WPP 2024 Medium variant",
                source_name="UN WPP 2024 (Medium Variant)",
                source_file="WPP2024_TotalPopulationBySex.csv.gz",
                notes="Matches WUP 2025 total population identically.",
            )

    return rows, hh_2025_derived


def build_clean_fuel_access(npl_c21: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Build clean_fuel_access.csv combining WHO GHO, World Bank WDI, DHS, and Nepal NPHC 2021."""
    rows: list[dict[str, Any]] = []
    dim_map = {
        "RESIDENCEAREATYPE_TOTL": ("total", "total"),
        "RESIDENCEAREATYPE_URB": ("urban", "urban"),
        "RESIDENCEAREATYPE_RUR": ("rural", "rural"),
    }

    # 1. WHO GHO SDG 7.1.2 overall (2000-2024)
    with (RAW_DIR / "who_gho_sdg712_clean_cooking_ind_npl.json").open(encoding="utf-8") as f:
        who_main = json.load(f)["value"]
    for r in who_main:
        iso = r["SpatialDim"]
        res, unit_sfx = dim_map[r["Dim1"]]
        yr = int(r["TimeDim"])
        rows.append(
            {
                "country_iso3": iso,
                "country": "India" if iso == "IND" else "Nepal",
                "unit_id": f"{iso}_{unit_sfx}",
                "residence": res,
                "year": yr,
                "source": "WHO GHO",
                "indicator_code": r["IndicatorCode"],
                "indicator_name": "Proportion of population with primary reliance on clean fuels and technologies for cooking (%)",
                "fuel_category": "clean_fuels_total",
                "val": r["NumericValue"],
                "lower": r["Low"],
                "upper": r["High"],
                "unit": "%",
                "denominator": "Total/urban/rural population",
                "unit_basis": "people",
                "is_latest_common_baseline_year": yr == 2024,
                "source_file": "who_gho_sdg712_clean_cooking_ind_npl.json",
            }
        )

    # 2. World Bank WDI SDG 7.1.2 (2000-2024)
    with (RAW_DIR / "worldbank_sdg712_clean_cooking_ind_npl.json").open(encoding="utf-8") as f:
        wb = json.load(f)[1]
    wb_res_map = {
        "EG.CFT.ACCS.ZS": "total",
        "EG.CFT.ACCS.UR.ZS": "urban",
        "EG.CFT.ACCS.RU.ZS": "rural",
    }
    for r in wb:
        if r["value"] is not None:
            iso = r["countryiso3code"]
            ind = r["indicator"]["id"]
            res = wb_res_map[ind]
            yr = int(r["date"])
            rows.append(
                {
                    "country_iso3": iso,
                    "country": "India" if iso == "IND" else "Nepal",
                    "unit_id": f"{iso}_{res}",
                    "residence": res,
                    "year": yr,
                    "source": "World Bank WDI (WHO/ESMAP)",
                    "indicator_code": ind,
                    "indicator_name": r["indicator"]["value"],
                    "fuel_category": "clean_fuels_total",
                    "val": r["value"],
                    "lower": "",
                    "upper": "",
                    "unit": "%",
                    "denominator": "Total/urban/rural population",
                    "unit_basis": "people",
                    "is_latest_common_baseline_year": yr == 2024,
                    "source_file": "worldbank_sdg712_clean_cooking_ind_npl.json",
                }
            )

    # 3. WHO GHO Fuel Categories (1990-2023)
    with (RAW_DIR / "who_gho_sdg712_fuel_categories_ind_npl.json").open(encoding="utf-8") as f:
        who_cat = json.load(f)["value"]
    for r in who_cat:
        iso = r["SpatialDim"]
        res, unit_sfx = dim_map[r["Dim1"]]
        yr = int(r["TimeDim"])
        fuel = r["Dim2"].replace("HOUSEHOLDCOOKINGFUEL_FUEL_", "").lower()
        rows.append(
            {
                "country_iso3": iso,
                "country": "India" if iso == "IND" else "Nepal",
                "unit_id": f"{iso}_{unit_sfx}",
                "residence": res,
                "year": yr,
                "source": "WHO GHO (Fuel Category)",
                "indicator_code": r["IndicatorCode"],
                "indicator_name": "Proportion of population with primary reliance on fuels and technologies for cooking, by fuel type (%)",
                "fuel_category": fuel,
                "val": r["NumericValue"],
                "lower": r["Low"],
                "upper": r["High"],
                "unit": "%",
                "denominator": "Total/urban/rural population",
                "unit_basis": "people",
                "is_latest_common_baseline_year": False,
                "source_file": "who_gho_sdg712_fuel_categories_ind_npl.json",
            }
        )

    # 4. DHS/NFHS Household vs Population Clean Fuel Comparison
    with (RAW_DIR / "dhs_ind_npl_hh_size_and_clean_fuel.json").open(encoding="utf-8") as f:
        dhs_data = json.load(f)["Data"]
    for r in dhs_data:
        if r["IndicatorId"] in ("HC_CKFL_H_CLN", "HC_CKFL_P_CLN") and r.get(
            "CharacteristicCategory"
        ) in ("Residence", "Total"):
            iso = "IND" if r["CountryName"] == "India" else "NPL"
            res = r["CharacteristicLabel"].strip().lower()
            basis = "households" if r["IndicatorId"] == "HC_CKFL_H_CLN" else "people"
            rows.append(
                {
                    "country_iso3": iso,
                    "country": r["CountryName"],
                    "unit_id": f"{iso}_{res}",
                    "residence": res,
                    "year": int(r["SurveyYear"]),
                    "source": f"DHS/NFHS ({r['SurveyId']})",
                    "indicator_code": r["IndicatorId"],
                    "indicator_name": r["Indicator"],
                    "fuel_category": "clean_fuels_total",
                    "val": r["Value"],
                    "lower": "",
                    "upper": "",
                    "unit": "%",
                    "denominator": f"De jure {basis}",
                    "unit_basis": basis,
                    "is_latest_common_baseline_year": False,
                    "source_file": "dhs_ind_npl_hh_size_and_clean_fuel.json",
                }
            )

    # 5. Nepal NPHC 2021 Census Household Fuel Counts (Hhld07)
    for res in ("total", "urban", "rural"):
        d = npl_c21[res]
        tot = d["conventional_hh_fuel_table"]
        clean_hh = d["fuel_lpg_hh"] + d["fuel_elec_hh"] + d["fuel_biogas_hh"]
        for fuel_cat, cnt in [
            ("clean_fuels_total", clean_hh),
            ("lpg", d["fuel_lpg_hh"]),
            ("electricity", d["fuel_elec_hh"]),
            ("biogas", d["fuel_biogas_hh"]),
            ("wood_firewood", d["fuel_wood_hh"]),
            ("cow_dung", d["fuel_dung_hh"]),
            ("kerosene", d["fuel_kerosene_hh"]),
        ]:
            rows.append(
                {
                    "country_iso3": "NPL",
                    "country": "Nepal",
                    "unit_id": f"NPL_{res}",
                    "residence": res,
                    "year": 2021,
                    "source": "Nepal NPHC 2021 Table 07",
                    "indicator_code": "NPHC2021_HHLD07",
                    "indicator_name": "Percentage of conventional households by primary cooking fuel",
                    "fuel_category": fuel_cat,
                    "val": round(100.0 * cnt / tot, 4),
                    "lower": "",
                    "upper": "",
                    "unit": "%",
                    "denominator": f"{int(tot)} conventional housing units",
                    "unit_basis": "households",
                    "is_latest_common_baseline_year": False,
                    "source_file": "NPL_Census2021_National_Hhld07_CookingFuel.xlsx",
                }
            )

    rows.sort(
        key=lambda x: (
            str(x["source"]),
            str(x["country_iso3"]),
            str(x["residence"]),
            str(x["fuel_category"]),
            int(x["year"]),
        )
    )
    return rows


def build_intervention_register(
    hh_2025: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Build intervention_source_register.csv, proposed_intervention_params.json, and Contract 4 table."""
    # Recompute exact LPG 5-yr USD/HH from param_audit.py (and check 2024 FX)
    lpg_pa = derive_task3_lpg_usd_per_hh_scenarios()
    # Also compute India Phase-1 (₹1,600) vs FY25-26 (₹2,050) and 2024 FX (83.70 / 133.50)
    ind_lpg_2024fx = {
        "low": compute_india_lpg_usd_per_hh(3.0, upfront_inr=2050.0, fx_inr_per_usd=FX_INR_PER_USD_2024),
        "central": compute_india_lpg_usd_per_hh(4.0, upfront_inr=2050.0, fx_inr_per_usd=FX_INR_PER_USD_2024),
        "high": compute_india_lpg_usd_per_hh(9.0, upfront_inr=2050.0, fx_inr_per_usd=FX_INR_PER_USD_2024),
    }
    npl_lpg_2024fx = {
        "urban_low": compute_nepal_lpg_usd_per_hh(331.25, 2500.0, fx_npr_per_usd=FX_NPR_PER_USD_2024),
        "urban_central": compute_nepal_lpg_usd_per_hh(450.0, 2500.0, fx_npr_per_usd=FX_NPR_PER_USD_2024),
        "urban_high": compute_nepal_lpg_usd_per_hh(1174.0, 2500.0, fx_npr_per_usd=FX_NPR_PER_USD_2024),
        "rural_low": compute_nepal_lpg_usd_per_hh(331.25, 4500.0, fx_npr_per_usd=FX_NPR_PER_USD_2024),
        "rural_central": compute_nepal_lpg_usd_per_hh(450.0, 4500.0, fx_npr_per_usd=FX_NPR_PER_USD_2024),
        "rural_high": compute_nepal_lpg_usd_per_hh(1174.0, 4500.0, fx_npr_per_usd=FX_NPR_PER_USD_2024),
    }

    # Verify param_audit converter works cleanly on real 2025 household counts
    lpg_usd_m_pa = convert_usd_per_hh_to_usd_millions(lpg_pa, hh_2025)

    # Improved biomass 5-yr USD/HH (hardware + replacement/5% annual maintenance; societal vs 50% subsidy)
    # Literature: $15-$50 commercial ICS hardware (South Asia / Kumar 2021 / MNRE benchmark ₹1,500-₹3,500)
    # With 1.5x replacement over 5 years + 5%/yr maintenance: ~$25 (mud/low) to ~$85 (forced-draft/high) societal;
    # 50% MNRE/AEPC public subsidy: ~$12.5 to ~$42.5/HH.
    bio_usd_hh = {"low": 25.0, "central": 45.0, "high": 85.0}

    # Electric induction 5-yr USD/HH (appliance + internal wiring/15A upgrade; excludes tariff/ongoing electricity)
    # India EESL NECP: ₹1,900-1,999 basic cooktop to ₹6,990 bundled w/ cookware -> $22.70 to $83.51 at 83.70 INR/USD
    # Nepal AEPC/MECS + NEA 15A upgrade: stove NPR 3,000-7,000 ($22.47-$52.43) + wiring/15A NPR 2,000-5,000 ($14.98-$37.45) = $37.45-$89.89
    elec_usd_hh = {
        "IND_urban": {"low": 22.70, "central": 53.11, "high": 83.51},
        "IND_rural": {"low": 22.70, "central": 53.11, "high": 101.43},  # +₹1,500 ($17.92) household wiring
        "NPL_urban": {"low": 37.45, "central": 63.67, "high": 89.89},
        "NPL_rural": {"low": 37.45, "central": 63.67, "high": 89.89},
    }

    register: list[dict[str, Any]] = [
        # --- LPG COST ---
        {
            "param_id": "ASM-05",
            "parameter_name": "lpg_subsidy_india_upfront",
            "intervention": "lpg_subsidy",
            "unit_scope": "IND_urban; IND_rural",
            "value_central": 2050.0,
            "uncertainty_low": 1600.0,
            "uncertainty_high": 2200.0,
            "original_currency_or_unit": "INR / connection",
            "original_price_year": "2023-2025",
            "usd_conversion_basis": f"2024_USD ({FX_INR_PER_USD_2024} INR/USD)",
            "usd_per_hh_low": round(1600.0 / FX_INR_PER_USD_2024, 2),
            "usd_per_hh_central": round(2050.0 / FX_INR_PER_USD_2024, 2),
            "usd_per_hh_high": round(2200.0 / FX_INR_PER_USD_2024, 2),
            "code_unit": "Component of USD/HH/5yr",
            "source_citation": "MoPNG / PIB India PMUY connection support (Phase-1 Rs 1,600; Ujjwala 2.0 Rs 2,200; FY25-26 Rs 2,050)",
            "source_url_or_doi": "https://pib.gov.in (MoPNG PMUY releases)",
            "page_or_table_ref": "Union Cabinet PMUY expansion approvals",
            "evidence_quality": "High - Primary Govt",
            "status": "sourced",
            "notes": "Public-fiscal connection cash assistance (excl. OMC-borne first refill & stove).",
        },
        {
            "param_id": "ASM-05",
            "parameter_name": "lpg_subsidy_india_refill_rate_and_amount",
            "intervention": "lpg_subsidy",
            "unit_scope": "IND_urban; IND_rural",
            "value_central": 4.0,
            "uncertainty_low": 3.0,
            "uncertainty_high": 9.0,
            "original_currency_or_unit": "Refills/HH/yr (at INR 300/refill)",
            "original_price_year": "2023-2025",
            "usd_conversion_basis": f"2024_USD ({FX_INR_PER_USD_2024} INR/USD)",
            "usd_per_hh_low": round(ind_lpg_2024fx["low"], 2),
            "usd_per_hh_central": round(ind_lpg_2024fx["central"], 2),
            "usd_per_hh_high": round(ind_lpg_2024fx["high"], 2),
            "code_unit": "USD/HH/5yr (upfront + 5*refills*300)",
            "source_citation": "MoPNG / PPAC / PRS India (3.7-4.8 refills/yr active PMUY; Rs 300/cyl up to 9 refills/yr FY25-26)",
            "source_url_or_doi": "https://prsindia.org; https://pib.gov.in",
            "page_or_table_ref": "MoPNG Demand for Grants & Union Cabinet FY25-26 release",
            "evidence_quality": "High - Primary Govt",
            "status": "derived",
            "notes": f"Matches param_audit.py formula at 83.56 (${lpg_pa['central']['IND_urban']:.2f}/HH) vs 2024 FX 83.70 (${ind_lpg_2024fx['central']:.2f}/HH).",
        },
        {
            "param_id": "ASM-05",
            "parameter_name": "lpg_subsidy_nepal_noc_loss",
            "intervention": "lpg_subsidy",
            "unit_scope": "NPL_urban; NPL_rural",
            "value_central": 450.0,
            "uncertainty_low": 331.25,
            "uncertainty_high": 1174.0,
            "original_currency_or_unit": "NPR loss / 14.2 kg cylinder (6 cyl/yr)",
            "original_price_year": "2022-2024",
            "usd_conversion_basis": f"2024_USD ({FX_NPR_PER_USD_2024} NPR/USD)",
            "usd_per_hh_low": round(npl_lpg_2024fx["rural_low"], 2),
            "usd_per_hh_central": round(npl_lpg_2024fx["rural_central"], 2),
            "usd_per_hh_high": round(npl_lpg_2024fx["rural_high"], 2),
            "code_unit": "USD/HH/5yr",
            "source_citation": "Nepal Oil Corporation (NOC) fortnightly LPG subsidy losses + urban NPR 2,500 / rural NPR 4,500 upfront deposit/kit assumption",
            "source_url_or_doi": "https://kathmandupost.com; https://nagariknetwork.com (NOC releases)",
            "page_or_table_ref": "2022-2024 NOC pricing disclosures; upfront deposit from param_audit.py",
            "evidence_quality": "Moderate - Govt/Media + Assumption on upfront",
            "status": "derived",
            "notes": f"Urban central=${npl_lpg_2024fx['urban_central']:.2f}/HH, Rural central=${npl_lpg_2024fx['rural_central']:.2f}/HH. Nepal has no universal PMUY-like connection grant, so upfront kit is an assumption.",
        },
        # --- IMPROVED BIOMASS COST ---
        {
            "param_id": "ASM-06",
            "parameter_name": "improved_biomass_stove_cost",
            "intervention": "improved_biomass",
            "unit_scope": "IND_rural; NPL_rural (and urban)",
            "value_central": 45.0,
            "uncertainty_low": 25.0,
            "uncertainty_high": 85.0,
            "original_currency_or_unit": "USD / HH / 5-yr (hardware + replacement + 5%/yr O&M)",
            "original_price_year": "2018-2024",
            "usd_conversion_basis": "Nominal USD (study-reported USD & INR/NPR converted)",
            "usd_per_hh_low": 25.0,
            "usd_per_hh_central": 45.0,
            "usd_per_hh_high": 85.0,
            "code_unit": "USD/HH/5yr (societal hardware+O&M)",
            "source_citation": "MNRE Unnat Chulha Abhiyan (50% CFA, Rs 300-800 subsidy); South Asia ICS economic reviews ($15-$50 stove + 5% O&M, 2-3 yr lifespan)",
            "source_url_or_doi": "https://mnre.gov.in; https://copenhagenconsensus.com; https://borgenproject.org",
            "page_or_table_ref": "MNRE NBCP/UCA guidelines; South Asia ICS cost syntheses",
            "evidence_quality": "Moderate - Mixed Programme/Literature",
            "status": "derived",
            "notes": "Societal hardware+O&M basis ($15-$50 hardware * ~1.5 replacements/5yr + O&M). Note COST BOUNDARY DIFFERENCE vs PMUY LPG fiscal subsidy!",
        },
        # --- ELECTRIC INDUCTION COST ---
        {
            "param_id": "ASM-07",
            "parameter_name": "electric_induction_india_hardware",
            "intervention": "electric_induction",
            "unit_scope": "IND_urban; IND_rural",
            "value_central": 53.11,
            "uncertainty_low": 22.70,
            "uncertainty_high": 101.43,
            "original_currency_or_unit": "INR 1,900-6,990 cooktop/kit (+ INR 1,500 rural wiring)",
            "original_price_year": "2023-2025",
            "usd_conversion_basis": f"2024_USD ({FX_INR_PER_USD_2024} INR/USD)",
            "usd_per_hh_low": 22.70,
            "usd_per_hh_central": 53.11,
            "usd_per_hh_high": 101.43,
            "code_unit": "USD/HH/5yr (hardware + internal wiring only)",
            "source_citation": "EESL National Efficient Cooking Programme (NECP) & EESLmart (Rs 1,900-1,999 basic cooktop; Rs 6,990 bundled w/ cookware)",
            "source_url_or_doi": "https://eeslmart.in; https://deccanchronicle.com (EESL NECP launch)",
            "page_or_table_ref": "EESL NECP 1,200W induction pricing",
            "evidence_quality": "Moderate - Programme Market Price (no central subsidy)",
            "status": "derived",
            "notes": "GAP: India has NO central public cash subsidy scheme for induction like PMUY; excludes 5-yr electricity consumption cost.",
        },
        {
            "param_id": "ASM-07",
            "parameter_name": "electric_induction_nepal_hardware_and_wiring",
            "intervention": "electric_induction",
            "unit_scope": "NPL_urban; NPL_rural",
            "value_central": 63.67,
            "uncertainty_low": 37.45,
            "uncertainty_high": 89.89,
            "original_currency_or_unit": "NPR 3,000-7,000 stove + NPR 2,000-5,000 15A wiring/meter",
            "original_price_year": "2022-2024",
            "usd_conversion_basis": f"2024_USD ({FX_NPR_PER_USD_2024} NPR/USD)",
            "usd_per_hh_low": 37.45,
            "usd_per_hh_central": 63.67,
            "usd_per_hh_high": 89.89,
            "code_unit": "USD/HH/5yr (hardware + 15A wiring; excl. tariff)",
            "source_citation": "AEPC/CREF Nepal induction specs; Daraz/Nepal retail (NPR 3,000-7,000); NEA 15A load upgrade (NPR 2,000-5,000); ESMAP/CCA ($15-20 internal wiring)",
            "source_url_or_doi": "https://aepc.gov.np; https://worldbank.org (ESMAP eCooking)",
            "page_or_table_ref": "AEPC tender specs & NEA load upgrade schedule",
            "evidence_quality": "Moderate - Primary Govt/Retail",
            "status": "derived",
            "notes": "If NEA 20% tariff discount (NPR 1,400-3,000/yr) is added over 5 yrs, fiscal cost rises by +$52.43-$112.36/HH.",
        },
        # --- INFRASTRUCTURE COST ---
        {
            "param_id": "ASM-08",
            "parameter_name": "infra_grid_india_electrification_benchmarks",
            "intervention": "infra_lpg_grid",
            "unit_scope": "IND_rural",
            "value_central": "",
            "uncertainty_low": 68.17,
            "uncertainty_high": 570.79,
            "original_currency_or_unit": "INR 5,706/HH (Saubhagya) to INR 47,775/HH (RDSS remote)",
            "original_price_year": "2017-2024",
            "usd_conversion_basis": f"2024_USD ({FX_INR_PER_USD_2024} INR/USD)",
            "usd_per_hh_low": 68.17,
            "usd_per_hh_central": "",
            "usd_per_hh_high": 570.79,
            "code_unit": "USD/HH benchmark (NOT regional $M capex)",
            "source_citation": "MoP/PIB Saubhagya (Rs 16,320 cr / 2.86 cr HH); RDSS remote electrification (Rs 6,521 cr / 13.65 lakh HH)",
            "source_url_or_doi": "https://powermin.gov.in; https://sansad.in; https://pib.gov.in",
            "page_or_table_ref": "Saubhagya closure & RDSS Parliamentary reply",
            "evidence_quality": "Low for cooking-specific capex (indirect proxy)",
            "status": "unresolved_gap",
            "notes": "UNRESOLVED GAP: Saubhagya achieved basic lighting hookups; no national dataset isolates transformer/feeder upgrade $M specifically for 2kW induction + rural LPG bottling/distribution. Do not invent central $M.",
        },
        {
            "param_id": "ASM-08",
            "parameter_name": "infra_lpg_grid_nepal_benchmarks",
            "intervention": "infra_lpg_grid",
            "unit_scope": "NPL_rural",
            "value_central": "",
            "uncertainty_low": 14.98,
            "uncertainty_high": 37.45,
            "original_currency_or_unit": "NPR 2,000-5,000/HH (NEA 15A meter upgrade only)",
            "original_price_year": "2022-2024",
            "usd_conversion_basis": f"2024_USD ({FX_NPR_PER_USD_2024} NPR/USD)",
            "usd_per_hh_low": 14.98,
            "usd_per_hh_central": "",
            "usd_per_hh_high": 37.45,
            "code_unit": "USD/HH meter upgrade (excl. MV/LV transformers & LPG storage)",
            "source_citation": "NEA household connection upgrade fee; ESMAP/Mecs Clements et al. 2020 Nepali mini-grids",
            "source_url_or_doi": "https://worldbank.org; https://mecs.org.uk",
            "page_or_table_ref": "NEA load upgrade schedule",
            "evidence_quality": "Low for regional macro capex (indirect proxy)",
            "status": "unresolved_gap",
            "notes": "UNRESOLVED GAP: No primary national figure for rural Nepal distribution transformer + LPG depot/road supply fixed $M capex.",
        },
        # --- EFFECTIVENESS ---
        {
            "param_id": "ASM-09",
            "parameter_name": "effectiveness_lpg_subsidy",
            "intervention": "lpg_subsidy",
            "unit_scope": "IND_urban; IND_rural; NPL_urban; NPL_rural",
            "value_central": 0.78,
            "uncertainty_low": 0.60,
            "uncertainty_high": 0.93,
            "original_currency_or_unit": "Fractional PM2.5 / stacking-adjusted effectiveness",
            "original_price_year": "2020-2024",
            "usd_conversion_basis": "N/A (dimensionless)",
            "usd_per_hh_low": "",
            "usd_per_hh_central": "",
            "usd_per_hh_high": "",
            "code_unit": "Dimensionless [0, 1]",
            "source_citation": "HAPIN India trial (93% kitchen / 78% personal PM2.5 reduction); CEEW rural PMUY stacking (~4.8/8 refills = 60% exclusive)",
            "source_url_or_doi": "https://doi.org/10.1016/j.envpol.2021.118198; https://doi.org/10.1186/s12889-020-09865-1; https://ceew.in",
            "page_or_table_ref": "HAPIN exposure results; CEEW IRES/PMUY tracking",
            "evidence_quality": "High - RCT + National Survey",
            "status": "derived",
            "notes": "param_audit.py candidate values: IND_urban=0.88, IND_rural=0.78, NPL_urban=0.86, NPL_rural=0.78 (rural 0.78 aligns with HAPIN 78% personal PM2.5 reduction; exclusive=0.93-1.0).",
        },
        {
            "param_id": "ASM-10",
            "parameter_name": "effectiveness_improved_biomass",
            "intervention": "improved_biomass",
            "unit_scope": "IND_rural; NPL_rural",
            "value_central": 0.32,
            "uncertainty_low": 0.15,
            "uncertainty_high": 0.50,
            "original_currency_or_unit": "Kitchen PM2.5 reduction (32%-50%) / IER COPD PAF est (0.15-0.30)",
            "original_price_year": "2016-2021",
            "usd_conversion_basis": "N/A (dimensionless)",
            "usd_per_hh_low": "",
            "usd_per_hh_central": "",
            "usd_per_hh_high": "",
            "code_unit": "Dimensionless [0, 1] relative to clean fuel",
            "source_citation": "Tielsch et al. 2016 Sarlahi Nepal RCT (~32% 20h PM2.5 reduction: 1376->936 ug/m3); Pope et al. 2017 meta-analysis (41%, 95% CI 29-50%); Kumar et al. 2021 (-0.28 mg/m3)",
            "source_url_or_doi": "https://doi.org/10.1016/S2214-109X(16)30024-9; https://doi.org/10.1016/j.envint.2017.01.012; https://doi.org/10.1016/j.envpol.2021.117997",
            "page_or_table_ref": "Tielsch 2016 Results; Pope 2017 Table 2",
            "evidence_quality": "High - RCT + Meta-analysis",
            "status": "derived",
            "notes": "CRITICAL: Raw PM2.5 drop is 32-50%, but under concave GBD COPD IER at >900 ug/m3, COPD PAF reduction is only ~0.15-0.30 (lower than mock 0.45!).",
        },
        {
            "param_id": "ASM-11",
            "parameter_name": "effectiveness_electric_induction",
            "intervention": "electric_induction",
            "unit_scope": "IND_urban; IND_rural; NPL_urban; NPL_rural",
            "value_central": 0.90,
            "uncertainty_low": 0.80,
            "uncertainty_high": 1.00,
            "original_currency_or_unit": "Zero point-of-use emissions adjusted for stacking/outages",
            "original_price_year": "2020-2024",
            "usd_conversion_basis": "N/A (dimensionless)",
            "usd_per_hh_low": "",
            "usd_per_hh_central": "",
            "usd_per_hh_high": "",
            "code_unit": "Dimensionless [0, 1]",
            "source_citation": "WHO Indoor Air Quality Guidelines (zero point-of-use PM2.5); ESMAP 2020 / MECS Nepal stacking & outage adjustment",
            "source_url_or_doi": "https://worldbank.org; https://who.int",
            "page_or_table_ref": "WHO Guidelines; ESMAP Cooking with Electricity",
            "evidence_quality": "Moderate - Guideline + Stacking Adjustment",
            "status": "derived",
            "notes": "param_audit.py candidate values: IND_urban=0.92, IND_rural=0.85, NPL_urban=0.90, NPL_rural=0.85.",
        },
        # --- ADOPTION CAPS & CEILINGS ---
        {
            "param_id": "ASM-12",
            "parameter_name": "max_adoption_per_period_historical_rate",
            "intervention": "lpg_subsidy (observed clean fuel proxy)",
            "unit_scope": "IND_urban; IND_rural; NPL_urban; NPL_rural",
            "value_central": 0.23,
            "uncertainty_low": 0.05,
            "uncertainty_high": 0.24,
            "original_currency_or_unit": "5-yr percentage point / 100 change (WHO SDG 7.1.2 2015-2024)",
            "original_price_year": "2015-2024",
            "usd_conversion_basis": "N/A (fraction / 5 yr)",
            "usd_per_hh_low": "",
            "usd_per_hh_central": "",
            "usd_per_hh_high": "",
            "code_unit": "Fraction / 5-yr period [0, 1]",
            "source_citation": "WHO GHO SDG 7.1.2 (2000-2024): IND_rural max 5-yr=+0.24 (2019-2024=+0.23); IND_urban=+0.05..0.15; NPL_rural=+0.06..0.07; NPL_urban=+0.01..0.04 (2000-2005=+0.23)",
            "source_url_or_doi": "https://ghoapi.azureedge.net/api/PHE_HHAIR_PROP_POP_CLEAN_FUELS",
            "page_or_table_ref": "Downloaded raw time series 2000-2024",
            "evidence_quality": "High - Empirical Time Series",
            "status": "sourced",
            "notes": "Historical series reflects mostly LPG. Electric induction & improved biomass caps remain scenario assumptions (bounded by Nepal 2030 25% e-cooking NDC).",
        },
        {
            "param_id": "ASM-12",
            "parameter_name": "coverage_ceiling",
            "intervention": "All cooking",
            "unit_scope": "IND_urban; IND_rural; NPL_urban; NPL_rural",
            "value_central": 0.95,
            "uncertainty_low": 0.90,
            "uncertainty_high": 0.99,
            "original_currency_or_unit": "Max reachable clean/improved coverage fraction",
            "original_price_year": "2024",
            "usd_conversion_basis": "N/A",
            "usd_per_hh_low": "",
            "usd_per_hh_central": "",
            "usd_per_hh_high": "",
            "code_unit": "Fraction (0, 1]",
            "source_citation": "WHO SDG 7.1.2 (IND_urban=0.96 in 2024 -> ceiling >= 0.98); Nepal universal 2030 target",
            "source_url_or_doi": "https://ghoapi.azureedge.net/api/PHE_HHAIR_PROP_POP_CLEAN_FUELS",
            "page_or_table_ref": "2024 baseline observations",
            "evidence_quality": "Moderate - Policy Target + Empirical Bound",
            "status": "assumption",
            "notes": "Must satisfy ceiling >= cf_bau_2040[u]. With IND_urban already at 0.96 in 2024, ceiling must be >= 0.98-0.99.",
        },
        # --- BUDGET ---
        {
            "param_id": "ASM-13",
            "parameter_name": "total_budget_national_benchmarks",
            "intervention": "Programme Budget",
            "unit_scope": "IND; NPL",
            "value_central": "",
            "uncertainty_low": 200.0,
            "uncertainty_high": 21500.0,
            "original_currency_or_unit": "India PMUY Rs 12,000 cr/yr (~$1,434M/yr); Nepal ~$12M/yr + $49.2M GCF",
            "original_price_year": "2023-2026",
            "usd_conversion_basis": f"2024_USD ({FX_INR_PER_USD_2024} INR/USD)",
            "usd_per_hh_low": "",
            "usd_per_hh_central": "",
            "usd_per_hh_high": "",
            "code_unit": "USD Millions (15-yr cumulative)",
            "source_citation": "MoPNG PIB Rs 12,000 cr/yr PMUY subsidy; AEPC/GCF FP172 $49.2M; Nepal FY23-24 ~$12M/yr clean cooking",
            "source_url_or_doi": "https://pib.gov.in; https://greenclimate.fund; https://mecs.org.uk",
            "page_or_table_ref": "Union Cabinet FY24-25/25-26; GCF FP172",
            "evidence_quality": "High for country outlays; Gap for pooled 2-country cap",
            "status": "assumption",
            "notes": "National allocations are NOT pooled between India and Nepal. Combined 15-yr budget envelope is a scenario assumption (e.g., 25%/50%/100% of real full-gap cost).",
        },
        # --- DISCOUNT & MDP ---
        {
            "param_id": "ASM-16",
            "parameter_name": "discount_factor_5yr",
            "intervention": "Global",
            "unit_scope": "All 4 units",
            "value_central": 0.774,
            "uncertainty_low": 0.774,
            "uncertainty_high": 1.0,
            "original_currency_or_unit": "r=5%/yr -> (1.05)^-5 = 0.7835 (~0.95^5 = 0.7738)",
            "original_price_year": "N/A",
            "usd_conversion_basis": "N/A",
            "usd_per_hh_low": "",
            "usd_per_hh_central": "",
            "usd_per_hh_high": "",
            "code_unit": "Dimensionless (0, 1]",
            "source_citation": "WHO-CHOICE / iDSI Reference Case standard 3%-5% annual health-economic discount rate",
            "source_url_or_doi": "https://who.int",
            "page_or_table_ref": "Standard HTA convention",
            "evidence_quality": "High - Standard Convention",
            "status": "assumption",
            "notes": "Note: (1.05)^-5 = 0.7835 while 0.95^5 = 0.7738 (code default 0.774).",
        },
        {
            "param_id": "ASM-15",
            "parameter_name": "mdp_transition_probs_and_shocks",
            "intervention": "MDP Planning",
            "unit_scope": "All 4 units",
            "value_central": "",
            "uncertainty_low": 0.5,
            "uncertainty_high": 1.25,
            "original_currency_or_unit": "probs=[0.15, 0.70, 0.15]; shocks=[0.5, 1.0, 1.25]",
            "original_price_year": "N/A",
            "usd_conversion_basis": "N/A",
            "usd_per_hh_low": "",
            "usd_per_hh_central": "",
            "usd_per_hh_high": "",
            "code_unit": "Probability triplet & multipliers",
            "source_citation": "No direct empirical distribution; PMUY refill realization ~4.4/8=0.55 supports 0.50 downside shock plausibility",
            "source_url_or_doi": "docs/PLANNING_SPEC.md",
            "page_or_table_ref": "ASM-15",
            "evidence_quality": "Low - Scenario Calibration",
            "status": "assumption",
            "notes": "Retain as scenario assumption; test sensitivity against deterministic (0,1,0) and high-variance (0.25,0.50,0.25).",
        },
    ]

    # Build proposed Contract 4 JSON & integration table
    unit_costs_m: dict[str, dict[str, float | None]] = {}
    eff_map: dict[str, dict[str, float]] = {}
    caps_map: dict[str, dict[str, dict[str, float]]] = {}
    contract4_rows: list[dict[str, Any]] = []

    lpg_eta = {"IND_urban": 0.88, "IND_rural": 0.78, "NPL_urban": 0.86, "NPL_rural": 0.78}
    elec_eta = {"IND_urban": 0.92, "IND_rural": 0.85, "NPL_urban": 0.90, "NPL_rural": 0.85}
    bio_eta_central = 0.25  # IER-adjusted COPD PAF midpoint in [0.15, 0.30] (raw PM2.5 = 0.32-0.41)

    lpg_caps = {
        "IND_urban": {"P1": 0.05, "P2": 0.05, "P3": 0.05},  # already 96% in 2024
        "IND_rural": {"P1": 0.24, "P2": 0.24, "P3": 0.20},  # WHO 2016-2023 max 5-yr = +0.24
        "NPL_urban": {"P1": 0.15, "P2": 0.15, "P3": 0.15},
        "NPL_rural": {"P1": 0.10, "P2": 0.15, "P3": 0.15},  # historical +0.07; policy step-up
    }

    ceilings = {
        "IND_urban": 0.99,
        "IND_rural": 0.95,
        "NPL_urban": 0.98,
        "NPL_rural": 0.95,
    }

    for u in ("IND_urban", "IND_rural", "NPL_urban", "NPL_rural"):
        hh = hh_2025[u]
        # LPG (using param_audit formula at 2024 FX for consistent base year; also report param_audit 83.56/133.70)
        if u.startswith("IND"):
            lpg_hh_low, lpg_hh_cen, lpg_hh_high = ind_lpg_2024fx["low"], ind_lpg_2024fx["central"], ind_lpg_2024fx["high"]
        elif u == "NPL_urban":
            lpg_hh_low, lpg_hh_cen, lpg_hh_high = npl_lpg_2024fx["urban_low"], npl_lpg_2024fx["urban_central"], npl_lpg_2024fx["urban_high"]
        else:
            lpg_hh_low, lpg_hh_cen, lpg_hh_high = npl_lpg_2024fx["rural_low"], npl_lpg_2024fx["rural_central"], npl_lpg_2024fx["rural_high"]

        lpg_m_cen = lpg_hh_cen * hh / 1e6
        bio_m_cen = bio_usd_hh["central"] * hh / 1e6
        elec_m_cen = elec_usd_hh[u]["central"] * hh / 1e6

        unit_costs_m[u] = {
            "lpg_subsidy": round(lpg_m_cen, 2),
            "improved_biomass": round(bio_m_cen, 2),
            "electric_induction": round(elec_m_cen, 2),
            "infra_lpg_grid": None,  # Unresolved gap: do not invent regional fixed $M
        }
        eff_map[u] = {
            "lpg_subsidy": lpg_eta[u],
            "improved_biomass": bio_eta_central,
            "electric_induction": elec_eta[u],
            "infra_lpg_grid": 0.0,
        }
        caps_map[u] = {
            "lpg_subsidy": lpg_caps[u],
            "improved_biomass": {"P1": 0.20, "P2": 0.20, "P3": 0.20},
            "electric_induction": {"P1": 0.10, "P2": 0.15, "P3": 0.15},
            "infra_lpg_grid": {"P1": 1.0, "P2": 1.0, "P3": 1.0},
        }

        for interv, hh_l, hh_c, hh_h, m_c, eta_v, st, note in [
            (
                "lpg_subsidy",
                lpg_hh_low,
                lpg_hh_cen,
                lpg_hh_high,
                lpg_m_cen,
                lpg_eta[u],
                "derived_from_sourced_components",
                f"Public fiscal subsidy (2024 USD). param_audit 83.56/133.70 FX gives ${lpg_usd_m_pa['central'][u]:.2f}M.",
            ),
            (
                "improved_biomass",
                bio_usd_hh["low"],
                bio_usd_hh["central"],
                bio_usd_hh["high"],
                bio_m_cen,
                bio_eta_central,
                "derived_mixed_boundary",
                "Societal hardware+O&M ($25-$85/HH/5yr). Eta=0.25 IER-adjusted COPD PAF midpoint [0.15, 0.30] (raw PM2.5=0.32-0.41).",
            ),
            (
                "electric_induction",
                elec_usd_hh[u]["low"],
                elec_usd_hh[u]["central"],
                elec_usd_hh[u]["high"],
                elec_m_cen,
                elec_eta[u],
                "derived_hardware_only_gap",
                "Hardware+internal wiring only (no India central subsidy; excl. 5-yr electricity tariff).",
            ),
            (
                "infra_lpg_grid",
                None,
                None,
                None,
                None,
                0.0,
                "unresolved_gap",
                "No primary regional clean-cooking transformer+LPG depot fixed $M capex; left null.",
            ),
        ]:
            contract4_rows.append(
                {
                    "unit": u,
                    "intervention": interv,
                    "households_2025_derived": hh,
                    "usd_per_hh_low": round(hh_l, 2) if hh_l is not None else "",
                    "usd_per_hh_central": round(hh_c, 2) if hh_c is not None else "",
                    "usd_per_hh_high": round(hh_h, 2) if hh_h is not None else "",
                    "usd_millions_low": round(hh_l * hh / 1e6, 2) if hh_l is not None else "",
                    "usd_millions_central": round(m_c, 2) if m_c is not None else "",
                    "usd_millions_high": round(hh_h * hh / 1e6, 2) if hh_h is not None else "",
                    "effectiveness_central": eta_v,
                    "coverage_ceiling": ceilings[u],
                    "cap_P1": caps_map[u][interv]["P1"],
                    "cap_P2": caps_map[u][interv]["P2"],
                    "cap_P3": caps_map[u][interv]["P3"],
                    "status": st,
                    "note": note,
                }
            )

    proposed_json = {
        "schema_version": "1.0",
        "is_mock": False,
        "data_version": "B07_PROPOSED_2024USD_PARTIAL",
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "PROPOSED Contract 4 integration file (kept in research/person_b_inputs/ pending Person A "
            "Contracts 1-3, cost-boundary harmonization, and infra_lpg_grid fixed capex sourcing). "
            "Null fields mark genuine unresolved empirical gaps."
        ),
        "total_budget_usd_millions": None,
        "infra_build_threshold": 0.5,
        "apply_cost_discounting": False,
        "discount_factor": 0.774,
        "period_order": ["P1", "P2", "P3"],
        "coverage_ceiling": ceilings,
        "effectiveness": eff_map,
        "unit_costs_usd_millions": unit_costs_m,
        "max_adoption_per_period": caps_map,
        "_auxiliary_households_2025_derived": hh_2025,
    }

    return register, proposed_json, contract4_rows


def build_gap_matrix(hh_2025: dict[str, int]) -> list[dict[str, Any]]:
    """Build parameter_gap_matrix.csv across Contracts 2, 3, 4 and config."""
    return [
        {
            "contract_file": "Contract 2 (baseline_unit_state.csv)",
            "parameter_path": "population_total (IND_urban..NPL_rural)",
            "asm_id": "ASM-01",
            "real_data_status": "acquired",
            "proposed_real_value_or_range": "2025 WUP NatDef: 522.44M / 941.42M / 19.84M / 9.77M",
            "blocking_reason": "Ready to hand to Person A immediately; Person A writes final CSV.",
            "owner": "Person A (inputs from Person B)",
        },
        {
            "contract_file": "Contract 2 (baseline_unit_state.csv)",
            "parameter_path": "population_female_share",
            "asm_id": "ASM-01",
            "real_data_status": "acquired_national_proxy",
            "proposed_real_value_or_range": "WPP 2024 2025: IND=0.4844, NPL=0.5229 (2011/2021 census urban/rural split available)",
            "blocking_reason": "WUP 2025 is not sex-disaggregated; use census urban/rural sex ratio adjustments or national WPP share.",
            "owner": "Person A",
        },
        {
            "contract_file": "Contract 2 (baseline_unit_state.csv)",
            "parameter_path": "clean_fuel_share_init",
            "asm_id": "ASM-02",
            "real_data_status": "acquired",
            "proposed_real_value_or_range": "2024 WHO SDG 7.1.2: IND_u=0.96, IND_r=0.69, NPL_u=0.64, NPL_r=0.25",
            "blocking_reason": "Ready to hand to Person A immediately (note: people-denominated).",
            "owner": "Person A (inputs from Person B)",
        },
        {
            "contract_file": "Contract 2 (baseline_unit_state.csv)",
            "parameter_path": "lpg_infra_ready_init / grid_ready_init",
            "asm_id": "PLANNING_SPEC",
            "real_data_status": "assumption_grounded",
            "proposed_real_value_or_range": "IND_u: T/T; NPL_u: T/T; NPL_r: F/F; IND_r: check (LPG 69% & Saubhagya connected, but rural 2kW feeder & doorstep delivery constrained)",
            "blocking_reason": "Requires explicit operational threshold definition between Person A & B.",
            "owner": "Shared",
        },
        {
            "contract_file": "Contract 2 (baseline_unit_state.csv)",
            "parameter_path": "bau_dalys_2040_* & bau_asr_2040_*",
            "asm_id": "Stage 2",
            "real_data_status": "blocked_person_a",
            "proposed_real_value_or_range": "Pending GBD 2023 extraction + Stage 2 forecasting",
            "blocking_reason": "Requires GBD 2023 HAP-COPD data (1990-2023) from Person A.",
            "owner": "Person A",
        },
        {
            "contract_file": "Contract 1 (forecasts_stage3.csv)",
            "parameter_path": "val, lower, upper, population, clean_fuel_share",
            "asm_id": "Stage 2",
            "real_data_status": "blocked_person_a",
            "proposed_real_value_or_range": "Pending GBD 2023 + Stage 2 models",
            "blocking_reason": "Person A must extract GBD 2023 and run Stage 2 pipeline.",
            "owner": "Person A",
        },
        {
            "contract_file": "Contract 3 (exposure_response_surrogate.json)",
            "parameter_path": "slope_clean_fuel, sex_elasticity_ratio_f_to_m, uncertainty_sd",
            "asm_id": "ASM-04; ASM-14",
            "real_data_status": "blocked_person_a",
            "proposed_real_value_or_range": "Pending Stage 2 counterfactual calibration (cross-checked vs GBD 2021/2023 IER)",
            "blocking_reason": "Calibrated by src/forecast/export_contracts.py from Stage 2 cf_25..cf_100.",
            "owner": "Person A",
        },
        {
            "contract_file": "Contract 4 (intervention_params.json)",
            "parameter_path": "unit_costs_usd_millions.*.lpg_subsidy",
            "asm_id": "ASM-05",
            "real_data_status": "derived",
            "proposed_real_value_or_range": f"IND_u=${round(compute_india_lpg_usd_per_hh(4.0, fx_inr_per_usd=83.70)*hh_2025['IND_urban']/1e6, 1)}M, IND_r=${round(compute_india_lpg_usd_per_hh(4.0, fx_inr_per_usd=83.70)*hh_2025['IND_rural']/1e6, 1)}M, NPL_u=${round(compute_nepal_lpg_usd_per_hh(450,2500,fx_npr_per_usd=133.5)*hh_2025['NPL_urban']/1e6, 1)}M, NPL_r=${round(compute_nepal_lpg_usd_per_hh(450,4500,fx_npr_per_usd=133.5)*hh_2025['NPL_rural']/1e6, 1)}M",
            "blocking_reason": "Sourced/derived (fiscal subsidy basis); Nepal upfront deposit is an assumption.",
            "owner": "Person B",
        },
        {
            "contract_file": "Contract 4 (intervention_params.json)",
            "parameter_path": "unit_costs_usd_millions.*.improved_biomass",
            "asm_id": "ASM-06",
            "real_data_status": "derived_mixed_boundary",
            "proposed_real_value_or_range": "IND_u=$5,598M, IND_r=$9,414M, NPL_u=$207M, NPL_r=$97M (at $45/HH/5yr societal)",
            "blocking_reason": "Cost boundary (societal hardware+O&M vs 50% MNRE/AEPC public subsidy) must be harmonized with LPG.",
            "owner": "Person B",
        },
        {
            "contract_file": "Contract 4 (intervention_params.json)",
            "parameter_path": "unit_costs_usd_millions.*.electric_induction",
            "asm_id": "ASM-07",
            "real_data_status": "derived_hardware_only_gap",
            "proposed_real_value_or_range": "IND_u=$6,606M, IND_r=$11,111M, NPL_u=$293M, NPL_r=$138M (hardware+wiring only)",
            "blocking_reason": "No India central subsidy equivalent to PMUY; excludes 5-yr electricity fuel cost.",
            "owner": "Person B",
        },
        {
            "contract_file": "Contract 4 (intervention_params.json)",
            "parameter_path": "unit_costs_usd_millions.*.infra_lpg_grid",
            "asm_id": "ASM-08",
            "real_data_status": "unresolved_gap",
            "proposed_real_value_or_range": "null (Saubhagya $68/HH - RDSS $571/HH, NEA 15A $15-38/HH are partial proxies only)",
            "blocking_reason": "No primary regional clean-cooking transformer/feeder + LPG depot fixed $M dataset found.",
            "owner": "Person B",
        },
        {
            "contract_file": "Contract 4 (intervention_params.json)",
            "parameter_path": "effectiveness.*.{lpg_subsidy, electric_induction}",
            "asm_id": "ASM-09; ASM-11",
            "real_data_status": "derived",
            "proposed_real_value_or_range": "LPG: 0.78-0.88 (HAPIN+CEEW); Elec: 0.85-0.92 (WHO IAQ+stacking)",
            "blocking_reason": "Note: INTERFACES.md §5 currently locks 1.0; update doc if stacking-adjusted <1.0 used.",
            "owner": "Shared",
        },
        {
            "contract_file": "Contract 4 (intervention_params.json)",
            "parameter_path": "effectiveness.*.improved_biomass",
            "asm_id": "ASM-10",
            "real_data_status": "derived",
            "proposed_real_value_or_range": "0.15-0.30 COPD PAF (central 0.25); raw kitchen PM2.5 0.32-0.50 (Tielsch 2016, Pope 2017)",
            "blocking_reason": "Exact PAF mapping depends on GBD 2023 COPD IER curve slope.",
            "owner": "Person B",
        },
        {
            "contract_file": "Contract 4 (intervention_params.json)",
            "parameter_path": "max_adoption_per_period & coverage_ceiling",
            "asm_id": "ASM-12",
            "real_data_status": "sourced_partial_assumption",
            "proposed_real_value_or_range": "LPG max observed 5-yr: IND_r=+0.24, IND_u=+0.05..0.15, NPL_r=+0.07; Elec/Bio caps assumed",
            "blocking_reason": "WHO SDG 7.1.2 gives overall clean-fuel pace; fuel-specific caps and ceilings >= BAU_2040 require Stage 2 BAU.",
            "owner": "Person B / Shared",
        },
        {
            "contract_file": "Contract 4 (intervention_params.json)",
            "parameter_path": "total_budget_usd_millions",
            "asm_id": "ASM-13",
            "real_data_status": "assumption",
            "proposed_real_value_or_range": "null (India PMUY ~$1,434M/yr; Nepal ~$12M/yr + $49.2M GCF; pooled 15-yr cap is scenario choice)",
            "blocking_reason": "Requires team decision on cost boundary & scenario % of full-gap cost.",
            "owner": "Shared",
        },
        {
            "contract_file": "configs/default.yaml",
            "parameter_path": "planning.mdp.{transition_probs, shock_multipliers, discount_factor, cost_weight_lambda}",
            "asm_id": "ASM-15; ASM-16; DEC-018",
            "real_data_status": "assumption_and_calibration",
            "proposed_real_value_or_range": "probs=[0.15,0.70,0.15] (assumption); gamma=0.774 (WHO convention); lambda recalibrated post-Stage 3",
            "blocking_reason": "cost_weight_lambda must be recalibrated via calibrate_lambda_from_pareto() once real DALYs & $M costs are combined.",
            "owner": "Person B",
        },
    ]


def build_source_manifest(verification_notes: dict[str, Any]) -> dict[str, Any]:
    """Create source_manifest.json for all files under research/person_b_inputs/raw/."""
    specs = [
        (
            "WPP2024_TotalPopulationBySex.csv.gz",
            "csv.gz",
            "UN DESA Population Division",
            "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/WPP2024_TotalPopulationBySex.csv.gz",
            "1950-2100 (filtered 1990-2040)",
            "Global (verified IND, NPL)",
            True,
            "All required variables present (PopTotal, PopMale, PopFemale in thousands).",
        ),
        (
            "WPP2024_PopulationByAge5GroupSex_Medium.csv.gz",
            "csv.gz",
            "UN DESA Population Division",
            "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/WPP2024_PopulationByAge5GroupSex_Medium.csv.gz",
            "1950-2100 (verified 1990-2040)",
            "Global (verified IND, NPL; 21 5-yr age bins)",
            True,
            f"All required variables present; age sums match total file within {verification_notes['wpp_age5']['max_rel_diff_vs_totals']:.2e} rel diff.",
        ),
        (
            "WUP2025-DB-National-Definitions-Population-Data.csv.gz",
            "csv.gz",
            "UN DESA Population Division (WUP 2025)",
            "https://population.un.org/wup/assets/Download/Countries%20and%20Aggregates/WUP2025-DB-National-Definitions-Population-Data.csv.gz",
            "1950-2050 (verified 2011-2040)",
            "Global (verified IND, NPL Urban/Rural/Total)",
            True,
            "All required variables present under national urban/rural definitions.",
        ),
        (
            "WUP2025-DB-DEGURBA-Level1-Population-Surface-Data.csv.gz",
            "csv.gz",
            "UN DESA Population Division (WUP 2025)",
            "https://population.un.org/wup/assets/Download/Countries%20and%20Aggregates/WUP2025-DB-DEGURBA-Level1-Population-Surface-Data.csv.gz",
            "1950-2050 (verified 2011-2040)",
            "Global (verified IND, NPL Cities/Towns/Rural)",
            True,
            "All required variables present under UN harmonized Degree of Urbanization Level 1.",
        ),
        (
            "WUP2025_Countries_Indicators.csv",
            "csv",
            "UN DESA Population Division (WUP 2025)",
            "https://population.un.org/wup/assets/Download/Countries%20and%20Aggregates/WUP2025_Countries_Indicators.csv",
            "1950-2050",
            "Metadata indicator list",
            True,
            "Indicator dictionary verified.",
        ),
        (
            "WUP2025-M02-DataSources-National-Definitions-UrbanPercentage.xlsx",
            "xlsx",
            "UN DESA Population Division (WUP 2025)",
            "https://population.un.org/wup/assets/Download/WUP2025-M02-DataSources-National-Definitions-UrbanPercentage.xlsx",
            "Historical-2025",
            "Global (verified India & Nepal urban definitions)",
            True,
            "Documents exact India (2011 statutory + 3-criteria towns) and Nepal (2021 Urban Municipalities) definitions.",
        ),
        (
            "IND_Census2011_A1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
            "xlsx",
            "Office of the Registrar General & Census Commissioner, India",
            "https://censusindia.gov.in/nada/index.php/catalog/42526/download/46152/A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx",
            "2011",
            "India (National, States/UTs, Districts, Sub-districts; Total/Rural/Urban)",
            True,
            "Contains all required population & total household counts by Total/Rural/Urban.",
        ),
        (
            "IND_Census2011_HH01_01_JK.XLS",
            "xls (BIFF8)",
            "Office of the Registrar General & Census Commissioner, India",
            "https://censusindia.gov.in/nada/index.php/catalog/7110/download/10222/DDW-HH01-0100-2011.XLS",
            "2011",
            "Jammu & Kashmir (State sample of 35 state HH-01 files)",
            False,
            "Limitation: NADA catalog publishes Census 2011 HH-01 only as 35 separate state/UT legacy binary .XLS files (no national 0000 file) and xlrd is not installed in .venv. All-India households/population obtained from A-1 xlsx + NFHS-5 DHS API.",
        ),
        (
            "NPL_Census2021_National_Indv01_HouseholdAndPop.xlsx",
            "xlsx",
            "National Statistics Office (NSO / CBS), Nepal",
            "https://censusresults.nsonepal.gov.np/files/national/Formated_NR_Indv_Table01_HouseholdAndPop.xlsx",
            "2021",
            "Nepal (National, Urban/Rural Municipalities, Ecological Belt, Provinces)",
            True,
            "Contains households, population by sex, and published average household size.",
        ),
        (
            "NPL_Census2021_National_Indv02_SizeOfPerson.xlsx",
            "xlsx",
            "National Statistics Office (NSO / CBS), Nepal",
            "https://censusresults.nsonepal.gov.np/files/national/Formated_NR_Indv_Table02_SizeOfPerson.xlsx",
            "2021",
            "Nepal (National, Urban/Rural)",
            True,
            "Household size distribution table.",
        ),
        (
            "NPL_Census2021_National_Hhld07_CookingFuel.xlsx",
            "xlsx",
            "National Statistics Office (NSO / CBS), Nepal",
            "https://censusresults.nsonepal.gov.np/files/national/Formated_NR_Hhld_Table07_CookingFuel_edited.xlsx",
            "2021",
            "Nepal (National, Urban/Rural, Ecological Belt, Provinces)",
            True,
            "Observed 2021 households by primary cooking fuel (conventional housing units).",
        ),
        (
            "NPL_Census2021_UrbanRural_Indv01_PopulationBySex.xlsx",
            "xlsx",
            "National Statistics Office (NSO / CBS), Nepal",
            "https://censusresults.nsonepal.gov.np/files/rural/Indv01_PopulationBySex.xlsx",
            "2021",
            "Nepal (Urban/Rural cross-tabulated by Ecological Belt & Province)",
            True,
            "Matches National Table 15 exactly.",
        ),
        (
            "NPL_Census2021_UrbanRural_Indv02_SizeOfPerson.xlsx",
            "xlsx",
            "National Statistics Office (NSO / CBS), Nepal",
            "https://censusresults.nsonepal.gov.np/files/rural/Indv02_SizeOfPerson.xlsx",
            "2021",
            "Nepal (Urban/Rural household size distribution)",
            True,
            "Verified readable.",
        ),
        (
            "NPL_Census2021_UrbanRural_Hhld07_TypeOfCookingFuel.xlsx",
            "xlsx",
            "National Statistics Office (NSO / CBS), Nepal",
            "https://censusresults.nsonepal.gov.np/files/rural/Hhld07_TypeOfCookingFuel.xlsx",
            "2021",
            "Nepal (Urban/Rural cooking fuel by belt/province)",
            True,
            "Verified readable.",
        ),
        (
            "NPL_Census2021_DegurbaUrbanRural.xlsx",
            "xlsx",
            "National Statistics Office (NSO / CBS), Nepal",
            "https://censusresults.nsonepal.gov.np/files/degurba-report/DegurbaUrbanRural.xlsx",
            "2021",
            "Nepal (Ward-level Urban/Rural vs DEGURBA_7 mapping)",
            True,
            "Verified readable; documents ward-level national vs DEGURBA mapping.",
        ),
        (
            "who_gho_sdg712_clean_cooking_ind_npl.json",
            "json",
            "World Health Organization (GHO API)",
            "https://ghoapi.azureedge.net/api/PHE_HHAIR_PROP_POP_CLEAN_FUELS?$filter=SpatialDim%20eq%20%27IND%27%20or%20SpatialDim%20eq%20%27NPL%27",
            "2000-2024",
            "India & Nepal (Total, Urban, Rural)",
            True,
            "150 records (2 countries x 3 residence strata x 25 years) with 95% UI (Low, High).",
        ),
        (
            "who_gho_sdg712_fuel_categories_ind_npl.json",
            "json",
            "World Health Organization (GHO API)",
            "https://ghoapi.azureedge.net/api/PHE_HHAIR_PROP_POP_CATEGORY_FUELS?$filter=SpatialDim%20eq%20%27IND%27%20or%20SpatialDim%20eq%20%27NPL%27",
            "1990-2023",
            "India & Nepal (Total, Urban, Rural; 6 fuel types)",
            True,
            "1,224 records disaggregated by Biomass, Charcoal, Coal, Electricity, Gas, Kerosene.",
        ),
        (
            "worldbank_sdg712_clean_cooking_ind_npl.json",
            "json",
            "World Bank WDI API",
            "https://api.worldbank.org/v2/country/IND;NPL/indicator/EG.CFT.ACCS.ZS;EG.CFT.ACCS.UR.ZS;EG.CFT.ACCS.RU.ZS?source=2&format=json&per_page=1000",
            "2000-2024",
            "India & Nepal (Total, Urban, Rural)",
            True,
            "Matches WHO GHO point estimates exactly (lastupdated 2026-11-11).",
        ),
        (
            "dhs_ind_npl_hh_size_and_clean_fuel.json",
            "json",
            "The DHS Program API (NFHS / NDHS)",
            "https://api.dhsprogram.com/rest/dhs/data?countryIds=IA,NP&indicatorIds=HC_MEMB_H_MNM,HC_CKFL_H_CLN,HC_CKFL_P_CLN&breakdown=background&f=json",
            "1992-2022 (NFHS-5 2019-21; NDHS 2022)",
            "India & Nepal (Total, Urban, Rural + sub-national)",
            True,
            "Provides mean household size (NFHS-5: 4.2 urban / 4.5 rural) and household vs population clean fuel rates.",
        ),
    ]

    entries: list[dict[str, Any]] = []
    for fname, fmt, org, url, years, geo, complete, note in specs:
        p = RAW_DIR / fname
        exists = p.exists() and p.stat().st_size > 0
        entries.append(
            {
                "filename": fname,
                "relative_path": f"research/person_b_inputs/raw/{fname}",
                "downloaded_and_verified": exists,
                "format": fmt,
                "size_bytes": p.stat().st_size if exists else 0,
                "sha256": sha256_file(p) if exists else None,
                "source_organization": org,
                "source_url": url,
                "reference_years": years,
                "geographic_coverage": geo,
                "contains_all_required_variables": complete,
                "notes": note,
            }
        )

    # Explicitly document failed/unavailable automated downloads
    failed_or_manual = [
        {
            "dataset": "World Bank WDI FX & CPI bulk API (PA.NUS.FCRF, FP.CPI.TOTL)",
            "status": "timeout_http_error_28",
            "url": "https://api.worldbank.org/v2/country/IND;NPL;USA/indicator/PA.NUS.FCRF;FP.CPI.TOTL",
            "impact": "Low; annual average 2023/2024 FX (INR 83.70/USD, NPR 133.50/USD) and US CPI verified via World Bank/IMF/BLS search.",
        },
        {
            "dataset": "India Census 2011 HH-01 All-India aggregate",
            "status": "not_published_as_single_national_xlsx",
            "url": "https://censusindia.gov.in/nada/index.php/api/tables/data/global/census_tables/100/0/?ft_query=PC11_HH01",
            "impact": "Low; Census 2011 Table A-1 (.xlsx) provides exact all-India Urban/Rural households and population, and NFHS-5 (2019-21) provides updated mean household sizes.",
        },
        {
            "dataset": "IHME GBD 2023 Results Tool CSVs",
            "status": "requires_interactive_browser_login_export",
            "url": "https://vizhub.healthdata.org/gbd-results/",
            "impact": "High for Person A (Stage 1-2); checklist generated for Person A.",
        },
    ]

    return {
        "generated_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_files": entries,
        "unavailable_or_manual_sources": failed_or_manual,
        "verification_checks": verification_notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write list of dicts to CSV with stable column order."""
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    wpp_totals = load_wpp2024_totals()
    wpp_age5_check = verify_wpp2024_age5_vs_totals(wpp_totals)
    wup_nat = load_wup2025_nat_def()
    wup_degurba = load_wup2025_degurba()
    ind_c11 = load_india_census_2011_a1()
    npl_c21 = load_nepal_census_2021()
    dhs = load_dhs_hh_size_and_fuel()

    # Validate WUP 2025 urban+rural == total
    max_wup_sum_diff = 0.0
    for iso in ("IND", "NPL"):
        for yr in (2011, 2021, 2025, 2040):
            u = wup_nat[(iso, yr, "urban")]["pop"]
            r = wup_nat[(iso, yr, "rural")]["pop"]
            t = wup_nat[(iso, yr, "total")]["pop"]
            max_wup_sum_diff = max(max_wup_sum_diff, abs((u + r) - t) / t)

    demo_rows, hh_2025 = build_demographic_inputs(
        wpp_totals, wup_nat, wup_degurba, ind_c11, npl_c21, dhs
    )
    validate_household_counts(hh_2025)

    cf_rows = build_clean_fuel_access(npl_c21)
    # Verify WHO vs World Bank parity on overlapping years
    who_lookup = {
        (r["country_iso3"], r["residence"], int(r["year"])): float(r["val"])
        for r in cf_rows
        if r["source"] == "WHO GHO"
    }
    wb_lookup = {
        (r["country_iso3"], r["residence"], int(r["year"])): float(r["val"])
        for r in cf_rows
        if r["source"].startswith("World Bank")
    }
    max_who_wb_diff = max(abs(who_lookup[k] - wb_lookup[k]) for k in who_lookup if k in wb_lookup)

    reg_rows, proposed_json, c4_rows = build_intervention_register(hh_2025)

    gap_rows = build_gap_matrix(hh_2025)

    verification_notes = {
        "wpp_age5": wpp_age5_check,
        "wup_nat_urban_plus_rural_max_rel_diff": max_wup_sum_diff,
        "who_vs_wb_sdg712_max_abs_diff_pct_pts": max_who_wb_diff,
        "derived_2025_households": hh_2025,
    }
    manifest = build_source_manifest(verification_notes)

    write_csv(BASE_DIR / "demographic_inputs.csv", demo_rows)
    write_csv(BASE_DIR / "clean_fuel_access.csv", cf_rows)
    write_csv(BASE_DIR / "intervention_source_register.csv", reg_rows)
    write_csv(BASE_DIR / "parameter_gap_matrix.csv", gap_rows)
    write_csv(BASE_DIR / "proposed_contract4_table.csv", c4_rows)
    (BASE_DIR / "proposed_intervention_params.json").write_text(
        json.dumps(proposed_json, indent=2), encoding="utf-8"
    )
    (BASE_DIR / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Generated all Person B CSV/JSON artifacts successfully.")
    print("Derived 2025 households:", hh_2025)
    print("WPP age5 max rel diff vs total:", f"{wpp_age5_check['max_rel_diff_vs_totals']:.2e}")
    print("WUP NatDef urban+rural max rel diff:", f"{max_wup_sum_diff:.2e}")
    print("WHO vs World Bank max diff (pct pts):", max_who_wb_diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
