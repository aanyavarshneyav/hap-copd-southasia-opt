"""Raw dataset validation utility for AEGIS-GBD.

Audits downloaded datasets in data/raw against requirements in docs/DATA_SPEC.md.
Validates file integrity, schemas, geographic coverage, temporal coverage,
indicator variables, uncertainty invariants, and duplicate records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_PANEL_COUNTRIES: frozenset[str] = frozenset(
    {"India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"}
)
PROJECT_PANEL_ISO3: frozenset[str] = frozenset({"IND", "NPL", "PAK", "BGD", "BTN"})
HISTORICAL_YEARS: range = range(1990, 2024)  # 1990 to 2023 inclusive
WPP_PROJECTION_YEARS: range = range(1990, 2041)  # 1990 to 2040 inclusive


@dataclass
class DatasetValidationResult:
    """Detailed validation result for a single raw dataset."""

    name: str
    target_role: str
    filepath: str
    status: str  # PASS, PASS WITH WARNINGS, FAIL, NOT TESTED
    file_exists: bool
    size_bytes: int = 0
    sha256: str = ""
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    locations_found: list[str] = field(default_factory=list)
    missing_locations: list[str] = field(default_factory=list)
    years_min: int | None = None
    years_max: int | None = None
    years_count: int = 0
    missing_years: list[int] = field(default_factory=list)
    duplicates_count: int = 0
    null_count_key_series: int = 0
    invariant_violations: int = 0
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash in 8MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_gbd_zip(
    zip_path: Path,
    expected_cause: str,
    expected_rei: str | None,
    expected_role: str,
    require_all_ages_5yr: bool = False,
    require_sex_split: bool = True,
    require_level3_causes: bool = False,
) -> DatasetValidationResult:
    """Validate an IHME GBD 2023 Results Tool export zip archive."""
    res = DatasetValidationResult(
        name=zip_path.name,
        target_role=expected_role,
        filepath=str(zip_path),
        status="PASS",
        file_exists=zip_path.exists(),
    )
    if not zip_path.exists():
        res.status = "FAIL"
        res.critical_errors.append(f"File not found: {zip_path}")
        return res

    res.size_bytes = zip_path.stat().st_size
    res.sha256 = compute_sha256(zip_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                res.status = "FAIL"
                res.critical_errors.append("Zip archive contains no CSV files.")
                return res

            cname = csv_names[0]
            with zf.open(cname) as f:
                df = pd.read_csv(f)
    except Exception as e:
        res.status = "FAIL"
        res.critical_errors.append(f"Failed to read/decompress zip archive: {e}")
        return res

    res.row_count = len(df)
    res.columns = list(df.columns)

    # Invariant checks: 0 <= lower <= val <= upper
    if {"val", "lower", "upper"}.issubset(df.columns):
        num_val = pd.to_numeric(df["val"], errors="coerce")
        num_low = pd.to_numeric(df["lower"], errors="coerce")
        num_upp = pd.to_numeric(df["upper"], errors="coerce")

        if num_val.isna().sum() > 0 or num_low.isna().sum() > 0 or num_upp.isna().sum() > 0:
            res.critical_errors.append("Non-numeric values found in val/lower/upper.")

        neg_vals = (num_val < 0).sum()
        inv_violations = ((num_low > num_val) | (num_val > num_upp)).sum()
        res.invariant_violations = int(inv_violations)
        if neg_vals > 0:
            res.critical_errors.append(f"Found {neg_vals} negative values in val.")
        if inv_violations > 0:
            res.critical_errors.append(
                f"Found {inv_violations} violations of invariant 0 <= lower <= val <= upper."
            )

    # Geographic coverage check
    if "location_name" in df.columns:
        found_locs = set(df["location_name"].unique())
        res.locations_found = sorted(found_locs)
        missing_locs = sorted(PROJECT_PANEL_COUNTRIES - found_locs)
        res.missing_locations = missing_locs
        if missing_locs:
            res.critical_errors.append(f"Missing required panel countries: {missing_locs}")

    # Temporal coverage check
    if "year" in df.columns:
        years = sorted(df["year"].unique())
        res.years_min = int(min(years)) if years else None
        res.years_max = int(max(years)) if years else None
        res.years_count = len(years)
        missing_yrs = sorted(set(HISTORICAL_YEARS) - set(years))
        res.missing_years = missing_yrs
        if missing_yrs:
            res.critical_errors.append(f"Missing historical years (1990-2023): {missing_yrs}")

    # Cause & REI check
    if "cause_name" in df.columns:
        causes = df["cause_name"].unique().tolist()
        if expected_cause not in causes:
            res.critical_errors.append(
                f"Expected cause '{expected_cause}' not found. Present causes: {causes}"
            )
        if require_level3_causes and len(causes) <= 1:
            res.critical_errors.append(
                f"Expected multiple Level 3 linked causes, but only found: {causes}"
            )

    if expected_rei and "rei_name" in df.columns:
        reis = df["rei_name"].unique().tolist()
        if expected_rei not in reis:
            res.critical_errors.append(
                f"Expected risk factor '{expected_rei}' not found. Present risks: {reis}"
            )

    # Sex coverage check
    if "sex_name" in df.columns:
        sexes = set(df["sex_name"].unique())
        if require_sex_split:
            expected_sexes = {"Both", "Male", "Female"}
            missing_sexes = expected_sexes - sexes
            if missing_sexes:
                res.critical_errors.append(
                    f"Missing required sexes: {sorted(missing_sexes)}. Found only: {sorted(sexes)}."
                )

    # Age granularity check
    if "age_name" in df.columns:
        ages = set(df["age_name"].unique())
        if require_all_ages_5yr:
            has_5yr = any(any(char.isdigit() for char in a) for a in ages)
            if not has_5yr:
                res.critical_errors.append(
                    f"Missing 5-year age groups. Found only aggregate ages: {sorted(ages)}."
                )
        if require_level3_causes and "Age-standardized" not in ages:
            res.critical_errors.append(
                f"Missing 'Age-standardized' age group. Found only: {sorted(ages)}."
            )

    # Duplicates check
    dup_cols = ["location_id", "year", "sex_id", "age_id", "cause_id", "measure_id", "metric_id"]
    if "rei_id" in df.columns:
        dup_cols.append("rei_id")
    if set(dup_cols).issubset(df.columns):
        res.duplicates_count = int(df.duplicated(subset=dup_cols).sum())
        if res.duplicates_count > 0:
            res.critical_errors.append(f"Found {res.duplicates_count} duplicate rows.")

    if res.critical_errors:
        res.status = "FAIL"
    elif res.warnings:
        res.status = "PASS WITH WARNINGS"
    else:
        res.status = "PASS"

    return res


def validate_who_data(file_path: Path) -> DatasetValidationResult:
    """Validate the WHO Global Health Observatory / Household Energy Database CSV."""
    res = DatasetValidationResult(
        name=file_path.name,
        target_role="WHO Clean Cooking Fuel Access (SDG 7.1.2)",
        filepath=str(file_path),
        status="PASS",
        file_exists=file_path.exists(),
    )
    if not file_path.exists():
        res.status = "FAIL"
        res.critical_errors.append(f"File not found: {file_path}")
        return res

    res.size_bytes = file_path.stat().st_size
    res.sha256 = compute_sha256(file_path)

    if file_path.name != "who_sdg712_clean_cooking.csv":
        res.warnings.append(
            f"Non-standard filename '{file_path.name}'. Expected 'who_sdg712_clean_cooking.csv'."
        )

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        res.status = "FAIL"
        res.critical_errors.append(f"Failed to read CSV: {e}")
        return res

    res.row_count = len(df)
    res.columns = list(df.columns)

    req_cols = ["IndicatorCode", "SpatialDimValueCode", "Period", "Dim1", "FactValueNumeric"]
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        res.critical_errors.append(f"Missing required columns: {missing_cols}")
        res.status = "FAIL"
        return res

    df_sa = df[df["SpatialDimValueCode"].isin(PROJECT_PANEL_ISO3)]
    found_iso = set(df_sa["SpatialDimValueCode"].unique())
    res.locations_found = sorted(df_sa["Location"].unique())
    missing_iso = PROJECT_PANEL_ISO3 - found_iso
    if missing_iso:
        res.critical_errors.append(f"Missing project countries: {sorted(missing_iso)}")

    years = sorted(df_sa["Period"].unique())
    res.years_min = int(min(years)) if years else None
    res.years_max = int(max(years)) if years else None
    res.years_count = len(years)
    missing_yrs = sorted(set(HISTORICAL_YEARS) - set(years))
    if missing_yrs:
        res.critical_errors.append(f"Missing historical years (1990-2023): {missing_yrs}")

    # Proportions bounds check: 0 <= val <= 100
    df_prop = df_sa[df_sa["IndicatorCode"] == "PHE_HHAIR_PROP_POP_CATEGORY_FUELS"]
    if len(df_prop) > 0:
        vals = df_prop["FactValueNumeric"].dropna()
        if (vals < 0).sum() > 0 or (vals > 100).sum() > 0:
            res.critical_errors.append("Fuel proportion values outside valid range [0, 100].")

    # Duplicates check
    dup_keys = ["SpatialDimValueCode", "Period", "Dim1", "Dim2", "IndicatorCode"]
    res.duplicates_count = int(df_sa.duplicated(subset=dup_keys).sum())
    if res.duplicates_count > 0:
        res.critical_errors.append(f"Found {res.duplicates_count} duplicate rows in SA subset.")

    if len(df) > 10000:
        res.warnings.append(
            f"File contains global data ({len(df)} rows, {res.size_bytes / (1024 * 1024):.1f} MB); "
            "recommend filtering and saving as pre-processed parquet during Stage 0."
        )

    if res.critical_errors:
        res.status = "FAIL"
    elif res.warnings:
        res.status = "PASS WITH WARNINGS"
    else:
        res.status = "PASS"

    return res


def validate_wpp_files(raw_dir: Path) -> list[DatasetValidationResult]:
    """Validate UN World Population Prospects 2024 directories and CSVs."""
    results = []
    wpp_dirs = [
        ("WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv", 1950, 2023),
        ("WPP2024_PopulationBySingleAgeSex_Medium_2024-2100.csv", 2024, 2100),
    ]

    for dir_name, min_expected_yr, max_expected_yr in wpp_dirs:
        dir_path = raw_dir / dir_name
        csv_path = dir_path / dir_name if dir_path.is_dir() else dir_path

        res = DatasetValidationResult(
            name=dir_name,
            target_role=f"UN WPP 2024 Population Projections ({min_expected_yr}-{max_expected_yr})",
            filepath=str(csv_path),
            status="PASS",
            file_exists=csv_path.exists(),
        )

        if not csv_path.exists():
            res.status = "FAIL"
            res.critical_errors.append(f"WPP CSV not found at {csv_path}")
            results.append(res)
            continue

        res.size_bytes = csv_path.stat().st_size
        res.sha256 = compute_sha256(csv_path)

        if dir_path.is_dir():
            res.warnings.append(
                f"File is nested inside directory with .csv extension: '{dir_name}/{dir_name}'. "
                "Target specification expects consolidated 'un_wpp2024_pop_by_age_sex.csv'."
            )

        # Process chunks to extract project panel
        chunks = []
        total_rows = 0
        for chunk in pd.read_csv(csv_path, chunksize=150000, low_memory=False):
            total_rows += len(chunk)
            m = chunk[chunk["ISO3_code"].isin(PROJECT_PANEL_ISO3)]
            if len(m) > 0:
                chunks.append(m)

        res.row_count = total_rows
        if not chunks:
            res.status = "FAIL"
            res.critical_errors.append("No project panel countries found in WPP export.")
            results.append(res)
            continue

        df_sa = pd.concat(chunks, ignore_index=True)
        res.columns = list(df_sa.columns)
        res.locations_found = sorted(df_sa["Location"].unique())
        missing_iso = PROJECT_PANEL_ISO3 - set(df_sa["ISO3_code"].unique())
        if missing_iso:
            res.critical_errors.append(f"Missing panel countries in WPP: {sorted(missing_iso)}")

        years = sorted(df_sa["Time"].unique())
        res.years_min = int(min(years))
        res.years_max = int(max(years))
        res.years_count = len(years)

        # Internal consistency invariant: PopMale + PopFemale ≈ PopTotal
        diff = np.abs(df_sa["PopMale"] + df_sa["PopFemale"] - df_sa["PopTotal"])
        if diff.max() > 0.01:
            res.critical_errors.append(
                f"Discrepancy: PopMale + PopFemale vs PopTotal (max diff: {diff.max()}k persons)."
            )

        # Granularity check
        if df_sa["AgeGrp"].nunique() > 50:
            res.warnings.append(
                "Contains single-year ages (101 bins: 0..100+). "
                "DATA_SPEC.md §2.4 specifies 5-year groups (21 bins: 0-4..100+). "
                "Aggregate to 5-year bins during Stage 0 ingestion."
            )

        # Duplicates
        dup_keys = ["ISO3_code", "Time", "AgeGrp", "Variant"]
        res.duplicates_count = int(df_sa.duplicated(subset=dup_keys).sum())
        if res.duplicates_count > 0:
            res.critical_errors.append(
                f"Found {res.duplicates_count} duplicate rows in WPP SA subset."
            )

        if res.critical_errors:
            res.status = "FAIL"
        elif res.warnings:
            res.status = "PASS WITH WARNINGS"
        else:
            res.status = "PASS"

        results.append(res)

    return results


def validate_worldbank_json(json_path: Path) -> DatasetValidationResult:
    """Validate World Bank Open Data JSON indicator files."""
    res = DatasetValidationResult(
        name=json_path.name,
        target_role="World Bank Clean Cooking Access (Alternative/Supplementary)",
        filepath=str(json_path),
        status="PASS WITH WARNINGS",
        file_exists=json_path.exists(),
    )
    if not json_path.exists():
        res.status = "FAIL"
        res.critical_errors.append(f"File not found: {json_path}")
        return res

    res.size_bytes = json_path.stat().st_size
    res.sha256 = compute_sha256(json_path)

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        records = data[1]
    except Exception as e:
        res.status = "FAIL"
        res.critical_errors.append(f"Failed to parse JSON: {e}")
        return res

    res.row_count = len(records)
    df = pd.DataFrame(records)
    res.columns = list(df.columns)

    df_sa = df[df["countryiso3code"].isin(PROJECT_PANEL_ISO3)]
    res.locations_found = sorted(df_sa["country"].apply(lambda x: x.get("value")).unique())

    res.warnings.append(
        "Alternative JSON format and indicator source (World Bank API rounded integers, 2000-2024);"
        " WHO GHO data in data (1).csv provides primary historical coverage (1990-2023)."
    )
    return res


def run_full_validation(raw_dir: Path) -> list[DatasetValidationResult]:
    """Run validation across all datasets present in raw directory."""
    results: list[DatasetValidationResult] = []

    # 1. Primary GBD HAP-COPD
    gbd_hap_path = raw_dir / "IHME-GBD_2023_DATA-010d0c57-1.zip"
    results.append(
        validate_gbd_zip(
            gbd_hap_path,
            expected_cause="Chronic obstructive pulmonary disease",
            expected_rei="Household air pollution from solid fuels",
            expected_role="Primary historical burden outcome (Stage 1-2)",
            require_all_ages_5yr=True,
            require_sex_split=True,
        )
    )

    # 2. GBD Total COPD
    gbd_total_path = raw_dir / "IHME-GBD_2023_DATA-7546b8be-1.zip"
    results.append(
        validate_gbd_zip(
            gbd_total_path,
            expected_cause="Chronic obstructive pulmonary disease",
            expected_rei=None,
            expected_role="Total COPD denominator for attributable share (Stage 1)",
            require_all_ages_5yr=False,
            require_sex_split=True,
        )
    )

    # 3. GBD HAP All Linked Causes
    gbd_all_path = raw_dir / "IHME-GBD_2023_DATA-89044603-1.zip"
    results.append(
        validate_gbd_zip(
            gbd_all_path,
            expected_cause="All causes",
            expected_rei="Household air pollution from solid fuels",
            expected_role="Context: COPD share of total HAP burden (Stage 1)",
            require_all_ages_5yr=False,
            require_sex_split=True,
            require_level3_causes=True,
        )
    )

    # 4. WHO SDG 7.1.2 CSV
    who_csv_path = raw_dir / "data (1).csv"
    results.append(validate_who_data(who_csv_path))

    # 5. UN WPP 2024
    results.extend(validate_wpp_files(raw_dir))

    # 6. World Bank JSONs
    for jf in ["EG.CFT.ACCS.json", "EG.CFT.ACCS.RU.json", "EG.CFT.ACCS.UR.json"]:
        p = raw_dir / jf
        if p.exists():
            results.append(validate_worldbank_json(p))

    # 7. Codebook check
    codebook_p = raw_dir / "IHME_GBD_2021_AIR_POLLUTION_1990_2021_CODEBOOK_Y2024M06D06.csv"
    if codebook_p.exists():
        results.append(
            DatasetValidationResult(
                name=codebook_p.name,
                target_role="GBD 2021 Air Pollution Codebook (Unmatched auxiliary reference)",
                filepath=str(codebook_p),
                status="NOT TESTED",
                file_exists=True,
                size_bytes=codebook_p.stat().st_size,
                sha256=compute_sha256(codebook_p),
                row_count=771,
                warnings=["Auxiliary GBD 2021 codebook; not used in active pipeline."],
            )
        )

    # 8. Manifest check
    manifest_p = raw_dir / "manifest.json"
    manifest_res = DatasetValidationResult(
        name="manifest.json",
        target_role="Raw Data Manifest & Provenance Registry (AGENTS.md §5 / DATA_SPEC.md §1)",
        filepath=str(manifest_p),
        status="FAIL" if not manifest_p.exists() else "PASS",
        file_exists=manifest_p.exists(),
    )
    if not manifest_p.exists():
        manifest_res.critical_errors.append(
            "Missing data/raw/manifest.json. Required by AGENTS.md §5 and DATA_SPEC.md §1 & §4."
        )
    else:
        manifest_res.size_bytes = manifest_p.stat().st_size
        manifest_res.sha256 = compute_sha256(manifest_p)
    results.append(manifest_res)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AEGIS-GBD raw datasets.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument(
        "--report-only", action="store_true", help="Always exit 0, just print report."
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    raw_dir = Path("data/raw")
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            raw_dir = Path(cfg.get("paths", {}).get("raw_dir", "data/raw"))

    print(f"Auditing raw datasets in: {raw_dir}")
    results = run_full_validation(raw_dir)

    print("\n" + "=" * 80)
    print("DATASET VALIDATION SUMMARY TABLE")
    print("=" * 80)
    summary_rows = []
    for r in results:
        summary_rows.append(
            {
                "Dataset": r.name,
                "Status": r.status,
                "Size (KB)": f"{r.size_bytes / 1024:.1f}",
                "Rows": r.row_count,
                "Errors": len(r.critical_errors),
                "Warnings": len(r.warnings),
            }
        )
    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))

    has_failures = any(r.status == "FAIL" for r in results)
    print("\n" + "=" * 80)
    for r in results:
        print(f"\n--- {r.name} [{r.status}] ---")
        print(f"  Target Role: {r.target_role}")
        print(f"  SHA-256: {r.sha256}")
        if r.critical_errors:
            print("  CRITICAL ERRORS:")
            for err in r.critical_errors:
                print(f"    - [FAIL] {err}")
        if r.warnings:
            print("  WARNINGS / CONCERNS:")
            for warn in r.warnings:
                print(f"    - [WARN] {warn}")

    print("\n" + "=" * 80)
    if has_failures and not args.report_only:
        print("OVERALL RESULT: FAILED (Critical dataset gaps detected)")
        sys.exit(1)
    else:
        print("OVERALL RESULT: COMPLETED")


if __name__ == "__main__":
    main()
