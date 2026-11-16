"""Production validation runner for AEGIS-GBD raw datasets.

Verifies raw data integrity against data/raw/manifest.json and asserts schema,
completeness, and invariant bounds required by docs/DATA_SPEC.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PANEL_COUNTRIES: frozenset[str] = frozenset({"India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"})
PANEL_ISO3: frozenset[str] = frozenset({"IND", "NPL", "PAK", "BGD", "BTN"})


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash in 8MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(raw_dir: Path) -> dict[str, Any]:
    """Verify that manifest.json exists and all registered files match SHA-256."""
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files", {})
    verified_files = []
    for rel_path, meta in files.items():
        file_path = raw_dir / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"File listed in manifest missing on disk: {file_path}")
        actual_sha = compute_sha256(file_path)
        expected_sha = meta.get("sha256")
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"
            )
        verified_files.append(rel_path)

    return {"verified_files": len(verified_files), "status": "OK"}


def validate_gbd_data(raw_dir: Path) -> dict[str, Any]:
    """Validate GBD 2023 primary HAP-COPD dataset."""
    hap_zip = raw_dir / "IHME-GBD_2023_DATA-010d0c57-1.zip"
    if not hap_zip.exists():
        raise FileNotFoundError(f"Missing GBD HAP-COPD zip: {hap_zip}")

    with zipfile.ZipFile(hap_zip) as zf:
        cnames = [n for n in zf.namelist() if n.endswith(".csv")]
        with zf.open(cnames[0]) as f:
            df = pd.read_csv(f)

    # Required columns
    req_cols = {"location_name", "year", "measure_name", "metric_name", "val", "lower", "upper"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"GBD dataset missing required columns: {req_cols - set(df.columns)}")

    # Countries
    found_countries = set(df["location_name"].unique())
    if not PANEL_COUNTRIES.issubset(found_countries):
        missing = PANEL_COUNTRIES - found_countries
        raise ValueError(f"GBD dataset missing panel countries: {missing}")

    # Years
    years = set(df["year"].unique())
    expected_years = set(range(1990, 2024))
    if not expected_years.issubset(years):
        raise ValueError(f"GBD dataset missing historical years: {expected_years - years}")

    # Invariants: 0 <= lower <= val <= upper
    num_val = pd.to_numeric(df["val"], errors="coerce")
    num_low = pd.to_numeric(df["lower"], errors="coerce")
    num_upp = pd.to_numeric(df["upper"], errors="coerce")
    if (num_val < 0).any() or (num_low > num_val).any() or (num_val > num_upp).any():
        raise ValueError("GBD uncertainty bounds invariant violation (0 <= lower <= val <= upper)")

    return {"rows": len(df), "years": len(years), "status": "OK"}


def validate_who_data(raw_dir: Path) -> dict[str, Any]:
    """Validate WHO Clean Cooking dataset (data (1).csv)."""
    who_file = raw_dir / "data (1).csv"
    if not who_file.exists():
        raise FileNotFoundError(f"Missing WHO clean cooking dataset: {who_file}")

    df = pd.read_csv(who_file, low_memory=False)
    sa = df[df["SpatialDimValueCode"].isin(PANEL_ISO3)]
    if len(sa) == 0:
        raise ValueError("No South Asian panel records found in WHO clean cooking dataset")

    # Check residence and fuels
    residences = set(sa["Dim1"].dropna().unique())
    if not {"Total", "Urban", "Rural"}.issubset(residences):
        raise ValueError(f"Missing residence breakdown in WHO data: {residences}")

    return {"sa_rows": len(sa), "status": "OK"}


def run_validation(config_path: str = "configs/default.yaml") -> bool:
    """Run full validation suite on data/raw."""
    cfg_p = Path(config_path)
    raw_dir = Path("data/raw")
    if cfg_p.exists():
        with open(cfg_p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            raw_dir = Path(cfg.get("paths", {}).get("raw_dir", "data/raw"))

    print(f"=== AEGIS-GBD Data Validation: {raw_dir} ===")
    m_res = validate_manifest(raw_dir)
    print(f"  [PASS] Manifest verified: {m_res['verified_files']} raw files matched SHA-256.")

    g_res = validate_gbd_data(raw_dir)
    print(
        f"  [PASS] GBD 2023 HAP-COPD verified: {g_res['rows']} rows, {g_res['years']} annual years."
    )

    w_res = validate_who_data(raw_dir)
    print(f"  [PASS] WHO SDG 7.1.2 Clean Cooking verified: {w_res['sa_rows']} SA rows.")

    print("=== All Raw Validation Checks Passed Successfully ===")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS-GBD Raw Data Validation Runner")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to default.yaml")
    args = parser.parse_args()

    try:
        run_validation(args.config)
        sys.exit(0)
    except Exception as e:
        print(f"Validation FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
