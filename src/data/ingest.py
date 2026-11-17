"""Data ingestion and urban/rural apportionment pipeline for AEGIS-GBD.

Reads verified raw archives from data/raw/, harmonises historical time-series
(1990-2023) across South Asia panel countries and 4 allocation units (IND_urban,
IND_rural, NPL_urban, NPL_rural), applies empirical sex splits (ASM-04b),
performs population & exposure-weighted urban/rural apportionment (ASM-01),
and saves cleaned tables to data/interim/.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PANEL_COUNTRIES: list[str] = ["India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"]
COUNTRY_TO_ISO3: dict[str, str] = {
    "India": "IND",
    "Nepal": "NPL",
    "Pakistan": "PAK",
    "Bangladesh": "BGD",
    "Bhutan": "BTN",
}
ISO3_TO_COUNTRY: dict[str, str] = {v: k for k, v in COUNTRY_TO_ISO3.items()}

# Empirical sex fractions from WHO GHE 2021 (data (2).csv DALYs, data (3).csv Deaths)
# Logged in docs/ASSUMPTIONS.md (ASM-04b)
WHO_GHE_SEX_SPLITS: dict[str, dict[str, dict[str, float]]] = {
    "DALYs": {
        "India": {"female": 0.5346, "male": 0.4654},
        "Nepal": {"female": 0.5834, "male": 0.4166},
        "Pakistan": {"female": 0.4209, "male": 0.5791},
        "Bangladesh": {"female": 0.5080, "male": 0.4920},
        "Bhutan": {"female": 0.5426, "male": 0.4574},
    },
    "Deaths": {
        "India": {"female": 0.5157, "male": 0.4843},
        "Nepal": {"female": 0.5401, "male": 0.4599},
        "Pakistan": {"female": 0.4844, "male": 0.5156},
        "Bangladesh": {"female": 0.4795, "male": 0.5205},
        "Bhutan": {"female": 0.4915, "male": 0.5085},
    },
}

# Urban population share trajectories (UN WUP / World Bank historical baseline)
# Linear interpolation between 1990 and 2023:
# India: 25.5% in 1990 -> 36.4% in 2023
# Nepal: 8.9% in 1990 -> 21.9% in 2023
URBAN_SHARE_BOUNDS: dict[str, tuple[float, float]] = {
    "India": (0.255, 0.364),
    "Nepal": (0.089, 0.219),
}


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load project YAML configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_gbd_zip(zip_path: Path) -> pd.DataFrame:
    """Read primary CSV file from an IHME GBD Results zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV found inside {zip_path}")
        with zf.open(csv_names[0]) as f:
            return pd.read_csv(f)


def ingest_wpp_population(raw_dir: Path, interim_dir: Path) -> pd.DataFrame:
    """Extract and aggregate UN WPP 2024 population series (1990-2023) for South Asia."""
    cache_path = interim_dir / "wpp_population.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    wpp_base = raw_dir / "WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv"
    wpp_file = wpp_base / wpp_base.name if wpp_base.is_dir() else wpp_base
    if not wpp_file.exists():
        raise FileNotFoundError(f"WPP population file not found: {wpp_file}")

    target_iso3 = set(COUNTRY_TO_ISO3.values())
    chunks: list[pd.DataFrame] = []

    # Read in chunks to keep memory footprint minimal (< 200MB)
    for chunk in pd.read_csv(
        wpp_file,
        chunksize=100_000,
        usecols=["ISO3_code", "Time", "PopMale", "PopFemale", "PopTotal"],
        low_memory=False,
    ):
        filtered = chunk[
            (chunk["ISO3_code"].isin(target_iso3))
            & (chunk["Time"] >= 1990)
            & (chunk["Time"] <= 2023)
        ]
        if not filtered.empty:
            chunks.append(filtered)

    df_raw = pd.concat(chunks, ignore_index=True)
    # WPP single-age rows need aggregation to national annual totals
    # Convert from thousands of persons to individual persons
    agg = (
        df_raw.groupby(["ISO3_code", "Time"])[["PopMale", "PopFemale", "PopTotal"]]
        .sum()
        .reset_index()
    )
    agg["PopMale"] = agg["PopMale"] * 1000.0
    agg["PopFemale"] = agg["PopFemale"] * 1000.0
    agg["PopTotal"] = agg["PopTotal"] * 1000.0
    agg.rename(
        columns={
            "ISO3_code": "iso3",
            "Time": "year",
            "PopMale": "pop_male",
            "PopFemale": "pop_female",
            "PopTotal": "pop_total",
        },
        inplace=True,
    )
    agg["country"] = agg["iso3"].map(ISO3_TO_COUNTRY)
    agg.to_parquet(cache_path, index=False)
    return agg


def ingest_clean_cooking(raw_dir: Path, interim_dir: Path) -> pd.DataFrame:
    """Ingest World Bank clean cooking access series (1990-2023) for national, urban, rural."""
    cache_path = interim_dir / "clean_cooking.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    wb_files = {
        "national": raw_dir / "EG.CFT.ACCS.json",
        "rural": raw_dir / "EG.CFT.ACCS.RU.json",
        "urban": raw_dir / "EG.CFT.ACCS.UR.json",
    }

    records: list[dict[str, Any]] = []
    target_iso3 = set(COUNTRY_TO_ISO3.values())

    for domain, fpath in wb_files.items():
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
            entries = data[1]
            for e in entries:
                iso3 = e.get("countryiso3code")
                if iso3 in target_iso3:
                    try:
                        yr = int(e["date"])
                        val = float(e["value"]) if e["value"] is not None else np.nan
                        records.append(
                            {
                                "iso3": iso3,
                                "country": ISO3_TO_COUNTRY[iso3],
                                "year": yr,
                                "domain": domain,
                                "clean_fuel_percent": val,
                            }
                        )
                    except (ValueError, TypeError):
                        continue

    df_wb = pd.DataFrame(records)
    # Pivot so columns are national, rural, urban
    piv = df_wb.pivot_table(
        index=["country", "iso3", "year"],
        columns="domain",
        values="clean_fuel_percent",
    ).reset_index()

    # Filter to 1990-2023
    piv = piv[(piv["year"] >= 1990) & (piv["year"] <= 2023)].sort_values(["country", "year"])

    # Interpolate backwards for 1990-1999 where survey series begins ~2000
    # World Bank series typically covers 2000-2023; backfill smoothly
    full_idx = (
        pd.MultiIndex.from_product(
            [PANEL_COUNTRIES, range(1990, 2024)],
            names=["country", "year"],
        )
        .to_frame()
        .reset_index(drop=True)
    )
    full_idx["iso3"] = full_idx["country"].map(COUNTRY_TO_ISO3)

    merged = pd.merge(full_idx, piv, on=["country", "iso3", "year"], how="left")
    merged = merged.sort_values(["country", "year"])

    for col in ["national", "rural", "urban"]:
        # Backfill then forward-fill per country
        merged[col] = merged.groupby("country")[col].transform(lambda s: s.bfill().ffill())

    # Convert to fraction [0, 1]
    merged["clean_fuel_share_national"] = (merged["national"] / 100.0).clip(0.0, 1.0)
    merged["clean_fuel_share_rural"] = (merged["rural"] / 100.0).clip(0.0, 1.0)
    merged["clean_fuel_share_urban"] = (merged["urban"] / 100.0).clip(0.0, 1.0)

    merged.to_parquet(cache_path, index=False)
    return merged


def ingest_gbd_series(
    raw_dir: Path,
    interim_dir: Path,
    wpp_pop: pd.DataFrame,
    clean_cooking: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ingest primary HAP-COPD, Total COPD, and All-cause HAP GBD 2023 tables."""
    hap_zip = raw_dir / "IHME-GBD_2023_DATA-010d0c57-1.zip"
    tot_zip = raw_dir / "IHME-GBD_2023_DATA-7546b8be-1.zip"
    all_zip = raw_dir / "IHME-GBD_2023_DATA-89044603-1.zip"

    df_hap_raw = _read_gbd_zip(hap_zip)
    df_tot_raw = _read_gbd_zip(tot_zip)
    df_all_raw = _read_gbd_zip(all_zip)

    # Standardize measure names
    def _clean_measure(val: str) -> str:
        if "DALY" in val:
            return "DALYs"
        if "Death" in val:
            return "Deaths"
        return val

    for df in [df_hap_raw, df_tot_raw, df_all_raw]:
        df["measure_name"] = df["measure_name"].map(_clean_measure)
        # Filter to 5 panel countries and 1990-2023
        df.drop(
            df[
                (~df["location_name"].isin(PANEL_COUNTRIES))
                | (df["year"] < 1990)
                | (df["year"] > 2023)
            ].index,
            inplace=True,
        )

    # Standardize column naming
    std_cols = [
        "location_name",
        "year",
        "measure_name",
        "metric_name",
        "age_name",
        "val",
        "lower",
        "upper",
    ]
    df_hap = df_hap_raw[std_cols].copy()
    df_tot = df_tot_raw[std_cols].copy()
    df_all = df_all_raw[std_cols].copy()

    # Expand HAP-COPD into sex-stratified (Female, Male, Both)
    # Using WHO GHE empirical splits (ASM-04b)
    expanded_rows: list[dict[str, Any]] = []
    pop_lookup = wpp_pop.set_index(["country", "year"])

    for _, row in df_hap.iterrows():
        country = row["location_name"]
        year = int(row["year"])
        measure = row["measure_name"]
        metric = row["metric_name"]
        age = row["age_name"]
        val = float(row["val"])
        low = float(row["lower"])
        upp = float(row["upper"])

        # Base row: 'both' sexes
        base_dict = {
            "location": country,
            "year": year,
            "sex": "both",
            "measure": measure,
            "metric": "Rate_per_100k" if metric == "Rate" else metric,
            "age": age,
            "val": val,
            "lower": low,
            "upper": upp,
        }
        expanded_rows.append(base_dict)

        # Sex split for female and male
        splits = WHO_GHE_SEX_SPLITS.get(measure, {}).get(country, {"female": 0.5, "male": 0.5})
        f_share = splits["female"]
        m_share = splits["male"]

        # Look up population
        pop_row = pop_lookup.loc[(country, year)] if (country, year) in pop_lookup.index else None
        p_tot = float(pop_row["pop_total"]) if pop_row is not None else 1.0
        p_fem = float(pop_row["pop_female"]) if pop_row is not None else p_tot * 0.49
        p_mal = float(pop_row["pop_male"]) if pop_row is not None else p_tot * 0.51

        if metric == "Number":
            # Direct count split
            expanded_rows.append(
                {
                    "location": country,
                    "year": year,
                    "sex": "female",
                    "measure": measure,
                    "metric": "Number",
                    "age": age,
                    "val": val * f_share,
                    "lower": low * f_share,
                    "upper": upp * f_share,
                }
            )
            expanded_rows.append(
                {
                    "location": country,
                    "year": year,
                    "sex": "male",
                    "measure": measure,
                    "metric": "Number",
                    "age": age,
                    "val": val * m_share,
                    "lower": low * m_share,
                    "upper": upp * m_share,
                }
            )
        elif metric == "Rate":
            # Rate per 100k: rate_F = (val_both_count * f_share / pop_fem) * 100k
            # If age is Age-standardized, standard scaling preserves sex ratio
            f_rate_factor = (f_share / (p_fem / p_tot)) if p_tot > 0 else 1.0
            m_rate_factor = (m_share / (p_mal / p_tot)) if p_tot > 0 else 1.0

            expanded_rows.append(
                {
                    "location": country,
                    "year": year,
                    "sex": "female",
                    "measure": measure,
                    "metric": "Rate_per_100k",
                    "age": age,
                    "val": val * f_rate_factor,
                    "lower": low * f_rate_factor,
                    "upper": upp * f_rate_factor,
                }
            )
            expanded_rows.append(
                {
                    "location": country,
                    "year": year,
                    "sex": "male",
                    "measure": measure,
                    "metric": "Rate_per_100k",
                    "age": age,
                    "val": val * m_rate_factor,
                    "lower": low * m_rate_factor,
                    "upper": upp * m_rate_factor,
                }
            )

    df_hap_expanded = pd.DataFrame(expanded_rows)

    # Urban/rural apportionment for India and Nepal (ASM-01)
    # w_k = (Pop_k * (1 - CleanFuel_k)) / sum(Pop_j * (1 - CleanFuel_j))
    apportioned_rows: list[dict[str, Any]] = []
    cc_lookup = clean_cooking.set_index(["country", "year"])

    for country, prefix in [("India", "IND"), ("Nepal", "NPL")]:
        u_min, u_max = URBAN_SHARE_BOUNDS[country]
        for year in range(1990, 2024):
            # Evolving urban population fraction
            alpha = (year - 1990) / 33.0
            u_pop_share = u_min + alpha * (u_max - u_min)
            r_pop_share = 1.0 - u_pop_share

            # Clean fuel shares
            cc_row = cc_lookup.loc[(country, year)]
            c_u = float(cc_row["clean_fuel_share_urban"])
            c_r = float(cc_row["clean_fuel_share_rural"])
            s_u = 1.0 - c_u
            s_r = 1.0 - c_r

            # Weight apportionment
            denom = (u_pop_share * s_u) + (r_pop_share * s_r)
            if denom <= 0:
                w_u, w_r = u_pop_share, r_pop_share
            else:
                w_u = (u_pop_share * s_u) / denom
                w_r = (r_pop_share * s_r) / denom

            # Filter country rows for this year
            c_sub = df_hap_expanded[
                (df_hap_expanded["location"] == country) & (df_hap_expanded["year"] == year)
            ]

            for _, r in c_sub.iterrows():
                sex = r["sex"]
                measure = r["measure"]
                metric = r["metric"]
                age = r["age"]
                v, lo, hi = r["val"], r["lower"], r["upper"]

                if metric == "Number":
                    # Urban unit
                    apportioned_rows.append(
                        {
                            "location": f"{prefix}_urban",
                            "year": year,
                            "sex": sex,
                            "measure": measure,
                            "metric": metric,
                            "age": age,
                            "val": v * w_u,
                            "lower": lo * w_u,
                            "upper": hi * w_u,
                        }
                    )
                    # Rural unit
                    apportioned_rows.append(
                        {
                            "location": f"{prefix}_rural",
                            "year": year,
                            "sex": sex,
                            "measure": measure,
                            "metric": metric,
                            "age": age,
                            "val": v * w_r,
                            "lower": lo * w_r,
                            "upper": hi * w_r,
                        }
                    )
                elif metric == "Rate_per_100k":
                    # Rate per 100k = (Burden_k / Pop_k) * 100k
                    # Burden_k = Count * w_k
                    # Pop_k = Pop * (u_pop_share or r_pop_share)
                    u_rate_mult = (w_u / u_pop_share) if u_pop_share > 0 else 1.0
                    r_rate_mult = (w_r / r_pop_share) if r_pop_share > 0 else 1.0

                    apportioned_rows.append(
                        {
                            "location": f"{prefix}_urban",
                            "year": year,
                            "sex": sex,
                            "measure": measure,
                            "metric": metric,
                            "age": age,
                            "val": v * u_rate_mult,
                            "lower": lo * u_rate_mult,
                            "upper": hi * u_rate_mult,
                        }
                    )
                    apportioned_rows.append(
                        {
                            "location": f"{prefix}_rural",
                            "year": year,
                            "sex": sex,
                            "measure": measure,
                            "metric": metric,
                            "age": age,
                            "val": v * r_rate_mult,
                            "lower": lo * r_rate_mult,
                            "upper": hi * r_rate_mult,
                        }
                    )

    df_apportioned = pd.DataFrame(apportioned_rows)
    df_full_hap = pd.concat([df_hap_expanded, df_apportioned], ignore_index=True)

    # Save interim files
    df_full_hap.to_parquet(interim_dir / "historical_burden.parquet", index=False)
    df_tot.to_parquet(interim_dir / "total_copd.parquet", index=False)
    df_all.to_parquet(interim_dir / "all_cause_hap.parquet", index=False)

    return df_full_hap, df_tot, df_all


def run_ingest(config_path: str | Path = "configs/default.yaml") -> dict[str, Path]:
    """Execute complete ingestion and apportionment pipeline."""
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    print("=== [AEGIS-GBD] Step 1: Ingestion & Apportionment ===")
    print(f"  Ingesting WPP population from {raw_dir}...")
    wpp_pop = ingest_wpp_population(raw_dir, interim_dir)
    print(f"  WPP population extracted ({len(wpp_pop)} country-years).")

    print(f"  Ingesting Clean Cooking access from {raw_dir}...")
    clean_cooking = ingest_clean_cooking(raw_dir, interim_dir)
    print(f"  Clean cooking timeseries processed ({len(clean_cooking)} rows).")

    print("  Ingesting GBD 2023 archives & apportioning urban/rural (ASM-01, ASM-04b)...")
    df_hap, df_tot, df_all = ingest_gbd_series(raw_dir, interim_dir, wpp_pop, clean_cooking)
    print(f"  HAP-COPD panel complete ({len(df_hap)} rows across 9 units).")
    print(f"  Total COPD complete ({len(df_tot)} rows).")
    print(f"  All-cause HAP complete ({len(df_all)} rows).")
    print("=== Ingestion Finished Successfully ===")

    return {
        "historical_burden": interim_dir / "historical_burden.parquet",
        "total_copd": interim_dir / "total_copd.parquet",
        "all_cause_hap": interim_dir / "all_cause_hap.parquet",
        "clean_cooking": interim_dir / "clean_cooking.parquet",
        "wpp_population": interim_dir / "wpp_population.parquet",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw data into data/interim/")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config")
    args = parser.parse_args()
    run_ingest(args.config)
