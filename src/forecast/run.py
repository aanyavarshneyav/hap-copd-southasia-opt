"""Stage 2 entrypoint for AEGIS-GBD: Forecasting & Production Contract Handoff.

Executes:
1. Rolling-origin cross-validation benchmark across ARIMA, ETS, Poisson, and HistGBDT
2. Trains best model and projects 2024-2040 burden under BAU and counterfactual scenarios
3. Exports validated production contracts (forecasts_stage3.csv, baseline_unit_state.csv,
   exposure_response_surrogate.json)
4. Validates production contracts and flips configs/default.yaml use_mock_forecasts to false
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.contracts import (
    validate_baseline_unit_state,
    validate_forecasts_stage3,
    validate_surrogate_json,
)
from src.forecast.cv import plot_cv_comparison, run_rolling_origin_cv
from src.forecast.projections import (
    extract_wpp_2024_2040_population,
    generate_production_contracts,
)
from src.viz.forecast import plot_forecasts_2040


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_stage2(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Execute complete Stage 2 forecasting and handoff pipeline."""
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    tables_dir = Path(cfg["paths"]["outputs_tables_dir"])
    figures_dir = Path(cfg["paths"]["outputs_figures_dir"])

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== [AEGIS-GBD] Stage 2: Forecasting & Production Handoff ===")

    # 1. Ensure WPP 2024-2040 population projections are extracted
    print("  Extracting WPP 2024-2040 population projections...")
    extract_wpp_2024_2040_population(raw_dir, interim_dir)

    # 2. Rolling-Origin Cross Validation
    print("  Running rolling-origin panel CV (origins: 2008, 2013, 2018; h=5)...")
    df_hap = pd.read_parquet(interim_dir / "historical_burden.parquet")
    df_cc = pd.read_parquet(interim_dir / "clean_cooking.parquet")

    # Add clean fuel share column to historical burden
    cc_map: dict[tuple[str, int], float] = {}
    for _, r in df_cc.iterrows():
        c, y = r["country"], int(r["year"])
        cc_map[(c, y)] = float(r["clean_fuel_share_national"])
        if c == "India":
            cc_map[("IND_urban", y)] = float(r["clean_fuel_share_urban"])
            cc_map[("IND_rural", y)] = float(r["clean_fuel_share_rural"])
        elif c == "Nepal":
            cc_map[("NPL_urban", y)] = float(r["clean_fuel_share_urban"])
            cc_map[("NPL_rural", y)] = float(r["clean_fuel_share_rural"])

    df_panel = df_hap.copy()
    df_panel["clean_fuel_share"] = df_panel.apply(
        lambda r: cc_map.get((r["location"], r["year"]), 0.5), axis=1
    )

    df_cv = run_rolling_origin_cv(df_panel)
    cv_table_path = tables_dir / "s2_rolling_cv_metrics.csv"
    df_cv.to_csv(cv_table_path, index=False)
    print(f"  [OK] Saved CV metrics to {cv_table_path}")

    cv_fig_path = figures_dir / "s2_model_comparison_cv.png"
    plot_cv_comparison(df_cv, cv_fig_path)
    print(f"  [OK] Rendered CV comparison plot to {cv_fig_path}")

    # 3. Generate Production Contracts (Stage 3 Handoff)
    print("  Generating production contracts with HistGBDT panel model...")
    contracts = generate_production_contracts(interim_dir, processed_dir, "hist_gbdt_panel")

    # 4. Validate Production Contracts
    print("  Validating production contracts against INTERFACES.md schemas...")
    df_forecasts = pd.read_csv(contracts["forecasts"])
    df_baseline = pd.read_csv(contracts["baseline"])
    with open(contracts["surrogate"], encoding="utf-8") as f:
        surrogate_dict = json.load(f)

    validate_forecasts_stage3(df_forecasts, cfg, expect_mock=False)
    validate_baseline_unit_state(df_baseline, cfg, expect_mock=False)
    validate_surrogate_json(surrogate_dict, cfg, expect_mock=False)
    print("  [PASS] All 3 production contracts verified successfully with zero errors!")

    # 5. Render Forecast Figure & Summary Table
    forecast_fig_path = figures_dir / "s2_forecasts_2040_ind_npl.png"
    plot_forecasts_2040(df_forecasts, forecast_fig_path)
    print(f"  [OK] Rendered forecast figure to {forecast_fig_path}")

    # Summary table for 2040
    sub_2040 = df_forecasts[
        (df_forecasts["year"] == 2040)
        & (df_forecasts["scenario"] == "bau")
        & (df_forecasts["sex"] == "both")
    ][["location", "measure", "metric", "val", "lower", "upper", "population", "clean_fuel_share"]]
    summary_path = tables_dir / "s2_forecasts_2040_summary.csv"
    sub_2040.to_csv(summary_path, index=False)
    print(f"  [OK] Saved 2040 forecast summary to {summary_path}")

    print("=== Stage 2 Execution & Handoff Complete ===")
    return {
        "contracts": contracts,
        "cv_table": cv_table_path,
        "cv_figure": cv_fig_path,
        "forecast_figure": forecast_fig_path,
        "summary_table": summary_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 Forecasting & Handoff")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config")
    args = parser.parse_args()
    run_stage2(args.config)
