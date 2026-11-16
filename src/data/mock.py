"""Deterministic synthetic mock fixture generator for Stage 3 and Planning contracts.

Generates clearly flagged synthetic fixtures (``is_mock=True``, ``MOCK_V1``) matching
``docs/INTERFACES.md`` Contracts 1-4 so optimisation and planning can be developed and
tested before real GBD/WPP/WHO data ingestion and Stage 2 forecasting finish.

All numbers generated here are synthetic placeholders — DO NOT CITE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import (
    load_config,
    validate_baseline_unit_state,
    validate_forecasts_stage3,
    validate_intervention_params_json,
    validate_surrogate_json,
)

MOCK_DATA_VERSION: str = "MOCK_V1"
MOCK_MODEL_NAME: str = "mock_generator"
MOCK_GENERATED_AT: str = "2026-01-01T00:00:00Z"
MOCK_WARNING: str = (
    "SYNTHETIC MOCK DATA — PLACEHOLDER VALUES FOR PIPELINE TESTING ONLY; DO NOT CITE."
)

# Synthetic order-of-magnitude unit baselines (strictly non-empirical placeholders).
_SYNTHETIC_UNIT_PROFILES: dict[str, dict[str, Any]] = {
    "IND_urban": {
        "country": "India",
        "residence": "urban",
        "pop_2025": 5.0e8,
        "pop_annual_growth": 0.012,
        "female_share": 0.485,
        "cf_2025": 0.78,
        "cf_2040_bau": 0.88,
        "lpg_infra_ready_init": True,
        "grid_ready_init": True,
        "daly_asr_2025_female": 240.0,
        "daly_asr_2025_male": 220.0,
        "death_asr_2025_female": 11.0,
        "death_asr_2025_male": 10.2,
        "annual_decline": 0.018,
        "slope_clean_fuel": -1.30,
        "sex_ratio_f_to_m": 1.10,
        "uncertainty_sd": 0.09,
        "cost_lpg": 320.0,
        "cost_biomass": 120.0,
        "cost_electric": 410.0,
        "cost_infra": 40.0,
        "cov_ceiling": 0.98,
    },
    "IND_rural": {
        "country": "India",
        "residence": "rural",
        "pop_2025": 9.0e8,
        "pop_annual_growth": 0.004,
        "female_share": 0.488,
        "cf_2025": 0.42,
        "cf_2040_bau": 0.58,
        "lpg_infra_ready_init": False,
        "grid_ready_init": False,
        "daly_asr_2025_female": 620.0,
        "daly_asr_2025_male": 510.0,
        "death_asr_2025_female": 29.0,
        "death_asr_2025_male": 24.5,
        "annual_decline": 0.014,
        "slope_clean_fuel": -1.45,
        "sex_ratio_f_to_m": 1.15,
        "uncertainty_sd": 0.11,
        "cost_lpg": 620.0,
        "cost_biomass": 210.0,
        "cost_electric": 780.0,
        "cost_infra": 95.0,
        "cov_ceiling": 0.95,
    },
    "NPL_urban": {
        "country": "Nepal",
        "residence": "urban",
        "pop_2025": 7.0e6,
        "pop_annual_growth": 0.018,
        "female_share": 0.505,
        "cf_2025": 0.64,
        "cf_2040_bau": 0.76,
        "lpg_infra_ready_init": True,
        "grid_ready_init": True,
        "daly_asr_2025_female": 310.0,
        "daly_asr_2025_male": 270.0,
        "death_asr_2025_female": 14.5,
        "death_asr_2025_male": 12.8,
        "annual_decline": 0.016,
        "slope_clean_fuel": -1.25,
        "sex_ratio_f_to_m": 1.10,
        "uncertainty_sd": 0.09,
        "cost_lpg": 18.0,
        "cost_biomass": 7.0,
        "cost_electric": 22.0,
        "cost_infra": 5.0,
        "cov_ceiling": 0.98,
    },
    "NPL_rural": {
        "country": "Nepal",
        "residence": "rural",
        "pop_2025": 2.3e7,
        "pop_annual_growth": 0.006,
        "female_share": 0.515,
        "cf_2025": 0.24,
        "cf_2040_bau": 0.40,
        "lpg_infra_ready_init": False,
        "grid_ready_init": False,
        "daly_asr_2025_female": 740.0,
        "daly_asr_2025_male": 590.0,
        "death_asr_2025_female": 34.0,
        "death_asr_2025_male": 27.5,
        "annual_decline": 0.012,
        "slope_clean_fuel": -1.50,
        "sex_ratio_f_to_m": 1.18,
        "uncertainty_sd": 0.12,
        "cost_lpg": 48.0,
        "cost_biomass": 18.0,
        "cost_electric": 60.0,
        "cost_infra": 16.0,
        "cov_ceiling": 0.95,
    },
}

_SCENARIO_TARGETS: dict[str, float | None] = {
    "bau": None,
    "cf_25": 0.25,
    "cf_50": 0.50,
    "cf_75": 0.75,
    "cf_100": 1.00,
}

_MILESTONE_YEARS: tuple[int, ...] = (2025, 2030, 2035, 2040)
_ALL_FORECAST_YEARS: tuple[int, ...] = tuple(range(2025, 2041))


def _write_meta_sidecar(csv_path: Path, seed: int, row_count: int) -> Path:
    meta_path = Path(f"{csv_path}.meta.json")
    payload = {
        "file": csv_path.name,
        "is_mock": True,
        "data_version": MOCK_DATA_VERSION,
        "generated_at": MOCK_GENERATED_AT,
        "seed": seed,
        "row_count": row_count,
        "warning": MOCK_WARNING,
    }
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return meta_path


def build_mock_forecasts(config: dict[str, Any], seed: int) -> pd.DataFrame:
    """Build deterministic synthetic Contract 1 ``forecasts_stage3`` DataFrame."""
    rng = np.random.default_rng(seed)
    units: list[str] = [str(u) for u in config["geography"]["allocation_units"]]
    rows: list[dict[str, Any]] = []

    for unit in units:
        prof = _SYNTHETIC_UNIT_PROFILES[unit]
        # Small deterministic synthetic jitter per unit so values are clearly synthetic.
        unit_jitter = float(rng.uniform(0.98, 1.02))

        for scenario, cf_target in _SCENARIO_TARGETS.items():
            years = _ALL_FORECAST_YEARS if scenario == "bau" else _MILESTONE_YEARS
            for year in years:
                h = year - 2025
                frac = h / 15.0
                pop_total = prof["pop_2025"] * ((1.0 + prof["pop_annual_growth"]) ** h)
                f_share = prof["female_share"]
                pop_female = pop_total * f_share
                pop_male = pop_total * (1.0 - f_share)

                cf_bau = prof["cf_2025"] + frac * (prof["cf_2040_bau"] - prof["cf_2025"])
                cf_val = (
                    cf_bau
                    if cf_target is None
                    else (prof["cf_2025"] + frac * (cf_target - prof["cf_2025"]))
                )
                delta_cf = cf_val - cf_bau

                mult_male = float(np.exp(prof["slope_clean_fuel"] * delta_cf))
                mult_female = float(
                    np.exp(prof["sex_ratio_f_to_m"] * prof["slope_clean_fuel"] * delta_cf)
                )
                decline = ((1.0 - prof["annual_decline"]) ** h) * unit_jitter

                # Widening 95% UI with forecast horizon h.
                rel_half_width = 0.08 + 0.012 * h

                for measure in ("DALYs", "Deaths"):
                    base_f = (
                        prof["daly_asr_2025_female"]
                        if measure == "DALYs"
                        else prof["death_asr_2025_female"]
                    )
                    base_m = (
                        prof["daly_asr_2025_male"]
                        if measure == "DALYs"
                        else prof["death_asr_2025_male"]
                    )
                    asr_f = base_f * decline * mult_female
                    asr_m = base_m * decline * mult_male
                    asr_both = f_share * asr_f + (1.0 - f_share) * asr_m

                    # Synthetic crude-vs-ASR scaling factor (1.0 for clean equivalence in mock).
                    num_f = asr_f * (pop_female / 100_000.0)
                    num_m = asr_m * (pop_male / 100_000.0)
                    num_both = num_f + num_m

                    sex_specs = (
                        ("female", pop_female, num_f, asr_f),
                        ("male", pop_male, num_m, asr_m),
                        ("both", pop_total, num_both, asr_both),
                    )
                    for sex, pop_s, num_s, asr_s in sex_specs:
                        for metric, val in (("Number", num_s), ("Rate_per_100k", asr_s)):
                            lower = max(0.0, val * (1.0 - rel_half_width))
                            upper = val * (1.0 + rel_half_width)
                            rows.append(
                                {
                                    "location": unit,
                                    "year": int(year),
                                    "sex": sex,
                                    "scenario": scenario,
                                    "measure": measure,
                                    "metric": metric,
                                    "val": round(float(val), 6),
                                    "lower": round(float(lower), 6),
                                    "upper": round(float(upper), 6),
                                    "population": round(float(pop_s), 2),
                                    "clean_fuel_share": round(float(cf_val), 6),
                                    "model_name": MOCK_MODEL_NAME,
                                    "is_mock": True,
                                    "data_version": MOCK_DATA_VERSION,
                                }
                            )

    df = pd.DataFrame(rows)
    return validate_forecasts_stage3(
        df, config, file_label="forecasts_stage3_mock", expect_mock=True
    )


def build_mock_baseline(
    forecasts_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build Contract 2 ``baseline_unit_state`` consistent with ``forecasts_df``."""
    units: list[str] = [str(u) for u in config["geography"]["allocation_units"]]
    rows: list[dict[str, Any]] = []

    def _lookup_2040(unit: str, sex: str, metric: str) -> float:
        mask = (
            (forecasts_df["location"] == unit)
            & (forecasts_df["year"] == 2040)
            & (forecasts_df["scenario"] == "bau")
            & (forecasts_df["measure"] == "DALYs")
            & (forecasts_df["sex"] == sex)
            & (forecasts_df["metric"] == metric)
        )
        return float(forecasts_df.loc[mask, "val"].iloc[0])

    for unit in units:
        prof = _SYNTHETIC_UNIT_PROFILES[unit]
        rows.append(
            {
                "location": unit,
                "country": prof["country"],
                "residence": prof["residence"],
                "baseline_year": 2025,
                "population_total": round(float(prof["pop_2025"]), 2),
                "population_female_share": float(prof["female_share"]),
                "clean_fuel_share_init": float(prof["cf_2025"]),
                "lpg_infra_ready_init": bool(prof["lpg_infra_ready_init"]),
                "grid_ready_init": bool(prof["grid_ready_init"]),
                "bau_dalys_2040_female": _lookup_2040(unit, "female", "Number"),
                "bau_dalys_2040_male": _lookup_2040(unit, "male", "Number"),
                "bau_asr_2040_female": _lookup_2040(unit, "female", "Rate_per_100k"),
                "bau_asr_2040_male": _lookup_2040(unit, "male", "Rate_per_100k"),
                "is_mock": True,
                "data_version": MOCK_DATA_VERSION,
            }
        )

    df = pd.DataFrame(rows)
    return validate_baseline_unit_state(
        df, config, file_label="baseline_unit_state_mock", expect_mock=True
    )


def build_mock_surrogate(
    forecasts_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build Contract 3 ``exposure_response_surrogate`` JSON aligned with 2040 BAU."""
    units: list[str] = [str(u) for u in config["geography"]["allocation_units"]]
    unit_payload: dict[str, dict[str, float]] = {}

    for unit in units:
        prof = _SYNTHETIC_UNIT_PROFILES[unit]
        mask = (
            (forecasts_df["location"] == unit)
            & (forecasts_df["year"] == 2040)
            & (forecasts_df["scenario"] == "bau")
        )
        cf_bau_2040 = float(forecasts_df.loc[mask, "clean_fuel_share"].iloc[0])
        unit_payload[unit] = {
            "slope_clean_fuel": float(prof["slope_clean_fuel"]),
            "baseline_clean_fuel": round(cf_bau_2040, 6),
            "sex_elasticity_ratio_f_to_m": float(prof["sex_ratio_f_to_m"]),
            "uncertainty_sd": float(prof["uncertainty_sd"]),
        }

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "is_mock": True,
        "data_version": MOCK_DATA_VERSION,
        "functional_form": "log_linear",
        "generated_at": MOCK_GENERATED_AT,
        "source_model": MOCK_MODEL_NAME,
        "warning": MOCK_WARNING,
        "units": unit_payload,
    }
    return validate_surrogate_json(
        payload, config, file_label="exposure_response_surrogate_mock", expect_mock=True
    )


def build_mock_intervention_params(config: dict[str, Any]) -> dict[str, Any]:
    """Build Contract 4 ``intervention_params`` mock JSON with placeholder values."""
    units: list[str] = [str(u) for u in config["geography"]["allocation_units"]]
    periods: list[str] = [str(p["id"]) for p in config["time"]["decision_periods"]]

    coverage_ceiling: dict[str, float] = {}
    effectiveness: dict[str, dict[str, float]] = {}
    unit_costs: dict[str, dict[str, float]] = {}
    max_adoption: dict[str, dict[str, dict[str, float]]] = {}

    for unit in units:
        prof = _SYNTHETIC_UNIT_PROFILES[unit]
        coverage_ceiling[unit] = float(prof["cov_ceiling"])
        effectiveness[unit] = {
            "lpg_subsidy": 1.0,
            "improved_biomass": 0.45,  # synthetic placeholder (ASM-10)
            "electric_induction": 1.0,
            "infra_lpg_grid": 0.0,
        }
        unit_costs[unit] = {
            "lpg_subsidy": float(prof["cost_lpg"]),
            "improved_biomass": float(prof["cost_biomass"]),
            "electric_induction": float(prof["cost_electric"]),
            "infra_lpg_grid": float(prof["cost_infra"]),
        }
        max_adoption[unit] = {
            "lpg_subsidy": {p: 0.20 for p in periods},
            "improved_biomass": {p: 0.25 for p in periods},
            "electric_induction": {p: 0.15 for p in periods},
            "infra_lpg_grid": {p: 1.0 for p in periods},
        }

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "is_mock": True,
        "data_version": MOCK_DATA_VERSION,
        "generated_at": MOCK_GENERATED_AT,
        "note": MOCK_WARNING,
        "total_budget_usd_millions": 500.0,  # synthetic placeholder (ASM-13)
        "infra_build_threshold": 0.5,
        "apply_cost_discounting": False,
        "discount_factor": float(config["planning"]["mdp"]["discount_factor"]),
        "period_order": periods,
        "coverage_ceiling": coverage_ceiling,
        "effectiveness": effectiveness,
        "unit_costs_usd_millions": unit_costs,
        "max_adoption_per_period": max_adoption,
    }
    return validate_intervention_params_json(
        payload,
        config,
        file_label="intervention_params_mock",
        expect_mock=True,
        require_populated=True,
    )


def generate_mock_fixtures(
    config_path: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate and validate all mock fixtures configured in ``config_path``."""
    config = load_config(config_path)
    seed = int(config["seeds"]["global"])
    interfaces = config["interfaces"]
    root = Path(base_dir) if base_dir is not None else Path(".")

    def _out_path(key: str) -> Path:
        raw = Path(interfaces[key])
        p = raw if raw.is_absolute() else root / raw
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    forecasts_path = _out_path("forecast_mock_file")
    baseline_path = _out_path("baseline_state_mock_file")
    surrogate_path = _out_path("surrogate_mock_file")
    params_path = _out_path("intervention_params_mock_file")

    forecasts_df = build_mock_forecasts(config, seed=seed)
    baseline_df = build_mock_baseline(forecasts_df, config)
    surrogate_payload = build_mock_surrogate(forecasts_df, config)
    params_payload = build_mock_intervention_params(config)

    forecasts_df.to_csv(forecasts_path, index=False)
    forecasts_meta = _write_meta_sidecar(forecasts_path, seed=seed, row_count=len(forecasts_df))

    baseline_df.to_csv(baseline_path, index=False)
    baseline_meta = _write_meta_sidecar(baseline_path, seed=seed, row_count=len(baseline_df))

    surrogate_path.write_text(json.dumps(surrogate_payload, indent=2) + "\n", encoding="utf-8")
    params_path.write_text(json.dumps(params_payload, indent=2) + "\n", encoding="utf-8")

    return {
        "forecasts_stage3_mock": forecasts_path,
        "forecasts_stage3_mock_meta": forecasts_meta,
        "baseline_unit_state_mock": baseline_path,
        "baseline_unit_state_mock_meta": baseline_meta,
        "exposure_response_surrogate_mock": surrogate_path,
        "intervention_params_mock": params_path,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``python -m src.data.mock --config configs/default.yaml``."""
    parser = argparse.ArgumentParser(description="Generate synthetic Stage 3 mock fixtures.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    args = parser.parse_args(argv)
    written = generate_mock_fixtures(args.config)
    for name, path in written.items():
        print(f"[MOCK] {name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
