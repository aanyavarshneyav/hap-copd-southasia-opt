"""Contract schemas, validators, and loaders for Stage 3 / Planning hand-off files.

Validates CSV and JSON contracts documented in ``docs/INTERFACES.md`` (Contracts 1-4)
and resolves mock vs. production file paths strictly from ``configs/default.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ALLOWED_SEXES: frozenset[str] = frozenset({"female", "male", "both"})
ALLOWED_SCENARIOS: frozenset[str] = frozenset({"bau", "cf_25", "cf_50", "cf_75", "cf_100"})
ALLOWED_MEASURES: frozenset[str] = frozenset({"DALYs", "Deaths"})
ALLOWED_METRICS: frozenset[str] = frozenset({"Number", "Rate_per_100k"})
ALLOWED_COUNTRIES: frozenset[str] = frozenset({"India", "Nepal"})
ALLOWED_RESIDENCES: frozenset[str] = frozenset({"urban", "rural"})
ALLOWED_FUNCTIONAL_FORMS: frozenset[str] = frozenset(
    {"log_linear", "log_linear_or_piecewise_paf", "piecewise_paf"}
)
REQUIRED_MILESTONE_YEARS: frozenset[int] = frozenset({2025, 2030, 2035, 2040})
FORECAST_YEAR_MIN: int = 2025
FORECAST_YEAR_MAX: int = 2040
BASELINE_YEAR_MIN: int = 2023
BASELINE_YEAR_MAX: int = 2025

FORECASTS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "location",
    "year",
    "sex",
    "scenario",
    "measure",
    "metric",
    "val",
    "lower",
    "upper",
    "population",
    "clean_fuel_share",
    "model_name",
    "is_mock",
    "data_version",
)

BASELINE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "location",
    "country",
    "residence",
    "baseline_year",
    "population_total",
    "population_female_share",
    "clean_fuel_share_init",
    "lpg_infra_ready_init",
    "grid_ready_init",
    "bau_dalys_2040_female",
    "bau_dalys_2040_male",
    "bau_asr_2040_female",
    "bau_asr_2040_male",
    "is_mock",
    "data_version",
)

SURROGATE_TOP_KEYS: tuple[str, ...] = (
    "schema_version",
    "is_mock",
    "data_version",
    "functional_form",
    "generated_at",
    "source_model",
    "units",
)

SURROGATE_UNIT_KEYS: tuple[str, ...] = (
    "slope_clean_fuel",
    "baseline_clean_fuel",
    "sex_elasticity_ratio_f_to_m",
    "uncertainty_sd",
)

INTERVENTION_PARAMS_TOP_KEYS: tuple[str, ...] = (
    "schema_version",
    "is_mock",
    "data_version",
    "generated_at",
    "total_budget_usd_millions",
    "infra_build_threshold",
    "apply_cost_discounting",
    "discount_factor",
    "period_order",
    "coverage_ceiling",
    "effectiveness",
    "unit_costs_usd_millions",
    "max_adoption_per_period",
)


class ContractValidationError(ValueError):
    """Raised when a contract file or payload violates ``docs/INTERFACES.md``."""


@dataclass(frozen=True)
class ContractPaths:
    """Resolved file paths for Contracts 1-4."""

    forecasts_stage3: Path
    baseline_unit_state: Path
    exposure_response_surrogate: Path
    intervention_params: Path
    use_mock: bool


@dataclass(frozen=True)
class Stage3Contracts:
    """Validated hand-off bundle consumed by Stage 3 optimisation and planning."""

    forecasts_stage3: pd.DataFrame
    baseline_unit_state: pd.DataFrame
    surrogate: dict[str, Any]
    intervention_params: dict[str, Any]
    is_mock: bool
    paths: ContractPaths


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and return YAML configuration from ``config_path``."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ContractValidationError(f"[{path}] Config must parse to a mapping/dict.")
    return data


def resolve_contract_paths(
    config: dict[str, Any],
    base_dir: str | Path | None = None,
) -> ContractPaths:
    """Resolve Contract 1-4 paths from config honoring ``interfaces.use_mock_forecasts``."""
    interfaces = config.get("interfaces")
    if not isinstance(interfaces, dict):
        raise ContractValidationError("[config] Missing 'interfaces' section.")

    use_mock = interfaces.get("use_mock_forecasts")
    if not isinstance(use_mock, bool):
        raise ContractValidationError("[config] 'interfaces.use_mock_forecasts' must be a boolean.")

    key_map = (
        ("forecast_mock_file", "forecast_contract_file"),
        ("baseline_state_mock_file", "baseline_state_file"),
        ("surrogate_mock_file", "surrogate_file"),
        ("intervention_params_mock_file", "intervention_params_file"),
    )
    resolved: list[Path] = []
    root = Path(base_dir) if base_dir is not None else Path(".")
    for mock_key, real_key in key_map:
        selected_key = mock_key if use_mock else real_key
        raw_path = interfaces.get(selected_key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ContractValidationError(
                f"[config] Missing or invalid path for 'interfaces.{selected_key}'."
            )
        p = Path(raw_path)
        resolved.append(p if p.is_absolute() else root / p)

    return ContractPaths(
        forecasts_stage3=resolved[0],
        baseline_unit_state=resolved[1],
        exposure_response_surrogate=resolved[2],
        intervention_params=resolved[3],
        use_mock=use_mock,
    )


def _format_rows(indices: pd.Index | list[Any], limit: int = 5) -> str:
    items = list(indices[:limit])
    suffix = "..." if len(indices) > limit else ""
    return f"{items}{suffix}"


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], file_label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ContractValidationError(
            f"[{file_label}] Missing required columns: {missing}. Found: {list(df.columns)}"
        )


def _require_non_empty(df: pd.DataFrame, file_label: str) -> None:
    if df.empty:
        raise ContractValidationError(f"[{file_label}] Table is empty (0 rows).")


def _require_no_nulls(df: pd.DataFrame, columns: tuple[str, ...], file_label: str) -> None:
    for col in columns:
        null_mask = df[col].isna()
        if null_mask.any():
            bad_rows = df.index[null_mask]
            raise ContractValidationError(
                f"[{file_label}] Column '{col}' contains nulls at row(s) {_format_rows(bad_rows)}."
            )


def _require_string_column(
    df: pd.DataFrame,
    col: str,
    file_label: str,
    allowed: frozenset[str] | set[str] | None = None,
) -> None:
    series = df[col]
    if not pd.api.types.is_string_dtype(series) and series.dtype != object:
        raise ContractValidationError(
            f"[{file_label}] Column '{col}' must have string dtype, got {series.dtype}."
        )
    non_str = ~series.map(lambda v: isinstance(v, str) and bool(v.strip()))
    if non_str.any():
        bad_rows = df.index[non_str]
        raise ContractValidationError(
            f"[{file_label}] Column '{col}' has non-string or blank values at row(s) "
            f"{_format_rows(bad_rows)}."
        )
    if allowed is not None:
        invalid_mask = ~series.isin(allowed)
        if invalid_mask.any():
            bad_rows = df.index[invalid_mask]
            bad_vals = sorted(set(series[invalid_mask].tolist()))
            raise ContractValidationError(
                f"[{file_label}] Column '{col}' has disallowed value(s) {bad_vals} "
                f"at row(s) {_format_rows(bad_rows)}. Allowed: {sorted(allowed)}."
            )


def _require_int_column(
    df: pd.DataFrame,
    col: str,
    file_label: str,
    min_val: int,
    max_val: int,
) -> None:
    series = df[col]
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_integer_dtype(series):
        raise ContractValidationError(
            f"[{file_label}] Column '{col}' must have integer dtype, got {series.dtype}."
        )
    out_of_range = (series < min_val) | (series > max_val)
    if out_of_range.any():
        bad_rows = df.index[out_of_range]
        raise ContractValidationError(
            f"[{file_label}] Column '{col}' out of range [{min_val}, {max_val}] "
            f"at row(s) {_format_rows(bad_rows)}."
        )


def _require_bool_column(
    df: pd.DataFrame,
    col: str,
    file_label: str,
) -> None:
    series = df[col]
    if not pd.api.types.is_bool_dtype(series):
        raise ContractValidationError(
            f"[{file_label}] Column '{col}' must have boolean dtype, got {series.dtype}."
        )


def _require_float_column(
    df: pd.DataFrame,
    col: str,
    file_label: str,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    strictly_positive: bool = False,
    strictly_below_max: bool = False,
) -> None:
    series = df[col]
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        raise ContractValidationError(
            f"[{file_label}] Column '{col}' must have numeric dtype, got {series.dtype}."
        )
    values = series.to_numpy(dtype=float)
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        bad_rows = df.index[non_finite]
        raise ContractValidationError(
            f"[{file_label}] Column '{col}' has non-finite values at row(s) "
            f"{_format_rows(bad_rows)}."
        )
    if min_val is not None:
        bad = values <= min_val if strictly_positive else values < min_val
        op = "<=" if strictly_positive else "<"
        if bad.any():
            bad_rows = df.index[bad]
            raise ContractValidationError(
                f"[{file_label}] Column '{col}' violates {col} {op} {min_val} invalidly "
                f"at row(s) {_format_rows(bad_rows)}."
            )
    if max_val is not None:
        bad = values >= max_val if strictly_below_max else values > max_val
        op = ">=" if strictly_below_max else ">"
        if bad.any():
            bad_rows = df.index[bad]
            raise ContractValidationError(
                f"[{file_label}] Column '{col}' exceeds max {max_val} "
                f"at row(s) {_format_rows(bad_rows)}."
            )


def _check_mock_column(
    df: pd.DataFrame,
    file_label: str,
    expect_mock: bool | None,
) -> bool:
    unique_flags = set(bool(v) for v in df["is_mock"].unique())
    if len(unique_flags) != 1:
        raise ContractValidationError(
            f"[{file_label}] Column 'is_mock' mixes true and false values in one file."
        )
    actual_mock = next(iter(unique_flags))
    if expect_mock is not None and actual_mock is not expect_mock:
        raise ContractValidationError(
            f"[{file_label}] Expected is_mock={expect_mock}, found is_mock={actual_mock}."
        )
    return actual_mock


def _allowed_units(config: dict[str, Any]) -> list[str]:
    units = config.get("geography", {}).get("allocation_units")
    if not isinstance(units, list) or not units:
        raise ContractValidationError("[config] 'geography.allocation_units' must be non-empty.")
    return [str(u) for u in units]


def _allowed_interventions(config: dict[str, Any]) -> list[str]:
    interventions = config.get("interventions")
    if not isinstance(interventions, list) or not interventions:
        raise ContractValidationError("[config] 'interventions' must be non-empty.")
    return [str(i) for i in interventions]


def _allowed_periods(config: dict[str, Any]) -> list[str]:
    periods = config.get("time", {}).get("decision_periods")
    if not isinstance(periods, list) or not periods:
        raise ContractValidationError("[config] 'time.decision_periods' must be non-empty.")
    return [str(p["id"]) for p in periods]


def validate_forecasts_stage3(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    file_label: str = "forecasts_stage3",
    expect_mock: bool | None = None,
) -> pd.DataFrame:
    """Validate Contract 1 (``forecasts_stage3``) DataFrame against ``docs/INTERFACES.md``."""
    _require_columns(df, FORECASTS_REQUIRED_COLUMNS, file_label)
    _require_non_empty(df, file_label)
    _require_no_nulls(df, FORECASTS_REQUIRED_COLUMNS, file_label)

    units = set(_allowed_units(config))
    _require_string_column(df, "location", file_label, allowed=units)
    _require_int_column(df, "year", file_label, FORECAST_YEAR_MIN, FORECAST_YEAR_MAX)
    _require_string_column(df, "sex", file_label, allowed=ALLOWED_SEXES)
    _require_string_column(df, "scenario", file_label, allowed=ALLOWED_SCENARIOS)
    _require_string_column(df, "measure", file_label, allowed=ALLOWED_MEASURES)
    _require_string_column(df, "metric", file_label, allowed=ALLOWED_METRICS)
    _require_string_column(df, "model_name", file_label)
    _require_string_column(df, "data_version", file_label)

    _require_float_column(df, "val", file_label, min_val=0.0)
    _require_float_column(df, "lower", file_label, min_val=0.0)
    _require_float_column(df, "upper", file_label, min_val=0.0)
    _require_float_column(df, "population", file_label, min_val=0.0, strictly_positive=True)
    _require_float_column(df, "clean_fuel_share", file_label, min_val=0.0, max_val=1.0)
    _require_bool_column(df, "is_mock", file_label)
    _check_mock_column(df, file_label, expect_mock)

    low_gt_val = df["lower"] > df["val"]
    if low_gt_val.any():
        bad_rows = df.index[low_gt_val]
        raise ContractValidationError(
            f"[{file_label}] Interval violation lower > val in columns ('lower', 'val') "
            f"at row(s) {_format_rows(bad_rows)}."
        )
    val_gt_up = df["val"] > df["upper"]
    if val_gt_up.any():
        bad_rows = df.index[val_gt_up]
        raise ContractValidationError(
            f"[{file_label}] Interval violation val > upper in columns ('val', 'upper') "
            f"at row(s) {_format_rows(bad_rows)}."
        )

    key_cols = ["location", "year", "sex", "scenario", "measure", "metric"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        bad_rows = df.index[dup_mask]
        raise ContractValidationError(
            f"[{file_label}] Duplicate composite keys {key_cols} at row(s) "
            f"{_format_rows(bad_rows)}."
        )

    present_scenarios = set(df["scenario"].unique())
    missing_scenarios = ALLOWED_SCENARIOS - present_scenarios
    if missing_scenarios:
        raise ContractValidationError(
            f"[{file_label}] Missing required scenario(s) in column 'scenario': "
            f"{sorted(missing_scenarios)}."
        )

    for scenario in sorted(ALLOWED_SCENARIOS):
        sub = df[df["scenario"] == scenario]
        for loc in sorted(units):
            loc_sub = sub[sub["location"] == loc]
            if loc_sub.empty:
                raise ContractValidationError(
                    f"[{file_label}] Missing location '{loc}' for scenario '{scenario}'."
                )
            for sex in sorted(ALLOWED_SEXES):
                for measure in sorted(ALLOWED_MEASURES):
                    for metric in sorted(ALLOWED_METRICS):
                        cell = loc_sub[
                            (loc_sub["sex"] == sex)
                            & (loc_sub["measure"] == measure)
                            & (loc_sub["metric"] == metric)
                        ]
                        years_present = set(int(y) for y in cell["year"].tolist())
                        missing_years = REQUIRED_MILESTONE_YEARS - years_present
                        if missing_years:
                            raise ContractValidationError(
                                f"[{file_label}] Missing milestone year(s) {sorted(missing_years)} "
                                f"for ({loc}, {scenario}, {sex}, {measure}, {metric})."
                            )

    return df


def validate_baseline_unit_state(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    file_label: str = "baseline_unit_state",
    expect_mock: bool | None = None,
) -> pd.DataFrame:
    """Validate Contract 2 (``baseline_unit_state``) DataFrame against ``docs/INTERFACES.md``."""
    _require_columns(df, BASELINE_REQUIRED_COLUMNS, file_label)
    _require_non_empty(df, file_label)
    _require_no_nulls(df, BASELINE_REQUIRED_COLUMNS, file_label)

    units = _allowed_units(config)
    _require_string_column(df, "location", file_label, allowed=set(units))
    _require_string_column(df, "country", file_label, allowed=ALLOWED_COUNTRIES)
    _require_string_column(df, "residence", file_label, allowed=ALLOWED_RESIDENCES)
    _require_int_column(df, "baseline_year", file_label, BASELINE_YEAR_MIN, BASELINE_YEAR_MAX)

    _require_float_column(df, "population_total", file_label, min_val=0.0, strictly_positive=True)
    _require_float_column(
        df,
        "population_female_share",
        file_label,
        min_val=0.0,
        max_val=1.0,
        strictly_positive=True,
        strictly_below_max=True,
    )
    _require_float_column(df, "clean_fuel_share_init", file_label, min_val=0.0, max_val=1.0)
    _require_bool_column(df, "lpg_infra_ready_init", file_label)
    _require_bool_column(df, "grid_ready_init", file_label)
    for col in (
        "bau_dalys_2040_female",
        "bau_dalys_2040_male",
        "bau_asr_2040_female",
        "bau_asr_2040_male",
    ):
        _require_float_column(df, col, file_label, min_val=0.0)
    _require_bool_column(df, "is_mock", file_label)
    _require_string_column(df, "data_version", file_label)
    _check_mock_column(df, file_label, expect_mock)

    dup_mask = df.duplicated(subset=["location"], keep=False)
    if dup_mask.any():
        bad_rows = df.index[dup_mask]
        raise ContractValidationError(
            f"[{file_label}] Duplicate 'location' rows at {_format_rows(bad_rows)}."
        )

    missing_units = set(units) - set(df["location"].tolist())
    if missing_units:
        raise ContractValidationError(
            f"[{file_label}] Missing allocation unit(s) in column 'location': "
            f"{sorted(missing_units)}."
        )

    prefix_to_country = {"IND": "India", "NPL": "Nepal"}
    for idx, row in df.iterrows():
        loc = str(row["location"])
        prefix, res_expected = loc.split("_", maxsplit=1)
        expected_country = prefix_to_country.get(prefix)
        if expected_country is not None and row["country"] != expected_country:
            raise ContractValidationError(
                f"[{file_label}] Row {idx}: location '{loc}' expects country "
                f"'{expected_country}', got '{row['country']}'."
            )
        if row["residence"] != res_expected:
            raise ContractValidationError(
                f"[{file_label}] Row {idx}: location '{loc}' expects residence "
                f"'{res_expected}', got '{row['residence']}'."
            )

    return df


def _check_finite_number(value: Any, path_label: str, file_label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(
            f"[{file_label}] Key '{path_label}' must be numeric, got {type(value).__name__}."
        )
    num = float(value)
    if not np.isfinite(num):
        raise ContractValidationError(f"[{file_label}] Key '{path_label}' must be finite.")
    return num


def validate_surrogate_json(
    data: dict[str, Any],
    config: dict[str, Any],
    *,
    file_label: str = "exposure_response_surrogate",
    expect_mock: bool | None = None,
) -> dict[str, Any]:
    """Validate Contract 3 (``exposure_response_surrogate``) JSON payload."""
    if not isinstance(data, dict):
        raise ContractValidationError(f"[{file_label}] Top-level JSON must be an object.")

    for key in SURROGATE_TOP_KEYS:
        if key not in data:
            raise ContractValidationError(f"[{file_label}] Missing required top-level key '{key}'.")

    for str_key in (
        "schema_version",
        "data_version",
        "functional_form",
        "generated_at",
        "source_model",
    ):
        val = data[str_key]
        if not isinstance(val, str) or not val.strip():
            raise ContractValidationError(
                f"[{file_label}] Key '{str_key}' must be a non-empty string."
            )

    if data["functional_form"] not in ALLOWED_FUNCTIONAL_FORMS:
        raise ContractValidationError(
            f"[{file_label}] Invalid 'functional_form': {data['functional_form']!r}. "
            f"Allowed: {sorted(ALLOWED_FUNCTIONAL_FORMS)}."
        )

    is_mock = data["is_mock"]
    if not isinstance(is_mock, bool):
        raise ContractValidationError(
            f"[{file_label}] Key 'is_mock' must be a boolean, got {type(is_mock).__name__}."
        )
    if expect_mock is not None and is_mock is not expect_mock:
        raise ContractValidationError(
            f"[{file_label}] Expected is_mock={expect_mock}, got is_mock={is_mock}."
        )

    expected_units = set(_allowed_units(config))
    units_block = data["units"]
    if not isinstance(units_block, dict):
        raise ContractValidationError(f"[{file_label}] Key 'units' must be an object.")

    actual_units = set(units_block.keys())
    if actual_units != expected_units:
        missing = sorted(expected_units - actual_units)
        extra = sorted(actual_units - expected_units)
        raise ContractValidationError(
            f"[{file_label}] 'units' keys mismatch. Missing={missing}, Unknown={extra}."
        )

    for unit, params in units_block.items():
        if not isinstance(params, dict):
            raise ContractValidationError(f"[{file_label}] 'units.{unit}' must be an object.")
        for ukey in SURROGATE_UNIT_KEYS:
            if ukey not in params or params[ukey] is None:
                raise ContractValidationError(
                    f"[{file_label}] Missing or null key 'units.{unit}.{ukey}'."
                )
        slope = _check_finite_number(
            params["slope_clean_fuel"], f"units.{unit}.slope_clean_fuel", file_label
        )
        if slope > 0.0:
            raise ContractValidationError(
                f"[{file_label}] 'units.{unit}.slope_clean_fuel' must be <= 0.0, got {slope}."
            )
        baseline_cf = _check_finite_number(
            params["baseline_clean_fuel"], f"units.{unit}.baseline_clean_fuel", file_label
        )
        if not (0.0 <= baseline_cf <= 1.0):
            raise ContractValidationError(
                f"[{file_label}] 'units.{unit}.baseline_clean_fuel' must be in [0, 1], "
                f"got {baseline_cf}."
            )
        ratio = _check_finite_number(
            params["sex_elasticity_ratio_f_to_m"],
            f"units.{unit}.sex_elasticity_ratio_f_to_m",
            file_label,
        )
        if ratio <= 0.0:
            raise ContractValidationError(
                f"[{file_label}] 'units.{unit}.sex_elasticity_ratio_f_to_m' must be > 0, "
                f"got {ratio}."
            )
        sd = _check_finite_number(
            params["uncertainty_sd"], f"units.{unit}.uncertainty_sd", file_label
        )
        if sd < 0.0:
            raise ContractValidationError(
                f"[{file_label}] 'units.{unit}.uncertainty_sd' must be >= 0, got {sd}."
            )

    return data


def validate_intervention_params_json(
    data: dict[str, Any],
    config: dict[str, Any],
    *,
    file_label: str = "intervention_params",
    expect_mock: bool | None = None,
    require_populated: bool = True,
) -> dict[str, Any]:
    """Validate Contract 4 (``intervention_params``) JSON payload."""
    if not isinstance(data, dict):
        raise ContractValidationError(f"[{file_label}] Top-level JSON must be an object.")

    for key in INTERVENTION_PARAMS_TOP_KEYS:
        if key not in data:
            raise ContractValidationError(f"[{file_label}] Missing required top-level key '{key}'.")

    is_mock = data["is_mock"]
    if not isinstance(is_mock, bool):
        raise ContractValidationError(
            f"[{file_label}] Key 'is_mock' must be a boolean, got {type(is_mock).__name__}."
        )
    if expect_mock is not None and is_mock is not expect_mock:
        raise ContractValidationError(
            f"[{file_label}] Expected is_mock={expect_mock}, got is_mock={is_mock}."
        )

    for str_key in ("schema_version", "data_version"):
        if not isinstance(data[str_key], str) or not data[str_key].strip():
            raise ContractValidationError(
                f"[{file_label}] Key '{str_key}' must be a non-empty string."
            )

    expected_units = _allowed_units(config)
    expected_interventions = _allowed_interventions(config)
    expected_periods = _allowed_periods(config)

    if data["period_order"] != expected_periods:
        raise ContractValidationError(
            f"[{file_label}] 'period_order' must equal {expected_periods}, "
            f"got {data['period_order']}."
        )
    if not isinstance(data["apply_cost_discounting"], bool):
        raise ContractValidationError(f"[{file_label}] 'apply_cost_discounting' must be a boolean.")

    if not require_populated:
        return data

    if not isinstance(data["generated_at"], str) or not data["generated_at"].strip():
        raise ContractValidationError(
            f"[{file_label}] 'generated_at' must be a non-empty string when populated."
        )

    budget = _check_finite_number(
        data["total_budget_usd_millions"], "total_budget_usd_millions", file_label
    )
    if budget <= 0.0:
        raise ContractValidationError(
            f"[{file_label}] 'total_budget_usd_millions' must be > 0, got {budget}."
        )

    thresh = _check_finite_number(
        data["infra_build_threshold"], "infra_build_threshold", file_label
    )
    if not (0.0 < thresh <= 1.0):
        raise ContractValidationError(
            f"[{file_label}] 'infra_build_threshold' must be in (0, 1], got {thresh}."
        )

    disc = _check_finite_number(data["discount_factor"], "discount_factor", file_label)
    if not (0.0 < disc <= 1.0):
        raise ContractValidationError(
            f"[{file_label}] 'discount_factor' must be in (0, 1], got {disc}."
        )

    unit_set = set(expected_units)
    interv_set = set(expected_interventions)
    period_set = set(expected_periods)

    # coverage_ceiling
    cov_ceil = data["coverage_ceiling"]
    if not isinstance(cov_ceil, dict) or set(cov_ceil.keys()) != unit_set:
        raise ContractValidationError(
            f"[{file_label}] 'coverage_ceiling' must have keys {expected_units}."
        )
    for u in expected_units:
        cmax = _check_finite_number(cov_ceil[u], f"coverage_ceiling.{u}", file_label)
        if not (0.0 < cmax <= 1.0):
            raise ContractValidationError(
                f"[{file_label}] 'coverage_ceiling.{u}' must be in (0, 1], got {cmax}."
            )

    # effectiveness & unit_costs_usd_millions
    for map_name in ("effectiveness", "unit_costs_usd_millions"):
        mapping = data[map_name]
        if not isinstance(mapping, dict) or set(mapping.keys()) != unit_set:
            raise ContractValidationError(
                f"[{file_label}] '{map_name}' must have unit keys {expected_units}."
            )
        for u in expected_units:
            u_map = mapping[u]
            if not isinstance(u_map, dict) or set(u_map.keys()) != interv_set:
                raise ContractValidationError(
                    f"[{file_label}] '{map_name}.{u}' must have intervention keys "
                    f"{expected_interventions}."
                )
            for interv in expected_interventions:
                val = _check_finite_number(u_map[interv], f"{map_name}.{u}.{interv}", file_label)
                if map_name == "unit_costs_usd_millions" and val < 0.0:
                    raise ContractValidationError(
                        f"[{file_label}] '{map_name}.{u}.{interv}' must be >= 0, got {val}."
                    )
                if map_name == "effectiveness":
                    if not (0.0 <= val <= 1.0):
                        raise ContractValidationError(
                            f"[{file_label}] 'effectiveness.{u}.{interv}' must be in [0, 1], "
                            f"got {val}."
                        )
                    if interv == "infra_lpg_grid" and val != 0.0:
                        raise ContractValidationError(
                            f"[{file_label}] 'effectiveness.{u}.infra_lpg_grid' must be 0.0, "
                            f"got {val}."
                        )

    # max_adoption_per_period
    caps = data["max_adoption_per_period"]
    if not isinstance(caps, dict) or set(caps.keys()) != unit_set:
        raise ContractValidationError(
            f"[{file_label}] 'max_adoption_per_period' must have unit keys {expected_units}."
        )
    for u in expected_units:
        u_caps = caps[u]
        if not isinstance(u_caps, dict) or set(u_caps.keys()) != interv_set:
            raise ContractValidationError(
                f"[{file_label}] 'max_adoption_per_period.{u}' must have intervention keys "
                f"{expected_interventions}."
            )
        for interv in expected_interventions:
            p_caps = u_caps[interv]
            if not isinstance(p_caps, dict) or set(p_caps.keys()) != period_set:
                raise ContractValidationError(
                    f"[{file_label}] 'max_adoption_per_period.{u}.{interv}' must have period "
                    f"keys {expected_periods}."
                )
            for p in expected_periods:
                cap = _check_finite_number(
                    p_caps[p], f"max_adoption_per_period.{u}.{interv}.{p}", file_label
                )
                if not (0.0 <= cap <= 1.0):
                    raise ContractValidationError(
                        f"[{file_label}] 'max_adoption_per_period.{u}.{interv}.{p}' "
                        f"must be in [0, 1], got {cap}."
                    )

    return data


def load_forecasts_stage3(
    path: str | Path,
    config: dict[str, Any],
    *,
    expect_mock: bool | None = None,
) -> pd.DataFrame:
    """Load and validate Contract 1 CSV from ``path``."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Forecasts contract file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return validate_forecasts_stage3(df, config, file_label=str(csv_path), expect_mock=expect_mock)


def load_baseline_unit_state(
    path: str | Path,
    config: dict[str, Any],
    *,
    expect_mock: bool | None = None,
) -> pd.DataFrame:
    """Load and validate Contract 2 CSV from ``path``."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline unit state contract file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return validate_baseline_unit_state(
        df, config, file_label=str(csv_path), expect_mock=expect_mock
    )


def load_surrogate(
    path: str | Path,
    config: dict[str, Any],
    *,
    expect_mock: bool | None = None,
) -> dict[str, Any]:
    """Load and validate Contract 3 surrogate JSON from ``path``."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Surrogate contract file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_surrogate_json(data, config, file_label=str(json_path), expect_mock=expect_mock)


def load_intervention_params(
    path: str | Path,
    config: dict[str, Any],
    *,
    expect_mock: bool | None = None,
    require_populated: bool = True,
) -> dict[str, Any]:
    """Load and validate Contract 4 intervention parameters JSON from ``path``."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Intervention params contract file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_intervention_params_json(
        data,
        config,
        file_label=str(json_path),
        expect_mock=expect_mock,
        require_populated=require_populated,
    )


def load_stage3_contracts(
    config: dict[str, Any],
    base_dir: str | Path | None = None,
) -> Stage3Contracts:
    """Load and validate all Stage 3 contracts honoring ``interfaces.use_mock_forecasts``."""
    paths = resolve_contract_paths(config, base_dir=base_dir)
    expect_mock = paths.use_mock
    forecasts = load_forecasts_stage3(paths.forecasts_stage3, config, expect_mock=expect_mock)
    baseline = load_baseline_unit_state(paths.baseline_unit_state, config, expect_mock=expect_mock)
    surrogate = load_surrogate(paths.exposure_response_surrogate, config, expect_mock=expect_mock)
    intervention_params = load_intervention_params(
        paths.intervention_params,
        config,
        expect_mock=expect_mock,
        require_populated=True,
    )
    return Stage3Contracts(
        forecasts_stage3=forecasts,
        baseline_unit_state=baseline,
        surrogate=surrogate,
        intervention_params=intervention_params,
        is_mock=expect_mock,
        paths=paths,
    )
