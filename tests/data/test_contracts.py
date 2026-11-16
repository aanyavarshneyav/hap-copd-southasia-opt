"""Unit tests for Stage 3 contract validators and loaders (``src/data/contracts.py``)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from src.data.contracts import (
    ContractValidationError,
    load_config,
    load_stage3_contracts,
    validate_baseline_unit_state,
    validate_forecasts_stage3,
    validate_intervention_params_json,
    validate_surrogate_json,
)
from src.data.mock import (
    build_mock_baseline,
    build_mock_forecasts,
    build_mock_intervention_params,
    build_mock_surrogate,
    generate_mock_fixtures,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture()
def config() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


@pytest.fixture()
def valid_forecasts(config: dict[str, Any]) -> pd.DataFrame:
    return build_mock_forecasts(config, seed=int(config["seeds"]["global"]))


@pytest.fixture()
def valid_baseline(valid_forecasts: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return build_mock_baseline(valid_forecasts, config)


@pytest.fixture()
def valid_surrogate(valid_forecasts: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    return build_mock_surrogate(valid_forecasts, config)


@pytest.fixture()
def valid_params(config: dict[str, Any]) -> dict[str, Any]:
    return build_mock_intervention_params(config)


def test_valid_mocks_pass_all_validators(
    config: dict[str, Any],
    valid_forecasts: pd.DataFrame,
    valid_baseline: pd.DataFrame,
    valid_surrogate: dict[str, Any],
    valid_params: dict[str, Any],
) -> None:
    validate_forecasts_stage3(valid_forecasts, config, expect_mock=True)
    validate_baseline_unit_state(valid_baseline, config, expect_mock=True)
    validate_surrogate_json(valid_surrogate, config, expect_mock=True)
    validate_intervention_params_json(valid_params, config, expect_mock=True)


def test_forecasts_accepts_full_annual_superset(
    config: dict[str, Any],
    valid_forecasts: pd.DataFrame,
) -> None:
    # Add intermediate years to cf_100 to verify Person A's full annual table is accepted.
    extra = valid_forecasts[
        (valid_forecasts["scenario"] == "cf_100") & (valid_forecasts["year"] == 2030)
    ].copy()
    extra["year"] = 2027
    superset = pd.concat([valid_forecasts, extra], ignore_index=True)
    validated = validate_forecasts_stage3(superset, config, expect_mock=True)
    assert len(validated) == len(superset)


def test_forecasts_missing_column_fails(
    config: dict[str, Any], valid_forecasts: pd.DataFrame
) -> None:
    corrupted = valid_forecasts.drop(columns=["clean_fuel_share"])
    with pytest.raises(ContractValidationError, match="clean_fuel_share"):
        validate_forecasts_stage3(corrupted, config, file_label="forecasts_test")


def test_forecasts_wrong_dtype_fails(config: dict[str, Any], valid_forecasts: pd.DataFrame) -> None:
    corrupted = valid_forecasts.copy()
    corrupted["year"] = corrupted["year"].astype(float)
    with pytest.raises(ContractValidationError, match="year.*integer"):
        validate_forecasts_stage3(corrupted, config, file_label="forecasts_test")


def test_forecasts_lower_gt_upper_fails_with_row(
    config: dict[str, Any], valid_forecasts: pd.DataFrame
) -> None:
    corrupted = valid_forecasts.copy()
    corrupted.loc[3, "lower"] = corrupted.loc[3, "upper"] + 10.0
    with pytest.raises(ContractValidationError, match=r"lower > val.*\[3\]"):
        validate_forecasts_stage3(corrupted, config, file_label="forecasts_test")


def test_forecasts_negative_count_fails_with_row(
    config: dict[str, Any], valid_forecasts: pd.DataFrame
) -> None:
    corrupted = valid_forecasts.copy()
    corrupted.loc[5, "lower"] = -1.0
    with pytest.raises(ContractValidationError, match=r"lower.*\[5\]"):
        validate_forecasts_stage3(corrupted, config, file_label="forecasts_test")


def test_forecasts_unknown_location_fails(
    config: dict[str, Any], valid_forecasts: pd.DataFrame
) -> None:
    corrupted = valid_forecasts.copy()
    corrupted.loc[0, "location"] = "PAK_urban"
    with pytest.raises(ContractValidationError, match="PAK_urban"):
        validate_forecasts_stage3(corrupted, config, file_label="forecasts_test")


def test_forecasts_null_value_fails(config: dict[str, Any], valid_forecasts: pd.DataFrame) -> None:
    corrupted = valid_forecasts.copy()
    corrupted.loc[2, "val"] = None
    with pytest.raises(ContractValidationError, match=r"val.*nulls.*\[2\]"):
        validate_forecasts_stage3(corrupted, config, file_label="forecasts_test")


def test_baseline_missing_unit_and_bad_bool_dtype(
    config: dict[str, Any], valid_baseline: pd.DataFrame
) -> None:
    missing_unit = valid_baseline[valid_baseline["location"] != "NPL_rural"].copy()
    with pytest.raises(ContractValidationError, match="NPL_rural"):
        validate_baseline_unit_state(missing_unit, config)

    bad_bool = valid_baseline.copy()
    bad_bool["lpg_infra_ready_init"] = bad_bool["lpg_infra_ready_init"].astype(str)
    with pytest.raises(ContractValidationError, match="lpg_infra_ready_init.*boolean"):
        validate_baseline_unit_state(bad_bool, config)


def test_surrogate_validation_failures(
    config: dict[str, Any], valid_surrogate: dict[str, Any]
) -> None:
    bad_slope = copy.deepcopy(valid_surrogate)
    bad_slope["units"]["IND_rural"]["slope_clean_fuel"] = 0.5
    with pytest.raises(ContractValidationError, match="slope_clean_fuel"):
        validate_surrogate_json(bad_slope, config)

    bad_unit = copy.deepcopy(valid_surrogate)
    del bad_unit["units"]["NPL_urban"]
    with pytest.raises(ContractValidationError, match="NPL_urban"):
        validate_surrogate_json(bad_unit, config)


def test_intervention_params_validation_failures(
    config: dict[str, Any], valid_params: dict[str, Any]
) -> None:
    bad_budget = copy.deepcopy(valid_params)
    bad_budget["total_budget_usd_millions"] = 0.0
    with pytest.raises(ContractValidationError, match="total_budget_usd_millions"):
        validate_intervention_params_json(bad_budget, config)

    bad_cost = copy.deepcopy(valid_params)
    bad_cost["unit_costs_usd_millions"]["IND_urban"]["lpg_subsidy"] = -5.0
    with pytest.raises(ContractValidationError, match="unit_costs_usd_millions"):
        validate_intervention_params_json(bad_cost, config)


def test_loader_rejects_mock_files_when_use_mock_false(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    generate_mock_fixtures(CONFIG_PATH, base_dir=tmp_path)
    prod_cfg = copy.deepcopy(config)
    prod_cfg["interfaces"]["use_mock_forecasts"] = False
    prod_cfg["interfaces"]["forecast_contract_file"] = prod_cfg["interfaces"]["forecast_mock_file"]
    prod_cfg["interfaces"]["baseline_state_file"] = prod_cfg["interfaces"][
        "baseline_state_mock_file"
    ]
    prod_cfg["interfaces"]["surrogate_file"] = prod_cfg["interfaces"]["surrogate_mock_file"]
    prod_cfg["interfaces"]["intervention_params_file"] = prod_cfg["interfaces"][
        "intervention_params_mock_file"
    ]
    with pytest.raises(ContractValidationError, match="Expected is_mock=False"):
        load_stage3_contracts(prod_cfg, base_dir=tmp_path)
