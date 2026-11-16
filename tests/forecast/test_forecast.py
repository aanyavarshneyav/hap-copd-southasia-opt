"""Unit & smoke tests for Stage 2 Forecasting and Contract Export."""

from pathlib import Path

import numpy as np
import pandas as pd
from src.data.contracts import (
    load_config,
    validate_baseline_unit_state,
    validate_forecasts_stage3,
    validate_surrogate_json,
)
from src.forecast.models import fit_predict_arima, fit_predict_ets, fit_predict_hist_gbdt


def test_arima_and_ets_forecast() -> None:
    """Verify univariate models produce valid non-negative forecasts."""
    series = pd.Series(np.linspace(100.0, 50.0, 30) + np.random.normal(0, 1, 30))
    arima_pred = fit_predict_arima(series, horizon=5)
    ets_pred = fit_predict_ets(series, horizon=5)

    assert len(arima_pred) == 5
    assert len(ets_pred) == 5
    assert (arima_pred >= 0).all()
    assert (ets_pred >= 0).all()


def test_hist_gbdt_panel() -> None:
    """Verify panel model trains and predicts correctly."""
    train_df = pd.DataFrame(
        {
            "year": np.repeat(range(1990, 2020), 4),
            "clean_fuel_share": np.random.uniform(0.1, 0.9, 120),
            "location": ["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"] * 30,
            "sex": ["both"] * 120,
            "val": np.random.uniform(500, 1500, 120),
        }
    )
    test_df = pd.DataFrame(
        {
            "year": [2025, 2026, 2027, 2028],
            "clean_fuel_share": [0.8, 0.85, 0.9, 0.95],
            "location": ["IND_urban"] * 4,
            "sex": ["both"] * 4,
        }
    )

    preds, model = fit_predict_hist_gbdt(train_df, test_df)
    assert len(preds) == 4
    assert (preds > 0).all()


def test_production_contracts_exist_and_validate() -> None:
    """Verify production contracts in data/processed validate cleanly under expect_mock=False."""
    cfg = load_config("configs/default.yaml")

    fc_path = Path(cfg["paths"]["processed_dir"]) / "forecasts_stage3.csv"
    base_path = Path(cfg["paths"]["processed_dir"]) / "baseline_unit_state.csv"
    surr_path = Path(cfg["paths"]["processed_dir"]) / "exposure_response_surrogate.json"

    assert fc_path.exists()
    assert base_path.exists()
    assert surr_path.exists()

    df_fc = pd.read_csv(fc_path)
    df_base = pd.read_csv(base_path)
    import json

    with open(surr_path, encoding="utf-8") as f:
        surr_dict = json.load(f)

    # Must pass schema validation with expect_mock=False
    validate_forecasts_stage3(df_fc, cfg, expect_mock=False)
    validate_baseline_unit_state(df_base, cfg, expect_mock=False)
    validate_surrogate_json(surr_dict, cfg, expect_mock=False)
