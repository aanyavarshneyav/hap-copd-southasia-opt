"""Regression tests verifying smooth Log-OLS replacement & HistGBDT snapshot preservation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data.contracts import load_config, validate_surrogate_json
from src.forecast.b13_surrogate import evaluate_identical_folds_4unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_production_surrogate_is_log_ols_smooth_and_monotone() -> None:
    """Confirm Contract 3 uses smooth Log-OLS with strictly negative slopes across all 4 units."""
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    surr_path = REPO_ROOT / "data" / "processed" / "exposure_response_surrogate.json"
    surr = json.loads(surr_path.read_text(encoding="utf-8"))
    validate_surrogate_json(surr, cfg, expect_mock=False)

    assert surr["source_model"] == "log_ols_smooth"
    for unit in ("IND_urban", "IND_rural", "NPL_urban", "NPL_rural"):
        assert surr["units"][unit]["slope_clean_fuel"] < -1.0
        assert surr["units"][unit]["uncertainty_sd"] > 0.15


def test_histgbdt_baseline_snapshot_preserved() -> None:
    """Ensure HistGBDT surrogate snapshot & baseline outputs remain preserved for audit."""
    snap_surr = REPO_ROOT / "data" / "processed" / "exposure_response_surrogate_histgbdt.json"
    snap_dir = REPO_ROOT / "outputs" / "tables" / "histgbdt_baseline_snapshot"
    if not snap_surr.exists() or not snap_dir.exists():
        pytest.skip("HistGBDT preservation snapshot not present.")

    old = json.loads(snap_surr.read_text(encoding="utf-8"))
    assert old["source_model"] == "hist_gbdt_panel"
    assert old["units"]["IND_urban"]["slope_clean_fuel"] == 0.0
    assert (snap_dir / "s3_selected_plans.csv").exists()
    assert (snap_dir / "comp_ga_vs_mdp.csv").exists()


def test_identical_fold_counterfactual_cv_log_ols_beats_histgbdt() -> None:
    """Verify identical-fold 4-unit CV confirms Log-OLS accuracy advantage over HistGBDT."""
    cv_path = REPO_ROOT / "outputs" / "tables" / "s2_counterfactual_cv_identical_b13.csv"
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
    else:
        from src.forecast.b12_benchmark import reconstruct_historical_panel_b12

        cv_df = evaluate_identical_folds_4unit(reconstruct_historical_panel_b12(REPO_ROOT))

    h5 = cv_df[cv_df["horizon"] == 5].groupby("model")["mae"].mean()
    assert float(h5["log_ols_smooth"]) < float(h5["hist_gbdt_panel"])
    assert np.isfinite(float(h5["log_ols_smooth"]))
