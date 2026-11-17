"""Unit & regression tests for B-12 forecasting fixes and multi-horizon CV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.forecast.cv import compute_mase, run_multihorizon_cv_b12
from src.forecast.models import (
    ModelFitError,
    fit_predict_arima,
    fit_predict_drift,
    fit_predict_ets,
    fit_predict_log_ols,
    fit_predict_theta,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_synthetic_panel(seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    years = list(range(1990, 2024))
    for loc, base, cf_slope, cf_effect in (
        ("India", 1200.0, 0.024, 1.5),
        ("Nepal", 1500.0, 0.011, 1.5),
    ):
        for i, yr in enumerate(years):
            # Distinct clean-fuel trajectories across locations so t and cf are identified
            cf = float(np.clip(0.15 + cf_slope * i + 0.03 * np.sin(0.25 * i), 0.05, 0.96))
            val = base * np.exp(-0.01 * i - cf_effect * cf) + rng.normal(0.0, 5.0)
            rows.append(
                {
                    "location": loc,
                    "year": yr,
                    "sex": "both",
                    "measure": "DALYs",
                    "metric": "Rate_per_100k",
                    "age": "Age-standardized",
                    "val": max(float(val), 10.0),
                    "clean_fuel_share": cf,
                }
            )
    return pd.DataFrame(rows)


def test_ets_not_identical_to_arima() -> None:
    """B-12 Task 2: Fixed ExponentialSmoothing(optimized=True) differs from ARIMA."""
    rng = np.random.default_rng(7)
    t = np.arange(25, dtype=float)
    series = pd.Series(900.0 * np.exp(-0.03 * t) + 18.0 * np.sin(0.6 * t) + rng.normal(0, 8, 25))
    pred_ets = fit_predict_ets(series, horizon=5, damped_trend=True)
    pred_arima = fit_predict_arima(series, horizon=5)
    assert pred_ets.shape == (5,)
    assert np.all(np.isfinite(pred_ets))
    assert np.max(np.abs(pred_ets - pred_arima)) > 1.0


def test_ets_failure_logged_no_silent_arima_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-12 Task 2: ETS failures raise ModelFitError and are logged explicitly in CV."""
    import src.forecast.models as m_mod

    class _BrokenETS:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated optimizer crash")

    monkeypatch.setattr(m_mod, "ExponentialSmoothing", _BrokenETS)

    series = pd.Series(np.linspace(500.0, 300.0, 20))
    with pytest.raises(ModelFitError, match="simulated optimizer crash"):
        fit_predict_ets(series, horizon=5)

    panel = _make_synthetic_panel()
    metrics_df, _, _, failures_df = run_multihorizon_cv_b12(
        panel, origins=[2013], horizons=(5,), sexes=("both",)
    )
    ets_fail = failures_df[failures_df["model"] == "ets_damped"]
    assert not ets_fail.empty
    assert (ets_fail["location"] == "India").any()
    assert ets_fail["error"].str.contains("simulated optimizer crash").any()

    ets_rows = metrics_df[metrics_df["model"] == "ets_damped"]
    assert (ets_rows["status"] == "FAILED").all()
    assert ets_rows["mae"].isna().all()


def test_drift_theta_and_causal_mase_and_skill_scores() -> None:
    """B-12 Tasks 3 & 4: Drift/Theta work cleanly, MASE uses train only, skill = 1 - relMAE."""
    series = pd.Series(np.linspace(800.0, 400.0, 21) + np.sin(np.arange(21)) * 5.0)
    pred_d = fit_predict_drift(series, horizon=5)
    pred_t = fit_predict_theta(series, horizon=5)
    assert pred_d.shape == (5,) and np.all(pred_d >= 0.0)
    assert pred_t.shape == (5,) and np.all(pred_t >= 0.0)

    train_hist = series.to_numpy(dtype=float)
    y_true = np.array([380.0, 360.0, 340.0, 320.0, 300.0])
    # In-sample scale is independent of test horizon length / values
    scale_expected = float(np.mean(np.abs(np.diff(train_hist))))
    mase_obs = compute_mase(y_true, pred_d, train_hist)
    assert np.isclose(mase_obs, float(np.mean(np.abs(y_true - pred_d))) / scale_expected, atol=1e-9)

    panel = _make_synthetic_panel()
    metrics_df, summary_df, _, _ = run_multihorizon_cv_b12(
        panel, origins=[2013, 2018], horizons=(1, 5), sexes=("both",)
    )
    ok = metrics_df[metrics_df["status"] == "OK"].dropna(subset=["relative_mae", "skill_score"])
    assert np.allclose(ok["relative_mae"], ok["mae"] / ok["naive_mae"], atol=1e-10)
    assert np.allclose(ok["skill_score"], 1.0 - ok["relative_mae"], atol=1e-10)
    assert set(summary_df["horizon"].unique()) == {1, 5}


def test_log_ols_counterfactual_has_strictly_negative_high_cf_slope() -> None:
    """B-12 Task 5: Smooth log-OLS does not saturate at clean_fuel_share >= 0.98."""
    panel = _make_synthetic_panel()
    grid = pd.DataFrame(
        {
            "location": ["India", "India"],
            "year": [2040, 2040],
            "sex": ["both", "both"],
            "clean_fuel_share": [0.95, 1.00],
        }
    )
    preds, _ = fit_predict_log_ols(panel, grid)
    assert preds[1] < preds[0]


def test_b12_outputs_exist_when_generated() -> None:
    """Verify generated B-12 benchmark artifacts pass structural invariants."""
    sum_path = REPO_ROOT / "outputs" / "tables" / "s2_cv_summary_b12.csv"
    if not sum_path.exists():
        pytest.skip("B-12 benchmark tables not yet generated.")
    df_sum = pd.read_csv(sum_path)
    assert {1, 3, 5, 10}.issubset(set(df_sum["horizon"].unique()))
    assert (df_sum["n_failed"] == 0).all()

    h5 = df_sum[df_sum["horizon"] == 5].set_index("model")
    assert abs(h5.loc["ets_damped", "mean_mae"] - h5.loc["arima", "mean_mae"]) > 0.1
    assert h5.loc["ets_undamped", "pooled_skill_score"] > 0.5
