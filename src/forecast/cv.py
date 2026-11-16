"""Rolling-origin cross-validation for AEGIS-GBD Stage 2.

Evaluates ARIMA, ETS, Poisson Rate GLM, and HistGradientBoosting across
rolling temporal cutoffs (2008, 2013, 2018; horizon = 5) on panel ASR series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.forecast.models import (
    fit_predict_arima,
    fit_predict_ets,
    fit_predict_hist_gbdt,
    fit_predict_poisson_glm,
)


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray,
) -> float:
    """Compute Mean Absolute Scaled Error relative to naive random walk on train."""
    mae = np.mean(np.abs(y_true - y_pred))
    if len(y_train_history) < 2:
        return float(mae)
    scale = np.mean(np.abs(np.diff(y_train_history)))
    if scale < 1e-6:
        return float(mae)
    return float(mae / scale)


def run_rolling_origin_cv(
    df_panel: pd.DataFrame,
    origins: list[int] | None = None,
    horizon: int = 5,
) -> pd.DataFrame:
    """Execute rolling-origin panel cross-validation.

    df_panel must contain: location, year, sex, measure, metric, age, val, clean_fuel_share.
    """
    if origins is None:
        origins = [2008, 2013, 2018]

    # Focus on Rate_per_100k, Age-standardized, DALYs across 5 countries + 4 units
    sub = df_panel[
        (df_panel["metric"] == "Rate_per_100k")
        & (df_panel["age"] == "Age-standardized")
        & (df_panel["measure"] == "DALYs")
        & (df_panel["sex"].isin(["female", "male", "both"]))
    ].copy()

    results: list[dict[str, Any]] = []

    for origin in origins:
        test_start = origin + 1
        test_end = origin + horizon

        train_data = sub[sub["year"] <= origin].copy()
        test_data = sub[(sub["year"] >= test_start) & (sub["year"] <= test_end)].copy()

        if test_data.empty:
            continue

        # 1. HistGradientBoosting Panel Model
        gbdt_preds, _ = fit_predict_hist_gbdt(train_data, test_data)
        test_data_gbdt = test_data.copy()
        test_data_gbdt["pred_gbdt"] = gbdt_preds

        # 2. Poisson Rate GLM
        poisson_preds = fit_predict_poisson_glm(train_data, test_data)
        test_data["pred_poisson"] = poisson_preds
        test_data["pred_gbdt"] = gbdt_preds

        # Evaluate per series (location x sex)
        for loc in sorted(sub["location"].unique()):
            for s in ["female", "male", "both"]:
                t_sub = test_data[
                    (test_data["location"] == loc) & (test_data["sex"] == s)
                ].sort_values("year")
                if len(t_sub) != horizon:
                    continue

                y_true = t_sub["val"].to_numpy(dtype=float)
                hist_series = (
                    train_data[(train_data["location"] == loc) & (train_data["sex"] == s)]
                    .sort_values("year")["val"]
                    .to_numpy(dtype=float)
                )

                if len(hist_series) < 5:
                    continue

                # Univariate Models
                pred_arima = fit_predict_arima(pd.Series(hist_series), horizon=horizon)
                pred_ets = fit_predict_ets(pd.Series(hist_series), horizon=horizon)
                pred_naive = np.repeat(hist_series[-1], horizon)

                models_dict = {
                    "naive_baseline": pred_naive,
                    "arima": pred_arima,
                    "ets": pred_ets,
                    "poisson_rate_glm": t_sub["pred_poisson"].to_numpy(dtype=float),
                    "hist_gbdt_panel": t_sub["pred_gbdt"].to_numpy(dtype=float),
                }

                for m_name, y_pred in models_dict.items():
                    mae = float(np.mean(np.abs(y_true - y_pred)))
                    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                    mase = compute_mase(y_true, y_pred, hist_series)

                    results.append(
                        {
                            "origin_year": origin,
                            "test_window": f"{test_start}-{test_end}",
                            "location": loc,
                            "sex": s,
                            "model": m_name,
                            "mae": mae,
                            "rmse": rmse,
                            "mase": mase,
                        }
                    )

    return pd.DataFrame(results)


def plot_cv_comparison(df_cv: pd.DataFrame, output_path: Path) -> Path:
    """Plot cross-validation benchmark comparisons across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregated metrics per model
    agg = df_cv.groupby("model")[["mae", "rmse", "mase"]].mean().reset_index()
    agg = agg.sort_values("mase")

    # Panel A: MASE Comparison
    sns.barplot(
        data=agg,
        x="model",
        y="mase",
        ax=axes[0],
        palette="viridis",
        hue="model",
        legend=False,
    )
    axes[0].set_title(
        "Mean Absolute Scaled Error (MASE) across Models", fontsize=11, fontweight="bold"
    )
    axes[0].set_ylabel("MASE (Lower is Better)")
    axes[0].set_xlabel("Model Architecture")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].axhline(
        1.0, color="red", linestyle="--", linewidth=1.0, label="Naive Benchmark (MASE=1.0)"
    )
    axes[0].legend()

    # Panel B: MAE by Origin Year
    origin_agg = df_cv.groupby(["origin_year", "model"])["mae"].mean().reset_index()
    sns.barplot(
        data=origin_agg,
        x="origin_year",
        y="mae",
        hue="model",
        ax=axes[1],
        palette="tab10",
    )
    axes[1].set_title("Mean Absolute Error (MAE) by Rolling Origin", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("MAE (DALYs per 100k)")
    axes[1].set_xlabel("Origin Year")
    axes[1].legend(title="Model", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
