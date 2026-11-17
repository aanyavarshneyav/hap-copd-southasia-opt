"""Rolling-origin cross-validation for AEGIS-GBD Stage 2.

Evaluates Naive, Drift, ARIMA, ETS (damped & undamped), Theta, Poisson Rate GLM,
Log-OLS, and HistGradientBoosting across rolling temporal cutoffs and multiple
horizons (h = 1, 3, 5, 10) with explicit failure tracking and no silent fallbacks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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
    fit_predict_drift,
    fit_predict_ets,
    fit_predict_hist_gbdt,
    fit_predict_log_ols,
    fit_predict_naive,
    fit_predict_poisson_glm,
    fit_predict_theta,
)

logger = logging.getLogger(__name__)


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray,
) -> float:
    """Compute MASE using strictly in-sample (year <= origin) 1-step naive scale."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if len(y_train_history) < 2:
        return mae
    scale = float(np.mean(np.abs(np.diff(y_train_history))))
    if scale < 1e-6:
        return mae
    return float(mae / scale)


def _safe_fit_series(
    model_name: str,
    fn: Callable[[], np.ndarray],
    loc: str,
    sex: str,
    origin: int,
    horizon: int,
    failures: list[dict[str, Any]],
) -> tuple[np.ndarray | None, str | None]:
    """Run a model fitter and explicitly record failures without fallback masking."""
    try:
        pred = np.asarray(fn(), dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Prediction contained NaN or Inf values.")
        return pred, None
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "CV fit failure model=%s loc=%s sex=%s origin=%d h=%d: %s",
            model_name,
            loc,
            sex,
            origin,
            horizon,
            err_msg,
        )
        failures.append(
            {
                "horizon": horizon,
                "origin_year": origin,
                "location": loc,
                "sex": sex,
                "model": model_name,
                "error": err_msg,
            }
        )
        return None, err_msg


def run_multihorizon_cv_b12(
    df_panel: pd.DataFrame,
    origins: list[int] | None = None,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    sexes: tuple[str, ...] = ("female", "male", "both"),
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute rolling-origin multi-horizon CV with explicit failure audit and skill scores.

    Returns ``(metrics_df, summary_df, per_series_winners_df, failures_df)``.
    """
    if origins is None:
        origins = [2008, 2013, 2018]

    sub = df_panel[
        (df_panel["metric"] == "Rate_per_100k")
        & (df_panel["age"] == "Age-standardized")
        & (df_panel["measure"] == "DALYs")
        & (df_panel["sex"].isin(list(sexes)))
    ].copy()

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    max_year = int(sub["year"].max())

    for horizon in horizons:
        for origin in origins:
            test_start = origin + 1
            test_end = origin + horizon
            if test_end > max_year:
                continue

            train_data = sub[sub["year"] <= origin].copy()
            test_data = sub[(sub["year"] >= test_start) & (sub["year"] <= test_end)].copy()
            if test_data.empty:
                continue

            # Fit panel models strictly on train_data (year <= origin)
            panel_preds: dict[str, pd.Series | None] = {}
            panel_errs: dict[str, str | None] = {}

            for pname, pfn in [
                (
                    "hist_gbdt_panel",
                    lambda td=train_data, vd=test_data: fit_predict_hist_gbdt(
                        td, vd, random_state=random_state
                    )[0],
                ),
                (
                    "poisson_rate_glm",
                    lambda td=train_data, vd=test_data: fit_predict_poisson_glm(td, vd),
                ),
                (
                    "log_ols_panel",
                    lambda td=train_data, vd=test_data: fit_predict_log_ols(td, vd)[0],
                ),
            ]:
                p_arr, err = _safe_fit_series(
                    pname,
                    pfn,
                    loc="ALL_PANEL",
                    sex=",".join(sexes),
                    origin=origin,
                    horizon=horizon,
                    failures=failures,
                )
                panel_preds[pname] = (
                    pd.Series(p_arr, index=test_data.index) if p_arr is not None else None
                )
                panel_errs[pname] = err

            for loc in sorted(sub["location"].unique()):
                for s in sexes:
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

                    hist_pd = pd.Series(hist_series)
                    scale_in = float(np.mean(np.abs(np.diff(hist_series))))
                    pred_naive = fit_predict_naive(hist_pd, horizon)
                    naive_mae = float(np.mean(np.abs(y_true - pred_naive)))

                    model_candidates: list[tuple[str, np.ndarray | None, str | None]] = [
                        ("naive_baseline", pred_naive, None),
                    ]
                    for uname, ufn in [
                        ("drift", lambda h=hist_pd, hz=horizon: fit_predict_drift(h, hz)),
                        ("arima", lambda h=hist_pd, hz=horizon: fit_predict_arima(h, hz)),
                        (
                            "ets_damped",
                            lambda h=hist_pd, hz=horizon: fit_predict_ets(h, hz, damped_trend=True),
                        ),
                        (
                            "ets_undamped",
                            lambda h=hist_pd, hz=horizon: fit_predict_ets(
                                h, hz, damped_trend=False
                            ),
                        ),
                        ("theta", lambda h=hist_pd, hz=horizon: fit_predict_theta(h, hz)),
                    ]:
                        u_pred, u_err = _safe_fit_series(
                            uname,
                            ufn,
                            loc=loc,
                            sex=s,
                            origin=origin,
                            horizon=horizon,
                            failures=failures,
                        )
                        model_candidates.append((uname, u_pred, u_err))

                    for pname in ("poisson_rate_glm", "log_ols_panel", "hist_gbdt_panel"):
                        p_ser = panel_preds[pname]
                        if p_ser is None:
                            model_candidates.append((pname, None, panel_errs[pname]))
                        else:
                            model_candidates.append(
                                (pname, p_ser.loc[t_sub.index].to_numpy(dtype=float), None)
                            )

                    for m_name, y_pred, err_msg in model_candidates:
                        if y_pred is None:
                            rows.append(
                                {
                                    "horizon": horizon,
                                    "origin_year": origin,
                                    "test_window": f"{test_start}-{test_end}",
                                    "location": loc,
                                    "sex": s,
                                    "model": m_name,
                                    "mae": float("nan"),
                                    "rmse": float("nan"),
                                    "mase": float("nan"),
                                    "naive_mae": naive_mae,
                                    "relative_mae": float("nan"),
                                    "skill_score": float("nan"),
                                    "scale_in_sample": scale_in,
                                    "status": "FAILED",
                                    "error": err_msg,
                                }
                            )
                            continue

                        mae = float(np.mean(np.abs(y_true - y_pred)))
                        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                        mase = compute_mase(y_true, y_pred, hist_series)
                        rel_mae = float(mae / naive_mae) if naive_mae > 1e-9 else float("nan")
                        skill = float(1.0 - rel_mae) if np.isfinite(rel_mae) else float("nan")
                        rows.append(
                            {
                                "horizon": horizon,
                                "origin_year": origin,
                                "test_window": f"{test_start}-{test_end}",
                                "location": loc,
                                "sex": s,
                                "model": m_name,
                                "mae": mae,
                                "rmse": rmse,
                                "mase": mase,
                                "naive_mae": naive_mae,
                                "relative_mae": rel_mae,
                                "skill_score": skill,
                                "scale_in_sample": scale_in,
                                "status": "OK",
                                "error": "",
                            }
                        )

    metrics_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(
        failures,
        columns=["horizon", "origin_year", "location", "sex", "model", "error"],
    )

    if metrics_df.empty:
        return metrics_df, pd.DataFrame(), pd.DataFrame(), failures_df

    # Macro + Pooled Summary per (horizon, model)
    summary_rows: list[dict[str, Any]] = []
    for (hz, m_name), grp in metrics_df.groupby(["horizon", "model"], sort=False):
        ok = grp[grp["status"] == "OK"]
        n_ok = len(ok)
        n_fail = int((grp["status"] == "FAILED").sum())
        mean_mae = float(ok["mae"].mean()) if n_ok else float("nan")
        mean_naive = float(ok["naive_mae"].mean()) if n_ok else float("nan")
        pooled_rel = (mean_mae / mean_naive) if (n_ok and mean_naive > 1e-9) else float("nan")
        summary_rows.append(
            {
                "horizon": int(hz),
                "model": str(m_name),
                "n_eval": n_ok,
                "n_failed": n_fail,
                "mean_mae": round(mean_mae, 4) if np.isfinite(mean_mae) else float("nan"),
                "mean_rmse": round(float(ok["rmse"].mean()), 4) if n_ok else float("nan"),
                "mean_mase": round(float(ok["mase"].mean()), 4) if n_ok else float("nan"),
                "macro_relative_mae": round(float(ok["relative_mae"].mean()), 4)
                if n_ok
                else float("nan"),
                "macro_skill_score": round(float(ok["skill_score"].mean()), 4)
                if n_ok
                else float("nan"),
                "pooled_relative_mae": round(pooled_rel, 4)
                if np.isfinite(pooled_rel)
                else float("nan"),
                "pooled_skill_score": round(1.0 - pooled_rel, 4)
                if np.isfinite(pooled_rel)
                else float("nan"),
            }
        )
    summary_df = (
        pd.DataFrame(summary_rows).sort_values(["horizon", "mean_mae"]).reset_index(drop=True)
    )

    # Per-series best vs global best (Task 4)
    winner_rows: list[dict[str, Any]] = []
    ok_all = metrics_df[metrics_df["status"] == "OK"]
    for hz, h_grp in ok_all.groupby("horizon"):
        global_agg = h_grp.groupby("model")["mae"].mean()
        global_best_model = str(global_agg.idxmin())
        for (loc, s), s_grp in h_grp.groupby(["location", "sex"]):
            m_agg = s_grp.groupby("model")["mae"].mean()
            best_m = str(m_agg.idxmin())
            best_mae = float(m_agg.loc[best_m])
            glob_mae = (
                float(m_agg.loc[global_best_model]) if global_best_model in m_agg else float("nan")
            )
            naive_s = float(s_grp[s_grp["model"] == "naive_baseline"]["mae"].mean())
            winner_rows.append(
                {
                    "horizon": int(hz),
                    "location": str(loc),
                    "sex": str(s),
                    "per_series_best_model": best_m,
                    "per_series_best_mae": round(best_mae, 4),
                    "global_best_model": global_best_model,
                    "global_best_mae": round(glob_mae, 4),
                    "per_series_improvement_mae": round(glob_mae - best_mae, 4),
                    "naive_mae": round(naive_s, 4),
                }
            )
    winners_df = pd.DataFrame(winner_rows)
    return metrics_df, summary_df, winners_df, failures_df


def run_rolling_origin_cv(
    df_panel: pd.DataFrame,
    origins: list[int] | None = None,
    horizon: int = 5,
) -> pd.DataFrame:
    """Execute rolling-origin panel cross-validation cleanly (fixed ETS, no silent fallbacks).

    Returns per-(origin, location, sex, model) MAE, RMSE, MASE, relative_mae, and skill_score.
    """
    metrics_df, _, _, _ = run_multihorizon_cv_b12(
        df_panel=df_panel,
        origins=origins,
        horizons=(horizon,),
    )
    if metrics_df.empty:
        return metrics_df
    cols = [
        "origin_year",
        "test_window",
        "location",
        "sex",
        "model",
        "mae",
        "rmse",
        "mase",
        "relative_mae",
        "skill_score",
        "status",
        "error",
    ]
    return metrics_df[cols].copy()


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
