"""Forecasting models for AEGIS-GBD Stage 2.

Implements statistical (Naive, Drift, ARIMA, ETS damped/undamped, Theta, Poisson Rate GLM),
smooth parametric (Log-OLS, Log-Solid), and ML panel models (HistGradientBoostingRegressor
unconstrained and monotonic) for projecting HAP-attributable COPD ASRs.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ModelFitError(RuntimeError):
    """Raised explicitly when a time-series or panel model fails to fit."""


def fit_predict_naive(history: pd.Series, horizon: int) -> np.ndarray:
    """Persistence forecast repeating the last observed historical value."""
    vals = history.to_numpy(dtype=float)
    if len(vals) == 0:
        raise ModelFitError("Naive forecast received empty history.")
    return np.maximum(np.repeat(vals[-1], horizon), 0.0)


def fit_predict_drift(history: pd.Series, horizon: int) -> np.ndarray:
    """Random-walk with drift interpolating average historical step change."""
    vals = history.to_numpy(dtype=float)
    if len(vals) == 0:
        raise ModelFitError("Drift forecast received empty history.")
    if len(vals) == 1:
        return np.maximum(np.repeat(vals[-1], horizon), 0.0)
    slope = (vals[-1] - vals[0]) / (len(vals) - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    return np.maximum(vals[-1] + slope * steps, 0.0)


def fit_predict_arima(
    history: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 0),
) -> np.ndarray:
    """Fit ARIMA on historical series and forecast forward without silent fallbacks."""
    vals = history.to_numpy(dtype=float)
    if len(vals) < 5 or np.all(vals == vals[0]):
        return np.maximum(np.repeat(vals[-1], horizon), 0.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(vals, order=order)
            res = model.fit()
            pred = np.asarray(res.forecast(steps=horizon), dtype=float)
            if not np.all(np.isfinite(pred)):
                raise ModelFitError("ARIMA produced non-finite forecasts.")
            return np.maximum(pred, 0.0)
    except ModelFitError:
        raise
    except Exception as exc:
        raise ModelFitError(f"ARIMA{order} failed: {type(exc).__name__}: {exc}") from exc


def fit_predict_ets(
    history: pd.Series,
    horizon: int,
    trend: str = "add",
    damped_trend: bool = True,
) -> np.ndarray:
    """Fit Holt-Winters Exponential Smoothing without silent ARIMA fallback."""
    vals = history.to_numpy(dtype=float)
    if len(vals) < 5 or np.all(vals == vals[0]):
        return np.maximum(np.repeat(vals[-1], horizon), 0.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                vals,
                trend=trend,
                damped_trend=damped_trend,
                initialization_method="estimated",
            )
            res = model.fit(optimized=True)
            pred = np.asarray(res.forecast(horizon), dtype=float)
            if not np.all(np.isfinite(pred)):
                raise ModelFitError("ETS produced non-finite forecasts.")
            return np.maximum(pred, 0.0)
    except ModelFitError:
        raise
    except Exception as exc:
        raise ModelFitError(
            f"ETS(trend={trend}, damped={damped_trend}) failed: {type(exc).__name__}: {exc}"
        ) from exc


def fit_predict_theta(history: pd.Series, horizon: int) -> np.ndarray:
    """Fit Assimakopoulos-Nikolopoulos Theta model on non-seasonal annual series."""
    vals = history.to_numpy(dtype=float)
    if len(vals) < 5 or np.all(vals == vals[0]):
        return np.maximum(np.repeat(vals[-1], horizon), 0.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ThetaModel(vals, period=1, deseasonalize=False)
            res = model.fit()
            pred = np.asarray(res.forecast(horizon), dtype=float)
            if not np.all(np.isfinite(pred)):
                raise ModelFitError("ThetaModel produced non-finite forecasts.")
            return np.maximum(pred, 0.0)
    except ModelFitError:
        raise
    except Exception as exc:
        raise ModelFitError(f"ThetaModel failed: {type(exc).__name__}: {exc}") from exc


def _panel_formula_suffix(df: pd.DataFrame) -> str:
    """Build categorical control suffix using only varying grouping columns."""
    terms: list[str] = []
    if df["location"].nunique() > 1:
        terms.append("C(location)")
    if df["sex"].nunique() > 1:
        terms.append("C(sex)")
    return (" + " + " + ".join(terms)) if terms else ""


def fit_predict_poisson_glm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Fit log-linear Poisson Rate GLM: log(ASR) ~ t + clean_fuel_share [+ loc] [+ sex]."""
    df_fit = train_df.copy()
    df_pred = test_df.copy()
    df_fit["t"] = df_fit["year"] - 1990
    df_pred["t"] = df_pred["year"] - 1990
    suffix = _panel_formula_suffix(df_fit)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fmla = f"val ~ t + clean_fuel_share{suffix}"
            res = sm.GLM.from_formula(
                fmla,
                data=df_fit,
                family=sm.families.Poisson(link=sm.families.links.Log()),
            ).fit()
            preds = np.asarray(res.predict(df_pred), dtype=float)
            if not np.all(np.isfinite(preds)):
                raise ModelFitError("Poisson GLM produced non-finite forecasts.")
            return np.maximum(preds, 0.0)
    except ModelFitError:
        raise
    except Exception as exc:
        raise ModelFitError(f"Poisson GLM failed: {type(exc).__name__}: {exc}") from exc


def fit_predict_log_ols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, Any]:
    """Fit smooth parametric log-OLS panel: log(ASR) ~ t + clean_fuel_share [+ loc] [+ sex]."""
    df_fit = train_df.copy()
    df_pred = test_df.copy()
    df_fit["t"] = df_fit["year"] - 1990
    df_pred["t"] = df_pred["year"] - 1990
    df_fit["log_val"] = np.log(np.maximum(df_fit["val"], 1e-6))
    suffix = _panel_formula_suffix(df_fit)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fmla = f"log_val ~ t + clean_fuel_share{suffix}"
            res = sm.OLS.from_formula(fmla, data=df_fit).fit()
            preds = np.exp(np.asarray(res.predict(df_pred), dtype=float))
            if not np.all(np.isfinite(preds)):
                raise ModelFitError("Log-OLS produced non-finite forecasts.")
            return np.maximum(preds, 0.0), res
    except ModelFitError:
        raise
    except Exception as exc:
        raise ModelFitError(f"Log-OLS failed: {type(exc).__name__}: {exc}") from exc


def fit_predict_log_solid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, Any]:
    """Fit saturating solid-fuel log-log model: log(ASR) ~ t + log(1 - cf) [+ loc] [+ sex]."""
    df_fit = train_df.copy()
    df_pred = test_df.copy()
    df_fit["t"] = df_fit["year"] - 1990
    df_pred["t"] = df_pred["year"] - 1990
    df_fit["log_val"] = np.log(np.maximum(df_fit["val"], 1e-6))
    df_fit["log_solid"] = np.log(np.clip(1.0 - df_fit["clean_fuel_share"], 0.005, 1.0))
    df_pred["log_solid"] = np.log(np.clip(1.0 - df_pred["clean_fuel_share"], 0.005, 1.0))
    suffix = _panel_formula_suffix(df_fit)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fmla = f"log_val ~ t + log_solid{suffix}"
            res = sm.OLS.from_formula(fmla, data=df_fit).fit()
            preds = np.exp(np.asarray(res.predict(df_pred), dtype=float))
            if not np.all(np.isfinite(preds)):
                raise ModelFitError("Log-Solid OLS produced non-finite forecasts.")
            return np.maximum(preds, 0.0), res
    except ModelFitError:
        raise
    except Exception as exc:
        raise ModelFitError(f"Log-Solid OLS failed: {type(exc).__name__}: {exc}") from exc


def _prepare_gbdt_features(
    df: pd.DataFrame,
    loc_map: dict[str, int],
) -> pd.DataFrame:
    feature_cols = ["year", "clean_fuel_share", "loc_code", "sex_code"]
    sex_map = {"female": 0, "male": 1, "both": 2}
    x = df.copy()
    x["loc_code"] = x["location"].map(loc_map).fillna(0).astype(int)
    x["sex_code"] = x["sex"].map(sex_map).fillna(2).astype(int)
    return x[feature_cols]


def fit_predict_hist_gbdt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
    monotonic_clean_fuel: bool = False,
) -> tuple[np.ndarray, HistGradientBoostingRegressor]:
    """Fit HistGradientBoostingRegressor on panel features (optionally monotonic in cf)."""
    loc_map = {loc: idx for idx, loc in enumerate(sorted(train_df["location"].unique()))}
    x_train = _prepare_gbdt_features(train_df, loc_map)
    y_train = np.log(np.maximum(train_df["val"].to_numpy(dtype=float), 1.0))
    x_test = _prepare_gbdt_features(test_df, loc_map)

    monotonic_cst = [0, -1, 0, 0] if monotonic_clean_fuel else None
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_depth=4,
        monotonic_cst=monotonic_cst,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_pred_log = model.predict(x_test)
    preds = np.exp(y_pred_log)
    return preds, model
