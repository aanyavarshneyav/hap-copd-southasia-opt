"""Forecasting models for AEGIS-GBD Stage 2.

Implements statistical (ARIMA, ETS, Poisson Rate GLM) and ML panel models
(HistGradientBoostingRegressor) for projecting HAP-attributable COPD ASRs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_predict_arima(
    history: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 0),
) -> np.ndarray:
    """Fit ARIMA on historical series and forecast forward."""
    vals = history.to_numpy(dtype=float)
    if len(vals) < 5 or np.all(vals == vals[0]):
        return np.repeat(vals[-1], horizon)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(vals, order=order)
            res = model.fit()
            pred = res.forecast(steps=horizon)
            # Clip negative rates
            return np.maximum(np.asarray(pred, dtype=float), 0.0)
    except Exception:
        # Fallback to naive random walk with drift
        drift = (vals[-1] - vals[0]) / max(len(vals) - 1, 1)
        steps = np.arange(1, horizon + 1)
        return np.maximum(vals[-1] + drift * steps, 0.0)


def fit_predict_ets(
    history: pd.Series,
    horizon: int,
    trend: str = "add",
    damped_trend: bool = True,
) -> np.ndarray:
    """Fit Holt-Winters Exponential Smoothing on historical series and forecast."""
    vals = history.to_numpy(dtype=float)
    if len(vals) < 5 or np.all(vals == vals[0]):
        return np.repeat(vals[-1], horizon)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                vals,
                trend=trend,
                damped_trend=damped_trend,
                initialization_method="estimated",
            )
            res = model.fit(disp=False)
            pred = res.forecast(horizon)
            return np.maximum(np.asarray(pred, dtype=float), 0.0)
    except Exception:
        return fit_predict_arima(history, horizon)


def fit_predict_poisson_glm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Fit log-linear Poisson Rate GLM: log(ASR) ~ t + clean_fuel_share + loc + sex."""
    df_fit = train_df.copy()
    df_pred = test_df.copy()

    # Model: log(val) = beta_0 + beta_yr * t + beta_cf * clean_fuel_share + loc/sex
    df_fit["t"] = df_fit["year"] - 1990
    df_pred["t"] = df_pred["year"] - 1990

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit using sm.GLM with Poisson family
            fmla = "val ~ t + clean_fuel_share + C(location) + C(sex)"
            res = sm.GLM.from_formula(
                fmla,
                data=df_fit,
                family=sm.families.Poisson(link=sm.families.links.Log()),
            ).fit()
            preds = res.predict(df_pred)
            return np.maximum(np.asarray(preds, dtype=float), 0.0)
    except Exception:
        # Robust fallback using log-OLS
        df_fit["log_val"] = np.log(np.maximum(df_fit["val"], 1.0))
        fmla = "log_val ~ t + clean_fuel_share + C(location) + C(sex)"
        res = sm.OLS.from_formula(fmla, data=df_fit).fit()
        log_preds = res.predict(df_pred)
        return np.maximum(np.exp(np.asarray(log_preds, dtype=float)), 0.0)


def fit_predict_hist_gbdt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[np.ndarray, HistGradientBoostingRegressor]:
    """Fit HistGradientBoostingRegressor on panel features and predict."""
    feature_cols = ["year", "clean_fuel_share", "loc_code", "sex_code"]

    loc_map = {loc: idx for idx, loc in enumerate(sorted(train_df["location"].unique()))}
    sex_map = {"female": 0, "male": 1, "both": 2}

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["loc_code"] = x["location"].map(loc_map).fillna(0).astype(int)
        x["sex_code"] = x["sex"].map(sex_map).fillna(2).astype(int)
        return x[feature_cols]

    x_train = _prepare(train_df)
    y_train = np.log(np.maximum(train_df["val"].to_numpy(dtype=float), 1.0))

    x_test = _prepare(test_df)

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_pred_log = model.predict(x_test)
    preds = np.exp(y_pred_log)
    return preds, model
