"""Side-by-side HistGradientBoosting vs Smooth Log-OLS Counterfactual Audit & Surrogate Builder.

Compares existing HistGradientBoosting (production & monotonic) against smooth parametric
Log-OLS (pooled and unit-interacted) and Poisson Rate GLM across identical 4-unit rolling
CV folds, response curves, monotonicity, extrapolation behaviour, and analytical uncertainty.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.contracts import load_config
from src.forecast.b12_benchmark import TARGET_UNITS, reconstruct_historical_panel_b12
from src.forecast.models import (
    fit_predict_hist_gbdt,
    fit_predict_log_ols,
    fit_predict_naive,
    fit_predict_poisson_glm,
)


def fit_pooled_log_ols_surrogate(panel: pd.DataFrame) -> dict[str, Any]:
    """Fit pooled log-linear OLS counterfactual model on 4 intervention units (both sex)."""
    sub = panel[
        (panel["location"].isin(TARGET_UNITS))
        & (panel["sex"] == "both")
        & (panel["measure"] == "DALYs")
        & (panel["metric"] == "Rate_per_100k")
        & (panel["year"] <= 2023)
    ].copy()
    sub["t"] = sub["year"] - 1990
    sub["log_val"] = np.log(np.maximum(sub["val"], 1e-6))

    res = sm.OLS.from_formula("log_val ~ t + clean_fuel_share + C(location)", data=sub).fit()
    slope_cf = float(res.params["clean_fuel_share"])
    slope_se = float(res.bse["clean_fuel_share"])
    slope_pval = float(res.pvalues["clean_fuel_share"])
    ci = np.asarray(res.conf_int().loc["clean_fuel_share"], dtype=float)
    return {
        "model": res,
        "sub": sub,
        "slope_clean_fuel": round(slope_cf, 4),
        "slope_se": round(slope_se, 4),
        "slope_ci95_low": round(float(ci[0]), 4),
        "slope_ci95_high": round(float(ci[1]), 4),
        "slope_pvalue": slope_pval,
        "r_squared": round(float(res.rsquared), 4),
        "adj_r_squared": round(float(res.rsquared_adj), 4),
        "residual_log_sd": round(float(np.sqrt(res.mse_resid)), 4),
        "n_obs": len(sub),
    }


def evaluate_identical_folds_4unit(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare all candidate counterfactual models on identical 4-unit folds & horizons."""
    sub = panel[
        (panel["location"].isin(TARGET_UNITS))
        & (panel["sex"] == "both")
        & (panel["measure"] == "DALYs")
        & (panel["metric"] == "Rate_per_100k")
        & (panel["year"] <= 2023)
    ].copy()
    sub["t"] = sub["year"] - 1990
    sub["log_val"] = np.log(np.maximum(sub["val"], 1e-6))

    rows: list[dict[str, Any]] = []
    for horizon in (1, 3, 5, 10):
        for origin in (2008, 2013, 2018):
            if origin + horizon > 2023:
                continue
            tr = sub[sub["year"] <= origin].copy()
            te = sub[(sub["year"] >= origin + 1) & (sub["year"] <= origin + horizon)].copy()

            p_g0, _ = fit_predict_hist_gbdt(tr, te, monotonic_clean_fuel=False, random_state=42)
            p_gm, _ = fit_predict_hist_gbdt(tr, te, monotonic_clean_fuel=True, random_state=42)
            p_lo_pool, _ = fit_predict_log_ols(tr, te)
            p_pois = fit_predict_poisson_glm(tr, te)

            m_int = sm.OLS.from_formula(
                "log_val ~ t + clean_fuel_share * C(location)", data=tr
            ).fit()
            p_lo_int = np.maximum(np.exp(m_int.predict(te).to_numpy(dtype=float)), 0.0)

            for loc in TARGET_UNITS:
                msk = (te["location"] == loc).to_numpy()
                y_true = te.loc[msk, "val"].to_numpy(dtype=float)
                hist = tr[tr["location"] == loc].sort_values("year")["val"].to_numpy(dtype=float)
                p_nv = fit_predict_naive(pd.Series(hist), horizon=horizon)
                nv_mae = float(np.mean(np.abs(y_true - p_nv)))

                preds_map = {
                    "naive_baseline": p_nv,
                    "hist_gbdt_panel": p_g0[msk],
                    "hist_gbdt_monotonic": p_gm[msk],
                    "log_ols_smooth": p_lo_pool[msk],
                    "log_ols_interacted": p_lo_int[msk],
                    "poisson_rate_glm": p_pois[msk],
                }
                for m_name, y_pred in preds_map.items():
                    mae = float(np.mean(np.abs(y_true - y_pred)))
                    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                    rows.append(
                        {
                            "horizon": horizon,
                            "origin_year": origin,
                            "location": loc,
                            "model": m_name,
                            "mae": round(mae, 4),
                            "rmse": round(rmse, 4),
                            "naive_mae": round(nv_mae, 4),
                            "relative_mae": round(mae / nv_mae, 4)
                            if nv_mae > 1e-9
                            else float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def build_response_curve_table(panel: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Compare clean-fuel response curves, monotonicity & slopes for all 4 units."""
    fit_info = fit_pooled_log_ols_surrogate(panel)
    sub = fit_info["sub"]
    m_pool = fit_info["model"]
    m_int = sm.OLS.from_formula("log_val ~ t + clean_fuel_share * C(location)", data=sub).fit()

    fc_rec = pd.read_csv(root / "data" / "processed" / "forecasts_stage3_reconciled.csv")
    b40 = (
        fc_rec[
            (fc_rec["year"] == 2040)
            & (fc_rec["scenario"] == "bau")
            & (fc_rec["measure"] == "DALYs")
            & (fc_rec["metric"] == "Rate_per_100k")
            & (fc_rec["sex"] == "both")
        ]
        .set_index("location")
        .to_dict(orient="index")
    )

    surr_gbdt_path = root / "data" / "processed" / "exposure_response_surrogate_histgbdt.json"
    if not surr_gbdt_path.exists():
        surr_gbdt_path = root / "data" / "processed" / "exposure_response_surrogate.json"
    surr_gbdt = json.loads(surr_gbdt_path.read_text(encoding="utf-8"))

    cf_grid = [0.25, 0.50, 0.75, 0.90, 0.95, 0.985, 0.99, 1.00]
    rows: list[dict[str, Any]] = []

    for unit in TARGET_UNITS:
        cf_bau = float(b40[unit]["clean_fuel_share"])
        bau_asr = float(b40[unit]["val"])

        # HistGBDT full-1990-2023 direct predictions across grid
        grid_df = pd.DataFrame(
            {
                "location": [unit] * len(cf_grid),
                "year": [2040] * len(cf_grid),
                "sex": ["both"] * len(cf_grid),
                "clean_fuel_share": cf_grid,
                "t": [2040 - 1990] * len(cf_grid),
            }
        )
        p_g0, _ = fit_predict_hist_gbdt(sub, grid_df, monotonic_clean_fuel=False, random_state=42)
        p_gm, _ = fit_predict_hist_gbdt(sub, grid_df, monotonic_clean_fuel=True, random_state=42)
        p_pool = np.exp(m_pool.predict(grid_df).to_numpy(dtype=float))
        p_int = np.exp(m_int.predict(grid_df).to_numpy(dtype=float))

        # Exact Stage 3 surrogate exponential trajectory anchored on BAU 2040
        g_slope = float(surr_gbdt["units"][unit]["slope_clean_fuel"])
        l_slope = fit_info["slope_clean_fuel"]

        for idx, cf in enumerate(cf_grid):
            delta_c = cf - cf_bau
            rows.append(
                {
                    "unit": unit,
                    "cf_share": cf,
                    "cf_bau_2040": cf_bau,
                    "stage2_bau_asr_2040": bau_asr,
                    "hist_gbdt_direct_asr": round(float(p_g0[idx]), 2),
                    "hist_gbdt_mono_asr": round(float(p_gm[idx]), 2),
                    "log_ols_pooled_direct_asr": round(float(p_pool[idx]), 2),
                    "log_ols_interacted_direct_asr": round(float(p_int[idx]), 2),
                    "surrogate_hist_gbdt_asr": round(bau_asr * float(np.exp(g_slope * delta_c)), 2),
                    "surrogate_log_ols_asr": round(bau_asr * float(np.exp(l_slope * delta_c)), 2),
                    "hist_gbdt_prod_slope": g_slope,
                    "log_ols_pooled_slope": l_slope,
                }
            )
    return pd.DataFrame(rows)


def plot_surrogate_comparison_b13(
    curve_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Render side-by-side surrogate response curves (4 units) & identical-fold CV MAE."""
    fig, axes = plt.subplots(
        1, 5, figsize=(19.5, 4.3), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]}
    )

    for idx, unit in enumerate(TARGET_UNITS):
        ax = axes[idx]
        sub = curve_df[curve_df["unit"] == unit].sort_values("cf_share")
        cf = sub["cf_share"].to_numpy()
        ax.plot(
            cf,
            sub["surrogate_hist_gbdt_asr"],
            color="#d97706",
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=2.0,
            label="HistGBDT Surrogate" if idx == 0 else None,
        )
        ax.plot(
            cf,
            sub["surrogate_log_ols_asr"],
            color="#059669",
            linestyle="-",
            marker="s",
            markersize=4,
            linewidth=2.2,
            label="Smooth Log-OLS (-2.1933)" if idx == 0 else None,
        )
        cf_b = float(sub["cf_bau_2040"].iloc[0])
        asr_b = float(sub["stage2_bau_asr_2040"].iloc[0])
        ax.scatter(
            [cf_b],
            [asr_b],
            color="#0f172a",
            zorder=5,
            s=45,
            label="2040 BAU Anchor" if idx == 0 else None,
        )
        ax.set_title(f"{unit}\n(BAU cf={cf_b:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Clean Fuel Share")
        if idx == 0:
            ax.set_ylabel("Projected 2040 DALY ASR (/100k)")
            ax.legend(fontsize=7.8, loc="upper right", frameon=True)
        ax.grid(alpha=0.25)

    # Panel 5: Identical-fold 4-unit h=5 CV MAE comparison
    ax_cv = axes[4]
    h5 = (
        cv_df[cv_df["horizon"] == 5]
        .groupby("model")["mae"]
        .mean()
        .reindex(
            [
                "log_ols_interacted",
                "poisson_rate_glm",
                "naive_baseline",
                "log_ols_smooth",
                "hist_gbdt_monotonic",
                "hist_gbdt_panel",
            ]
        )
    )
    cols = ["#0284c7", "#64748b", "#94a3b8", "#059669", "#f59e0b", "#dc2626"]
    ax_cv.barh(list(h5.index)[::-1], list(h5.values)[::-1], color=cols[::-1])
    ax_cv.set_title("4-Unit CV MAE (h=5)\nIdentical Folds", fontsize=10, fontweight="bold")
    ax_cv.set_xlabel("Mean MAE (DALYs/100k)")
    ax_cv.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Side-by-Side Comparison: HistGradientBoosting vs Smooth Log-OLS Counterfactual Surrogate",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_log_ols_surrogate_contract(
    panel: pd.DataFrame,
    out_path: Path,
    base_fc_path: Path,
) -> dict[str, Any]:
    """Write validated Contract 3 JSON using pooled smooth Log-OLS slope & analytical SE."""
    fit_info = fit_pooled_log_ols_surrogate(panel)
    fc_rec = pd.read_csv(base_fc_path)
    bau40 = (
        fc_rec[
            (fc_rec["year"] == 2040)
            & (fc_rec["scenario"] == "bau")
            & (fc_rec["measure"] == "DALYs")
            & (fc_rec["metric"] == "Rate_per_100k")
            & (fc_rec["sex"] == "both")
        ]
        .set_index("location")["clean_fuel_share"]
        .to_dict()
    )

    units_dict: dict[str, Any] = {}
    for u in TARGET_UNITS:
        units_dict[u] = {
            "slope_clean_fuel": fit_info["slope_clean_fuel"],
            "baseline_clean_fuel": round(float(bau40[u]), 4),
            "sex_elasticity_ratio_f_to_m": 1.05,
            "uncertainty_sd": fit_info["slope_se"],
        }

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "is_mock": False,
        "data_version": "GBD_2023_WPP_2024",
        "functional_form": "log_linear",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_model": "log_ols_smooth",
        "calibration_note": (
            f"Pooled smooth Log-OLS (log_val ~ t + clean_fuel_share + C(location), "
            f"N={fit_info['n_obs']}, R2={fit_info['r_squared']}, "
            f"slope={fit_info['slope_clean_fuel']} +/- {fit_info['slope_se']} SE, "
            f"p={fit_info['slope_pvalue']:.2e}); replaces HistGradientBoosting artifacts."
        ),
        "units": units_dict,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_comparison_only(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Run non-destructive comparison, saving CSV tables & figure without overwriting production."""
    cfg = load_config(config_path)
    root = Path(config_path).resolve().parents[1]
    tables_dir = root / cfg["paths"]["outputs_tables_dir"]
    figures_dir = root / cfg["paths"]["outputs_figures_dir"]

    panel = reconstruct_historical_panel_b12(root)
    fit_info = fit_pooled_log_ols_surrogate(panel)
    cv_df = evaluate_identical_folds_4unit(panel)
    curve_df = build_response_curve_table(panel, root)

    cv_path = tables_dir / "s2_counterfactual_cv_identical_b13.csv"
    curve_path = tables_dir / "s2_surrogate_comparison_b13.csv"
    fig_path = figures_dir / "s2_surrogate_comparison_b13.png"

    cv_df.to_csv(cv_path, index=False)
    curve_df.to_csv(curve_path, index=False)
    plot_surrogate_comparison_b13(curve_df, cv_df, fig_path)

    return {
        "slope_clean_fuel": fit_info["slope_clean_fuel"],
        "slope_se": fit_info["slope_se"],
        "r_squared": fit_info["r_squared"],
        "cv_rows": len(cv_df),
        "curve_rows": len(curve_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side HistGBDT vs Smooth Log-OLS audit")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--replace-surrogate", action="store_true")
    args = parser.parse_args()

    res = run_comparison_only(args.config)
    print("Comparison complete:", res)

    if args.replace_surrogate:
        root = Path(args.config).resolve().parents[1]
        cfg = load_config(args.config)
        processed = root / cfg["paths"]["processed_dir"]
        panel = reconstruct_historical_panel_b12(root)
        payload = write_log_ols_surrogate_contract(
            panel,
            processed / "exposure_response_surrogate.json",
            root / cfg["interfaces"]["forecast_contract_file"],
        )
        print(
            "Replaced exposure_response_surrogate.json with:",
            payload["source_model"],
            payload["units"]["IND_urban"],
        )


if __name__ == "__main__":
    main()
