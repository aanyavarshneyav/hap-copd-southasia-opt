"""B-12 Multi-Horizon Forecasting CV, Failure Audit & Counterfactual Analysis.

Reconstructs the 1990-2023 South Asia 9-location HAP-COPD panel cleanly from
Stage 1 outputs + clean-fuel inputs (or interim parquet files when present),
executes multi-horizon CV (h = 1, 3, 5, 10) with explicit ETS & failure logging,
compares smooth parametric vs tree-based counterfactual models (including
IND_urban high-coverage behaviour), and exports labelled B-12 tables & figures
without altering any original Stage 2 exports.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.contracts import load_config, load_stage3_contracts
from src.data.ingest import URBAN_SHARE_BOUNDS
from src.forecast.cv import run_multihorizon_cv_b12, run_rolling_origin_cv
from src.forecast.models import (
    fit_predict_hist_gbdt,
    fit_predict_log_ols,
    fit_predict_log_solid,
)
from src.optimise.evaluator import Stage3Evaluator
from src.optimise.final_real_suite import bundle_final_real_outputs
from src.optimise.nsga2 import run_nsga2_multi_seed, select_representative_plans

TARGET_UNITS = ("IND_urban", "IND_rural", "NPL_urban", "NPL_rural")
ALL_9_LOCS = (
    "Bangladesh",
    "Bhutan",
    "India",
    "Nepal",
    "Pakistan",
    "IND_urban",
    "IND_rural",
    "NPL_urban",
    "NPL_rural",
)


def _get_linear_urban_share(country: str, year: int) -> float:
    lo, hi = URBAN_SHARE_BOUNDS[country]
    frac = np.clip((year - 1990) / (2023 - 1990), 0.0, 1.0)
    return float(lo + frac * (hi - lo))


def reconstruct_historical_panel_b12(root: Path) -> pd.DataFrame:
    """Load interim historical panel or reconstruct 9-location 1990-2023 panel cleanly."""
    interim_hap = root / "data" / "interim" / "historical_burden.parquet"
    interim_cc = root / "data" / "interim" / "clean_cooking.parquet"
    if interim_hap.exists() and interim_cc.exists():
        df_hap = pd.read_parquet(interim_hap)
        df_cc = pd.read_parquet(interim_cc)
        cc_map: dict[tuple[str, int], float] = {}
        for _, r in df_cc.iterrows():
            c, y = str(r["country"]), int(r["year"])
            cc_map[(c, y)] = float(r["clean_fuel_share_national"])
            if c == "India":
                cc_map[("IND_urban", y)] = float(r["clean_fuel_share_urban"])
                cc_map[("IND_rural", y)] = float(r["clean_fuel_share_rural"])
            elif c == "Nepal":
                cc_map[("NPL_urban", y)] = float(r["clean_fuel_share_urban"])
                cc_map[("NPL_rural", y)] = float(r["clean_fuel_share_rural"])
        out = df_hap.copy()
        out["clean_fuel_share"] = [
            cc_map.get((str(loc), int(yr)), 0.5)
            for loc, yr in zip(out["location"], out["year"], strict=True)
        ]
        return out

    # Reconstruct strictly from committed Stage 1 outputs + clean_fuel_access.csv
    s1 = pd.read_csv(root / "outputs" / "tables" / "s1_attributable_share.csv")
    s1 = s1[(s1["measure"] == "DALYs") & (s1["year"] <= 2023)].copy()

    cfa = pd.read_csv(root / "research" / "person_b_inputs" / "clean_fuel_access.csv")
    cfa = cfa[
        (cfa["source"] == "World Bank WDI (WHO/ESMAP)")
        & (cfa["fuel_category"] == "clean_fuels_total")
        & (cfa["year"] <= 2023)
    ].copy()
    cfa["cf_share"] = cfa["val"].astype(float) / 100.0

    # Extrapolate 2000 values backward cleanly to 1990 (matching ingest pre-2000 fill)
    cf_lookup: dict[tuple[str, int], float] = {}
    for uid, grp in cfa.groupby("unit_id"):
        s_cf = grp.set_index("year")["cf_share"].sort_index()
        s_full = s_cf.reindex(range(1990, 2024)).bfill().ffill()
        for yr, v in s_full.items():
            key_loc = (
                "India" if uid == "IND_total" else ("Nepal" if uid == "NPL_total" else str(uid))
            )
            cf_lookup[(key_loc, int(yr))] = float(v)

    rows: list[dict[str, Any]] = []
    # National series (all 5 countries, sex='both')
    for _, r in s1.iterrows():
        c = str(r["location"])
        y = int(r["year"])
        asr = float(r["hap_copd_asr"])
        rows.append(
            {
                "location": c,
                "year": y,
                "sex": "both",
                "measure": "DALYs",
                "metric": "Rate_per_100k",
                "age": "Age-standardized",
                "val": asr,
                "clean_fuel_share": cf_lookup.get((c, y), 0.5),
            }
        )

    # Apportioned subnational urban/rural units for India & Nepal (exact ingest.py formula)
    for country, prefix in (("India", "IND"), ("Nepal", "NPL")):
        c_df = s1[s1["location"] == country].set_index("year")["hap_copd_asr"]
        for y in range(1990, 2024):
            nat_asr = float(c_df.loc[y])
            u_share = _get_linear_urban_share(country, y)
            r_share = 1.0 - u_share
            cf_u = cf_lookup[(f"{prefix}_urban", y)]
            cf_r = cf_lookup[(f"{prefix}_rural", y)]
            s_u = max(1.0 - cf_u, 0.01)
            s_r = max(1.0 - cf_r, 0.01)
            denom = u_share * s_u + r_share * s_r
            w_u = (u_share * s_u) / denom
            w_r = (r_share * s_r) / denom
            rows.append(
                {
                    "location": f"{prefix}_urban",
                    "year": y,
                    "sex": "both",
                    "measure": "DALYs",
                    "metric": "Rate_per_100k",
                    "age": "Age-standardized",
                    "val": nat_asr * (w_u / u_share),
                    "clean_fuel_share": cf_u,
                }
            )
            rows.append(
                {
                    "location": f"{prefix}_rural",
                    "year": y,
                    "sex": "both",
                    "measure": "DALYs",
                    "metric": "Rate_per_100k",
                    "age": "Age-standardized",
                    "val": nat_asr * (w_r / r_share),
                    "clean_fuel_share": cf_r,
                }
            )

    return pd.DataFrame(rows)


def run_counterfactual_comparison_b12(
    df_panel: pd.DataFrame,
    root: Path,
) -> pd.DataFrame:
    """Compare counterfactual response shapes & 4-unit CV error (Task 5)."""
    sub = df_panel[
        (df_panel["location"].isin(TARGET_UNITS))
        & (df_panel["sex"] == "both")
        & (df_panel["measure"] == "DALYs")
        & (df_panel["metric"] == "Rate_per_100k")
        & (df_panel["year"] <= 2023)
    ].copy()

    fc_rec = pd.read_csv(root / "data" / "processed" / "forecasts_stage3_reconciled.csv")
    fc_2040_both = fc_rec[
        (fc_rec["year"] == 2040)
        & (fc_rec["scenario"] == "bau")
        & (fc_rec["measure"] == "DALYs")
        & (fc_rec["metric"] == "Number")
        & (fc_rec["sex"] == "both")
    ].set_index("location")
    pop_2040 = fc_2040_both["population"].to_dict()
    cf_bau_2040 = {str(k): float(v) for k, v in fc_2040_both["clean_fuel_share"].items()}

    # Evaluate 4-unit h=5 CV on sub
    cv_errors: dict[str, list[float]] = {
        "hist_gbdt_unconstrained": [],
        "hist_gbdt_monotonic": [],
        "log_ols_smooth": [],
        "log_solid_saturating": [],
    }
    for origin in (2008, 2013, 2018):
        tr = sub[sub["year"] <= origin]
        te = sub[(sub["year"] >= origin + 1) & (sub["year"] <= origin + 5)]
        y_true = te["val"].to_numpy(dtype=float)
        p_g0, _ = fit_predict_hist_gbdt(tr, te, monotonic_clean_fuel=False)
        p_gm, _ = fit_predict_hist_gbdt(tr, te, monotonic_clean_fuel=True)
        p_lo, _ = fit_predict_log_ols(tr, te)
        p_ls, _ = fit_predict_log_solid(tr, te)
        for k, p in (
            ("hist_gbdt_unconstrained", p_g0),
            ("hist_gbdt_monotonic", p_gm),
            ("log_ols_smooth", p_lo),
            ("log_solid_saturating", p_ls),
        ):
            cv_errors[k].append(float(np.mean(np.abs(y_true - p))))

    # Fit on full 1990-2023 4-unit history and evaluate 2040 counterfactual curve
    grid = [0.50, 0.70, 0.90, 0.95, 0.98, 0.99, 1.00]
    rows: list[dict[str, Any]] = []
    model_specs = [
        (
            "hist_gbdt_unconstrained",
            lambda tr, te: fit_predict_hist_gbdt(tr, te, monotonic_clean_fuel=False),
        ),
        (
            "hist_gbdt_monotonic",
            lambda tr, te: fit_predict_hist_gbdt(tr, te, monotonic_clean_fuel=True),
        ),
        ("log_ols_smooth", fit_predict_log_ols),
        ("log_solid_saturating", fit_predict_log_solid),
    ]

    for mname, mfn in model_specs:
        cv_mae = float(np.mean(cv_errors[mname]))
        for unit in TARGET_UNITS:
            grid_df = pd.DataFrame(
                {
                    "location": [unit] * len(grid),
                    "year": [2040] * len(grid),
                    "sex": ["both"] * len(grid),
                    "clean_fuel_share": grid,
                }
            )
            preds_grid, _ = mfn(sub, grid_df)

            bau_df = pd.DataFrame(
                {
                    "location": [unit],
                    "year": [2040],
                    "sex": ["both"],
                    "clean_fuel_share": [cf_bau_2040[unit]],
                }
            )
            bau_asr = float(mfn(sub, bau_df)[0][0])
            bau_dalys = bau_asr * float(pop_2040[unit]) / 100_000.0

            p90 = float(preds_grid[grid.index(0.90)])
            p98 = float(preds_grid[grid.index(0.98)])
            p99 = float(preds_grid[grid.index(0.99)])
            p100 = float(preds_grid[grid.index(1.00)])
            slope_985_100 = (p100 - p98) / 0.02
            rows.append(
                {
                    "model": mname,
                    "unit": unit,
                    "cv_4unit_h5_mae": round(cv_mae, 2),
                    "bau_cf_2040": round(cf_bau_2040[unit], 4),
                    "bau_asr_2040": round(bau_asr, 2),
                    "bau_dalys_2040": round(bau_dalys, 2),
                    "asr_cf_0_50": round(float(preds_grid[0]), 2),
                    "asr_cf_0_90": round(p90, 2),
                    "asr_cf_0_98": round(p98, 2),
                    "asr_cf_0_99": round(p99, 2),
                    "asr_cf_1_00": round(p100, 2),
                    "delta_asr_0_90_to_1_00": round(p100 - p90, 2),
                    "local_slope_0_98_to_1_00": round(slope_985_100, 2),
                    "plateaus_above_0_98": abs(p100 - p98) < 1e-6,
                }
            )

    return pd.DataFrame(rows)


def plot_multihorizon_cv_b12(summary_df: pd.DataFrame, out_path: Path) -> Path:
    """Plot B-12 multi-horizon CV MAE and horizon-matched Skill Score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    order = [
        "ets_undamped",
        "ets_damped",
        "arima",
        "drift",
        "theta",
        "naive_baseline",
        "log_ols_panel",
        "poisson_rate_glm",
        "hist_gbdt_panel",
    ]
    present = [m for m in order if m in set(summary_df["model"])]

    # Panel A: Grouped MAE across horizons h = 1, 3, 5, 10
    piv_mae = summary_df.pivot(index="model", columns="horizon", values="mean_mae").loc[present]
    piv_mae.plot(kind="bar", ax=axes[0], colormap="viridis", width=0.78)
    axes[0].set_title(
        "Multi-Horizon Out-of-Sample MAE (DALYs/100k)", fontsize=11, fontweight="bold"
    )
    axes[0].set_ylabel("Mean CV MAE (Lower is Better)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="Horizon (yr)", frameon=True)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel B: h = 5 Pooled Skill Score = 1 - relative_MAE vs Naive
    h5 = summary_df[summary_df["horizon"] == 5].set_index("model").loc[present]
    colors = [
        "#059669" if v > 0 else ("#64748b" if v == 0 else "#dc2626")
        for v in h5["pooled_skill_score"]
    ]
    axes[1].barh(present[::-1], h5["pooled_skill_score"].iloc[::-1], color=colors[::-1])
    axes[1].axvline(
        0.0, color="#0f172a", linestyle="--", linewidth=1.1, label="Naive h=5 (Skill = 0)"
    )
    axes[1].set_title(
        "5-Year Horizon Skill Score (1 - Model_MAE / Naive_MAE)", fontsize=11, fontweight="bold"
    )
    axes[1].set_xlabel("Pooled Skill Score (Higher is Better)")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def run_stage3_slope_sensitivity_b12(
    config: dict[str, Any],
    root: Path,
    ind_urban_log_ols_slope: float = -2.1933,
) -> pd.DataFrame:
    """Compare Stage 3 NSGA-II outcomes with vs without IND_urban 0.0 surrogate plateau."""
    base_contracts = load_stage3_contracts(config, base_dir=root)
    mod_surrogate = copy.deepcopy(base_contracts.surrogate)
    mod_surrogate["units"]["IND_urban"]["slope_clean_fuel"] = ind_urban_log_ols_slope
    mod_contracts = replace(base_contracts, surrogate=mod_surrogate)

    rows: list[dict[str, Any]] = []
    for label, ctr in (
        ("reconciled_baseline_ind_urban_slope_0", base_contracts),
        ("b12_counterfactual_ind_urban_slope_-2.1933", mod_contracts),
    ):
        ev = Stage3Evaluator(config=config, contracts=ctr)
        ms = run_nsga2_multi_seed(ev, config)
        sel_df, _ = select_representative_plans(ms.combined_X, ms.combined_evaluation, ev)
        for _, r in sel_df.iterrows():
            rows.append(
                {
                    "variant": label,
                    "ind_urban_slope": ctr.surrogate["units"]["IND_urban"]["slope_clean_fuel"],
                    "plan_name": str(r["plan_name"]),
                    "bau_dalys_2040": round(float(ms.bau_evaluation.dalys_2040_mean), 2),
                    "dalys_2040_mean": round(float(r["dalys_2040_mean"]), 2),
                    "dalys_averted_mean": round(float(r["dalys_averted_mean"]), 2),
                    "total_cost_usd_millions": round(float(r["total_cost_usd_millions"]), 2),
                    "equity_gap_asr": round(float(r["equity_gap_asr"]), 4),
                    "equity_gap_dalys_abs": round(float(r["equity_gap_dalys_abs"]), 2),
                }
            )
    return pd.DataFrame(rows)


def run_b12_suite(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Execute complete B-12 forecasting benchmark, failure audit, and counterfactual check."""
    cfg = load_config(config_path)
    root = Path(config_path).resolve().parents[1]
    tables_dir = root / Path(cfg["paths"]["outputs_tables_dir"])
    figures_dir = root / Path(cfg["paths"]["outputs_figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = reconstruct_historical_panel_b12(root)

    # Run clean non-redundant (sex='both', 9 real locations) multi-horizon CV
    metrics_df, summary_df, winners_df, failures_df = run_multihorizon_cv_b12(
        panel,
        origins=[2008, 2013, 2018],
        horizons=(1, 3, 5, 10),
        sexes=("both",),
        random_state=int(cfg.get("seeds", {}).get("numpy", 42)),
    )

    # Also verify exact backwards-compatible h=5 run_rolling_origin_cv call works cleanly
    cv_h5_compat = run_rolling_origin_cv(panel, origins=[2008, 2013, 2018], horizon=5)

    cf_df = run_counterfactual_comparison_b12(panel, root)
    s3_sens_df = run_stage3_slope_sensitivity_b12(cfg, root)

    metrics_path = tables_dir / "s2_rolling_cv_metrics_b12.csv"
    summary_path = tables_dir / "s2_cv_summary_b12.csv"
    winners_path = tables_dir / "s2_cv_per_series_winners_b12.csv"
    failures_path = tables_dir / "s2_model_failures_b12.csv"
    cf_path = tables_dir / "s2_counterfactual_comparison_b12.csv"
    s3_sens_path = tables_dir / "s3_b12_counterfactual_sensitivity.csv"
    fig_path = figures_dir / "s2_model_comparison_cv_b12.png"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    winners_df.to_csv(winners_path, index=False)
    failures_df.to_csv(failures_path, index=False)
    cf_df.to_csv(cf_path, index=False)
    s3_sens_df.to_csv(s3_sens_path, index=False)
    plot_multihorizon_cv_b12(summary_df, fig_path)
    bundle_final_real_outputs(cfg, root=root, final_dir=root / "outputs" / "final_real_data")

    return {
        "panel_rows": len(panel),
        "cv_rows": len(metrics_df),
        "compat_h5_rows": len(cv_h5_compat),
        "failures": len(failures_df),
        "summary_h5": summary_df[summary_df["horizon"] == 5].to_dict(orient="records"),
        "cf_rows": len(cf_df),
        "s3_sens_rows": len(s3_sens_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run B-12 multi-horizon CV & counterfactual benchmark"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    res = run_b12_suite(args.config)
    print(
        f"B-12 CV suite complete: panel={res['panel_rows']}, "
        f"cv_rows={res['cv_rows']}, failures={res['failures']}"
    )


if __name__ == "__main__":
    main()
