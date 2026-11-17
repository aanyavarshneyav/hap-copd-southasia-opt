"""Build compact static JSON datasets and downloadable artifacts for the research website.

Reads the real-data repository state (configs/default.yaml, configs/intervention_params.json,
data/processed/*.{csv,json}, outputs/final_real_data/{tables,figures}/*, research/person_b_inputs/*,
and pddl/*) and writes clean JSON files to site/data/, downloadable assets to site/downloads/,
and publication PNGs to site/assets/images/real/.

Usage:
    .venv/bin/python site/scripts/prepare_data.py
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = REPO_ROOT / "site"
DATA_OUT = SITE_DIR / "data"
DOWNLOADS_OUT = SITE_DIR / "downloads"
IMAGES_REAL_OUT = SITE_DIR / "assets" / "images" / "real"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.contracts import load_stage3_contracts  # noqa: E402
from src.optimise.evaluator import Stage3Evaluator  # noqa: E402
from src.optimise.nsga2 import decode_allocation_spend, run_nsga2_multi_seed  # noqa: E402


def _clean_value(val: Any) -> Any:
    """Convert numpy/pandas scalars and NaNs into JSON-safe Python primitives."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return round(fval, 8)
    if isinstance(val, str):
        return val
    return val


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""
    records: list[dict[str, Any]] = []
    for raw in df.to_dict(orient="records"):
        records.append({str(k): _clean_value(v) for k, v in raw.items()})
    return records


def _write_json(filename: str, payload: dict[str, Any]) -> Path:
    """Write formatted compact JSON to site/data/."""
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out_path = DATA_OUT / filename
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _parse_pddl_solution(text: str) -> dict[str, Any]:
    """Parse PDDL solution text into metadata, raw text, and structured steps."""
    meta: dict[str, str] = {}
    steps: list[dict[str, Any]] = []
    solve_time: float | None = None

    action_descriptions = {
        "build-lpg-infra": (
            "Construct regional LPG storage, bottling, and distribution prerequisite "
            "(unlocks LPG subsidy in strictly later periods)."
        ),
        "build-grid-infra": (
            "Upgrade rural/peri-urban distribution transformers for electric induction "
            "(unlocks induction in strictly later periods)."
        ),
        "rollout-improved-biomass": (
            "Deploy improved biomass cookstoves (+0.10 coverage tier; available without "
            "LPG or grid prerequisite)."
        ),
        "rollout-lpg-init-ready": (
            "Scale up targeted LPG cylinder & stove subsidy in an initially LPG-ready unit "
            "(+0.10 coverage tier)."
        ),
        "rollout-lpg-after-infra": (
            "Scale up targeted LPG subsidy after prior-period LPG infrastructure build "
            "(+0.10 coverage tier)."
        ),
        "rollout-electric-init-ready": (
            "Deploy electric induction cookstoves in an initially grid-ready unit "
            "(+0.10 coverage tier)."
        ),
        "rollout-electric-after-infra": (
            "Deploy electric induction cookstoves after prior-period grid upgrade "
            "(+0.10 coverage tier)."
        ),
        "advance-period": "Transition planning clock to the next 5-year period.",
    }

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            clean = stripped.lstrip("# ").strip()
            if "Generated:" in clean:
                meta["generated_line"] = clean
            elif "Planner:" in clean:
                meta["planner_line"] = clean
                m_time = re.search(r"solve_time=([0-9.]+)s", clean)
                if m_time:
                    solve_time = float(m_time.group(1))
            elif "Replay Verification:" in clean:
                meta["replay_verification"] = clean
            elif "Stage3Evaluator Check:" in clean:
                meta["evaluator_check"] = clean
            continue

        match = re.match(r"^(\d+):\s*\(([^)]+)\)$", stripped)
        if match:
            step_num = int(match.group(1))
            tokens = match.group(2).split()
            op = tokens[0]
            args = tokens[1:]
            unit = None
            period = None
            for tok in args:
                if tok in {"ind_urban", "ind_rural", "npl_urban", "npl_rural"}:
                    unit = tok.replace("ind_", "IND_").replace("npl_", "NPL_")
                if tok in {"p1", "p2", "p3"}:
                    period = tok.upper()
            steps.append(
                {
                    "step": step_num,
                    "raw": f"({match.group(2)})",
                    "operator": op,
                    "args": args,
                    "period": period,
                    "unit": unit,
                    "description": action_descriptions.get(op, op),
                }
            )

    return {
        "meta": meta,
        "solve_time_seconds": solve_time,
        "raw_text": text,
        "steps": steps,
    }


def _build_metadata_and_overview(
    config: dict[str, Any],
    evaluator: Stage3Evaluator,
    selected_plans: pd.DataFrame,
    bau_2040: dict[str, float],
    mdp_row: dict[str, Any],
    pareto_reeval: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create metadata.json and overview.json from validated real-data outputs."""
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    use_mock = bool(config["interfaces"]["use_mock_forecasts"])
    budget_cap = float(evaluator.total_budget_usd_millions)

    knee_row = selected_plans[selected_plans["plan_name"] == "knee"].iloc[0]
    daly_min_row = selected_plans[selected_plans["plan_name"] == "daly_minimal"].iloc[0]
    bau_mean = float(bau_2040["dalys_2040_mean"])
    bau_lo = float(bau_2040["dalys_2040_lower"])
    bau_hi = float(bau_2040["dalys_2040_upper"])

    knee_averted = float(knee_row["dalys_averted_mean"])
    max_averted = float(daly_min_row["dalys_averted_mean"])
    max_pct = (max_averted / bau_mean) * 100.0
    knee_pct = (knee_averted / bau_mean) * 100.0

    bau_mc = pareto_reeval[pareto_reeval["plan_name"] == "bau_zero_action"].iloc[0]

    metadata = {
        "project_name": config["project"]["name"],
        "version": config["project"]["version"],
        "generated_at": generated_at,
        "use_mock_forecasts": use_mock,
        "data_version": "GBD_2023_WPP_2024",
        "parameter_version": "REAL_V1_2024USD",
        "pipeline_status": {
            "stage0_ingestion": {
                "status": "complete",
                "label": "Stage 0: Data Ingestion & Harmonisation",
                "detail": "GBD 2023 (1990–2023), UN WPP 2024, WHO SDG 7.1.2, NFHS-5, NDHS 2022, India/Nepal Censuses validated.",
            },
            "stage1_historical": {
                "status": "complete",
                "label": "Stage 1: Historical Burden (1990–2023)",
                "detail": "5-country ASRs, EAPC, PWLF breakpoints, HAP-COPD shares & apportioned IND/NPL units complete.",
            },
            "stage2_forecast": {
                "status": "complete",
                "label": "Stage 2: 2040 Forecasting & Smooth Log-OLS Surrogate",
                "detail": "Rolling-origin CV (fixed ETS, ARIMA, Drift, Theta), reconciled 2.70M BAU, smooth Log-OLS surrogate (-2.1933) & 500 draws.",
            },
            "stage3_optimise": {
                "status": "complete",
                "label": "Stage 3: NSGA-II Optimisation (Scenario A)",
                "detail": "251 real Pareto solutions ($5,985.73M cap, Log-OLS surrogate), 500-draw joint MC & 8-factor sensitivity.",
            },
            "stage4_planning": {
                "status": "complete",
                "label": "Stage 4: AI Planning (MDP, HTN, PDDL)",
                "detail": "MDP λ=116.0 (22,136 states, 0.0% viol.), HTN (9 leaves), PDDL (7 steps, 0.0798s) & 1,000 CRN rollouts.",
            },
            "param_sourcing": {
                "status": "complete",
                "label": "Contract 4 Parameters (REAL_V1_2024USD)",
                "detail": "Active real unit costs ($22.50–$106.00/HH) & separate HH/pop baselines; infra_lpg_grid null (disabled Scen. A).",
            },
        },
    }

    overview = {
        "title": "Household Air Pollution & COPD",
        "subtitle": "Forecasting Burden. Optimising Intervention. Planning for 2040.",
        "kpis": [
            {
                "id": "bau_2040",
                "label": "2040 BAU Burden (F+M Sum)",
                "value": f"{bau_mean / 1e6:.2f}M",
                "detail": (
                    f"{bau_mean:,.0f} DALYs [{bau_lo / 1e6:.2f}M–{bau_hi / 1e6:.2f}M]; "
                    f"500 MC draws mean {float(bau_mc['stage2_draws_dalys_mean']):,.0f}"
                ),
                "status": "real",
            },
            {
                "id": "max_averted",
                "label": "Max NSGA-II DALYs Averted",
                "value": f"{max_averted / 1e3:,.1f}k",
                "detail": (
                    f"{max_pct:.1f}% reduction (to {float(daly_min_row['dalys_2040_mean']) / 1e6:.2f}M) "
                    f"at ${float(daly_min_row['total_cost_usd_millions']):,.1f}M spend"
                ),
                "status": "real",
            },
            {
                "id": "knee_plan",
                "label": "NSGA-II Knee Plan",
                "value": f"{knee_averted / 1e3:,.1f}k averted",
                "detail": (
                    f"{knee_pct:.1f}% reduction ({float(knee_row['dalys_2040_mean']) / 1e6:.2f}M DALYs) "
                    f"at ${float(knee_row['total_cost_usd_millions']):,.1f}M; eq gap {float(knee_row['equity_gap_asr']):.2f}"
                ),
                "status": "real",
            },
            {
                "id": "budget_cap",
                "label": "Real Budget Cap (50% Base)",
                "value": f"${budget_cap:,.1f}M",
                "detail": f"50% of $11,971.5M 2025 HH LPG fiscal gap; Scenario A saturates at ${float(daly_min_row['total_cost_usd_millions']):,.1f}M",
                "status": "real",
            },
            {
                "id": "mdp_advantage",
                "label": "MDP Adaptive Policy (λ=116)",
                "value": f"{float(mdp_row['constraint_violation_rate']) * 100:.1f}% viol.",
                "detail": (
                    f"{float(mdp_row['mean_dalys_averted']) / 1e3:,.1f}k mean averted at "
                    f"${float(mdp_row['mean_cost_usd_millions']):,.1f}M vs. 40.0% viol. for GA matched"
                ),
                "status": "real",
            },
            {
                "id": "real_data",
                "label": "Real Data & Parameters",
                "value": "B-10 Complete",
                "detail": "GBD 2023 + 1,585 WHO/DHS records + REAL_V1_2024USD + 18 sensitivity runs",
                "status": "real",
            },
        ],
        "objectives": [
            {
                "id": "rq1",
                "title": "Historical Trends (1990–2023)",
                "summary": (
                    "All five South Asian countries saw steep DALY ASR declines (Bhutan −94.2%, India −56.4%, "
                    "Pakistan −44.0%, Nepal −40.0%, Bangladesh −32.2%), yet population ageing raised absolute DALY "
                    "counts in India (+24.8% to 6.96M) and Nepal (+75.7% to 254.0k)."
                ),
                "status": "complete",
            },
            {
                "id": "rq2",
                "title": "Forecasting to 2040",
                "summary": (
                    "B-12 multi-horizon CV fixes ETS (0 failures): fixed ETS (47.73/48.78 MAE; +53.9%/+52.8% skill) and ARIMA (49.33; +52.3% skill) lead short-term "
                    "accuracy, while smooth Log-OLS (4-unit CV MAE 102.34 vs. 151.00 for HistGBDT; slope −2.1933 ± 0.2557 SE) replaces HistGBDT as the Stage 3 counterfactual surrogate."
                ),
                "status": "complete",
            },
            {
                "id": "rq3",
                "title": "Multi-Objective Optimisation",
                "summary": (
                    "5-seed NSGA-II discovers 251 feasible Scenario A Pareto plans (combined HV 2.7965e+10), beating "
                    "all three heuristic/random baselines. Knee saves 311.7k DALYs ($1,695.9M); DALY-min (=Equity-min) "
                    "saves 453.5k DALYs ($3,097.3M) and lowers WHO-reconciled equity gap from 31.41 to 24.91/100k."
                ),
                "status": "complete",
            },
            {
                "id": "rq4",
                "title": "Sequential Adaptive Planning",
                "summary": (
                    "Recalibrated Finite-Horizon MDP (λ=116.0, 22,136 states) achieves 0.0% constraint violations "
                    "across 1,000 uptake-shock rollouts ($600.4M mean cost, 127.5k averted), whereas open-loop GA achieves "
                    "higher mean scalarized value (90.7k vs. 46.7k) with 40.0% cap/ceiling overshoot under 1.25× shocks."
                ),
                "status": "complete",
            },
        ],
        "research_questions": [
            {
                "id": "RQ1",
                "question": "How have HAP-attributable COPD deaths and DALYs changed in India, Nepal, and wider South Asia from 1990 to 2023 by sex?",
                "note": "Answered using GBD 2023 + UN WPP 2024 + NFHS/Census rural-urban apportionment across 5 countries and 4 subnational units.",
            },
            {
                "id": "RQ2",
                "question": "Which forecasting approach (statistical or ML) projects attributable burden to 2040 most reliably on held-out rolling windows?",
                "note": "Answered via B-12 multi-horizon CV (fixed ETS 47.73 MAE / +53.9% skill vs ARIMA 49.33; smooth Log-OLS surrogate −2.1933 ± 0.2557 replacing HistGBDT + 500 draws).",
            },
            {
                "id": "RQ3",
                "question": "Under a real 50% fiscal clean-cooking budget ($5,985.73M), which allocation minimises projected 2040 DALYs vs. cost and urban–rural equity?",
                "note": "Answered via 5-seed NSGA-II (251 Pareto plans), 500-draw joint MC re-evaluation, and 8-factor sensitivity (Scenario A vs. B1/B2 & Alt B opex).",
            },
            {
                "id": "RQ4",
                "question": "How does an adaptive MDP policy compare against open-loop GA schedules, HTN decomposition, and classical PDDL planning?",
                "note": "Answered via 1,000 CRN rollouts under ±uptake shocks, showing MDP's 0.0% violation robustness vs. 40.0% (GA) and 26.9% (PDDL).",
            },
        ],
        "pipeline_nodes": [
            {
                "id": "stage0",
                "step": "Stage 0",
                "title": "Data Ingestion",
                "subtitle": "GBD 2023, WPP 2024, WHO, NFHS/NDHS",
                "status": "complete",
                "target": "#/historical",
            },
            {
                "id": "stage1",
                "step": "Stage 1",
                "title": "Historical Trends",
                "subtitle": "ASRs, EAPC & PWLF breakpoints",
                "status": "complete",
                "target": "#/historical",
            },
            {
                "id": "stage2",
                "step": "Stage 2",
                "title": "2040 Forecasting",
                "subtitle": "CV, HistGBDT surrogate & 500 draws",
                "status": "complete",
                "target": "#/forecast",
            },
            {
                "id": "params",
                "step": "Contract 4",
                "title": "Intervention Params",
                "subtitle": "REAL_V1_2024USD & Scenario A/B",
                "status": "complete",
                "target": "#/assumptions",
            },
            {
                "id": "stage3",
                "step": "Stage 3",
                "title": "NSGA-II Optimisation",
                "subtitle": "251 Pareto plans & 500-draw MC",
                "status": "complete",
                "target": "#/optimise",
            },
            {
                "id": "stage4",
                "step": "Stage 4",
                "title": "AI Planning",
                "subtitle": "MDP (λ=116), HTN & PDDL",
                "status": "complete",
                "target": "#/planning",
            },
            {
                "id": "research",
                "step": "Archive",
                "title": "Research & Files",
                "subtitle": "Methods, limitations & 60+ exports",
                "status": "complete",
                "target": "#/research",
            },
        ],
    }
    return metadata, overview


def _build_historical_clean_fuel() -> dict[str, Any]:
    """Load real clean-fuel, demographic, and Stage 1 GBD historical datasets."""
    cf_df = pd.read_csv(REPO_ROOT / "research/person_b_inputs/clean_fuel_access.csv")
    demo_df = pd.read_csv(REPO_ROOT / "research/person_b_inputs/demographic_inputs.csv")
    s1_sum = pd.read_csv(REPO_ROOT / "outputs/final_real_data/tables/s1_historical_summary.csv")
    s1_eapc = pd.read_csv(REPO_ROOT / "outputs/final_real_data/tables/s1_eapc_joinpoint.csv")
    s1_pwlf = pd.read_csv(REPO_ROOT / "outputs/final_real_data/tables/s1_pwlf_breakpoints.csv")
    s1_share = pd.read_csv(REPO_ROOT / "outputs/final_real_data/tables/s1_attributable_share.csv")

    return {
        "is_mock": False,
        "data_status": "REAL_ACQUIRED_DATA",
        "sources": [
            "WHO GHO (PHE_HHAIR_PROP_POP_CLEAN_FUELS, 2000–2024)",
            "India NFHS-5 (2019–21, IIPS/DHS)",
            "India Census 2011 Houselisting",
            "Nepal NDHS 2022 (MoHP / ICF DHS)",
            "Nepal NPHC 2021 Census",
            "UN World Population Prospects 2024",
            "GBD 2023 (IHME, 1990–2023 HAP-COPD & Total COPD)",
        ],
        "gbd_burden_status": {
            "status": "complete",
            "countries": ["India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"],
            "subnational_units": ["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"],
            "period": "1990–2023",
            "expected_files": [
                "s1_historical_summary.csv",
                "s1_eapc_joinpoint.csv",
                "s1_pwlf_breakpoints.csv",
                "s1_attributable_share.csv",
            ],
            "note": (
                "Complete Stage 1 GBD 2023 historical analysis across 5 South Asian countries "
                "and 4 apportioned India/Nepal subnational decision units (1990–2023)."
            ),
        },
        "clean_fuel_records": _df_to_records(cf_df),
        "demographic_records": _df_to_records(demo_df),
        "gbd_summary": _df_to_records(s1_sum),
        "eapc_joinpoint": _df_to_records(s1_eapc),
        "pwlf_breakpoints": _df_to_records(s1_pwlf),
        "attributable_share": _df_to_records(s1_share),
    }


def _build_forecast_real(config: dict[str, Any]) -> dict[str, Any]:
    """Load real Stage 2 forecasts, CV metrics, baseline states, and surrogate."""
    ifaces = config["interfaces"]
    fc_df = pd.read_csv(REPO_ROOT / ifaces["forecast_contract_file"])
    base_df = pd.read_csv(REPO_ROOT / ifaces["baseline_state_file"])
    surr = json.loads((REPO_ROOT / ifaces["surrogate_file"]).read_text(encoding="utf-8"))
    cv_df = pd.read_csv(REPO_ROOT / "outputs/final_real_data/tables/s2_rolling_cv_metrics.csv")
    sum_df = pd.read_csv(REPO_ROOT / "outputs/final_real_data/tables/s2_forecasts_2040_summary.csv")
    cv_b12_sum = pd.read_csv(REPO_ROOT / "outputs/final_real_data/tables/s2_cv_summary_b12.csv")
    cf_b12_df = pd.read_csv(
        REPO_ROOT / "outputs/final_real_data/tables/s2_counterfactual_comparison_b12.csv"
    )
    audit_b12_df = pd.read_csv(
        REPO_ROOT / "outputs/final_real_data/tables/s2_hierarchical_audit_b12.csv"
    )

    return {
        "is_mock": False,
        "data_version": "GBD_2023_WPP_2024",
        "real_stage2_status": {
            "status": "complete",
            "models": [
                {
                    "id": "ets_undamped",
                    "name": "ETS (Undamped & Damped, Fixed B-12)",
                    "family": "Statistical Time Series",
                    "implementation": "statsmodels ExponentialSmoothing (optimized=True, no ARIMA fallback): h=5 MAE 47.73 undamped (+53.9% skill) & 48.78 damped (+52.8% skill)",
                },
                {
                    "id": "arima",
                    "name": "ARIMA(1,1,0)",
                    "family": "Statistical Time Series",
                    "implementation": "statsmodels ARIMA(1,1,0): h=1 winner (11.31 MAE, 0.67 MASE); h=5 MAE 49.33 (+52.3% skill)",
                },
                {
                    "id": "drift_theta",
                    "name": "Drift & Theta Benchmarks",
                    "family": "Statistical Time Series",
                    "implementation": "Random-walk with drift (h=5 MAE 63.74; +38.4% skill) and statsmodels ThetaModel (79.28; +23.3% skill)",
                },
                {
                    "id": "poisson_rate_glm",
                    "name": "Poisson Rate GLM & Log-OLS",
                    "family": "Reduced-Form Covariate Panel",
                    "implementation": "Log-linear panel with clean_fuel_share: h=5 MAE 147.55 / 140.96; h=10 reaches positive +4.9% / +8.8% skill",
                },
                {
                    "id": "log_ols_smooth",
                    "name": "Smooth Log-OLS (Selected Production Surrogate)",
                    "family": "Parametric Log-Linear Counterfactual Panel",
                    "implementation": "log(ASR) ~ t + clean_fuel_share + C(location): 4-unit CV MAE 102.34 vs. 151.00 (HistGBDT); slope -2.1933 +/- 0.2557 SE",
                },
                {
                    "id": "hist_gbdt_panel",
                    "name": "HistGBDT Panel (Superseded Surrogate Benchmark)",
                    "family": "Machine Learning Panel",
                    "implementation": "Preserved historical benchmark (h=5 MAE 172.23, -66.5% skill; 4-unit 151.00); superseded due to IND_urban plateau",
                },
            ],
            "cv_folds": ["2009-2013", "2014-2018", "2019-2023"],
            "cv_metrics_required": ["MAE", "RMSE", "MASE", "Relative MAE", "Skill Score"],
            "note": (
                "Stage 2 forecasting & B-12/B-13 audit complete (0 fit failures). Fixed ETS (47.73/48.78) & ARIMA (49.33) lead univariate CV; "
                "smooth Log-OLS (slope -2.1933 +/- 0.2557 SE; R2=0.8938) replaces HistGBDT as Contract 3 surrogate while preserving 2.70M BAU."
            ),
        },
        "forecasts": _df_to_records(fc_df),
        "baseline_state": _df_to_records(base_df),
        "surrogate": surr,
        "cv_metrics": _df_to_records(cv_df),
        "cv_summary_b12": _df_to_records(cv_b12_sum),
        "counterfactual_comparison_b12": _df_to_records(cf_b12_df),
        "reconciliation_audit_b12": _df_to_records(audit_b12_df),
        "forecasts_2040_summary": _df_to_records(sum_df),
    }


def _build_assumptions(config: dict[str, Any], evaluator: Stage3Evaluator) -> dict[str, Any]:
    """Load real active Contract 4 parameters, B-09 research tables, and mock comparison."""
    pdir = REPO_ROOT / "research/person_b_inputs"
    ifaces = config["interfaces"]
    active_params = json.loads(
        (REPO_ROOT / ifaces["intervention_params_file"]).read_text(encoding="utf-8")
    )
    proposed_params = json.loads(
        (pdir / "proposed_intervention_params.json").read_text(encoding="utf-8")
    )
    mock_params = json.loads(
        (REPO_ROOT / ifaces["intervention_params_mock_file"]).read_text(encoding="utf-8")
    )
    manifest = json.loads((pdir / "source_manifest.json").read_text(encoding="utf-8"))

    tables: dict[str, list[dict[str, Any]]] = {}
    csv_map = {
        "proposed_contract4_table": "proposed_contract4_table.csv",
        "baseline_comparison": "b09_baseline_comparison.csv",
        "budget_scenarios": "b09_budget_scenarios.csv",
        "feasibility_audit": "b09_feasibility_audit.csv",
        "parameter_gap_matrix": "parameter_gap_matrix.csv",
        "source_register": "intervention_source_register.csv",
        "demographic_summary": "demographic_inputs.csv",
    }
    for key, fname in csv_map.items():
        tables[key] = _df_to_records(pd.read_csv(pdir / fname))

    # Reflect B-10 resolution status inside gap matrix display rows accurately
    for row in tables["parameter_gap_matrix"]:
        param_path = str(row.get("parameter_path", ""))
        if "infra_lpg_grid" in param_path:
            row["real_data_status"] = "UNRESOLVED_SCENARIO_A_DISABLED"
        elif any(
            tok in param_path
            for tok in (
                "lpg_subsidy",
                "improved_biomass",
                "electric_induction",
                "budget",
                "discount",
                "baseline",
                "bridge",
            )
        ):
            if (
                "partial" in str(row.get("real_data_status", "")).lower()
                or "proposed" in str(row.get("real_data_status", "")).lower()
                or "ready" in str(row.get("real_data_status", "")).lower()
            ):
                row["real_data_status"] = "ACTIVE_IN_REAL_V1"

    return {
        "units": list(evaluator.units),
        "interventions": [
            {
                "id": "lpg_subsidy",
                "name": "Targeted LPG Subsidy",
                "role": "Immediate high-effectiveness fuel switch (η = 0.78)",
                "infra_required": True,
                "description": (
                    "PAHAL / Ujjwala-style targeted LPG cylinder & refill support ($84.33–$90.30/HH India; "
                    "$99.00–$106.00/HH Nepal). Alt A models 1-time programme capex; Alt B sensitivity tests recurring refill opex."
                ),
            },
            {
                "id": "improved_biomass",
                "name": "Improved Biomass Cookstove",
                "role": "Interim transitional harm reduction (η = 0.25)",
                "infra_required": False,
                "description": (
                    "Forced-draft / Tier 2–3 biomass stove at 50% public co-finance ($22.50/HH co-financed; "
                    "$45.00/HH full cost). Only direct adoption option for unready NPL_rural in Scenario A."
                ),
            },
            {
                "id": "electric_induction",
                "name": "Electric Induction Cooking",
                "role": "Zero-PM2.5 household cooking transition (η = 0.85)",
                "infra_required": True,
                "description": (
                    "1800W single-burner induction stove + starter cookware at 50% public co-finance ($26.56/HH India; "
                    "$31.84/HH Nepal). Requires grid-ready status (IND_urban, NPL_urban in Scenario A)."
                ),
            },
            {
                "id": "infra_lpg_grid",
                "name": "LPG / Grid Infrastructure Build",
                "role": "Unlocks prerequisite-gated adoption from period p+1",
                "infra_required": False,
                "description": (
                    "Unresolved real capital cost (null in REAL_V1_2024USD, caps P1=P2=P3=0 in Scenario A). "
                    "Tested under explicit Scenario B1 ($150/HH LPG, $250/HH grid) & B2 ($300/$400) proxies."
                ),
            },
        ],
        **tables,
        "source_manifest_raw_files": manifest.get("files", []),
        "proposed_params": active_params,
        "b09_proposed_params": proposed_params,
        "production_template": active_params,
        "mock_params": mock_params,
    }


def _build_optimise_real(
    config: dict[str, Any],
    evaluator: Stage3Evaluator,
    bau_2040: dict[str, float],
    selected_df: pd.DataFrame,
    reeval_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    opex_df: pd.DataFrame,
) -> dict[str, Any]:
    """Load real Stage 3 Pareto front, baseline comparisons, MC re-evaluation, and 8-factor sensitivity."""
    tdir = REPO_ROOT / "outputs/final_real_data/tables"
    pareto_df = pd.read_csv(tdir / "s3_pareto_front.csv")
    base_comp_df = pd.read_csv(tdir / "s3_baseline_comparison.csv")
    hv_seeds_df = pd.read_csv(tdir / "s3_hypervolume_seeds.csv")

    multi_seed = run_nsga2_multi_seed(evaluator, config)

    units = list(evaluator.units)
    interventions = list(evaluator.interventions)
    periods = list(evaluator.periods)

    sel_by_id: dict[int, list[str]] = {}
    for _, row in selected_df.iterrows():
        sid = int(row["solution_id"])
        sel_by_id.setdefault(sid, []).append(str(row["plan_name"]))

    pareto_solutions: list[dict[str, Any]] = []
    for _, row in pareto_df.iterrows():
        sid = int(row["solution_id"])
        x_vals: list[float] = []
        for u in units:
            for iv in interventions:
                for p in periods:
                    x_vals.append(float(row[f"{u}__{iv}__{p}"]))
        alloc_df = decode_allocation_spend(np.asarray(x_vals, dtype=float), evaluator)
        allocations = [
            {
                "unit": str(ar["unit"]),
                "intervention": str(ar["intervention"]),
                "period": str(ar["period"]),
                "allocation_value": round(float(ar["allocation_value"]), 5),
                "cost_usd_millions": round(float(ar["cost_usd_millions"]), 3),
            }
            for ar in alloc_df.to_dict(orient="records")
        ]

        pareto_solutions.append(
            {
                "solution_id": sid,
                "seed": int(row["seed"]),
                "selected_plans": ", ".join(sel_by_id.get(sid, [])) if sid in sel_by_id else None,
                "dalys_2040_mean": _clean_value(row["dalys_2040_mean"]),
                "dalys_2040_lower": _clean_value(row["dalys_2040_lower"]),
                "dalys_2040_upper": _clean_value(row["dalys_2040_upper"]),
                "dalys_averted_mean": _clean_value(
                    row["dalys_averted_mean"]
                    if "dalys_averted_mean" in row
                    else float(bau_2040["dalys_2040_mean"]) - float(row["dalys_2040_mean"])
                ),
                "total_cost_usd_millions": _clean_value(row["total_cost_usd_millions"]),
                "equity_gap_asr": _clean_value(row["equity_gap_asr"]),
                "equity_gap_dalys_abs": _clean_value(row["equity_gap_dalys_abs"]),
                "is_feasible": bool(row["is_feasible"]),
                "violations": {
                    "budget": _clean_value(row["violation_budget"]),
                    "adoption_cap": _clean_value(row["violation_adoption_cap"]),
                    "coverage_ceiling": _clean_value(row["violation_coverage_ceiling"]),
                    "infra_prereq": _clean_value(row["violation_infra_prereq"]),
                },
                "allocations": allocations,
            }
        )

    hv_convergence: list[dict[str, Any]] = []
    for sres in multi_seed.seed_results:
        hv_convergence.append(
            {
                "seed": int(sres.seed),
                "history": [
                    {
                        "generation": int(h["gen"]),
                        "gen": int(h["gen"]),
                        "n_eval": int(h["n_eval"]),
                        "hypervolume": round(float(h["hypervolume"]), 4),
                    }
                    for h in sres.hv_history
                ],
            }
        )

    sens_records = _df_to_records(sens_df)
    selected_cols = [
        "plan_name",
        "solution_id",
        "dalys_2040_mean",
        "dalys_2040_lower",
        "dalys_2040_upper",
        "dalys_averted_mean",
        "total_cost_usd_millions",
        "equity_gap_asr",
        "equity_gap_dalys_abs",
        "is_feasible",
    ]
    return {
        "is_mock": False,
        "warning_mock_inputs": False,
        "data_version": "GBD_2023_WPP_2024",
        "parameter_version": "REAL_V1_2024USD",
        "scenario_name": "Scenario A (existing infra-ready only; infra_lpg_grid disabled)",
        "budget_cap_usd_millions": float(evaluator.total_budget_usd_millions),
        "bau_2040": bau_2040,
        "units": units,
        "interventions": interventions,
        "periods": periods,
        "pareto_solutions": pareto_solutions,
        "selected_plans": _df_to_records(selected_df[selected_cols]),
        "baseline_comparison": _df_to_records(base_comp_df),
        "hypervolume_seeds": _df_to_records(hv_seeds_df),
        "hypervolume_convergence": hv_convergence,
        "pareto_reeval": _df_to_records(reeval_df),
        "sensitivity_real": sens_records,
        "recurring_opex_sensitivity": _df_to_records(opex_df),
        "budget_sweep": [
            r for r in sens_records if r.get("factor_group") in {"0_base_reference", "1_budget"}
        ],
        "eta_biomass_sweep": [
            r
            for r in sens_records
            if r.get("factor_group") in {"0_base_reference", "3_eta_biomass"}
        ],
        "induction_cap_sweep": [
            r
            for r in sens_records
            if r.get("factor_group") in {"0_base_reference", "4_induction_caps"}
        ],
        "bridge_sensitivity": [
            r
            for r in sens_records
            if r.get("factor_group") in {"0_base_reference", "2_bridge_factor"}
        ],
    }


def _build_planning_real(config: dict[str, Any]) -> dict[str, Any]:
    """Load real Stage 4 MDP, HTN, PDDL, and GA-vs-MDP comparison artifacts."""
    tdir = REPO_ROOT / "outputs/final_real_data/tables"
    policy_df = pd.read_csv(tdir / "plan_mdp_policy.csv")
    rollouts_df = pd.read_csv(tdir / "plan_mdp_rollouts.csv")
    mdp_sens_df = pd.read_csv(tdir / "plan_mdp_sensitivity.csv")
    htn_trace = json.loads((tdir / "plan_htn_trace.json").read_text(encoding="utf-8"))
    comp_df = pd.read_csv(tdir / "comp_ga_vs_mdp.csv")
    comp_sens_df = pd.read_csv(tdir / "comp_sensitivity.csv")

    pddl_dir = REPO_ROOT / config["paths"]["pddl_dir"]
    pddl_domain = (pddl_dir / "domain.pddl").read_text(encoding="utf-8")
    pddl_problem = (pddl_dir / "problem_ind_npl_existing_infra.pddl").read_text(encoding="utf-8")
    pddl_solution_raw = (tdir / "plan_pddl_solution.txt").read_text(encoding="utf-8")
    parsed_sol = _parse_pddl_solution(pddl_solution_raw)

    decision_rows = policy_df[policy_df["period"].isin(["P1", "P2", "P3"])].copy()
    policy_summary: list[dict[str, Any]] = []
    for period in ["P1", "P2", "P3"]:
        sub = decision_rows[decision_rows["period"] == period]
        counts = sub["action_label"].value_counts().head(8)
        policy_summary.append(
            {
                "period": period,
                "n_states": int(len(sub)),
                "top_actions": [
                    {"action": str(act), "state_count": int(cnt)} for act, cnt in counts.items()
                ],
            }
        )

    keep_policy_cols = [
        "period_idx",
        "period",
        "state_id",
        "budget_bin",
        "remaining_budget_tier_usd_millions",
        "state_dalys_2040_mean",
        "state_value",
        "action_id",
        "action_label",
        "action_cost_usd_millions",
        "IND_urban__raw_cov",
        "IND_urban__infra_ready",
        "IND_rural__raw_cov",
        "IND_rural__infra_ready",
        "NPL_urban__raw_cov",
        "NPL_urban__infra_ready",
        "NPL_rural__raw_cov",
        "NPL_rural__infra_ready",
    ]

    mdp_cfg = config["planning"]["mdp"]
    return {
        "is_mock": False,
        "warning_mock_inputs": False,
        "data_version": "GBD_2023_WPP_2024",
        "parameter_version": "REAL_V1_2024USD",
        "mdp_config": {
            "coverage_bins": mdp_cfg["coverage_bins"],
            "budget_bins": int(mdp_cfg["budget_bins"]),
            "discount_factor": float(mdp_cfg["discount_factor"]),
            "cost_weight_lambda": float(mdp_cfg["cost_weight_lambda"]),
            "transition_probs": mdp_cfg["transition_probs"],
            "shock_multipliers": config["planning"]["compare"]["shock_multipliers"],
            "n_rollouts": int(mdp_cfg["n_rollouts"]),
            "n_decision_states_b6": int(len(decision_rows)),
            "n_total_states_b6": int(len(policy_df)),
        },
        "mdp_policy_summary": policy_summary,
        "mdp_policy_rows": _df_to_records(decision_rows[keep_policy_cols]),
        "mdp_rollouts": _df_to_records(rollouts_df),
        "mdp_sensitivity": _df_to_records(mdp_sens_df),
        "htn_trace": htn_trace,
        "pddl": {
            "domain_name": "south-asia-clean-cooking",
            "problem_name": "ind-npl-clean-cooking-2026-2040-existing-infra",
            "requirements": [":strips", ":typing"],
            "planner": "pyperplan (astar + hff)",
            "solve_time_seconds": parsed_sol["solve_time_seconds"] or 0.0798,
            "predicates": [
                {
                    "name": "(lpg-init-ready ?u)",
                    "meaning": "Unit starts 2025 with LPG supply infrastructure ready",
                },
                {
                    "name": "(grid-init-ready ?u)",
                    "meaning": "Unit starts 2025 with electric grid ready for induction",
                },
                {
                    "name": "(lpg-infra-built-at ?u ?p)",
                    "meaning": "LPG infrastructure was built in unit ?u during period ?p",
                },
                {
                    "name": "(grid-infra-built-at ?u ?p)",
                    "meaning": "Grid upgrade was built in unit ?u during period ?p",
                },
                {
                    "name": "(cov-level ?u ?c)",
                    "meaning": "Unit ?u is at discrete clean-fuel coverage tier ?c (c0..c2)",
                },
                {
                    "name": "(clean-fuel-deployed ?u)",
                    "meaning": "Unit ?u has received an active clean-cooking deployment",
                },
                {
                    "name": "(budget-level ?b)",
                    "meaning": "Remaining discrete sequential action budget token ?b",
                },
                {
                    "name": "(active-period ?p)",
                    "meaning": "Current discrete planning period ?p (p1 -> p2 -> p3)",
                },
                {
                    "name": "(precedes ?p1 ?p2)",
                    "meaning": "Period ?p1 strictly precedes period ?p2",
                },
            ],
            "actions": [
                {
                    "name": "build-lpg-infra",
                    "params": "(?u ?p ?b1 ?b2)",
                    "preconditions": "Active period ?p; budget ?b1 -> ?b2",
                    "effects": "Marks LPG infra built at (?u, ?p); Scenario A omits build actions from solution via disabled budget/cap",
                    "meaning": "Unlocks LPG subsidy rollouts in strictly later periods (p > q).",
                },
                {
                    "name": "build-grid-infra",
                    "params": "(?u ?p ?b1 ?b2)",
                    "preconditions": "Active period ?p; budget ?b1 -> ?b2",
                    "effects": "Marks grid infra built at (?u, ?p)",
                    "meaning": "Unlocks electric induction rollouts in strictly later periods (p > q).",
                },
                {
                    "name": "rollout-improved-biomass",
                    "params": "(?u ?p ?c1 ?c2 ?b1 ?b2)",
                    "preconditions": "cov ?c1; active ?p; budget ?b1 -> ?b2",
                    "effects": "Advances cov ?c1 -> ?c2",
                    "meaning": "Always enabled (no LPG/grid prereq; sole direct option in unready NPL_rural in Scenario A).",
                },
                {
                    "name": "rollout-lpg-init-ready",
                    "params": "(?u ?p ?c1 ?c2 ?b1 ?b2)",
                    "preconditions": "(lpg-init-ready ?u); cov ?c1; budget ?b1 -> ?b2",
                    "effects": "Advances cov ?c1 -> ?c2",
                    "meaning": "Deploys LPG immediately in IND_urban, IND_rural, and NPL_urban.",
                },
                {
                    "name": "rollout-lpg-after-infra",
                    "params": "(?u ?q ?p ?c1 ?c2 ?b1 ?b2)",
                    "preconditions": "(lpg-infra-built-at ?u ?q); (precedes ?q ?p); active ?p",
                    "effects": "Advances cov ?c1 -> ?c2",
                    "meaning": "Enforces causal lag after LPG infrastructure build.",
                },
                {
                    "name": "rollout-electric-init-ready",
                    "params": "(?u ?p ?c1 ?c2 ?b1 ?b2)",
                    "preconditions": "(grid-init-ready ?u); cov ?c1; budget ?b1 -> ?b2",
                    "effects": "Advances cov ?c1 -> ?c2",
                    "meaning": "Deploys electric induction in initially grid-ready urban units.",
                },
                {
                    "name": "advance-period",
                    "params": "(?p1 ?p2)",
                    "preconditions": "(precedes ?p1 ?p2); (active-period ?p1)",
                    "effects": "Moves active clock p1 -> p2",
                    "meaning": "Monotonic forward progression across 5-year periods.",
                },
            ],
            "initial_state": {
                "ready_units": [
                    "ind_urban (LPG + Grid)",
                    "ind_rural (LPG only)",
                    "npl_urban (LPG + Grid)",
                ],
                "unready_units": ["npl_rural (neither; infra disabled in Scenario A)"],
                "coverage_start": "All 4 units at c0 (+0.00 above BAU)",
                "budget_tokens": "b8 (8 sequential action budget levels)",
                "active_period": "p1",
            },
            "goal_state": {
                "urban_targets": "ind_urban -> c1 (+0.10), npl_urban -> c1 (+0.10), clean-fuel-deployed",
                "rural_targets": "ind_rural -> c2 (+0.20), npl_rural -> c2 (+0.20), clean-fuel-deployed",
            },
            "domain_pddl": pddl_domain,
            "problem_pddl": pddl_problem,
            "solution": parsed_sol,
        },
        "comparison": _df_to_records(comp_df),
        "comparison_sensitivity": _df_to_records(comp_sens_df),
    }


def _copy_downloads() -> list[dict[str, str]]:
    """Copy real-data tables, figures, configs, PDDL, and docs to site/downloads/ & assets/images/real/."""
    DOWNLOADS_OUT.mkdir(parents=True, exist_ok=True)
    IMAGES_REAL_OUT.mkdir(parents=True, exist_ok=True)

    # Copy all final real figures into site/assets/images/real/
    fig_src_dir = REPO_ROOT / "outputs/final_real_data/figures"
    for fig_path in sorted(fig_src_dir.glob("*.png")):
        shutil.copy2(fig_path, IMAGES_REAL_OUT / fig_path.name)

    files_to_copy: list[tuple[Path, str, str]] = [
        # Stage 1 real tables
        (
            REPO_ROOT / "outputs/final_real_data/tables/s1_historical_summary.csv",
            "Stage 1: Historical GBD 2023 Summary (1990–2023)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s1_eapc_joinpoint.csv",
            "Stage 1: EAPC & Period Trends",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s1_pwlf_breakpoints.csv",
            "Stage 1: PWLF Structural Breakpoints",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s1_attributable_share.csv",
            "Stage 1: HAP-COPD Attributable Share (1990–2023)",
            "CSV",
        ),
        # Stage 2 real tables & contracts
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_rolling_cv_metrics.csv",
            "Stage 2: Original Rolling-Origin CV Metrics (MAE, RMSE, MASE)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_cv_summary_b12.csv",
            "Stage 2 (B-12): Multi-Horizon CV Summary (h=1,3,5,10 + Skill Score)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_counterfactual_comparison_b12.csv",
            "Stage 2 (B-12): Counterfactual Models Comparison (HistGBDT vs Monotonic vs Log-OLS/Log-Solid)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_hierarchical_audit_b12.csv",
            "Stage 2 (B-12): Full-Precision Demographic Hierarchy Audit (H1-H8)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_reconciliation_methods_b12.csv",
            "Stage 2 (B-12): Reconciliation Comparison (Bottom-Up vs OLS/WLS MinT vs WHO GHE Top-Down)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s3_b12_counterfactual_sensitivity.csv",
            "Stage 3 (B-12): IND_urban Surrogate Slope Sensitivity (-1.08% Impact)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_forecasts_2040_summary.csv",
            "Stage 2: 2040 Forecast Summary",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_sex_reconciliation_audit.csv",
            "Stage 2 (B-11): Sex-Specific Reconciliation Discrepancy Audit",
            "CSV",
        ),
        (
            REPO_ROOT / "data/processed/forecasts_stage3_reconciled.csv",
            "Contract 1 (Derived Reconciled): Stage 2 Forecasts Reconciled to Authoritative Both",
            "CSV",
        ),
        (
            REPO_ROOT / "data/processed/baseline_unit_state_reconciled.csv",
            "Contract 2 (Derived Reconciled): Subnational Baseline State",
            "CSV",
        ),
        (
            REPO_ROOT / "data/processed/exposure_response_surrogate.json",
            "Contract 3: Exposure–Response Surrogate Parameters (Smooth Log-OLS)",
            "JSON",
        ),
        (
            REPO_ROOT / "data/processed/exposure_response_surrogate_histgbdt.json",
            "Contract 3 (Preserved Baseline): Historical HistGBDT Surrogate Snapshot",
            "JSON",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_surrogate_comparison_b13.csv",
            "Stage 2 (B-13): Side-by-Side HistGBDT vs Smooth Log-OLS Response Curves",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s2_counterfactual_cv_identical_b13.csv",
            "Stage 2 (B-13): 4-Unit Identical-Fold Counterfactual CV Comparison",
            "CSV",
        ),
        (
            REPO_ROOT / "configs/intervention_params.json",
            "Contract 4: Active Real Intervention Parameters (REAL_V1_2024USD)",
            "JSON",
        ),
        # Stage 3 real optimisation & sensitivity tables
        (
            REPO_ROOT / "outputs/final_real_data/tables/s3_pareto_front.csv",
            "Stage 3: 251 Real NSGA-II Pareto Solutions",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s3_selected_plans.csv",
            "Stage 3: Selected Plans (Knee, Cost-Min, DALY-Min)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s3_baseline_comparison.csv",
            "Stage 3: NSGA-II vs. Random/Equal/Greedy Baselines",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s3_pareto_reeval.csv",
            "Stage 3: 500-Draw Joint Monte Carlo Uncertainty Re-evaluation",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s3_sensitivity_real.csv",
            "Stage 3: 8-Factor Real Sensitivity Suite (18 Scenarios)",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/s3_recurring_opex_sensitivity.csv",
            "Stage 3: Alt B Recurring LPG Opex Sensitivity",
            "CSV",
        ),
        # Stage 4 planning & comparison tables
        (
            REPO_ROOT / "outputs/final_real_data/tables/plan_htn_trace.json",
            "Stage 4: HTN Hierarchical Plan Trace",
            "JSON",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/plan_pddl_solution.txt",
            "Stage 4: PDDL Optimal 7-Step Plan (pyperplan)",
            "TXT",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/plan_mdp_sensitivity.csv",
            "Stage 4: MDP Finite- vs. Infinite-Horizon Sensitivity",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/comp_ga_vs_mdp.csv",
            "Stage 4: GA vs. MDP vs. PDDL 1,000-Rollout Comparison",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/comp_mdp_clamp_audit_b14.csv",
            "Stage 4 (B-14): MDP Pre-Clamp vs Post-Clamp & Symmetric GA Audit",
            "CSV",
        ),
        (
            REPO_ROOT / "outputs/final_real_data/tables/comp_sensitivity.csv",
            "Stage 4: Planning Horizon & λ Sensitivity Sweep",
            "CSV",
        ),
        # Clean-cooking inputs & PDDL/Configs/Docs
        (
            REPO_ROOT / "research/person_b_inputs/clean_fuel_access.csv",
            "WHO + DHS Clean-Cooking Series (1,585 records)",
            "CSV",
        ),
        (
            REPO_ROOT / "research/person_b_inputs/demographic_inputs.csv",
            "WPP + NFHS/NDHS/Census Household & Sex Panel",
            "CSV",
        ),
        (
            REPO_ROOT / "research/person_b_inputs/intervention_source_register.csv",
            "Literature Intervention Parameter Source Register",
            "CSV",
        ),
        (
            REPO_ROOT / "configs/default.yaml",
            "Active Real-Data Configuration (configs/default.yaml)",
            "YAML",
        ),
        (REPO_ROOT / "pddl/domain.pddl", "STRIPS PDDL Domain Definition", "PDDL"),
        (
            REPO_ROOT / "pddl/problem_ind_npl_existing_infra.pddl",
            "STRIPS Scenario A Existing-Infra Problem",
            "PDDL",
        ),
        (REPO_ROOT / "docs/INTERFACES.md", "Stage 1–4 Interface Contracts Specification", "MD"),
        (
            REPO_ROOT / "docs/ASSUMPTIONS.md",
            "Methodological & Parameter Assumptions Register",
            "MD",
        ),
        (REPO_ROOT / "docs/DECISIONS.md", "Architectural Decision Log", "MD"),
    ]

    entries: list[dict[str, str]] = []
    for src, label, fmt in files_to_copy:
        if not src.exists():
            continue
        dest = DOWNLOADS_OUT / src.name
        shutil.copy2(src, dest)
        entries.append(
            {
                "filename": src.name,
                "href": f"downloads/{src.name}",
                "label": label,
                "format": fmt,
                "size_kb": f"{dest.stat().st_size / 1024.0:.1f} KB",
            }
        )

    # Also register final PNG figures under downloads/figures/ for convenient direct downloads
    fig_dl_dir = DOWNLOADS_OUT / "figures"
    fig_dl_dir.mkdir(parents=True, exist_ok=True)
    for fig_path in sorted(fig_src_dir.glob("*.png")):
        dest = fig_dl_dir / fig_path.name
        shutil.copy2(fig_path, dest)
        entries.append(
            {
                "filename": fig_path.name,
                "href": f"downloads/figures/{fig_path.name}",
                "label": f"Figure: {fig_path.stem.replace('_', ' ')}",
                "format": "PNG",
                "size_kb": f"{dest.stat().st_size / 1024.0:.1f} KB",
            }
        )

    return entries


def _build_research(config: dict[str, Any], downloads: list[dict[str, str]]) -> dict[str, Any]:
    """Assemble real methodology stages, sources, limitations, reproducibility, and downloads."""
    return {
        "methodology_stages": [
            {
                "stage": "Stage 0",
                "name": "Data Ingestion & Validation",
                "owner": "Person A & Person B",
                "status": "complete",
                "inputs": "GBD 2023 (1990–2023), UN WPP 2024, WHO GHO, NFHS-5, NDHS 2022, India/Nepal Censuses",
                "outputs": "baseline_unit_state.csv, manifest.json, clean_fuel_access.csv (all schema-validated)",
            },
            {
                "stage": "Stage 1",
                "name": "Historical Epidemiological Analysis",
                "owner": "Person A (Rudraksh)",
                "status": "complete",
                "inputs": "5-country GBD 2023 panel + apportioned rural/urban IND/NPL rates",
                "outputs": "s1_historical_summary.csv, s1_eapc_joinpoint.csv, s1_pwlf_breakpoints.csv, s1_attributable_share.csv",
            },
            {
                "stage": "Stage 2",
                "name": "2040 Burden Forecasting & Surrogate",
                "owner": "Person A (Rudraksh)",
                "status": "complete",
                "inputs": "1990–2023 rates, clean-fuel trajectories, 3 rolling CV origins",
                "outputs": "forecasts_stage3.csv, exposure_response_surrogate.json, stage2_uncertainty_draws.csv (500 draws)",
            },
            {
                "stage": "Stage 3",
                "name": "Multi-Objective NSGA-II Optimisation",
                "owner": "Person B (Aanya)",
                "status": "complete",
                "inputs": "Real Contracts 1–4 (REAL_V1_2024USD, 50% cap $5,985.73M, Scenario A)",
                "outputs": "251 Pareto solutions, s3_pareto_reeval.csv (500 MC draws), s3_sensitivity_real.csv (18 runs)",
            },
            {
                "stage": "Stage 4",
                "name": "Sequential & Symbolic AI Planning",
                "owner": "Person B (Aanya)",
                "status": "complete",
                "inputs": "Real evaluator, recalibrated λ=116.0, existing-infra PDDL problem",
                "outputs": "MDP policy (22,136 states), 1,000 CRN rollouts, HTN 9-leaf trace, PDDL 7-step plan",
            },
        ],
        "data_sources": [
            {
                "name": "IHME Global Burden of Disease (GBD) 2023",
                "role": "1990–2023 HAP–COPD & total COPD DALYs, deaths, ASRs (5 countries)",
                "countries": "India, Nepal, Pakistan, Bangladesh, Bhutan",
                "status": "verified",
                "url": "https://vizhub.healthdata.org/gbd-results/",
            },
            {
                "name": "UN World Population Prospects 2024",
                "role": "Female/male population denominators & 2024–2040 demographic projections",
                "countries": "India, Nepal",
                "status": "verified",
                "url": "https://population.un.org/wpp/",
            },
            {
                "name": "WHO Global Health Observatory (PHE_HHAIR_PROP_POP_CLEAN_FUELS)",
                "role": "National, urban, rural population clean-cooking access (2000–2022)",
                "countries": "India, Nepal, Pakistan, Bangladesh, Bhutan",
                "status": "verified",
                "url": "https://www.who.int/data/gho/data/indicators/indicator-details/GHO/proportion-of-population-with-primary-reliance-on-clean-fuels-and-technologies-for-cooking-(-)",
            },
            {
                "name": "India NFHS-5 (2019–21) & Census 2011",
                "role": "Household clean-cooking baseline, grid readiness & household size",
                "countries": "India (Urban & Rural)",
                "status": "verified",
                "url": "https://dhsprogram.com/methodology/survey/survey-display-541.cfm",
            },
            {
                "name": "Nepal NDHS 2022 & NPHC 2021 Census",
                "role": "Household clean-cooking baseline, LPG/grid readiness & household size",
                "countries": "Nepal (Urban & Rural)",
                "status": "verified",
                "url": "https://dhsprogram.com/publications/publication-FR379-DHS-Final-Reports.cfm",
            },
            {
                "name": "PPAC / Union Budget & World Bank LPG Targeting Studies",
                "role": "India PAHAL/Ujjwala targeted LPG fiscal cost ($84.33–$90.30/HH)",
                "countries": "India",
                "status": "verified",
                "url": "https://ppac.gov.in/",
            },
            {
                "name": "Sharma et al. (2024) / AEPC Electric Cooking Nepal",
                "role": "Nepal LPG cost & Tier-2 improved biomass ($45/HH) / induction ($63.67/HH full)",
                "countries": "Nepal",
                "status": "verified",
                "url": "https://www.aepc.gov.np/",
            },
            {
                "name": "Pope et al. / WHO Indoor Air Quality Guidelines",
                "role": "Clean-fuel HAP exposure–response effectiveness weights (η)",
                "countries": "Global / South Asia",
                "status": "verified",
                "url": "https://www.who.int/publications/i/item/9789240034228",
            },
        ],
        "limitations": [
            {
                "id": "L1",
                "title": "GBD 2023 Modelled Estimates & Reduced-Form GLM Compromise",
                "detail": "GBD 2023 outputs are modelled epidemiological estimates rather than vital-registry observations. Because age-stratified HAP–COPD subnational exports were unavailable, Stage 2 uses a reduced-form Poisson rate GLM instead of full Age-Period-Cohort (APC) decomposition, keeping all-age DALY counts and ASRs distinct.",
            },
            {
                "id": "L2",
                "title": "B-12 Multi-Horizon CV, Fixed ETS & Counterfactual Separation",
                "detail": "Original Stage 2 ExponentialSmoothing.fit(disp=False) threw a TypeError causing silent fallback to ARIMA. B-12 removes disp=False and eliminates silent fallbacks (0 failures across 891 multi-horizon fits): at h=5 (both sex), ETS Undamped (MAE 47.73, +53.9% skill) and Damped (48.78, +52.8% skill) edge out ARIMA(1,1,0) (49.33, +52.3% skill). MASE > 1 at h=5 reflects dividing a 5-year cumulative horizon error by a 1-year in-sample step; horizon-matched relative MAE (< 0.48) confirms strong statistical skill. HistGBDT retains Stage 3 counterfactual duties because univariate models omit clean-fuel covariates.",
            },
            {
                "id": "L3",
                "title": "High-Coverage Plateau, Counterfactual Sensitivity & Hierarchical Reconciliation",
                "detail": "Both unconstrained and monotonic HistGBDT (monotonic_cst=[0,-1,0,0]) plateau above clean_fuel_share >= 0.96 (historical max = 0.958 in IND_urban), yielding slope = 0.0 in IND_urban. Smooth Log-OLS testing (-2.1933 slope) alters GA DALY-Minimal 2040 burden by only -1.08% without changing rural-first priorities. B-12 also enforces float64 pre-rounding additivity (max_pop_err = 0.0, max_count_err = 1.86e-9, weighted_rate_err = 2.27e-13) and compares Top-Down WHO GHE 2021 against Bottom-Up (+50,254 DALY bias) and OLS MinT.",
            },
            {
                "id": "L4",
                "title": "Unresolved Regional Infrastructure Capex (Scenario A vs. B1/B2)",
                "detail": "Regional LPG bulk storage and distribution-grid capex (infra_lpg_grid) has no authoritative verified cost and remains null in REAL_V1_2024USD. Scenario A enforces zero infra builds (keeping NPL_rural on improved_biomass), while Scenario B1/B2 test explicit proxy capex ($64.6M–$117.6M, unlocking +74.9k more DALYs).",
            },
            {
                "id": "L5",
                "title": "Population vs. Household Coverage Bridge & Alt B Recurring LPG Opex",
                "detail": "Exposure-response uses WHO population clean-fuel shares while adoption costing uses NFHS/NDHS household shares, connected via Δpop = bridge × Δhh (default 1.0; tested at 0.8 and 1.2). Alt A treats LPG subsidies as 1-time programme costs; Alt B (+$508.5M Knee, +$2,213.5M DALY-min) tests recurring P1–P3 refill opex.",
            },
            {
                "id": "L6",
                "title": "Proxy Sex Elasticity Ratio & Scenario Framing",
                "detail": "Female-vs-male clean-fuel responsiveness uses a uniform proxy elasticity ratio (r_fm = 1.05). All intervention costs and averted DALY projections represent comparative decision scenarios for academic evaluation, not prescriptive public budgets.",
            },
        ],
        "reproducibility": {
            "python_requirement": "Python 3.12+ (.venv)",
            "dependencies": [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "statsmodels",
                "pymoo",
                "pyperplan",
                "matplotlib",
                "seaborn",
                "pwlf",
                "pytest",
                "ruff",
            ],
            "seeds": config["seeds"],
            "commands": [
                {
                    "cmd": ".venv/bin/python -c \"from src.data.contracts import load_config, load_stage3_contracts; load_stage3_contracts(load_config('configs/default.yaml'))\"",
                    "desc": "Validate all processed Stage 1–3 contracts against configs/default.yaml",
                },
                {
                    "cmd": ".venv/bin/python -m src.optimise.run --config configs/default.yaml",
                    "desc": "Run 5-seed NSGA-II (251 Pareto plans) + 3 baselines + Stage 3 plots",
                },
                {
                    "cmd": ".venv/bin/python -m src.planning.run --config configs/default.yaml",
                    "desc": "Solve Finite MDP (λ=116.0), HTN tree & PDDL (pyperplan)",
                },
                {
                    "cmd": ".venv/bin/python -m src.planning.compare --config configs/default.yaml",
                    "desc": "Compare MDP vs. matched GA & PDDL over 1,000 CRN shock rollouts",
                },
                {
                    "cmd": ".venv/bin/python -m src.optimise.final_real_suite --config configs/default.yaml",
                    "desc": "Run 500-draw MC re-eval, 8-factor sensitivity (18 scenarios) & Alt B opex",
                },
                {
                    "cmd": ".venv/bin/python site/scripts/prepare_data.py",
                    "desc": "Rebuild offline website JSON payloads & downloadable asset bundle",
                },
                {
                    "cmd": "make test",
                    "desc": "Run 59 core/fixture tests (excluding external raw-archive test_stage1_pipeline_run)",
                },
            ],
            "verified_status": {
                "pytest": "59/59 core/fixture tests passing via `make test` (excluding 3 interim-parquet skipped + 1 raw-archive test when data/raw/* zips are absent)",
                "ruff_lint": "0 issues on src/ and tests/",
                "ruff_format": "clean",
            },
        },
        "downloads": downloads,
    }


def main() -> None:
    """Build all static JSON files and copy downloadable artifacts for the website."""
    config = yaml.safe_load((REPO_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    tdir = REPO_ROOT / "outputs/final_real_data/tables"

    selected_df = pd.read_csv(tdir / "s3_selected_plans.csv")
    reeval_df = pd.read_csv(tdir / "s3_pareto_reeval.csv")
    sens_df = pd.read_csv(tdir / "s3_sensitivity_real.csv")
    opex_df = pd.read_csv(tdir / "s3_recurring_opex_sensitivity.csv")
    comp_df = pd.read_csv(tdir / "comp_ga_vs_mdp.csv")

    contracts = load_stage3_contracts(config, base_dir=REPO_ROOT)
    evaluator = Stage3Evaluator(config=config, contracts=contracts)
    bau_eval = evaluator.evaluate(np.zeros(evaluator.n_vars, dtype=float))
    bau_2040 = {
        "dalys_2040_mean": round(float(np.asarray(bau_eval.dalys_2040_mean).item()), 4),
        "dalys_2040_lower": round(float(np.asarray(bau_eval.dalys_2040_lower).item()), 4),
        "dalys_2040_upper": round(float(np.asarray(bau_eval.dalys_2040_upper).item()), 4),
        "equity_gap_asr": round(float(np.asarray(bau_eval.equity_gap_asr).item()), 6),
        "equity_gap_dalys_abs": round(float(np.asarray(bau_eval.equity_gap_dalys_abs).item()), 4),
    }

    mdp_row = comp_df[comp_df["method"] == "mdp_adaptive_b6"].iloc[0].to_dict()

    metadata, overview = _build_metadata_and_overview(
        config, evaluator, selected_df, bau_2040, mdp_row, reeval_df
    )
    hist = _build_historical_clean_fuel()
    forecast = _build_forecast_real(config)
    assumptions = _build_assumptions(config, evaluator)
    optimise = _build_optimise_real(
        config, evaluator, bau_2040, selected_df, reeval_df, sens_df, opex_df
    )
    planning = _build_planning_real(config)
    downloads = _copy_downloads()
    research = _build_research(config, downloads)

    # Remove stale mock JSON files in site/data/ if present so only real-data JSONs remain active
    for stale_mock in DATA_OUT.glob("*_mock.json"):
        stale_mock.unlink()

    outputs = [
        _write_json("metadata.json", metadata),
        _write_json("overview.json", overview),
        _write_json("historical_clean_fuel.json", hist),
        _write_json("forecast_stage3.json", forecast),
        _write_json("assumptions.json", assumptions),
        _write_json("optimise.json", optimise),
        _write_json("planning.json", planning),
        _write_json("research.json", research),
    ]
    print(
        f"Wrote {len(outputs)} JSON payloads to {DATA_OUT} and {len(downloads)} downloads to {DOWNLOADS_OUT}."
    )


if __name__ == "__main__":
    main()
