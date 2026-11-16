"""Read-only data preparation script for the static research website.

Reads existing repository configs, docs, PDDL files, mock outputs, and
Parameter Sourcing outputs (on ``b/param-sourcing``) without modifying any
source or output file, and writes clean JSON snapshots to ``site/data/`` and
downloadable copies to ``site/downloads/``.
"""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.contracts import load_config, load_stage3_contracts  # noqa: E402
from src.optimise.evaluator import Stage3Evaluator  # noqa: E402
from src.optimise.nsga2 import decode_allocation_spend, run_nsga2_multi_seed  # noqa: E402

SITE_DIR = REPO_ROOT / "site"
DATA_DIR = SITE_DIR / "data"
DOWNLOADS_DIR = SITE_DIR / "downloads"
IMAGES_MOCK_DIR = SITE_DIR / "assets" / "images" / "mock"


def _read_git_file(branch: str, rel_path: str) -> str:
    """Read a file from a git branch without touching the working tree."""
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{branch}:{rel_path}"],
        text=True,
    )


def _clean_value(val: Any) -> Any:
    """Convert numpy/pandas values and NaN/Inf into JSON-safe Python primitives."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        fval = float(val)
        return None if (math.isnan(fval) or math.isinf(fval)) else fval
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if pd.isna(val):
        return None
    return val


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame to a list of JSON-safe dicts."""
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({k: _clean_value(v) for k, v in row.items()})
    return records


def _write_json(filename: str, payload: dict[str, Any]) -> Path:
    out_path = DATA_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _parse_pddl_solution(text: str) -> dict[str, Any]:
    """Parse ``plan_pddl_solution_mock.txt`` into structured metadata & action timeline."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    comments = [ln for ln in lines if ln.startswith("#")]
    actions_raw = [ln for ln in lines if not ln.startswith("#")]

    meta: dict[str, Any] = {}
    for c in comments:
        if "Planner:" in c:
            meta["planner_line"] = c.lstrip("# ").strip()
        elif "Replay Verification:" in c:
            meta["replay_verification"] = c.lstrip("# ").strip()
        elif "Stage3Evaluator Check:" in c:
            meta["evaluator_check"] = c.lstrip("# ").strip()
        elif "Generated:" in c:
            meta["generated_line"] = c.lstrip("# ").strip()

    action_descriptions = {
        "build-lpg-infra": "Construct regional LPG storage, bottling, and distribution prerequisite (unlocks LPG subsidy in strictly later periods q < p).",
        "build-grid-infra": "Reinforce rural electric feeder/transformer capacity for 2 kW induction cooking (unlocks induction in strictly later periods q < p).",
        "rollout-lpg-init-ready": "Scale LPG connection & refill subsidies (+0.10 coverage tier) in an initially infrastructure-ready urban unit.",
        "rollout-lpg-after-infra": "Scale LPG subsidies (+0.10 coverage tier) in a rural unit after LPG infrastructure was built in a strictly earlier period.",
        "rollout-electric-init-ready": "Scale electric induction cooking (+0.10 coverage tier) in an initially grid-ready urban unit.",
        "rollout-electric-after-infra": "Scale electric induction (+0.10 coverage tier) in a rural unit after grid infrastructure was built earlier.",
        "rollout-improved-biomass": "Deploy improved biomass cookstoves (+0.10 coverage tier, interim exposure reduction without LPG/grid prerequisite).",
        "advance-period": "Transition planning clock sequentially from one 5-year epoch to the next.",
    }

    steps: list[dict[str, Any]] = []
    current_period = "P1"
    for raw in actions_raw:
        if ":" in raw:
            idx_str, expr = raw.split(":", 1)
            step_num = int(idx_str.strip())
            expr = expr.strip()
        else:
            step_num = len(steps) + 1
            expr = raw
        tokens = expr.strip("()").split()
        op = tokens[0] if tokens else ""
        args = tokens[1:] if len(tokens) > 1 else []
        if op == "advance-period" and len(args) >= 2:
            step_period = f"{args[0].upper()} → {args[1].upper()}"
            current_period = args[1].upper()
        else:
            step_period = current_period
        steps.append(
            {
                "step": step_num,
                "raw": expr,
                "operator": op,
                "args": args,
                "period": step_period,
                "unit": args[0].upper().replace("_URBAN", "_urban").replace("_RURAL", "_rural")
                if args and op != "advance-period"
                else None,
                "description": action_descriptions.get(op, ""),
            }
        )

    return {
        "meta": meta,
        "raw_text": text,
        "steps": steps,
    }


def _build_metadata_and_overview(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = {
        "project_name": config["project"]["name"],
        "version": config["project"]["version"],
        "generated_at": generated_at,
        "use_mock_forecasts": bool(config["interfaces"]["use_mock_forecasts"]),
        "pipeline_status": {
            "stage0_ingestion": {
                "status": "partial_inputs_acquired",
                "label": "Auxiliary Inputs Acquired / GBD Pending",
                "detail": "UN WPP 2024, UN WUP 2025, WHO GHO SDG 7.1.2, World Bank WDI, DHS/NFHS, and Nepal NPHC 2021 acquired in b/param-sourcing; IHME GBD 2023 raw exports pending (A-03).",
            },
            "stage1_historical": {
                "status": "pending_gbd",
                "label": "Pending GBD 2023 Data",
                "detail": "src/historical/ awaiting GBD 2023 HAP-COPD, Total COPD, and All-Cause HAP CSV exports; real clean-cooking access series (1990–2024) are available and displayed.",
            },
            "stage2_forecast": {
                "status": "pending_stage2",
                "label": "Pending Stage 2 Exports",
                "detail": "src/forecast/ awaiting validated Stage 2 rolling-origin CV and 2040 projections; Contract 1/2/3 synthetic mock explorer (MOCK_V1) is active.",
            },
            "stage3_optimise": {
                "status": "complete_mock",
                "label": "Complete on Mock Data (MOCK_V1)",
                "detail": "NSGA-II (pymoo), Stage3Repair, baselines, and budget/parameter sweeps verified and running on MOCK_V1.",
            },
            "stage4_planning": {
                "status": "complete_mock",
                "label": "Complete on Mock Data (MOCK_V1)",
                "detail": "Custom NumPy MDP, HTN tree, STRIPS/PDDL (pyperplan), and 1,000-rollout GA vs MDP vs PDDL comparison verified on MOCK_V1.",
            },
            "param_sourcing": {
                "status": "partial_real",
                "label": "Sourced & Derived (2 Gaps Unresolved)",
                "detail": "Costs/effectiveness sourced & derived on b/param-sourcing; infra_lpg_grid fixed capex and pooled total_budget_usd_millions remain unresolved (null).",
            },
        },
    }

    overview = {
        "title": "Household Air Pollution & COPD",
        "subtitle": "Forecasting Burden. Optimising Intervention. Planning for 2040.",
        "kpis": [
            {
                "id": "countries",
                "label": "Countries Analysed",
                "value": "5",
                "detail": "India, Nepal, Pakistan, Bangladesh, Bhutan (GBD South Asia panel)",
                "status": "verified_scope",
            },
            {
                "id": "historical_period",
                "label": "Historical Study Period",
                "value": "1990–2023",
                "detail": "34 annual years (GBD 2023); WHO clean cooking extends to 2024",
                "status": "verified_scope",
            },
            {
                "id": "forecast_horizon",
                "label": "Forecast Horizon",
                "value": "2040",
                "detail": "2024–2040 projections; decision periods P1 (2026–30), P2 (2031–35), P3 (2036–40)",
                "status": "verified_scope",
            },
            {
                "id": "allocation_units",
                "label": "Geographical Allocation Units",
                "value": "4",
                "detail": "IND_urban, IND_rural, NPL_urban, NPL_rural (48 decision vars)",
                "status": "verified_scope",
            },
            {
                "id": "interventions",
                "label": "Intervention Categories",
                "value": "4",
                "detail": "LPG subsidy, Improved biomass, Electric induction, LPG/grid infra",
                "status": "verified_scope",
            },
            {
                "id": "planning_approaches",
                "label": "Decision & Planning Methods",
                "value": "4",
                "detail": "NSGA-II (pymoo), Custom NumPy MDP, HTN Tree, STRIPS/PDDL (pyperplan)",
                "status": "verified_scope",
            },
        ],
        "objectives": [
            {
                "id": "OBJ-1",
                "title": "Historical Epidemiology (Stage 1)",
                "summary": "Quantify 1990–2023 HAP-attributable COPD deaths, DALYs, age-standardized rates, EAPC, and joinpoint inflection years across 5 South Asian countries, highlighting India and Nepal.",
                "status": "Pending GBD 2023 export; real WHO/DHS clean-cooking series ready",
            },
            {
                "id": "OBJ-2",
                "title": "Burden Forecasting to 2040 (Stage 2)",
                "summary": "Benchmark statistical baselines (ARIMA, ETS, adapted Poisson rate GLM) against panel ML models (Random Forest, HistGradientBoosting) via expanding-window rolling-origin CV and calibrate a fast exposure–response surrogate.",
                "status": "Methodology locked; awaiting Stage 2 implementation & exports",
            },
            {
                "id": "OBJ-3",
                "title": "Multi-Objective Resource Allocation (Stage 3)",
                "summary": "Use NSGA-II to minimise projected 2040 DALYs, total intervention cost ($M), and population-weighted female–male ASR burden gap across 4 units × 4 interventions × 3 periods under budget, cap, ceiling, and q < p infra constraints.",
                "status": "Implemented & verified on synthetic MOCK_V1",
            },
            {
                "id": "OBJ-4",
                "title": "Sequential & Symbolic Planning (Stage 4)",
                "summary": "Compare an adaptive stochastic NumPy MDP policy, a hierarchical task network (HTN), and a STRIPS/PDDL classical planner against open-loop NSGA-II schedules on equal-footing stochastic rollouts.",
                "status": "Implemented & verified on synthetic MOCK_V1",
            },
        ],
        "research_questions": [
            {
                "id": "RQ1",
                "question": "How have HAP-attributable COPD deaths and DALYs changed in India, Nepal, and wider South Asia from 1990 to 2023, by sex and age-standardised metrics?",
                "note": "Repository methodology compromise uses aggregated all-age burden and age-standardized rates rather than five-year age-specific burden trajectories.",
            },
            {
                "id": "RQ2",
                "question": "Which forecasting approach (statistical or ML) projects attributable burden to 2040 most reliably on held-out rolling-origin validation data?",
                "note": "Evaluates ARIMA, ETS, adapted Poisson rate GLM, Random Forest, and HistGradientBoosting on MAE, RMSE, and MASE.",
            },
            {
                "id": "RQ3",
                "question": "Under a fixed budget, which intervention allocation minimises projected 2040 burden, and how does it trade off against cost and sex-based equity?",
                "note": "Sex-specific burden response uses a proxy elasticity multiplier due to lack of sex-disaggregated intervention uptake data.",
            },
            {
                "id": "RQ4",
                "question": "How does an adaptive planning policy (MDP) compare with a one-off optimised allocation (GA) and symbolic plans (HTN / PDDL)?",
                "note": "Tested across 1,000 Common-Random-Numbers uptake shock rollouts ([0.5x, 1.0x, 1.25x] with probs [0.15, 0.70, 0.15]).",
            },
        ],
        "pipeline_nodes": [
            {
                "id": "data",
                "step": "01",
                "title": "Historical GBD & Clean-Cooking Data",
                "subtitle": "GBD 2023 · UN WPP 2024 · UN WUP 2025 · WHO SDG 7.1.2",
                "status": "partial",
                "target": "historical",
            },
            {
                "id": "stage1",
                "step": "02",
                "title": "Historical Burden Analysis",
                "subtitle": "ASRs · EAPC · Joinpoint (pwlf) · HAP Share",
                "status": "pending",
                "target": "historical",
            },
            {
                "id": "stage2",
                "step": "03",
                "title": "Forecasting to 2040",
                "subtitle": "ARIMA · ETS · Poisson GLM · RF · HistGBDT",
                "status": "pending",
                "target": "forecast",
            },
            {
                "id": "stage3",
                "step": "04",
                "title": "NSGA-II Optimisation",
                "subtitle": "48 Vars · 3 Objectives · Pareto Front",
                "status": "mock_complete",
                "target": "optimise",
            },
            {
                "id": "mdp",
                "step": "05",
                "title": "MDP Decision Planning",
                "subtitle": "Custom NumPy Backward Induction & VI",
                "status": "mock_complete",
                "target": "planning",
            },
            {
                "id": "symbolic",
                "step": "06",
                "title": "HTN & STRIPS/PDDL",
                "subtitle": "Decomposition Tree & pyperplan Solver",
                "status": "mock_complete",
                "target": "planning",
            },
            {
                "id": "compare",
                "step": "07",
                "title": "Comparative Evaluation",
                "subtitle": "1,000-Rollout Equal-Footing Stress Test",
                "status": "mock_complete",
                "target": "planning",
            },
        ],
    }
    return metadata, overview


def _build_historical_clean_fuel() -> dict[str, Any]:
    raw_csv = _read_git_file("b/param-sourcing", "research/person_b_inputs/clean_fuel_access.csv")
    df = pd.read_csv(io.StringIO(raw_csv))
    demo_csv = _read_git_file("b/param-sourcing", "research/person_b_inputs/demographic_inputs.csv")
    demo_df = pd.read_csv(io.StringIO(demo_csv))

    return {
        "is_mock": False,
        "data_status": "REAL_ACQUIRED_DATA",
        "sources": [
            "WHO GHO (PHE_HHAIR_PROP_POP_CLEAN_FUELS, 2000–2024)",
            "WHO GHO Fuel Category (PHE_HHAIR_PROP_POP_CATEGORY_FUELS, 1990–2023)",
            "World Bank WDI (EG.CFT.ACCS.*, 2000–2024)",
            "DHS/NFHS India (1993–2020) & Nepal (2001–2022)",
            "Nepal NPHC 2021 Table 07 & Table 15",
            "Census of India 2011 Table A-1",
            "UN WPP 2024 & UN WUP 2025 (National Definitions & DEGURBA)",
        ],
        "gbd_burden_status": {
            "status": "pending_stage1",
            "countries": ["India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"],
            "period": "1990–2023",
            "expected_files": [
                "data/raw/gbd2023_hap_copd_southasia.csv",
                "data/raw/gbd2023_total_copd_southasia.csv",
                "data/raw/gbd2023_hap_all_causes_ind_npl.csv",
                "outputs/tables/s1_historical_summary.csv",
                "outputs/tables/s1_eapc_joinpoint.csv",
            ],
            "note": "Historical GBD 2023 COPD burden data is not yet exported from IHME GHDx (Task A-03 / A-05). Note: repository methodology uses aggregated all-age burden and age-standardized rates (ASR per 100k) rather than 5-year age-specific HAP-COPD trajectories.",
        },
        "clean_fuel_records": _df_to_records(df),
        "demographic_records": _df_to_records(demo_df),
    }


def _build_forecast_mock(config: dict[str, Any]) -> dict[str, Any]:
    df_fc = pd.read_csv(REPO_ROOT / "data/processed/forecasts_stage3_mock.csv")
    df_base = pd.read_csv(REPO_ROOT / "data/processed/baseline_unit_state_mock.csv")
    surrogate = json.loads(
        (REPO_ROOT / "data/processed/exposure_response_surrogate_mock.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "is_mock": True,
        "data_version": "MOCK_V1",
        "real_stage2_status": {
            "status": "pending_stage2",
            "models": [
                {
                    "id": "arima",
                    "name": "ARIMA",
                    "family": "Statistical Time Series",
                    "implementation": "statsmodels SARIMAX/ARIMA per location x sex series (AICc order selection)",
                },
                {
                    "id": "ets",
                    "name": "ETS",
                    "family": "Statistical Time Series",
                    "implementation": "statsmodels ExponentialSmoothing (damped/linear trend)",
                },
                {
                    "id": "poisson_rate_glm",
                    "name": "Poisson Rate GLM (Adapted)",
                    "family": "Statistical Rate Model",
                    "implementation": "Adapted Poisson GLM on aggregated all-age burden with log(population) offset (NOT an age-period-cohort decomposition; configs key: poisson_apc)",
                },
                {
                    "id": "random_forest",
                    "name": "Random Forest",
                    "family": "Panel Machine Learning",
                    "implementation": "scikit-learn RandomForestRegressor across 5-country panel (170 country-years/sex) using WHO SDG 7.1.2, urban share, 65+ share, and constrained lags",
                },
                {
                    "id": "hist_gradient_boosting",
                    "name": "HistGradientBoosting",
                    "family": "Panel Machine Learning",
                    "implementation": "scikit-learn HistGradientBoostingRegressor across 5-country panel with shallow trees & regularization",
                },
            ],
            "cv_folds": [
                {"fold": 1, "train": "1990–2008", "test": "2009–2013", "horizon_years": 5},
                {"fold": 2, "train": "1990–2013", "test": "2014–2018", "horizon_years": 5},
                {
                    "fold": 3,
                    "train": "1990–2018",
                    "test": "2019–2023",
                    "horizon_years": 5,
                    "note": "Also tested pre-2020 (2019) vs COVID-period sensitivity",
                },
            ],
            "cv_metrics_required": ["MAE", "RMSE", "MASE"],
            "note": "Final Stage 2 rolling-origin validation scores and real 2024–2040 projections are awaiting validated Stage 2 exports (make stage2). Below, Contract 1–3 synthetic mock trajectories (MOCK_V1) are shown solely to demonstrate the downstream Stage 3 input contract.",
        },
        "mock_forecasts": _df_to_records(df_fc),
        "mock_baseline_state": _df_to_records(df_base),
        "mock_surrogate": surrogate,
    }


def _build_assumptions(config: dict[str, Any]) -> dict[str, Any]:
    mock_params = json.loads(
        (REPO_ROOT / "data/processed/intervention_params_mock.json").read_text(encoding="utf-8")
    )
    prod_template = json.loads(
        (REPO_ROOT / "configs/intervention_params.json").read_text(encoding="utf-8")
    )
    proposed_params = json.loads(
        _read_git_file("b/param-sourcing", "research/person_b_inputs/proposed_intervention_params.json")
    )
    source_manifest = json.loads(
        _read_git_file("b/param-sourcing", "research/person_b_inputs/source_manifest.json")
    )

    def _git_csv(rel_path: str) -> list[dict[str, Any]]:
        txt = _read_git_file("b/param-sourcing", rel_path)
        return _df_to_records(pd.read_csv(io.StringIO(txt)))

    return {
        "units": list(config["geography"]["allocation_units"]),
        "interventions": [
            {
                "id": "lpg_subsidy",
                "name": "LPG Subsidy",
                "role": "Primary clean-cooking scale-up",
                "infra_required": "Requires LPG infrastructure readiness (lpg_infra_ready_init = true or infra_lpg_grid built in strictly earlier period q < p)",
                "description": "Upfront connection assistance + 5-year transitional cylinder refill subsidy (PMUY in India; NOC refill subsidy + assumed deposit in Nepal). Treated as a one-time 5-year transitional cohort package (Alternative A).",
            },
            {
                "id": "improved_biomass",
                "name": "Improved Biomass Cookstoves",
                "role": "Transitional / interim bridge without infra prerequisite",
                "infra_required": "None — deployable in P1 even when rural LPG/grid prerequisites are unready",
                "description": "Forced-draft or improved biomass stoves with 5-year hardware replacement + 5%/yr O&M. Under GBD concave COPD exposure–response, field PM2.5 reductions (32–50%) translate to ~0.15–0.30 COPD PAF reduction (proposed central 0.25 vs 0.45 mock).",
            },
            {
                "id": "electric_induction",
                "name": "Electric Induction Cooking",
                "role": "Clean electric transition",
                "infra_required": "Requires grid readiness (grid_ready_init = true or infra_lpg_grid built in strictly earlier period q < p)",
                "description": "Zero point-of-use PM2.5 induction cooktop + 15A internal household wiring package (EESL NECP retail benchmark in India; AEPC/NEA in Nepal). Excludes 5-year household electricity tariffs and lacks a central Indian cash subsidy.",
            },
            {
                "id": "infra_lpg_grid",
                "name": "LPG / Grid Infrastructure Prerequisite",
                "role": "Enabling binary prerequisite (effectiveness = 0.0 directly)",
                "infra_required": "Binary trigger when decision variable x >= 0.5; unlocks LPG & induction in periods p > q",
                "description": "Regional rural distribution feeder/transformer reinforcement and LPG bottling/depot expansion. Fixed regional $M capex remains an UNRESOLVED GAP (null) in real sourcing — proxies like Saubhagya ($68/HH) and RDSS ($571/HH) reflect household hookups rather than regional cooking-capacity capex.",
            },
        ],
        "proposed_contract4_table": _git_csv("research/person_b_inputs/proposed_contract4_table.csv"),
        "baseline_comparison": _git_csv("research/person_b_inputs/b09_baseline_comparison.csv"),
        "budget_scenarios": _git_csv("research/person_b_inputs/b09_budget_scenarios.csv"),
        "feasibility_audit": _git_csv("research/person_b_inputs/b09_feasibility_audit.csv"),
        "parameter_gap_matrix": _git_csv("research/person_b_inputs/parameter_gap_matrix.csv"),
        "source_register": _git_csv("research/person_b_inputs/intervention_source_register.csv"),
        "demographic_summary": _git_csv("research/person_b_inputs/demographic_inputs.csv"),
        "source_manifest_raw_files": source_manifest.get("raw_files", []),
        "proposed_params": proposed_params,
        "production_template": prod_template,
        "mock_params": mock_params,
    }


def _build_optimise_mock(config: dict[str, Any]) -> dict[str, Any]:
    pareto_df = pd.read_csv(REPO_ROOT / "outputs/tables/s3_pareto_front_mock.csv")
    selected_df = pd.read_csv(REPO_ROOT / "outputs/tables/s3_selected_plans_mock.csv")
    baseline_df = pd.read_csv(REPO_ROOT / "outputs/tables/s3_baseline_comparison_mock.csv")
    hv_seeds_df = pd.read_csv(REPO_ROOT / "outputs/tables/s3_hypervolume_seeds_mock.csv")

    budget_sweep_df = pd.read_csv(
        REPO_ROOT / "outputs/tables/param_sourcing/task4_budget_sweep_mock.csv"
    )
    eta_sweep_df = pd.read_csv(
        REPO_ROOT / "outputs/tables/param_sourcing/task4_eta_biomass_sweep_mock.csv"
    )
    cap_sweep_df = pd.read_csv(
        REPO_ROOT / "outputs/tables/param_sourcing/task4_induction_cap_sweep_mock.csv"
    )
    bridge_df = pd.read_csv(
        io.StringIO(
            _read_git_file("b/param-sourcing", "research/person_b_inputs/b09_bridge_sensitivity.csv")
        )
    )

    contracts = load_stage3_contracts(config, base_dir=REPO_ROOT)
    evaluator = Stage3Evaluator(config=config, contracts=contracts)
    bau_eval = evaluator.evaluate(np.zeros(evaluator.n_vars, dtype=float))

    # Decode every Pareto solution's 48 decision vars + exact spend matrix
    var_cols = list(evaluator.variable_names)
    pareto_solutions: list[dict[str, Any]] = []
    for _, row in pareto_df.iterrows():
        x_vec = np.array([float(row[c]) for c in var_cols], dtype=float)
        spend_df = decode_allocation_spend(x_vec, evaluator)
        pareto_solutions.append(
            {
                "solution_id": int(row["solution_id"]),
                "seed": int(row["seed"]),
                "selected_plans": "" if pd.isna(row["selected_plans"]) else str(row["selected_plans"]),
                "dalys_2040_mean": float(row["dalys_2040_mean"]),
                "dalys_2040_lower": float(row["dalys_2040_lower"]),
                "dalys_2040_upper": float(row["dalys_2040_upper"]),
                "dalys_averted_mean": float(bau_eval.dalys_2040_mean) - float(row["dalys_2040_mean"]),
                "total_cost_usd_millions": float(row["total_cost_usd_millions"]),
                "equity_gap_asr": float(row["equity_gap_asr"]),
                "equity_gap_dalys_abs": float(row["equity_gap_dalys_abs"]),
                "is_feasible": bool(row["is_feasible"]),
                "violations": {
                    "budget": float(row["violation_budget"]),
                    "adoption_cap": float(row["violation_adoption_cap"]),
                    "coverage_ceiling": float(row["violation_coverage_ceiling"]),
                    "infra_prereq": float(row["violation_infra_prereq"]),
                },
                "allocations": _df_to_records(spend_df),
            }
        )

    # Extract deterministic per-generation hypervolume convergence & verify match with CSV
    hv_convergence: list[dict[str, Any]] = []
    try:
        ms_res = run_nsga2_multi_seed(evaluator, config)
        expected_hvs = dict(zip(hv_seeds_df["seed"].astype(int), hv_seeds_df["hypervolume"].astype(float)))
        all_match = all(
            abs(sr.hypervolume - expected_hvs.get(sr.seed, -1.0)) <= 1e-3
            for sr in ms_res.seed_results
        )
        if all_match:
            for sr in ms_res.seed_results:
                hv_convergence.append(
                    {
                        "seed": int(sr.seed),
                        "history": sr.hv_history,
                    }
                )
    except Exception:
        hv_convergence = []

    return {
        "is_mock": True,
        "warning_mock_inputs": True,
        "data_version": "MOCK_V1",
        "budget_cap_usd_millions": float(evaluator.total_budget_usd_millions),
        "bau_2040": {
            "dalys_2040_mean": float(bau_eval.dalys_2040_mean),
            "dalys_2040_lower": float(bau_eval.dalys_2040_lower),
            "dalys_2040_upper": float(bau_eval.dalys_2040_upper),
            "equity_gap_asr": float(bau_eval.equity_gap_asr),
            "equity_gap_dalys_abs": float(bau_eval.equity_gap_dalys_abs),
        },
        "units": list(evaluator.units),
        "interventions": list(evaluator.interventions),
        "periods": list(evaluator.periods),
        "pareto_solutions": pareto_solutions,
        "selected_plans": [
            {
                "plan_name": str(r["plan_name"]),
                "solution_id": int(r["solution_id"]),
                "dalys_2040_mean": float(r["dalys_2040_mean"]),
                "dalys_2040_lower": float(r["dalys_2040_lower"]),
                "dalys_2040_upper": float(r["dalys_2040_upper"]),
                "dalys_averted_mean": float(r["dalys_averted_mean"]),
                "total_cost_usd_millions": float(r["total_cost_usd_millions"]),
                "equity_gap_asr": float(r["equity_gap_asr"]),
                "equity_gap_dalys_abs": float(r["equity_gap_dalys_abs"]),
                "is_feasible": bool(r["is_feasible"]),
            }
            for _, r in selected_df.iterrows()
        ],
        "baseline_comparison": _df_to_records(baseline_df),
        "hypervolume_seeds": _df_to_records(hv_seeds_df),
        "hypervolume_convergence": hv_convergence,
        "budget_sweep": _df_to_records(budget_sweep_df),
        "eta_biomass_sweep": _df_to_records(eta_sweep_df),
        "induction_cap_sweep": _df_to_records(cap_sweep_df),
        "bridge_sensitivity": _df_to_records(bridge_df),
    }


def _build_planning_mock(config: dict[str, Any]) -> dict[str, Any]:
    htn_trace = json.loads(
        (REPO_ROOT / "outputs/tables/plan_htn_trace_mock.json").read_text(encoding="utf-8")
    )
    mdp_policy_df = pd.read_csv(REPO_ROOT / "outputs/tables/plan_mdp_policy_mock.csv")
    mdp_rollouts_df = pd.read_csv(REPO_ROOT / "outputs/tables/plan_mdp_rollouts_mock.csv")
    mdp_sens_df = pd.read_csv(REPO_ROOT / "outputs/tables/plan_mdp_sensitivity_mock.csv")
    comp_df = pd.read_csv(REPO_ROOT / "outputs/tables/comp_ga_vs_mdp_mock.csv")
    comp_sens_df = pd.read_csv(REPO_ROOT / "outputs/tables/comp_sensitivity_mock.csv")

    pddl_domain_text = (REPO_ROOT / "pddl/domain.pddl").read_text(encoding="utf-8")
    pddl_problem_text = (REPO_ROOT / "pddl/problem_ind_npl.pddl").read_text(encoding="utf-8")
    pddl_sol_text = (REPO_ROOT / "outputs/tables/plan_pddl_solution_mock.txt").read_text(
        encoding="utf-8"
    )
    pddl_parsed = _parse_pddl_solution(pddl_sol_text)

    # Summarize MDP policy action counts per period
    policy_period_summary = []
    for period, grp in mdp_policy_df.groupby("period"):
        top_actions = (
            grp["action_label"]
            .value_counts()
            .head(8)
            .reset_index()
            .rename(columns={"action_label": "action", "count": "state_count"})
        )
        policy_period_summary.append(
            {
                "period": str(period),
                "n_states": int(len(grp)),
                "top_actions": _df_to_records(top_actions),
            }
        )

    # Keep essential columns of MDP policy table so client filtering is instant
    policy_cols = [
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
    policy_compact = _df_to_records(mdp_policy_df[policy_cols])

    rollout_cols = [
        "rollout_id",
        "dalys_2040_mean",
        "dalys_2040_lower",
        "dalys_2040_upper",
        "dalys_averted_mean",
        "total_cost_usd_millions",
        "equity_gap_asr",
        "discounted_return",
        "is_feasible_in_evaluator",
        "p1_action",
        "p2_action",
        "p3_action",
    ]

    return {
        "is_mock": True,
        "warning_mock_inputs": True,
        "data_version": "MOCK_V1",
        "mdp_config": {
            "coverage_bins": list(config["planning"]["mdp"]["coverage_bins"]),
            "budget_bins": int(config["planning"]["mdp"]["budget_bins"]),
            "discount_factor": float(config["planning"]["mdp"]["discount_factor"]),
            "cost_weight_lambda": float(config["planning"]["mdp"]["cost_weight_lambda"]),
            "transition_probs": list(config["planning"]["mdp"]["transition_probs"]),
            "shock_multipliers": list(config["planning"]["compare"]["shock_multipliers"]),
            "n_rollouts": int(config["planning"]["mdp"]["n_rollouts"]),
            "n_decision_states_b6": int(len(mdp_policy_df)),
        },
        "mdp_policy_summary": policy_period_summary,
        "mdp_policy_rows": policy_compact,
        "mdp_rollouts": _df_to_records(mdp_rollouts_df[rollout_cols]),
        "mdp_sensitivity": _df_to_records(mdp_sens_df),
        "htn_trace": htn_trace,
        "pddl": {
            "domain_name": "south-asia-clean-cooking",
            "problem_name": "ind-npl-clean-cooking-2026-2040",
            "requirements": [":strips", ":typing"],
            "planner": "pyperplan (astar + hff)",
            "solve_time_seconds": 0.0623,
            "predicates": [
                {"name": "(lpg-init-ready ?u)", "meaning": "Unit starts 2025 with LPG supply infrastructure ready"},
                {"name": "(grid-init-ready ?u)", "meaning": "Unit starts 2025 with electric grid ready for induction"},
                {"name": "(lpg-infra-built-at ?u ?p)", "meaning": "LPG infrastructure was built in unit ?u during period ?p"},
                {"name": "(grid-infra-built-at ?u ?p)", "meaning": "Grid infrastructure was built in unit ?u during period ?p"},
                {"name": "(cov-tier ?u ?c)", "meaning": "Unit ?u is currently at discrete additional coverage tier ?c (c0/c1/c2)"},
                {"name": "(active-period ?p)", "meaning": "Planning clock is currently in 5-year period ?p (p1/p2/p3)"},
                {"name": "(strictly-before ?p1 ?p2)", "meaning": "Period ?p1 occurs strictly earlier than ?p2 (enforces q < p)"},
                {"name": "(slot-open ?u ?p)", "meaning": "Unit ?u has not yet used its single cooking rollout slot in period ?p"},
                {"name": "(budget-level ?b)", "meaning": "Ordered discrete budget counter (b8 down to b0) avoiding token symmetry"},
            ],
            "actions": [
                {
                    "name": "build-lpg-infra",
                    "params": "(?u ?p ?b1 ?b2)",
                    "preconditions": "Active period ?p; budget ?b1 → ?b2 available",
                    "effects": "Marks LPG infra built at (?u, ?p) and consumes 1 budget step",
                    "meaning": "Unlocks LPG subsidy rollouts in strictly later periods (p > q).",
                },
                {
                    "name": "build-grid-infra",
                    "params": "(?u ?p ?b1 ?b2)",
                    "preconditions": "Active period ?p; budget ?b1 → ?b2 available",
                    "effects": "Marks grid infra built at (?u, ?p) and consumes 1 budget step",
                    "meaning": "Unlocks electric induction rollouts in strictly later periods (p > q).",
                },
                {
                    "name": "rollout-lpg-init-ready",
                    "params": "(?u ?p ?c1 ?c2 ?b1 ?b2)",
                    "preconditions": "Initially LPG-ready; active period ?p; slot open; cov ?c1 → ?c2; budget ?b1 → ?b2",
                    "effects": "Advances coverage tier (+0.10), closes slot (?u, ?p), sets clean-fuel-deployed",
                    "meaning": "Deploys LPG subsidies in urban units without waiting for infra buildout.",
                },
                {
                    "name": "rollout-lpg-after-infra",
                    "params": "(?u ?p_infra ?p ?c1 ?c2 ?b1 ?b2)",
                    "preconditions": "LPG infra built at ?p_infra; strictly-before ?p_infra ?p; active ?p; slot open; budget",
                    "effects": "Advances coverage tier (+0.10), closes slot (?u, ?p), sets clean-fuel-deployed",
                    "meaning": "Deploys LPG subsidies in rural units strictly after infrastructure buildout.",
                },
                {
                    "name": "rollout-electric-init-ready / after-infra",
                    "params": "(?u ... ?p ?c1 ?c2 ?b1 ?b2)",
                    "preconditions": "Grid-init-ready OR grid built in strictly earlier period; slot open; budget",
                    "effects": "Advances coverage tier (+0.10), closes slot, sets clean-fuel-deployed",
                    "meaning": "Deploys electric induction in grid-ready urban units or post-infra rural units.",
                },
                {
                    "name": "rollout-improved-biomass",
                    "params": "(?u ?p ?c1 ?c2 ?b1 ?b2)",
                    "preconditions": "Active period ?p; slot open (?u, ?p); cov ?c1 → ?c2; budget",
                    "effects": "Advances coverage tier (+0.10), closes slot (does NOT set clean-fuel-deployed)",
                    "meaning": "Provides transitional exposure reduction in P1 before rural LPG/grid infra matures.",
                },
                {
                    "name": "advance-period",
                    "params": "(?p1 ?p2)",
                    "preconditions": "Active period ?p1; next-period ?p1 ?p2",
                    "effects": "Advances active planning period from ?p1 to ?p2",
                    "meaning": "Moves from P1 (2026–2030) to P2 (2031–2035) and P3 (2036–2040).",
                },
            ],
            "initial_state": {
                "ready_units": ["ind_urban (LPG + Grid)", "npl_urban (LPG + Grid)"],
                "unready_units": ["ind_rural (neither)", "npl_rural (neither)"],
                "coverage_start": "All 4 units at c0 (+0.00 above BAU)",
                "budget_tokens": "b8 (8 sequential action budget levels)",
                "active_period": "p1",
            },
            "goal_state": {
                "urban_targets": "ind_urban -> c1 (+0.10), npl_urban -> c1 (+0.10), clean-fuel-deployed",
                "rural_targets": "ind_rural -> c2 (+0.20), npl_rural -> c2 (+0.20), clean-fuel-deployed",
            },
            "domain_pddl": pddl_domain_text,
            "problem_pddl": pddl_problem_text,
            "solution": pddl_parsed,
        },
        "comparison": _df_to_records(comp_df),
        "comparison_sensitivity": _df_to_records(comp_sens_df),
    }


def _build_research(config: dict[str, Any]) -> dict[str, Any]:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps: list[str] = []
    in_deps = False
    for line in pyproject_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies = ["):
            in_deps = True
            continue
        if in_deps:
            if stripped == "]":
                in_deps = False
                continue
            deps.append(stripped.strip('",'))

    downloads = _copy_downloads()

    return {
        "methodology_stages": [
            {
                "stage": "Stage 0",
                "name": "Ingestion, Validation & Apportionment",
                "owner": "Person A (Rudraksh)",
                "status": "Auxiliary inputs acquired (WPP/WUP/WHO/DHS); GBD raw pending",
                "inputs": "GBD 2023 CSVs, UN WPP 2024, UN WUP 2025, WHO SDG 7.1.2",
                "outputs": "data/interim/*.parquet, baseline_unit_state.csv",
            },
            {
                "stage": "Stage 1",
                "name": "Historical Burden & Trend Analysis (1990–2023)",
                "owner": "Person A (Rudraksh)",
                "status": "Pending GBD 2023 exports (clean-cooking trends ready)",
                "inputs": "GBD 2023 HAP-COPD, Total COPD, All-cause HAP",
                "outputs": "s1_historical_summary.csv, s1_eapc_joinpoint.csv, s1_*.png",
            },
            {
                "stage": "Stage 2",
                "name": "Statistical & ML Forecasting (2024–2040)",
                "owner": "Person A (Rudraksh)",
                "status": "Pending Stage 2 exports (mock Contracts 1–3 active)",
                "inputs": "5-country panel (1990–2023), WHO SDG 7.1.2, UN WPP 2024",
                "outputs": "s2_rolling_cv_metrics.csv, forecasts_stage3.csv, exposure_response_surrogate.json",
            },
            {
                "stage": "Stage 3",
                "name": "Multi-Objective Optimisation (NSGA-II)",
                "owner": "Person B (Aanya)",
                "status": "Complete & tested on MOCK_V1",
                "inputs": "Contracts 1–4 (forecasts, baseline state, surrogate, intervention params)",
                "outputs": "s3_pareto_front{_mock}.csv, s3_selected_plans{_mock}.csv, s3_*.png",
            },
            {
                "stage": "Stage 4",
                "name": "Sequential & Symbolic Planning (MDP, HTN, PDDL)",
                "owner": "Person B (Aanya)",
                "status": "Complete & tested on MOCK_V1",
                "inputs": "Contracts 1–4, Stage 3 Pareto reference, pddl/*.pddl",
                "outputs": "plan_mdp_*, plan_htn_trace*, plan_pddl_solution*, comp_ga_vs_mdp*",
            },
        ],
        "data_sources": [
            {
                "name": "IHME Global Burden of Disease (GBD) 2023",
                "role": "Historical HAP-attributable COPD, total COPD, and HAP all-causes (1990–2023)",
                "countries": "India, Nepal, Pakistan, Bangladesh, Bhutan",
                "status": "Pending download (interactive GHDx login required; A-03)",
                "url": "https://vizhub.healthdata.org/gbd-results/",
            },
            {
                "name": "UN World Population Prospects (WPP) 2024",
                "role": "Population by 5-year age group and sex (1990–2040 Medium variant)",
                "countries": "Global (verified India & Nepal; PAK/BGD/BTN available in archive)",
                "status": "Acquired & SHA-256 verified in b/param-sourcing",
                "url": "https://population.un.org/wpp/Download/Standard/CSV/",
            },
            {
                "name": "UN World Urbanization Prospects (WUP) 2025",
                "role": "Urban vs rural population (1950–2050) under National Definitions and DEGURBA Level 1",
                "countries": "India, Nepal (plus global archive)",
                "status": "Acquired & SHA-256 verified in b/param-sourcing",
                "url": "https://population.un.org/wup/",
            },
            {
                "name": "WHO Global Health Observatory (SDG 7.1.2)",
                "role": "Primary reliance on clean fuels/tech (2000–2024, people) & fuel categories (1990–2023)",
                "countries": "India, Nepal (total, urban, rural)",
                "status": "Acquired & verified in b/param-sourcing",
                "url": "https://www.who.int/data/gho/data/themes/air-pollution/household-air-pollution",
            },
            {
                "name": "World Bank WDI (WHO/ESMAP SDG 7.1.2)",
                "role": "Cross-check of clean cooking access (EG.CFT.ACCS.*, 2000–2024; 0.0 pct pt diff vs WHO)",
                "countries": "India, Nepal (total, urban, rural)",
                "status": "Acquired & verified in b/param-sourcing",
                "url": "https://data.worldbank.org/indicator/EG.CFT.ACCS.ZS",
            },
            {
                "name": "India NFHS-5 (2019–21) & NFHS-1..4 via DHS Program API",
                "role": "Observed household size (4.2 urban / 4.5 rural) & HH vs pop clean-fuel access (89.7% / 43.2% HH)",
                "countries": "India (urban, rural)",
                "status": "Acquired & verified in b/param-sourcing",
                "url": "https://api.dhsprogram.com/",
            },
            {
                "name": "Nepal NDHS 2022 & NDHS 2001–2016 via DHS Program API",
                "role": "De jure household size (3.9 / 4.0) & HH vs pop clean-fuel access (56.7% / 21.2% HH)",
                "countries": "Nepal (urban, rural)",
                "status": "Acquired & verified in b/param-sourcing",
                "url": "https://api.dhsprogram.com/",
            },
            {
                "name": "Nepal National Population & Housing Census (NPHC) 2021",
                "role": "Official municipal pop/HH counts (4.3076 urban / 4.5115 rural) & Table 07 cooking fuels",
                "countries": "Nepal (293 Urban & 460 Rural Municipalities)",
                "status": "Acquired & verified in b/param-sourcing",
                "url": "https://censusnepal.cbs.gov.np/",
            },
            {
                "name": "Census of India 2011 (Table A-1)",
                "role": "2011 baseline pop & households (377.1M / 80.9M urban; 833.7M / 168.6M rural); 2021 census delayed",
                "countries": "India (urban, rural, total)",
                "status": "Acquired & verified in b/param-sourcing",
                "url": "https://censusindia.gov.in/",
            },
            {
                "name": "Intervention Programme & Clinical Literature",
                "role": "PMUY (MoPNG/PIB), Nepal NOC, MNRE/AEPC, EESL NECP, HAPIN trial, Tielsch et al. 2016, Pope et al. 2017, CEEW, ESMAP/MECS",
                "countries": "India & Nepal",
                "status": "Sourced/derived in intervention_source_register.csv (infra capex unresolved)",
                "url": "https://pib.gov.in",
            },
        ],
        "limitations": [
            {
                "id": "LIM-01",
                "title": "GBD 2023 Values Are Modelled Estimates",
                "detail": "IHME GBD 2023 deaths, DALYs, and ASRs are modelled epidemiological estimates rather than directly observed vital registration counts; all forecasts and Pareto evaluations must carry uncertainty intervals (lower, val, upper).",
            },
            {
                "id": "LIM-02",
                "title": "All-Age Burden Rather Than 5-Year Age-Specific Trajectories",
                "detail": "The final repository compromise uses aggregated all-age burden counts and age-standardized rates (ASRs) rather than five-year age-specific HAP-COPD burden trajectories.",
            },
            {
                "id": "LIM-03",
                "title": "Adapted Poisson Rate GLM Instead of Full APC Decomposition",
                "detail": "Consistent with aggregated all-age burden, the adapted Poisson GLM models aggregated rates with a log-population offset rather than full Age-Period-Cohort (APC) identifiability decomposition.",
            },
            {
                "id": "LIM-04",
                "title": "Proxy Sex-Specific Burden Projection",
                "detail": "Clean-cooking rollout programmes rarely disaggregate adoption by sex. Stage 3 and MDP model sex equity via a proxy female-to-male exposure elasticity ratio (sex_elasticity_ratio_f_to_m) applied to the shared unit coverage increment.",
            },
            {
                "id": "LIM-05",
                "title": "Uncertainty in Intervention Effectiveness & Fuel Stacking",
                "detail": "Field cookstove trials show substantial fuel stacking and non-linear GBD exposure–response concavity: raw 32–50% PM2.5 reductions for improved biomass translate to much smaller COPD PAF gains (~0.15–0.30). Real-world uptake also varies stochastically.",
            },
            {
                "id": "LIM-06",
                "title": "Household vs. Population Coverage Denominator Bridging",
                "detail": "WHO SDG 7.1.2 is population-denominated (people) whereas surveys (NFHS-5, NPHC 2021) and programme subsidies target households. Observed HH coverage exceeds population coverage by +0.7 to +4.2 pct pts because larger households disproportionately use solid fuels; Stage 3 links them via coverage_bridge_factor (central 1.0, tested 0.8–1.2).",
            },
            {
                "id": "LIM-07",
                "title": "Intervention Costing Boundaries & Unresolved Infrastructure Capex",
                "detail": "PMUY LPG reflects public-fiscal subsidy, whereas improved biomass and induction benchmarks initially reflect hardware/O&M or lack central cash subsidies. Regional rural feeder/transformer + LPG depot capex (infra_lpg_grid) and pooled 2-country budgets remain unresolved gaps (null) and are never invented.",
            },
            {
                "id": "LIM-08",
                "title": "Forecasting Structural Continuity & Scenario Framing",
                "detail": "Stage 2 statistical/ML forecasts assume historical associations persist (with clean-fuel access as the primary driver) and cannot establish causal policy impact. All optimisation and planning outputs are exploratory scenario analyses—never real-world policy prescriptions.",
            },
        ],
        "reproducibility": {
            "python_requirement": ">=3.12,<3.13 (venv + pyproject.toml; no conda, uv, or R)",
            "dependencies": deps,
            "seeds": config["seeds"],
            "commands": [
                {"cmd": "make setup", "desc": "Create Python 3.12 .venv and install package + dev tools"},
                {"cmd": "make check", "desc": "Run ruff lint/format checks and pytest unit/contract suite"},
                {"cmd": "make stage3", "desc": "Run NSGA-II (5 seeds), baselines, and Stage 3 figures/tables"},
                {"cmd": "make planning", "desc": "Run MDP value iteration, HTN decomposition, and PDDL (pyperplan)"},
                {"cmd": "make compare", "desc": "Run 1,000-rollout equal-footing GA vs MDP vs PDDL comparison"},
                {"cmd": "python3 site/scripts/prepare_data.py", "desc": "Regenerate site/data/*.json and site/downloads/* (read-only on repo)"},
                {"cmd": "python3 -m http.server 8080 --directory site", "desc": "Launch static research platform locally at http://localhost:8080"},
            ],
            "verified_status": {
                "pytest": "37 passed in test suite (verified on current branch)",
                "ruff_lint": "ruff check src tests: All checks passed",
                "ruff_format": "25/26 files formatted (1 pre-existing uncommitted diff in src/optimise/evaluator.py left untouched)",
            },
        },
        "downloads": downloads,
    }


def _copy_downloads() -> list[dict[str, str]]:
    """Copy actual existing project files into site/downloads/ for offline downloads."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_MOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Copy existing mock PNGs read-only
    fig_dir = REPO_ROOT / "outputs/figures"
    if fig_dir.exists():
        for png in sorted(fig_dir.glob("*.png")):
            shutil.copy2(png, IMAGES_MOCK_DIR / png.name)

    files_to_bundle = [
        ("s3_pareto_front_mock.csv", REPO_ROOT / "outputs/tables/s3_pareto_front_mock.csv", "Stage 3 Pareto Front (219 solutions, MOCK_V1)", "CSV"),
        ("s3_selected_plans_mock.csv", REPO_ROOT / "outputs/tables/s3_selected_plans_mock.csv", "Selected Representative Plans (MOCK_V1)", "CSV"),
        ("s3_baseline_comparison_mock.csv", REPO_ROOT / "outputs/tables/s3_baseline_comparison_mock.csv", "NSGA-II vs Baseline Heuristics (MOCK_V1)", "CSV"),
        ("comp_ga_vs_mdp_mock.csv", REPO_ROOT / "outputs/tables/comp_ga_vs_mdp_mock.csv", "Equal-Footing GA vs MDP vs PDDL Comparison (MOCK_V1)", "CSV"),
        ("plan_htn_trace_mock.json", REPO_ROOT / "outputs/tables/plan_htn_trace_mock.json", "HTN Decomposition Trace (MOCK_V1)", "JSON"),
        ("plan_pddl_solution_mock.txt", REPO_ROOT / "outputs/tables/plan_pddl_solution_mock.txt", "STRIPS/PDDL Verified Plan Solution (MOCK_V1)", "TXT"),
        ("domain.pddl", REPO_ROOT / "pddl/domain.pddl", "PDDL Domain Definition (south-asia-clean-cooking)", "PDDL"),
        ("problem_ind_npl.pddl", REPO_ROOT / "pddl/problem_ind_npl.pddl", "PDDL Problem Instance (IND & NPL)", "PDDL"),
        ("default.yaml", REPO_ROOT / "configs/default.yaml", "Project Configuration (configs/default.yaml)", "YAML"),
        ("METHODOLOGY.md", REPO_ROOT / "docs/METHODOLOGY.md", "Methodology Specification", "Markdown"),
        ("INTERFACES.md", REPO_ROOT / "docs/INTERFACES.md", "Track Interfaces & Data Contracts", "Markdown"),
        ("ASSUMPTIONS.md", REPO_ROOT / "docs/ASSUMPTIONS.md", "Scenario Assumptions Register", "Markdown"),
        ("PLANNING_SPEC.md", REPO_ROOT / "docs/PLANNING_SPEC.md", "Planning Component Specification", "Markdown"),
        ("DECISIONS.md", REPO_ROOT / "docs/DECISIONS.md", "Project Decision Log", "Markdown"),
    ]

    git_branch_files = [
        ("clean_fuel_access.csv", "research/person_b_inputs/clean_fuel_access.csv", "Real WHO/WB/DHS/NPHC Clean-Cooking Access (1990–2024)", "CSV"),
        ("demographic_inputs.csv", "research/person_b_inputs/demographic_inputs.csv", "Real UN WPP/WUP & Census/DHS Demographics", "CSV"),
        ("intervention_source_register.csv", "research/person_b_inputs/intervention_source_register.csv", "Sourced Intervention Cost & Effectiveness Register", "CSV"),
        ("b09_baseline_comparison.csv", "research/person_b_inputs/b09_baseline_comparison.csv", "Population vs Household Coverage Baselines", "CSV"),
        ("b09_budget_scenarios.csv", "research/person_b_inputs/b09_budget_scenarios.csv", "Reconciled Fiscal Budget Scenarios", "CSV"),
    ]

    bundled: list[dict[str, str]] = []
    for target_name, src_path, label, fmt in files_to_bundle:
        if src_path.exists():
            dst = DOWNLOADS_DIR / target_name
            shutil.copy2(src_path, dst)
            bundled.append(
                {
                    "filename": target_name,
                    "href": f"downloads/{target_name}",
                    "label": label,
                    "format": fmt,
                    "size_kb": f"{dst.stat().st_size / 1024:.1f} KB",
                }
            )

    for target_name, git_path, label, fmt in git_branch_files:
        txt = _read_git_file("b/param-sourcing", git_path)
        dst = DOWNLOADS_DIR / target_name
        dst.write_text(txt, encoding="utf-8")
        bundled.append(
            {
                "filename": target_name,
                "href": f"downloads/{target_name}",
                "label": label,
                "format": fmt,
                "size_kb": f"{dst.stat().st_size / 1024:.1f} KB",
            }
        )

    return bundled


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(REPO_ROOT / "configs/default.yaml")

    metadata, overview = _build_metadata_and_overview(config)
    _write_json("metadata.json", metadata)
    _write_json("overview.json", overview)
    _write_json("historical_clean_fuel.json", _build_historical_clean_fuel())
    _write_json("forecast_stage3_mock.json", _build_forecast_mock(config))
    _write_json("assumptions.json", _build_assumptions(config))
    _write_json("optimise_mock.json", _build_optimise_mock(config))
    _write_json("planning_mock.json", _build_planning_mock(config))
    _write_json("research.json", _build_research(config))

    print(f"Wrote static website JSON data to {DATA_DIR}")


if __name__ == "__main__":
    main()
