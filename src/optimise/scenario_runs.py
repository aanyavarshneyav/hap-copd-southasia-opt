"""Task 4 scenario experiments on the candidate parameter set (mock forecasts).

Runs NSGA-II and the equal-footing MDP/GA comparator across:
1. Total budget at 25%, 50%, and 100% of full-gap LPG + infra cost.
2. Sweep of ``eta_biomass`` over ``{0.10, 0.20, 0.30, 0.40, 0.48}``.
3. ``electric_induction`` adoption caps at Central (0.15) vs. halved (0.075).

Outputs deterministic summary CSVs and JSON under ``outputs/tables/param_sourcing/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import Stage3Contracts, load_config, load_stage3_contracts
from src.optimise.evaluator import Stage3Evaluator
from src.optimise.nsga2 import (
    run_nsga2_multi_seed,
    select_representative_plans,
)
from src.optimise.param_audit import (
    build_candidate_intervention_params,
    build_task3_audit_spec,
    run_parameter_audit,
)
from src.planning.compare import (
    _pick_best_ga_for_scalarization,
    draw_shock_branches,
    simulate_open_loop_schedule,
    summarize_rollouts,
)
from src.planning.mdp import (
    MDPConfig,
    build_layered_mdp,
    format_action_label,
    simulate_mdp_rollouts,
    solve_finite_horizon_backward_induction,
)


def _with_params(base: Stage3Contracts, params: dict[str, Any]) -> Stage3Contracts:
    return Stage3Contracts(
        forecasts_stage3=base.forecasts_stage3,
        baseline_unit_state=base.baseline_unit_state,
        surrogate=base.surrogate,
        intervention_params=params,
        is_mock=base.is_mock,
        paths=base.paths,
    )


def _allocation_shares(x_vec: np.ndarray, evaluator: Stage3Evaluator) -> dict[str, float]:
    """Compute total coverage increments per intervention and infra build count."""
    tensor = x_vec.reshape(evaluator.n_units, evaluator.n_interventions, evaluator.n_periods)
    lpg_sum = float(np.sum(tensor[:, evaluator._idx_lpg, :]))
    bio_sum = float(np.sum(tensor[:, evaluator._idx_bio, :]))
    elec_sum = float(np.sum(tensor[:, evaluator._idx_elec, :]))
    infra_cnt = int(np.sum(tensor[:, evaluator._idx_infra, :] >= evaluator.infra_build_threshold))
    return {
        "lpg_cov_sum": lpg_sum,
        "bio_cov_sum": bio_sum,
        "elec_cov_sum": elec_sum,
        "infra_builds": float(infra_cnt),
    }


def _mdp_policy_shares(
    mdp: Any,
    sol: Any,
    evaluator: Stage3Evaluator,
    branches_mat: np.ndarray,
) -> dict[str, Any]:
    """Simulate MDP rollouts and extract nominal-path & mean realized intervention totals."""
    roll_df = simulate_mdp_rollouts(
        mdp,
        sol,
        evaluator,
        n_rollouts=len(branches_mat),
        seed=42,
        precomputed_branches=branches_mat,
    )
    # Also trace exact nominal branch (mult_mode=1) to inspect nominal coverage allocation
    nom_tensor = np.zeros(
        (evaluator.n_units, evaluator.n_interventions, evaluator.n_periods),
        dtype=float,
    )
    s_idx = 0
    trans_lookup: list[list[dict[int, tuple[float, np.ndarray]]]] = []
    for t in range(mdp.n_periods):
        tr = mdp.transitions[t]
        per_state: list[dict[int, tuple[float, np.ndarray]]] = [
            {} for _ in range(len(mdp.layers[t]))
        ]
        for edge_idx in range(len(tr.src_state)):
            per_state[int(tr.src_state[edge_idx])][int(tr.action_id[edge_idx])] = (
                float(tr.cost[edge_idx]),
                tr.next_state[edge_idx],
            )
        trans_lookup.append(per_state)

    p_labels: list[str] = []
    for t in range(mdp.n_periods):
        st_curr = mdp.layers[t][s_idx]
        a_id = int(sol.policy_action_ids[t][s_idx])
        combo = mdp.templates[a_id]
        p_labels.append(format_action_label(combo, evaluator.units))
        _, next_triple = trans_lookup[t][s_idx][a_id]
        ns_idx = int(next_triple[1])  # nominal
        st_next = mdp.layers[t + 1][ns_idx]
        n_u = mdp.n_units
        for u, act in enumerate(combo):
            if act == 1:  # infra
                nom_tensor[u, evaluator._idx_infra, t] = evaluator.infra_build_threshold
            elif act == 3:  # bio
                nom_tensor[u, evaluator._idx_bio, t] = float(
                    st_next[2 * n_u + u] - st_curr[2 * n_u + u]
                ) * float(mdp.step_u[u])
            elif act in (2, 5):  # lpg
                nom_tensor[u, evaluator._idx_lpg, t] = float(st_next[u] - st_curr[u]) * float(
                    mdp.step_u[u]
                )
            elif act == 4:  # elec
                nom_tensor[u, evaluator._idx_elec, t] = float(
                    st_next[n_u + u] - st_curr[n_u + u]
                ) * float(mdp.step_u[u])
        s_idx = ns_idx

    shares = _allocation_shares(nom_tensor.reshape(-1), evaluator)
    return {
        "roll_df": roll_df,
        "nom_shares": shares,
        "nom_actions": " | ".join(p_labels),
    }


def run_single_scenario(
    label: str,
    config: dict[str, Any],
    contracts: Stage3Contracts,
    *,
    n_rollouts: int = 1000,
) -> dict[str, Any]:
    """Run NSGA-II + MDP/GA comparator for one parameter configuration."""
    t0 = time.perf_counter()
    evaluator = Stage3Evaluator(config=config, contracts=contracts)
    mdp_cfg = MDPConfig.from_config(config)
    comp_cfg = config.get("planning", {}).get("compare", {})
    mult_raw = comp_cfg.get("shock_multipliers", [0.5, 1.0, 1.25])
    shock_mults = (float(mult_raw[0]), float(mult_raw[1]), float(mult_raw[2]))

    # 1. NSGA-II multi-seed
    nsga2_res = run_nsga2_multi_seed(evaluator, config)
    selected_df, selected_idx = select_representative_plans(
        nsga2_res.combined_X, nsga2_res.combined_evaluation, evaluator
    )
    var_cols = list(evaluator.variable_names)
    from src.optimise.nsga2 import pareto_front_to_dataframe

    pareto_df = pareto_front_to_dataframe(nsga2_res, evaluator, selected_indices=selected_idx)

    # 2. Equal-footing CRN branches
    branches_mat = draw_shock_branches(
        mdp_cfg.transition_probs, n_rollouts, evaluator.n_periods, mdp_cfg.seed
    )

    # 3. MDP (b=6 and b=11)
    mdp_b6 = build_layered_mdp(evaluator, replace(mdp_cfg, budget_bins=6))
    sol_b6 = solve_finite_horizon_backward_induction(
        mdp_b6,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    mdp_b6_out = _mdp_policy_shares(mdp_b6, sol_b6, evaluator, branches_mat)
    sum_b6 = summarize_rollouts(
        "mdp_b6",
        "MDP",
        mdp_b6_out["roll_df"],
        branches_mat,
        evaluator,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )

    mdp_b11 = build_layered_mdp(evaluator, replace(mdp_cfg, budget_bins=11))
    sol_b11 = solve_finite_horizon_backward_induction(
        mdp_b11,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    mdp_b11_out = _mdp_policy_shares(mdp_b11, sol_b11, evaluator, branches_mat)
    sum_b11 = summarize_rollouts(
        "mdp_b11",
        "MDP",
        mdp_b11_out["roll_df"],
        branches_mat,
        evaluator,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )

    # 4. Matched GA best-scalarized + knee + daly_minimal
    best_idx, x_ga_best, roll_ga_best = _pick_best_ga_for_scalarization(
        pareto_df,
        evaluator,
        branches_mat,
        shock_mults,
        mdp_cfg.cost_weight_lambda,
        mdp_cfg.discount_factor,
    )
    sum_ga_best = summarize_rollouts(
        f"ga_matched_best (id={best_idx})",
        "GA",
        roll_ga_best,
        branches_mat,
        evaluator,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    ga_best_shares = _allocation_shares(x_ga_best, evaluator)

    sel_by_name = selected_df.set_index("plan_name")
    x_knee = sel_by_name.loc["knee", var_cols].to_numpy(dtype=float)
    x_daly_min = sel_by_name.loc["daly_minimal", var_cols].to_numpy(dtype=float)
    knee_shares = _allocation_shares(x_knee, evaluator)
    daly_min_shares = _allocation_shares(x_daly_min, evaluator)

    roll_knee = simulate_open_loop_schedule(
        x_knee,
        evaluator,
        branches_mat,
        shock_multipliers=shock_mults,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    sum_knee = summarize_rollouts(
        "ga_knee",
        "GA",
        roll_knee,
        branches_mat,
        evaluator,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )

    dt = time.perf_counter() - t0
    return {
        "scenario": label,
        "is_mock": bool(evaluator.is_mock),
        "warning_mock_inputs": bool(evaluator.warning_mock_inputs),
        "data_version": str(
            contracts.intervention_params.get("data_version", evaluator.data_version)
        ),
        "forecast_basis": f"mock_forecasts ({evaluator.data_version})"
        if evaluator.is_mock
        else evaluator.data_version,
        "cost_basis": "mock_intervention_costs_usd_millions",
        "effectiveness_basis": (
            "lpg=0.88/0.78/0.86/0.78; elec=0.92/0.85/0.90/0.85; "
            f"bio={contracts.intervention_params['effectiveness']['IND_rural']['improved_biomass']:.2f}"
        ),
        "budget_basis": "pct_of_mock_full_gap_lpg_infra_cost (not_real_world_estimate)",
        "budget_usd_millions": float(evaluator.total_budget_usd_millions),
        "eta_biomass": float(
            contracts.intervention_params["effectiveness"]["IND_rural"]["improved_biomass"]
        ),
        "cap_electric_induction": float(
            contracts.intervention_params["max_adoption_per_period"]["IND_rural"][
                "electric_induction"
            ]["P1"]
        ),
        "runtime_seconds": round(dt, 2),
        "nsga2_n_pareto": int(len(pareto_df)),
        "nsga2_combined_hv": float(nsga2_res.combined_hypervolume),
        "knee_dalys_2040": float(sel_by_name.loc["knee", "dalys_2040_mean"]),
        "knee_cost_usd_m": float(sel_by_name.loc["knee", "total_cost_usd_millions"]),
        "knee_eq_gap_asr": float(sel_by_name.loc["knee", "equity_gap_asr"]),
        "knee_lpg_cov": knee_shares["lpg_cov_sum"],
        "knee_bio_cov": knee_shares["bio_cov_sum"],
        "knee_elec_cov": knee_shares["elec_cov_sum"],
        "daly_min_dalys_2040": float(sel_by_name.loc["daly_minimal", "dalys_2040_mean"]),
        "daly_min_cost_usd_m": float(sel_by_name.loc["daly_minimal", "total_cost_usd_millions"]),
        "daly_min_lpg_cov": daly_min_shares["lpg_cov_sum"],
        "daly_min_bio_cov": daly_min_shares["bio_cov_sum"],
        "daly_min_elec_cov": daly_min_shares["elec_cov_sum"],
        "ga_best_mean_dalys_2040": sum_ga_best["mean_dalys_2040"],
        "ga_best_mean_cost": sum_ga_best["mean_cost_usd_millions"],
        "ga_best_scalarized": sum_ga_best["mean_scalarized_value"],
        "ga_best_viol_rate": sum_ga_best["constraint_violation_rate"],
        "ga_best_lpg_cov": ga_best_shares["lpg_cov_sum"],
        "ga_best_bio_cov": ga_best_shares["bio_cov_sum"],
        "ga_best_elec_cov": ga_best_shares["elec_cov_sum"],
        "ga_knee_scalarized": sum_knee["mean_scalarized_value"],
        "mdp_b6_mean_dalys_2040": sum_b6["mean_dalys_2040"],
        "mdp_b6_mean_cost": sum_b6["mean_cost_usd_millions"],
        "mdp_b6_scalarized": sum_b6["mean_scalarized_value"],
        "mdp_b6_nom_lpg_cov": mdp_b6_out["nom_shares"]["lpg_cov_sum"],
        "mdp_b6_nom_bio_cov": mdp_b6_out["nom_shares"]["bio_cov_sum"],
        "mdp_b6_nom_elec_cov": mdp_b6_out["nom_shares"]["elec_cov_sum"],
        "mdp_b6_nom_actions": mdp_b6_out["nom_actions"],
        "mdp_b11_mean_dalys_2040": sum_b11["mean_dalys_2040"],
        "mdp_b11_mean_cost": sum_b11["mean_cost_usd_millions"],
        "mdp_b11_scalarized": sum_b11["mean_scalarized_value"],
        "mdp_b11_nom_lpg_cov": mdp_b11_out["nom_shares"]["lpg_cov_sum"],
        "mdp_b11_nom_bio_cov": mdp_b11_out["nom_shares"]["bio_cov_sum"],
        "mdp_b11_nom_elec_cov": mdp_b11_out["nom_shares"]["elec_cov_sum"],
        "mdp_b11_nom_actions": mdp_b11_out["nom_actions"],
    }


def run_all_task4_experiments(
    config_path: str | Path = "configs/default.yaml",
    out_dir: str | Path = "outputs/tables/param_sourcing",
) -> dict[str, pd.DataFrame]:
    """Run all Task 4 candidate experiments on mock forecasts and save summary CSVs."""
    config = load_config(config_path)
    base_contracts = load_stage3_contracts(config)
    cand_params = build_candidate_intervention_params(base_contracts.intervention_params)
    cand_contracts = _with_params(base_contracts, cand_params)

    audit_spec = build_task3_audit_spec(config, base_contracts, use_candidate_overrides=True)
    audit_report = run_parameter_audit(audit_spec)
    full_gap_cost = audit_report.full_gap_lpg.total_full_gap_lpg_cost_usd_millions

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "candidate_audit.json").write_text(
        json.dumps(audit_report.to_dict(), indent=2), encoding="utf-8"
    )

    # 1. Budget sweep: 25%, 50%, 100% of full-gap LPG cost
    budget_rows: list[dict[str, Any]] = []
    for frac in (0.25, 0.50, 1.00):
        p = copy.deepcopy(cand_params)
        p["total_budget_usd_millions"] = round(frac * full_gap_cost, 4)
        res = run_single_scenario(
            f"budget_{int(frac * 100)}pct", config, _with_params(cand_contracts, p)
        )
        budget_rows.append(res)
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(out_path / "task4_budget_sweep_mock.csv", index=False)

    # 2. eta_biomass sweep at 50% budget (binding) and 100% budget (full gap)
    bio_rows: list[dict[str, Any]] = []
    for budget_frac in (0.50, 1.00):
        for eta_bio in (0.10, 0.20, 0.30, 0.40, 0.48):
            p = copy.deepcopy(cand_params)
            p["total_budget_usd_millions"] = round(budget_frac * full_gap_cost, 4)
            for u in config["geography"]["allocation_units"]:
                p["effectiveness"][u]["improved_biomass"] = float(eta_bio)
            res = run_single_scenario(
                f"bio_{eta_bio:.2f}_b{int(budget_frac * 100)}pct",
                config,
                _with_params(cand_contracts, p),
            )
            bio_rows.append(res)
    bio_df = pd.DataFrame(bio_rows)
    bio_df.to_csv(out_path / "task4_eta_biomass_sweep_mock.csv", index=False)

    # 3. Induction adoption caps: Central (0.15) vs halved (0.075) at 50% and 100% budget
    cap_rows: list[dict[str, Any]] = []
    for budget_frac in (0.50, 1.00):
        for cap_label, cap_mult in (("central", 1.0), ("halved", 0.5)):
            p = copy.deepcopy(cand_params)
            p["total_budget_usd_millions"] = round(budget_frac * full_gap_cost, 4)
            for u in config["geography"]["allocation_units"]:
                for p_id in ("P1", "P2", "P3"):
                    p["max_adoption_per_period"][u]["electric_induction"][p_id] = float(
                        cand_params["max_adoption_per_period"][u]["electric_induction"][p_id]
                        * cap_mult
                    )
            res = run_single_scenario(
                f"elec_cap_{cap_label}_b{int(budget_frac * 100)}pct",
                config,
                _with_params(cand_contracts, p),
            )
            cap_rows.append(res)
    cap_df = pd.DataFrame(cap_rows)
    cap_df.to_csv(out_path / "task4_induction_cap_sweep_mock.csv", index=False)

    write_task4_safeguard_metadata(out_path, full_gap_cost=full_gap_cost)
    return {"budget": budget_df, "eta_biomass": bio_df, "induction_caps": cap_df}


def build_task4_safeguard_metadata(full_gap_cost: float = 402.76) -> dict[str, Any]:
    """Build explicit mock-safeguard provenance metadata for Task 4 outputs."""
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_mock": True,
        "warning_mock_inputs": True,
        "data_version": "MOCK_V1_CANDIDATE_T3",
        "forecast_data_basis": {
            "is_mock": True,
            "data_version": "MOCK_V1",
            "files": [
                "data/processed/forecasts_stage3_mock.csv",
                "data/processed/baseline_unit_state_mock.csv",
                "data/processed/exposure_response_surrogate_mock.json",
            ],
        },
        "intervention_cost_basis": {
            "is_mock": True,
            "source_file": "data/processed/intervention_params_mock.json",
            "note": (
                "unit_costs_usd_millions use synthetic mock $M values (LPG 320/620/18/48M, "
                "bio 120/210/7/18M, induction 410/780/22/60M, infra 40/95/5/16M) because "
                "unit household counts and non-LPG USD/HH were not supplied."
            ),
        },
        "effectiveness_assumptions": {
            "lpg_subsidy": {
                "IND_urban": 0.88,
                "IND_rural": 0.78,
                "NPL_urban": 0.86,
                "NPL_rural": 0.78,
            },
            "electric_induction": {
                "IND_urban": 0.92,
                "IND_rural": 0.85,
                "NPL_urban": 0.90,
                "NPL_rural": 0.85,
            },
            "improved_biomass": (
                "0.45 mock baseline in budget & induction-cap runs; swept over "
                "{0.10, 0.20, 0.30, 0.40, 0.48} in eta_biomass sweep."
            ),
        },
        "budget_basis": {
            "mock_full_gap_lpg_infra_cost_usd_millions": round(float(full_gap_cost), 2),
            "budget_scenarios_usd_millions": {
                "25_pct": round(0.25 * float(full_gap_cost), 2),
                "50_pct": round(0.50 * float(full_gap_cost), 2),
                "100_pct": round(float(full_gap_cost), 2),
            },
            "disclaimer": (
                "SYNTHETIC MOCK SCENARIO ONLY: The $402.76M full-gap LPG+infra cost and its "
                "25%/50%/100% budget scenarios ($100.69M, $201.38M, $402.76M) are computed from "
                "mock intervention costs and mock 2040 BAU forecasts and MUST NOT be treated as "
                "real-world cost estimates."
            ),
        },
    }


def write_task4_safeguard_metadata(
    out_dir: str | Path,
    *,
    full_gap_cost: float = 402.76,
    stamp_existing_csvs: bool = False,
) -> dict[str, Path]:
    """Write safeguard metadata sidecars and optionally stamp existing Task 4 CSVs."""
    out_path = Path(out_dir)
    meta = build_task4_safeguard_metadata(full_gap_cost=full_gap_cost)
    written: dict[str, Path] = {}

    summary_meta = out_path / "task4_summary.meta.json"
    summary_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    written["summary_meta"] = summary_meta

    csv_names = (
        "task4_budget_sweep_mock.csv",
        "task4_eta_biomass_sweep_mock.csv",
        "task4_induction_cap_sweep_mock.csv",
    )
    for name in csv_names:
        csv_path = out_path / name
        if csv_path.exists():
            if stamp_existing_csvs:
                df = pd.read_csv(csv_path)
                df["is_mock"] = True
                df["warning_mock_inputs"] = True
                df["data_version"] = "MOCK_V1_CANDIDATE_T3"
                df["forecast_basis"] = "mock_forecasts (MOCK_V1)"
                df["cost_basis"] = "mock_intervention_costs_usd_millions"
                df["effectiveness_basis"] = df["eta_biomass"].map(
                    lambda e: (
                        f"lpg=0.88/0.78/0.86/0.78; elec=0.92/0.85/0.90/0.85; bio={float(e):.2f}"
                    )
                )
                df["budget_basis"] = (
                    "pct_of_mock_full_gap_lpg_infra_cost_402.76M (not_real_world_estimate)"
                )
                df.to_csv(csv_path, index=False)
            sidecar = Path(f"{csv_path}.meta.json")
            sidecar.write_text(
                json.dumps({"artifact": name, **meta}, indent=2) + "\n", encoding="utf-8"
            )
            written[f"{name}.meta.json"] = sidecar

    audit_path = out_path / "candidate_audit.json"
    if audit_path.exists() and stamp_existing_csvs:
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if "safeguards" not in audit_payload:
            audit_payload = {"safeguards": meta, **audit_payload}
            audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Task 4 candidate parameter experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tables/param_sourcing"))
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    dfs = run_all_task4_experiments(args.config, args.out_dir)
    print(f"Task 4 experiments complete in {time.perf_counter() - t0:.2f}s")
    for name, df in dfs.items():
        print(f"\n=== {name.upper()} ===")
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
