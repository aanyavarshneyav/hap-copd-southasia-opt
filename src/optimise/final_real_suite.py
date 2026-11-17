"""Final real-data Stage 2 uncertainty re-evaluation, 8-factor sensitivity suite,
and curated ``outputs/final_real_data/`` bundle export (Phase 3 & Phase 4).

Reads validated real-data outputs from ``configs/default.yaml`` and
``data/processed/``, runs:
1. 500-draw Stage 2 Monte Carlo Pareto re-evaluation (``s3_pareto_reeval.csv``).
2. Targeted one-factor-at-a-time 8-factor sensitivity suite (``s3_sensitivity_real.csv``
   and ``s3_recurring_opex_sensitivity.csv``).
3. Additional Phase 4 presentation figures (``s1_country_period_heatmap.png``,
   ``s3_pareto_uncertainty_reeval.png``, ``plan_sensitivity_tornado.png``).
4. Curated export to ``outputs/final_real_data/{tables,figures}/`` with rich
   provenance metadata sidecars (``is_mock: false``).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.contracts import Stage3Contracts, load_config, load_stage3_contracts
from src.optimise.evaluator import Stage3Evaluator
from src.optimise.nsga2 import (
    run_nsga2_multi_seed,
    select_representative_plans,
)
from src.planning.mdp import (
    MDPConfig,
    build_layered_mdp,
    simulate_mdp_rollouts,
    solve_finite_horizon_backward_induction,
)

PARAM_VERSION = "REAL_V1_2024USD"


def _write_sidecar(
    target: Path,
    *,
    source: str,
    config: dict[str, Any],
    data_version: str,
    scenario_assumptions: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    meta_path = Path(f"{target}.meta.json")
    payload: dict[str, Any] = {
        "artifact": target.name,
        "source": source,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_mock": False,
        "warning_mock_inputs": False,
        "data_version": data_version,
        "parameter_version": PARAM_VERSION,
        "seeds": config.get("seeds", {}),
        "stage3_seeds": config.get("stage3", {}).get("seeds", []),
        "total_budget_usd_millions": config.get("stage3", {}).get("total_budget_usd_millions"),
        "mdp_cost_weight_lambda": config.get("planning", {})
        .get("mdp", {})
        .get("cost_weight_lambda"),
        "scenario_assumptions": scenario_assumptions,
    }
    if extra:
        payload.update(extra)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def run_pareto_uncertainty_reevaluation(
    config: dict[str, Any],
    contracts: Stage3Contracts,
    evaluator: Stage3Evaluator,
    *,
    pareto_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    draws_path: Path,
    out_csv: Path,
) -> pd.DataFrame:
    """Re-evaluate BAU + selected Pareto plans over 500 real Stage 2 MC draws."""
    draws_df = pd.read_csv(draws_path)
    var_cols = list(evaluator.variable_names)

    # Extract (4, 2, 500) array for (unit, sex=['female','male'], draw_id)
    n_u = evaluator.n_units
    draw_ids = sorted(draws_df["draw_id"].unique().tolist())
    n_draws = len(draw_ids)
    draw_counts = np.zeros((n_u, 2, n_draws), dtype=float)
    draw_asrs = np.zeros((n_u, 2, n_draws), dtype=float)

    for u_i, unit in enumerate(evaluator.units):
        for s_i, sex in enumerate(("female", "male")):
            sub = draws_df[
                (draws_df["location"] == unit)
                & (draws_df["sex"] == sex)
                & (draws_df["year"] == 2040)
            ].sort_values("draw_id")
            if len(sub) != n_draws:
                raise ValueError(
                    f"Expected {n_draws} Stage 2 draws for ({unit}, {sex}), got {len(sub)}."
                )
            draw_counts[u_i, s_i, :] = sub["dalys_count_draw"].to_numpy(dtype=float)
            draw_asrs[u_i, s_i, :] = sub["rate_per_100k_draw"].to_numpy(dtype=float)

    # Plans to re-evaluate: BAU + 4 selected plans + 4 Pareto quantiles
    plans: list[tuple[str, np.ndarray]] = [("bau_zero_action", np.zeros(evaluator.n_vars))]
    for _, row in selected_df.iterrows():
        plans.append((f"ga_{row['plan_name']}", row[var_cols].to_numpy(dtype=float)))

    pareto_sorted = pareto_df.sort_values("total_cost_usd_millions").reset_index(drop=True)
    sel_ids = set(int(v) for v in selected_df["solution_id"].tolist())
    for q_idx, pos in enumerate(np.linspace(0.20, 0.80, 4)):
        r_idx = int(round(pos * (len(pareto_sorted) - 1)))
        sol_id = int(pareto_sorted.loc[r_idx, "solution_id"])
        if sol_id not in sel_ids:
            plans.append(
                (
                    f"ga_pareto_q{q_idx + 1} (id={sol_id})",
                    pareto_sorted.loc[r_idx, var_cols].to_numpy(dtype=float),
                )
            )

    # Fixed surrogate noise draws (500 x 4 units) for joint Stage 2 + surrogate UI
    rng = np.random.default_rng(int(config.get("seeds", {}).get("global", 42)))
    surr_eps = rng.normal(0.0, 1.0, size=(n_u, 2, n_draws))

    max_headroom = np.maximum(0.0, 1.0 - evaluator._baseline_clean_fuel_population)
    rows: list[dict[str, Any]] = []

    for plan_label, x_vec in plans:
        ev_res = evaluator.evaluate(x_vec)
        tensor = x_vec.reshape(n_u, evaluator.n_interventions, evaluator.n_periods)
        x_cook = tensor[:, evaluator._non_infra_indices, :]
        eff_gain = np.sum(
            x_cook * evaluator._non_infra_effectiveness[:, :, np.newaxis], axis=(1, 2)
        )
        delta_c = np.minimum(eff_gain * evaluator._coverage_bridge_factor, max_headroom)

        # Deterministic surrogate multiplier per (unit, sex)
        mult_f = np.exp(evaluator._slope_female * delta_c)
        mult_m = np.exp(evaluator._slope_male * delta_c)
        mult_det = np.stack([mult_f, mult_m], axis=1)  # (U, 2)

        # Stage 2 forecast-draw uncertainty only
        dalys_s2_draws = np.sum(draw_counts * mult_det[:, :, np.newaxis], axis=(0, 1))
        asr_gap_s2_draws = np.sum(
            np.abs(
                draw_asrs[:, 0, :] * mult_f[:, np.newaxis]
                - draw_asrs[:, 1, :] * mult_m[:, np.newaxis]
            )
            * evaluator._pop_weights[:, np.newaxis],
            axis=0,
        )

        # Joint Stage 2 draws + surrogate residual SD (sigma * delta_c)
        sd_term = (
            evaluator._surrogate_sd[:, np.newaxis, np.newaxis] * delta_c[:, np.newaxis, np.newaxis]
        )
        mult_joint = mult_det[:, :, np.newaxis] * np.exp(surr_eps * sd_term)
        dalys_joint_draws = np.sum(draw_counts * mult_joint, axis=(0, 1))
        asr_gap_joint_draws = np.sum(
            np.abs(
                draw_asrs[:, 0, :] * mult_joint[:, 0, :] - draw_asrs[:, 1, :] * mult_joint[:, 1, :]
            )
            * evaluator._pop_weights[:, np.newaxis],
            axis=0,
        )

        rows.append(
            {
                "plan_name": plan_label,
                "total_cost_usd_millions": float(ev_res.total_cost_usd_millions),
                "is_feasible": bool(ev_res.is_feasible),
                "analytical_dalys_mean": float(ev_res.dalys_2040_mean),
                "analytical_dalys_lower": float(ev_res.dalys_2040_lower),
                "analytical_dalys_upper": float(ev_res.dalys_2040_upper),
                "stage2_draws_n": n_draws,
                "stage2_draws_dalys_mean": float(np.mean(dalys_s2_draws)),
                "stage2_draws_dalys_p025": float(np.percentile(dalys_s2_draws, 2.5)),
                "stage2_draws_dalys_p975": float(np.percentile(dalys_s2_draws, 97.5)),
                "joint_draws_dalys_mean": float(np.mean(dalys_joint_draws)),
                "joint_draws_dalys_p025": float(np.percentile(dalys_joint_draws, 2.5)),
                "joint_draws_dalys_p975": float(np.percentile(dalys_joint_draws, 97.5)),
                "analytical_equity_gap_asr": float(ev_res.equity_gap_asr),
                "stage2_draws_equity_gap_mean": float(np.mean(asr_gap_s2_draws)),
                "joint_draws_equity_gap_mean": float(np.mean(asr_gap_joint_draws)),
                "joint_draws_equity_gap_p025": float(np.percentile(asr_gap_joint_draws, 2.5)),
                "joint_draws_equity_gap_p975": float(np.percentile(asr_gap_joint_draws, 97.5)),
                "is_mock": False,
                "data_version": evaluator.data_version,
                "parameter_version": PARAM_VERSION,
            }
        )

    df_out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    _write_sidecar(
        out_csv,
        source="src.optimise.final_real_suite.run_pareto_uncertainty_reevaluation",
        config=config,
        data_version=evaluator.data_version,
        scenario_assumptions=(
            "500 real Stage 2 2040 MC draws (stage2_uncertainty_draws.csv) "
            "+ log-linear surrogate residual SD=0.08."
        ),
        extra={"n_draws": n_draws, "n_plans": len(df_out)},
    )
    return df_out


def _evaluate_one_sensitivity_case(
    *,
    factor_group: str,
    scenario_id: str,
    scenario_label: str,
    config: dict[str, Any],
    contracts: Stage3Contracts,
    baseline_hh_override: dict[str, float] | None = None,
    bridge_override: dict[str, float] | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Run 5-seed NSGA-II + finite-horizon MDP for a single sensitivity variant."""
    t0 = time.perf_counter()
    ev = Stage3Evaluator(
        config=config,
        contracts=contracts,
        baseline_clean_fuel_households=baseline_hh_override,
        coverage_bridge_factor=bridge_override,
    )
    bau = ev.evaluate(np.zeros(ev.n_vars))
    multi = run_nsga2_multi_seed(ev, config)
    sel_df, _ = select_representative_plans(multi.combined_X, multi.combined_evaluation, ev)
    sel_by = sel_df.set_index("plan_name")

    var_cols = list(ev.variable_names)
    daly_vec = (
        sel_by.loc["daly_minimal", var_cols]
        .to_numpy(dtype=float)
        .reshape(ev.n_units, ev.n_interventions, ev.n_periods)
    )
    knee_vec = (
        sel_by.loc["knee", var_cols]
        .to_numpy(dtype=float)
        .reshape(ev.n_units, ev.n_interventions, ev.n_periods)
    )

    mdp_cfg = MDPConfig.from_config(config)
    mdp = build_layered_mdp(ev, mdp_cfg)
    sol = solve_finite_horizon_backward_induction(
        mdp,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    mdp_roll = simulate_mdp_rollouts(mdp, sol, ev, n_rollouts=500, seed=mdp_cfg.seed)

    def _spend_summary(ten: np.ndarray) -> dict[str, float]:
        return {
            "lpg_cov": float(np.sum(ten[:, ev._idx_lpg, :])),
            "bio_cov": float(np.sum(ten[:, ev._idx_bio, :])),
            "elec_cov": float(np.sum(ten[:, ev._idx_elec, :])),
            "infra_builds": float(np.sum(ten[:, ev._idx_infra, :] >= ev.infra_build_threshold)),
            "npl_rural_lpg_cov": float(np.sum(ten[ev.units.index("NPL_rural"), ev._idx_lpg, :])),
            "npl_rural_bio_cov": float(np.sum(ten[ev.units.index("NPL_rural"), ev._idx_bio, :])),
        }

    k_sp = _spend_summary(knee_vec)
    d_sp = _spend_summary(daly_vec)

    return {
        "factor_group": factor_group,
        "scenario_id": scenario_id,
        "scenario_label": scenario_label,
        "budget_usd_millions": float(ev.total_budget_usd_millions),
        "discount_factor_mdp": float(mdp_cfg.discount_factor),
        "apply_cost_discounting_ga": bool(ev.apply_cost_discounting),
        "bridge_factor": float(ev._coverage_bridge_factor[0]),
        "eta_biomass": float(
            contracts.intervention_params["effectiveness"]["IND_rural"]["improved_biomass"]
        ),
        "infra_enabled_npl_rural": bool(ev._infra_enabled[ev.units.index("NPL_rural")]),
        "bau_dalys_2040": float(bau.dalys_2040_mean),
        "ga_n_pareto": int(len(multi.combined_X)),
        "ga_combined_hv": float(multi.combined_hypervolume),
        "ga_knee_dalys_2040": float(sel_by.loc["knee", "dalys_2040_mean"]),
        "ga_knee_dalys_averted": float(bau.dalys_2040_mean)
        - float(sel_by.loc["knee", "dalys_2040_mean"]),
        "ga_knee_cost_usd_m": float(sel_by.loc["knee", "total_cost_usd_millions"]),
        "ga_knee_equity_gap": float(sel_by.loc["knee", "equity_gap_asr"]),
        "ga_knee_lpg_cov": k_sp["lpg_cov"],
        "ga_knee_bio_cov": k_sp["bio_cov"],
        "ga_knee_elec_cov": k_sp["elec_cov"],
        "ga_daly_min_dalys_2040": float(sel_by.loc["daly_minimal", "dalys_2040_mean"]),
        "ga_daly_min_dalys_averted": float(bau.dalys_2040_mean)
        - float(sel_by.loc["daly_minimal", "dalys_2040_mean"]),
        "ga_daly_min_cost_usd_m": float(sel_by.loc["daly_minimal", "total_cost_usd_millions"]),
        "ga_daly_min_equity_gap": float(sel_by.loc["daly_minimal", "equity_gap_asr"]),
        "ga_daly_min_infra_builds": d_sp["infra_builds"],
        "ga_daly_min_npl_rural_lpg": d_sp["npl_rural_lpg_cov"],
        "ga_daly_min_npl_rural_bio": d_sp["npl_rural_bio_cov"],
        "mdp_mean_dalys_2040": float(mdp_roll["dalys_2040_mean"].mean()),
        "mdp_mean_dalys_averted": float(mdp_roll["dalys_averted_mean"].mean()),
        "mdp_mean_cost_usd_m": float(mdp_roll["total_cost_usd_millions"].mean()),
        "mdp_mean_equity_gap": float(mdp_roll["equity_gap_asr"].mean()),
        "mdp_all_feasible": bool(mdp_roll["is_feasible_in_evaluator"].all()),
        "mdp_p1_action_s0": str(mdp_roll["p1_action"].iloc[0]),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
        "is_mock": False,
        "data_version": ev.data_version,
        "parameter_version": PARAM_VERSION,
        "note": note,
    }


def run_targeted_sensitivity_suite(
    base_config: dict[str, Any],
    base_contracts: Stage3Contracts,
    out_csv: Path,
    recurring_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all 8 targeted one-factor-at-a-time sensitivity scenarios on real data."""
    rows: list[dict[str, Any]] = []

    def _with_params(p_dict: dict[str, Any]) -> Stage3Contracts:
        return Stage3Contracts(
            forecasts_stage3=base_contracts.forecasts_stage3,
            baseline_unit_state=base_contracts.baseline_unit_state,
            surrogate=base_contracts.surrogate,
            intervention_params=p_dict,
            is_mock=False,
            paths=base_contracts.paths,
        )

    # 0. Reference Base Case (50% budget $5985.73M, bridge=1.0, bio=0.25, gamma=0.774)
    rows.append(
        _evaluate_one_sensitivity_case(
            factor_group="0_base_reference",
            scenario_id="base_50pct_scenario_a",
            scenario_label="Base (50% $5,985.7M, bridge=1.0, eta_bio=0.25, Scenario A)",
            config=copy.deepcopy(base_config),
            contracts=base_contracts,
            note="Central real-data configuration (Scenario A existing infra-ready only).",
        )
    )

    # 1. Budget scenarios: 25% Tight, 100% Full-gap, Derived-2024 50%, WHO-2024 50%
    for sid, label, b_val, b_note in [
        (
            "budget_25pct_tight",
            "Budget 25% Tight ($2,992.87M)",
            2992.87,
            "25% of H_observed_hh_lpg_fiscal ($11,971.47M)",
        ),
        (
            "budget_100pct_full",
            "Budget 100% Full-Gap ($11,971.47M)",
            11971.47,
            "100% of H_observed_hh_lpg_fiscal",
        ),
        (
            "budget_derived2024_50pct",
            "Budget Derived-2024 HH 50% ($2,703.61M)",
            2703.61,
            "50% of H_2024_derived_hh_lpg_fiscal ($5,407.22M)",
        ),
        (
            "budget_who2024_50pct",
            "Budget WHO-2024 Pop 50% ($2,991.24M)",
            2991.24,
            "50% of P_who2024_pop_lpg_fiscal ($5,982.48M)",
        ),
    ]:
        p = copy.deepcopy(base_contracts.intervention_params)
        p["total_budget_usd_millions"] = b_val
        c = copy.deepcopy(base_config)
        c["stage3"]["total_budget_usd_millions"] = b_val
        rows.append(
            _evaluate_one_sensitivity_case(
                factor_group="1_budget_scenario",
                scenario_id=sid,
                scenario_label=label,
                config=c,
                contracts=_with_params(p),
                note=b_note,
            )
        )

    # 2. Coverage bridge factor: 0.8 and 1.2 (1.0 is Base)
    for bf in (0.8, 1.2):
        rows.append(
            _evaluate_one_sensitivity_case(
                factor_group="2_coverage_bridge",
                scenario_id=f"bridge_{bf:.1f}".replace(".", "_"),
                scenario_label=f"Coverage Bridge Factor = {bf:.1f}",
                config=copy.deepcopy(base_config),
                contracts=base_contracts,
                bridge_override={u: bf for u in base_config["geography"]["allocation_units"]},
                note=f"delta_pop = {bf:.1f} * delta_hh capped at 1 - pop_bau_2040.",
            )
        )

    # 3. Improved biomass effectiveness: Low=0.15, High IER=0.30, Raw PM2.5=0.32 & 0.50
    for eta_b, tag in [
        (0.15, "Low IER PAF (0.15)"),
        (0.30, "High IER PAF (0.30)"),
        (0.32, "Tielsch Raw PM2.5 (0.32)"),
        (0.50, "Pope Upper Raw PM2.5 (0.50)"),
    ]:
        p = copy.deepcopy(base_contracts.intervention_params)
        for u in p["effectiveness"]:
            p["effectiveness"][u]["improved_biomass"] = eta_b
        rows.append(
            _evaluate_one_sensitivity_case(
                factor_group="3_biomass_effectiveness",
                scenario_id=f"eta_bio_{int(round(eta_b * 100)):02d}",
                scenario_label=f"Biomass eta = {eta_b:.2f} ({tag})",
                config=copy.deepcopy(base_config),
                contracts=_with_params(p),
                note=f"ASM-10 sensitivity: {tag}.",
            )
        )

    # 4. Electric induction adoption caps: Halved (P1=0.05, P2=0.075, P3=0.075)
    p_halved = copy.deepcopy(base_contracts.intervention_params)
    for u in p_halved["max_adoption_per_period"]:
        for pid in ("P1", "P2", "P3"):
            p_halved["max_adoption_per_period"][u]["electric_induction"][pid] *= 0.5
    rows.append(
        _evaluate_one_sensitivity_case(
            factor_group="4_induction_caps",
            scenario_id="induction_caps_halved",
            scenario_label="Electric Induction Caps Halved (0.05/0.075/0.075)",
            config=copy.deepcopy(base_config),
            contracts=_with_params(p_halved),
            note="Halved per-period adoption caps on electric_induction across all 4 units.",
        )
    )

    # 5. Discount factor: gamma=1.0 (undiscounted MDP) + GA apply_cost_discounting=True
    c_g1 = copy.deepcopy(base_config)
    c_g1["planning"]["mdp"]["discount_factor"] = 1.0
    rows.append(
        _evaluate_one_sensitivity_case(
            factor_group="5_discount_factor",
            scenario_id="mdp_gamma_1_0",
            scenario_label="MDP Undiscounted (gamma = 1.0)",
            config=c_g1,
            contracts=base_contracts,
            note="Tests MDP gamma=1.0 vs primary WHO-CHOICE gamma=0.774.",
        )
    )
    p_ga_disc = copy.deepcopy(base_contracts.intervention_params)
    p_ga_disc["apply_cost_discounting"] = True
    rows.append(
        _evaluate_one_sensitivity_case(
            factor_group="5_discount_factor",
            scenario_id="ga_cost_discounted_0_774",
            scenario_label="Stage 3 GA Cost Discounted (0.774^t)",
            config=copy.deepcopy(base_config),
            contracts=_with_params(p_ga_disc),
            note="Optional apply_cost_discounting=true sensitivity in Stage 3 evaluator.",
        )
    )

    # 6. Household baseline scenario: Derived 2024 vs Observed NFHS-5/NPHC
    hh_derived_2024 = {
        "IND_urban": 0.967,
        "IND_rural": 0.713,
        "NPL_urban": 0.682,
        "NPL_rural": 0.269,
    }
    rows.append(
        _evaluate_one_sensitivity_case(
            factor_group="6_household_baseline",
            scenario_id="hh_baseline_derived_2024",
            scenario_label="Derived 2024 HH Baseline (0.967/0.713/0.682/0.269)",
            config=copy.deepcopy(base_config),
            contracts=base_contracts,
            baseline_hh_override=hh_derived_2024,
            note=(
                "Derived WHO 2024 + DHS spread HH baseline; tighter HH headroom on "
                "IND_rural/NPL_urban."
            ),
        )
    )

    # 7. Infrastructure availability & cost scenarios (Scenario B1 Low / B2 High Proxy)
    # Using unserved households: IND_rural (2024 WHO unserved 31% of 209.21M = 64.85M HH),
    # NPL_rural (NPHC unserved 79.64% of 2.167M = 1.725M HH)
    ind_r_unserved_hh = 209_205_395 * (1.0 - 0.69)
    npl_r_unserved_hh = 2_166_519 * (1.0 - 0.2036)
    b1_ind_r_cost = round(68.17 * ind_r_unserved_hh / 1e6, 2)  # $4,421.08M (Saubhagya $68.17/HH)
    b1_npl_r_cost = round(37.45 * npl_r_unserved_hh / 1e6, 2)  # $64.62M (NEA 15A $37.45/HH)
    b2_ind_r_cost = round(570.79 * ind_r_unserved_hh / 1e6, 2)  # $37,017.80M (RDSS $570.79/HH)
    b2_npl_r_cost = round(68.17 * npl_r_unserved_hh / 1e6, 2)  # $117.62M (Saubhagya proxy)

    for sid, label, c_ind_r, c_npl_r, inf_note in [
        (
            "infra_scenario_b1_low_proxy",
            (
                f"Scenario B1 Infra Enabled (Low Proxy: IND_r=${b1_ind_r_cost:,.0f}M, "
                f"NPL_r=${b1_npl_r_cost:.1f}M)"
            ),
            b1_ind_r_cost,
            b1_npl_r_cost,
            (
                "PROXY ASSUMPTION (NOT verified regional capex): Saubhagya $68.17/HH on 2024 "
                "unserved IND_r; NEA 15A $37.45/HH on NPHC unserved NPL_r."
            ),
        ),
        (
            "infra_scenario_b2_high_proxy",
            (
                f"Scenario B2 Infra Enabled (High RDSS Proxy: IND_r=${b2_ind_r_cost:,.0f}M, "
                f"NPL_r=${b2_npl_r_cost:.1f}M)"
            ),
            b2_ind_r_cost,
            b2_npl_r_cost,
            (
                "PROXY ASSUMPTION (NOT verified regional capex): RDSS $570.79/HH on unserved "
                "IND_r; Saubhagya $68.17/HH on unserved NPL_r."
            ),
        ),
    ]:
        p_inf = copy.deepcopy(base_contracts.intervention_params)
        p_inf["unit_costs_usd_millions"]["IND_urban"]["infra_lpg_grid"] = 0.0
        p_inf["unit_costs_usd_millions"]["NPL_urban"]["infra_lpg_grid"] = 0.0
        p_inf["unit_costs_usd_millions"]["IND_rural"]["infra_lpg_grid"] = c_ind_r
        p_inf["unit_costs_usd_millions"]["NPL_rural"]["infra_lpg_grid"] = c_npl_r
        for u in ("IND_rural", "NPL_rural"):
            p_inf["max_adoption_per_period"][u]["infra_lpg_grid"] = {
                "P1": 1.0,
                "P2": 1.0,
                "P3": 1.0,
            }
        rows.append(
            _evaluate_one_sensitivity_case(
                factor_group="7_infra_scenarios",
                scenario_id=sid,
                scenario_label=label,
                config=copy.deepcopy(base_config),
                contracts=_with_params(p_inf),
                note=inf_note,
            )
        )

    # 8a. 100% Societal/Appliance cost boundary (vs 50% public co-finance)
    p_soc = copy.deepcopy(base_contracts.intervention_params)
    for u, c_bio, c_elec in [
        ("IND_urban", 5597.58, 6606.39),
        ("IND_rural", 9414.24, 11110.90),
        ("NPL_urban", 207.30, 293.31),
        ("NPL_rural", 97.49, 137.94),
    ]:
        p_soc["unit_costs_usd_millions"][u]["improved_biomass"] = c_bio
        p_soc["unit_costs_usd_millions"][u]["electric_induction"] = c_elec
    rows.append(
        _evaluate_one_sensitivity_case(
            factor_group="8_cost_boundary_and_recurring",
            scenario_id="societal_100pct_bio_elec",
            scenario_label="100% Appliance+Wiring Cost Boundary (No 50% Bio/Elec Co-Finance)",
            config=copy.deepcopy(base_config),
            contracts=_with_params(p_soc),
            note=(
                "Sensitivity replacing 50% public co-finance assumptions with 100% "
                "hardware+O&M/wiring costs from proposed_intervention_params.json."
            ),
        )
    )

    df_sens = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sens.to_csv(out_csv, index=False)
    _write_sidecar(
        out_csv,
        source="src.optimise.final_real_suite.run_targeted_sensitivity_suite",
        config=base_config,
        data_version=base_contracts.surrogate["data_version"],
        scenario_assumptions=(
            "One-factor-at-a-time 8-factor sensitivity suite around Base Scenario A "
            "($5,985.73M, bridge=1.0, bio=0.25, gamma=0.774)."
        ),
        extra={"n_scenarios": len(df_sens)},
    )

    # 8b. Recurring subsidy expenditure sensitivity (Alternative B vs Alternative A)
    # Using exact Base selected plans & recomputing P1/P2/P3 recurring LPG subsidy obligations
    base_ev = Stage3Evaluator(config=base_config, contracts=base_contracts)
    sel_df = pd.read_csv(Path(base_config["paths"]["outputs_tables_dir"]) / "s3_selected_plans.csv")
    var_cols = list(base_ev.variable_names)

    # Upfront vs 5-yr recurring opex shares per unit for LPG subsidy (B-08 Section 3.2 & 4):
    # IND: upfront=$24.49/HH, 5-yr opex=$71.69/HH -> P1=2.49x, P2=1.75x, P3=1.00x
    # NPL_u: upfront=$18.70/HH, 5-yr opex=$101.12/HH -> P1=2.69x, P2=1.84x, P3=1.00x
    # NPL_r: upfront=$33.66/HH, 5-yr opex=$101.12/HH -> P1=2.50x, P2=1.75x, P3=1.00x
    lpg_rec_mult = {
        "IND_urban": np.array([2.4908, 1.7454, 1.0]),
        "IND_rural": np.array([2.4908, 1.7454, 1.0]),
        "NPL_urban": np.array([2.6880, 1.8440, 1.0]),
        "NPL_rural": np.array([2.5008, 1.7504, 1.0]),
    }

    rec_rows: list[dict[str, Any]] = []
    for _, srow in sel_df.iterrows():
        pname = str(srow["plan_name"])
        ten = (
            srow[var_cols]
            .to_numpy(dtype=float)
            .reshape(base_ev.n_units, base_ev.n_interventions, base_ev.n_periods)
        )
        alt_a_cost = float(srow["total_cost_usd_millions"])
        extra_lpg_opex = 0.0
        for u_i, u in enumerate(base_ev.units):
            lpg_uc = float(base_ev._unit_costs[u_i, base_ev._idx_lpg])
            for p_i in range(base_ev.n_periods):
                extra_lpg_opex += (
                    float(ten[u_i, base_ev._idx_lpg, p_i])
                    * lpg_uc
                    * float(lpg_rec_mult[u][p_i] - 1.0)
                )
        alt_b_cost = alt_a_cost + extra_lpg_opex

        # Counterfactual P3 deferral of LPG (identical 2040 terminal DALYs under Alternative B)
        ten_p3_shift = ten.copy()
        for u_i in range(base_ev.n_units):
            tot_lpg = float(np.sum(ten_p3_shift[u_i, base_ev._idx_lpg, :]))
            cap_p3 = float(base_ev._max_adoption[u_i, base_ev._idx_lpg, 2])
            if tot_lpg <= cap_p3 + 1e-12 and (
                base_ev._lpg_infra_init[u_i] >= 1.0 or base_ev._infra_enabled[u_i]
            ):
                ten_p3_shift[u_i, base_ev._idx_lpg, :] = [0.0, 0.0, tot_lpg]
        extra_p3 = 0.0
        for u_i, u in enumerate(base_ev.units):
            lpg_uc = float(base_ev._unit_costs[u_i, base_ev._idx_lpg])
            for p_i in range(base_ev.n_periods):
                extra_p3 += (
                    float(ten_p3_shift[u_i, base_ev._idx_lpg, p_i])
                    * lpg_uc
                    * float(lpg_rec_mult[u][p_i] - 1.0)
                )
        rec_rows.append(
            {
                "plan_name": pname,
                "dalys_2040_mean": float(srow["dalys_2040_mean"]),
                "alt_a_one_time_cost_usd_m": alt_a_cost,
                "alt_b_recurring_lpg_cost_usd_m": alt_b_cost,
                "alt_b_extra_lpg_opex_usd_m": extra_lpg_opex,
                "alt_b_cost_ratio_vs_alt_a": alt_b_cost / max(alt_a_cost, 1e-9),
                "alt_b_p3_deferred_cost_usd_m": alt_a_cost + extra_p3,
                "fits_base_budget_under_alt_b": alt_b_cost
                <= float(base_ev.total_budget_usd_millions),
                "is_mock": False,
                "data_version": base_ev.data_version,
                "parameter_version": PARAM_VERSION,
            }
        )

    df_rec = pd.DataFrame(rec_rows)
    df_rec.to_csv(recurring_csv, index=False)
    _write_sidecar(
        recurring_csv,
        source="src.optimise.final_real_suite.run_targeted_sensitivity_suite",
        config=base_config,
        data_version=base_contracts.surrogate["data_version"],
        scenario_assumptions=(
            "Alternative B recurring 5-yr LPG opex sensitivity (P1=2.49-2.69x, "
            "P2=1.75-1.84x, P3=1.0x) vs Alternative A one-time cohort package."
        ),
    )
    return df_sens, df_rec


def plot_country_period_heatmap(
    eapc_csv: Path,
    summary_csv: Path,
    out_path: Path,
) -> Path:
    """Render Country x Period EAPC & 1990-2023 ASR change heatmap (Phase 4)."""
    eapc_df = pd.read_csv(eapc_csv)
    sum_df = pd.read_csv(summary_csv)

    locs = [
        "India",
        "Nepal",
        "Pakistan",
        "Bangladesh",
        "Bhutan",
        "IND_urban",
        "IND_rural",
        "NPL_urban",
        "NPL_rural",
    ]
    periods = ["1990-2023", "1990-2010", "2010-2023"]

    sub_e = eapc_df[
        (eapc_df["measure"] == "DALYs")
        & (eapc_df["sex"] == "both")
        & (eapc_df["location"].isin(locs))
    ]
    piv = sub_e.pivot(index="location", columns="period", values="eapc").reindex(
        index=locs, columns=periods
    )

    sub_s = (
        sum_df[(sum_df["measure"] == "DALYs") & (sum_df["sex"] == "both")]
        .set_index("location")
        .reindex(locs)
    )
    piv["ASR % Change\n(1990–2023)"] = sub_s["asr_pct_change"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [3, 1.15]}
    )
    mat1 = piv[periods].to_numpy(dtype=float)
    im1 = ax1.imshow(mat1, cmap="RdYlGn_r", vmin=-14.0, vmax=0.0, aspect="auto")
    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels(periods, fontsize=9.5)
    ax1.set_yticks(range(len(locs)))
    ax1.set_yticklabels(locs, fontsize=9.5)
    ax1.set_title("DALY ASR EAPC (%/yr) by Period", fontsize=11, fontweight="bold")
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            val = mat1[i, j]
            txt_col = "white" if val < -9.0 else "black"
            ax1.text(
                j,
                i,
                f"{val:.2f}%",
                ha="center",
                va="center",
                color=txt_col,
                fontsize=9,
                fontweight="bold",
            )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="EAPC (%/yr)")

    mat2 = piv[["ASR % Change\n(1990–2023)"]].to_numpy(dtype=float)
    im2 = ax2.imshow(mat2, cmap="RdYlGn_r", vmin=-95.0, vmax=-20.0, aspect="auto")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["ASR % Change\n(1990–2023)"], fontsize=9.5)
    ax2.set_yticks(range(len(locs)))
    ax2.set_yticklabels([""] * len(locs))
    ax2.set_title("Cumulative ASR Shift", fontsize=11, fontweight="bold")
    for i in range(mat2.shape[0]):
        val = mat2[i, 0]
        txt_col = "white" if val < -75.0 else "black"
        ax2.text(
            0,
            i,
            f"{val:.1f}%",
            ha="center",
            va="center",
            color=txt_col,
            fontsize=9,
            fontweight="bold",
        )
    fig.colorbar(im2, ax=ax2, fraction=0.12, pad=0.08, label="% Change")

    fig.suptitle(
        "South Asia HAP–COPD DALY ASR Trends by Country/Stratum & Epoch (1990–2023)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        (
            "Source: IHME GBD 2023 (modelled estimates, both sexes, Age-standardized Rate "
            "per 100k); IND/NPL urban-rural apportioned (ASM-01)."
        ),
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_uncertainty_reeval(reeval_df: pd.DataFrame, out_path: Path) -> Path:
    """Plot analytical vs 500-draw Stage 2 & joint 95% UI intervals across selected plans."""
    sub = reeval_df[
        reeval_df["plan_name"].isin(
            ["bau_zero_action", "ga_cost_minimal", "ga_knee", "ga_daly_minimal"]
        )
    ].copy()
    title_map = {
        "bau_zero_action": "2040 BAU",
        "ga_cost_minimal": "GA Cost-Minimal",
        "ga_knee": "GA Knee",
        "ga_daly_minimal": "GA DALY-Minimal",
    }
    sub["label"] = [
        (
            f"{title_map.get(str(r['plan_name']), str(r['plan_name']))}\n"
            f"(${float(r['total_cost_usd_millions']):,.1f}M)"
        )
        for _, r in sub.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(sub))
    h = 0.24

    for _idx, (col_m, col_l, col_u, lbl, col, dy) in enumerate(
        [
            (
                "analytical_dalys_mean",
                "analytical_dalys_lower",
                "analytical_dalys_upper",
                "Analytical Surrogate UI",
                "#0072B2",
                -h,
            ),
            (
                "stage2_draws_dalys_mean",
                "stage2_draws_dalys_p025",
                "stage2_draws_dalys_p975",
                "Stage 2 Forecast Draws (n=500)",
                "#009E73",
                0.0,
            ),
            (
                "joint_draws_dalys_mean",
                "joint_draws_dalys_p025",
                "joint_draws_dalys_p975",
                "Joint Stage 2 + Surrogate UI",
                "#D55E00",
                h,
            ),
        ]
    ):
        means = sub[col_m].to_numpy() / 1e6
        err_lo = (sub[col_m] - sub[col_l]).to_numpy() / 1e6
        err_hi = (sub[col_u] - sub[col_m]).to_numpy() / 1e6
        ax.errorbar(
            means,
            y + dy,
            xerr=[err_lo, err_hi],
            fmt="o",
            color=col,
            capsize=4,
            markersize=6,
            linewidth=1.8,
            label=lbl,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Projected 2040 HAP–COPD DALYs (Millions, India + Nepal)")
    ax.set_title(
        "2040 Burden Uncertainty: Analytical vs. 500 Stage 2 Monte Carlo Draws", fontweight="bold"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True)
    fig.text(
        0.5,
        0.01,
        (
            "Note: Real GBD 2023 / HistGBDT 2040 draws (stage2_uncertainty_draws.csv, "
            "is_mock=false); scenario interventions, not policy predictions."
        ),
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_sensitivity_tornado(sens_df: pd.DataFrame, out_path: Path) -> Path:
    """Plot 8-factor sensitivity impacts on GA DALY-minimal and MDP DALYs averted."""
    base = sens_df[sens_df["scenario_id"] == "base_50pct_scenario_a"].iloc[0]
    base_ga = float(base["ga_daly_min_dalys_averted"]) / 1e3
    base_mdp = float(base["mdp_mean_dalys_averted"]) / 1e3

    sub = sens_df[sens_df["scenario_id"] != "base_50pct_scenario_a"].copy()
    sub["ga_k"] = sub["ga_daly_min_dalys_averted"] / 1e3
    sub["mdp_k"] = sub["mdp_mean_dalys_averted"] / 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    y = np.arange(len(sub))

    ax1.axvline(
        base_ga,
        color="#0f172a",
        linestyle="--",
        linewidth=1.4,
        label=f"Base GA DALY-min ({base_ga:.1f}k)",
    )
    diffs_ga = sub["ga_k"] - base_ga
    cols_ga = ["#009E73" if d >= 0 else "#D55E00" for d in diffs_ga]
    ax1.barh(y, diffs_ga, left=base_ga, color=cols_ga, alpha=0.85, height=0.65)
    ax1.set_yticks(y)
    ax1.set_yticklabels(sub["scenario_label"], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("GA DALY-Minimal: 2040 DALYs Averted (Thousands)")
    ax1.set_title("NSGA-II (DALY-Minimal) Sensitivity", fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2.axvline(
        base_mdp,
        color="#0f172a",
        linestyle="--",
        linewidth=1.4,
        label=f"Base MDP ({base_mdp:.1f}k)",
    )
    diffs_mdp = sub["mdp_k"] - base_mdp
    cols_mdp = ["#0072B2" if d >= 0 else "#E69F00" for d in diffs_mdp]
    ax2.barh(y, diffs_mdp, left=base_mdp, color=cols_mdp, alpha=0.85, height=0.65)
    ax2.set_xlabel("Adaptive MDP: Expected 2040 DALYs Averted (Thousands)")
    ax2.set_title("Adaptive MDP (b=6, lambda=116) Sensitivity", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Targeted 8-Factor Sensitivity Analysis (Real Data, Scenario A Base Reference)",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.01,
        (
            "One-factor-at-a-time shifts around Base (50% budget $5,985.73M, "
            "bridge=1.0, eta_bio=0.25, gamma=0.774, Scenario A)."
        ),
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def bundle_final_real_outputs(
    config: dict[str, Any],
    *,
    root: Path,
    final_dir: Path,
) -> dict[str, list[str]]:
    """Copy all verified real-data tables and figures into ``outputs/final_real_data/``."""
    src_tables = root / config["paths"]["outputs_tables_dir"]
    src_figs = root / config["paths"]["outputs_figures_dir"]
    dst_tables = final_dir / "tables"
    dst_figs = final_dir / "figures"
    dst_tables.mkdir(parents=True, exist_ok=True)
    dst_figs.mkdir(parents=True, exist_ok=True)

    copied_tables: list[str] = []
    copied_figs: list[str] = []

    for p in sorted(src_tables.iterdir()):
        if not p.is_file() or "_mock" in p.name or p.name == ".gitkeep":
            continue
        shutil.copy2(p, dst_tables / p.name)
        copied_tables.append(p.name)

    for p in sorted(src_figs.iterdir()):
        if not p.is_file() or "_mock" in p.name or p.name == ".gitkeep":
            continue
        shutil.copy2(p, dst_figs / p.name)
        copied_figs.append(p.name)

    manifest = {
        "bundle": "outputs/final_real_data",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_mock": False,
        "data_version": "GBD_2023_WPP_2024",
        "parameter_version": PARAM_VERSION,
        "tables": copied_tables,
        "figures": copied_figs,
    }
    (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"tables": copied_tables, "figures": copied_figs}


def run_final_real_suite(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Execute Stage 2 uncertainty re-eval, 8-factor sensitivity, Phase 4 figs, and final bundle."""
    root = Path(".")
    config = load_config(config_path)
    contracts = load_stage3_contracts(config, base_dir=root)
    if contracts.is_mock:
        raise ValueError("final_real_suite requires use_mock_forecasts: false in config.")
    evaluator = Stage3Evaluator(config=config, contracts=contracts)

    tables_dir = root / config["paths"]["outputs_tables_dir"]
    figures_dir = root / config["paths"]["outputs_figures_dir"]
    pareto_df = pd.read_csv(tables_dir / "s3_pareto_front.csv")
    selected_df = pd.read_csv(tables_dir / "s3_selected_plans.csv")

    # 1. Stage 2 uncertainty draws re-evaluation
    reeval_csv = tables_dir / "s3_pareto_reeval.csv"
    reeval_df = run_pareto_uncertainty_reevaluation(
        config,
        contracts,
        evaluator,
        pareto_df=pareto_df,
        selected_df=selected_df,
        draws_path=root
        / config.get("interfaces", {}).get(
            "stage2_draws_file", "data/processed/stage2_uncertainty_draws.csv"
        ),
        out_csv=reeval_csv,
    )

    # 2. Targeted 8-factor sensitivity suite
    sens_csv = tables_dir / "s3_sensitivity_real.csv"
    rec_csv = tables_dir / "s3_recurring_opex_sensitivity.csv"
    sens_df, rec_df = run_targeted_sensitivity_suite(config, contracts, sens_csv, rec_csv)

    # 3. Additional Phase 4 figures + sidecars
    hm_fig = plot_country_period_heatmap(
        tables_dir / "s1_eapc_joinpoint.csv",
        tables_dir / "s1_historical_summary.csv",
        figures_dir / "s1_country_period_heatmap.png",
    )
    _write_sidecar(
        hm_fig,
        source="src.optimise.final_real_suite.plot_country_period_heatmap",
        config=config,
        data_version="GBD_2023_WPP_2024",
        scenario_assumptions=(
            "GBD 2023 Age-standardized DALY rate EAPC and cumulative 1990-2023 change."
        ),
    )

    uu_fig = plot_uncertainty_reeval(reeval_df, figures_dir / "s3_pareto_uncertainty_reeval.png")
    _write_sidecar(
        uu_fig,
        source="src.optimise.final_real_suite.plot_uncertainty_reeval",
        config=config,
        data_version=evaluator.data_version,
        scenario_assumptions=(
            "Analytical surrogate 95% UI vs 500 Stage 2 forecast draws vs joint uncertainty."
        ),
    )

    tor_fig = plot_sensitivity_tornado(sens_df, figures_dir / "plan_sensitivity_tornado.png")
    _write_sidecar(
        tor_fig,
        source="src.optimise.final_real_suite.plot_sensitivity_tornado",
        config=config,
        data_version=evaluator.data_version,
        scenario_assumptions="8-factor targeted sensitivity analysis around Scenario A Base case.",
    )

    # 4. Curated bundle in outputs/final_real_data/
    final_dir = root / "outputs" / "final_real_data"
    bundled = bundle_final_real_outputs(config, root=root, final_dir=final_dir)

    return {
        "reeval_df": reeval_df,
        "sens_df": sens_df,
        "rec_df": rec_df,
        "bundled": bundled,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 3-4 real-data uncertainty & sensitivity suite."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args(argv)
    res = run_final_real_suite(args.config)
    print("Stage 2 Pareto re-evaluation complete:", len(res["reeval_df"]), "plans")
    print("Targeted 8-factor sensitivity complete:", len(res["sens_df"]), "scenarios")
    print("Recurring opex sensitivity complete:", len(res["rec_df"]), "plans")
    print(
        "Bundled into outputs/final_real_data/:",
        len(res["bundled"]["tables"]),
        "tables,",
        len(res["bundled"]["figures"]),
        "figures",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
