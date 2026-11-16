"""B-09 Final Baseline Harmonisation & Bridge-Factor Sensitivity Analysis.

Produces:
- ``research/person_b_inputs/b09_baseline_comparison.csv``
- ``research/person_b_inputs/b09_budget_scenarios.csv``
- ``research/person_b_inputs/b09_bridge_sensitivity.csv``
- ``research/person_b_inputs/b09_feasibility_audit.csv``

Does not modify shared configs, mock fixtures, or Person A files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.contracts import load_config, load_stage3_contracts  # noqa: E402
from src.optimise.evaluator import Stage3Evaluator  # noqa: E402
from src.optimise.nsga2 import (  # noqa: E402
    run_nsga2_multi_seed,
    select_representative_plans,
)
from src.planning.mdp import (  # noqa: E402
    MDPConfig,
    build_layered_mdp,
    simulate_mdp_rollouts,
    solve_finite_horizon_backward_induction,
)

BASE_DIR = Path(__file__).resolve().parent
UNITS: tuple[str, ...] = ("IND_urban", "IND_rural", "NPL_urban", "NPL_rural")


def build_baseline_and_budget_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce separate population vs household baseline, budget, and feasibility tables."""
    prop = json.loads((BASE_DIR / "proposed_intervention_params.json").read_text(encoding="utf-8"))
    hh_2025 = prop["_auxiliary_households_2025_derived"]
    costs = prop["unit_costs_usd_millions"]
    ceil = prop["coverage_ceiling"]
    caps = prop["max_adoption_per_period"]

    # Verified baselines from B-07/B-08 raw files
    pop_who_2024 = {"IND_urban": 0.96, "IND_rural": 0.69, "NPL_urban": 0.64, "NPL_rural": 0.25}
    hh_obs = {"IND_urban": 0.897, "IND_rural": 0.432, "NPL_urban": 0.5846, "NPL_rural": 0.2036}
    hh_obs_src = {
        "IND_urban": "India NFHS-5 (2019-21, DHS HC_CKFL_H_CLN)",
        "IND_rural": "India NFHS-5 (2019-21, DHS HC_CKFL_H_CLN)",
        "NPL_urban": "Nepal NPHC 2021 Table 07 (conventional HH)",
        "NPL_rural": "Nepal NPHC 2021 Table 07 (conventional HH)",
    }
    hh_obs_yr = {"IND_urban": 2020, "IND_rural": 2020, "NPL_urban": 2021, "NPL_rural": 2021}
    # DHS observed (HH - Pop) gap in survey year for explicit sensitivity reference
    dhs_spread = {"IND_urban": 0.007, "IND_rural": 0.023, "NPL_urban": 0.042, "NPL_rural": 0.019}

    baseline_rows: list[dict[str, Any]] = []
    feas_rows: list[dict[str, Any]] = []
    for u in UNITS:
        pop_b = pop_who_2024[u]
        hh_b = hh_obs[u]
        c_max = ceil[u]
        hh_gap_obs = c_max - hh_b
        pop_gap_2024 = c_max - pop_b
        hh_2024_sens = min(1.0, pop_b + dhs_spread[u])

        baseline_rows.append(
            {
                "unit": u,
                "households_2025_derived": hh_2025[u],
                "baseline_clean_fuel_population": pop_b,
                "population_baseline_year": 2024,
                "population_baseline_source": "WHO GHO / WB SDG 7.1.2 (2024, people)",
                "baseline_clean_fuel_households_obs": hh_b,
                "household_baseline_year": hh_obs_yr[u],
                "household_baseline_source": hh_obs_src[u],
                "dhs_hh_minus_pop_spread_obs": dhs_spread[u],
                "household_2024_sensitivity_derived": round(hh_2024_sens, 4),
                "coverage_ceiling_households": c_max,
                "gap_households_from_obs": round(hh_gap_obs, 4),
                "gap_population_from_2024": round(pop_gap_2024, 4),
                "bau_2040_population_status": "blocked_person_a (Stage 2 pending)",
            }
        )

        # Feasibility check under adoption caps & strictly-earlier infra rule (q < p)
        lpg_all = sum(caps[u]["lpg_subsidy"].values())
        lpg_p23 = caps[u]["lpg_subsidy"]["P2"] + caps[u]["lpg_subsidy"]["P3"]
        elec_all = sum(caps[u]["electric_induction"].values())
        elec_p23 = caps[u]["electric_induction"]["P2"] + caps[u]["electric_induction"]["P3"]
        bio_all = sum(caps[u]["improved_biomass"].values())
        init_ready = u in ("IND_urban", "NPL_urban")
        lpg_unlocked_cap = lpg_all if init_ready else lpg_p23
        elec_unlocked_cap = elec_all if init_ready else elec_p23
        combined_unlocked = lpg_unlocked_cap + elec_unlocked_cap + bio_all

        feas_rows.append(
            {
                "unit": u,
                "infra_ready_init_assumed": init_ready,
                "hh_obs_baseline": hh_b,
                "pop_2024_baseline": pop_b,
                "coverage_ceiling": c_max,
                "gap_hh_obs": round(hh_gap_obs, 4),
                "gap_pop_2024": round(pop_gap_2024, 4),
                "lpg_unlocked_cap_sum": round(lpg_unlocked_cap, 4),
                "elec_unlocked_cap_sum": round(elec_unlocked_cap, 4),
                "bio_cap_sum": round(bio_all, 4),
                "combined_unlocked_cap_sum": round(combined_unlocked, 4),
                "lpg_alone_closes_hh_obs_gap": lpg_unlocked_cap + 1e-12 >= hh_gap_obs,
                "lpg_alone_closes_pop_2024_gap": lpg_unlocked_cap + 1e-12 >= pop_gap_2024,
                "combined_closes_hh_obs_gap": combined_unlocked + 1e-12 >= hh_gap_obs,
                "combined_closes_pop_2024_gap": combined_unlocked + 1e-12 >= pop_gap_2024,
                "ceiling_margin_over_pop_2024": round(c_max - pop_b, 4),
                "notes": (
                    "CRITICAL: 2024 pop=0.96 -> 2040 BAU likely ~0.98-0.99; verify ceiling >= cf_bau_2040."
                    if u == "IND_urban"
                    else (
                        "CRITICAL: q<p blocks P1 LPG/Elec; unlocked P2+P3 LPG+Elec=0.60 < 0.70/0.7464 gap; needs P1 improved_biomass."
                        if u == "NPL_rural"
                        else "Unlocked P2+P3 or P1-P3 caps sufficient for 2024 pop gap."
                    )
                ),
            }
        )

    # Budget Scenarios: Household Baseline vs Population Baseline vs Sensitivity (excl. infra)
    scenarios = [
        (
            "H_observed_hh_lpg_fiscal",
            "Observed Household Baseline (NFHS-5 2020 IND / NPHC 2021 NPL)",
            "observed_household",
            "households",
            "India NFHS-5 (2019-21, DHS HC_CKFL_H_CLN, 2020) & Nepal NPHC 2021 Table 07",
            hh_obs,
            "lpg_subsidy",
            1.0,
            "Primary unadjusted household-denominated observed gap (note IND 2020 predates PMUY 2.0).",
        ),
        (
            "H_2024_derived_hh_lpg_fiscal",
            "2024 Derived Household Baseline Sensitivity (WHO 2024 + DHS HH-Pop spread)",
            "derived_household_sensitivity",
            "households_sensitivity",
            "Derived (NOT observed): WHO SDG 7.1.2 (2024) + NFHS-5/NDHS observed HH-Pop spread",
            {u: min(1.0, pop_who_2024[u] + dhs_spread[u]) for u in UNITS},
            "lpg_subsidy",
            1.0,
            "Derived sensitivity only (never classified as observation): shifts WHO 2024 by DHS spread.",
        ),
        (
            "P_who2024_pop_lpg_fiscal",
            "Population Baseline Reference (WHO SDG 7.1.2 2024)",
            "observed_modelled_population",
            "people",
            "WHO GHO / WB SDG 7.1.2 (2024, people)",
            pop_who_2024,
            "lpg_subsidy",
            1.0,
            "Population-denominated baseline reference (do not confuse with household adoption denominator).",
        ),
        (
            "H_observed_hh_bio_50pct_cfa",
            "Observed Household Baseline - Improved Biomass (50% Public CFA)",
            "observed_household",
            "households",
            "India NFHS-5 (2020) & Nepal NPHC 2021 + 50% MNRE/AEPC CFA assumption",
            hh_obs,
            "improved_biomass",
            0.5,
            "Scenario assumption: 50% MNRE/AEPC CFA applied to $45/HH 5-yr societal ICS cost ($22.50/HH).",
        ),
        (
            "H_observed_hh_elec_50pct_hw",
            "Observed Household Baseline - Electric Induction (50% Public HW)",
            "observed_household",
            "households",
            "India NFHS-5 (2020) & Nepal NPHC 2021 + 50% public hardware co-finance assumption",
            hh_obs,
            "electric_induction",
            0.5,
            "Scenario assumption: 50% public hardware co-finance (India lacks central induction subsidy).",
        ),
    ]

    budget_rows: list[dict[str, Any]] = []
    for sid, label, base_cls, denom_basis, prov_src, base_map, interv, mult, note in scenarios:
        u_costs = {}
        for u in UNITS:
            gap = max(0.0, ceil[u] - base_map[u])
            u_costs[u] = gap * costs[u][interv] * mult
        tot = sum(u_costs.values())
        budget_rows.append(
            {
                "scenario_id": sid,
                "scenario_label": label,
                "baseline_classification": base_cls,
                "denominator_basis": denom_basis,
                "baseline_provenance": prov_src,
                "intervention_basis": interv,
                "cost_multiplier": mult,
                "IND_urban_cost_usd_m": round(u_costs["IND_urban"], 2),
                "IND_rural_cost_usd_m": round(u_costs["IND_rural"], 2),
                "NPL_urban_cost_usd_m": round(u_costs["NPL_urban"], 2),
                "NPL_rural_cost_usd_m": round(u_costs["NPL_rural"], 2),
                "full_gap_100pct_usd_m": round(tot, 2),
                "budget_50pct_usd_m": round(tot * 0.50, 2),
                "budget_25pct_usd_m": round(tot * 0.25, 2),
                "infra_cost_status": "excluded_unresolved_gap (null)",
                "note": note,
            }
        )

    return (
        pd.DataFrame(baseline_rows),
        pd.DataFrame(budget_rows),
        pd.DataFrame(feas_rows),
    )


def run_bridge_sensitivity() -> pd.DataFrame:
    """Test bridge_factor in {0.8, 1.0, 1.2} on Stage 3 NSGA-II and MDP (mock contracts)."""
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    base_contracts = load_stage3_contracts(config, base_dir=REPO_ROOT)

    # Reduce pop/gen slightly? No, full mock NSGA-II + MDP runs in ~3s total!
    # Also test BOTH same baseline (default mock) AND separated hh vs pop baseline!
    mock_pop_2040 = {"IND_urban": 0.88, "IND_rural": 0.58, "NPL_urban": 0.76, "NPL_rural": 0.40}
    # Illustrative separated household baseline slightly higher (+0.01..+0.03 as in DHS)
    # Wait: keep IND_urban <= 0.90 so ceiling 0.98 has headroom
    hh_sep_mock = {"IND_urban": 0.89, "IND_rural": 0.60, "NPL_urban": 0.79, "NPL_rural": 0.42}

    rows: list[dict[str, Any]] = []
    for mode_name, hh_override in [
        ("mock_aligned_baselines", None),
        ("mock_separated_hh_baseline", hh_sep_mock),
    ]:
        for bf in (0.8, 1.0, 1.2):
            ev = Stage3Evaluator(
                config=config,
                contracts=base_contracts,
                baseline_clean_fuel_households=hh_override,
                coverage_bridge_factor={u: bf for u in UNITS},
            )
            bau_res = ev.evaluate(np.zeros(ev.n_vars))
            multi = run_nsga2_multi_seed(ev, config)
            sel_df, _ = select_representative_plans(
                multi.combined_X, multi.combined_evaluation, ev
            )
            var_cols = list(ev.variable_names)

            # Check feasibility under BOTH household baseline (ev) and population ceiling (<= 1.0)
            for _, srow in sel_df.iterrows():
                x_plan = srow[var_cols].to_numpy(dtype=float)
                r_eval = ev.evaluate(x_plan)
                ten = x_plan.reshape(ev.n_units, ev.n_interventions, ev.n_periods)
                raw_gain = np.sum(ten[:, ev._non_infra_indices, :], axis=(1, 2))
                pop_after = np.array([mock_pop_2040[u] for u in UNITS]) + bf * raw_gain
                pop_overshoot = float(np.sum(np.maximum(0.0, pop_after - 1.0)))

                rows.append(
                    {
                        "baseline_mode": mode_name,
                        "bridge_factor": bf,
                        "method": f"GA_{srow['plan_name']}",
                        "dalys_2040_mean": round(float(r_eval.dalys_2040_mean), 2),
                        "dalys_averted_vs_bau": round(
                            float(bau_res.dalys_2040_mean - r_eval.dalys_2040_mean), 2
                        ),
                        "total_cost_usd_millions": round(float(r_eval.total_cost_usd_millions), 2),
                        "equity_gap_asr": round(float(r_eval.equity_gap_asr), 2),
                        "feasible_hh_rules": bool(r_eval.is_feasible),
                        "pop_100pct_overshoot": round(pop_overshoot, 6),
                        "feasible_both_hh_and_pop100": bool(
                            r_eval.is_feasible and pop_overshoot <= 1e-9
                        ),
                        "n_pareto_solutions": int(len(multi.combined_X)),
                        "hypervolume": round(float(multi.combined_hypervolume), 2),
                    }
                )

            # Solve MDP (b=6) on same evaluator & simulate nominal/stochastic rollouts
            mdp_cfg = MDPConfig.from_config(config)
            mdp = build_layered_mdp(ev, mdp_cfg)
            sol = solve_finite_horizon_backward_induction(
                mdp,
                cost_weight_lambda=mdp_cfg.cost_weight_lambda,
                discount_factor=mdp_cfg.discount_factor,
            )
            roll = simulate_mdp_rollouts(
                mdp, sol, ev, n_rollouts=200, seed=mdp_cfg.seed
            )
            # Compute worst-case terminal population overshoot across reachable MDP terminal states
            term_states = np.asarray(mdp.layers[-1], dtype=float)
            term_raw_bins = (
                term_states[:, : ev.n_units]
                + term_states[:, ev.n_units : 2 * ev.n_units]
                + term_states[:, 2 * ev.n_units : 3 * ev.n_units]
            )
            term_raw_gain = term_raw_bins * mdp.step_u[np.newaxis, :]
            term_pop_after = (
                np.array([mock_pop_2040[u] for u in UNITS])[np.newaxis, :] + bf * term_raw_gain
            )
            max_mdp_pop_overshoot = float(np.max(np.sum(np.maximum(0.0, term_pop_after - 1.0), axis=1)))
            rows.append(
                {
                    "baseline_mode": mode_name,
                    "bridge_factor": bf,
                    "method": "MDP_b6_mean_rollout",
                    "dalys_2040_mean": round(float(roll["dalys_2040_mean"].mean()), 2),
                    "dalys_averted_vs_bau": round(float(roll["dalys_averted_mean"].mean()), 2),
                    "total_cost_usd_millions": round(float(roll["total_cost_usd_millions"].mean()), 2),
                    "equity_gap_asr": round(float(roll["equity_gap_asr"].mean()), 2),
                    "feasible_hh_rules": bool(roll["is_feasible_in_evaluator"].all()),
                    "pop_100pct_overshoot": round(max_mdp_pop_overshoot, 6),
                    "feasible_both_hh_and_pop100": bool(
                        roll["is_feasible_in_evaluator"].all() and max_mdp_pop_overshoot <= 1e-9
                    ),
                    "n_pareto_solutions": int(len(multi.combined_X)),
                    "hypervolume": round(float(multi.combined_hypervolume), 2),
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    base_df, budget_df, feas_df = build_baseline_and_budget_tables()
    base_df.to_csv(BASE_DIR / "b09_baseline_comparison.csv", index=False)
    budget_df.to_csv(BASE_DIR / "b09_budget_scenarios.csv", index=False)
    feas_df.to_csv(BASE_DIR / "b09_feasibility_audit.csv", index=False)

    sens_df = run_bridge_sensitivity()
    sens_df.to_csv(BASE_DIR / "b09_bridge_sensitivity.csv", index=False)

    print("=== BASELINE COMPARISON ===")
    print(base_df[["unit", "households_2025_derived", "baseline_clean_fuel_population", "baseline_clean_fuel_households_obs", "household_2024_sensitivity_derived", "coverage_ceiling_households", "gap_households_from_obs", "gap_population_from_2024"]].to_string(index=False))
    print("\n=== BUDGET SCENARIOS ===")
    print(budget_df[["scenario_id", "denominator_basis", "full_gap_100pct_usd_m", "budget_50pct_usd_m", "budget_25pct_usd_m"]].to_string(index=False))
    print("\n=== BRIDGE SENSITIVITY (GA_knee, GA_daly_minimal, MDP_b6) ===")
    print(sens_df[sens_df["method"].isin(["GA_knee", "GA_daly_minimal", "MDP_b6_mean_rollout"])].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
