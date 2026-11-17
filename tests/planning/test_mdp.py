"""Unit and integration tests for Stage 4 custom NumPy MDP (``src/planning/mdp.py``)."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from src.data.contracts import Stage3Contracts, load_config, load_stage3_contracts
from src.optimise.evaluator import Stage3Evaluator
from src.planning.mdp import (
    ACT_LPG_1,
    MDPConfig,
    _apply_action_branch,
    build_layered_mdp,
    build_stationary_mdp,
    run_mdp_pipeline,
    simulate_mdp_rollouts,
    solve_finite_horizon_backward_induction,
    solve_infinite_horizon_value_iteration,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mock.yaml"


def _make_tiny_1unit_evaluator(
    *,
    budget: float = 100.0,
    lpg_init_ready: bool = False,
) -> tuple[dict[str, Any], Stage3Evaluator]:
    """Create a minimal 1-unit, 2-period evaluator for exact brute-force verification."""
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    cfg["geography"]["allocation_units"] = ["NPL_urban"]
    cfg["time"]["decision_periods"] = [
        {"id": "P1", "label": "2026-2030", "start_year": 2026, "end_year": 2030},
        {"id": "P2", "label": "2031-2035", "start_year": 2031, "end_year": 2035},
    ]
    base_contracts = load_stage3_contracts(load_config(CONFIG_PATH), base_dir=REPO_ROOT)

    f_df = base_contracts.forecasts_stage3[
        base_contracts.forecasts_stage3["location"] == "NPL_urban"
    ].copy()
    b_df = base_contracts.baseline_unit_state[
        base_contracts.baseline_unit_state["location"] == "NPL_urban"
    ].copy()
    b_df["lpg_infra_ready_init"] = bool(lpg_init_ready)
    b_df["grid_ready_init"] = bool(lpg_init_ready)

    surr = copy.deepcopy(base_contracts.surrogate)
    surr["units"] = {"NPL_urban": surr["units"]["NPL_urban"]}

    params = copy.deepcopy(base_contracts.intervention_params)
    params["total_budget_usd_millions"] = float(budget)
    params["period_order"] = ["P1", "P2"]
    for map_key in ("coverage_ceiling", "effectiveness", "unit_costs_usd_millions"):
        params[map_key] = {"NPL_urban": params[map_key]["NPL_urban"]}
    params["max_adoption_per_period"] = {
        "NPL_urban": {
            interv: {p: float(vals[p]) for p in ("P1", "P2")}
            for interv, vals in params["max_adoption_per_period"]["NPL_urban"].items()
        }
    }

    tiny_contracts = Stage3Contracts(
        forecasts_stage3=f_df,
        baseline_unit_state=b_df,
        surrogate=surr,
        intervention_params=params,
        is_mock=True,
        paths=base_contracts.paths,
    )
    return cfg, Stage3Evaluator(config=cfg, contracts=tiny_contracts)


def _tiny_mdp_cfg(lam: float = 1500.0, gamma: float = 0.774) -> MDPConfig:
    return MDPConfig(
        coverage_bins=(0.0, 0.5, 1.0),
        budget_bins=3,
        discount_factor=gamma,
        cost_weight_lambda=lam,
        transition_probs=(0.2, 0.6, 0.2),
        n_rollouts=20,
        vi_tol=1e-6,
        vi_max_iter=200,
        lambda_sensitivity=(0.0, lam),
        discount_sensitivity=(gamma,),
        seed=42,
    )


def test_mdp_matches_brute_force_enumeration_on_tiny_instance() -> None:
    _, ev = _make_tiny_1unit_evaluator(budget=80.0, lpg_init_ready=False)
    mdp_cfg = _tiny_mdp_cfg(lam=1500.0, gamma=0.774)
    mdp = build_layered_mdp(ev, mdp_cfg)
    sol = solve_finite_horizon_backward_induction(
        mdp, cost_weight_lambda=mdp_cfg.cost_weight_lambda, discount_factor=mdp_cfg.discount_factor
    )

    # Brute-force enumerate all valid 2-period non-anticipative policy trees from s0
    tr0 = mdp.transitions[0]
    tr1 = mdp.transitions[1]
    edges_t1_by_state: dict[int, list[tuple[float, np.ndarray, np.ndarray]]] = {
        s_i: [] for s_i in range(len(mdp.layers[1]))
    }
    for e_i in range(len(tr1.src_state)):
        edges_t1_by_state[int(tr1.src_state[e_i])].append(
            (float(tr1.cost[e_i]), tr1.dalys_averted[e_i], tr1.next_state[e_i])
        )

    best_brute_val = -np.inf
    probs = mdp.probs
    lam = mdp_cfg.cost_weight_lambda
    gamma = mdp_cfg.discount_factor

    for e0 in range(len(tr0.src_state)):
        c0 = float(tr0.cost[e0])
        da0 = tr0.dalys_averted[e0]
        ns0 = tr0.next_state[e0]
        # For each stochastic branch b in {0, 1, 2}, pick best t=1 action independently
        b_opts = [edges_t1_by_state[int(ns0[b])] for b in range(3)]
        for e1_0, e1_1, e1_2 in itertools.product(*b_opts):
            chosen_t1 = (e1_0, e1_1, e1_2)
            expected_val = 0.0
            for b in range(3):
                r0 = float(da0[b]) - lam * c0
                c1, da1, _ = chosen_t1[b]
                r1 = float(np.sum(probs * (da1 - lam * c1)))
                expected_val += float(probs[b]) * (r0 + gamma * r1)
            best_brute_val = max(best_brute_val, expected_val)

    assert sol.expected_value_s0 > 0.0
    assert sol.expected_value_s0 == pytest.approx(best_brute_val, rel=1e-9, abs=1e-6)


def test_mdp_value_monotone_in_budget_and_zero_budget_gives_bau() -> None:
    mdp_cfg = _tiny_mdp_cfg(lam=100.0, gamma=0.774)
    values_by_budget: list[float] = []

    for budget in (0.0, 2.0, 20.0):
        _, ev = _make_tiny_1unit_evaluator(budget=budget, lpg_init_ready=True)
        mdp = build_layered_mdp(ev, mdp_cfg)
        sol = solve_finite_horizon_backward_induction(
            mdp,
            cost_weight_lambda=mdp_cfg.cost_weight_lambda,
            discount_factor=mdp_cfg.discount_factor,
        )
        values_by_budget.append(sol.expected_value_s0)

        if budget == 0.0:
            roll_df = simulate_mdp_rollouts(mdp, sol, ev, n_rollouts=10, seed=42)
            bau_res = ev.evaluate(np.zeros(ev.n_vars, dtype=float))
            assert np.allclose(
                roll_df["dalys_2040_mean"].to_numpy(), float(bau_res.dalys_2040_mean)
            )
            assert np.allclose(roll_df["total_cost_usd_millions"].to_numpy(), 0.0)
            assert bool(roll_df["is_feasible_in_evaluator"].all()) is True

    assert values_by_budget[0] <= values_by_budget[1] <= values_by_budget[2]
    assert values_by_budget[2] > values_by_budget[0]


def test_mdp_transitions_valid_distributions_and_seeded_reproducibility() -> None:
    _, ev = _make_tiny_1unit_evaluator(budget=100.0, lpg_init_ready=False)
    mdp_cfg = _tiny_mdp_cfg(lam=1500.0, gamma=0.774)
    mdp = build_layered_mdp(ev, mdp_cfg)

    assert np.all(mdp.probs >= 0.0)
    assert float(np.sum(mdp.probs)) == pytest.approx(1.0)
    for t in range(mdp.n_periods):
        assert np.all(mdp.transitions[t].state_counts >= 1)

    sol = solve_finite_horizon_backward_induction(
        mdp, cost_weight_lambda=mdp_cfg.cost_weight_lambda, discount_factor=mdp_cfg.discount_factor
    )
    stat_mdp = build_stationary_mdp(mdp, ev)
    ih_sol = solve_infinite_horizon_value_iteration(
        stat_mdp,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
        tol=1e-6,
    )
    assert ih_sol.converged is True
    assert ih_sol.expected_value_s0 == pytest.approx(sol.expected_value_s0, rel=1e-6)

    df_a = simulate_mdp_rollouts(mdp, sol, ev, n_rollouts=30, seed=42)
    df_b = simulate_mdp_rollouts(mdp, sol, ev, n_rollouts=30, seed=42)
    pd.testing.assert_frame_equal(df_a, df_b)
    assert bool(df_a["is_feasible_in_evaluator"].all()) is True


def test_run_mdp_pipeline_writes_mock_csvs_and_sidecars(tmp_path: Path) -> None:
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    for key in (
        "forecast_mock_file",
        "baseline_state_mock_file",
        "surrogate_mock_file",
        "intervention_params_mock_file",
    ):
        cfg["interfaces"][key] = str((REPO_ROOT / cfg["interfaces"][key]).resolve())

    cfg["paths"]["outputs_tables_dir"] = str(tmp_path / "tables")
    cfg["paths"]["outputs_figures_dir"] = str(tmp_path / "figures")
    cfg["planning"]["mdp"]["n_rollouts"] = 25
    cfg["planning"]["mdp"]["lambda_sensitivity"] = [0.0, 3500.0]
    cfg["planning"]["mdp"]["discount_sensitivity"] = [0.774]

    cfg_path = tmp_path / "mdp_test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    res = run_mdp_pipeline(cfg_path, base_dir=tmp_path, generate_figure=True)
    files = res["files"]

    for csv_key in ("policy_csv", "rollouts_csv", "sensitivity_csv"):
        csv_path: Path = files[csv_key]
        meta_path = Path(f"{csv_path}.meta.json")
        assert csv_path.exists()
        assert "_mock" in csv_path.stem
        assert meta_path.exists()
        df = pd.read_csv(csv_path)
        assert bool(df["is_mock"].all()) is True
        assert set(df["data_version"].unique()) == {"MOCK_V1"}
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["is_mock"] is True

    assert files["fig_mdp"].exists()
    assert files["fig_mdp"].stat().st_size > 1024
    assert bool(res["rollouts_df"]["is_feasible_in_evaluator"].all()) is True


def test_mdp_respects_stack_adjusted_lpg_and_electric_eta() -> None:
    cfg, base_ev = _make_tiny_1unit_evaluator(budget=80.0, lpg_init_ready=True)
    base_contracts = load_stage3_contracts(load_config(CONFIG_PATH), base_dir=REPO_ROOT)
    params = copy.deepcopy(base_contracts.intervention_params)
    params["total_budget_usd_millions"] = 80.0
    params["period_order"] = ["P1", "P2"]
    for map_key in ("coverage_ceiling", "effectiveness", "unit_costs_usd_millions"):
        params[map_key] = {"NPL_urban": params[map_key]["NPL_urban"]}
    params["effectiveness"]["NPL_urban"]["lpg_subsidy"] = 0.86
    params["effectiveness"]["NPL_urban"]["electric_induction"] = 0.90
    params["max_adoption_per_period"] = {
        "NPL_urban": {
            interv: {p: float(vals[p]) for p in ("P1", "P2")}
            for interv, vals in params["max_adoption_per_period"]["NPL_urban"].items()
        }
    }
    f_df = base_contracts.forecasts_stage3[
        base_contracts.forecasts_stage3["location"] == "NPL_urban"
    ].copy()
    b_df = base_contracts.baseline_unit_state[
        base_contracts.baseline_unit_state["location"] == "NPL_urban"
    ].copy()
    b_df["lpg_infra_ready_init"] = True
    b_df["grid_ready_init"] = True
    surr = copy.deepcopy(base_contracts.surrogate)
    surr["units"] = {"NPL_urban": surr["units"]["NPL_urban"]}
    contracts = Stage3Contracts(
        forecasts_stage3=f_df,
        baseline_unit_state=b_df,
        surrogate=surr,
        intervention_params=params,
        is_mock=True,
        paths=base_contracts.paths,
    )
    ev = Stage3Evaluator(config=cfg, contracts=contracts)
    mdp_cfg = _tiny_mdp_cfg(lam=1500.0, gamma=1.0)
    mdp = build_layered_mdp(ev, mdp_cfg)
    sol = solve_finite_horizon_backward_induction(
        mdp, cost_weight_lambda=1500.0, discount_factor=1.0
    )
    branches = np.ones((5, ev.n_periods), dtype=int)  # nominal branch everywhere
    roll_df = simulate_mdp_rollouts(
        mdp, sol, ev, n_rollouts=5, seed=42, precomputed_branches=branches
    )
    assert bool(roll_df["is_feasible_in_evaluator"].all()) is True
    expected_ret = float(roll_df["dalys_averted_mean"].iloc[0]) - 1500.0 * float(
        roll_df["total_cost_usd_millions"].iloc[0]
    )
    assert float(roll_df["discounted_return"].iloc[0]) == pytest.approx(expected_ret, rel=1e-9)


def test_mdp_and_comparator_enforce_equivalent_population_saturation() -> None:
    from src.planning.compare import draw_shock_branches, simulate_open_loop_schedule

    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    base_contracts = load_stage3_contracts(load_config(CONFIG_PATH), base_dir=REPO_ROOT)
    f_df = base_contracts.forecasts_stage3.copy()
    pop_map = {"IND_urban": 0.96, "IND_rural": 0.69, "NPL_urban": 0.64, "NPL_rural": 0.25}
    for unit, val in pop_map.items():
        mask = (f_df["location"] == unit) & (f_df["year"] == 2040) & (f_df["scenario"] == "bau")
        f_df.loc[mask, "clean_fuel_share"] = val
    params = copy.deepcopy(base_contracts.intervention_params)
    params["coverage_ceiling"]["IND_urban"] = 0.99
    contracts = Stage3Contracts(
        forecasts_stage3=f_df,
        baseline_unit_state=base_contracts.baseline_unit_state.copy(),
        surrogate=copy.deepcopy(base_contracts.surrogate),
        intervention_params=params,
        is_mock=True,
        paths=base_contracts.paths,
    )
    hh_base = {"IND_urban": 0.897, "IND_rural": 0.432, "NPL_urban": 0.5846, "NPL_rural": 0.2036}
    mdp_cfg = MDPConfig.from_config(cfg)

    for bf in (0.8, 1.0, 1.2):
        ev = Stage3Evaluator(
            config=cfg,
            contracts=contracts,
            baseline_clean_fuel_households=hh_base,
            coverage_bridge_factor={u: bf for u in hh_base},
        )
        mdp = build_layered_mdp(ev, mdp_cfg)
        assert np.all(mdp.max_bin * mdp.step_u <= ev.effective_adoption_headroom() + 1e-12)

        sol = solve_finite_horizon_backward_induction(
            mdp,
            cost_weight_lambda=mdp_cfg.cost_weight_lambda,
            discount_factor=mdp_cfg.discount_factor,
        )
        branches = draw_shock_branches(mdp_cfg.transition_probs, 40, ev.n_periods, 42)
        roll_df = simulate_mdp_rollouts(
            mdp, sol, ev, n_rollouts=40, seed=42, precomputed_branches=branches
        )
        assert bool(roll_df["is_feasible_in_evaluator"].all()) is True

        term_states = np.asarray(mdp.layers[-1], dtype=float)
        term_raw_bins = (
            term_states[:, : ev.n_units]
            + term_states[:, ev.n_units : 2 * ev.n_units]
            + term_states[:, 2 * ev.n_units : 3 * ev.n_units]
        )
        term_raw_gain = term_raw_bins * mdp.step_u[np.newaxis, :]
        assert np.all(
            ev._baseline_clean_fuel_population[np.newaxis, :] + bf * term_raw_gain <= 1.0 + 1e-12
        )

    # Comparator enforces evaluator saturation ceiling under nominal execution
    ev_10 = Stage3Evaluator(
        config=cfg,
        contracts=contracts,
        baseline_clean_fuel_households=hh_base,
        coverage_bridge_factor={u: 1.0 for u in hh_base},
    )
    x_oversat = np.zeros(ev_10.n_vars, dtype=float)
    ind_u = ev_10.units.index("IND_urban")
    lpg_i = ev_10.interventions.index("lpg_subsidy")
    x_oversat[ind_u * ev_10.n_interventions * ev_10.n_periods + lpg_i * ev_10.n_periods] = 0.06
    nom_branches = np.ones((4, ev_10.n_periods), dtype=int)
    sim_over = simulate_open_loop_schedule(
        x_oversat,
        ev_10,
        nom_branches,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    assert bool(sim_over["is_feasible_in_evaluator"].any()) is False

    # Saturated unit (pop=1.0 -> headroom=0.0) must build MDP safely and spend zero
    f_sat = f_df.copy()
    mask_iu = (
        (f_sat["location"] == "IND_urban") & (f_sat["year"] == 2040) & (f_sat["scenario"] == "bau")
    )
    f_sat.loc[mask_iu, "clean_fuel_share"] = 1.0
    ev_sat = Stage3Evaluator(
        config=cfg,
        contracts=Stage3Contracts(
            forecasts_stage3=f_sat,
            baseline_unit_state=base_contracts.baseline_unit_state.copy(),
            surrogate=copy.deepcopy(base_contracts.surrogate),
            intervention_params=params,
            is_mock=True,
            paths=base_contracts.paths,
        ),
        baseline_clean_fuel_households=hh_base,
    )
    assert ev_sat.effective_adoption_headroom()[ind_u] == pytest.approx(0.0)
    mdp_sat = build_layered_mdp(ev_sat, mdp_cfg)
    assert mdp_sat.step_u[ind_u] == pytest.approx(0.0)


def test_mdp_preclamp_high_uptake_vs_postclamp_realized() -> None:
    """B-14 audit: _apply_action_branch clips 1.25x overshoots before evaluator scoring."""
    _, ev = _make_tiny_1unit_evaluator(budget=200.0, lpg_init_ready=True)
    lpg_pos = int(np.flatnonzero(ev._non_infra_indices == ev._idx_lpg)[0])
    # 5 bins (max_bin = 4) and P1 cap = 1.1 * step5 -> cap_bins = 1, raw 1.25x = 2 steps
    mdp_cfg_5 = MDPConfig(
        coverage_bins=(0.0, 0.25, 0.5, 0.75, 1.0),
        budget_bins=3,
        discount_factor=0.774,
        cost_weight_lambda=1500.0,
        transition_probs=(0.15, 0.70, 0.15),
        n_rollouts=10,
        vi_tol=1e-6,
        vi_max_iter=200,
        lambda_sensitivity=(1500.0,),
        discount_sensitivity=(0.774,),
        seed=42,
    )
    step5 = float(ev.effective_adoption_headroom()[0]) / 4.0
    ev._non_infra_max_adoption[0, lpg_pos, 0] = step5 * 1.1
    mdp = build_layered_mdp(ev, mdp_cfg_5)

    s0 = mdp.layers[0][0]
    raw_attempted_steps = int(np.ceil(1.25 * 1))  # mult_mode == 2 for ACT_LPG_1 (steps=1) -> 2
    assert raw_attempted_steps * mdp.step_u[0] > ev._non_infra_max_adoption[0, lpg_pos, 0]

    s_post = _apply_action_branch(
        state=s0,
        combo=(ACT_LPG_1,),
        b_next=1,
        mult_mode=2,
        period_idx=0,
        n_units=ev.n_units,
        unready_units=mdp.unready_units,
        max_bin=mdp.max_bin,
        step_u=mdp.step_u,
        evaluator=ev,
    )
    # Post-clamp gain is repaired down to cap_bins (=1), keeping evaluator violation_total == 0.0
    assert s_post[0] == 1
    x_post = np.zeros(ev.n_vars, dtype=float)
    x_post[lpg_pos * ev.n_periods] = s_post[0] * mdp.step_u[0]
    res_post = ev.evaluate(x_post)
    assert res_post.is_feasible is True

    # Without clamping, raw attempted gain (=2 steps) triggers an adoption-cap violation
    x_pre = np.zeros(ev.n_vars, dtype=float)
    x_pre[lpg_pos * ev.n_periods] = raw_attempted_steps * mdp.step_u[0]
    res_pre = ev.evaluate(x_pre)
    assert res_pre.is_feasible is False
    assert res_pre.violation_adoption_cap > 0.0
