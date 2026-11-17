"""Unit and integration tests for equal-footing GA vs. MDP comparison."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from src.data.contracts import load_config
from src.optimise.evaluator import Stage3Evaluator
from src.planning.compare import (
    draw_shock_branches,
    run_comparison_pipeline,
    simulate_open_loop_schedule,
)
from src.planning.mdp import (
    MDPConfig,
    build_layered_mdp,
    simulate_mdp_rollouts,
    solve_finite_horizon_backward_induction,
)
from src.planning.pddl_runner import plan_to_decision_vector, solve_pddl

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mock.yaml"


def _tiny_compare_config(tmp_path: Path) -> Path:
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
    cfg["paths"]["pddl_dir"] = str((REPO_ROOT / "pddl").resolve())

    # Copy existing Stage 3 mock CSVs if available or use tiny NSGA-II settings
    cfg["stage3"]["population_size"] = 12
    cfg["stage3"]["n_generations"] = 5
    cfg["stage3"]["seeds"] = [42]
    cfg["planning"]["compare"]["n_rollouts"] = 40
    cfg["planning"]["compare"]["n_pareto_sample"] = 2
    cfg["planning"]["compare"]["budget_bins_sensitivity"] = [6]
    cfg["planning"]["compare"]["lambda_eq_sensitivity"] = [0.0, 25000.0]
    cfg["planning"]["mdp"]["discount_sensitivity"] = [0.774]

    cfg_path = tmp_path / "compare_test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_zero_stochasticity_simulator_matches_stage3_evaluator() -> None:
    evaluator = Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)
    actions, _ = solve_pddl(
        REPO_ROOT / "pddl" / "domain.pddl",
        REPO_ROOT / "pddl" / "problem_ind_npl.pddl",
    )
    x_pddl = plan_to_decision_vector(actions, evaluator)
    direct = evaluator.evaluate(x_pddl)

    # Zero stochasticity: all periods realize nominal multiplier 1.0 (branch == 1)
    zero_shock_branches = np.ones((5, evaluator.n_periods), dtype=int)
    sim_df = simulate_open_loop_schedule(
        x_pddl,
        evaluator,
        zero_shock_branches,
        cost_weight_lambda=3500.0,
        discount_factor=1.0,
    )

    assert np.allclose(sim_df["dalys_2040_mean"].to_numpy(), float(direct.dalys_2040_mean))
    assert np.allclose(sim_df["dalys_2040_lower"].to_numpy(), float(direct.dalys_2040_lower))
    assert np.allclose(sim_df["dalys_2040_upper"].to_numpy(), float(direct.dalys_2040_upper))
    assert np.allclose(
        sim_df["total_cost_usd_millions"].to_numpy(), float(direct.total_cost_usd_millions)
    )
    assert np.allclose(sim_df["equity_gap_asr"].to_numpy(), float(direct.equity_gap_asr))
    assert bool(sim_df["is_feasible_in_evaluator"].all()) is True

    bau = evaluator.evaluate(np.zeros(evaluator.n_vars, dtype=float))
    expected_scalarized = (
        float(bau.dalys_2040_mean) - float(direct.dalys_2040_mean)
    ) - 3500.0 * float(direct.total_cost_usd_millions)
    assert float(sim_df["discounted_return"].iloc[0]) == pytest.approx(
        expected_scalarized, rel=1e-9
    )


def test_mdp_simulated_mean_matches_solver_expected_value() -> None:
    cfg = load_config(CONFIG_PATH)
    evaluator = Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)
    mdp_cfg = MDPConfig.from_config(cfg)
    mdp = build_layered_mdp(evaluator, mdp_cfg)
    sol = solve_finite_horizon_backward_induction(
        mdp,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )

    n_rollouts = 1500
    branches = draw_shock_branches(mdp_cfg.transition_probs, n_rollouts, evaluator.n_periods, 42)
    roll_df = simulate_mdp_rollouts(
        mdp,
        sol,
        evaluator,
        n_rollouts=n_rollouts,
        seed=42,
        precomputed_branches=branches,
    )
    sim_mean = float(roll_df["discounted_return"].mean())
    sim_se = float(roll_df["discounted_return"].std(ddof=1)) / np.sqrt(n_rollouts)
    assert abs(sim_mean - sol.expected_value_s0) <= 3.5 * sim_se


def test_comparison_pipeline_reproducible_and_propagates_mock_flags(tmp_path: Path) -> None:
    cfg_path = _tiny_compare_config(tmp_path)
    out_a = run_comparison_pipeline(cfg_path, base_dir=tmp_path, generate_figures=True)
    out_b = run_comparison_pipeline(cfg_path, base_dir=tmp_path, generate_figures=False)

    pd.testing.assert_frame_equal(out_a["comp_df"], out_b["comp_df"])
    timing_cols = ["build_seconds", "solve_seconds"]
    pd.testing.assert_frame_equal(
        out_a["sens_df"].drop(columns=timing_cols),
        out_b["sens_df"].drop(columns=timing_cols),
    )

    files = out_a["files"]
    for csv_key in ("comp_csv", "sens_csv"):
        csv_path: Path = files[csv_key]
        meta_path = Path(f"{csv_path}.meta.json")
        assert csv_path.exists()
        assert "_mock" in csv_path.stem
        assert meta_path.exists()

        df = pd.read_csv(csv_path)
        assert bool(df["is_mock"].all()) is True
        assert bool(df["warning_mock_inputs"].all()) is True
        assert set(df["data_version"].unique()) == {"MOCK_V1"}

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["is_mock"] is True
        assert meta["warning_mock_inputs"] is True
        assert meta["data_version"] == "MOCK_V1"

    for fig_key in ("fig_dist", "fig_trade", "fig_sens"):
        fig_path: Path = files[fig_key]
        meta_path = Path(f"{fig_path}.meta.json")
        assert fig_path.exists()
        assert fig_path.stat().st_size > 1024
        assert "_mock" in fig_path.stem
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["is_mock"] is True
