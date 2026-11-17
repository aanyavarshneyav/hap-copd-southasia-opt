"""Unit and integration tests for HTN and STRIPS/PDDL planners."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from src.data.contracts import load_config
from src.optimise.evaluator import Stage3Evaluator
from src.planning.htn import build_htn_plan, collect_htn_leaves, run_htn_pipeline, verify_htn_tree
from src.planning.pddl_runner import (
    PDDLVerificationError,
    parse_pddl_domain_and_problem,
    plan_to_decision_vector,
    replay_and_verify_pddl_plan,
    run_pddl_pipeline,
    solve_pddl,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mock.yaml"
DOMAIN_PATH = REPO_ROOT / "pddl" / "domain.pddl"
PROBLEM_PATH = REPO_ROOT / "pddl" / "problem_ind_npl.pddl"


def test_htn_tree_is_complete_and_every_leaf_is_valid_cell() -> None:
    evaluator = Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)
    tree, x_vec, ev_res = build_htn_plan(evaluator)
    leaves = collect_htn_leaves(tree)

    assert len(leaves) > 0
    country_names = [ch.name for ch in tree.children if ch.name.startswith("PlanCountryProgram")]
    assert "PlanCountryProgram(India)" in country_names
    assert "PlanCountryProgram(Nepal)" in country_names

    valid_units = set(evaluator.units)
    valid_intervs = set(evaluator.interventions)
    valid_periods = set(evaluator.periods)
    for leaf in leaves:
        assert leaf.unit in valid_units
        assert leaf.intervention in valid_intervs
        assert leaf.period in valid_periods
        assert 0.0 <= leaf.allocation_value <= 1.0

    verified_vec, verified_res = verify_htn_tree(tree, evaluator)
    assert np.array_equal(x_vec, verified_vec)
    assert verified_res.is_feasible is True
    assert ev_res.is_feasible is True
    assert float(ev_res.total_cost_usd_millions) <= evaluator.total_budget_usd_millions


def test_pddl_parses_solves_replays_and_passes_evaluator() -> None:
    strips = parse_pddl_domain_and_problem(DOMAIN_PATH, PROBLEM_PATH)
    assert strips.domain_name == "south-asia-clean-cooking"
    assert "build-lpg-infra" in strips.actions
    assert "rollout-lpg-after-infra" in strips.actions

    actions, solve_s = solve_pddl(
        DOMAIN_PATH, PROBLEM_PATH, search_name="astar", heuristic_name="hff"
    )
    assert len(actions) > 0
    assert solve_s < 5.0

    final_state = replay_and_verify_pddl_plan(DOMAIN_PATH, PROBLEM_PATH, actions)
    assert all(goal in final_state for goal in strips.goal_facts)

    evaluator = Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)
    x_vec = plan_to_decision_vector(actions, evaluator)
    res = evaluator.evaluate(x_vec)
    assert res.is_feasible is True
    assert res.violation_budget == 0.0
    assert res.violation_adoption_cap == 0.0
    assert res.violation_coverage_ceiling == 0.0
    assert res.violation_infra_prereq == 0.0


def test_pddl_replay_checker_rejects_invalid_sequences() -> None:
    # 1. Rolling out LPG in SAME period as building infra (violates strictly-before)
    same_period_seq = [
        "(build-lpg-infra ind_rural p1 b8 b7)",
        "(rollout-lpg-after-infra ind_rural p1 p1 c0 c1 b7 b6)",
    ]
    with pytest.raises(PDDLVerificationError, match="missing precondition"):
        replay_and_verify_pddl_plan(DOMAIN_PATH, PROBLEM_PATH, same_period_seq)

    # 2. Rolling out LPG in P2 without building infra first
    no_infra_seq = [
        "(advance-period p1 p2)",
        "(rollout-lpg-after-infra ind_rural p1 p2 c0 c1 b8 b7)",
    ]
    with pytest.raises(PDDLVerificationError, match="missing precondition"):
        replay_and_verify_pddl_plan(DOMAIN_PATH, PROBLEM_PATH, no_infra_seq)

    # 3. Double rollout in the same (unit, period) slot violates slot-open guard
    double_slot_seq = [
        "(rollout-lpg-init-ready npl_urban p1 c0 c1 b8 b7)",
        "(rollout-improved-biomass npl_urban p1 c1 c2 b7 b6)",
    ]
    with pytest.raises(PDDLVerificationError, match="missing precondition"):
        replay_and_verify_pddl_plan(DOMAIN_PATH, PROBLEM_PATH, double_slot_seq)

    # 4. Valid prefix that fails to reach goal is rejected
    incomplete_seq = ["(rollout-lpg-init-ready ind_urban p1 c0 c1 b8 b7)"]
    with pytest.raises(PDDLVerificationError, match="unsatisfied goal"):
        replay_and_verify_pddl_plan(DOMAIN_PATH, PROBLEM_PATH, incomplete_seq)


def test_htn_and_pddl_pipelines_write_mock_outputs(tmp_path: Path) -> None:
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

    cfg_path = tmp_path / "plan_test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    htn_out = run_htn_pipeline(cfg_path, base_dir=tmp_path, generate_figure=True)
    pddl_out = run_pddl_pipeline(cfg_path, base_dir=tmp_path)

    htn_json = htn_out["files"]["htn_trace_json"]
    assert htn_json.exists()
    assert "_mock" in htn_json.stem
    htn_payload = json.loads(htn_json.read_text(encoding="utf-8"))
    assert htn_payload["is_mock"] is True
    assert htn_payload["data_version"] == "MOCK_V1"
    assert htn_out["files"]["fig_htn"].exists()

    pddl_txt = pddl_out["files"]["pddl_solution_txt"]
    pddl_meta = pddl_out["files"]["pddl_solution_meta"]
    assert pddl_txt.exists()
    assert "_mock" in pddl_txt.stem
    assert pddl_meta.exists()
    meta_payload = json.loads(pddl_meta.read_text(encoding="utf-8"))
    assert meta_payload["is_mock"] is True
    assert meta_payload["replay_verified"] is True
    assert meta_payload["evaluator_feasible"] is True
