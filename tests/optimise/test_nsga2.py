"""Unit and integration tests for Stage 3 NSGA-II, baselines, and outputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from src.data.contracts import Stage3Contracts, load_config, load_stage3_contracts
from src.optimise.baselines import (
    compare_nsga2_vs_baselines,
    run_all_baselines,
    run_greedy_dalys_per_dollar,
    run_proportional_equal_split,
    run_random_search,
)
from src.optimise.evaluator import Stage3Evaluator
from src.optimise.nsga2 import (
    NSGA2Config,
    compute_reference_point,
    run_nsga2_multi_seed,
    run_nsga2_seed,
    select_representative_plans,
)
from src.optimise.run import artifact_path, run_stage3

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _tiny_test_config() -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    cfg["stage3"]["population_size"] = 12
    cfg["stage3"]["n_generations"] = 6
    cfg["stage3"]["seeds"] = [42, 43]
    cfg["stage3"]["greedy_step_size"] = 0.1
    return cfg


def _assert_non_dominated(F: np.ndarray) -> None:
    """Assert no row in ``F`` strictly dominates any other row (all minimised)."""
    n = len(F)
    for i in range(n):
        le = F <= F[i]
        lt = F < F[i]
        dominated_by_other = np.all(le, axis=1) & np.any(lt, axis=1)
        assert not np.any(dominated_by_other), f"Row {i} is dominated by another solution."


def test_seeded_run_reproduces_identical_front() -> None:
    cfg = _tiny_test_config()
    evaluator = Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)
    nsga2_cfg = NSGA2Config.from_config(cfg)
    ref_point = compute_reference_point(evaluator)

    res_a = run_nsga2_seed(evaluator, nsga2_cfg, seed=42, ref_point=ref_point)
    res_b = run_nsga2_seed(evaluator, nsga2_cfg, seed=42, ref_point=ref_point)

    assert np.array_equal(res_a.X, res_b.X)
    assert np.array_equal(res_a.F, res_b.F)
    assert res_a.hypervolume == pytest.approx(res_b.hypervolume)
    assert len(res_a.hv_history) == nsga2_cfg.n_generations


def test_returned_front_is_feasible_non_dominated_and_non_zero_daly_opt() -> None:
    cfg = _tiny_test_config()
    evaluator = Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)
    multi = run_nsga2_multi_seed(evaluator, cfg)

    ev = multi.combined_evaluation
    assert np.all(np.asarray(ev.is_feasible, dtype=bool))
    assert np.all(np.asarray(ev.violation_budget, dtype=float) == 0.0)
    assert np.all(np.asarray(ev.violation_adoption_cap, dtype=float) == 0.0)
    assert np.all(np.asarray(ev.violation_coverage_ceiling, dtype=float) == 0.0)
    assert np.all(np.asarray(ev.violation_infra_prereq, dtype=float) == 0.0)

    _assert_non_dominated(multi.combined_F)

    selected_df, _ = select_representative_plans(
        multi.combined_X, multi.combined_evaluation, evaluator
    )
    assert list(selected_df["plan_name"]) == [
        "knee",
        "cost_minimal",
        "daly_minimal",
        "equity_minimal",
    ]
    assert np.all(selected_df["dalys_2040_lower"] <= selected_df["dalys_2040_mean"])
    assert np.all(selected_df["dalys_2040_mean"] <= selected_df["dalys_2040_upper"])

    # Zero vector must never be reported as optimal for DALYs
    daly_row = selected_df[selected_df["plan_name"] == "daly_minimal"].iloc[0]
    daly_vec = daly_row[list(evaluator.variable_names)].to_numpy(dtype=float)
    assert not np.allclose(daly_vec, 0.0)
    assert float(daly_row["dalys_2040_mean"]) < float(multi.bau_evaluation.dalys_2040_mean)
    assert float(daly_row["total_cost_usd_millions"]) > 0.0


def test_baselines_are_feasible_and_comparable() -> None:
    cfg = _tiny_test_config()
    evaluator = Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)
    multi = run_nsga2_multi_seed(evaluator, cfg)
    ref_point = multi.ref_point

    prop = run_proportional_equal_split(evaluator, ref_point)
    greedy = run_greedy_dalys_per_dollar(evaluator, ref_point, step_size=0.1)
    rand = run_random_search(
        evaluator,
        ref_point,
        n_eval=multi.seed_results[0].n_eval,
        seed=42,
    )

    for b in (prop, greedy, rand):
        assert np.all(np.asarray(b.evaluation.is_feasible, dtype=bool))
        _assert_non_dominated(b.F)

    baselines = {"proportional": prop, "greedy": greedy, "random": rand}
    comp_df = compare_nsga2_vs_baselines(multi, baselines, evaluator)
    assert set(comp_df["method"]) == {
        "nsga2_combined_5seeds",
        "nsga2_seed_42",
        "baseline_proportional_split",
        "baseline_greedy_dalys_per_dollar",
        "baseline_random_search",
    }
    assert bool(comp_df["is_mock"].all()) is True


def test_run_stage3_writes_mock_flagged_outputs_and_sidecars(tmp_path: Path) -> None:
    cfg = _tiny_test_config()
    # Point contract inputs to repo mock files (absolute paths) and outputs to tmp_path
    for key in (
        "forecast_mock_file",
        "baseline_state_mock_file",
        "surrogate_mock_file",
        "intervention_params_mock_file",
    ):
        cfg["interfaces"][key] = str((REPO_ROOT / cfg["interfaces"][key]).resolve())

    cfg["paths"]["outputs_tables_dir"] = str(tmp_path / "tables")
    cfg["paths"]["outputs_figures_dir"] = str(tmp_path / "figures")

    tmp_cfg_path = tmp_path / "test_config.yaml"
    tmp_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = run_stage3(tmp_cfg_path, base_dir=tmp_path)
    assert result["is_mock"] is True

    files = result["files"]
    expected_csv_keys = ("pareto_csv", "selected_csv", "hv_csv", "comparison_csv")
    expected_fig_keys = ("fig_pareto", "fig_hv", "fig_alloc", "fig_baseline")

    for k in expected_csv_keys:
        csv_path: Path = files[k]
        meta_path: Path = Path(f"{csv_path}.meta.json")
        assert csv_path.exists()
        assert "_mock" in csv_path.stem
        assert meta_path.exists()

        df = pd.read_csv(csv_path)
        assert "is_mock" in df.columns
        assert bool(df["is_mock"].all()) is True
        assert "data_version" in df.columns
        assert set(df["data_version"].unique()) == {"MOCK_V1"}

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["is_mock"] is True
        assert meta["warning_mock_inputs"] is True
        assert meta["data_version"] == "MOCK_V1"

    pareto_df = pd.read_csv(files["pareto_csv"])
    evaluator = Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)
    for var_name in evaluator.variable_names:
        assert var_name in pareto_df.columns
    assert bool(pareto_df["is_feasible"].all()) is True

    for k in expected_fig_keys:
        fig_path: Path = files[k]
        meta_path = Path(f"{fig_path}.meta.json")
        assert fig_path.exists()
        assert fig_path.stat().st_size > 1024
        assert "_mock" in fig_path.stem
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["is_mock"] is True

    # Verify non-mock helper omits _mock suffix
    real_path = artifact_path(tmp_path, "s3_pareto_front", ".csv", is_mock=False)
    assert real_path.name == "s3_pareto_front.csv"

    # Exercise run_all_baselines wrapper with tiny config
    baselines = run_all_baselines(evaluator, cfg, result["nsga2_result"].ref_point)
    assert set(baselines.keys()) == {"proportional", "greedy", "random"}


def test_nsga2_and_repair_respect_population_saturation_across_bridge_factors() -> None:
    from src.optimise.nsga2 import repair_decision_vectors

    cfg = _tiny_test_config()
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

    for bf in (0.8, 1.0, 1.2):
        ev = Stage3Evaluator(
            config=cfg,
            contracts=contracts,
            baseline_clean_fuel_households=hh_base,
            coverage_bridge_factor={u: bf for u in hh_base},
        )
        # Over-saturated raw vector (<= hh ceiling gap, > pop headroom) must repair cleanly
        x_over = np.zeros(ev.n_vars)
        ind_u = ev.units.index("IND_urban")
        lpg_i = ev.interventions.index("lpg_subsidy")
        x_over[ind_u * ev.n_interventions * ev.n_periods + lpg_i * ev.n_periods] = 0.09
        x_rep = repair_decision_vectors(x_over, ev, repair_budget=True)
        res_rep = ev.evaluate(x_rep)
        assert res_rep.is_feasible is True
        assert res_rep.violation_coverage_ceiling == pytest.approx(0.0)
        ten = x_rep.reshape(ev.n_units, ev.n_interventions, ev.n_periods)
        raw_gain = np.sum(ten[:, ev._non_infra_indices, :], axis=(1, 2))
        assert np.all(raw_gain <= ev.effective_adoption_headroom() + 1e-12)
        assert np.all(ev._baseline_clean_fuel_population + bf * raw_gain <= 1.0 + 1e-12)

        # Full NSGA-II front obeys joint headroom across all 4 units
        multi = run_nsga2_multi_seed(ev, cfg)
        assert np.all(np.asarray(multi.combined_evaluation.is_feasible, dtype=bool))
        ten_all = multi.combined_X.reshape(-1, ev.n_units, ev.n_interventions, ev.n_periods)
        raw_all = np.sum(ten_all[:, :, ev._non_infra_indices, :], axis=(2, 3))
        assert np.all(
            ev._baseline_clean_fuel_population[np.newaxis, :] + bf * raw_all <= 1.0 + 1e-9
        )
        assert np.all(raw_all <= ev.effective_adoption_headroom()[np.newaxis, :] + 1e-9)
