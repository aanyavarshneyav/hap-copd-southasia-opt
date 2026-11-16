"""Unit tests for the vectorized Stage 3 evaluator (``src/optimise/evaluator.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.data.contracts import load_config, load_stage3_contracts
from src.optimise.evaluator import (
    Stage3Evaluator,
    tensor_to_vector,
    vector_to_tensor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture()
def evaluator() -> Stage3Evaluator:
    return Stage3Evaluator.from_config(CONFIG_PATH, base_dir=REPO_ROOT)


def _idx(evaluator: Stage3Evaluator, unit: str, intervention: str, period: str) -> int:
    u = evaluator.units.index(unit)
    i = evaluator.interventions.index(intervention)
    p = evaluator.periods.index(period)
    return u * (evaluator.n_interventions * evaluator.n_periods) + i * evaluator.n_periods + p


def test_vector_tensor_round_trip(evaluator: Stage3Evaluator) -> None:
    rng = np.random.default_rng(42)
    vec_1d = rng.uniform(0.0, 0.2, size=(evaluator.n_vars,))
    ten_3d = vector_to_tensor(
        vec_1d, evaluator.n_units, evaluator.n_interventions, evaluator.n_periods
    )
    assert ten_3d.shape == (4, 4, 3)
    assert np.array_equal(tensor_to_vector(ten_3d), vec_1d)

    # Check exact position mapping for a specific entry
    probe = np.zeros(evaluator.n_vars)
    probe[_idx(evaluator, "NPL_rural", "electric_induction", "P2")] = 0.123
    probe_t = vector_to_tensor(probe, 4, 4, 3)
    assert probe_t[3, 2, 1] == pytest.approx(0.123)

    batch_2d = rng.uniform(0.0, 0.2, size=(15, evaluator.n_vars))
    ten_4d = vector_to_tensor(batch_2d, 4, 4, 3)
    assert ten_4d.shape == (15, 4, 4, 3)
    assert np.array_equal(tensor_to_vector(ten_4d), batch_2d)


def test_zero_investment_returns_baseline_dalys(evaluator: Stage3Evaluator) -> None:
    config = load_config(CONFIG_PATH)
    bundle = load_stage3_contracts(config, base_dir=REPO_ROOT)
    bau_2040 = bundle.forecasts_stage3[
        (bundle.forecasts_stage3["year"] == 2040)
        & (bundle.forecasts_stage3["scenario"] == "bau")
        & (bundle.forecasts_stage3["measure"] == "DALYs")
        & (bundle.forecasts_stage3["metric"] == "Number")
        & (bundle.forecasts_stage3["sex"].isin(["female", "male"]))
    ]
    expected_mean = float(bau_2040["val"].sum())
    expected_low = float(bau_2040["lower"].sum())
    expected_up = float(bau_2040["upper"].sum())

    res = evaluator.evaluate(np.zeros(evaluator.n_vars))
    assert res.is_mock is True
    assert res.warning_mock_inputs is True
    assert res.dalys_2040_mean == pytest.approx(expected_mean)
    assert res.dalys_2040_lower == pytest.approx(expected_low)
    assert res.dalys_2040_upper == pytest.approx(expected_up)
    assert res.total_cost_usd_millions == pytest.approx(0.0)
    assert res.violation_budget == pytest.approx(0.0)
    assert res.violation_adoption_cap == pytest.approx(0.0)
    assert res.violation_coverage_ceiling == pytest.approx(0.0)
    assert res.violation_infra_prereq == pytest.approx(0.0)
    assert res.is_feasible is True

    as_dict = res.to_dict()
    assert as_dict["is_mock"] is True
    assert as_dict["warning_mock_inputs"] is True


def test_monotonicity_spending_more_never_increases_dalys(evaluator: Stage3Evaluator) -> None:
    rng = np.random.default_rng(7)
    base = rng.uniform(0.0, 0.1, size=(20, evaluator.n_vars))
    delta = rng.uniform(0.0, 0.1, size=(20, evaluator.n_vars))
    higher = np.clip(base + delta, 0.0, 1.0)

    res_base = evaluator.evaluate(base)
    res_higher = evaluator.evaluate(higher)
    assert np.all(res_higher.dalys_2040_mean <= res_base.dalys_2040_mean + 1e-12)
    assert np.all(res_higher.dalys_2040_lower <= res_higher.dalys_2040_mean)
    assert np.all(res_higher.dalys_2040_mean <= res_higher.dalys_2040_upper)


def test_budget_adoption_and_ceiling_violations(evaluator: Stage3Evaluator) -> None:
    # Budget violation: massive rollout in IND_rural (costly) + infra
    over_budget = np.zeros(evaluator.n_vars)
    over_budget[_idx(evaluator, "IND_rural", "infra_lpg_grid", "P1")] = 1.0
    for p in ("P1", "P2", "P3"):
        over_budget[_idx(evaluator, "IND_rural", "lpg_subsidy", p)] = 0.20
        over_budget[_idx(evaluator, "IND_rural", "electric_induction", p)] = 0.15
    res_budget = evaluator.evaluate(over_budget)
    assert res_budget.violation_budget > 0.0
    assert res_budget.is_feasible is False

    # Per-period adoption cap violation (NPL_urban lpg cap is 0.20; set 0.35)
    over_cap = np.zeros(evaluator.n_vars)
    over_cap[_idx(evaluator, "NPL_urban", "lpg_subsidy", "P1")] = 0.35
    res_cap = evaluator.evaluate(over_cap)
    assert res_cap.violation_adoption_cap == pytest.approx(0.15)
    assert res_cap.is_feasible is False

    # Coverage ceiling violation (IND_urban BAU 2040 is 0.88, ceiling is 0.98; add 0.20)
    over_ceil = np.zeros(evaluator.n_vars)
    over_ceil[_idx(evaluator, "IND_urban", "lpg_subsidy", "P1")] = 0.10
    over_ceil[_idx(evaluator, "IND_urban", "electric_induction", "P2")] = 0.10
    res_ceil = evaluator.evaluate(over_ceil)
    assert res_ceil.violation_coverage_ceiling == pytest.approx(0.10)
    assert res_ceil.is_feasible is False


def test_infrastructure_prerequisite_mirrors_pddl(evaluator: Stage3Evaluator) -> None:
    # NPL_rural has lpg_infra_ready_init=False, grid_ready_init=False
    # 1. Rolling out LPG without infra -> violation
    no_infra = np.zeros(evaluator.n_vars)
    no_infra[_idx(evaluator, "NPL_rural", "lpg_subsidy", "P1")] = 0.10
    res_no = evaluator.evaluate(no_infra)
    assert res_no.violation_infra_prereq == pytest.approx(0.10)
    assert res_no.is_feasible is False

    # 2. Building infra strictly later (P2) than rollout (P1) -> still violates!
    late_infra = no_infra.copy()
    late_infra[_idx(evaluator, "NPL_rural", "infra_lpg_grid", "P2")] = 0.60
    res_late = evaluator.evaluate(late_infra)
    assert res_late.violation_infra_prereq == pytest.approx(0.10)
    assert res_late.is_feasible is False

    # 3. Building infra in SAME period (P1, x >= 0.5) unlocks P1 LPG & P2 electric -> feasible
    same_period = no_infra.copy()
    same_period[_idx(evaluator, "NPL_rural", "infra_lpg_grid", "P1")] = 0.50
    same_period[_idx(evaluator, "NPL_rural", "electric_induction", "P2")] = 0.10
    res_same = evaluator.evaluate(same_period)
    assert res_same.violation_infra_prereq == pytest.approx(0.0)
    assert res_same.is_feasible is True

    # 4. Below-threshold infra (0.49 < 0.50) does NOT build infra and pays zero infra cost
    below_thresh = no_infra.copy()
    below_thresh[_idx(evaluator, "NPL_rural", "infra_lpg_grid", "P1")] = 0.49
    res_below = evaluator.evaluate(below_thresh)
    assert res_below.violation_infra_prereq == pytest.approx(0.10)
    assert res_below.total_cost_usd_millions == pytest.approx(res_no.total_cost_usd_millions)


def test_vectorised_batch_matches_single_evaluations(evaluator: Stage3Evaluator) -> None:
    rng = np.random.default_rng(123)
    pop = rng.uniform(0.0, 0.25, size=(120, evaluator.n_vars))
    batch_res = evaluator.evaluate(pop)

    for idx in (0, 17, 63, 119):
        single_res = evaluator.evaluate(pop[idx])
        assert batch_res.dalys_2040_mean[idx] == pytest.approx(single_res.dalys_2040_mean)
        assert batch_res.dalys_2040_lower[idx] == pytest.approx(single_res.dalys_2040_lower)
        assert batch_res.dalys_2040_upper[idx] == pytest.approx(single_res.dalys_2040_upper)
        assert batch_res.total_cost_usd_millions[idx] == pytest.approx(
            single_res.total_cost_usd_millions
        )
        assert batch_res.equity_gap_asr[idx] == pytest.approx(single_res.equity_gap_asr)
        assert batch_res.violation_budget[idx] == pytest.approx(single_res.violation_budget)
        assert batch_res.violation_adoption_cap[idx] == pytest.approx(
            single_res.violation_adoption_cap
        )
        assert batch_res.violation_coverage_ceiling[idx] == pytest.approx(
            single_res.violation_coverage_ceiling
        )
        assert batch_res.violation_infra_prereq[idx] == pytest.approx(
            single_res.violation_infra_prereq
        )
        assert bool(batch_res.is_feasible[idx]) == single_res.is_feasible


def test_optional_cost_discounting_flag() -> None:
    config = load_config(CONFIG_PATH)
    contracts = load_stage3_contracts(config, base_dir=REPO_ROOT)

    eval_undisc = Stage3Evaluator(config=config, contracts=contracts)
    contracts.intervention_params["apply_cost_discounting"] = True
    eval_disc = Stage3Evaluator(config=config, contracts=contracts)

    x = np.zeros(eval_undisc.n_vars)
    x[_idx(eval_undisc, "IND_urban", "lpg_subsidy", "P3")] = 0.05
    c_undisc = eval_undisc.evaluate(x).total_cost_usd_millions
    c_disc = eval_disc.evaluate(x).total_cost_usd_millions
    assert c_disc == pytest.approx(c_undisc * (eval_disc.discount_factor**2))
