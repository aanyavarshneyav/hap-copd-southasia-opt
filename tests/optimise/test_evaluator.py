"""Unit tests for the vectorized Stage 3 evaluator (``src/optimise/evaluator.py``)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from src.data.contracts import Stage3Contracts, load_config, load_stage3_contracts
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

    # 3. Building infra in SAME period (P1) does NOT unlock P1 LPG -> infeasible (q < p)
    same_period = no_infra.copy()
    same_period[_idx(evaluator, "NPL_rural", "infra_lpg_grid", "P1")] = 0.50
    res_same = evaluator.evaluate(same_period)
    assert res_same.violation_infra_prereq == pytest.approx(0.10)
    assert res_same.is_feasible is False

    # 4. Infra built in STRICTLY EARLIER period (P1, x >= 0.5) unlocks P2 LPG & electric
    earlier_period = np.zeros(evaluator.n_vars)
    earlier_period[_idx(evaluator, "NPL_rural", "infra_lpg_grid", "P1")] = 0.50
    earlier_period[_idx(evaluator, "NPL_rural", "lpg_subsidy", "P2")] = 0.10
    earlier_period[_idx(evaluator, "NPL_rural", "electric_induction", "P2")] = 0.10
    res_earlier = evaluator.evaluate(earlier_period)
    assert res_earlier.violation_infra_prereq == pytest.approx(0.0)
    assert res_earlier.is_feasible is True

    # 5. Below-threshold infra (0.49 < 0.50) in P1 does NOT unlock P2 and pays zero infra cost
    below_thresh = np.zeros(evaluator.n_vars)
    below_thresh[_idx(evaluator, "NPL_rural", "lpg_subsidy", "P2")] = 0.10
    res_no_p2 = evaluator.evaluate(below_thresh)
    below_thresh[_idx(evaluator, "NPL_rural", "infra_lpg_grid", "P1")] = 0.49
    res_below = evaluator.evaluate(below_thresh)
    assert res_below.violation_infra_prereq == pytest.approx(0.10)
    assert res_below.total_cost_usd_millions == pytest.approx(res_no_p2.total_cost_usd_millions)


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


def test_separate_household_baseline_and_bridge_factor() -> None:
    config = load_config(CONFIG_PATH)
    contracts = load_stage3_contracts(config, base_dir=REPO_ROOT)

    hh_base = {"IND_urban": 0.90, "IND_rural": 0.45, "NPL_urban": 0.58, "NPL_rural": 0.21}
    eval_08 = Stage3Evaluator(
        config=config,
        contracts=contracts,
        baseline_clean_fuel_households=hh_base,
        coverage_bridge_factor={u: 0.8 for u in hh_base},
    )
    eval_10 = Stage3Evaluator(
        config=config,
        contracts=contracts,
        baseline_clean_fuel_households=hh_base,
        coverage_bridge_factor={u: 1.0 for u in hh_base},
    )
    eval_12 = Stage3Evaluator(
        config=config,
        contracts=contracts,
        baseline_clean_fuel_households=hh_base,
        coverage_bridge_factor={u: 1.2 for u in hh_base},
    )

    x = np.zeros(eval_10.n_vars)
    x[_idx(eval_10, "IND_urban", "lpg_subsidy", "P1")] = 0.09  # 0.90 + 0.09 = 0.99 > 0.98 ceiling
    r_08 = eval_08.evaluate(x)
    r_10 = eval_10.evaluate(x)
    r_12 = eval_12.evaluate(x)

    # Household ceiling uses hh_base (0.90 + 0.09 - 0.98 = 0.01), independent of bridge factor
    assert r_10.violation_coverage_ceiling == pytest.approx(0.01)
    assert r_08.violation_coverage_ceiling == pytest.approx(0.01)
    # Cost semantics remain unchanged across bridge factors
    assert r_08.total_cost_usd_millions == pytest.approx(r_10.total_cost_usd_millions)
    assert r_12.total_cost_usd_millions == pytest.approx(r_10.total_cost_usd_millions)
    # Higher bridge factor yields greater population exposure shift -> lower 2040 DALYs
    assert r_12.dalys_2040_mean < r_10.dalys_2040_mean < r_08.dalys_2040_mean


def _make_ind_urban_saturation_evaluator(
    bridge_factor: float = 1.0,
) -> Stage3Evaluator:
    config = load_config(CONFIG_PATH)
    base_contracts = load_stage3_contracts(config, base_dir=REPO_ROOT)
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
    return Stage3Evaluator(
        config=config,
        contracts=contracts,
        baseline_clean_fuel_households=hh_base,
        coverage_bridge_factor={u: bridge_factor for u in hh_base},
    )


def test_population_saturation_enforces_joint_headroom_and_positive_marginal_benefit() -> None:
    ev_10 = _make_ind_urban_saturation_evaluator(bridge_factor=1.0)
    idx_ind_u = ev_10.units.index("IND_urban")
    # IND_urban: hh headroom = 0.99 - 0.897 = 0.093; pop headroom = (1.0 - 0.96) / 1.0 = 0.04
    assert ev_10.effective_adoption_headroom()[idx_ind_u] == pytest.approx(0.04)

    # At exact population saturation limit (0.04), allocation is feasible and gains DALYs
    x_at_limit = np.zeros(ev_10.n_vars)
    x_at_limit[_idx(ev_10, "IND_urban", "lpg_subsidy", "P1")] = 0.04
    x_below = np.zeros(ev_10.n_vars)
    x_below[_idx(ev_10, "IND_urban", "lpg_subsidy", "P1")] = 0.03
    res_lim = ev_10.evaluate(x_at_limit)
    res_below = ev_10.evaluate(x_below)
    assert res_lim.is_feasible is True
    assert res_lim.violation_coverage_ceiling == pytest.approx(0.0)
    assert res_lim.dalys_2040_mean < res_below.dalys_2040_mean

    # Above pop headroom (0.06 > 0.04) even below hh ceiling (0.06 <= 0.093) MUST be infeasible
    x_over_pop = np.zeros(ev_10.n_vars)
    x_over_pop[_idx(ev_10, "IND_urban", "lpg_subsidy", "P1")] = 0.06
    res_over = ev_10.evaluate(x_over_pop)
    assert res_over.is_feasible is False
    assert res_over.violation_coverage_ceiling == pytest.approx(0.02)

    # Across bridge factors 0.8, 1.0, 1.2, max feasible adoption keeps implied pop <= 1.0
    # and infrastructure prerequisite q < p in NPL_rural remains unchanged.
    expected_ind_u_limits = {0.8: 0.05, 1.0: 0.04, 1.2: 0.04 / 1.2}
    for bf, expected_lim in expected_ind_u_limits.items():
        ev_bf = _make_ind_urban_saturation_evaluator(bridge_factor=bf)
        headroom = ev_bf.effective_adoption_headroom()
        assert headroom[idx_ind_u] == pytest.approx(expected_lim)
        pop_base = ev_bf._baseline_clean_fuel_population
        assert np.all(pop_base + bf * headroom <= 1.0 + 1e-12)

        x_just_over = np.zeros(ev_bf.n_vars)
        x_just_over[_idx(ev_bf, "IND_urban", "lpg_subsidy", "P1")] = expected_lim + 0.005
        r_over = ev_bf.evaluate(x_just_over)
        assert r_over.is_feasible is False
        assert r_over.violation_coverage_ceiling == pytest.approx(0.005)

        x_npl_no_infra = np.zeros(ev_bf.n_vars)
        x_npl_no_infra[_idx(ev_bf, "NPL_rural", "lpg_subsidy", "P1")] = 0.05
        assert ev_bf.evaluate(x_npl_no_infra).violation_infra_prereq == pytest.approx(0.05)
