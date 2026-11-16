"""Plain-Python Hierarchical Task Network (HTN) planner for Stage 4.

Implements total-order forward task decomposition following ``docs/PLANNING_SPEC.md`` §2:
``AchieveSouthAsiaCleanCookingPlan -> PlanCountryProgram -> PlanUnitProgram -> Primitive Operators``
plus ``VerifyBudgetAndEquityGuard``, mapping every primitive leaf to a valid
``(unit, intervention, period)`` cell in :class:`Stage3Evaluator`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.data.contracts import load_config, load_stage3_contracts
from src.optimise.evaluator import EvaluationResult, Stage3Evaluator

_FLOAT_EPS: float = 1e-12


@dataclass
class HTNNode:
    """Node in the hierarchical task decomposition tree."""

    name: str
    node_type: str  # "compound" | "primitive"
    method: str | None = None
    unit: str | None = None
    intervention: str | None = None
    period: str | None = None
    allocation_value: float = 0.0
    cost_usd_millions: float = 0.0
    children: list[HTNNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node and subtree to a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "name": self.name,
            "node_type": self.node_type,
        }
        if self.method is not None:
            payload["method"] = self.method
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.intervention is not None:
            payload["intervention"] = self.intervention
        if self.period is not None:
            payload["period"] = self.period
        if self.node_type == "primitive":
            payload["allocation_value"] = round(self.allocation_value, 6)
            payload["cost_usd_millions"] = round(self.cost_usd_millions, 4)
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload


@dataclass
class _PlanningState:
    """Mutable planning world state maintained during forward HTN decomposition."""

    remaining_budget: float
    remaining_headroom: dict[str, float]
    lpg_ready_from_period: dict[str, int]
    grid_ready_from_period: dict[str, int]
    allocated: dict[tuple[str, str, str], float] = field(default_factory=dict)


def _country_of(unit: str) -> str:
    if unit.startswith("IND_"):
        return "India"
    if unit.startswith("NPL_"):
        return "Nepal"
    return unit.split("_", maxsplit=1)[0]


def _pick_preferred_clean_intervention(
    u_idx: int,
    evaluator: Stage3Evaluator,
) -> str:
    """Choose cheaper of lpg_subsidy and electric_induction (equal effectiveness)."""
    c_lpg = float(evaluator._unit_costs[u_idx, evaluator._idx_lpg])
    c_elec = float(evaluator._unit_costs[u_idx, evaluator._idx_elec])
    return "lpg_subsidy" if c_lpg <= c_elec else "electric_induction"


def _decompose_unit_period(
    unit: str,
    u_idx: int,
    period: str,
    p_idx: int,
    state: _PlanningState,
    evaluator: Stage3Evaluator,
) -> HTNNode:
    """Apply HTN methods M1-M4 for ``PlanUnitProgram(unit, period)``."""
    disc = float(evaluator._period_discount[p_idx])
    headroom_rem = state.remaining_headroom[unit]
    task_node = HTNNode(name=f"PlanUnitProgram({unit}, {period})", node_type="compound")

    if headroom_rem <= 1e-6 or state.remaining_budget <= 1e-6:
        task_node.method = "M4_NoOp_Exhausted"
        return task_node

    clean_interv = _pick_preferred_clean_intervention(u_idx, evaluator)
    clean_i_idx = evaluator.interventions.index(clean_interv)
    clean_ready_from = (
        state.lpg_ready_from_period[unit]
        if clean_interv == "lpg_subsidy"
        else state.grid_ready_from_period[unit]
    )
    is_clean_ready_now = p_idx >= clean_ready_from

    infra_cost = float(evaluator._unit_costs[u_idx, evaluator._idx_infra]) * disc
    clean_unit_cost = float(evaluator._unit_costs[u_idx, clean_i_idx]) * disc
    clean_cap = float(evaluator._max_adoption[u_idx, clean_i_idx, p_idx])

    bio_i_idx = evaluator._idx_bio
    bio_unit_cost = float(evaluator._unit_costs[u_idx, bio_i_idx]) * disc
    bio_cap = float(evaluator._max_adoption[u_idx, bio_i_idx, p_idx])

    # Method M2: Rural / unready unit in P1/P2 -> build infra for strictly later periods
    if not is_clean_ready_now and p_idx < evaluator.n_periods - 1:
        # Check if budget covers infra + at least modest clean rollout later
        min_clean_reserve = 0.05 * clean_unit_cost
        if state.remaining_budget >= infra_cost + min_clean_reserve + _FLOAT_EPS:
            task_node.method = "M2_BuildInfraPrerequisite"
            prereq_task = HTNNode(
                name=f"BuildPrerequisites({unit}, {period})",
                node_type="compound",
                method="M2a_BuildLpgGridInfra",
            )
            op_infra = HTNNode(
                name=f"!op_build_lpg_grid_infra({unit}, {period})",
                node_type="primitive",
                unit=unit,
                intervention="infra_lpg_grid",
                period=period,
                allocation_value=evaluator.infra_build_threshold,
                cost_usd_millions=infra_cost,
            )
            prereq_task.children.append(op_infra)
            task_node.children.append(prereq_task)

            state.remaining_budget -= infra_cost
            state.lpg_ready_from_period[unit] = min(state.lpg_ready_from_period[unit], p_idx + 1)
            state.grid_ready_from_period[unit] = min(state.grid_ready_from_period[unit], p_idx + 1)
            state.allocated[(unit, "infra_lpg_grid", period)] = evaluator.infra_build_threshold
            return task_node

    # Method M1: Ready unit -> deploy clean cooking (lpg_subsidy or electric_induction)
    if is_clean_ready_now and clean_cap > 0.0:
        max_affordable = (
            (state.remaining_budget - _FLOAT_EPS) / clean_unit_cost
            if clean_unit_cost > 0.0
            else clean_cap
        )
        delta = min(headroom_rem, clean_cap, max_affordable)
        if delta > 1e-4:
            cost = delta * clean_unit_cost
            task_node.method = "M1_DeployCleanCooking"
            scale_task = HTNNode(
                name=f"ScaleCleanFuel({unit}, {period})",
                node_type="compound",
                method=f"M1a_{clean_interv}",
            )
            op_clean = HTNNode(
                name=f"!op_deploy_{clean_interv}({unit}, {period})",
                node_type="primitive",
                unit=unit,
                intervention=clean_interv,
                period=period,
                allocation_value=delta,
                cost_usd_millions=cost,
            )
            scale_task.children.append(op_clean)
            task_node.children.append(scale_task)

            state.remaining_budget -= cost
            state.remaining_headroom[unit] -= delta
            state.allocated[(unit, clean_interv, period)] = delta
            return task_node

    # Method M3: Transitional / budget-constrained improved biomass bridge
    if bio_cap > 0.0:
        max_affordable_bio = (
            (state.remaining_budget - _FLOAT_EPS) / bio_unit_cost
            if bio_unit_cost > 0.0
            else bio_cap
        )
        # In early period before infra, cap interim biomass so clean fuel still has headroom later
        interim_cap = (
            min(bio_cap, 0.5 * headroom_rem)
            if (not is_clean_ready_now and p_idx < evaluator.n_periods - 1)
            else bio_cap
        )
        delta_bio = min(headroom_rem, interim_cap, max_affordable_bio)
        if delta_bio > 1e-4:
            cost_bio = delta_bio * bio_unit_cost
            task_node.method = "M3_InterimImprovedBiomass"
            op_bio = HTNNode(
                name=f"!op_deploy_improved_biomass({unit}, {period})",
                node_type="primitive",
                unit=unit,
                intervention="improved_biomass",
                period=period,
                allocation_value=delta_bio,
                cost_usd_millions=cost_bio,
            )
            task_node.children.append(op_bio)

            state.remaining_budget -= cost_bio
            state.remaining_headroom[unit] -= delta_bio
            state.allocated[(unit, "improved_biomass", period)] = delta_bio
            return task_node

    task_node.method = "M4_NoOp_BudgetConstrained"
    return task_node


def build_htn_plan(evaluator: Stage3Evaluator) -> tuple[HTNNode, np.ndarray, EvaluationResult]:
    """Decompose the Stage 3 goal into an executable HTN tree and verify feasibility."""
    state = _PlanningState(
        remaining_budget=evaluator.total_budget_usd_millions - _FLOAT_EPS,
        remaining_headroom={
            u: max(
                0.0,
                float(evaluator._coverage_ceiling[idx] - evaluator._cf_bau_2040[idx]) - _FLOAT_EPS,
            )
            for idx, u in enumerate(evaluator.units)
        },
        lpg_ready_from_period={
            u: (0 if evaluator._lpg_infra_init[idx] >= 1.0 else evaluator.n_periods)
            for idx, u in enumerate(evaluator.units)
        },
        grid_ready_from_period={
            u: (0 if evaluator._grid_ready_init[idx] >= 1.0 else evaluator.n_periods)
            for idx, u in enumerate(evaluator.units)
        },
    )

    root = HTNNode(
        name="AchieveSouthAsiaCleanCookingPlan(2026-2040)",
        node_type="compound",
        method="TotalOrderForwardDecomposition",
    )

    countries: list[str] = list(dict.fromkeys(_country_of(u) for u in evaluator.units))
    # Prioritize rural units before urban units inside each country, and P1 -> P2 -> P3
    # so infra prerequisites unlock early clean cooking and budget isn't exhausted trivially.
    for country in countries:
        c_node = HTNNode(
            name=f"PlanCountryProgram({country})",
            node_type="compound",
            method="DecomposeUnitsByPeriod",
        )
        country_units = [
            (idx, u) for idx, u in enumerate(evaluator.units) if _country_of(u) == country
        ]
        # Rural first (high burden & infra prereqs), then urban
        country_units.sort(key=lambda pair: (0 if "rural" in pair[1] else 1, pair[0]))
        for p_idx, period in enumerate(evaluator.periods):
            for u_idx, unit in country_units:
                unit_node = _decompose_unit_period(unit, u_idx, period, p_idx, state, evaluator)
                c_node.children.append(unit_node)
        root.children.append(c_node)

    x_vec, ev_res = verify_htn_tree(root, evaluator)
    guard_node = HTNNode(
        name=(
            f"VerifyBudgetAndEquityGuard(feasible={ev_res.is_feasible}, "
            f"cost=${float(ev_res.total_cost_usd_millions):.1f}M)"
        ),
        node_type="compound",
        method="EvaluatorGuardCheck",
    )
    root.children.append(guard_node)
    return root, x_vec, ev_res


def collect_htn_leaves(node: HTNNode) -> list[HTNNode]:
    """Collect all primitive leaf action nodes in left-to-right execution order."""
    if node.node_type == "primitive":
        return [node]
    leaves: list[HTNNode] = []
    for child in node.children:
        leaves.extend(collect_htn_leaves(child))
    return leaves


def verify_htn_tree(
    root: HTNNode,
    evaluator: Stage3Evaluator,
) -> tuple[np.ndarray, EvaluationResult]:
    """Verify every HTN leaf maps to a valid ``(unit, intervention, period)`` and is feasible."""
    leaves = collect_htn_leaves(root)
    if not leaves:
        raise ValueError("HTN plan produced zero primitive leaf actions.")

    tensor = np.zeros(
        (evaluator.n_units, evaluator.n_interventions, evaluator.n_periods),
        dtype=float,
    )
    valid_units = set(evaluator.units)
    valid_intervs = set(evaluator.interventions)
    valid_periods = set(evaluator.periods)

    for leaf in leaves:
        if leaf.unit not in valid_units:
            raise ValueError(f"Invalid HTN leaf unit: {leaf.unit!r}")
        if leaf.intervention not in valid_intervs:
            raise ValueError(f"Invalid HTN leaf intervention: {leaf.intervention!r}")
        if leaf.period not in valid_periods:
            raise ValueError(f"Invalid HTN leaf period: {leaf.period!r}")
        u_i = evaluator.units.index(leaf.unit)
        i_i = evaluator.interventions.index(leaf.intervention)
        p_i = evaluator.periods.index(leaf.period)
        tensor[u_i, i_i, p_i] += float(leaf.allocation_value)

    x_vec = tensor.reshape(-1)
    res = evaluator.evaluate(x_vec)
    if not bool(res.is_feasible):
        raise ValueError(
            "HTN plan violated Stage3Evaluator constraints: "
            f"budget={res.violation_budget}, cap={res.violation_adoption_cap}, "
            f"ceiling={res.violation_coverage_ceiling}, infra={res.violation_infra_prereq}"
        )
    return x_vec, res


def run_htn_pipeline(
    config_path: str | Path,
    base_dir: str | Path | None = None,
    *,
    generate_figure: bool = True,
) -> dict[str, Any]:
    """Run HTN planning, verify leaves, and write JSON trace and tree figure."""
    root_dir = Path(base_dir) if base_dir is not None else Path(".")
    config = load_config(config_path)
    contracts = load_stage3_contracts(config, base_dir=root_dir)
    evaluator = Stage3Evaluator(config=config, contracts=contracts)

    tables_dir = root_dir / config["paths"]["outputs_tables_dir"]
    figures_dir = root_dir / config["paths"]["outputs_figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tree, x_vec, ev_res = build_htn_plan(evaluator)
    leaves = collect_htn_leaves(tree)

    suffix = "_mock" if evaluator.is_mock else ""
    trace_path = tables_dir / f"plan_htn_trace{suffix}.json"
    meta_path = Path(f"{trace_path}.meta.json")

    payload = {
        "schema_version": "1.0",
        "source": "src.planning.htn",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_mock": evaluator.is_mock,
        "warning_mock_inputs": evaluator.warning_mock_inputs,
        "data_version": evaluator.data_version,
        "summary": {
            "n_primitive_leaves": len(leaves),
            "dalys_2040_mean": float(ev_res.dalys_2040_mean),
            "dalys_2040_lower": float(ev_res.dalys_2040_lower),
            "dalys_2040_upper": float(ev_res.dalys_2040_upper),
            "total_cost_usd_millions": float(ev_res.total_cost_usd_millions),
            "equity_gap_asr": float(ev_res.equity_gap_asr),
            "is_feasible": bool(ev_res.is_feasible),
        },
        "leaves": [leaf.to_dict() for leaf in leaves],
        "tree": tree.to_dict(),
    }
    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "artifact": trace_path.name,
                "source": "src.planning.htn",
                "generated_at": payload["generated_at"],
                "is_mock": evaluator.is_mock,
                "warning_mock_inputs": evaluator.warning_mock_inputs,
                "data_version": evaluator.data_version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    files: dict[str, Path] = {"htn_trace_json": trace_path, "htn_trace_meta": meta_path}
    if generate_figure:
        from src.viz.decision import plot_htn_tree

        fig_path = figures_dir / f"plan_htn_tree{suffix}.png"
        plot_htn_tree(
            tree,
            ev_res,
            fig_path,
            is_mock=evaluator.is_mock,
            data_version=evaluator.data_version,
        )
        fig_meta = Path(f"{fig_path}.meta.json")
        fig_meta.write_text(
            json.dumps(
                {
                    "artifact": fig_path.name,
                    "source": "src.planning.htn",
                    "generated_at": payload["generated_at"],
                    "is_mock": evaluator.is_mock,
                    "warning_mock_inputs": evaluator.warning_mock_inputs,
                    "data_version": evaluator.data_version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        files["fig_htn"] = fig_path
        files["fig_htn_meta"] = fig_meta

    return {
        "tree": tree,
        "leaves": leaves,
        "x_vec": x_vec,
        "evaluation": ev_res,
        "files": files,
    }
