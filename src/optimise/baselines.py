"""Baseline heuristics for Stage 3 comparison against NSGA-II.

Implements three comparators evaluated under identical contracts, constraints,
and total budget via :class:`src.optimise.evaluator.Stage3Evaluator`:
1. Proportional / equal split across units, cooking interventions, and periods.
2. Greedy by marginal DALYs averted per USD million (bundling infra prerequisites).
3. Random search over the repaired feasible space with matched evaluation count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

from src.optimise.evaluator import EvaluationResult, Stage3Evaluator
from src.optimise.nsga2 import (
    MultiSeedResult,
    NSGA2Config,
    extract_non_dominated_feasible,
    repair_decision_vectors,
)

_FLOAT_EPS: float = 1e-12


@dataclass(frozen=True)
class BaselineResult:
    """Evaluated Pareto set (or single solution) and metadata for one baseline method."""

    method: str
    X: np.ndarray
    F: np.ndarray
    evaluation: EvaluationResult
    hypervolume: float
    n_evaluations: int
    runtime_seconds: float


def _make_baseline_result(
    method: str,
    X: np.ndarray,
    evaluator: Stage3Evaluator,
    ref_point: np.ndarray,
    n_evaluations: int,
    runtime_seconds: float,
) -> BaselineResult:
    X_nd, F_nd, eval_nd, _ = extract_non_dominated_feasible(np.atleast_2d(X), evaluator)
    hv_val = float(HV(ref_point=ref_point)(F_nd))
    return BaselineResult(
        method=method,
        X=X_nd,
        F=F_nd,
        evaluation=eval_nd,
        hypervolume=hv_val,
        n_evaluations=int(n_evaluations),
        runtime_seconds=float(runtime_seconds),
    )


def run_proportional_equal_split(
    evaluator: Stage3Evaluator,
    ref_point: np.ndarray,
) -> BaselineResult:
    """Allocate budget equally across units and uniformly across feasible slots.

    Unready units receive ``infra_lpg_grid = infra_build_threshold`` in ``P1``
    first (lowest infra cost first if budget is tight), and remaining budget is
    divided equally across units with iterative redistribution of any ceiling-
    capped surplus.
    """
    t0 = time.perf_counter()
    tensor = np.zeros(
        (evaluator.n_units, evaluator.n_interventions, evaluator.n_periods),
        dtype=float,
    )
    disc = evaluator._period_discount
    infra_idx = evaluator._idx_infra
    cook_idx = evaluator._non_infra_indices

    # 1. Build P1 infrastructure for units that are not initially fully ready (when infra enabled)
    infra_spent = 0.0
    unready_units = [
        u_idx
        for u_idx in range(evaluator.n_units)
        if evaluator._infra_enabled[u_idx]
        and ((evaluator._lpg_infra_init[u_idx] < 1.0) or (evaluator._grid_ready_init[u_idx] < 1.0))
    ]
    unready_units.sort(key=lambda u: float(evaluator._infra_costs[u] * disc[0]))
    for u_idx in unready_units:
        build_cost = float(evaluator._infra_costs[u_idx] * disc[0])
        if infra_spent + build_cost <= evaluator.total_budget_usd_millions - _FLOAT_EPS:
            tensor[u_idx, infra_idx, 0] = evaluator.infra_build_threshold
            infra_spent += build_cost

    # 2. Identify ready status per (u, cook_i, p) using strictly earlier infra (q < p)
    infra_built = (tensor[:, infra_idx, :] >= evaluator.infra_build_threshold).astype(float)
    cum_built = np.maximum.accumulate(infra_built, axis=1)
    earlier_built = np.zeros_like(cum_built)
    if evaluator.n_periods > 1:
        earlier_built[:, 1:] = cum_built[:, :-1]
    lpg_ready = np.maximum(evaluator._lpg_infra_init[:, np.newaxis], earlier_built)
    grid_ready = np.maximum(evaluator._grid_ready_init[:, np.newaxis], earlier_built)

    allow_mask = np.ones((evaluator.n_units, len(cook_idx), evaluator.n_periods), dtype=bool)
    lpg_pos = int(np.flatnonzero(cook_idx == evaluator._idx_lpg)[0])
    elec_pos = int(np.flatnonzero(cook_idx == evaluator._idx_elec)[0])
    allow_mask[:, lpg_pos, :] = lpg_ready >= 1.0
    allow_mask[:, elec_pos, :] = grid_ready >= 1.0

    # Unit max feasible cooking vector respecting adoption caps & coverage ceiling
    unit_full_cook = np.where(allow_mask, evaluator._non_infra_max_adoption, 0.0)
    headroom = np.maximum(0.0, evaluator.effective_adoption_headroom() - _FLOAT_EPS)
    for u_idx in range(evaluator.n_units):
        total_u = float(np.sum(unit_full_cook[u_idx]))
        if total_u > headroom[u_idx] and total_u > 0.0:
            unit_full_cook[u_idx] *= headroom[u_idx] / total_u

    unit_max_cost = np.sum(
        unit_full_cook
        * evaluator._non_infra_costs[:, :, np.newaxis]
        * disc[np.newaxis, np.newaxis, :],
        axis=(1, 2),
    )

    # 3. Equal budget split across units with iterative redistribution of surplus
    rem_budget = max(0.0, evaluator.total_budget_usd_millions - infra_spent - _FLOAT_EPS)
    unit_budgets = np.zeros(evaluator.n_units, dtype=float)
    active = unit_max_cost > _FLOAT_EPS
    while rem_budget > 1e-9 and np.any(active):
        share = rem_budget / float(np.sum(active))
        added = np.zeros(evaluator.n_units, dtype=float)
        for u_idx in np.flatnonzero(active):
            cap_left = unit_max_cost[u_idx] - unit_budgets[u_idx]
            grant = min(share, cap_left)
            unit_budgets[u_idx] += grant
            added[u_idx] = grant
            if unit_budgets[u_idx] >= unit_max_cost[u_idx] - 1e-12:
                active[u_idx] = False
        spent_round = float(np.sum(added))
        if spent_round <= 1e-12:
            break
        rem_budget -= spent_round

    for u_idx in range(evaluator.n_units):
        if unit_max_cost[u_idx] > 0.0:
            frac = min(1.0, unit_budgets[u_idx] / unit_max_cost[u_idx])
            tensor[u_idx, cook_idx, :] = unit_full_cook[u_idx] * frac

    x = repair_decision_vectors(tensor.reshape(-1), evaluator, repair_budget=True)
    runtime_s = time.perf_counter() - t0
    return _make_baseline_result(
        method="baseline_proportional_split",
        X=x[np.newaxis, :],
        evaluator=evaluator,
        ref_point=ref_point,
        n_evaluations=1,
        runtime_seconds=runtime_s,
    )


def run_greedy_dalys_per_dollar(
    evaluator: Stage3Evaluator,
    ref_point: np.ndarray,
    *,
    step_size: float = 0.05,
) -> BaselineResult:
    """Greedy heuristic maximizing marginal DALYs averted per USD million.

    Evaluates feasible increments of size ``<= step_size`` across all
    ``(unit, cooking_intervention, period)`` slots, automatically bundling the
    required ``infra_lpg_grid`` build in period ``p`` when an unready unit
    selects ``lpg_subsidy`` or ``electric_induction``.
    """
    t0 = time.perf_counter()
    tensor = np.zeros(
        (evaluator.n_units, evaluator.n_interventions, evaluator.n_periods),
        dtype=float,
    )
    cook_idx = evaluator._non_infra_indices
    infra_idx = evaluator._idx_infra
    disc = evaluator._period_discount

    curr_eval = evaluator.evaluate(tensor.reshape(-1))
    curr_dalys = float(curr_eval.dalys_2040_mean)
    curr_cost = float(curr_eval.total_cost_usd_millions)
    n_eval = 1

    headroom = np.maximum(0.0, evaluator.effective_adoption_headroom() - _FLOAT_EPS)

    while True:
        infra_built = (tensor[:, infra_idx, :] >= evaluator.infra_build_threshold).astype(float)
        cum_built = np.maximum.accumulate(infra_built, axis=1)
        earlier_built = np.zeros_like(cum_built)
        if evaluator.n_periods > 1:
            earlier_built[:, 1:] = cum_built[:, :-1]
        lpg_ready = np.maximum(evaluator._lpg_infra_init[:, np.newaxis], earlier_built)
        grid_ready = np.maximum(evaluator._grid_ready_init[:, np.newaxis], earlier_built)

        candidates: list[np.ndarray] = []
        for u_idx in range(evaluator.n_units):
            used_u = float(np.sum(tensor[u_idx, cook_idx, :]))
            rem_headroom = max(0.0, float(headroom[u_idx]) - used_u)
            if rem_headroom <= 1e-9:
                continue

            for c_pos, i_idx in enumerate(cook_idx):
                unit_cost_i = float(evaluator._non_infra_costs[u_idx, c_pos])
                for p_idx in range(evaluator.n_periods):
                    cap_rem = float(evaluator._non_infra_max_adoption[u_idx, c_pos, p_idx]) - float(
                        tensor[u_idx, i_idx, p_idx]
                    )
                    delta = min(step_size, cap_rem, rem_headroom)
                    if delta <= 1e-9:
                        continue

                    needs_infra = (
                        i_idx == evaluator._idx_lpg and lpg_ready[u_idx, p_idx] < 1.0
                    ) or (i_idx == evaluator._idx_elec and grid_ready[u_idx, p_idx] < 1.0)
                    if needs_infra and (p_idx == 0 or not evaluator._infra_enabled[u_idx]):
                        continue
                    infra_p_idx = p_idx - 1
                    infra_delta_cost = (
                        float(evaluator._infra_costs[u_idx] * disc[infra_p_idx])
                        if needs_infra
                        else 0.0
                    )
                    slot_unit_cost = unit_cost_i * float(disc[p_idx])
                    rem_budget = max(
                        0.0, evaluator.total_budget_usd_millions - curr_cost - _FLOAT_EPS
                    )
                    if infra_delta_cost >= rem_budget:
                        continue
                    if slot_unit_cost > 0.0:
                        max_affordable_delta = (rem_budget - infra_delta_cost) / slot_unit_cost
                        delta = min(delta, max_affordable_delta)
                    if delta <= 1e-9:
                        continue

                    cand = tensor.copy()
                    cand[u_idx, i_idx, p_idx] += delta
                    if needs_infra:
                        cand[u_idx, infra_idx, infra_p_idx] = evaluator.infra_build_threshold
                    candidates.append(cand.reshape(-1))

        if not candidates:
            break

        batch = np.vstack(candidates)
        batch_eval = evaluator.evaluate(batch)
        n_eval += len(batch)

        dalys_gain = curr_dalys - np.asarray(batch_eval.dalys_2040_mean, dtype=float)
        cost_gain = np.asarray(batch_eval.total_cost_usd_millions, dtype=float) - curr_cost
        feas = np.asarray(batch_eval.is_feasible, dtype=bool)
        valid = feas & (dalys_gain > 0.0) & (cost_gain > 1e-12)
        if not np.any(valid):
            break

        ratios = np.where(valid, dalys_gain / np.maximum(cost_gain, 1e-12), -np.inf)
        best_pos = int(np.argmax(ratios))
        if not np.isfinite(ratios[best_pos]) or ratios[best_pos] <= 0.0:
            break

        tensor = batch[best_pos].reshape(
            evaluator.n_units, evaluator.n_interventions, evaluator.n_periods
        )
        curr_dalys = float(np.asarray(batch_eval.dalys_2040_mean, dtype=float)[best_pos])
        curr_cost = float(np.asarray(batch_eval.total_cost_usd_millions, dtype=float)[best_pos])

    x = repair_decision_vectors(tensor.reshape(-1), evaluator, repair_budget=True)
    runtime_s = time.perf_counter() - t0
    return _make_baseline_result(
        method="baseline_greedy_dalys_per_dollar",
        X=x[np.newaxis, :],
        evaluator=evaluator,
        ref_point=ref_point,
        n_evaluations=n_eval,
        runtime_seconds=runtime_s,
    )


def run_random_search(
    evaluator: Stage3Evaluator,
    ref_point: np.ndarray,
    *,
    n_eval: int,
    seed: int,
    repair_budget: bool = True,
) -> BaselineResult:
    """Sample ``n_eval`` repaired random vectors in ``[0, 1]^48`` and keep Pareto front."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    raw_X = rng.uniform(0.0, 1.0, size=(int(n_eval), evaluator.n_vars))
    repaired_X = repair_decision_vectors(raw_X, evaluator, repair_budget=repair_budget)
    runtime_s = time.perf_counter() - t0
    return _make_baseline_result(
        method="baseline_random_search",
        X=repaired_X,
        evaluator=evaluator,
        ref_point=ref_point,
        n_evaluations=int(n_eval),
        runtime_seconds=runtime_s,
    )


def _fraction_dominated_by(front_a: np.ndarray, front_b: np.ndarray) -> float:
    """Return fraction of points in ``front_b`` strictly dominated by ``front_a``."""
    if len(front_b) == 0:
        return 0.0
    # Point b is dominated by a iff all(a <= b) and any(a < b)
    le = front_a[:, np.newaxis, :] <= front_b[np.newaxis, :, :]
    lt = front_a[:, np.newaxis, :] < front_b[np.newaxis, :, :]
    dom_matrix = np.all(le, axis=2) & np.any(lt, axis=2)
    return float(np.mean(np.any(dom_matrix, axis=0)))


def run_all_baselines(
    evaluator: Stage3Evaluator,
    config: dict[str, Any],
    ref_point: np.ndarray,
    *,
    n_eval_match: int | None = None,
) -> dict[str, BaselineResult]:
    """Run proportional split, greedy DALYs/$, and matched-budget random search."""
    nsga2_cfg = NSGA2Config.from_config(config)
    s3_cfg = config.get("stage3", {})
    step_size = float(s3_cfg.get("greedy_step_size", 0.05))
    random_evals = (
        int(n_eval_match)
        if n_eval_match is not None
        else nsga2_cfg.population_size * nsga2_cfg.n_generations
    )
    random_seed = int(config.get("seeds", {}).get("numpy", nsga2_cfg.seeds[0]))

    prop = run_proportional_equal_split(evaluator, ref_point)
    greedy = run_greedy_dalys_per_dollar(evaluator, ref_point, step_size=step_size)
    rand = run_random_search(
        evaluator,
        ref_point,
        n_eval=random_evals,
        seed=random_seed,
        repair_budget=nsga2_cfg.repair_budget,
    )
    return {
        "proportional": prop,
        "greedy": greedy,
        "random": rand,
    }


def compare_nsga2_vs_baselines(
    nsga2_result: MultiSeedResult,
    baselines: dict[str, BaselineResult],
    evaluator: Stage3Evaluator,
) -> pd.DataFrame:
    """Build comparison table across NSGA-II (combined + primary seed) and baselines."""
    bau_dalys = float(nsga2_result.bau_evaluation.dalys_2040_mean)
    combined_hv = nsga2_result.combined_hypervolume
    primary = nsga2_result.seed_results[0]
    total_nsga2_evals = sum(sr.n_eval for sr in nsga2_result.seed_results)

    entries: list[tuple[str, np.ndarray, EvaluationResult, float, int, float]] = [
        (
            "nsga2_combined_5seeds",
            nsga2_result.combined_F,
            nsga2_result.combined_evaluation,
            combined_hv,
            total_nsga2_evals,
            sum(sr.runtime_seconds for sr in nsga2_result.seed_results),
        ),
        (
            f"nsga2_seed_{primary.seed}",
            primary.F,
            primary.evaluation,
            primary.hypervolume,
            primary.n_eval,
            primary.runtime_seconds,
        ),
    ]
    for key in ("proportional", "greedy", "random"):
        b = baselines[key]
        entries.append(
            (b.method, b.F, b.evaluation, b.hypervolume, b.n_evaluations, b.runtime_seconds)
        )

    rows: list[dict[str, Any]] = []
    for method, F_pts, ev_res, hv_val, n_eval, rt_s in entries:
        dalys_mean = np.asarray(ev_res.dalys_2040_mean, dtype=float)
        dalys_low = np.asarray(ev_res.dalys_2040_lower, dtype=float)
        dalys_up = np.asarray(ev_res.dalys_2040_upper, dtype=float)
        best_idx = int(np.argmin(dalys_mean))
        hv_vs_nsga2 = 100.0 * (hv_val - combined_hv) / combined_hv if combined_hv > 0.0 else 0.0
        is_nsga = method.startswith("nsga2_")
        frac_dom = 0.0 if is_nsga else _fraction_dominated_by(nsga2_result.combined_F, F_pts)
        rows.append(
            {
                "method": method,
                "n_evaluations": n_eval,
                "n_pareto_points": int(len(F_pts)),
                "hypervolume": float(hv_val),
                "hypervolume_diff_vs_nsga2_pct": float(hv_vs_nsga2),
                "nsga2_beats_hypervolume": bool(not is_nsga and combined_hv > hv_val),
                "fraction_dominated_by_nsga2": float(frac_dom),
                "min_dalys_2040_mean": float(dalys_mean[best_idx]),
                "min_dalys_2040_lower": float(dalys_low[best_idx]),
                "min_dalys_2040_upper": float(dalys_up[best_idx]),
                "max_dalys_averted_mean": float(bau_dalys - dalys_mean[best_idx]),
                "cost_at_min_dalys_usd_millions": float(F_pts[best_idx, 1]),
                "min_cost_usd_millions": float(F_pts[:, 1].min()),
                "max_cost_usd_millions": float(F_pts[:, 1].max()),
                "min_equity_gap_asr": float(F_pts[:, 2].min()),
                "runtime_seconds": float(rt_s),
                "is_mock": evaluator.is_mock,
                "warning_mock_inputs": evaluator.warning_mock_inputs,
                "data_version": evaluator.data_version,
            }
        )
    return pd.DataFrame(rows)
