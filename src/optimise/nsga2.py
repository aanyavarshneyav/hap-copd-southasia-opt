"""NSGA-II multi-objective optimisation over the Stage 3 surrogate evaluator.

Optimises the 48-variable decision vector (4 units x 4 interventions x 3 periods)
using ``pymoo``'s ``NSGA2`` in vectorized mode, minimising:
1. Projected 2040 HAP-attributable COPD DALYs (mean),
2. Total intervention cost (USD millions),
3. Female-male burden equity gap (population-weighted ASR gap).

All 4 evaluator constraints (budget, per-period adoption caps, coverage ceilings,
and infrastructure prerequisites) are passed to ``pymoo`` via ``out["G"]``, paired
with an optional vectorized feasibility repair operator (``Stage3Repair``) logged
in ``docs/DECISIONS.md`` so feasible solutions are reliably discovered inside the
48D ``[0, 1]`` hypercube.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.optimise.evaluator import (
    EvaluationResult,
    Stage3Evaluator,
    tensor_to_vector,
    vector_to_tensor,
)

# Small margin to avoid IEEE-754 float overshoot (e.g., 0.40 + 0.55 > 0.95 by 1e-16)
# when scaling continuous allocations down to exact ceilings or budget limits.
_FLOAT_EPS: float = 1e-12


@dataclass(frozen=True)
class NSGA2Config:
    """Hyperparameters and seeds for Stage 3 NSGA-II runs loaded from YAML config."""

    population_size: int
    n_generations: int
    seeds: tuple[int, ...]
    crossover_prob: float
    crossover_eta: float
    mutation_eta: float
    eliminate_duplicates: bool
    use_repair: bool
    repair_budget: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NSGA2Config:
        """Parse Stage 3 NSGA-II configuration from ``configs/default.yaml``."""
        s3 = config.get("stage3", {})
        seeds_cfg = s3.get("seeds")
        if isinstance(seeds_cfg, list) and seeds_cfg:
            seeds = tuple(int(s) for s in seeds_cfg)
        else:
            base_seed = int(config.get("seeds", {}).get("pymoo", 42))
            seeds = tuple(base_seed + offset for offset in range(5))

        return cls(
            population_size=int(s3.get("population_size", 80)),
            n_generations=int(s3.get("n_generations", 100)),
            seeds=seeds,
            crossover_prob=float(s3.get("crossover_prob", 0.9)),
            crossover_eta=float(s3.get("crossover_eta", 15.0)),
            mutation_eta=float(s3.get("mutation_eta", 20.0)),
            eliminate_duplicates=bool(s3.get("eliminate_duplicates", True)),
            use_repair=bool(s3.get("use_repair", True)),
            repair_budget=bool(s3.get("repair_budget", True)),
        )


@dataclass(frozen=True)
class SingleSeedResult:
    """Outputs and convergence trace from one seeded NSGA-II optimisation."""

    seed: int
    X: np.ndarray
    F: np.ndarray
    evaluation: EvaluationResult
    hypervolume: float
    hv_history: list[dict[str, float | int]]
    n_eval: int
    runtime_seconds: float


@dataclass(frozen=True)
class MultiSeedResult:
    """Aggregated multi-seed NSGA-II results and combined global Pareto front."""

    seed_results: list[SingleSeedResult]
    combined_X: np.ndarray
    combined_F: np.ndarray
    combined_seeds: np.ndarray
    combined_evaluation: EvaluationResult
    combined_hypervolume: float
    ref_point: np.ndarray
    bau_evaluation: EvaluationResult


def repair_decision_vectors(
    X: np.ndarray,
    evaluator: Stage3Evaluator,
    *,
    repair_budget: bool = True,
) -> np.ndarray:
    """Project arbitrary ``(N, 48)`` or ``(48,)`` vectors into the feasible domain.

    Steps applied in order:
    1. Clip all entries to ``[0, 1]``.
    2. Clip cooking variables to per-period adoption caps ``max_adoption_per_period``.
    3. If infrastructure builds alone exceed ``total_budget_usd_millions``, drop
       latest-period / highest-cost infra builds until infra cost fits budget.
    4. Zero out ``lpg_subsidy`` and ``electric_induction`` in periods where
       infrastructure is not yet ready (initial readiness or ``infra_lpg_grid`` built
       in a strictly earlier period ``q < p``).
    5. Proportionally scale down per-unit cooking allocations exceeding unit
       ``coverage_ceiling`` headroom.
    6. Optionally scale down cooking allocations when ``total_cost > total_budget``.
    """
    arr = np.asarray(X, dtype=float)
    is_single = arr.ndim == 1
    batch = arr[np.newaxis, :] if is_single else arr
    clipped = np.clip(batch, 0.0, 1.0)

    tensor = vector_to_tensor(
        clipped, evaluator.n_units, evaluator.n_interventions, evaluator.n_periods
    ).copy()

    cook_idx = evaluator._non_infra_indices
    infra_idx = evaluator._idx_infra
    disc = evaluator._period_discount

    # 2. Per-period adoption caps on cooking & infrastructure interventions
    tensor[:, :, cook_idx, :] = np.minimum(
        tensor[:, :, cook_idx, :],
        evaluator._non_infra_max_adoption[np.newaxis, :, :, :],
    )
    tensor[:, :, infra_idx, :] = np.minimum(
        tensor[:, :, infra_idx, :],
        evaluator._infra_max_adoption[np.newaxis, :, :],
    )
    tensor[:, ~evaluator._infra_enabled, infra_idx, :] = 0.0

    # 3. Ensure infrastructure builds themselves fit within total budget
    infra_built = (
        (tensor[:, :, infra_idx, :] >= evaluator.infra_build_threshold)
        & evaluator._infra_enabled[np.newaxis, :, np.newaxis]
    ).astype(float)
    infra_cost = np.sum(
        infra_built
        * evaluator._infra_costs[np.newaxis, :, np.newaxis]
        * disc[np.newaxis, np.newaxis, :],
        axis=(1, 2),
    )
    over_infra = infra_cost > evaluator.total_budget_usd_millions
    if np.any(over_infra):
        for row_idx in np.flatnonzero(over_infra):
            running = float(infra_cost[row_idx])
            # Drop latest-period, highest-cost infra builds first
            candidates = [
                (
                    p_idx,
                    float(evaluator._infra_costs[u_idx] * disc[p_idx]),
                    u_idx,
                )
                for p_idx in range(evaluator.n_periods - 1, -1, -1)
                for u_idx in range(evaluator.n_units)
                if infra_built[row_idx, u_idx, p_idx] > 0.0
            ]
            candidates.sort(key=lambda item: (-item[0], -item[1]))
            for p_idx, slot_cost, u_idx in candidates:
                if running <= evaluator.total_budget_usd_millions:
                    break
                tensor[row_idx, u_idx, infra_idx, p_idx] = 0.0
                infra_built[row_idx, u_idx, p_idx] = 0.0
                running -= slot_cost

    # 4. Enforce data-driven infrastructure prerequisites (strictly earlier period q < p)
    cum_built = np.maximum.accumulate(infra_built, axis=2)
    earlier_built = np.zeros_like(cum_built)
    if evaluator.n_periods > 1:
        earlier_built[:, :, 1:] = cum_built[:, :, :-1]
    lpg_ready = np.maximum(evaluator._lpg_infra_init[np.newaxis, :, np.newaxis], earlier_built)
    grid_ready = np.maximum(evaluator._grid_ready_init[np.newaxis, :, np.newaxis], earlier_built)
    tensor[:, :, evaluator._idx_lpg, :] *= lpg_ready
    tensor[:, :, evaluator._idx_elec, :] *= grid_ready

    # 5. Enforce joint household-ceiling & population-headroom limits with float margin
    headroom = evaluator.effective_adoption_headroom()[np.newaxis, :]
    raw_sum = np.sum(tensor[:, :, cook_idx, :], axis=(2, 3))
    safe_headroom = np.maximum(0.0, headroom - _FLOAT_EPS)
    ceil_scale = np.where(raw_sum > headroom, safe_headroom / np.maximum(raw_sum, _FLOAT_EPS), 1.0)
    tensor[:, :, cook_idx, :] *= ceil_scale[:, :, np.newaxis, np.newaxis]

    # 6. Optional budget scaling on cooking allocations
    if repair_budget:
        infra_cost = np.sum(
            infra_built
            * evaluator._infra_costs[np.newaxis, :, np.newaxis]
            * disc[np.newaxis, np.newaxis, :],
            axis=(1, 2),
        )
        cook_cost = np.sum(
            tensor[:, :, cook_idx, :]
            * evaluator._non_infra_costs[np.newaxis, :, :, np.newaxis]
            * disc[np.newaxis, np.newaxis, np.newaxis, :],
            axis=(1, 2, 3),
        )
        total_cost = infra_cost + cook_cost
        over_budget = total_cost > evaluator.total_budget_usd_millions
        if np.any(over_budget):
            rem_budget = np.maximum(
                0.0, evaluator.total_budget_usd_millions - infra_cost - _FLOAT_EPS
            )
            b_scale = np.where(over_budget, rem_budget / np.maximum(cook_cost, _FLOAT_EPS), 1.0)
            tensor[:, :, cook_idx, :] *= b_scale[:, np.newaxis, np.newaxis, np.newaxis]

    repaired = tensor_to_vector(tensor)
    return repaired[0] if is_single else repaired


class Stage3Repair(Repair):
    """Pymoo vectorized repair wrapper around :func:`repair_decision_vectors`."""

    def __init__(self, evaluator: Stage3Evaluator, *, repair_budget: bool = True) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.repair_budget = repair_budget

    def _do(self, problem: Problem, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return repair_decision_vectors(X, self.evaluator, repair_budget=self.repair_budget)


class Stage3Problem(Problem):
    """Vectorized pymoo multi-objective problem for Stage 3 allocation."""

    def __init__(self, evaluator: Stage3Evaluator) -> None:
        super().__init__(
            n_var=evaluator.n_vars,
            n_obj=3,
            n_ieq_constr=4,
            xl=0.0,
            xu=1.0,
        )
        self.evaluator = evaluator

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        res = self.evaluator.evaluate(X)
        out["F"] = np.column_stack(
            [
                res.dalys_2040_mean,
                res.total_cost_usd_millions,
                res.equity_gap_asr,
            ]
        )
        out["G"] = np.column_stack(
            [
                res.violation_budget,
                res.violation_adoption_cap,
                res.violation_coverage_ceiling,
                res.violation_infra_prereq,
            ]
        )


class _HypervolumeCallback(Callback):
    """Record feasible non-dominated hypervolume at each generation."""

    def __init__(self, ref_point: np.ndarray) -> None:
        super().__init__()
        self.hv_indicator = HV(ref_point=ref_point)
        self.history: list[dict[str, float | int]] = []

    def notify(self, algorithm: Any) -> None:
        pop = algorithm.pop
        F = pop.get("F")
        CV = pop.get("CV")
        if F is None or len(F) == 0:
            hv_val = 0.0
        else:
            feas_mask = np.ones(len(F), dtype=bool) if CV is None else (CV.reshape(-1) <= 0.0)
            F_feas = F[feas_mask]
            if len(F_feas) == 0:
                hv_val = 0.0
            else:
                nd_idx = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)
                hv_val = float(self.hv_indicator(F_feas[nd_idx]))
        self.history.append(
            {
                "gen": int(algorithm.n_gen),
                "n_eval": int(algorithm.evaluator.n_eval),
                "hypervolume": hv_val,
            }
        )


def compute_reference_point(
    evaluator: Stage3Evaluator,
    extra_F: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a deterministic hypervolume reference point anchored to 2040 BAU & budget."""
    bau = evaluator.evaluate(np.zeros(evaluator.n_vars, dtype=float))
    max_dalys = float(bau.dalys_2040_mean)
    max_cost = float(evaluator.total_budget_usd_millions)
    max_eq = float(bau.equity_gap_asr)
    if extra_F is not None and len(extra_F) > 0:
        max_dalys = max(max_dalys, float(np.max(extra_F[:, 0])))
        max_cost = max(max_cost, float(np.max(extra_F[:, 1])))
        max_eq = max(max_eq, float(np.max(extra_F[:, 2])))
    return np.array([max_dalys * 1.05, max_cost * 1.05, max_eq * 1.10], dtype=float)


def extract_non_dominated_feasible(
    X: np.ndarray,
    evaluator: Stage3Evaluator,
    seeds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, EvaluationResult, np.ndarray | None]:
    """Filter to strictly feasible, deduplicated, non-dominated solutions sorted by cost."""
    arr = np.asarray(X, dtype=float)
    if arr.size == 0:
        raise RuntimeError("Optimisation returned an empty solution set.")

    res = evaluator.evaluate(arr)
    feas_mask = np.asarray(res.is_feasible, dtype=bool)
    if not np.any(feas_mask):
        raise RuntimeError("No feasible solutions satisfy all Stage 3 constraints.")

    X_feas = arr[feas_mask]
    F_feas = np.column_stack(
        [
            np.asarray(res.dalys_2040_mean, dtype=float)[feas_mask],
            np.asarray(res.total_cost_usd_millions, dtype=float)[feas_mask],
            np.asarray(res.equity_gap_asr, dtype=float)[feas_mask],
        ]
    )
    seeds_feas = seeds[feas_mask] if seeds is not None else None

    # Deduplicate identical objective points deterministically before NDS
    F_rounded = np.round(F_feas, decimals=10)
    _, unique_idx = np.unique(F_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    X_uniq = X_feas[unique_idx]
    F_uniq = F_feas[unique_idx]
    seeds_uniq = seeds_feas[unique_idx] if seeds_feas is not None else None

    nd_idx = NonDominatedSorting().do(F_uniq, only_non_dominated_front=True)
    X_nd = X_uniq[nd_idx]
    F_nd = F_uniq[nd_idx]
    seeds_nd = seeds_uniq[nd_idx] if seeds_uniq is not None else None

    # Sort deterministically by (cost, dalys, equity_gap)
    order = np.lexsort((F_nd[:, 2], F_nd[:, 0], F_nd[:, 1]))
    X_sorted = X_nd[order]
    F_sorted = F_nd[order]
    seeds_sorted = seeds_nd[order] if seeds_nd is not None else None
    eval_sorted = evaluator.evaluate(X_sorted)
    return X_sorted, F_sorted, eval_sorted, seeds_sorted


def run_nsga2_seed(
    evaluator: Stage3Evaluator,
    nsga2_cfg: NSGA2Config,
    seed: int,
    ref_point: np.ndarray | None = None,
) -> SingleSeedResult:
    """Run one deterministic seeded NSGA-II optimisation over ``evaluator``."""
    resolved_ref = ref_point if ref_point is not None else compute_reference_point(evaluator)
    problem = Stage3Problem(evaluator)
    repair_op = (
        Stage3Repair(evaluator, repair_budget=nsga2_cfg.repair_budget)
        if nsga2_cfg.use_repair
        else None
    )
    algorithm = NSGA2(
        pop_size=nsga2_cfg.population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=nsga2_cfg.crossover_prob, eta=nsga2_cfg.crossover_eta),
        mutation=PM(eta=nsga2_cfg.mutation_eta),
        repair=repair_op,
        eliminate_duplicates=nsga2_cfg.eliminate_duplicates,
    )
    callback = _HypervolumeCallback(resolved_ref)

    t0 = time.perf_counter()
    pymoo_res = minimize(
        problem,
        algorithm,
        ("n_gen", nsga2_cfg.n_generations),
        seed=int(seed),
        callback=callback,
        verbose=False,
    )
    runtime_s = time.perf_counter() - t0

    if pymoo_res.X is None or len(pymoo_res.X) == 0:
        raise RuntimeError(
            f"NSGA-II (seed={seed}) found no feasible solutions. "
            "Check constraints or enable use_repair."
        )

    X_nd, F_nd, eval_nd, _ = extract_non_dominated_feasible(np.atleast_2d(pymoo_res.X), evaluator)
    hv_val = float(HV(ref_point=resolved_ref)(F_nd))
    n_eval = int(pymoo_res.algorithm.evaluator.n_eval)

    return SingleSeedResult(
        seed=int(seed),
        X=X_nd,
        F=F_nd,
        evaluation=eval_nd,
        hypervolume=hv_val,
        hv_history=callback.history,
        n_eval=n_eval,
        runtime_seconds=float(runtime_s),
    )


def run_nsga2_multi_seed(
    evaluator: Stage3Evaluator,
    config: dict[str, Any],
) -> MultiSeedResult:
    """Run NSGA-II across all configured seeds and merge the global non-dominated front."""
    nsga2_cfg = NSGA2Config.from_config(config)
    bau_eval = evaluator.evaluate(np.zeros(evaluator.n_vars, dtype=float))
    ref_point = compute_reference_point(evaluator)

    seed_results: list[SingleSeedResult] = []
    all_X: list[np.ndarray] = []
    all_seeds: list[np.ndarray] = []

    for seed in nsga2_cfg.seeds:
        single = run_nsga2_seed(evaluator, nsga2_cfg, seed=seed, ref_point=ref_point)
        seed_results.append(single)
        all_X.append(single.X)
        all_seeds.append(np.full(len(single.X), seed, dtype=int))

    stacked_X = np.vstack(all_X)
    stacked_seeds = np.concatenate(all_seeds)
    combined_X, combined_F, combined_eval, combined_seeds = extract_non_dominated_feasible(
        stacked_X, evaluator, seeds=stacked_seeds
    )
    assert combined_seeds is not None
    combined_hv = float(HV(ref_point=ref_point)(combined_F))

    return MultiSeedResult(
        seed_results=seed_results,
        combined_X=combined_X,
        combined_F=combined_F,
        combined_seeds=combined_seeds,
        combined_evaluation=combined_eval,
        combined_hypervolume=combined_hv,
        ref_point=ref_point,
        bau_evaluation=bau_eval,
    )


def select_representative_plans(
    X: np.ndarray,
    evaluation: EvaluationResult,
    evaluator: Stage3Evaluator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract knee, cost-minimal, DALY-minimal, and equity-minimal plans with UIs."""
    dalys_mean = np.asarray(evaluation.dalys_2040_mean, dtype=float)
    dalys_low = np.asarray(evaluation.dalys_2040_lower, dtype=float)
    dalys_up = np.asarray(evaluation.dalys_2040_upper, dtype=float)
    cost = np.asarray(evaluation.total_cost_usd_millions, dtype=float)
    eq_asr = np.asarray(evaluation.equity_gap_asr, dtype=float)
    eq_dalys = np.asarray(evaluation.equity_gap_dalys_abs, dtype=float)

    bau = evaluator.evaluate(np.zeros(evaluator.n_vars, dtype=float))
    bau_dalys = float(bau.dalys_2040_mean)

    F = np.column_stack([dalys_mean, cost, eq_asr])
    n_pts = len(F)

    # 1. Cost-minimal (break ties by DALYs, then equity)
    cost_min_idx = int(np.lexsort((eq_asr, dalys_mean, cost))[0])

    # 2. DALY-minimal (never select zero vector if any positive-spend solution exists)
    non_zero_mask = ~np.all(np.isclose(X, 0.0, atol=1e-12), axis=1)
    candidate_idxs = np.flatnonzero(non_zero_mask) if np.any(non_zero_mask) else np.arange(n_pts)
    daly_order = np.lexsort(
        (eq_asr[candidate_idxs], cost[candidate_idxs], dalys_mean[candidate_idxs])
    )
    daly_min_idx = int(candidate_idxs[daly_order[0]])

    # 3. Equity-minimal (break ties by DALYs, then cost)
    equity_min_idx = int(np.lexsort((cost, dalys_mean, eq_asr))[0])

    # 4. Knee point: closest to normalized utopia (0, 0, 0) in Euclidean distance
    if n_pts == 1:
        knee_idx = 0
    else:
        f_min = F.min(axis=0)
        f_max = F.max(axis=0)
        span = np.where((f_max - f_min) > 1e-12, f_max - f_min, 1.0)
        F_norm = (F - f_min) / span
        dist_utopia = np.linalg.norm(F_norm, axis=1)
        knee_idx = int(np.lexsort((cost, dalys_mean, dist_utopia))[0])

    selected_indices: dict[str, int] = {
        "knee": knee_idx,
        "cost_minimal": cost_min_idx,
        "daly_minimal": daly_min_idx,
        "equity_minimal": equity_min_idx,
    }

    rows: list[dict[str, Any]] = []
    v_budget = np.asarray(evaluation.violation_budget, dtype=float)
    v_cap = np.asarray(evaluation.violation_adoption_cap, dtype=float)
    v_ceil = np.asarray(evaluation.violation_coverage_ceiling, dtype=float)
    v_infra = np.asarray(evaluation.violation_infra_prereq, dtype=float)
    is_feas = np.asarray(evaluation.is_feasible, dtype=bool)

    for plan_name, idx in selected_indices.items():
        row: dict[str, Any] = {
            "plan_name": plan_name,
            "solution_id": idx,
            "dalys_2040_mean": float(dalys_mean[idx]),
            "dalys_2040_lower": float(dalys_low[idx]),
            "dalys_2040_upper": float(dalys_up[idx]),
            "dalys_averted_mean": float(bau_dalys - dalys_mean[idx]),
            "total_cost_usd_millions": float(cost[idx]),
            "equity_gap_asr": float(eq_asr[idx]),
            "equity_gap_dalys_abs": float(eq_dalys[idx]),
            "is_feasible": bool(is_feas[idx]),
            "violation_budget": float(v_budget[idx]),
            "violation_adoption_cap": float(v_cap[idx]),
            "violation_coverage_ceiling": float(v_ceil[idx]),
            "violation_infra_prereq": float(v_infra[idx]),
            "is_mock": evaluation.is_mock,
            "warning_mock_inputs": evaluation.warning_mock_inputs,
            "data_version": evaluation.data_version,
        }
        for v_idx, var_name in enumerate(evaluator.variable_names):
            row[var_name] = float(X[idx, v_idx])
        rows.append(row)

    return pd.DataFrame(rows), selected_indices


def pareto_front_to_dataframe(
    result: MultiSeedResult,
    evaluator: Stage3Evaluator,
    selected_indices: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Convert combined Pareto front and decoded decision variables to a DataFrame."""
    ev = result.combined_evaluation
    n_pts = len(result.combined_X)

    idx_to_labels: dict[int, list[str]] = {}
    if selected_indices is not None:
        for label, idx in selected_indices.items():
            idx_to_labels.setdefault(idx, []).append(label)

    df = pd.DataFrame(
        {
            "solution_id": np.arange(n_pts, dtype=int),
            "seed": result.combined_seeds.astype(int),
            "selected_plans": [",".join(idx_to_labels.get(i, [])) for i in range(n_pts)],
            "dalys_2040_mean": np.asarray(ev.dalys_2040_mean, dtype=float),
            "dalys_2040_lower": np.asarray(ev.dalys_2040_lower, dtype=float),
            "dalys_2040_upper": np.asarray(ev.dalys_2040_upper, dtype=float),
            "total_cost_usd_millions": np.asarray(ev.total_cost_usd_millions, dtype=float),
            "equity_gap_asr": np.asarray(ev.equity_gap_asr, dtype=float),
            "equity_gap_dalys_abs": np.asarray(ev.equity_gap_dalys_abs, dtype=float),
            "is_feasible": np.asarray(ev.is_feasible, dtype=bool),
            "violation_budget": np.asarray(ev.violation_budget, dtype=float),
            "violation_adoption_cap": np.asarray(ev.violation_adoption_cap, dtype=float),
            "violation_coverage_ceiling": np.asarray(ev.violation_coverage_ceiling, dtype=float),
            "violation_infra_prereq": np.asarray(ev.violation_infra_prereq, dtype=float),
            "is_mock": ev.is_mock,
            "warning_mock_inputs": ev.warning_mock_inputs,
            "data_version": ev.data_version,
        }
    )
    vars_df = pd.DataFrame(result.combined_X, columns=list(evaluator.variable_names))
    return pd.concat([df, vars_df], axis=1)


def hypervolume_seeds_to_dataframe(result: MultiSeedResult) -> pd.DataFrame:
    """Summarize per-seed hypervolume, objective ranges, evaluation count, and runtime."""
    rows: list[dict[str, Any]] = []
    for sr in result.seed_results:
        rows.append(
            {
                "seed": sr.seed,
                "n_pareto_points": int(len(sr.X)),
                "n_evaluations": sr.n_eval,
                "hypervolume": sr.hypervolume,
                "min_dalys_2040_mean": float(sr.F[:, 0].min()),
                "min_cost_usd_millions": float(sr.F[:, 1].min()),
                "max_cost_usd_millions": float(sr.F[:, 1].max()),
                "min_equity_gap_asr": float(sr.F[:, 2].min()),
                "runtime_seconds": sr.runtime_seconds,
                "is_mock": result.combined_evaluation.is_mock,
                "warning_mock_inputs": result.combined_evaluation.warning_mock_inputs,
                "data_version": result.combined_evaluation.data_version,
            }
        )
    return pd.DataFrame(rows)


def decode_allocation_spend(
    x: np.ndarray,
    evaluator: Stage3Evaluator,
) -> pd.DataFrame:
    """Decode one ``(48,)`` decision vector into long-form spend by (unit, intervention, period)."""
    tensor = vector_to_tensor(
        np.asarray(x, dtype=float),
        evaluator.n_units,
        evaluator.n_interventions,
        evaluator.n_periods,
    )
    rows: list[dict[str, Any]] = []
    for u_idx, unit in enumerate(evaluator.units):
        for i_idx, interv in enumerate(evaluator.interventions):
            unit_cost = float(evaluator._unit_costs[u_idx, i_idx])
            for p_idx, period in enumerate(evaluator.periods):
                val = float(tensor[u_idx, i_idx, p_idx])
                disc = float(evaluator._period_discount[p_idx])
                if interv == evaluator.interventions[evaluator._idx_infra]:
                    active = 1.0 if val >= evaluator.infra_build_threshold else 0.0
                    cost = active * unit_cost * disc
                else:
                    cost = val * unit_cost * disc
                rows.append(
                    {
                        "unit": unit,
                        "intervention": interv,
                        "period": period,
                        "allocation_value": val,
                        "cost_usd_millions": float(cost),
                    }
                )
    return pd.DataFrame(rows)
