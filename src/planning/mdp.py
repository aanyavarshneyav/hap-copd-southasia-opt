"""Custom vectorized NumPy MDP solver for Stage 4 sequential clean-cooking planning.

Implements finite-horizon backward induction over the 3 five-year decision periods
(P1: 2026-2030, P2: 2031-2035, P3: 2036-2040) plus infinite-horizon discounted
value iteration (``tol=1e-6``) for stationary policy diagnostics, strictly on equal
footing with Stage 3 NSGA-II:
- Reuses :meth:`Stage3Evaluator.evaluate_effective_gains` directly.
- Tracks raw coverage tiers separately from ``improved_biomass`` (``eta < 1``)
  effective exposure gains with the same ``1e-12`` ceiling safety margin.
- Enforces strictly-earlier-period infrastructure prerequisites (building infra
  at period ``t`` unlocks ``lpg_subsidy`` / ``electric_induction`` from ``t + 1``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import load_config, load_stage3_contracts
from src.optimise.evaluator import Stage3Evaluator

_FLOAT_EPS: float = 1e-12

# Per-unit action codes inside macro-action templates
ACT_NONE: int = 0
ACT_INFRA: int = 1
ACT_LPG_1: int = 2
ACT_BIOMASS_1: int = 3
ACT_ELEC_1: int = 4
ACT_LPG_2: int = 5

_ACT_CODE_LABELS: dict[int, str] = {
    ACT_NONE: "none",
    ACT_INFRA: "infra",
    ACT_LPG_1: "lpg(+1)",
    ACT_BIOMASS_1: "bio(+1)",
    ACT_ELEC_1: "elec(+1)",
    ACT_LPG_2: "lpg(+2)",
}


@dataclass(frozen=True)
class MDPConfig:
    """MDP discretization and solver settings parsed from YAML configuration."""

    coverage_bins: tuple[float, ...]
    budget_bins: int
    discount_factor: float
    cost_weight_lambda: float
    transition_probs: tuple[float, float, float]
    n_rollouts: int
    vi_tol: float
    vi_max_iter: int
    lambda_sensitivity: tuple[float, ...]
    discount_sensitivity: tuple[float, ...]
    seed: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MDPConfig:
        """Load MDP settings from ``config['planning']['mdp']`` and ``config['seeds']``."""
        mdp_cfg = config.get("planning", {}).get("mdp", {})
        probs_raw = mdp_cfg.get("transition_probs", [0.15, 0.70, 0.15])
        if len(probs_raw) != 3:
            raise ValueError(f"Expected 3 transition_probs [low, nom, high], got {probs_raw}.")
        probs = tuple(float(p) for p in probs_raw)
        if any(p < 0.0 for p in probs) or not np.isclose(sum(probs), 1.0, atol=1e-9):
            raise ValueError(f"transition_probs must be non-negative and sum to 1.0, got {probs}.")

        lam_raw = mdp_cfg.get("cost_weight_lambda")
        lam_val = 3500.0 if lam_raw is None else float(lam_raw)

        lam_sens = tuple(
            float(v)
            for v in mdp_cfg.get(
                "lambda_sensitivity", [0.0, 1000.0, 2500.0, 3500.0, 5000.0, 7500.0, 10000.0]
            )
        )
        disc_sens = tuple(float(v) for v in mdp_cfg.get("discount_sensitivity", [0.774, 1.0]))

        return cls(
            coverage_bins=tuple(
                float(x) for x in mdp_cfg.get("coverage_bins", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ),
            budget_bins=int(mdp_cfg.get("budget_bins", 6)),
            discount_factor=float(mdp_cfg.get("discount_factor", 0.774)),
            cost_weight_lambda=lam_val,
            transition_probs=(probs[0], probs[1], probs[2]),
            n_rollouts=int(mdp_cfg.get("n_rollouts", 1000)),
            vi_tol=float(mdp_cfg.get("vi_tol", 1.0e-6)),
            vi_max_iter=int(mdp_cfg.get("vi_max_iter", 500)),
            lambda_sensitivity=lam_sens,
            discount_sensitivity=disc_sens,
            seed=int(config.get("seeds", {}).get("mdp", 42)),
        )


@dataclass(frozen=True)
class PeriodTransitions:
    """Flat sparse NumPy arrays encoding transitions for one decision period."""

    src_state: np.ndarray  # int32 (E,)
    action_id: np.ndarray  # int32 (E,)
    cost: np.ndarray  # float64 (E,)
    next_state: np.ndarray  # int32 (E, 3)
    dalys_averted: np.ndarray  # float64 (E, 3)
    equity_gap_reduced: np.ndarray  # float64 (E, 3)
    state_starts: np.ndarray  # int32 (S_t,)
    state_counts: np.ndarray  # int32 (S_t,)


@dataclass(frozen=True)
class LayeredMDP:
    """Reachable finite-horizon MDP graph and sparse NumPy transition tables."""

    n_units: int
    n_periods: int
    max_bin: int
    n_budget_bins: int
    budget_step: float
    step_u: np.ndarray  # float64 (U,)
    lpg_eta: np.ndarray  # float64 (U,)
    elec_eta: np.ndarray  # float64 (U,)
    bio_eta: np.ndarray  # float64 (U,)
    unready_units: tuple[int, ...]
    templates: tuple[tuple[int, ...], ...]
    probs: np.ndarray  # float64 (3,)
    layers: list[list[tuple[int, ...]]]
    layer_dalys: list[np.ndarray]
    layer_equity_gaps: list[np.ndarray]
    transitions: list[PeriodTransitions]
    bau_dalys: float
    bau_equity_gap: float
    build_seconds: float


@dataclass(frozen=True)
class FiniteHorizonSolution:
    """Optimal value functions and greedy policy indices across periods ``0..H-1``."""

    cost_weight_lambda: float
    discount_factor: float
    values: list[np.ndarray]
    policy_action_ids: list[np.ndarray]
    policy_action_costs: list[np.ndarray]
    expected_value_s0: float
    solve_seconds: float
    equity_weight_lambda: float = 0.0


@dataclass(frozen=True)
class StationaryMDP:
    """Collapsed stationary state graph for infinite-horizon value iteration."""

    states: list[tuple[int, ...]]
    state_dalys: np.ndarray
    transitions: PeriodTransitions
    templates: tuple[tuple[int, ...], ...]
    probs: np.ndarray


@dataclass(frozen=True)
class InfiniteHorizonSolution:
    """Result from vectorized infinite-horizon discounted value iteration."""

    cost_weight_lambda: float
    discount_factor: float
    values: np.ndarray
    policy_action_ids: np.ndarray
    policy_action_costs: np.ndarray
    expected_value_s0: float
    n_iterations: int
    final_delta: float
    converged: bool
    solve_seconds: float


def build_action_templates(n_units: int) -> tuple[tuple[int, ...], ...]:
    """Build systematic, data-driven multi-unit portfolio action templates.

    For ``n_units <= 2``, includes all unit-action combinations over
    ``{none, infra, lpg(+1), bio(+1), elec(+1)}``. For ``n_units > 2`` (e.g., 4 units),
    includes ``no_op``, all single-unit actions (including ``lpg(+2)``), all 2-unit
    pairs over ``{none, infra, lpg(+1), bio(+1)}``, and symmetric all-unit /
    urban-rural bundles so action count stays compact (<= 75 actions).
    """
    if n_units <= 2:
        opts = (ACT_NONE, ACT_INFRA, ACT_LPG_1, ACT_BIOMASS_1, ACT_ELEC_1)
        return tuple(itertools.product(opts, repeat=n_units))

    templates: list[tuple[int, ...]] = [tuple([ACT_NONE] * n_units)]
    # Single-unit actions across all 5 active intervention codes
    for u in range(n_units):
        for act in (ACT_INFRA, ACT_LPG_1, ACT_BIOMASS_1, ACT_ELEC_1, ACT_LPG_2):
            vec = [ACT_NONE] * n_units
            vec[u] = act
            templates.append(tuple(vec))

    # All 2-unit pairs + symmetric urban/rural or uniform all-unit bundles
    pair_opts = (ACT_NONE, ACT_INFRA, ACT_LPG_1, ACT_BIOMASS_1)
    for combo in itertools.product(pair_opts, repeat=n_units):
        n_active = sum(1 for a in combo if a != ACT_NONE)
        is_all_same = len(set(combo)) == 1
        is_even_odd_sym = n_units == 4 and (combo[0] == combo[2] and combo[1] == combo[3])
        if n_active <= 2 or is_all_same or is_even_odd_sym:
            templates.append(combo)

    templates.append(tuple([ACT_ELEC_1] * n_units))
    return tuple(dict.fromkeys(templates))


def format_action_label(combo: tuple[int, ...], units: list[str]) -> str:
    """Format a portfolio action tuple into a human-readable string."""
    parts = [
        f"{units[u]}:{_ACT_CODE_LABELS[act]}" for u, act in enumerate(combo) if act != ACT_NONE
    ]
    return "no_op" if not parts else " + ".join(parts)


def _compute_states_dalys_and_equity(
    states: list[tuple[int, ...]],
    n_units: int,
    step_u: np.ndarray,
    lpg_eta: np.ndarray,
    elec_eta: np.ndarray,
    bio_eta: np.ndarray,
    evaluator: Stage3Evaluator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2040 DALYs and ASR equity gaps for state tuples via ``Stage3Evaluator``."""
    arr = np.asarray(states, dtype=float)
    lpg_bins = arr[:, :n_units]
    elec_bins = arr[:, n_units : 2 * n_units]
    bio_bins = arr[:, 2 * n_units : 3 * n_units]
    eff_gains = (
        lpg_bins * lpg_eta[np.newaxis, :]
        + elec_bins * elec_eta[np.newaxis, :]
        + bio_bins * bio_eta[np.newaxis, :]
    ) * step_u[np.newaxis, :]
    dalys_mean, _, _, eq_gap_asr, _ = evaluator.evaluate_effective_gains(eff_gains)
    return dalys_mean, eq_gap_asr


def _is_action_valid_and_cost(
    combo: tuple[int, ...],
    raws: list[int],
    lpg_ready: list[int],
    grid_ready: list[int],
    rem_usd: float,
    period_idx: int,
    max_bin: int,
    step_u: np.ndarray,
    evaluator: Stage3Evaluator,
) -> tuple[bool, float]:
    """Check preconditions, ceilings, adoption caps, and exact cost for ``combo``."""
    cost = 0.0
    disc = float(evaluator._period_discount[period_idx])
    lpg_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_lpg)[0])
    bio_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_bio)[0])
    elec_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_elec)[0])

    for u, act in enumerate(combo):
        if act == ACT_NONE:
            continue
        if act == ACT_INFRA:
            if lpg_ready[u] == 1 and grid_ready[u] == 1:
                return False, 0.0
            cost += float(evaluator._infra_costs[u]) * disc
            continue

        steps = 2 if act == ACT_LPG_2 else 1
        if act in (ACT_LPG_1, ACT_LPG_2):
            if lpg_ready[u] == 0:
                return False, 0.0
            c_pos = lpg_pos
        elif act == ACT_ELEC_1:
            if grid_ready[u] == 0:
                return False, 0.0
            c_pos = elec_pos
        else:
            c_pos = bio_pos

        if raws[u] + steps > max_bin:
            return False, 0.0
        cap = float(evaluator._non_infra_max_adoption[u, c_pos, period_idx])
        if steps * float(step_u[u]) > cap + _FLOAT_EPS:
            return False, 0.0
        cost += steps * float(step_u[u]) * float(evaluator._non_infra_costs[u, c_pos]) * disc

    if cost > rem_usd + 1e-9:
        return False, 0.0
    return True, cost


def _apply_action_branch(
    state: tuple[int, ...],
    combo: tuple[int, ...],
    b_next: int,
    mult_mode: int,
    period_idx: int,
    n_units: int,
    unready_units: tuple[int, ...],
    max_bin: int,
    step_u: np.ndarray,
    evaluator: Stage3Evaluator,
) -> tuple[int, ...]:
    """Compute successor state tuple for one stochastic uptake branch ``mult_mode``."""
    n_unready = len(unready_units)
    nl = list(state[:n_units])
    ne = list(state[n_units : 2 * n_units])
    nb = list(state[2 * n_units : 3 * n_units])
    n_lpg = list(state[3 * n_units : 3 * n_units + n_unready])
    n_grid = list(state[3 * n_units + n_unready : 3 * n_units + 2 * n_unready])

    lpg_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_lpg)[0])
    bio_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_bio)[0])
    elec_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_elec)[0])
    unready_lookup = {u: idx for idx, u in enumerate(unready_units)}

    for u, act in enumerate(combo):
        if act == ACT_NONE:
            continue
        if act == ACT_INFRA:
            if u in unready_lookup:
                idx = unready_lookup[u]
                n_lpg[idx] = 1
                n_grid[idx] = 1
            continue

        steps = 2 if act == ACT_LPG_2 else 1
        c_pos = (
            lpg_pos
            if act in (ACT_LPG_1, ACT_LPG_2)
            else (elec_pos if act == ACT_ELEC_1 else bio_pos)
        )
        cap_val = float(evaluator._non_infra_max_adoption[u, c_pos, period_idx])
        cap_bins = int(np.floor(cap_val / float(step_u[u]) + _FLOAT_EPS))

        if mult_mode == 0:  # low uptake: floor(0.5 * steps)
            gain = int(np.floor(0.5 * steps))
        elif mult_mode == 1:  # nominal uptake: steps
            gain = steps
        else:  # high uptake: ceil(1.25 * steps), capped at max adoption & ceiling
            gain = min(int(np.ceil(1.25 * steps)), cap_bins)

        raw_u = nl[u] + ne[u] + nb[u]
        gain = max(0, min(gain, cap_bins, max_bin - raw_u))
        if act == ACT_BIOMASS_1:
            nb[u] += gain
        elif act == ACT_ELEC_1:
            ne[u] += gain
        else:
            nl[u] += gain

    return tuple(nl) + tuple(ne) + tuple(nb) + tuple(n_lpg) + tuple(n_grid) + (b_next,)


def build_layered_mdp(
    evaluator: Stage3Evaluator,
    mdp_cfg: MDPConfig,
    *,
    templates: tuple[tuple[int, ...], ...] | None = None,
) -> LayeredMDP:
    """Build forward-reachable finite-horizon MDP with sparse NumPy transition tables."""
    t0 = time.perf_counter()
    n_u = evaluator.n_units
    n_p = evaluator.n_periods
    max_bin = len(mdp_cfg.coverage_bins) - 1
    n_b = mdp_cfg.budget_bins
    budget_step = evaluator.total_budget_usd_millions / float(n_b - 1) if n_b > 1 else 0.0

    headroom = np.maximum(0.0, evaluator._coverage_ceiling - evaluator._cf_bau_2040 - _FLOAT_EPS)
    step_u = headroom / float(max_bin)
    lpg_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_lpg)[0])
    elec_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_elec)[0])
    bio_pos = int(np.flatnonzero(evaluator._non_infra_indices == evaluator._idx_bio)[0])
    lpg_eta = evaluator._non_infra_effectiveness[:, lpg_pos]
    elec_eta = evaluator._non_infra_effectiveness[:, elec_pos]
    bio_eta = evaluator._non_infra_effectiveness[:, bio_pos]

    lpg_init = [int(v >= 1.0) for v in evaluator._lpg_infra_init]
    grid_init = [int(v >= 1.0) for v in evaluator._grid_ready_init]
    unready_units = tuple(u for u in range(n_u) if lpg_init[u] == 0 or grid_init[u] == 0)
    init_lpg_flags = tuple(lpg_init[u] for u in unready_units)
    init_grid_flags = tuple(grid_init[u] for u in unready_units)

    init_b_bin = (n_b - 1) if evaluator.total_budget_usd_millions > 0.0 else 0
    init_state = (
        tuple([0] * n_u)
        + tuple([0] * n_u)
        + tuple([0] * n_u)
        + init_lpg_flags
        + init_grid_flags
        + (init_b_bin,)
    )

    act_templates = templates if templates is not None else build_action_templates(n_u)
    probs = np.array(mdp_cfg.transition_probs, dtype=float)
    unready_lookup = {u: idx for idx, u in enumerate(unready_units)}

    layers: list[list[tuple[int, ...]]] = [[init_state]]
    init_dalys, init_eq = _compute_states_dalys_and_equity(
        [init_state], n_u, step_u, lpg_eta, elec_eta, bio_eta, evaluator
    )
    layer_dalys: list[np.ndarray] = [init_dalys]
    layer_equity_gaps: list[np.ndarray] = [init_eq]
    bau_dalys = float(init_dalys[0])
    bau_eq = float(init_eq[0])
    transitions: list[PeriodTransitions] = []

    for t in range(n_p):
        curr_states = layers[t]
        curr_dalys = layer_dalys[t]
        curr_eq = layer_equity_gaps[t]
        next_map: dict[tuple[int, ...], int] = {}
        next_states: list[tuple[int, ...]] = []

        src_list: list[int] = []
        act_list: list[int] = []
        cost_list: list[float] = []
        next_triples: list[tuple[int, int, int]] = []
        starts = np.zeros(len(curr_states), dtype=np.int32)
        counts = np.zeros(len(curr_states), dtype=np.int32)

        for s_idx, st in enumerate(curr_states):
            starts[s_idx] = len(src_list)
            l_part = st[:n_u]
            e_part = st[n_u : 2 * n_u]
            b_part = st[2 * n_u : 3 * n_u]
            n_unready = len(unready_units)
            lpg_part = st[3 * n_u : 3 * n_u + n_unready]
            grid_part = st[3 * n_u + n_unready : 3 * n_u + 2 * n_unready]
            rem_b = st[-1]

            raws = [l_part[u] + e_part[u] + b_part[u] for u in range(n_u)]
            lpg_ready = [
                lpg_part[unready_lookup[u]] if u in unready_lookup else 1 for u in range(n_u)
            ]
            grid_ready = [
                grid_part[unready_lookup[u]] if u in unready_lookup else 1 for u in range(n_u)
            ]
            rem_usd = float(rem_b) * budget_step

            for a_idx, combo in enumerate(act_templates):
                valid, cost = _is_action_valid_and_cost(
                    combo, raws, lpg_ready, grid_ready, rem_usd, t, max_bin, step_u, evaluator
                )
                if not valid:
                    continue

                if budget_step > 0.0:
                    b_next = max(0, int(np.floor((rem_usd - cost + 1e-9) / budget_step)))
                else:
                    b_next = 0

                branch_idxs: list[int] = []
                for mult_mode in (0, 1, 2):
                    ns = _apply_action_branch(
                        st,
                        combo,
                        b_next,
                        mult_mode,
                        t,
                        n_u,
                        unready_units,
                        max_bin,
                        step_u,
                        evaluator,
                    )
                    if ns not in next_map:
                        next_map[ns] = len(next_states)
                        next_states.append(ns)
                    branch_idxs.append(next_map[ns])

                src_list.append(s_idx)
                act_list.append(a_idx)
                cost_list.append(cost)
                next_triples.append((branch_idxs[0], branch_idxs[1], branch_idxs[2]))

            counts[s_idx] = len(src_list) - starts[s_idx]

        next_dalys, next_eq = _compute_states_dalys_and_equity(
            next_states, n_u, step_u, lpg_eta, elec_eta, bio_eta, evaluator
        )
        layers.append(next_states)
        layer_dalys.append(next_dalys)
        layer_equity_gaps.append(next_eq)

        next_arr = np.asarray(next_triples, dtype=np.int32)
        src_arr = np.asarray(src_list, dtype=np.int32)
        dalys_av = curr_dalys[src_arr][:, np.newaxis] - next_dalys[next_arr]
        eq_red = curr_eq[src_arr][:, np.newaxis] - next_eq[next_arr]

        transitions.append(
            PeriodTransitions(
                src_state=src_arr,
                action_id=np.asarray(act_list, dtype=np.int32),
                cost=np.asarray(cost_list, dtype=float),
                next_state=next_arr,
                dalys_averted=dalys_av,
                equity_gap_reduced=eq_red,
                state_starts=starts,
                state_counts=counts,
            )
        )

    build_s = time.perf_counter() - t0
    return LayeredMDP(
        n_units=n_u,
        n_periods=n_p,
        max_bin=max_bin,
        n_budget_bins=n_b,
        budget_step=budget_step,
        step_u=step_u,
        lpg_eta=lpg_eta,
        elec_eta=elec_eta,
        bio_eta=bio_eta,
        unready_units=unready_units,
        templates=act_templates,
        probs=probs,
        layers=layers,
        layer_dalys=layer_dalys,
        layer_equity_gaps=layer_equity_gaps,
        transitions=transitions,
        bau_dalys=bau_dalys,
        bau_equity_gap=bau_eq,
        build_seconds=float(build_s),
    )


def _segment_max_and_argmax(
    q_vals: np.ndarray,
    action_ids: np.ndarray,
    costs: np.ndarray,
    state_starts: np.ndarray,
    state_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute segment-wise maximum Q-value and argmax action/cost in NumPy."""
    n_states = len(state_starts)
    best_v = np.maximum.reduceat(q_vals, state_starts)
    best_a = np.zeros(n_states, dtype=np.int32)
    best_c = np.zeros(n_states, dtype=float)

    # First occurrence where q_vals equals segment maximum (ties broken by lowest action_id)
    src_expanded = np.repeat(np.arange(n_states, dtype=np.int32), state_counts)
    is_best = q_vals == best_v[src_expanded]
    best_edge_indices = np.flatnonzero(is_best)
    # Because src_expanded is sorted non-decreasing, keeping the first occurrence per state
    # is vectorized via np.unique(..., return_index=True).
    _, first_occ = np.unique(src_expanded[best_edge_indices], return_index=True)
    chosen_edges = best_edge_indices[first_occ]
    best_a[:] = action_ids[chosen_edges]
    best_c[:] = costs[chosen_edges]
    return best_v, best_a, best_c


def solve_finite_horizon_backward_induction(
    mdp: LayeredMDP,
    *,
    cost_weight_lambda: float,
    discount_factor: float,
    equity_weight_lambda: float = 0.0,
) -> FiniteHorizonSolution:
    """Solve finite-horizon MDP by vectorized NumPy backward induction."""
    t0 = time.perf_counter()
    n_p = mdp.n_periods
    values: list[np.ndarray] = [np.zeros(len(mdp.layers[n_p]), dtype=float)]
    policy_acts: list[np.ndarray] = []
    policy_costs: list[np.ndarray] = []

    probs = mdp.probs[np.newaxis, :]
    lam = float(cost_weight_lambda)
    lam_eq = float(equity_weight_lambda)
    gamma = float(discount_factor)

    for t in range(n_p - 1, -1, -1):
        tr = mdp.transitions[t]
        v_next = values[0]
        immediate = tr.dalys_averted - lam * tr.cost[:, np.newaxis] + lam_eq * tr.equity_gap_reduced
        q_vals = np.sum(probs * (immediate + gamma * v_next[tr.next_state]), axis=1)
        v_curr, a_curr, c_curr = _segment_max_and_argmax(
            q_vals, tr.action_id, tr.cost, tr.state_starts, tr.state_counts
        )
        values.insert(0, v_curr)
        policy_acts.insert(0, a_curr)
        policy_costs.insert(0, c_curr)

    solve_s = time.perf_counter() - t0
    return FiniteHorizonSolution(
        cost_weight_lambda=lam,
        discount_factor=gamma,
        values=values,
        policy_action_ids=policy_acts,
        policy_action_costs=policy_costs,
        expected_value_s0=float(values[0][0]),
        solve_seconds=float(solve_s),
        equity_weight_lambda=lam_eq,
    )


def build_stationary_mdp(
    layered: LayeredMDP,
    evaluator: Stage3Evaluator,
) -> StationaryMDP:
    """Build stationary reachable transition graph by merging layered transitions.

    Reuses precomputed sparse transitions from ``layered.transitions`` across
    periods ``0..H-1`` (preserving identical costs, preconditions, and DALY
    transitions) and attaches an absorbing ``no_op`` self-loop for terminal-only
    leaf states so infinite-horizon value iteration runs in milliseconds.
    """
    state_to_id: dict[tuple[int, ...], int] = {}
    states: list[tuple[int, ...]] = []
    for layer in layered.layers:
        for st in layer:
            if st not in state_to_id:
                state_to_id[st] = len(states)
                states.append(st)

    n_real_states = len(states)
    term_gid = n_real_states
    states.append(states[0])  # placeholder tuple for absorbing terminal sink (no-op stop)
    n_states = len(states)
    per_state_edges: list[dict[int, tuple[float, tuple[int, int, int]]]] = [
        {} for _ in range(n_states)
    ]

    for t, tr in enumerate(layered.transitions):
        src_layer = layered.layers[t]
        nxt_layer = layered.layers[t + 1]
        for edge_idx in range(len(tr.src_state)):
            s_gid = state_to_id[src_layer[int(tr.src_state[edge_idx])]]
            a_id = int(tr.action_id[edge_idx])
            if a_id in per_state_edges[s_gid]:
                continue
            if a_id == 0:
                # Stopping / no_op transitions to absorbing zero-reward sink so gamma=1.0
                # never creates a spurious V(s) == V(s) self-loop tie.
                per_state_edges[s_gid][0] = (0.0, (term_gid, term_gid, term_gid))
                continue
            nt = tr.next_state[edge_idx]
            n0 = state_to_id[nxt_layer[int(nt[0])]]
            n1 = state_to_id[nxt_layer[int(nt[1])]]
            n2 = state_to_id[nxt_layer[int(nt[2])]]
            per_state_edges[s_gid][a_id] = (float(tr.cost[edge_idx]), (n0, n1, n2))

    src_list: list[int] = []
    act_list: list[int] = []
    cost_list: list[float] = []
    next_triples: list[tuple[int, int, int]] = []
    starts = np.zeros(n_states, dtype=np.int32)
    counts = np.zeros(n_states, dtype=np.int32)

    for s_gid in range(n_states):
        starts[s_gid] = len(src_list)
        edges = per_state_edges[s_gid]
        if not edges:
            edges[0] = (0.0, (term_gid, term_gid, term_gid))
        for a_id in sorted(edges.keys()):
            cost, nxt_triple = edges[a_id]
            src_list.append(s_gid)
            act_list.append(a_id)
            cost_list.append(cost)
            next_triples.append(nxt_triple)
        counts[s_gid] = len(src_list) - starts[s_gid]

    state_dalys, state_eq = _compute_states_dalys_and_equity(
        states[:-1],
        layered.n_units,
        layered.step_u,
        layered.lpg_eta,
        layered.elec_eta,
        layered.bio_eta,
        evaluator,
    )
    state_dalys = np.append(state_dalys, 0.0)
    state_eq = np.append(state_eq, 0.0)
    src_arr = np.asarray(src_list, dtype=np.int32)
    next_arr = np.asarray(next_triples, dtype=np.int32)
    dalys_av = state_dalys[src_arr][:, np.newaxis] - state_dalys[next_arr]
    eq_red = state_eq[src_arr][:, np.newaxis] - state_eq[next_arr]
    # Transitions into absorbing sink carry zero DALY/equity gain (stop/retire action)
    sink_mask = next_arr == term_gid
    dalys_av = np.where(sink_mask, 0.0, dalys_av)
    eq_red = np.where(sink_mask, 0.0, eq_red)

    return StationaryMDP(
        states=states,
        state_dalys=state_dalys,
        transitions=PeriodTransitions(
            src_state=src_arr,
            action_id=np.asarray(act_list, dtype=np.int32),
            cost=np.asarray(cost_list, dtype=float),
            next_state=next_arr,
            dalys_averted=dalys_av,
            equity_gap_reduced=eq_red,
            state_starts=starts,
            state_counts=counts,
        ),
        templates=layered.templates,
        probs=layered.probs,
    )


def solve_infinite_horizon_value_iteration(
    stat_mdp: StationaryMDP,
    *,
    cost_weight_lambda: float,
    discount_factor: float,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> InfiniteHorizonSolution:
    """Solve infinite-horizon discounted value iteration in vectorized NumPy."""
    t0 = time.perf_counter()
    tr = stat_mdp.transitions
    n_states = len(stat_mdp.states)
    V = np.zeros(n_states, dtype=float)
    probs = stat_mdp.probs[np.newaxis, :]
    lam = float(cost_weight_lambda)
    gamma = float(discount_factor)
    immediate = tr.dalys_averted - lam * tr.cost[:, np.newaxis]

    best_a = np.zeros(n_states, dtype=np.int32)
    best_c = np.zeros(n_states, dtype=float)
    delta = np.inf
    n_iter = 0
    converged = False

    for it in range(1, max_iter + 1):
        n_iter = it
        q_vals = np.sum(probs * (immediate + gamma * V[tr.next_state]), axis=1)
        v_new, best_a, best_c = _segment_max_and_argmax(
            q_vals, tr.action_id, tr.cost, tr.state_starts, tr.state_counts
        )
        delta = float(np.max(np.abs(v_new - V)))
        V = v_new
        if delta < tol:
            converged = True
            break

    solve_s = time.perf_counter() - t0
    return InfiniteHorizonSolution(
        cost_weight_lambda=lam,
        discount_factor=gamma,
        values=V,
        policy_action_ids=best_a,
        policy_action_costs=best_c,
        expected_value_s0=float(V[0]),
        n_iterations=n_iter,
        final_delta=delta,
        converged=converged,
        solve_seconds=float(solve_s),
    )


def simulate_mdp_rollouts(
    mdp: LayeredMDP,
    solution: FiniteHorizonSolution,
    evaluator: Stage3Evaluator,
    *,
    n_rollouts: int,
    seed: int,
    precomputed_branches: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate ``n_rollouts`` stochastic trajectories and verify via ``Stage3Evaluator``."""
    rng = np.random.default_rng(int(seed))
    n_u = mdp.n_units
    n_p = mdp.n_periods
    n_i = evaluator.n_interventions

    if precomputed_branches is not None:
        if precomputed_branches.shape != (n_rollouts, n_p):
            raise ValueError(
                f"Expected precomputed_branches shape ({n_rollouts}, {n_p}), "
                f"got {precomputed_branches.shape}."
            )
        branches_mat = precomputed_branches
    else:
        cum_probs = np.cumsum(mdp.probs)
        u_draws = rng.random((n_rollouts, n_p))
        branches_mat = np.minimum(
            np.searchsorted(cum_probs, u_draws, side="right"),
            len(mdp.probs) - 1,
        )

    # Map each (t, s_idx) -> lookup dict of {action_id: (cost, next_triple)}
    trans_lookup: list[list[dict[int, tuple[float, np.ndarray]]]] = []
    for t in range(n_p):
        tr = mdp.transitions[t]
        per_state: list[dict[int, tuple[float, np.ndarray]]] = [
            {} for _ in range(len(mdp.layers[t]))
        ]
        for edge_idx in range(len(tr.src_state)):
            s_i = int(tr.src_state[edge_idx])
            a_i = int(tr.action_id[edge_idx])
            per_state[s_i][a_i] = (float(tr.cost[edge_idx]), tr.next_state[edge_idx])
        trans_lookup.append(per_state)

    rollout_vectors = np.zeros((n_rollouts, evaluator.n_vars), dtype=float)
    action_records: list[list[str]] = []
    discounted_returns = np.zeros(n_rollouts, dtype=float)

    for r_idx in range(n_rollouts):
        s_idx = 0
        tensor = np.zeros((n_u, n_i, n_p), dtype=float)
        labels: list[str] = []
        disc_ret = 0.0

        for t in range(n_p):
            st_curr = mdp.layers[t][s_idx]
            a_id = int(solution.policy_action_ids[t][s_idx])
            combo = mdp.templates[a_id]
            labels.append(format_action_label(combo, evaluator.units))
            cost, next_triple = trans_lookup[t][s_idx][a_id]

            branch = int(branches_mat[r_idx, t])
            ns_idx = int(next_triple[branch])
            st_next = mdp.layers[t + 1][ns_idx]

            da = float(mdp.layer_dalys[t][s_idx] - mdp.layer_dalys[t + 1][ns_idx])
            eq_red = float(mdp.layer_equity_gaps[t][s_idx] - mdp.layer_equity_gaps[t + 1][ns_idx])
            disc_ret += (solution.discount_factor**t) * (
                da - solution.cost_weight_lambda * cost + solution.equity_weight_lambda * eq_red
            )

            # Record realized decision vector increment in period t
            for u, act in enumerate(combo):
                if act == ACT_NONE:
                    continue
                if act == ACT_INFRA:
                    tensor[u, evaluator._idx_infra, t] = evaluator.infra_build_threshold
                elif act == ACT_BIOMASS_1:
                    delta_bins = st_next[2 * n_u + u] - st_curr[2 * n_u + u]
                    tensor[u, evaluator._idx_bio, t] = float(delta_bins) * float(mdp.step_u[u])
                elif act in (ACT_LPG_1, ACT_LPG_2):
                    delta_bins = st_next[u] - st_curr[u]
                    tensor[u, evaluator._idx_lpg, t] = float(delta_bins) * float(mdp.step_u[u])
                elif act == ACT_ELEC_1:
                    delta_bins = st_next[n_u + u] - st_curr[n_u + u]
                    tensor[u, evaluator._idx_elec, t] = float(delta_bins) * float(mdp.step_u[u])

            s_idx = ns_idx

        rollout_vectors[r_idx] = tensor.reshape(-1)
        action_records.append(labels)
        discounted_returns[r_idx] = disc_ret

    ev_res = evaluator.evaluate(rollout_vectors)
    dalys_mean = np.asarray(ev_res.dalys_2040_mean, dtype=float)
    dalys_low = np.asarray(ev_res.dalys_2040_lower, dtype=float)
    dalys_up = np.asarray(ev_res.dalys_2040_upper, dtype=float)
    total_cost = np.asarray(ev_res.total_cost_usd_millions, dtype=float)
    eq_gap = np.asarray(ev_res.equity_gap_asr, dtype=float)
    is_feas = np.asarray(ev_res.is_feasible, dtype=bool)

    df = pd.DataFrame(
        {
            "rollout_id": np.arange(n_rollouts, dtype=int),
            "dalys_2040_mean": dalys_mean,
            "dalys_2040_lower": dalys_low,
            "dalys_2040_upper": dalys_up,
            "dalys_averted_mean": mdp.bau_dalys - dalys_mean,
            "total_cost_usd_millions": total_cost,
            "equity_gap_asr": eq_gap,
            "discounted_return": discounted_returns,
            "is_feasible_in_evaluator": is_feas,
            "p1_action": [rec[0] for rec in action_records],
            "p2_action": [rec[1] if len(rec) > 1 else "" for rec in action_records],
            "p3_action": [rec[2] if len(rec) > 2 else "" for rec in action_records],
            "is_mock": evaluator.is_mock,
            "warning_mock_inputs": evaluator.warning_mock_inputs,
            "data_version": evaluator.data_version,
        }
    )
    return df


def policy_table_to_dataframe(
    mdp: LayeredMDP,
    solution: FiniteHorizonSolution,
    evaluator: Stage3Evaluator,
) -> pd.DataFrame:
    """Export optimal state x period -> action policy table as a DataFrame."""
    rows: list[dict[str, Any]] = []
    n_u = mdp.n_units
    unready_lookup = {u: idx for idx, u in enumerate(mdp.unready_units)}

    for t in range(mdp.n_periods):
        period_id = evaluator.periods[t]
        for s_idx, st in enumerate(mdp.layers[t]):
            a_id = int(solution.policy_action_ids[t][s_idx])
            combo = mdp.templates[a_id]
            row: dict[str, Any] = {
                "period_idx": t,
                "period": period_id,
                "state_id": s_idx,
                "budget_bin": int(st[-1]),
                "remaining_budget_tier_usd_millions": float(st[-1]) * mdp.budget_step,
                "state_dalys_2040_mean": float(mdp.layer_dalys[t][s_idx]),
                "state_value": float(solution.values[t][s_idx]),
                "action_id": a_id,
                "action_label": format_action_label(combo, evaluator.units),
                "action_cost_usd_millions": float(solution.policy_action_costs[t][s_idx]),
            }
            for u_idx, unit in enumerate(evaluator.units):
                l_b = int(st[u_idx])
                e_b = int(st[n_u + u_idx])
                b_b = int(st[2 * n_u + u_idx])
                raw_cov = float(evaluator._cf_bau_2040[u_idx]) + (l_b + e_b + b_b) * float(
                    mdp.step_u[u_idx]
                )
                eff_gain = (
                    l_b * float(mdp.lpg_eta[u_idx])
                    + e_b * float(mdp.elec_eta[u_idx])
                    + b_b * float(mdp.bio_eta[u_idx])
                ) * float(mdp.step_u[u_idx])
                lpg_r = bool(st[3 * n_u + unready_lookup[u_idx]] if u_idx in unready_lookup else 1)
                row[f"{unit}__raw_cov"] = raw_cov
                row[f"{unit}__eff_gain"] = eff_gain
                row[f"{unit}__infra_ready"] = lpg_r

            row["is_mock"] = evaluator.is_mock
            row["warning_mock_inputs"] = evaluator.warning_mock_inputs
            row["data_version"] = evaluator.data_version
            rows.append(row)

    return pd.DataFrame(rows)


def calibrate_lambda_from_pareto(pareto_csv_path: Path) -> dict[str, float]:
    """Derive Stage 3 Pareto DALYs-per-$1M trade-off slopes for ``cost_weight_lambda``."""
    df = pd.read_csv(pareto_csv_path).sort_values("total_cost_usd_millions")
    d_cost = float(df["total_cost_usd_millions"].iloc[-1] - df["total_cost_usd_millions"].iloc[0])
    d_dalys = float(df["dalys_2040_mean"].iloc[0] - df["dalys_2040_mean"].iloc[-1])
    overall_slope = d_dalys / max(d_cost, 1e-9)

    q_idxs = np.linspace(0, len(df) - 1, 6, dtype=int)
    pts = df.iloc[q_idxs]
    seg_slopes: list[float] = []
    for idx in range(len(pts) - 1):
        dc = float(
            pts["total_cost_usd_millions"].iloc[idx + 1] - pts["total_cost_usd_millions"].iloc[idx]
        )
        dd = float(pts["dalys_2040_mean"].iloc[idx] - pts["dalys_2040_mean"].iloc[idx + 1])
        if dc > 1e-9:
            seg_slopes.append(dd / dc)

    return {
        "overall_pareto_slope_dalys_per_usd_m": overall_slope,
        "median_quantile_slope_dalys_per_usd_m": float(np.median(seg_slopes))
        if seg_slopes
        else overall_slope,
        "upper_segment_slope_dalys_per_usd_m": seg_slopes[-1] if seg_slopes else overall_slope,
    }


def run_sensitivity_sweep(
    mdp: LayeredMDP,
    stat_mdp: StationaryMDP,
    evaluator: Stage3Evaluator,
    mdp_cfg: MDPConfig,
) -> pd.DataFrame:
    """Evaluate sensitivity over ``lambda`` and ``discount_factor`` plus finite vs infinite VI."""
    rows: list[dict[str, Any]] = []
    for gamma in mdp_cfg.discount_sensitivity:
        for lam in mdp_cfg.lambda_sensitivity:
            fh = solve_finite_horizon_backward_induction(
                mdp, cost_weight_lambda=lam, discount_factor=gamma
            )
            rollouts = simulate_mdp_rollouts(
                mdp,
                fh,
                evaluator,
                n_rollouts=min(mdp_cfg.n_rollouts, 300),
                seed=mdp_cfg.seed,
            )
            ih = solve_infinite_horizon_value_iteration(
                stat_mdp,
                cost_weight_lambda=lam,
                discount_factor=gamma,
                tol=mdp_cfg.vi_tol,
                max_iter=mdp_cfg.vi_max_iter,
            )
            p1_act = format_action_label(
                mdp.templates[int(fh.policy_action_ids[0][0])], evaluator.units
            )
            p1_inf_act = format_action_label(
                stat_mdp.templates[int(ih.policy_action_ids[0])], evaluator.units
            )
            rows.append(
                {
                    "discount_factor": float(gamma),
                    "cost_weight_lambda": float(lam),
                    "fh_expected_value_s0": fh.expected_value_s0,
                    "ih_expected_value_s0": ih.expected_value_s0,
                    "ih_iterations": ih.n_iterations,
                    "ih_converged": ih.converged,
                    "ih_final_delta": ih.final_delta,
                    "mean_dalys_2040": float(rollouts["dalys_2040_mean"].mean()),
                    "mean_dalys_averted": float(rollouts["dalys_averted_mean"].mean()),
                    "mean_cost_usd_millions": float(rollouts["total_cost_usd_millions"].mean()),
                    "mean_equity_gap_asr": float(rollouts["equity_gap_asr"].mean()),
                    "all_rollouts_feasible": bool(rollouts["is_feasible_in_evaluator"].all()),
                    "fh_p1_action_s0": p1_act,
                    "ih_p1_action_s0": p1_inf_act,
                    "is_mock": evaluator.is_mock,
                    "warning_mock_inputs": evaluator.warning_mock_inputs,
                    "data_version": evaluator.data_version,
                }
            )
    return pd.DataFrame(rows)


def _artifact_path(directory: Path, stem: str, ext: str, *, is_mock: bool) -> Path:
    suffix = "_mock" if is_mock else ""
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    return directory / f"{stem}{suffix}{clean_ext}"


def _write_sidecar(
    target_path: Path,
    evaluator: Stage3Evaluator,
    mdp_cfg: MDPConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    meta_path = Path(f"{target_path}.meta.json")
    payload: dict[str, Any] = {
        "artifact": target_path.name,
        "source": "src.planning.mdp",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_version": evaluator.data_version,
        "is_mock": evaluator.is_mock,
        "warning_mock_inputs": evaluator.warning_mock_inputs,
        "coverage_bins": list(mdp_cfg.coverage_bins),
        "budget_bins": mdp_cfg.budget_bins,
        "discount_factor": mdp_cfg.discount_factor,
        "cost_weight_lambda": mdp_cfg.cost_weight_lambda,
        "transition_probs": list(mdp_cfg.transition_probs),
        "seed": mdp_cfg.seed,
    }
    if extra:
        payload.update(extra)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def run_mdp_pipeline(
    config_path: str | Path,
    base_dir: str | Path | None = None,
    *,
    generate_figure: bool = True,
) -> dict[str, Any]:
    """Run full Stage 4 MDP build, solve, rollout simulation, sensitivity, and outputs."""
    t_start = time.perf_counter()
    root = Path(base_dir) if base_dir is not None else Path(".")
    config = load_config(config_path)
    contracts = load_stage3_contracts(config, base_dir=root)
    evaluator = Stage3Evaluator(config=config, contracts=contracts)
    mdp_cfg = MDPConfig.from_config(config)

    tables_dir = root / config["paths"]["outputs_tables_dir"]
    figures_dir = root / config["paths"]["outputs_figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    mdp = build_layered_mdp(evaluator, mdp_cfg)
    fh_sol = solve_finite_horizon_backward_induction(
        mdp,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    stat_mdp = build_stationary_mdp(mdp, evaluator)
    ih_sol = solve_infinite_horizon_value_iteration(
        stat_mdp,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
        tol=mdp_cfg.vi_tol,
        max_iter=mdp_cfg.vi_max_iter,
    )

    policy_df = policy_table_to_dataframe(mdp, fh_sol, evaluator)
    rollouts_df = simulate_mdp_rollouts(
        mdp,
        fh_sol,
        evaluator,
        n_rollouts=mdp_cfg.n_rollouts,
        seed=mdp_cfg.seed,
    )
    sensitivity_df = run_sensitivity_sweep(mdp, stat_mdp, evaluator, mdp_cfg)

    is_mock = evaluator.is_mock
    policy_path = _artifact_path(tables_dir, "plan_mdp_policy", ".csv", is_mock=is_mock)
    rollouts_path = _artifact_path(tables_dir, "plan_mdp_rollouts", ".csv", is_mock=is_mock)
    sens_path = _artifact_path(tables_dir, "plan_mdp_sensitivity", ".csv", is_mock=is_mock)

    policy_df.to_csv(policy_path, index=False)
    rollouts_df.to_csv(rollouts_path, index=False)
    sensitivity_df.to_csv(sens_path, index=False)

    layer_sizes = [len(L) for L in mdp.layers]
    extra_meta = {
        "layer_state_counts": layer_sizes,
        "n_decision_states": sum(layer_sizes[:-1]),
        "n_macro_actions": len(mdp.templates),
        "fh_expected_value_s0": fh_sol.expected_value_s0,
        "ih_expected_value_s0": ih_sol.expected_value_s0,
        "ih_iterations": ih_sol.n_iterations,
        "ih_converged": ih_sol.converged,
    }
    policy_meta = _write_sidecar(policy_path, evaluator, mdp_cfg, extra_meta)
    rollouts_meta = _write_sidecar(rollouts_path, evaluator, mdp_cfg, extra_meta)
    sens_meta = _write_sidecar(sens_path, evaluator, mdp_cfg, extra_meta)

    files: dict[str, Path] = {
        "policy_csv": policy_path,
        "policy_meta": policy_meta,
        "rollouts_csv": rollouts_path,
        "rollouts_meta": rollouts_meta,
        "sensitivity_csv": sens_path,
        "sensitivity_meta": sens_meta,
    }

    if generate_figure:
        from src.viz.decision import plot_mdp_policy_and_sensitivity

        fig_path = _artifact_path(
            figures_dir, "plan_mdp_policy_sensitivity", ".png", is_mock=is_mock
        )
        plot_mdp_policy_and_sensitivity(
            sensitivity_df,
            rollouts_df,
            mdp_cfg.cost_weight_lambda,
            mdp_cfg.discount_factor,
            fig_path,
            is_mock=is_mock,
            data_version=evaluator.data_version,
        )
        fig_meta = _write_sidecar(fig_path, evaluator, mdp_cfg, extra_meta)
        files["fig_mdp"] = fig_path
        files["fig_mdp_meta"] = fig_meta

    total_runtime = time.perf_counter() - t_start
    return {
        "is_mock": is_mock,
        "data_version": evaluator.data_version,
        "runtime_seconds": float(total_runtime),
        "mdp": mdp,
        "fh_solution": fh_sol,
        "ih_solution": ih_sol,
        "policy_df": policy_df,
        "rollouts_df": rollouts_df,
        "sensitivity_df": sensitivity_df,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``make mdp`` (``python -m src.planning.mdp``)."""
    parser = argparse.ArgumentParser(description="Run Stage 4 custom NumPy MDP solver.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args(argv)

    out = run_mdp_pipeline(args.config)
    mdp: LayeredMDP = out["mdp"]
    fh: FiniteHorizonSolution = out["fh_solution"]
    ih: InfiniteHorizonSolution = out["ih_solution"]
    roll_df: pd.DataFrame = out["rollouts_df"]

    mock_tag = " [MOCK DATA]" if out["is_mock"] else ""
    print(f"MDP complete{mock_tag} in {out['runtime_seconds']:.2f}s")
    print(
        f"State layers: {[len(L) for L in mdp.layers]} | "
        f"Macro-actions: {len(mdp.templates)} | "
        f"Build: {mdp.build_seconds:.2f}s | Solve: {fh.solve_seconds:.4f}s"
    )
    print(
        f"Finite-horizon E[V0]={fh.expected_value_s0:,.1f} | "
        f"Infinite-horizon VI E[V0]={ih.expected_value_s0:,.1f} "
        f"(iters={ih.n_iterations}, converged={ih.converged})"
    )
    print(
        f"Rollouts (N={len(roll_df)}): "
        f"mean DALYs averted={roll_df['dalys_averted_mean'].mean():,.0f}, "
        f"mean cost=${roll_df['total_cost_usd_millions'].mean():.1f}M, "
        f"feasible={bool(roll_df['is_feasible_in_evaluator'].all())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
