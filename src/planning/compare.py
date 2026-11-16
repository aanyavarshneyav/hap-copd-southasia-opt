"""Equal-footing GA vs. MDP vs. PDDL stochastic rollout comparator and sensitivity suite.

Evaluates open-loop NSGA-II plans (knee, extremes, best-scalarized, and Pareto
samples), adaptive custom NumPy MDP policies, and the STRIPS/PDDL plan inside the
identical stochastic uptake transition simulator (Common Random Numbers across
``N >= 1000`` seeded rollouts), and runs sensitivity sweeps over ``budget_bins``
(6 vs. 11), equity penalty ``lambda_eq > 0``, and discount factor ``gamma``
(0.774 vs. 1.0).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import load_config, load_stage3_contracts
from src.optimise.evaluator import Stage3Evaluator
from src.optimise.nsga2 import run_nsga2_multi_seed, select_representative_plans
from src.planning.mdp import (
    MDPConfig,
    build_layered_mdp,
    simulate_mdp_rollouts,
    solve_finite_horizon_backward_induction,
)
from src.planning.pddl_runner import plan_to_decision_vector, solve_pddl

_DEFAULT_SHOCK_MULTS: tuple[float, float, float] = (0.5, 1.0, 1.25)


def draw_shock_branches(
    probs: tuple[float, float, float],
    n_rollouts: int,
    n_periods: int,
    seed: int,
) -> np.ndarray:
    """Draw ``(n_rollouts, n_periods)`` Common-Random-Numbers shock indices ``0/1/2``."""
    rng = np.random.default_rng(int(seed))
    cum_probs = np.cumsum(np.asarray(probs, dtype=float))
    draws = rng.random((int(n_rollouts), int(n_periods)))
    return np.minimum(np.searchsorted(cum_probs, draws, side="right"), len(probs) - 1)


def simulate_open_loop_schedule(
    x_plan: np.ndarray,
    evaluator: Stage3Evaluator,
    branches_mat: np.ndarray,
    *,
    shock_multipliers: tuple[float, float, float] = _DEFAULT_SHOCK_MULTS,
    cost_weight_lambda: float = 3500.0,
    discount_factor: float = 0.774,
    equity_weight_lambda: float = 0.0,
) -> pd.DataFrame:
    """Simulate an open-loop ``(48,)`` schedule across stochastic uptake shocks.

    Under zero stochasticity (``branches_mat == 1`` everywhere, i.e. nominal
    multiplier ``1.0``), realized vectors match ``x_plan`` and every metric
    matches ``Stage3Evaluator.evaluate(x_plan)`` identically.
    """
    n_rollouts, n_periods = branches_mat.shape
    if n_periods != evaluator.n_periods:
        raise ValueError(
            f"Expected {evaluator.n_periods} periods in branches_mat, got {n_periods}."
        )

    plan_tensor = np.asarray(x_plan, dtype=float).reshape(
        evaluator.n_units, evaluator.n_interventions, evaluator.n_periods
    )
    mult_arr = np.asarray(shock_multipliers, dtype=float)[branches_mat]  # (N, P)

    # Realized decision tensors: infra is deterministic; cooking scales by period shock
    realized = np.repeat(plan_tensor[np.newaxis, :, :, :], n_rollouts, axis=0)
    cook_idx = evaluator._non_infra_indices
    realized[:, :, cook_idx, :] *= mult_arr[:, np.newaxis, np.newaxis, :]
    realized_vecs = realized.reshape(n_rollouts, evaluator.n_vars)

    ev_res = evaluator.evaluate(realized_vecs)

    # Committed period costs from planned schedule (equal-footing open-loop spend)
    disc = evaluator._period_discount
    cook_cost_t = np.sum(
        plan_tensor[:, cook_idx, :]
        * evaluator._non_infra_costs[:, :, np.newaxis]
        * disc[np.newaxis, np.newaxis, :],
        axis=(0, 1),
    )
    infra_built_plan = (
        plan_tensor[:, evaluator._idx_infra, :] >= evaluator.infra_build_threshold
    ).astype(float)
    infra_cost_t = np.sum(
        infra_built_plan * evaluator._infra_costs[:, np.newaxis] * disc[np.newaxis, :],
        axis=0,
    )
    planned_cost_t = cook_cost_t + infra_cost_t  # (P,)
    planned_total_cost = float(np.sum(planned_cost_t))

    # Period-by-period incremental DALYs averted & equity gap reduction for discounted return
    eff_per_period = np.sum(
        realized[:, :, cook_idx, :]
        * evaluator._non_infra_effectiveness[np.newaxis, :, :, np.newaxis],
        axis=2,
    )  # (N, U, P)
    cum_eff = np.cumsum(eff_per_period, axis=2)  # (N, U, P)

    bau_d, _, _, bau_eq, _ = evaluator.evaluate_effective_gains(
        np.zeros((1, evaluator.n_units), dtype=float)
    )
    prev_dalys = np.full(n_rollouts, float(bau_d[0]), dtype=float)
    prev_eq = np.full(n_rollouts, float(bau_eq[0]), dtype=float)
    discounted_returns = np.zeros(n_rollouts, dtype=float)

    for t in range(n_periods):
        d_t, _, _, eq_t, _ = evaluator.evaluate_effective_gains(cum_eff[:, :, t])
        da_t = prev_dalys - d_t
        eq_red_t = prev_eq - eq_t
        discounted_returns += (discount_factor**t) * (
            da_t - cost_weight_lambda * float(planned_cost_t[t]) + equity_weight_lambda * eq_red_t
        )
        prev_dalys = d_t
        prev_eq = eq_t

    dalys_mean = np.asarray(ev_res.dalys_2040_mean, dtype=float)
    dalys_low = np.asarray(ev_res.dalys_2040_lower, dtype=float)
    dalys_up = np.asarray(ev_res.dalys_2040_upper, dtype=float)
    eq_gap = np.asarray(ev_res.equity_gap_asr, dtype=float)
    is_feas = np.asarray(ev_res.is_feasible, dtype=bool)

    return pd.DataFrame(
        {
            "rollout_id": np.arange(n_rollouts, dtype=int),
            "dalys_2040_mean": dalys_mean,
            "dalys_2040_lower": dalys_low,
            "dalys_2040_upper": dalys_up,
            "dalys_averted_mean": float(bau_d[0]) - dalys_mean,
            "total_cost_usd_millions": np.full(n_rollouts, planned_total_cost, dtype=float),
            "realized_cost_if_pay_as_go": np.asarray(ev_res.total_cost_usd_millions, dtype=float),
            "equity_gap_asr": eq_gap,
            "discounted_return": discounted_returns,
            "is_feasible_in_evaluator": is_feas,
            "p1_branch": branches_mat[:, 0].astype(int),
        }
    )


def summarize_rollouts(
    method: str,
    paradigm: str,
    roll_df: pd.DataFrame,
    branches_mat: np.ndarray,
    evaluator: Stage3Evaluator,
    *,
    cost_weight_lambda: float,
    discount_factor: float,
    equity_weight_lambda: float = 0.0,
) -> dict[str, Any]:
    """Summarize rollout distributions, tail percentiles, low-uptake performance, and violations."""
    dalys = roll_df["dalys_2040_mean"].to_numpy(dtype=float)
    averted = roll_df["dalys_averted_mean"].to_numpy(dtype=float)
    cost = roll_df["total_cost_usd_millions"].to_numpy(dtype=float)
    eq = roll_df["equity_gap_asr"].to_numpy(dtype=float)
    ret = roll_df["discounted_return"].to_numpy(dtype=float)
    feas = roll_df["is_feasible_in_evaluator"].to_numpy(dtype=bool)

    low_p1_mask = branches_mat[:, 0] == 0
    nom_high_p1_mask = branches_mat[:, 0] >= 1

    return {
        "method": method,
        "paradigm": paradigm,
        "n_rollouts": int(len(roll_df)),
        "cost_weight_lambda": float(cost_weight_lambda),
        "discount_factor": float(discount_factor),
        "equity_weight_lambda": float(equity_weight_lambda),
        "mean_dalys_2040": float(np.mean(dalys)),
        "sd_dalys_2040": float(np.std(dalys, ddof=1)) if len(dalys) > 1 else 0.0,
        "p05_dalys_2040": float(np.percentile(dalys, 5)),
        "p95_dalys_2040": float(np.percentile(dalys, 95)),
        "mean_dalys_averted": float(np.mean(averted)),
        "mean_cost_usd_millions": float(np.mean(cost)),
        "mean_equity_gap_asr": float(np.mean(eq)),
        "sd_equity_gap_asr": float(np.std(eq, ddof=1)) if len(eq) > 1 else 0.0,
        "mean_scalarized_value": float(np.mean(ret)),
        "sd_scalarized_value": float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0,
        "p05_scalarized_value": float(np.percentile(ret, 5)),
        "p95_scalarized_value": float(np.percentile(ret, 95)),
        "constraint_violation_rate": float(1.0 - np.mean(feas)),
        "mean_dalys_averted_low_p1": float(np.mean(averted[low_p1_mask]))
        if np.any(low_p1_mask)
        else float("nan"),
        "mean_scalarized_low_p1": float(np.mean(ret[low_p1_mask]))
        if np.any(low_p1_mask)
        else float("nan"),
        "mean_scalarized_nom_high_p1": float(np.mean(ret[nom_high_p1_mask]))
        if np.any(nom_high_p1_mask)
        else float("nan"),
        "is_mock": evaluator.is_mock,
        "warning_mock_inputs": evaluator.warning_mock_inputs,
        "data_version": evaluator.data_version,
    }


def _load_or_compute_stage3_fronts(
    config: dict[str, Any],
    evaluator: Stage3Evaluator,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = "_mock" if evaluator.is_mock else ""
    pareto_path = tables_dir / f"s3_pareto_front{suffix}.csv"
    selected_path = tables_dir / f"s3_selected_plans{suffix}.csv"

    if pareto_path.exists() and selected_path.exists():
        return pd.read_csv(pareto_path), pd.read_csv(selected_path)

    multi = run_nsga2_multi_seed(evaluator, config)
    sel_df, sel_idx = select_representative_plans(
        multi.combined_X, multi.combined_evaluation, evaluator
    )
    from src.optimise.nsga2 import pareto_front_to_dataframe

    par_df = pareto_front_to_dataframe(multi, evaluator, selected_indices=sel_idx)
    return par_df, sel_df


def _pick_best_ga_for_scalarization(
    pareto_df: pd.DataFrame,
    evaluator: Stage3Evaluator,
    branches_mat: np.ndarray,
    shock_multipliers: tuple[float, float, float],
    cost_weight_lambda: float,
    discount_factor: float,
    equity_weight_lambda: float = 0.0,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    """Identify the GA Pareto solution with highest expected scalarized value."""
    var_cols = list(evaluator.variable_names)
    X_all = pareto_df[var_cols].to_numpy(dtype=float)

    # Fast nominal pre-screen across all Pareto points, then exact stochastic rollout on top-15
    bau_d, _, _, bau_eq, _ = evaluator.evaluate_effective_gains(
        np.zeros((1, evaluator.n_units), dtype=float)
    )
    dalys = pareto_df["dalys_2040_mean"].to_numpy(dtype=float)
    costs = pareto_df["total_cost_usd_millions"].to_numpy(dtype=float)
    eqs = pareto_df["equity_gap_asr"].to_numpy(dtype=float)
    approx_scores = (
        (float(bau_d[0]) - dalys)
        - cost_weight_lambda * costs
        + equity_weight_lambda * (float(bau_eq[0]) - eqs)
    )
    top_k = np.argsort(approx_scores)[::-1][: min(15, len(pareto_df))]

    best_idx = int(top_k[0])
    best_mean_ret = -np.inf
    best_roll_df: pd.DataFrame | None = None

    for cand_idx in top_k:
        roll = simulate_open_loop_schedule(
            X_all[cand_idx],
            evaluator,
            branches_mat,
            shock_multipliers=shock_multipliers,
            cost_weight_lambda=cost_weight_lambda,
            discount_factor=discount_factor,
            equity_weight_lambda=equity_weight_lambda,
        )
        mean_ret = float(roll["discounted_return"].mean())
        if mean_ret > best_mean_ret:
            best_mean_ret = mean_ret
            best_idx = int(cand_idx)
            best_roll_df = roll

    assert best_roll_df is not None
    return best_idx, X_all[best_idx], best_roll_df


def run_comparison_pipeline(
    config_path: str | Path,
    base_dir: str | Path | None = None,
    *,
    generate_figures: bool = True,
) -> dict[str, Any]:
    """Run equal-footing GA vs. MDP vs. PDDL comparison and sensitivity suite."""
    t_start = time.perf_counter()
    root = Path(base_dir) if base_dir is not None else Path(".")
    config = load_config(config_path)
    contracts = load_stage3_contracts(config, base_dir=root)
    evaluator = Stage3Evaluator(config=config, contracts=contracts)
    mdp_cfg = MDPConfig.from_config(config)

    comp_cfg = config.get("planning", {}).get("compare", {})
    n_rollouts = int(comp_cfg.get("n_rollouts", 1000))
    mult_raw = comp_cfg.get("shock_multipliers", list(_DEFAULT_SHOCK_MULTS))
    shock_multipliers = (float(mult_raw[0]), float(mult_raw[1]), float(mult_raw[2]))
    n_pareto_sample = int(comp_cfg.get("n_pareto_sample", 4))
    budget_bins_list = [int(b) for b in comp_cfg.get("budget_bins_sensitivity", [6, 11])]
    lambda_eq_list = [
        float(v)
        for v in comp_cfg.get("lambda_eq_sensitivity", [0.0, 2500.0, 10000.0, 25000.0, 50000.0])
    ]

    tables_dir = root / config["paths"]["outputs_tables_dir"]
    figures_dir = root / config["paths"]["outputs_figures_dir"]
    pddl_dir = root / config["paths"]["pddl_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Shared Common Random Numbers shock matrix across all methods
    branches_mat = draw_shock_branches(
        mdp_cfg.transition_probs, n_rollouts, evaluator.n_periods, mdp_cfg.seed
    )

    # 1. Load Stage 3 Pareto & selected plans
    pareto_df, selected_df = _load_or_compute_stage3_fronts(config, evaluator, tables_dir)
    var_cols = list(evaluator.variable_names)

    # 2. Build MDPs (budget_bins=6 baseline and budget_bins=11 sensitivity)
    mdp_by_bins: dict[int, Any] = {}
    for b_bins in sorted(set(budget_bins_list + [mdp_cfg.budget_bins])):
        cfg_b = replace(mdp_cfg, budget_bins=b_bins)
        mdp_by_bins[b_bins] = build_layered_mdp(evaluator, cfg_b)

    mdp_b6 = mdp_by_bins[6]
    sol_mdp_b6 = solve_finite_horizon_backward_induction(
        mdp_b6,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    roll_mdp_b6 = simulate_mdp_rollouts(
        mdp_b6,
        sol_mdp_b6,
        evaluator,
        n_rollouts=n_rollouts,
        seed=mdp_cfg.seed,
        precomputed_branches=branches_mat,
    )

    mdp_b11 = mdp_by_bins[11] if 11 in mdp_by_bins else mdp_b6
    sol_mdp_b11 = solve_finite_horizon_backward_induction(
        mdp_b11,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    roll_mdp_b11 = simulate_mdp_rollouts(
        mdp_b11,
        sol_mdp_b11,
        evaluator,
        n_rollouts=n_rollouts,
        seed=mdp_cfg.seed,
        precomputed_branches=branches_mat,
    )

    # 3. Matched GA best-scalarized plan + 4 selected plans + Pareto quantiles
    best_ga_idx, _, roll_ga_best = _pick_best_ga_for_scalarization(
        pareto_df,
        evaluator,
        branches_mat,
        shock_multipliers,
        mdp_cfg.cost_weight_lambda,
        mdp_cfg.discount_factor,
    )

    rollout_map: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    def _record(method_name: str, paradigm: str, rdf: pd.DataFrame) -> None:
        rollout_map[method_name] = rdf
        summary_rows.append(
            summarize_rollouts(
                method_name,
                paradigm,
                rdf,
                branches_mat,
                evaluator,
                cost_weight_lambda=mdp_cfg.cost_weight_lambda,
                discount_factor=mdp_cfg.discount_factor,
            )
        )

    _record("mdp_adaptive_b6", "MDP (adaptive, b=6)", roll_mdp_b6)
    if 11 in mdp_by_bins:
        _record("mdp_adaptive_b11", "MDP (adaptive, b=11)", roll_mdp_b11)
    _record(f"ga_matched_best (id={best_ga_idx})", "GA (open-loop)", roll_ga_best)

    for _, row in selected_df.iterrows():
        p_name = str(row["plan_name"])
        x_vec = row[var_cols].to_numpy(dtype=float)
        rdf = simulate_open_loop_schedule(
            x_vec,
            evaluator,
            branches_mat,
            shock_multipliers=shock_multipliers,
            cost_weight_lambda=mdp_cfg.cost_weight_lambda,
            discount_factor=mdp_cfg.discount_factor,
        )
        _record(f"ga_{p_name}", "GA (open-loop)", rdf)

    # Sample evenly spaced Pareto front points (excluding already-selected solution_ids)
    selected_ids = set(int(v) for v in selected_df["solution_id"].tolist()) | {best_ga_idx}
    pareto_sorted = pareto_df.sort_values("total_cost_usd_millions").reset_index(drop=True)
    if n_pareto_sample > 0 and len(pareto_sorted) > 4:
        q_idxs = np.linspace(0.15, 0.85, n_pareto_sample)
        sample_positions = [int(round(q * (len(pareto_sorted) - 1))) for q in q_idxs]
        for s_num, pos in enumerate(dict.fromkeys(sample_positions), start=1):
            sol_id = int(pareto_sorted.loc[pos, "solution_id"])
            if sol_id in selected_ids:
                continue
            x_vec = pareto_sorted.loc[pos, var_cols].to_numpy(dtype=float)
            rdf = simulate_open_loop_schedule(
                x_vec,
                evaluator,
                branches_mat,
                shock_multipliers=shock_multipliers,
                cost_weight_lambda=mdp_cfg.cost_weight_lambda,
                discount_factor=mdp_cfg.discount_factor,
            )
            _record(f"ga_pareto_s{s_num} (id={sol_id})", "GA (open-loop)", rdf)

    # 4. PDDL classical planning baseline
    pddl_cfg = config.get("planning", {}).get("pddl", {})
    pddl_actions, _ = solve_pddl(
        pddl_dir / "domain.pddl",
        pddl_dir / "problem_ind_npl.pddl",
        search_name=str(pddl_cfg.get("search", "astar")),
        heuristic_name=str(pddl_cfg.get("heuristic", "hff")),
    )
    x_pddl = plan_to_decision_vector(pddl_actions, evaluator)
    roll_pddl = simulate_open_loop_schedule(
        x_pddl,
        evaluator,
        branches_mat,
        shock_multipliers=shock_multipliers,
        cost_weight_lambda=mdp_cfg.cost_weight_lambda,
        discount_factor=mdp_cfg.discount_factor,
    )
    _record("pddl_classical_plan", "PDDL (classical)", roll_pddl)

    comp_df = pd.DataFrame(summary_rows)

    # 5. Task 2 Sensitivity suite: budget_bins (6 vs 11) x gamma (0.774 vs 1.0) + lambda_eq sweep
    sens_rows: list[dict[str, Any]] = []
    for b_bins in budget_bins_list:
        mdp_b = mdp_by_bins[b_bins]
        for gamma in mdp_cfg.discount_sensitivity:
            for lam_eq in lambda_eq_list:
                sol = solve_finite_horizon_backward_induction(
                    mdp_b,
                    cost_weight_lambda=mdp_cfg.cost_weight_lambda,
                    discount_factor=gamma,
                    equity_weight_lambda=lam_eq,
                )
                roll = simulate_mdp_rollouts(
                    mdp_b,
                    sol,
                    evaluator,
                    n_rollouts=n_rollouts,
                    seed=mdp_cfg.seed,
                    precomputed_branches=branches_mat,
                )
                sens_rows.append(
                    {
                        "sweep_type": "lambda_eq_and_discount"
                        if lam_eq > 0.0
                        else "budget_bins_and_discount",
                        "budget_bins": int(b_bins),
                        "n_decision_states": int(sum(len(L) for L in mdp_b.layers[:-1])),
                        "discount_factor": float(gamma),
                        "cost_weight_lambda": float(mdp_cfg.cost_weight_lambda),
                        "equity_weight_lambda": float(lam_eq),
                        "solver_expected_value_s0": float(sol.expected_value_s0),
                        "sim_mean_scalarized_return": float(roll["discounted_return"].mean()),
                        "mean_dalys_2040": float(roll["dalys_2040_mean"].mean()),
                        "p05_dalys_2040": float(np.percentile(roll["dalys_2040_mean"], 5)),
                        "p95_dalys_2040": float(np.percentile(roll["dalys_2040_mean"], 95)),
                        "mean_dalys_averted": float(roll["dalys_averted_mean"].mean()),
                        "mean_cost_usd_millions": float(roll["total_cost_usd_millions"].mean()),
                        "mean_equity_gap_asr": float(roll["equity_gap_asr"].mean()),
                        "build_seconds": float(mdp_b.build_seconds),
                        "solve_seconds": float(sol.solve_seconds),
                        "is_mock": evaluator.is_mock,
                        "warning_mock_inputs": evaluator.warning_mock_inputs,
                        "data_version": evaluator.data_version,
                    }
                )

    sens_df = pd.DataFrame(sens_rows)

    # 6. Write CSVs + metadata sidecars
    suffix = "_mock" if evaluator.is_mock else ""
    comp_csv = tables_dir / f"comp_ga_vs_mdp{suffix}.csv"
    sens_csv = tables_dir / f"comp_sensitivity{suffix}.csv"
    comp_df.to_csv(comp_csv, index=False)
    sens_df.to_csv(sens_csv, index=False)

    runtime_s = time.perf_counter() - t_start
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    base_meta = {
        "source": "src.planning.compare",
        "generated_at": now_iso,
        "is_mock": evaluator.is_mock,
        "warning_mock_inputs": evaluator.warning_mock_inputs,
        "data_version": evaluator.data_version,
        "n_rollouts": n_rollouts,
        "shock_multipliers": list(shock_multipliers),
        "transition_probs": list(mdp_cfg.transition_probs),
        "cost_weight_lambda": mdp_cfg.cost_weight_lambda,
        "discount_factor": mdp_cfg.discount_factor,
        "seed": mdp_cfg.seed,
        "runtime_seconds": round(runtime_s, 4),
    }

    comp_meta = Path(f"{comp_csv}.meta.json")
    sens_meta = Path(f"{sens_csv}.meta.json")
    comp_meta.write_text(
        json.dumps({"artifact": comp_csv.name, **base_meta}, indent=2), encoding="utf-8"
    )
    sens_meta.write_text(
        json.dumps({"artifact": sens_csv.name, **base_meta}, indent=2), encoding="utf-8"
    )

    files: dict[str, Path] = {
        "comp_csv": comp_csv,
        "comp_meta": comp_meta,
        "sens_csv": sens_csv,
        "sens_meta": sens_meta,
    }

    if generate_figures:
        from src.viz.decision import (
            plot_comp_distributions,
            plot_comp_sensitivity,
            plot_comp_tradeoffs,
        )

        fig_dist = figures_dir / f"plan_comp_distributions{suffix}.png"
        fig_trade = figures_dir / f"plan_comp_tradeoffs{suffix}.png"
        fig_sens = figures_dir / f"plan_comp_sensitivity{suffix}.png"

        plot_comp_distributions(
            rollout_map,
            comp_df,
            fig_dist,
            is_mock=evaluator.is_mock,
            data_version=evaluator.data_version,
        )
        plot_comp_tradeoffs(
            pareto_df,
            comp_df,
            sens_df,
            fig_trade,
            is_mock=evaluator.is_mock,
            data_version=evaluator.data_version,
        )
        plot_comp_sensitivity(
            sens_df,
            pareto_df,
            comp_df,
            fig_sens,
            is_mock=evaluator.is_mock,
            data_version=evaluator.data_version,
        )

        for key, fpath in (
            ("fig_dist", fig_dist),
            ("fig_trade", fig_trade),
            ("fig_sens", fig_sens),
        ):
            mpath = Path(f"{fpath}.meta.json")
            mpath.write_text(
                json.dumps({"artifact": fpath.name, **base_meta}, indent=2),
                encoding="utf-8",
            )
            files[key] = fpath
            files[f"{key}_meta"] = mpath

    total_runtime_s = time.perf_counter() - t_start
    return {
        "is_mock": evaluator.is_mock,
        "data_version": evaluator.data_version,
        "runtime_seconds": float(total_runtime_s),
        "comp_df": comp_df,
        "sens_df": sens_df,
        "rollout_map": rollout_map,
        "mdp_b6_solver_ev": float(sol_mdp_b6.expected_value_s0),
        "mdp_b11_solver_ev": float(sol_mdp_b11.expected_value_s0),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``make compare`` (``python -m src.planning.compare``)."""
    parser = argparse.ArgumentParser(
        description="Run equal-footing GA vs. MDP vs. PDDL stochastic comparison."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args(argv)

    out = run_comparison_pipeline(args.config)
    comp_df: pd.DataFrame = out["comp_df"]
    mock_tag = " [MOCK DATA]" if out["is_mock"] else ""
    print(f"Comparison complete{mock_tag} in {out['runtime_seconds']:.2f}s")
    print(
        comp_df[
            [
                "method",
                "mean_dalys_2040",
                "p05_dalys_2040",
                "p95_dalys_2040",
                "mean_cost_usd_millions",
                "mean_equity_gap_asr",
                "mean_scalarized_value",
                "constraint_violation_rate",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
