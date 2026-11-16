"""Publication-ready visualisations for Stage 3 NSGA-II and baseline comparisons.

Generates four Stage 3 figures in ``outputs/figures/``:
1. ``s3_pareto_tradeoffs{_mock}.png`` — Paired 2D projections + 3D view with UIs on selected plans.
2. ``s3_hypervolume_convergence{_mock}.png`` — Multi-seed HV convergence curves and stability.
3. ``s3_allocation_breakdown{_mock}.png`` — Spend stacked by unit, period, and intervention.
4. ``s3_baseline_comparison{_mock}.png`` — NSGA-II vs proportional, greedy, and random search.

All mock runs stamp a prominent diagonal ``SYNTHETIC DATA`` watermark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Patch

from src.optimise.baselines import BaselineResult
from src.optimise.evaluator import EvaluationResult, Stage3Evaluator
from src.optimise.nsga2 import MultiSeedResult, SingleSeedResult, decode_allocation_spend
from src.planning.htn import HTNNode

# Colorblind-safe Okabe-Ito palette
_PLAN_STYLE: dict[str, dict[str, Any]] = {
    "knee": {"label": "Knee (compromise)", "color": "#D55E00", "marker": "*", "size": 210},
    "cost_minimal": {"label": "Cost-minimal", "color": "#0072B2", "marker": "s", "size": 100},
    "daly_minimal": {"label": "DALY-minimal", "color": "#009E73", "marker": "^", "size": 120},
    "equity_minimal": {"label": "Equity-minimal", "color": "#CC79A7", "marker": "D", "size": 100},
}

_INTERVENTION_COLORS: dict[str, str] = {
    "lpg_subsidy": "#0072B2",
    "improved_biomass": "#E69F00",
    "electric_induction": "#009E73",
    "infra_lpg_grid": "#CC79A7",
}

_INTERVENTION_LABELS: dict[str, str] = {
    "lpg_subsidy": "LPG subsidy",
    "improved_biomass": "Improved biomass",
    "electric_induction": "Electric induction",
    "infra_lpg_grid": "LPG / grid infra",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _apply_mock_watermark(fig: Figure, is_mock: bool, data_version: str) -> None:
    """Stamp a visible SYNTHETIC DATA watermark onto mock figures."""
    if not is_mock:
        return
    fig.text(
        0.5,
        0.5,
        f"SYNTHETIC DATA ({data_version})",
        fontsize=36,
        fontweight="bold",
        color="#B91C1C",
        alpha=0.18,
        ha="center",
        va="center",
        rotation=25,
        zorder=1000,
    )
    fig.text(
        0.99,
        0.01,
        f"WARNING: SYNTHETIC MOCK INPUTS ({data_version}) — DO NOT CITE",
        fontsize=8,
        fontweight="bold",
        color="#991B1B",
        ha="right",
        va="bottom",
        zorder=1000,
    )


def _save_figure(fig: Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pareto_tradeoffs(
    pareto_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    bau_eval: EvaluationResult,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Plot paired 2D Pareto projections and a 3D trade-off scatter with UI bars."""
    _configure_style()
    fig = plt.figure(figsize=(13.5, 10.5))

    dalys_m = pareto_df["dalys_2040_mean"].to_numpy(dtype=float) / 1e6
    cost = pareto_df["total_cost_usd_millions"].to_numpy(dtype=float)
    eq = pareto_df["equity_gap_asr"].to_numpy(dtype=float)

    bau_dalys_m = float(bau_eval.dalys_2040_mean) / 1e6
    bau_eq = float(bau_eval.equity_gap_asr)

    # Panel A: DALYs vs Cost (with 95% UI error bars on selected plans)
    ax1 = fig.add_subplot(2, 2, 1)
    sc1 = ax1.scatter(cost, dalys_m, c=eq, cmap="viridis", s=28, alpha=0.82, edgecolors="none")
    ax1.scatter([0.0], [bau_dalys_m], color="#4B5563", marker="X", s=95, label="2040 BAU (x=0)")
    for _, row in selected_df.iterrows():
        st = _PLAN_STYLE[str(row["plan_name"])]
        y_mean = float(row["dalys_2040_mean"]) / 1e6
        y_lo = y_mean - float(row["dalys_2040_lower"]) / 1e6
        y_hi = float(row["dalys_2040_upper"]) / 1e6 - y_mean
        x_cost = float(row["total_cost_usd_millions"])
        ax1.errorbar(
            [x_cost],
            [y_mean],
            yerr=[[y_lo], [y_hi]],
            fmt="none",
            ecolor=st["color"],
            elinewidth=1.4,
            capsize=3,
            alpha=0.85,
            zorder=4,
        )
        ax1.scatter(
            [x_cost],
            [y_mean],
            color=st["color"],
            marker=st["marker"],
            s=st["size"],
            edgecolors="black",
            linewidths=0.8,
            label=st["label"],
            zorder=5,
        )
    ax1.set_xlabel("Total Intervention Cost (USD millions)")
    ax1.set_ylabel("Projected 2040 COPD DALYs (millions)")
    ax1.set_title("A. Burden vs. Cost (with 95% UI on Selected Plans)")
    cb1 = fig.colorbar(sc1, ax=ax1, pad=0.02)
    cb1.set_label("ASR Equity Gap (per 100k)", fontsize=8.5)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    # Panel B: Equity Gap vs Cost
    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(cost, eq, c=dalys_m, cmap="plasma", s=28, alpha=0.82, edgecolors="none")
    ax2.scatter([0.0], [bau_eq], color="#4B5563", marker="X", s=95, label="2040 BAU (x=0)")
    for _, row in selected_df.iterrows():
        st = _PLAN_STYLE[str(row["plan_name"])]
        ax2.scatter(
            [float(row["total_cost_usd_millions"])],
            [float(row["equity_gap_asr"])],
            color=st["color"],
            marker=st["marker"],
            s=st["size"],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
    ax2.set_xlabel("Total Intervention Cost (USD millions)")
    ax2.set_ylabel("Female–Male ASR Gap (DALYs per 100k)")
    ax2.set_title("B. Sex Equity Gap vs. Cost")
    cb2 = fig.colorbar(sc2, ax=ax2, pad=0.02)
    cb2.set_label("2040 DALYs (millions)", fontsize=8.5)

    # Panel C: Equity Gap vs DALYs
    ax3 = fig.add_subplot(2, 2, 3)
    sc3 = ax3.scatter(dalys_m, eq, c=cost, cmap="cividis", s=28, alpha=0.82, edgecolors="none")
    ax3.scatter([bau_dalys_m], [bau_eq], color="#4B5563", marker="X", s=95, label="2040 BAU")
    for _, row in selected_df.iterrows():
        st = _PLAN_STYLE[str(row["plan_name"])]
        ax3.scatter(
            [float(row["dalys_2040_mean"]) / 1e6],
            [float(row["equity_gap_asr"])],
            color=st["color"],
            marker=st["marker"],
            s=st["size"],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
    ax3.set_xlabel("Projected 2040 COPD DALYs (millions)")
    ax3.set_ylabel("Female–Male ASR Gap (DALYs per 100k)")
    ax3.set_title("C. Sex Equity Gap vs. Projected 2040 Burden")
    cb3 = fig.colorbar(sc3, ax=ax3, pad=0.02)
    cb3.set_label("Total Cost (USD M)", fontsize=8.5)

    # Panel D: 3D Pareto surface
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(cost, dalys_m, eq, c=dalys_m, cmap="viridis", s=22, alpha=0.75)
    for _, row in selected_df.iterrows():
        st = _PLAN_STYLE[str(row["plan_name"])]
        ax4.scatter(
            [float(row["total_cost_usd_millions"])],
            [float(row["dalys_2040_mean"]) / 1e6],
            [float(row["equity_gap_asr"])],
            color=st["color"],
            marker=st["marker"],
            s=st["size"] * 0.9,
            edgecolors="black",
            linewidths=0.8,
            label=st["label"],
        )
    ax4.set_xlabel("Cost (USD M)", labelpad=10)
    ax4.set_ylabel("2040 DALYs (M)", labelpad=10)
    ax4.set_zlabel("ASR Gap (/100k)", labelpad=10)
    ax4.tick_params(axis="both", labelsize=7.5)
    ax4.tick_params(axis="z", labelsize=7.5)
    ax4.set_title("D. 3D Pareto Front")
    ax4.view_init(elev=22, azim=-48)

    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}Stage 3 NSGA-II Pareto Trade-offs (IND + NPL, 2026–2040)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)


def plot_hypervolume_convergence(
    seed_results: list[SingleSeedResult],
    combined_hv: float,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Plot per-generation hypervolume trajectories and final stability across seeds."""
    _configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    gens = np.array([row["gen"] for row in seed_results[0].hv_history], dtype=int)
    hv_matrix = np.array(
        [[row["hypervolume"] for row in sr.hv_history] for sr in seed_results],
        dtype=float,
    )

    for idx, sr in enumerate(seed_results):
        ax1.plot(
            gens,
            hv_matrix[idx],
            alpha=0.5,
            linewidth=1.3,
            label=f"Seed {sr.seed}",
        )
    mean_curve = np.mean(hv_matrix, axis=0)
    min_curve = np.min(hv_matrix, axis=0)
    max_curve = np.max(hv_matrix, axis=0)
    ax1.fill_between(gens, min_curve, max_curve, color="#0072B2", alpha=0.14)
    ax1.plot(gens, mean_curve, color="#0072B2", linewidth=2.4, label="Mean across seeds")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Hypervolume")
    ax1.set_title("A. Generation-by-Generation Hypervolume Convergence")
    ax1.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.9)

    final_hvs = np.array([sr.hypervolume for sr in seed_results], dtype=float)
    seed_labels = [f"Seed {sr.seed}" for sr in seed_results]
    x_pos = np.arange(len(seed_results))
    bars = ax2.bar(x_pos, final_hvs, color="#0072B2", alpha=0.85, width=0.55)
    hv_mean = float(np.mean(final_hvs))
    hv_std = float(np.std(final_hvs))
    cv_pct = 100.0 * hv_std / hv_mean if hv_mean > 0 else 0.0
    ax2.axhline(
        hv_mean,
        color="#D55E00",
        linestyle="--",
        linewidth=1.6,
        label=f"Seed mean (CV={cv_pct:.2f}%)",
    )
    ax2.axhline(
        combined_hv, color="#009E73", linestyle=":", linewidth=1.8, label="Combined 5-seed front"
    )
    for bar, val in zip(bars, final_hvs, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val * 0.90,
            f"{val:.3e}",
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(seed_labels)
    ax2.set_ylabel("Final Hypervolume")
    ax2.set_ylim(0.0, max(combined_hv, float(np.max(final_hvs))) * 1.14)
    ax2.set_title("B. Final Hypervolume Stability Across Seeds")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9)

    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}NSGA-II Convergence & Seed Stability ({len(seed_results)} Seeds)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.94])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)


def plot_allocation_breakdown(
    selected_df: pd.DataFrame,
    evaluator: Stage3Evaluator,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Plot stacked spend by unit, intervention, and period for the 4 selected plans."""
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharey=True)
    axes_flat = axes.ravel()

    var_cols = list(evaluator.variable_names)
    units = evaluator.units
    periods = evaluator.periods
    interventions = evaluator.interventions

    # Compute consistent global y-max across all 4 plans
    spend_tables: list[pd.DataFrame] = []
    max_bar_cost = 1.0
    for _, row in selected_df.iterrows():
        x_vec = row[var_cols].to_numpy(dtype=float)
        spend_df = decode_allocation_spend(x_vec, evaluator)
        spend_tables.append(spend_df)
        grouped = spend_df.groupby(["unit", "period"], as_index=False)["cost_usd_millions"].sum()
        if not grouped.empty:
            max_bar_cost = max(max_bar_cost, float(grouped["cost_usd_millions"].max()))

    bar_width = 0.24
    period_offsets = [-bar_width, 0.0, bar_width]

    for ax, (_, row), spend_df in zip(axes_flat, selected_df.iterrows(), spend_tables, strict=True):
        plan_name = str(row["plan_name"])
        pivot = spend_df.pivot_table(
            index=["unit", "period"],
            columns="intervention",
            values="cost_usd_millions",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex(
            pd.MultiIndex.from_product([units, periods], names=["unit", "period"]),
            fill_value=0.0,
        )

        x_units = np.arange(len(units), dtype=float)
        for p_idx, period in enumerate(periods):
            x_pos = x_units + period_offsets[p_idx]
            bottom = np.zeros(len(units), dtype=float)
            for interv in interventions:
                vals = np.array(
                    [
                        float(pivot.loc[(u, period), interv]) if interv in pivot.columns else 0.0
                        for u in units
                    ],
                    dtype=float,
                )
                ax.bar(
                    x_pos,
                    vals,
                    width=bar_width * 0.92,
                    bottom=bottom,
                    color=_INTERVENTION_COLORS[interv],
                    edgecolor="white",
                    linewidth=0.4,
                )
                bottom += vals
            for u_idx, total_h in enumerate(bottom):
                if total_h > max_bar_cost * 0.02:
                    ax.text(
                        x_pos[u_idx],
                        total_h + max_bar_cost * 0.012,
                        period,
                        ha="center",
                        va="bottom",
                        fontsize=6.8,
                        color="#374151",
                    )

        ax.set_xticks(x_units)
        ax.set_xticklabels(units)
        ax.set_ylim(0.0, max_bar_cost * 1.15)
        dalys_m = float(row["dalys_2040_mean"]) / 1e6
        dalys_lo = float(row["dalys_2040_lower"]) / 1e6
        dalys_hi = float(row["dalys_2040_upper"]) / 1e6
        cost_m = float(row["total_cost_usd_millions"])
        eq_gap = float(row["equity_gap_asr"])
        title_lbl = _PLAN_STYLE[plan_name]["label"]
        ax.set_title(
            f"{title_lbl}\n"
            f"Cost=${cost_m:.1f}M | DALYs={dalys_m:.2f}M "
            f"[{dalys_lo:.2f}–{dalys_hi:.2f}] | Gap={eq_gap:.1f}",
            fontsize=10,
        )

    axes_flat[0].set_ylabel("Period Spend (USD millions)")
    axes_flat[2].set_ylabel("Period Spend (USD millions)")

    legend_handles = [
        Patch(facecolor=_INTERVENTION_COLORS[i], label=_INTERVENTION_LABELS[i])
        for i in interventions
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(interventions),
        frameon=True,
        bbox_to_anchor=(0.5, 0.025),
    )
    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}Selected Plan Spend Breakdown by Unit, Period (P1–P3), and Intervention",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.94])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)


def plot_baseline_comparison(
    nsga2_result: MultiSeedResult,
    baselines: dict[str, BaselineResult],
    comparison_df: pd.DataFrame,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Plot NSGA-II Pareto front vs proportional, greedy, and random search baselines."""
    _configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # Left panel: DALYs vs Cost Pareto comparison
    nsga_F = nsga2_result.combined_F
    ax1.scatter(
        nsga_F[:, 1],
        nsga_F[:, 0] / 1e6,
        color="#0072B2",
        s=32,
        alpha=0.85,
        label="NSGA-II (5-seed combined)",
        zorder=4,
    )

    rand_F = baselines["random"].F
    ax1.scatter(
        rand_F[:, 1],
        rand_F[:, 0] / 1e6,
        color="#E69F00",
        marker="o",
        facecolors="none",
        edgecolors="#E69F00",
        linewidths=1.5,
        s=45,
        label="Random search (repaired, matched evals)",
        zorder=3,
    )

    prop_F = baselines["proportional"].F[0]
    ax1.scatter(
        [prop_F[1]],
        [prop_F[0] / 1e6],
        color="#CC79A7",
        marker="D",
        s=110,
        edgecolors="black",
        label="Proportional equal split",
        zorder=5,
    )

    greedy_F = baselines["greedy"].F[0]
    ax1.scatter(
        [greedy_F[1]],
        [greedy_F[0] / 1e6],
        color="#009E73",
        marker="^",
        s=130,
        edgecolors="black",
        label="Greedy (DALYs averted / $)",
        zorder=5,
    )

    bau_dalys_m = float(nsga2_result.bau_evaluation.dalys_2040_mean) / 1e6
    ax1.scatter([0.0], [bau_dalys_m], color="#4B5563", marker="X", s=95, label="2040 BAU (x=0)")

    ax1.set_xlabel("Total Intervention Cost (USD millions)")
    ax1.set_ylabel("Projected 2040 COPD DALYs (millions)")
    ax1.set_title("A. Objective Space: NSGA-II vs. Baselines")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    # Right panel: Hypervolume comparison
    label_map = {
        "nsga2_combined_5seeds": "NSGA-II (5 seeds)",
        "baseline_random_search": "Random search",
        "baseline_greedy_dalys_per_dollar": "Greedy (DALYs/$)",
        "baseline_proportional_split": "Proportional split",
    }
    sub = comparison_df[comparison_df["method"].isin(label_map.keys())].copy()
    sub["display"] = sub["method"].map(label_map)
    sub = sub.sort_values("hypervolume", ascending=True)

    colors = ["#0072B2" if m.startswith("nsga2_") else "#9CA3AF" for m in sub["method"].tolist()]
    y_pos = np.arange(len(sub))
    bars = ax2.barh(y_pos, sub["hypervolume"].to_numpy(dtype=float), color=colors, height=0.55)
    max_hv = float(sub["hypervolume"].max())
    for bar, hv_val in zip(bars, sub["hypervolume"].to_numpy(dtype=float), strict=True):
        ax2.text(
            hv_val + max_hv * 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{hv_val:.3e}",
            va="center",
            ha="left",
            fontsize=8.5,
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sub["display"].tolist())
    ax2.set_xlim(0.0, max_hv * 1.24)
    ax2.set_xlabel("Dominated Hypervolume (higher is better)")
    ax2.set_title("B. Hypervolume Comparison")

    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}Stage 3 NSGA-II vs. Heuristic & Random Search Baselines",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.94])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)


def plot_mdp_policy_and_sensitivity(
    sensitivity_df: pd.DataFrame,
    rollouts_df: pd.DataFrame,
    chosen_lambda: float,
    chosen_gamma: float,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Plot MDP lambda/discount sensitivity and N-rollout stochastic outcome distribution."""
    _configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

    gamma_colors = {0.774: "#0072B2", 1.0: "#D55E00"}
    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)

    for gamma_val, sub in sensitivity_df.groupby("discount_factor"):
        sub_s = sub.sort_values("cost_weight_lambda")
        col = gamma_colors.get(round(float(gamma_val), 3), "#009E73")
        lams = sub_s["cost_weight_lambda"].to_numpy(dtype=float)
        da_m = sub_s["mean_dalys_averted"].to_numpy(dtype=float) / 1e6
        cost_m = sub_s["mean_cost_usd_millions"].to_numpy(dtype=float)
        ax1.plot(
            lams,
            da_m,
            marker="o",
            linewidth=2.1,
            color=col,
            label=f"DALYs averted (γ={gamma_val:.3f})",
        )
        ax1_r.plot(
            lams,
            cost_m,
            marker="s",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            color=col,
            label=f"Cost $M (γ={gamma_val:.3f})",
        )

    ax1.axvline(
        chosen_lambda,
        color="#374151",
        linestyle=":",
        linewidth=1.6,
        label=f"Calibrated λ={chosen_lambda:.0f}",
    )
    ax1.set_xlabel("Cost Penalty Weight λ (DALYs per USD million)")
    ax1.set_ylabel("Mean 2040 DALYs Averted (millions)")
    ax1_r.set_ylabel("Mean Programme Cost (USD millions)")
    ax1.set_title("A. MDP Policy Sensitivity to λ and Discount Factor γ")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.8, frameon=True, framealpha=0.92)

    # Panel B: Stochastic rollout distribution at calibrated (lambda, gamma)
    averted_m = rollouts_df["dalys_averted_mean"].to_numpy(dtype=float) / 1e6
    costs = rollouts_df["total_cost_usd_millions"].to_numpy(dtype=float)
    mean_da = float(np.mean(averted_m))
    q_lo, q_hi = np.percentile(averted_m, [2.5, 97.5])
    mean_cost = float(np.mean(costs))

    ax2.hist(averted_m, bins=24, color="#009E73", alpha=0.78, edgecolor="white")
    ax2.axvline(mean_da, color="#111827", linewidth=2.0, label=f"Mean={mean_da:.3f}M")
    ax2.axvline(
        q_lo,
        color="#D55E00",
        linestyle="--",
        linewidth=1.4,
        label=f"95% range [{q_lo:.3f}, {q_hi:.3f}]M",
    )
    ax2.axvline(q_hi, color="#D55E00", linestyle="--", linewidth=1.4)
    ax2.set_xlabel("Realized 2040 DALYs Averted (millions)")
    ax2.set_ylabel("Simulated Rollouts (count)")
    ax2.set_title(
        f"B. Stochastic Rollouts (N={len(rollouts_df)}, λ={chosen_lambda:.0f}, "
        f"γ={chosen_gamma:.3f}, Mean Cost=${mean_cost:.1f}M)"
    )
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92)

    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}Stage 4 Custom NumPy MDP: Sensitivity & Stochastic Rollout Outcomes",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.94])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)


def plot_htn_tree(
    root: HTNNode,
    evaluation: EvaluationResult,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Render the plain-Python HTN decomposition tree with mapped leaf actions."""
    _configure_style()

    # Assign vertical coordinates via post-order traversal
    leaf_counter = 0
    coords: dict[int, tuple[float, float, int]] = {}
    edges: list[tuple[int, int]] = []

    def _assign(node: HTNNode, depth: int) -> float:
        nonlocal leaf_counter
        if not node.children:
            y = float(leaf_counter)
            leaf_counter += 1
            coords[id(node)] = (float(depth), y, depth)
            return y
        child_ys = [_assign(ch, depth + 1) for ch in node.children]
        for ch in node.children:
            edges.append((id(node), id(ch)))
        y_mid = float(np.mean(child_ys))
        coords[id(node)] = (float(depth), y_mid, depth)
        return y_mid

    _assign(root, 0)
    max_depth = max(c[2] for c in coords.values())
    n_slots = max(leaf_counter, 1)

    fig_h = max(7.5, 0.52 * n_slots)
    fig, ax = plt.subplots(figsize=(15.0, fig_h))
    ax.axis("off")

    nodes_by_id: dict[int, HTNNode] = {}

    def _collect(node: HTNNode) -> None:
        nodes_by_id[id(node)] = node
        for ch in node.children:
            _collect(ch)

    _collect(root)

    for parent_id, child_id in edges:
        px, py, _ = coords[parent_id]
        cx, cy, _ = coords[child_id]
        ax.plot([px + 0.18, cx - 0.18], [py, cy], color="#9CA3AF", linewidth=1.1, zorder=1)

    for nid, (x, y, _) in coords.items():
        node = nodes_by_id[nid]
        if node.node_type == "primitive":
            face = _INTERVENTION_COLORS.get(node.intervention or "", "#0072B2")
            edge = "#111827"
            txt_color = "white"
            label = (
                f"{node.unit} | {node.period} | {node.intervention}\n"
                f"x={node.allocation_value:.2f}, ${node.cost_usd_millions:.1f}M"
            )
            box_w, box_h = 0.88, 0.68
        else:
            face = "#EFF6FF" if node.method else "#F3F4F6"
            edge = "#1D4ED8" if node.method else "#6B7280"
            txt_color = "#111827"
            short_name = node.name.replace(
                "AchieveSouthAsiaCleanCookingPlan", "Goal: SouthAsiaPlan"
            ).replace("VerifyBudgetAndEquityGuard(", "VerifyGuard(\n")
            label = f"{short_name}" + (f"\n[{node.method}]" if node.method else "")
            box_w, box_h = 0.78, 0.64

        patch = FancyBboxPatch(
            (x - box_w / 2.0, y - box_h / 2.0),
            box_w,
            box_h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.3,
            color=txt_color,
            fontweight="bold" if node.node_type == "primitive" else "normal",
            zorder=4,
        )

    ax.set_xlim(-0.6, float(max_depth) + 0.7)
    ax.set_ylim(-0.8, float(n_slots) - 0.2)
    ax.invert_yaxis()

    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}HTN Decomposition Tree | "
        f"Cost=${float(evaluation.total_cost_usd_millions):.1f}M | "
        f"2040 DALYs={float(evaluation.dalys_2040_mean) / 1e6:.2f}M | "
        f"Feasible={evaluation.is_feasible}",
        fontsize=12.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.96])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)


def plot_comp_distributions(
    rollout_map: dict[str, pd.DataFrame],
    comp_df: pd.DataFrame,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Plot rollout outcome distributions (violin/box) and low-uptake adaptive contrast."""
    _configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6))

    # Core comparable methods (MDP b=6/11, GA matched/knee/daly_min/cost_min, PDDL)
    ordered_keys = [
        k for k in rollout_map if not k.startswith("ga_pareto_s") and k != "ga_equity_minimal"
    ]
    if not ordered_keys:
        ordered_keys = list(rollout_map.keys())

    short_labels = [
        k.replace("ga_matched_best ", "GA best\n")
        .replace("mdp_adaptive_", "MDP ")
        .replace("pddl_classical_plan", "PDDL")
        .replace("ga_", "GA ")
        for k in ordered_keys
    ]

    dalys_data = [
        rollout_map[k]["dalys_2040_mean"].to_numpy(dtype=float) / 1e6 for k in ordered_keys
    ]

    parts = ax1.violinplot(dalys_data, showmeans=True, showextrema=False, widths=0.7)
    colors = [
        "#0072B2" if k.startswith("mdp_") else ("#CC79A7" if k.startswith("pddl_") else "#E69F00")
        for k in ordered_keys
    ]
    for body, col in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(col)
        body.set_alpha(0.65)
        body.set_edgecolor("#1F2937")
    ax1.boxplot(
        dalys_data,
        widths=0.22,
        patch_artist=False,
        medianprops={"color": "#111827", "linewidth": 1.6},
        whiskerprops={"color": "#374151"},
        capprops={"color": "#374151"},
        showfliers=False,
    )
    ax1.set_xticks(np.arange(1, len(ordered_keys) + 1))
    ax1.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=8.5)
    ax1.set_ylabel("Realized 2040 COPD DALYs (millions)")
    ax1.set_title("A. 2040 Burden Distribution Across N=1,000 Shocks")

    # Panel B: Scalarized return all rollouts vs P1 low-uptake shocks
    sub = comp_df[comp_df["method"].isin(ordered_keys)].copy()
    sub = sub.set_index("method").loc[ordered_keys].reset_index()
    x_pos = np.arange(len(ordered_keys), dtype=float)
    w = 0.36
    all_ret = sub["mean_scalarized_value"].to_numpy(dtype=float) / 1e3
    low_ret = sub["mean_scalarized_low_p1"].to_numpy(dtype=float) / 1e3
    ax2.bar(x_pos - w / 2, all_ret, width=w, color="#0072B2", alpha=0.88, label="All rollouts")
    ax2.bar(
        x_pos + w / 2,
        low_ret,
        width=w,
        color="#D55E00",
        alpha=0.88,
        label="Low-uptake P1 shock (p=0.15)",
    )
    ax2.axhline(0.0, color="#374151", linewidth=0.9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_labels, rotation=18, ha="right", fontsize=8.5)
    ax2.set_ylabel("Mean Scalarized Value (thousands)")
    ax2.set_title("B. Expected Value vs. Low-Uptake Shock Resilience")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92)

    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}Equal-Footing Stochastic Rollout Comparison (GA vs. MDP vs. PDDL)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.94])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)


def plot_comp_tradeoffs(
    pareto_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Plot NSGA-II Pareto front overlaid with stochastic MDP, GA, and PDDL outcomes."""
    _configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4))

    # Background deterministic NSGA-II front
    ax1.scatter(
        pareto_df["total_cost_usd_millions"],
        pareto_df["dalys_2040_mean"] / 1e6,
        color="#CBD5E1",
        s=22,
        label="NSGA-II Pareto front (nominal)",
        zorder=1,
    )
    ax2.scatter(
        pareto_df["dalys_2040_mean"] / 1e6,
        pareto_df["equity_gap_asr"],
        color="#CBD5E1",
        s=22,
        label="NSGA-II Pareto front (nominal)",
        zorder=1,
    )

    for _, row in comp_df.iterrows():
        method = str(row["method"])
        if method.startswith("ga_pareto_s"):
            continue
        cost = float(row["mean_cost_usd_millions"])
        dalys_m = float(row["mean_dalys_2040"]) / 1e6
        lo_m = dalys_m - float(row["p05_dalys_2040"]) / 1e6
        hi_m = float(row["p95_dalys_2040"]) / 1e6 - dalys_m
        eq = float(row["mean_equity_gap_asr"])

        if method.startswith("mdp_"):
            col, mk, sz = ("#0072B2" if "b6" in method else "#009E73"), "*", 190
            lbl = "MDP adaptive (b=6)" if "b6" in method else "MDP adaptive (b=11)"
        elif method.startswith("pddl_"):
            col, mk, sz = "#CC79A7", "P", 135
            lbl = "PDDL classical plan"
        elif method.startswith("ga_matched_best"):
            col, mk, sz = "#D55E00", "D", 105
            lbl = "GA best scalarized"
        else:
            col, mk, sz = "#E69F00", "o", 70
            lbl = method.replace("ga_", "GA ")

        ax1.errorbar(
            [cost],
            [dalys_m],
            yerr=[[lo_m], [hi_m]],
            fmt="none",
            ecolor=col,
            elinewidth=1.2,
            capsize=3,
            alpha=0.75,
            zorder=3,
        )
        ax1.scatter(
            [cost], [dalys_m], color=col, marker=mk, s=sz, edgecolors="black", label=lbl, zorder=4
        )
        ax2.scatter(
            [dalys_m], [eq], color=col, marker=mk, s=sz, edgecolors="black", label=lbl, zorder=4
        )

    # Overlay MDP lambda_eq > 0 trajectory at gamma=0.774, budget_bins=11 (or max available)
    max_b = int(sens_df["budget_bins"].max())
    eq_sweep = sens_df[
        (sens_df["budget_bins"] == max_b) & (np.isclose(sens_df["discount_factor"], 0.774))
    ]
    if len(eq_sweep) > 1:
        eq_s = eq_sweep.sort_values("equity_weight_lambda")
        ax2.plot(
            eq_s["mean_dalys_2040"] / 1e6,
            eq_s["mean_equity_gap_asr"],
            color="#009E73",
            linestyle="--",
            linewidth=1.6,
            marker="^",
            markersize=6,
            label=f"MDP λ_eq sweep (b={max_b})",
            zorder=5,
        )

    ax1.set_xlabel("Mean Programme Cost (USD millions)")
    ax1.set_ylabel("Mean 2040 COPD DALYs (millions, 5th–95th pct)")
    ax1.set_title("A. Cost vs. Stochastic 2040 Burden")
    ax1.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.92)

    ax2.set_xlabel("Mean 2040 COPD DALYs (millions)")
    ax2.set_ylabel("Mean Female–Male ASR Gap (per 100k)")
    ax2.set_title("B. Burden vs. Equity Trade-off (incl. MDP λ_eq Sweep)")
    ax2.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.92)

    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}MDP vs. NSGA-II vs. PDDL Objective-Space Comparison",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.94])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)


def plot_comp_sensitivity(
    sens_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    out_path: Path,
    *,
    is_mock: bool,
    data_version: str,
) -> Path:
    """Plot sensitivity across budget_bins (6 vs 11), lambda_eq, and discount factor."""
    _configure_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # Panel A: budget_bins 6 vs 11 at lambda_eq = 0
    sub_b = sens_df[np.isclose(sens_df["equity_weight_lambda"], 0.0)].copy()
    for gamma_val, col in ((0.774, "#0072B2"), (1.0, "#D55E00")):
        g_sub = sub_b[np.isclose(sub_b["discount_factor"], gamma_val)].sort_values("budget_bins")
        ax1.plot(
            g_sub["budget_bins"],
            g_sub["sim_mean_scalarized_return"] / 1e3,
            marker="o",
            linewidth=2.0,
            color=col,
            label=f"MDP γ={gamma_val:.3f}",
        )
    ga_best = comp_df[comp_df["method"].str.startswith("ga_matched_best")]
    if not ga_best.empty:
        ga_val = float(ga_best["mean_scalarized_value"].iloc[0]) / 1e3
        ax1.axhline(
            ga_val, color="#009E73", linestyle="--", linewidth=1.6, label="GA matched (γ=0.774)"
        )
    ax1.set_xticks(sorted(sub_b["budget_bins"].unique()))
    ax1.set_xlabel("MDP Budget Discretization (budget_bins)")
    ax1.set_ylabel("Mean Scalarized Return (thousands)")
    ax1.set_title("A. Discretization: budget_bins 6 vs. 11")
    ax1.legend(loc="best", fontsize=8, frameon=True)

    # Panel B: lambda_eq > 0 sweep at gamma=0.774
    sub_eq = sens_df[np.isclose(sens_df["discount_factor"], 0.774)].copy()
    for b_bins, col in ((6, "#0072B2"), (11, "#009E73")):
        b_sub = sub_eq[sub_eq["budget_bins"] == b_bins].sort_values("equity_weight_lambda")
        if b_sub.empty:
            continue
        ax2.plot(
            b_sub["equity_weight_lambda"] / 1e3,
            b_sub["mean_equity_gap_asr"],
            marker="s",
            linewidth=2.0,
            color=col,
            label=f"MDP b={b_bins}",
        )
    ax2.set_xlabel("Equity Weight λ_eq (thousands)")
    ax2.set_ylabel("Mean Female–Male ASR Gap (per 100k)")
    ax2.set_title("B. Equity Weight λ_eq Sensitivity (γ=0.774)")
    ax2.legend(loc="best", fontsize=8, frameon=True)

    # Panel C: Discount factor 0.774 vs 1.0 (DALYs averted & cost at lambda_eq=0)
    sub_g = sub_b.copy()
    labels_g = [
        f"b={int(r['budget_bins'])}, γ={r['discount_factor']:.3f}" for _, r in sub_g.iterrows()
    ]
    x_g = np.arange(len(sub_g), dtype=float)
    w = 0.36
    ax3_r = ax3.twinx()
    ax3_r.spines["top"].set_visible(False)
    ax3.bar(
        x_g - w / 2,
        sub_g["mean_dalys_averted"].to_numpy(dtype=float) / 1e6,
        width=w,
        color="#0072B2",
        alpha=0.85,
        label="DALYs averted (M)",
    )
    ax3_r.bar(
        x_g + w / 2,
        sub_g["mean_cost_usd_millions"].to_numpy(dtype=float),
        width=w,
        color="#E69F00",
        alpha=0.85,
        label="Mean Cost ($M)",
    )
    ax3.set_xticks(x_g)
    ax3.set_xticklabels(labels_g, rotation=18, ha="right", fontsize=8)
    ax3.set_ylabel("Mean 2040 DALYs Averted (M)")
    ax3_r.set_ylabel("Mean Cost (USD M)")
    ax3.set_title("C. Discount Factor γ=0.774 vs. 1.0")
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_r.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6, frameon=True)

    title_prefix = "[SYNTHETIC MOCK] " if is_mock else ""
    fig.suptitle(
        f"{title_prefix}MDP Sensitivity: Budget Bins, Equity Weight λ_eq, and Discounting",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.93])
    _apply_mock_watermark(fig, is_mock, data_version)
    return _save_figure(fig, out_path)
