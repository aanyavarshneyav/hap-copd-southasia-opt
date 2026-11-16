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
from matplotlib.patches import Patch

from src.optimise.baselines import BaselineResult
from src.optimise.evaluator import EvaluationResult, Stage3Evaluator
from src.optimise.nsga2 import MultiSeedResult, SingleSeedResult, decode_allocation_spend

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
