"""CLI and pipeline entrypoint for Stage 3 NSGA-II optimisation (``make stage3``).

Runs multi-seed NSGA-II via ``src.optimise.nsga2``, compares against three
baseline comparators via ``src.optimise.baselines``, exports tables and metadata
sidecars to ``outputs/tables/``, and generates figures via ``src.viz.decision``
to ``outputs/figures/``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import load_config, load_stage3_contracts
from src.optimise.baselines import compare_nsga2_vs_baselines, run_all_baselines
from src.optimise.evaluator import Stage3Evaluator
from src.optimise.nsga2 import (
    MultiSeedResult,
    NSGA2Config,
    hypervolume_seeds_to_dataframe,
    pareto_front_to_dataframe,
    run_nsga2_multi_seed,
    select_representative_plans,
)
from src.viz.decision import (
    plot_allocation_breakdown,
    plot_baseline_comparison,
    plot_hypervolume_convergence,
    plot_pareto_tradeoffs,
)

MOCK_OUTPUT_NOTE: str = (
    "SYNTHETIC MOCK DATA — PIPELINE TESTING OUTPUT ONLY; DO NOT CITE OR TREAT AS REAL."
)
REAL_OUTPUT_NOTE: str = (
    "Stage 3 scenario analysis output (costs and effects are scenario assumptions)."
)


def artifact_path(directory: Path, stem: str, ext: str, *, is_mock: bool) -> Path:
    """Return output path with ``_mock`` suffix whenever ``is_mock`` is True."""
    suffix = "_mock" if is_mock else ""
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    return directory / f"{stem}{suffix}{clean_ext}"


def _write_metadata_sidecar(
    target_path: Path,
    *,
    evaluator: Stage3Evaluator,
    nsga2_cfg: NSGA2Config,
    ref_point: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> Path:
    meta_path = Path(f"{target_path}.meta.json")
    payload: dict[str, Any] = {
        "artifact": target_path.name,
        "source": "src.optimise.run",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_version": evaluator.data_version,
        "is_mock": evaluator.is_mock,
        "warning_mock_inputs": evaluator.warning_mock_inputs,
        "note": MOCK_OUTPUT_NOTE if evaluator.is_mock else REAL_OUTPUT_NOTE,
        "algorithm": "NSGA2",
        "population_size": nsga2_cfg.population_size,
        "n_generations": nsga2_cfg.n_generations,
        "seeds": list(nsga2_cfg.seeds),
        "use_repair": nsga2_cfg.use_repair,
        "repair_budget": nsga2_cfg.repair_budget,
        "total_budget_usd_millions": evaluator.total_budget_usd_millions,
        "hypervolume_ref_point": [float(v) for v in ref_point],
    }
    if extra:
        payload.update(extra)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def _save_csv_with_sidecar(
    df: pd.DataFrame,
    out_path: Path,
    *,
    evaluator: Stage3Evaluator,
    nsga2_cfg: NSGA2Config,
    ref_point: np.ndarray,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    meta_path = _write_metadata_sidecar(
        out_path,
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=ref_point,
        extra=extra_meta,
    )
    return out_path, meta_path


def run_stage3(
    config_path: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute full Stage 3 NSGA-II + baseline comparison + plotting pipeline."""
    t_start = time.perf_counter()
    root = Path(base_dir) if base_dir is not None else Path(".")
    config = load_config(config_path)
    contracts = load_stage3_contracts(config, base_dir=root)
    evaluator = Stage3Evaluator(config=config, contracts=contracts)
    nsga2_cfg = NSGA2Config.from_config(config)

    paths_cfg = config["paths"]
    tables_dir = root / paths_cfg["outputs_tables_dir"]
    figures_dir = root / paths_cfg["outputs_figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Multi-seed NSGA-II optimisation
    nsga2_res: MultiSeedResult = run_nsga2_multi_seed(evaluator, config)

    # 2. Extract selected plans (knee, cost_minimal, daly_minimal, equity_minimal)
    selected_df, selected_indices = select_representative_plans(
        nsga2_res.combined_X, nsga2_res.combined_evaluation, evaluator
    )
    pareto_df = pareto_front_to_dataframe(nsga2_res, evaluator, selected_indices=selected_indices)
    hv_seeds_df = hypervolume_seeds_to_dataframe(nsga2_res)

    # 3. Baseline comparators (same evaluator, budget, ref_point, matched n_eval)
    matched_n_eval = nsga2_res.seed_results[0].n_eval
    baselines = run_all_baselines(
        evaluator,
        config,
        nsga2_res.ref_point,
        n_eval_match=matched_n_eval,
    )
    comparison_df = compare_nsga2_vs_baselines(nsga2_res, baselines, evaluator)

    total_runtime_s = time.perf_counter() - t_start
    common_meta = {"pipeline_runtime_seconds": round(total_runtime_s, 4)}

    # 4. Write CSV tables + metadata sidecars
    is_mock = evaluator.is_mock
    pareto_csv, pareto_meta = _save_csv_with_sidecar(
        pareto_df,
        artifact_path(tables_dir, "s3_pareto_front", ".csv", is_mock=is_mock),
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=nsga2_res.ref_point,
        extra_meta={**common_meta, "n_pareto_points": int(len(pareto_df))},
    )
    selected_csv, selected_meta = _save_csv_with_sidecar(
        selected_df,
        artifact_path(tables_dir, "s3_selected_plans", ".csv", is_mock=is_mock),
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=nsga2_res.ref_point,
        extra_meta=common_meta,
    )
    hv_csv, hv_meta = _save_csv_with_sidecar(
        hv_seeds_df,
        artifact_path(tables_dir, "s3_hypervolume_seeds", ".csv", is_mock=is_mock),
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=nsga2_res.ref_point,
        extra_meta={
            **common_meta,
            "combined_hypervolume": float(nsga2_res.combined_hypervolume),
        },
    )
    comp_csv, comp_meta = _save_csv_with_sidecar(
        comparison_df,
        artifact_path(tables_dir, "s3_baseline_comparison", ".csv", is_mock=is_mock),
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=nsga2_res.ref_point,
        extra_meta=common_meta,
    )

    # 5. Generate figures + metadata sidecars
    fig_pareto = plot_pareto_tradeoffs(
        pareto_df,
        selected_df,
        nsga2_res.bau_evaluation,
        artifact_path(figures_dir, "s3_pareto_tradeoffs", ".png", is_mock=is_mock),
        is_mock=is_mock,
        data_version=evaluator.data_version,
    )
    fig_pareto_meta = _write_metadata_sidecar(
        fig_pareto,
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=nsga2_res.ref_point,
        extra=common_meta,
    )

    fig_hv = plot_hypervolume_convergence(
        nsga2_res.seed_results,
        nsga2_res.combined_hypervolume,
        artifact_path(figures_dir, "s3_hypervolume_convergence", ".png", is_mock=is_mock),
        is_mock=is_mock,
        data_version=evaluator.data_version,
    )
    fig_hv_meta = _write_metadata_sidecar(
        fig_hv,
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=nsga2_res.ref_point,
        extra=common_meta,
    )

    fig_alloc = plot_allocation_breakdown(
        selected_df,
        evaluator,
        artifact_path(figures_dir, "s3_allocation_breakdown", ".png", is_mock=is_mock),
        is_mock=is_mock,
        data_version=evaluator.data_version,
    )
    fig_alloc_meta = _write_metadata_sidecar(
        fig_alloc,
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=nsga2_res.ref_point,
        extra=common_meta,
    )

    fig_base = plot_baseline_comparison(
        nsga2_res,
        baselines,
        comparison_df,
        artifact_path(figures_dir, "s3_baseline_comparison", ".png", is_mock=is_mock),
        is_mock=is_mock,
        data_version=evaluator.data_version,
    )
    fig_base_meta = _write_metadata_sidecar(
        fig_base,
        evaluator=evaluator,
        nsga2_cfg=nsga2_cfg,
        ref_point=nsga2_res.ref_point,
        extra=common_meta,
    )

    total_with_viz_s = time.perf_counter() - t_start
    return {
        "is_mock": is_mock,
        "data_version": evaluator.data_version,
        "runtime_seconds": total_with_viz_s,
        "nsga2_result": nsga2_res,
        "pareto_df": pareto_df,
        "selected_df": selected_df,
        "hv_seeds_df": hv_seeds_df,
        "comparison_df": comparison_df,
        "files": {
            "pareto_csv": pareto_csv,
            "pareto_meta": pareto_meta,
            "selected_csv": selected_csv,
            "selected_meta": selected_meta,
            "hv_csv": hv_csv,
            "hv_meta": hv_meta,
            "comparison_csv": comp_csv,
            "comparison_meta": comp_meta,
            "fig_pareto": fig_pareto,
            "fig_pareto_meta": fig_pareto_meta,
            "fig_hv": fig_hv,
            "fig_hv_meta": fig_hv_meta,
            "fig_alloc": fig_alloc,
            "fig_alloc_meta": fig_alloc_meta,
            "fig_baseline": fig_base,
            "fig_baseline_meta": fig_base_meta,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m src.optimise.run``."""
    parser = argparse.ArgumentParser(
        description="Run Stage 3 NSGA-II optimisation and baseline comparisons."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args(argv)

    summary = run_stage3(args.config)
    hv_df: pd.DataFrame = summary["hv_seeds_df"]
    sel_df: pd.DataFrame = summary["selected_df"]
    comp_df: pd.DataFrame = summary["comparison_df"]

    mock_banner = " [MOCK DATA]" if summary["is_mock"] else ""
    print(f"Stage 3 complete{mock_banner} in {summary['runtime_seconds']:.2f}s")
    print(
        f"Hypervolume across {len(hv_df)} seeds: "
        f"mean={hv_df['hypervolume'].mean():.4e}, "
        f"std={hv_df['hypervolume'].std(ddof=0):.4e}, "
        f"combined={summary['nsga2_result'].combined_hypervolume:.4e}"
    )
    print("\nSelected plans:")
    print(
        sel_df[
            [
                "plan_name",
                "dalys_2040_mean",
                "dalys_2040_lower",
                "dalys_2040_upper",
                "total_cost_usd_millions",
                "equity_gap_asr",
            ]
        ].to_string(index=False)
    )
    print("\nNSGA-II vs Baselines:")
    print(
        comp_df[
            [
                "method",
                "n_pareto_points",
                "hypervolume",
                "min_dalys_2040_mean",
                "min_equity_gap_asr",
                "nsga2_beats_hypervolume",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
