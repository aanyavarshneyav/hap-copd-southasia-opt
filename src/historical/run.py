"""Stage 1 entrypoint for AEGIS-GBD: Historical Analysis (1990-2023).

Executes end-to-end historical processing:
1. Ensures interim data tables exist (triggering ingestion if missing)
2. Computes historical summary, EAPC with 95% CIs, pwlf joinpoints, and shares
3. Exports tables to outputs/tables/s1_*.csv
4. Generates figures to outputs/figures/s1_*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.ingest import run_ingest
from src.historical.metrics import (
    compute_attributable_shares,
    compute_eapc,
    compute_historical_summary,
    compute_pwlf_joinpoints,
)
from src.viz.historical import (
    plot_hap_copd_share,
    plot_ind_npl_comparison,
    plot_joinpoints,
    plot_trends_southasia,
)


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_stage1(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Execute complete Stage 1 historical pipeline."""
    cfg = load_config(config_path)
    interim_dir = Path(cfg["paths"]["interim_dir"])
    tables_dir = Path(cfg["paths"]["outputs_tables_dir"])
    figures_dir = Path(cfg["paths"]["outputs_figures_dir"])

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    burden_file = interim_dir / "historical_burden.parquet"
    tot_file = interim_dir / "total_copd.parquet"
    all_file = interim_dir / "all_cause_hap.parquet"

    if not (burden_file.exists() and tot_file.exists() and all_file.exists()):
        print("  Interim files not found; triggering ingestion first...")
        run_ingest(config_path)

    print("=== [AEGIS-GBD] Stage 1: Historical Analysis (1990–2023) ===")
    df_hap = pd.read_parquet(burden_file)
    df_tot = pd.read_parquet(tot_file)
    df_all = pd.read_parquet(all_file)

    # 1. Historical Summary (1990 vs 2023)
    print("  Computing 1990 vs 2023 historical burden summary...")
    df_summary = compute_historical_summary(df_hap, df_tot, df_all)
    summary_path = tables_dir / "s1_historical_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"  [OK] Saved {summary_path} ({len(df_summary)} rows)")

    # 2. EAPC (Overall 1990-2023 + Periods 1990-2010 and 2010-2023)
    print("  Computing Estimated Annual Percentage Change (EAPC) with 95% CIs...")
    eapc_full = compute_eapc(df_hap, 1990, 2023, "1990-2023")
    eapc_early = compute_eapc(df_hap, 1990, 2010, "1990-2010")
    eapc_late = compute_eapc(df_hap, 2010, 2023, "2010-2023")
    eapc_combined = pd.concat([eapc_full, eapc_early, eapc_late], ignore_index=True)

    # 3. Piecewise linear joinpoint regression (pwlf)
    print("  Fitting piecewise linear joinpoints (pwlf)...")
    df_pwlf = compute_pwlf_joinpoints(df_hap)
    eapc_joinpoint_path = tables_dir / "s1_eapc_joinpoint.csv"
    # Merge or save EAPC + Joinpoint
    eapc_combined.to_csv(eapc_joinpoint_path, index=False)
    if not df_pwlf.empty:
        pwlf_path = tables_dir / "s1_pwlf_breakpoints.csv"
        df_pwlf.to_csv(pwlf_path, index=False)
        print(f"  [OK] Saved {pwlf_path} ({len(df_pwlf)} rows)")
    print(f"  [OK] Saved {eapc_joinpoint_path} ({len(eapc_combined)} rows)")

    # 4. Attributable shares of Total COPD and All-cause HAP
    print("  Computing attributable COPD shares...")
    df_shares = compute_attributable_shares(df_hap, df_tot, df_all)
    shares_path = tables_dir / "s1_attributable_share.csv"
    df_shares.to_csv(shares_path, index=False)
    print(f"  [OK] Saved {shares_path} ({len(df_shares)} rows)")

    # 5. Visualisations
    print("  Rendering publication figures...")
    f1 = plot_trends_southasia(df_hap, figures_dir / "s1_trends_southasia.png")
    print(f"  [OK] Rendered {f1}")

    f2 = plot_ind_npl_comparison(df_hap, figures_dir / "s1_ind_npl_comparison.png")
    print(f"  [OK] Rendered {f2}")

    f3 = plot_joinpoints(df_hap, figures_dir / "s1_joinpoints.png")
    print(f"  [OK] Rendered {f3}")

    f4 = plot_hap_copd_share(df_shares, figures_dir / "s1_hap_copd_share.png")
    print(f"  [OK] Rendered {f4}")

    print("=== Stage 1 Execution Complete ===")
    return {
        "summary_table": summary_path,
        "eapc_table": eapc_joinpoint_path,
        "shares_table": shares_path,
        "figures": [f1, f2, f3, f4],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 1 Historical Analysis")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config")
    args = parser.parse_args()
    run_stage1(args.config)
