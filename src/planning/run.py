"""Unified Stage 4 planning CLI entrypoint (``make planning``).

Runs MDP (B-03), HTN (B-04), and STRIPS/PDDL (B-04) pipelines against the
shared configuration and contracts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.planning.htn import run_htn_pipeline
from src.planning.mdp import run_mdp_pipeline
from src.planning.pddl_runner import run_pddl_pipeline


def run_planning(
    config_path: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute MDP, HTN, and STRIPS/PDDL pipelines end-to-end."""
    mdp_out = run_mdp_pipeline(config_path, base_dir=base_dir, generate_figure=True)
    htn_out = run_htn_pipeline(config_path, base_dir=base_dir, generate_figure=True)
    pddl_out = run_pddl_pipeline(config_path, base_dir=base_dir)
    return {
        "mdp": mdp_out,
        "htn": htn_out,
        "pddl": pddl_out,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m src.planning.run``."""
    parser = argparse.ArgumentParser(description="Run Stage 4 planning (MDP, HTN, STRIPS/PDDL).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args(argv)

    out = run_planning(args.config)
    mdp_roll = out["mdp"]["rollouts_df"]
    htn_ev = out["htn"]["evaluation"]
    pddl_ev = out["pddl"]["evaluation"]

    print(
        f"Planning complete (is_mock={out['mdp']['is_mock']}): "
        f"MDP={out['mdp']['runtime_seconds']:.2f}s "
        f"(mean DALYs averted={mdp_roll['dalys_averted_mean'].mean():,.0f}, "
        f"mean cost=${mdp_roll['total_cost_usd_millions'].mean():.1f}M) | "
        f"HTN={len(out['htn']['leaves'])} leaves "
        f"(cost=${float(htn_ev.total_cost_usd_millions):.1f}M, feasible={htn_ev.is_feasible}) | "
        f"PDDL={len(out['pddl']['actions'])} steps "
        f"(cost=${float(pddl_ev.total_cost_usd_millions):.1f}M, feasible={pddl_ev.is_feasible})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
