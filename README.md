# Forecasting & Optimal Intervention Planning for HAP-Attributable COPD Burden in South Asia

A two-person university AI Lab project combining statistical/ML disease-burden forecasting (GBD 2023) with multi-objective genetic optimisation (NSGA-II) and symbolic/sequential planning (MDP, HTN, STRIPS/PDDL) for clean-cooking interventions in India and Nepal (2024–2040).

> **Note:** GBD 2023 estimates are modelled values, not observed counts, and intervention costs/effects are scenario assumptions. Outputs are exploratory scenario analyses rather than policy recommendations.

## Team & Track Split
- **Person A (Rudraksh) — Data & Forecasting Track:** Data ingestion/validation (`data/`, `src/data/`), Stage 1 historical analysis (`src/historical/`), Stage 2 forecasting & surrogate calibration (`src/forecast/`).
- **Person B (Aanya) — Decision & Planning Track:** Stage 3 NSGA-II allocation (`src/optimise/`), MDP value iteration, HTN decomposition tree, STRIPS/PDDL planning (`src/planning/`, `pddl/`), Pareto & comparison plots (`src/viz/decision.py`).
- **Shared:** Report (`report/`), assumptions (`docs/ASSUMPTIONS.md`), decision log (`docs/DECISIONS.md`), interfaces (`docs/INTERFACES.md`), workplan (`docs/WORKPLAN.md`).

## Project Documentation
Start with [AGENTS.md](AGENTS.md) for agent/developer rules, then inspect `docs/`:
- [Project Brief](docs/PROJECT_BRIEF.md) — Objectives, research questions, scope, and limitations.
- [Methodology](docs/METHODOLOGY.md) — Stage 1–3 and planning specs, metrics, and file mapping.
- [Data Spec](docs/DATA_SPEC.md) — Raw dataset sources, exact IHME/UN/WHO export filters, and validation rules.
- [Interfaces](docs/INTERFACES.md) — Track A ↔ Track B hand-off schemas, mock files, and dependency timeline.
- [Planning Spec](docs/PLANNING_SPEC.md) — MDP state/action design, HTN tree, and STRIPS/PDDL encoding.
- [Assumptions](docs/ASSUMPTIONS.md) — Register of intervention costs, effectiveness, apportionment, and placeholders.
- [Workplan](docs/WORKPLAN.md) — Milestones, task dependencies, and parallel schedule.
- [Decisions](docs/DECISIONS.md) — Running log of methodological and tooling decisions.
- [Risks](docs/RISKS.md) — Critical brief review, technical pitfalls, and mitigations.

## Setup (Python 3.12)
The project uses **Python 3.12**, standard `venv`, `pyproject.toml`, `ruff`, and `pytest` (no `conda`, `uv`, or `R`).

```bash
make setup
source .venv/bin/activate
make check
```

## Reproducing the Pipeline
1. **Place raw exports** in `data/raw/` following exact filters and filenames in [docs/DATA_SPEC.md](docs/DATA_SPEC.md), and record provenance in `data/raw/manifest.json`.
2. **Validate data and run Stage 1–2** (Person A):
   ```bash
   make validate-data
   make stage1
   make stage2
   ```
3. **Run Stage 3 optimisation & planning** (Person B — works immediately against `data/processed/forecasts_stage3_mock.csv` when `interfaces.use_mock_forecasts: true` in `configs/default.yaml`):
   ```bash
   make stage3
   make planning
   ```
4. **Full end-to-end reproduction:**
   ```bash
   make all
   ```

All runs are designed to execute CPU-only on standard laptops in under a few minutes.
