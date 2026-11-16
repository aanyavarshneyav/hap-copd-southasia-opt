# AGENTS.md — Repository Source of Truth

Read this file first before touching code, data, or docs. Follow `docs/` for detailed specs.

## 1. Project Summary & Research Questions
This two-person university AI Lab project forecasts Household Air Pollution (HAP)-attributable COPD burden (deaths and DALYs) in South Asia from 2024 to 2040 using GBD 2023, UN WPP 2024, and WHO SDG 7.1.2 clean-cooking data. Historical trends (1990–2023) and forecasts are estimated over the 5-country GBD South Asia panel (India, Nepal, Pakistan, Bangladesh, Bhutan) in 100% Python. A decision layer then allocates a limited clean-cooking budget across India and Nepal (each split urban/rural = 4 units), 4 interventions, and three 5-year periods (2026–2040). It combines NSGA-II multi-objective optimisation (`pymoo`), a custom NumPy MDP, a Python HTN tree, and STRIPS/PDDL (`pyperplan`). All runs target CPU-only laptops and complete within a few minutes.

**Research questions:**
1. How have HAP-attributable COPD deaths and DALYs changed in India, Nepal and wider South Asia from 1990 to 2023, by age and sex?
2. Which forecasting approach (statistical or ML) projects attributable burden to 2040 most reliably on held-out data?
3. Under a fixed budget, which intervention allocation minimises projected 2040 burden, and how does it trade off against cost and sex-based equity?
4. How does an adaptive planning policy (MDP) compare with a one-off optimised allocation (GA)?

## 2. Repository Map & Directory Ownership
| Path | Stage / Purpose | Owner |
| --- | --- | --- |
| `data/{raw,interim,processed}/` | Immutable raw exports, cleaned tables, hand-off contracts | Person A (Rudraksh) |
| `src/data/` | Ingestion, validation, urban/rural apportionment | Person A |
| `src/historical/` | Stage 1: ASRs, EAPC, piecewise/joinpoint, shares | Person A |
| `src/forecast/` | Stage 2: ARIMA, ETS, Poisson APC, ML panel, surrogate | Person A |
| `src/optimise/` | Stage 3: NSGA-II (`pymoo`) & Pareto evaluation | Person B (Aanya) |
| `src/planning/` | MDP value iteration (NumPy), HTN tree, PDDL runner | Person B |
| `pddl/` | STRIPS domain & problem files (`pyperplan`) | Person B |
| `src/viz/` | Stage 1–2 plots (A) and Stage 3/planning plots (B) | Split by file (`historical.py`/`forecast.py` = A; `decision.py` = B) |
| `outputs/{figures,tables,models}/` | Generated artifacts (`s1_*`, `s2_*` = A; `s3_*`, `plan_*` = B) | Split by prefix |
| `tests/` | Unit & contract tests (`data/`, `historical/`, `forecast/` = A; `optimise/`, `planning/` = B) | Split by subdir |
| `configs/`, `Makefile`, `pyproject.toml` | Shared config, seeds, environment & pipeline commands | Shared (coordinate first) |
| `docs/`, `report/`, `notebooks/` | Specs, logs, Markdown report scaffold, exploratory scratch | Shared (`WORKPLAN` / `DECISIONS` / `ASSUMPTIONS`) |

## 3. Environment & Run Commands
Python 3.12 only (`venv` + `pyproject.toml`, no `conda`, `uv`, or `R`).
```bash
make setup          # python3.12 -m venv .venv && pip install -e ".[dev]"
make lint           # .venv/bin/ruff check & format --check on src/ and tests/
make format         # .venv/bin/ruff format src tests
make test           # .venv/bin/pytest
make check          # lint + test
```
Stage entrypoints (implemented in Phases 1–3; all read `configs/default.yaml`):
```bash
make validate-data  # python -m src.data.validate --config configs/default.yaml
make stage1         # python -m src.historical.run --config configs/default.yaml
make stage2         # python -m src.forecast.run --config configs/default.yaml
make stage3         # python -m src.optimise.run --config configs/default.yaml
make planning       # python -m src.planning.run --config configs/default.yaml
make all            # validate-data -> stage1 -> stage2 -> stage3 -> planning
```

## 4. Coding Standards
- **Python 3.12 type hints** on all public functions, dataclasses, and return types.
- **Small, single-purpose functions** with concise docstrings; keep logic modular and readable.
- **Config-driven:** Read paths, horizons, interventions, and hyperparameters from `configs/default.yaml`. Never hard-code file paths or magic numbers inside `src/`.
- **Fixed seeds:** Set seeds from `configs/default.yaml` (`seeds.*`) for NumPy, scikit-learn, and `pymoo`.
- **Pure transforms:** Data-cleaning and feature functions must be deterministic and side-effect free (read/write I/O lives in CLI/pipeline wrappers).
- **Reproducibility & speed:** Every figure/table must regenerate from `make` or a documented `python -m ...` command in under a few minutes on a CPU-only laptop.

## 5. Data Rules
- **Raw data is immutable:** Never edit files in `data/raw/`. Record source, download date, version, and SHA-256 in `data/raw/manifest.json`.
- **Derived outputs:** Write cleaned tables to `data/interim/` and analysis/contract tables to `data/processed/`.
- **No large commits:** `.gitignore` excludes `data/raw/*`, `data/interim/*`, `data/processed/*` (except `*_mock.*` fixtures and `.gitkeep`), and `outputs/*` binaries. Never force-add large files.
- **Metadata provenance:** Every exported table and contract artifact must carry or be paired with `source`, `download_date` (or `generated_at`), and `data_version` (e.g., `GBD_2023`).

## 6. Honesty & Scientific Integrity Rules
- **Never invent numbers:** Do not fabricate burden counts, costs, effect sizes, adoption rates, or citations.
- **Assumptions register:** Any assumed parameter (costs, effectiveness, adoption caps, apportionment weights, exposure–response parameters) must be logged in `docs/ASSUMPTIONS.md` with a verified source or marked `placeholder: needs literature`.
- **GBD framing:** GBD 2023 values are modelled estimates, not observed counts. Always propagate uncertainty intervals (`lower`, `upper`) where applicable.
- **Scenario framing:** Intervention costs and effects are scenario assumptions; never present outputs as real-world policy recommendations.

## 7. Git Workflow
- Create one branch per task using owner prefixes: `a/<task-slug>` for Person A, `b/<task-slug>` for Person B, `shared/<task-slug>` when agreed.
- Make small, focused commits with clear messages.
- **Never push directly to `main`.** Open a PR or merge cleanly after `make check` passes.
- **Respect ownership:** Never modify the other person's owned `src/`, `pddl/`, `data/`, or `tests/` files without asking first. When modifying `docs/INTERFACES.md`, notify the teammate before changing schemas.

## 8. Definition of Done (Every Task)
- [ ] `make check` (`ruff` lint/format + `pytest`) passes with zero warnings/failures.
- [ ] New modules/functions include Python 3.12 type hints and docstrings.
- [ ] Generated tables/contracts include metadata (`source`, date, `version`, `is_mock` where applicable).
- [ ] Relevant docs (`docs/WORKPLAN.md`, `docs/ASSUMPTIONS.md`, `docs/DECISIONS.md`, `docs/INTERFACES.md`) are updated.
- [ ] Outputs follow naming conventions (`s1_*`, `s2_*`, `s3_*`, `plan_*`) and run cleanly via `make`.

## 9. When Blocked
Stop immediately and ask the user or teammate. Do **not** guess missing GBD filters, invent literature values, alter interface contracts silently, or switch tools/languages outside the agreed Python 3.12 stack.
