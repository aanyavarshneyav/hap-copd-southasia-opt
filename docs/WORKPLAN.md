# Project Workplan & Milestones

## 1. Track Split & Ownership
- **Person A (Rudraksh) — Data & Forecasting Track (`a/...`):**
  - Owns: `data/`, `src/data/`, `src/historical/`, `src/forecast/`, `src/viz/{historical,forecast}.py`, `tests/{data,historical,forecast}/`.
- **Person B (Aanya) — Decision & Planning Track (`b/...`):**
  - Owns: `src/optimise/`, `src/planning/`, `pddl/`, `src/viz/decision.py`, `tests/{optimise,planning}/`.
- **Shared (`shared/...`, coordinate first):** `report/`, `docs/ASSUMPTIONS.md`, `docs/DECISIONS.md`, `docs/INTERFACES.md`, `docs/WORKPLAN.md`, `configs/`, `Makefile`.

---

## 2. Milestones (Dependency-Ordered; Dates Unscheduled)
> Calendar dates are not fixed yet; milestones below track dependency gates rather than deadlines.

| Milestone | Dependency Gate | Exit Criteria |
| --- | --- | --- |
| **M0: Repo, Interface & Mock Lock** | **Start (First Hour)** | Phase 0 merged; `configs/default.yaml` & `INTERFACES.md` locked; Aanya (or Aanya + Rudraksh together in hour 1) builds `*_mock.*` fixtures so Stage 3 / Planning work starts immediately. |
| **M1: Data Validated & Mock Pipeline Working** | **Post-M0** | Raw exports logged in `manifest.json`; `make validate-data` passes; Stage 1 EAPC done; Aanya has working NSGA-II + MDP + HTN + PDDL running on mock data. |
| **M2: Forecasting Complete & Real Handoff** | **Post-M1** | Stage 1 figures complete; Stage 2 CV benchmarks complete; real `forecasts_stage3.csv` + `surrogate.json` pass schema tests; flip `use_mock_forecasts: false`. |
| **M3: Final Experiments & Sensitivity Freeze** | **Post-M2** | Literature assumptions sourced in `ASSUMPTIONS.md`; final Stage 3 Pareto front, MDP vs GA comparison, HTN/PDDL outputs, and sensitivity checks complete. |
| **M4: Report & Final Reproduction** | **Post-M3** | `report/report.md` complete & converted to final format (`TODO(owner)`); `make check` and `make all` pass cleanly from a fresh `.venv`. |

---

## 3. Master Task Table

| ID | Task | Owner | Depends On | Output File | Status |
| --- | --- | --- | --- | --- | --- |
| `P0-01` | Phase 0 repository & documentation scaffolding | Setup | — | `AGENTS.md`, `docs/*.md`, `pyproject.toml` | `done` |
| `A-01` | Lock Python 3.12 `.venv` (`make setup`) & smoke-test `configs/default.yaml` loader | Rudraksh (A) | `P0-01` | `src/data/config.py`, `tests/test_config.py` | `todo` |
| `B-00` | **First-Hour Priority:** Build `*_mock.*` fixtures matching `INTERFACES.md` (with `A-02` schema validator alongside/paired with Rudraksh) | Aanya (B) *(or paired w/ Rudraksh)* | `P0-01` | `data/processed/*_mock.*` | `done` |
| `A-02` | Implement reusable contract schema validator (`src/data/contracts.py`) verifying mock & real parity | Rudraksh (A) *(Phase 1 impl by B)* | `A-01`, `B-00` | `src/data/contracts.py`, `tests/data/test_contracts.py` | `done` |
| `A-03` | Download GBD 2023 (HAP-COPD, Total COPD, All-cause HAP), UN WPP 2024, WHO SDG 7.1.2 & log `manifest.json` | Rudraksh (A) | `P0-01` | `data/raw/*.csv`, `data/raw/manifest.json` | `done` |
| `A-04` | Build ingestion, `validate-data`, and urban/rural apportionment pipeline (`ASM-01`) | Rudraksh (A) | `A-03` | `src/data/ingest.py`, `src/data/validate.py`, `data/interim/*.parquet` | `done` |
| `A-05` | Stage 1 historical analysis: ASRs, EAPC, `pwlf` joinpoint, shares, heatmaps | Rudraksh (A) | `A-04` | `src/historical/run.py`, `outputs/tables/s1_*.csv`, `outputs/figures/s1_*.png` | `done` |
| `A-06` | Stage 2 models (ARIMA, ETS, Poisson GLM APC, RF/HistGBDT) & rolling-origin CV | Rudraksh (A) | `A-04` | `src/forecast/models.py`, `src/forecast/cv.py`, `outputs/tables/s2_rolling_cv_metrics.csv` | `done` |
| `A-07` | Export 2024–2040 forecasts with UI propagation, `forecasts_stage3.csv`, and `exposure_response_surrogate.json` | Rudraksh (A) | `A-06`, `A-02` | `data/processed/forecasts_stage3.csv`, `data/processed/exposure_response_surrogate.json` | `done` |
| `A-08` | Re-evaluate top Pareto points from Stage 3 on full Stage 2 uncertainty draws | Rudraksh (A) | `A-07`, `B-02` | `outputs/tables/s3_pareto_reeval.csv` | `todo` |
| `B-01` | Build contract reader + fast surrogate objective/constraint evaluator against `*_mock.*` | Aanya (B) | `B-00` | `src/optimise/evaluator.py`, `tests/optimise/test_evaluator.py` | `done` |
| `B-02` | Implement NSGA-II (`pymoo`) problem with budget, adoption & infra constraints | Aanya (B) | `B-01` | `src/optimise/nsga2.py`, `src/optimise/run.py` | `todo` |
| `B-03` | Implement custom NumPy MDP value iteration / backward induction (`src/planning/mdp.py`) | Aanya (B) | `B-01` | `src/planning/mdp.py`, `tests/planning/test_mdp.py` | `todo` |
| `B-04` | Implement plain-Python HTN decomposition tree & STRIPS/PDDL files (`pyperplan`) | Aanya (B) | `P0-01` | `src/planning/htn.py`, `pddl/*.pddl`, `src/planning/pddl_runner.py` | `todo` |
| `B-05` | Build equal-footing MDP vs GA stochastic rollout comparator & Pareto/plan visualisations | Aanya (B) | `B-02`, `B-03` | `src/planning/compare.py`, `src/viz/decision.py`, `outputs/figures/{s3,plan}_*.png` | `todo` |
| `B-06` | Switch `use_mock_forecasts: false`, run final Stage 3 + MDP + sensitivity analyses | Aanya (B) | `A-07`, `B-05` | `outputs/tables/s3_pareto_front.csv`, `outputs/tables/plan_mdp_vs_ga_comparison.csv` | `todo` |
| `S-01` | Literature review to replace `placeholder: needs literature` costs & effectiveness in `ASSUMPTIONS.md` | Shared | `P0-01` | `docs/ASSUMPTIONS.md` | `todo` |
| `S-02` | Draft sections 1–3 of `report/report.md` in parallel; integrate final figures/tables & convert format | Shared | `A-08`, `B-06`, `S-01` | `report/report.md` (+ final submission artifact) | `todo` |
