# Reproducibility & Execution Guide

This repository reproduces the complete South Asia HAP–COPD decision pipeline in **Python 3.12** on CPU-only hardware with fixed random seeds (`42`).

## 1. Environment Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

All commands below assume execution from the repository root with `.venv/bin/`.

---

## 2. Prepared-Fixture Decision & Planning Reproduction (`configs/default.yaml`)

Validates all Stage 1–3 processed interface contracts (`GBD_2023_WPP_2024` + `REAL_V1_2024USD`) and executes sex-specific reconciliation (`src.optimise.reconcile_sex`), Stage 3 `NSGA-II` (`5` seeds), Stage 4 `MDP` / `HTN` / `PDDL`, `GA` vs. `MDP` stochastic rollout comparison, 500-draw joint Monte Carlo re-evaluation, 8-factor real sensitivity suite (`18` scenarios + Alt B opex), and builds the offline research website:

```bash
# 1. Reconcile Stage 2 sex-specific forecasts & 500 draws to authoritative 'both' via WHO GHE 2021
.venv/bin/python -m src.optimise.reconcile_sex --config configs/default.yaml

# 2. Validate active processed Stage 1-3 interface contracts against configs/default.yaml
.venv/bin/python -c "from src.data.contracts import load_config, load_stage3_contracts; load_stage3_contracts(load_config('configs/default.yaml'))"

# 3. Stage 3: Multi-objective NSGA-II + 3 heuristic baselines (Scenario A, $5,985.73M cap)
.venv/bin/python -m src.optimise.run --config configs/default.yaml

# 4. Stage 4: MDP (lambda=116.0), HTN decomposition, and pyperplan PDDL (existing infra problem)
.venv/bin/python -m src.planning.run --config configs/default.yaml

# 5. Stage 4 Comparator: 1,000 CRN shock rollouts (MDP b=6/11 vs matched GA vs PDDL)
.venv/bin/python -m src.planning.compare --config configs/default.yaml

# 6. Final B-10/B-11 uncertainty re-eval (500 draws), 8-factor sensitivity (18 scenarios), Alt B opex & curated bundle
.venv/bin/python -m src.optimise.final_real_suite --config configs/default.yaml

# 7. Rebuild offline website JSON payloads & downloadable asset bundle
.venv/bin/python site/scripts/prepare_data.py
```

Or run Stage 3 + Planning + Compare via `Makefile`:
```bash
make stage3 && make planning && make compare
```

Curated real outputs are exported to `outputs/final_real_data/` (with `manifest.json`). Raw Stage 2 exports (`data/processed/forecasts_stage3.csv`, `data/processed/baseline_unit_state.csv`, `data/processed/stage2_uncertainty_draws.csv`) are preserved unmodified alongside the derived reconciled files (`*_reconciled.csv`).

---

## 3. Isolated Mock Pipeline Reproduction (`configs/mock.yaml`)

The mock fixture pipeline remains completely separate (`_mock` filenames, `is_mock: true`, `use_mock_forecasts: true`, `cost_weight_lambda: 3500.0`) so synthetic contract regression tests never overwrite real outputs:

```bash
.venv/bin/python -m src.optimise.run --config configs/mock.yaml
.venv/bin/python -m src.planning.run --config configs/mock.yaml
.venv/bin/python -m src.planning.compare --config configs/mock.yaml
```

---

## 4. Upstream Raw Archives Required for Full Stage 0–1 Reproduction (`data/raw/`)

Per `AGENTS.md` §5, external raw archives (>930 MB total) and intermediate `.parquet` files (`data/interim/*.parquet`) are excluded via `.gitignore`. Prepared Stage 1–2 processed artifacts (`data/processed/*`, `outputs/tables/s1_*.csv`, `outputs/tables/s2_*.csv`) are present and validated on disk. To re-run Stage 0–1 from raw source archives (`make validate-data && make stage1 && make stage2 && make test-raw`), obtain the 11 files logged in `data/raw/manifest.json`:

| Filename (`data/raw/...`) | Status on Disk | Source & Acquisition URL / Tool | Size (bytes) |
| --- | --- | --- | ---: |
| `WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv` | **Restored (`2026-11-17`)** | UN DESA WPP 2024 (`https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz`) | `418,121,725` |
| `WPP2024_PopulationBySingleAgeSex_Medium_2024-2100.csv` | **Restored (`2026-11-17`)** | UN DESA WPP 2024 (`https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/WPP2024_PopulationBySingleAgeSex_Medium_2024-2100.csv.gz`) | `440,981,531` |
| `EG.CFT.ACCS.json`, `EG.CFT.ACCS.RU.json`, `EG.CFT.ACCS.UR.json` | Missing (API timed out `2026-11-17`) | World Bank Open Data API (`https://api.worldbank.org/v2/country/indicator/EG.CFT.ACCS*.ZS`) | `129,974` / `138,375` / `138,439` |
| `IHME-GBD_2023_DATA-010d0c57-1.zip` | Missing (interactive GBD portal) | IHME GBD 2023 Results Tool (`https://vizhub.healthdata.org/gbd-results/`) | `198,204` |
| `IHME-GBD_2023_DATA-7546b8be-1.zip` | Missing (interactive GBD portal) | IHME GBD 2023 Results Tool (`https://vizhub.healthdata.org/gbd-results/`) | `79,214` |
| `IHME-GBD_2023_DATA-89044603-1.zip` | Missing (interactive GBD portal) | IHME GBD 2023 Results Tool (`https://vizhub.healthdata.org/gbd-results/`) | `67,411` |
| `data (1).csv` | Missing (interactive WHO portal) | WHO GHO Household Energy DB (`https://www.who.int/data/gho/data/themes/air-pollution/household-air-pollution`) | `71,728,931` |
| `data (2).csv`, `data (3).csv` | Missing (interactive WHO GHE portal) | WHO Global Health Estimates (`https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates`) | `7,683` / `6,842` |

### Restored UN WPP 2024 Checksums & Reproducibility Fixes (`2026-11-17`)
- **UN WPP 2024 Medium Variant Single-Age Population (`1950-2023`):**
  - Compressed archive: `WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz` (`62,082,217` bytes, SHA-256 `f84c75789ccbd385122ad19bac70026d6306878442afbaa24e8e50a23b4bee2f`).
  - Extracted CSV: `data/raw/WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv` (`418,121,725` bytes, `4,148,070` data rows, SHA-256 `9e23aa12cfb8d4090cc0fc19b2d05cdc0df561142a55a3d8971c2d07578a55f6` — exact match to `data/raw/manifest.json`).
- **UN WPP 2024 Medium Variant Single-Age Population (`2024-2100`):**
  - Compressed archive: `WPP2024_PopulationBySingleAgeSex_Medium_2024-2100.csv.gz` (`66,954,143` bytes, SHA-256 `31804a296b663716236cd26c46415271cc9386fb3b2be56aff6467ad32283dc8`).
  - Extracted CSV: `data/raw/WPP2024_PopulationBySingleAgeSex_Medium_2024-2100.csv` (`440,981,531` bytes, `4,316,235` data rows, SHA-256 `b6ed024941d5a045bf25a350e1cfed0ec8949e2357111167c98ab9d846bc755f` — exact match to `data/raw/manifest.json`).
- **Reproducibility Fixes Applied:**
  - `src/data/ingest.py`, `src/forecast/projections.py`, and `src/data/validate.py` now accept both flat extracted `data/raw/WPP2024_*.csv` files and Person A's original nested `WPP2024_*.csv/WPP2024_*.csv` directories (matching `src/data/validate_raw.py`).
  - Added `pyarrow>=17.0` to `pyproject.toml` dependencies so `to_parquet` / `read_parquet` in `ingest.py` executes cleanly in fresh `.venv` installs.

Verify raw checksums and run Stage 1–2 integration tests after placing the remaining 9 files under `data/raw/`:
```bash
make validate-data
make test-raw
```

---

## 5. Offline Research Website

Serve `site/` locally without external internet access:

```bash
./site/start_site.sh
# Or directly:
.venv/bin/python -m http.server 8000 --directory site
```

---

## 6. Linting, Formatting, and Test Separation (`Makefile`)

- `make lint`: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests`
- `make test`: `.venv/bin/pytest -k "not test_stage1_pipeline_run"` (runs core fixture, contract, reconciliation, optimisation, planning, and forecast tests cleanly without external raw archives; interim-parquet tests in `tests/historical/test_historical.py` skip automatically when `data/interim/historical_burden.parquet` is absent).
- `make test-raw`: `.venv/bin/pytest tests/historical/test_historical.py -k test_stage1_pipeline_run` (end-to-end Stage 1 raw ingestion test; requires `data/raw/*`).
- `make test-all`: `.venv/bin/pytest` (all tests combined).
- `make check`: runs `make lint` followed by `make test`.
