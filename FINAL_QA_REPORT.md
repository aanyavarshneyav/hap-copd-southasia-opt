# B-11 Final Scientific Consistency & Submission QA Report

**Branch:** `b/b10-final-integration` | **Date:** `2026-11-17` | **Baseline Contracts:** `GBD_2023_WPP_2024` (Sex-Reconciled Derived) + `REAL_V1_2024USD`

---

## 1. Scientific Corrections Made
1. **Sex-Specific Forecast & 500-Draw Reconciliation (`src/optimise/reconcile_sex.py`, `DEC-026`):**
   - **Root Cause Diagnosed:** In raw `data/processed/forecasts_stage3.csv`, Person A fit a single `HistGradientBoostingRegressor` (`src/forecast/projections.py`) with `sex_code ∈ {0, 1, 2}` and predicted `female`, `male`, and `both` ASRs independently before multiplying by UN WPP 2024 sex-specific populations. Because decision-tree splits are nonlinear and unconstrained across `sex_code`, `female + male` exceeded authoritative `both` across all `256` `(unit × year × scenario × measure)` strata—by **`+50,254.13` DALYs (`+1.86%`)** in 2040 BAU (`2,750,091.39` F+M sum vs. `2,699,837.26` authoritative `both`). Concurrently, `stage2_uncertainty_draws.csv` sampled independent noise across sexes (draw-level `|F+M−both|` up to `961,988` DALYs in `IND_rural`), artificially doubling draw equity dispersion.
   - **Resolution Applied Without Modifying Person A's Raw/Source Files:** Created [`src/optimise/reconcile_sex.py`](src/optimise/reconcile_sex.py) and [`tests/optimise/test_reconcile_sex.py`](tests/optimise/test_reconcile_sex.py), preserving original `forecasts_stage3.csv`, `baseline_unit_state.csv`, and `stage2_uncertainty_draws.csv` untouched while exporting clearly labelled derived files (`forecasts_stage3_reconciled.csv`, `baseline_unit_state_reconciled.csv`, `stage2_uncertainty_draws_reconciled.csv`, `outputs/tables/s2_sex_reconciliation_audit.csv` + `.meta.json`). Reconciled splits apply documented WHO GHE 2021 ratios (`WHO_GHE_SEX_SPLITS`: India DALYs `53.46% F / 46.54% M`, Nepal `58.34% F / 41.66% M`) and WPP 2024 sex populations.
   - **Downstream Impact & Regeneration:** Reconciled 2040 BAU equity gap ASR increased **+151.3%** (`12.50 → 31.41`/100k) and BAU 2040 DALYs decreased `-50,254` (`2,750,091.39 → 2,699,837.26`). Pointed [`configs/default.yaml`](configs/default.yaml) to reconciled contracts, verified `calibrate_lambda_from_pareto` (`median_quantile_slope = 115.52 ≈ 116.0`), and regenerated all Stage 3 NSGA-II (`265` plans), Stage 4 MDP/HTN/PDDL, `compare.py` (`1,000` rollouts), `final_real_suite.py` (`500` draws + `18` sensitivities + Opex + bundle), website JSONs/HTML/JS, and reports.
2. **Stage 2 CV Diagnostic Clarification (`src/forecast/models.py`):** Identified why `arima` and `ets` reported identical `49.266` MAE (`max |arima − ets| = 0.0`): `ExponentialSmoothing.fit(disp=False)` raises `TypeError` in `statsmodels 0.14`, causing `fit_predict_ets` to fall back identically to `fit_predict_arima`. Left `src/forecast/` unmodified per ownership rules and disclosed this explicitly in [`report/report.md`](report/report.md), [`report/executive_summary.md`](report/executive_summary.md), [`site/scripts/prepare_data.py`](site/scripts/prepare_data.py), and [`site/assets/js/pages/forecast.js`](site/assets/js/pages/forecast.js).
3. **Test Separation Without Suppression (`Makefile`, `docs/REPRODUCIBILITY.md`):** Separated external raw-archive integration (`make test-raw`) from fixture-based tests (`make test`), documenting exact acquisition URLs and byte sizes for all 11 raw archives in `data/raw/manifest.json` while leaving Person A's `tests/historical/test_historical.py` intact.

---

## 2. Remaining Methodological Limitations
1. **Modelled Epidemiology & Aggregate Age Compromise:** GBD 2023 series are modelled estimates, not vital-registry counts; missing 5-year age bins forced all-age counts + ASRs + reduced-form `poisson_rate_glm` instead of APC decomposition.
2. **Counterfactual vs. Trend Model Trade-Off & `IND_urban` Saturation:** Univariate `ARIMA` beats `HistGBDT Panel` on CV accuracy (`49.27` vs. `114.25` MAE), while `HistGBDT` cannot extrapolate unbounded temporal trends (`year <= origin`) and reaches terminal leaf saturation (`slope = 0.0`) above `0.985` clean fuel in `IND_urban`.
3. **Proxy Sex Splits & Intervention Elasticity:** Both baseline sex splits (`WHO_GHE_SEX_SPLITS`) and differential clean-cooking responsiveness (`r_fm = 1.05`) are uniform proxies rather than directly measured sex-stratified HAP–COPD trials.
4. **Unverified Infrastructure Costs (`infra_lpg_grid`):** Remains `null` in active `REAL_V1_2024USD` (`Scenario A` disabled; bracketed in `Scenario B1/B2`).
5. **Programme Cost Boundary (`Alt A` vs. `Alt B`):** Base `Scenario A` treats LPG support as a 1-time 5-year enrolment cost; recurring multi-period cylinder refill subsidies (`Alt B`) add `+29.0%` to Knee cost (`\$2,259.40M`) and `+72.0%` to DALY-Minimal cost (`\$5,289.50M`).

---

## 3. Forecast Performance Interpretation (`s2_rolling_cv_metrics.csv`)
- **Verified CV Means (`81` origin × location × sex pairs across `2009–2013`, `2014–2018`, `2019–2023`):** `arima = 49.266` MAE (`57.019` RMSE, `2.905` MASE), `ets = 49.266` (fallback to `ARIMA`), `naive_baseline = 103.688` (`5.620` MASE), `hist_gbdt_panel = 114.250` (`6.535` MASE), `poisson_rate_glm = 123.733` (`6.457` MASE).
- **MASE Benchmark Definition (`compute_mase`, `src/forecast/cv.py`):** Divides `5`-year-ahead multi-step test MAE by `1`-year in-sample mean absolute first difference `mean(|y_t − y_{t−1}|)`. Smooth downward epidemiological trajectories accumulate multi-year drift (`h = 1..5`), causing even flat persistence (`naive_baseline`) to reach `MASE = 5.62` and `ARIMA(1,1,0)` to reach `2.91`.
- **Justification for Retaining `HistGBDT Panel`:** Historical extrapolation (`ARIMA`) ignores `clean_fuel_share` and therefore produces zero counterfactual leverage (`cf_25..cf_100`); `HistGBDT` learns cross-sectional/panel clean-fuel gradients (`-2.28` `IND_rural`, `-2.86` `NPL_urban`, `-3.73` `NPL_rural`) needed strictly for Stage 3 exposure–response calibration (outperforming `ARIMA` in `IND_rural` CV: `100.00` vs. `111.52`), while `IND_urban` plateauing above `0.985` is documented as a leaf-saturation ceiling.

---

## 4. Reconciled Sex-Specific Projections Summary (`s2_sex_reconciliation_audit.csv`)
- **Pre-reconciliation 2040 BAU `F + M − both` discrepancy:** `+50,254.13` DALYs (`IND_rural +28,075.23`, `IND_urban +15,221.14`, `NPL_rural +6,522.99`, `NPL_urban +434.77`).
- **Post-reconciliation max discrepancy (`256` strata):** `0.0000` on `forecasts_stage3_reconciled.csv` `Number`; `0.0100` across `2,000` `(unit × draw)` pairs in `stage2_uncertainty_draws_reconciled.csv`; `<= 0.05` across `baseline_unit_state_reconciled.csv`.
- **Reconciled 2040 BAU Totals:** `both = 2,699,837.26` (`female = 1,454,011.92`, `male = 1,245,825.34`; BAU weighted equity gap `31.4099`/100k).

---

## 5. GA / MDP / PDDL Fairness Verification (`comp_ga_vs_mdp.csv`)
- **Verified Shared Parameters:** All paradigms operate under `Scenario A` (`\$5,985.73M` budget cap, `infra_lpg_grid` disabled), identical 3-period horizon (`P1–P3`), identical CRN shock matrix (`N = 1,000`, `seed = 42`, multipliers `[0.5×, 1.0×, 1.25×]`, probabilities `[0.15, 0.70, 0.15]`), identical `λ = 116.0` and `γ = 0.774`, and identical evaluator cost/effectiveness coefficients.
- **Confirmed Violation Rates (`0.0%` MDP, `40.0%` GA, `26.9%` PDDL):** Zero violations occur on budget (`0.0%`) or infrastructure prerequisites (`0.0%`). Open-loop GA/PDDL apply shocks **without execution-time clamping** (`viol_cap = 0.400`, `viol_ceil = 0.267–0.348` over `3` active periods → `1 − 0.85^3 = 38.6%`; `26.9%` over `P1–P2` PDDL → `1 − 0.85^2 = 27.8%`), whereas `mdp.py` clamps realized steps inside `[cap_bins, max_bin − raw_u]` and re-optimises `P2–P3` given realized coverage.
- **Fair Balanced Interpretation:** Continuous open-loop GA achieves higher expected DALYs averted (`217k–453k` vs. `203.7k` MDP), lower equity gap (`24.86–28.12` vs. `28.40`), and higher mean scalarized value (`104,931` matched vs. `65,195` MDP) due to unbinned allocation flexibility, trading higher shock-overshoot exposure against MDP's `0.0%`-violation risk control.
- **PDDL Base vs. B1/B2 Distinction:** Base PDDL (`pddl/problem_ind_npl_existing_infra.pddl`, `0.0798s`, `7` steps, `\$3,262.52M`) reflects existing Scenario A infrastructure readiness (`NPL_rural` unready & blocked from LPG/grid), contrasting with explicit `Scenario B1/B2` proxy infrastructure runs in sensitivity (`+74,850` additional averted DALYs via P1 `NPL_rural` builds).

---

## 6. Headline Numerical Consistency Audit (Reconciled Real Pipeline)
Every table, JSON payload, website banner, `report/report.md`, and `report/executive_summary.md` matches the source CSVs below:

- **2040 BAU (`Deterministic` vs. `500-Draw MC`):** Deterministic `both = F+M`: **`2,699,837.26`** DALYs `[2,030,277.62, 3,369,396.91]`, equity gap `31.41`/100k | 500-draw empirical mean: **`2,714,694`** `[2,242,745, 3,170,924]`, equity gap `31.57`.
- **Budget Cap & Saturation:** `\$5,985.73M` (50% of `\$11,971.47M`); Scenario A saturates at `\$3,076.02M` (`51.4%` of cap).
- **NSGA-II (`265` Pareto pts, `HV = 2.9700e+10`):** Cost-Min (`#0`: `\$68.95M`, `84,040` averted, `2,615,798` DALYs, eq `30.01`); Knee (`#160`: `\$1,750.92M`, `333,513` averted, `2,366,324` DALYs, eq `26.49`, 500-draw joint mean `2,380,646` `[1.95M–2.78M]`); DALY-Min = Equity-Min (`#264`: `\$3,076.02M`, `469,179` averted, `2,230,658` DALYs, eq `24.64`, 500-draw joint mean `2,244,340` `[1.85M–2.61M]`).
- **MDP (`λ = 116.0`, `22,136` states):** `203,720` mean averted DALYs (`2,496,117` mean DALYs `[2.42M–2.55M]`), `\$1,078.83M` mean cost, eq `28.40`, scalarized `65,195`, `0.0%` violations.
- **HTN & PDDL:** HTN `9` leaves (`\$807.36M`, `2,478,516` DALYs, eq `28.05`); PDDL `7` steps (`0.0798s`, `\$3,262.52M`, `2,273,774` deterministic / `2,287,389` CRN mean DALYs, eq `25.35` / `25.54`, `26.9%` violations).
- **Sensitivity Highlights:** Scenario B1/B2 DALY-Min = `544,029–544,049` averted (`+74,850` gain over Scenario A); Alt B recurring LPG opex Knee = `\$2,259.40M` (`1.29×`), DALY-Min = `\$5,289.50M` (`1.72×`).

---

## 7. Test Results & Remaining Test Limitations (Latest Verified Run)
| Check | Command | Exact Result | Notes |
| --- | --- | --- | --- |
| **Ruff Lint** | `.venv/bin/ruff check src tests` | **Passed** (`All checks passed!`) | Zero warnings across `src/` & `tests/` |
| **Ruff Format** | `.venv/bin/ruff format --check src tests` | **Passed** (`45 files already formatted`) | All Python modules & tests formatted |
| **Core Suite** | `.venv/bin/pytest -q tests/data tests/optimise tests/planning tests/forecast` | **`59 passed`** (`0` skipped, `0` failed) | Includes new `test_reconcile_sex.py` |
| **Repo Fixture Tests** | `make test` (`pytest -k "not test_stage1_pipeline_run"`) | **`59 passed, 3 skipped, 1 deselected`** | Excludes raw-archive-dependent integration test |
| **Historical Raw Test** | `make test-raw` (`pytest tests/historical/test_historical.py -k test_stage1_pipeline_run`) | **`1 failed, 3 deselected`** | Fails strictly on missing gitignored `data/raw/WPP2024_...` (`FileNotFoundError`) |
| **Full Pytest** | `.venv/bin/pytest` | **`59 passed, 3 skipped, 1 failed`** | `3` interim-parquet tests skip; `test_stage1_pipeline_run` fails without `data/raw/` |
| **Real Contracts** | `load_stage3_contracts(load_config('configs/default.yaml'))` | **Passed (`is_mock: False`)** | Validates reconciled Scenario A contracts |
| **Mock Pipeline** | `load_stage3_contracts(load_config('configs/mock.yaml'))` | **Passed (`is_mock: True`)** | Mock fixtures & `_mock` outputs isolated from real data |
| **Website Static Check** | In-process `ThreadingHTTPServer` over `site/` | **`86 URLs` `200 OK`, `0` failures** | Automated Chromium visual rendering unavailable (`npx` absent) |

All external raw-archive acquisition steps (`11` files, `>930 MB` in `data/raw/manifest.json`) are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) without deleting or unconditionally skipping `test_stage1_pipeline_run`.

---

## 8. Website Verification Status
- Verified offline regeneration (`site/scripts/prepare_data.py` -> 8 JSON payloads + 49 downloads + 18 figures in `site/`), matching schemas against all 7 page modules (`overview.js`, `historical.js`, `forecast.js`, `assumptions.js`, `optimise.js`, `planning.js`, `research.js`), and offline HTTP 200 serving across all 86 static/JSON/asset URLs.
- **Visual Browser Limitation Disclosure:** Automated Chromium inspection via `chrome_devtools` MCP could not run because `npx`/`node` is not installed on this machine (even though `/Applications/Google Chrome.app` exists). Visual layout rendering in Chromium was therefore **not** executed via automated MCP browser tools.

---

## 9. Final Report Status
- [`report/report.md`](report/report.md) and [`report/executive_summary.md`](report/executive_summary.md) are updated, cross-checked, free of duplicate sections, preserve consistent B-09/B-11 terminology, and cite exact verified CSV/JSON numbers, URLs, DOIs, and disclosures.

---

## 10. Exact Remaining Submission Tasks
1. Optionally perform a manual visual spot-check of `http://127.0.0.1:8000` (`./site/start_site.sh`) in `/Applications/Google Chrome.app`.
2. Export `report/report.md` to final university PDF/DOCX submission formatting if required by course rules.
3. Review `git status` / `git diff` on `b/b10-final-integration`, obtain teammate sign-off (Rudraksh) on any shared config/Makefile changes (`configs/default.yaml`, `Makefile`), and commit + merge via PR into `main` (without committing gitignored `data/raw/*` or large outputs).
