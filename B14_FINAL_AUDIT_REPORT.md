# B-14: Final Scientific Audit and Release Preparation Report

**Branch:** `b/b10-final-integration` | **Active Production Surrogate:** `log_ols_smooth` (`slope = −2.1933 ± 0.2557 SE`) | **Parameter Baseline:** `REAL_V1_2024USD` | **Epidemiological Baseline:** `GBD_2023_WPP_2024` (WHO GHE 2021 Sex-Reconciled)

---

## 1. Executive Audit Summary & Development-Freeze Readiness

B-14 audited the post-B-13 computational pipeline across fairness, surrogate validity, uncertainty propagation, raw-data reproducibility, cross-artifact consistency, website integrity, and automated testing without resetting, committing, pushing, or introducing new algorithms.

**Key Audit Resolutions:**
1. **MDP `0.0%` vs. GA `36.5%–40.0%` Violation Asymmetry Resolved (`comp_mdp_clamp_audit_b14.csv`):** The MDP's reported `0.0%` violation rate occurs *strictly* because `_apply_action_branch` (`src/planning/mdp.py:344–362`) clips stochastic `1.25×` high-uptake transitions at `(cap_bins, max_bin - raw_u)` before `simulate_mdp_rollouts` passes realized states to `Stage3Evaluator`. Auditing pre-clamp transitions proves the MDP attempts `1.25×` overshoots on **`40.0%` (`400/1,000`)** of rollouts (`132,017.1` attempted averted DALYs vs. `127,463.3` post-clamp). When open-loop GA schedules undergo symmetric execution clamping, **`ga_matched_best` (`237,528.8` averted)**, **`ga_knee` (`294,245.9`)**, and **`ga_daly_minimal` (`426,023.7`)** also achieve **`100.0%` feasibility (`0.0%` violations)**.
2. **Smooth Log-OLS Surrogate Confirmed Fit for Production:** Pooled `log_ols_smooth` (`R^2 = 0.8938`, `h=5` 4-unit CV MAE `102.34` vs. `151.00` `HistGBDT`) eliminates `HistGBDT`'s discontinuous `124.08/100k` leaf collapse and `IND_urban = 0.0` plateau, avoids `log_ols_interacted`'s Stage-0 `(1 − cf)` apportionment slope inversion (`−4.1482` `IND_urban` vs. `−1.1162` `IND_rural`), and replaces the placeholder `0.08` SD with empirical slope SE `0.2557`. No change to production model selection is warranted.
3. **Uncertainty Propagation Audited Cleanly:** `Stage3Evaluator` quadrature (`np.hypot`) and 500-draw Monte Carlo re-evaluation (`run_pareto_uncertainty_reevaluation`) exhibit zero double-counting. Treating the 8 `(4 units × 2 sexes)` surrogate perturbations independently in MC (`[1,974,879, 2,799,565]` Knee) is slightly narrower (`−0.79%` Knee, `−2.31%` DALY-Min) than a 100%-correlated shared pooled slope draw (`[1,974,293, 2,805,483]`).
4. **Raw-Data Audit Preserved Without Fabrication (`2/11` Restored, `9/11` Missing):** Both `UN WPP 2024` single-age archives (`418,121,725` B and `440,981,531` B) are restored and SHA-256 verified (`170 country-years` pass in `make test-raw`), while `test_stage1_pipeline_run` fails deterministically on missing `data/raw/EG.CFT.ACCS.json`. World Bank JSONs (`3`) timed out on `api.worldbank.org`, and IHME GBD (`3`) / WHO (`3`) require interactive portal downloads.
5. **Cross-Artifact Synchronisation Completed:** Removed stale B-11 `HistGBDT` metrics in [`report/report.md`](report/report.md) (Abstract and Sections 6.1, 9.2, 12, 13), added pre-clamp vs. post-clamp findings to [`report/executive_summary.md`](report/executive_summary.md), updated `ASM-14` in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md), registered `comp_mdp_clamp_audit_b14.csv` in [`site/scripts/prepare_data.py`](site/scripts/prepare_data.py) (`60` downloads), fixed two stale tooltips in [`site/index.html`](site/index.html), and added regression coverage in [`tests/planning/test_mdp.py`](tests/planning/test_mdp.py).

**Verdict:** The codebase, active `log_ols_smooth` data contracts, curated outputs, offline static website, and documentation are **ready for development freeze** for academic submission, subject to transparent reporting of the three World Bank API timeouts and six interactive portal exports absent from `data/raw/` and the lack of `npx` for automated Chromium rendering.

---

## 2. Task 1 — MDP vs. GA Clamping Fairness Audit

### 2.1 Root Cause in Code
- **`src/planning/mdp.py` (`_apply_action_branch`, lines 344–362):** Under `mult_mode == 2` (`1.25×` high-uptake shock), transition states apply `gain = max(0, min(gain, cap_bins, max_bin - raw_u))` before constructing `st_next`. `simulate_mdp_rollouts` (lines 833–882) evaluates `Stage3Evaluator.evaluate` strictly on the **post-clamp** realized increment `delta_bins = st_next - st_curr`.
- **`src/planning/compare.py` (`simulate_open_loop_schedule`, lines 51–149):** Evaluates open-loop schedules as `x_realized = plan_tensor * mult` **without** clipping `1.25×` shocks against per-period adoption caps or unit coverage ceilings.

### 2.2 Pre-Clamp vs. Post-Clamp & Symmetric Execution-Clamped Results (`1,000` CRN Rollouts, `seed=42`)
Verified from [`outputs/final_real_data/tables/comp_mdp_clamp_audit_b14.csv`](outputs/final_real_data/tables/comp_mdp_clamp_audit_b14.csv) and [`outputs/tables/comp_mdp_clamp_audit_b14.csv`](outputs/tables/comp_mdp_clamp_audit_b14.csv):

| Strategy | Evaluation Mode | Feasible (%) | Violated (%) | Budget Viol (%) | Cap Viol (%) | Ceiling Viol (%) | Mean DALYs Averted | Mean Cost (`$M`) | Mean Eq Gap (`ASR`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mdp_adaptive_b6` | `pre_clamp_attempted (unclipped shock)` | `60.0` | `40.0` | `0.0` | `40.0` | `0.5` | `132,017.1` | `603.45` | `29.4369` |
| `mdp_adaptive_b6` | `post_clamp_realized (active transition clamping)` | **`100.0`** | **`0.0`** | `0.0` | `0.0` | `0.0` | `127,463.3` | `600.35` | `29.5133` |
| `ga_matched_best (id=113)` | `open_loop_unclamped (reported comp_ga_vs_mdp)` | `60.0` | `40.0` | `0.0` | `40.0` | `31.6` | `245,520.0` | `1,176.87` | `27.7787` |
| `ga_matched_best (id=113)` | `open_loop_execution_clamped (symmetric clamp)` | **`100.0`** | **`0.0`** | `0.0` | `0.0` | `0.0` | `237,528.8` | `1,090.45` | `27.8940` |
| `ga_knee` | `open_loop_unclamped (reported comp_ga_vs_mdp)` | `63.5` | `36.5` | `0.0` | `29.8` | `31.1` | `300,322.9` | `1,695.91` | `27.0156` |
| `ga_knee` | `open_loop_execution_clamped (symmetric clamp)` | **`100.0`** | **`0.0`** | `0.0` | `0.0` | `0.0` | `294,245.9` | `1,601.50` | `27.1038` |
| `ga_daly_minimal` | `open_loop_unclamped (reported comp_ga_vs_mdp)` | `60.0` | `40.0` | `0.0` | `40.0` | `33.6` | `437,157.1` | `3,097.29` | `25.1392` |
| `ga_daly_minimal` | `open_loop_execution_clamped (symmetric clamp)` | **`100.0`** | **`0.0`** | `0.0` | `0.0` | `0.0` | `426,023.7` | `2,891.56` | `25.2943` |

**Audit Conclusion:** Both paradigms share the exact same `1,000` CRN shock draws, `$5,985.73M` budget cap, cost table, horizon (`P1..P3`), and physical constraints. Pre-clamp MDP attempted overshoots happen identically on `40.0%` of rollouts (`460` period-cap repair events + `15` ceiling repair events across `P1=158`, `P2=164`, `P3=153`; `0.0%` budget overshoots). Under symmetric execution clamping, GA also achieves `0.0%` violations (`max ceil ULP diff = 2.78e-17`) and higher expected DALYs averted (`237.5k–426.0k` vs. `127.5k` MDP), proving neither method holds a strict dominance claim: MDP's genuine comparative advantage is state-contingent closed-loop `P2–P3` recourse, whereas continuous GA avoids 5-bin discretization loss.

---

## 3. Task 2 — Validation of Production Smooth Log-OLS (`log_ols_smooth`)

Side-by-side validation in [`B13_SURROGATE_COMPARISON_REPORT.md`](B13_SURROGATE_COMPARISON_REPORT.md) (`src/forecast/b13_surrogate.py`) confirms pooled `log_ols_smooth` (`log(ASR) ~ t + clean_fuel_share + C(location)`, `N = 136`, `R^2 = 0.8938`, `slope = −2.1933 ± 0.2557 SE`, `p = 2.51e-14`) is strictly preferable for production [`data/processed/exposure_response_surrogate.json`](data/processed/exposure_response_surrogate.json):
- **Versus `hist_gbdt_panel` & `hist_gbdt_monotonic`:** Both tree variants exhibit worse `4-unit h=5` rolling-origin CV MAE (`151.00` and `148.25` vs. **`102.34`**), saturate to slope `0.0` in `IND_urban` above `cf = 0.958`, and collapse all four units onto the `124.08/100k` leaf at `cf = 1.00` (causing the spurious `−3.7254` `NPL_rural` elasticity and requiring placeholder `0.08` SD).
- **Versus `log_ols_interacted`:** While interaction slopes reach `64.52` `h=5` MAE, Stage 0 urban/rural disaggregation constructed sub-national rates proportionally to `(1 − cf)`. Interaction terms absorb this apportionment artifact, producing `−4.1482` in 96%-clean `IND_urban` versus `−1.1162` in 65%-clean `IND_rural`, inverting diminishing-returns HAP exposure physics. Pooled `log_ols_smooth` enforces monotone cross-panel exponential attenuation across all four units and uses exact analytical parameter SE (`0.2557`).

---

## 4. Task 3 — End-to-End Uncertainty Propagation Audit

- **Analytical UI Bounds (`Stage3Evaluator.evaluate_effective_gains`):** Combines Stage 2 BAU prediction half-width (`bau_lo_hw`, `~24.8%` from `[lower, upper]`) and counterfactual slope uncertainty (`surr_hw = dalys_mean_us * 1.96 * slope_se * delta_c`) via quadrature `np.hypot(bau_lo_hw, surr_hw)` — avoiding linear double-counting.
- **500-Draw Joint Monte Carlo (`src/optimise/final_real_suite.py`, `run_pareto_uncertainty_reevaluation`):** Multiplies Stage 2 reconciled trajectory draws (`stage2_uncertainty_draws_reconciled.csv`) by deterministic counterfactual factor and `exp(surr_eps * slope_se * delta_c)`.
- **Independent Cell Draws vs. Shared Pooled Slope Draw (`500` draws, `seed=42`):** Because `log_ols_smooth` estimates one pooled slope `beta = -2.1933 ± 0.2557`, drawing 8 independent `surr_eps` perturbations (`4 units × 2 sexes`) slightly understates `95%` interval width compared to a single shared pooled `beta ~ N(-2.1933, 0.2557^2)` draw:
  - `ga_knee`: 8-cell independent `[1,974,879, 2,799,565]` (`width = 824,686`) vs. 100%-correlated shared slope `[1,974,293, 2,805,483]` (`width = 831,190`, **`+0.79%`**).
  - `ga_daly_minimal`: 8-cell independent `[1,868,489, 2,631,221]` (`width = 762,732`) vs. shared slope `[1,857,568, 2,637,912]` (`width = 780,344`, **`+2.31%`**).
- **Parameter vs. Predictive Residual Scatter:** The Stage 2 predictive scatter (`±24.8%` BAU spread across 500 draws) paired with analytical parameter SE (`0.2557`) is propagated; OLS residual log-SD (`0.2071`) is not added separately on top of BAU predictive draws, consistent with comparative intervention parameter uncertainty.
- **18-Scenario Real Sensitivity Suite (`s3_sensitivity_real.csv` + `s3_recurring_opex_sensitivity.csv`):** All 18 deterministic/planning sensitivity scenarios ran on `log_ols_smooth`, confirming Scenario B1/B2 infra (`+59,562` / `+59,383` DALYs to `513,035` / `512,856`) and `η_biomass` (`0.15 -> 0.50`: `437,586 -> 492,076` DALYs) remain the primary structural sensitivity drivers, while Alt B recurring LPG opex shifts DALY-Min spend by `1.68×` (`$3,097.29M -> $5,192.82M`, still `< $5,985.73M`).

---

## 5. Task 4 — Raw-Data Reproducibility & `data/raw/manifest.json` Audit

Inspected all `11` entries in [`data/raw/manifest.json`](data/raw/manifest.json) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md):
- **Restored & Verified (`2 / 11`):**
  1. `WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv` (`418,121,725` bytes, `4,148,070` rows, SHA-256 `9e23aa12cfb8d4090cc0fc19b2d05cdc0df561142a55a3d8971c2d07578a55f6`) — exact match.
  2. `WPP2024_PopulationBySingleAgeSex_Medium_2024-2100.csv` (`440,981,531` bytes, `4,316,235` rows, SHA-256 `b6ed024941d5a045bf25a350e1cfed0ec8949e2357111167c98ab9d846bc75f`) — exact match.
- **Missing External Archives (`9 / 11`, Not Fabricated):**
  - `EG.CFT.ACCS.json`, `EG.CFT.ACCS.RU.json`, `EG.CFT.ACCS.UR.json` (`World Bank Open Data API`, timed out with `0` bytes after TLS).
  - `IHME-GBD_2023_DATA-010d0c57-1.zip`, `7546b8be-1.zip`, `89044603-1.zip` (`IHME GBD 2023 Results Tool`, interactive portal export).
  - `data (1).csv`, `data (2).csv`, `data (3).csv` (`WHO Household Energy DB` / `WHO GHE 2021`, interactive portal exports).
- **`test_stage1_pipeline_run` Preserved Intact:** Running `make test-raw` extracts WPP population (`170 country-years`) cleanly and fails solely on missing `data/raw/EG.CFT.ACCS.json`.

---

## 6. Task 5 — Cross-Artifact Results Consistency Audit

Cross-checked active Stage 2–4 artifacts and fixed all residual pre-B-13/B-14 discrepancies:
- **Stage 2 Contracts:** `data/processed/forecasts_stage3_reconciled.csv` (`2,699,837.26` DALYs, equity gap `31.41`/100k) and `data/processed/exposure_response_surrogate.json` (`source_model: "log_ols_smooth"`, uniform `slope_clean_fuel: -2.1933`, `uncertainty_sd: 0.2557`) align identically.
- **Preserved HistGBDT Snapshots:** `data/processed/exposure_response_surrogate_histgbdt.json` and `outputs/tables/histgbdt_baseline_snapshot/` remain cleanly isolated and labelled. Historical phase reports (`FINAL_QA_REPORT.md`, `B12_FORECASTING_QA_REPORT.md`) remain as point-in-time records superseded by `B13_SURROGATE_COMPARISON_REPORT.md` and `B14_FINAL_AUDIT_REPORT.md`.
- **Stage 3 & 4 Curated Deliverables (`outputs/final_real_data/`):** `251` Pareto plans (`HV = 2.7965e+10`), Cost-Min (`#0`, `$70.75M`, `59,359` averted), Knee (`#156`, `$1,695.91M`, `311,707` averted, joint MC `2,401,031` `[1,974,879, 2,799,565]`), DALY-Min (`#250`, `$3,097.29M`, `453,474` averted, `24.91`/100k eq gap, joint MC `2,258,315` `[1,868,489, 2,631,221]`), MDP (`λ = 116.0`, `127,463.34` averted at `$600.35M`, `0.0%` post-clamp vs. `40.0%` pre-clamp), HTN (`9` leaves, `$807.36M`), and PDDL (`7` steps, `$3,262.52M`) all match across CSVs, `site/data/*.json`, and reports.
- **Documentation Edits Applied:**
  - [`report/report.md`](report/report.md): Replaced stale B-11 `HistGBDT` Abstract numbers (`265`, `84k/333.5k/469.2k`, `203.7k` MDP, `+74.9k` B1) and stale Section 6.1 "`HistGBDT` retained" wording; added B-14 MDP clamp fairness finding to Section 9.2; updated Section 12–13 website/test/surrogate disclosures.
  - [`report/executive_summary.md`](report/executive_summary.md): Added B-14 pre-clamp (`40.0%`) vs. post-clamp (`0.0%`) and symmetric execution-clamped GA (`100.0%` feasible) summary.
  - [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md): Updated `ASM-14` from `HistGBDT` slopes (`0.0 / -2.2794 / -2.8553 / -3.7254`, `sd=0.08`) to `log_ols_smooth` (`-2.1933`, `sd=0.2557`, noting preserved `HistGBDT` snapshot).

---

## 7. Task 6 — Website, Figures & Presentation Readiness Audit

- Updated [`site/scripts/prepare_data.py`](site/scripts/prepare_data.py) to include `comp_mdp_clamp_audit_b14.csv` in the downloadable bundle and re-ran `.venv/bin/python site/scripts/prepare_data.py` (`8 JSON payloads ... and 60 downloads`).
- Updated two stale sidebar tooltip `title=` attributes in [`site/index.html`](site/index.html) (`265` → `251` plans; `48` → `60` downloads); verified zero stale B-11 numbers across `site/assets/js/`, `site/index.html`, and `site/README.md`.
- Verified active JSON payloads: `site/data/forecast_stage3.json` (`log_ols_smooth`), `site/data/optimise.json` (`251` Pareto plans, `4` selected, `9` MC reeval, `18` sensitivity scenarios), `site/data/planning.json` (`mdp_adaptive_b6` & `mdp_adaptive_b11` `127,463.34` averted at `$600.35M`, `0.0%` violations), and `site/data/research.json` (`60` downloads, `0` missing files).
- **Local HTTP Verification (`127.0.0.1` Python `ThreadingHTTPServer`):** Requested all `100` served URLs (`index.html`, CSS/JS/Plotly vendor, `8` JSONs, `20` PNG figures in `site/assets/images/real/`, and all `60` downloads + `13` bundled figure downloads = `73` files in `site/downloads/`): **100/100 returned HTTP 200 (`bad: []`)**.
- **Browser Limitation:** Automated Chromium DOM/screenshot inspection via `chrome_devtools` could not execute because `npx` is not installed (`npx not found`).

---

## 8. Task 7 — Final Verification Results (Post-Final Edit)

| Check / Command | Scope | Outcome Observed |
| --- | --- | --- |
| `make lint` | `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests` | **Passed:** `All checks passed!`, `49 files already formatted` |
| `make test` | `.venv/bin/pytest -k "not test_stage1_pipeline_run"` | **69 passed, 3 skipped (interim parquet), 1 deselected** (`29.55s`) |
| `make test-raw` | `tests/historical/test_historical.py -k test_stage1_pipeline_run` | **1 failed, 3 deselected (`1.38s`):** WPP ingestion succeeded (`170 country-years`); failed on missing `data/raw/EG.CFT.ACCS.json` |
| `.venv/bin/pytest -q` | Full test suite (`73` total tests) | **`69` passed, `3` skipped, `1` failed** (`test_stage1_pipeline_run` missing `EG.CFT.ACCS.json`) |
| Contract validation | `load_stage3_contracts` on `configs/default.yaml` & `configs/mock.yaml` | **Passed:** Real (`is_mock=False`, `1536` rows, `-2.1933`, `$5,985.73M`) and Mock (`is_mock=True`, `1536` rows, `[-1.3, -1.45, -1.25, -1.5]`, `$500.0M`) isolated |
| Website HTTP check | `ThreadingHTTPServer` serving `site/` | **100/100 URLs HTTP 200 (`bad = []`)**, `60` downloads registered (`0` missing) |

*(Note: After running `site/index.html` tooltip edits, `.venv/bin/pytest -q` was re-executed cleanly to confirm `69 passed, 3 skipped, 1 failed` [missing `EG.CFT.ACCS.json`].)*

---

## 9. Preserved Snapshot vs. Active Deliverable Registry

- **Active Production Contracts & Outputs:** `data/processed/forecasts_stage3_reconciled.csv`, `data/processed/exposure_response_surrogate.json` (`log_ols_smooth`), `configs/intervention_params.json` (`REAL_V1_2024USD`, Scenario A), `outputs/final_real_data/`, `outputs/tables/comp_mdp_clamp_audit_b14.csv`, `report/report.md`, `report/executive_summary.md`, `site/`.
- **Preserved Reference Snapshots (Intentionally Kept Identifiable):** `data/processed/exposure_response_surrogate_histgbdt.json`, `outputs/tables/histgbdt_baseline_snapshot/`, historical pre-reconciliation raw Stage 2 CSVs (`forecasts_stage3.csv`, `baseline_unit_state.csv`, `stage2_uncertainty_draws.csv`), and prior milestone logs (`FINAL_QA_REPORT.md`, `B12_FORECASTING_QA_REPORT.md`, `B13_SURROGATE_COMPARISON_REPORT.md`).

---

## 10. Release & Presentation Readiness Recommendation

1. **Ready for Code & Report Freeze:** Merge/review cleanly on `b/b10-final-integration` only when ready; do not alter model selection (`log_ols_smooth`), Scenario A constraints (`infra_lpg_grid: null`), or raw manifest requirements.
2. **Disclosure Slide / Viva Guidance:** Present `(i)` GBD 2023 as modelled estimates and WHO GHE 2021 sex splits / `r_fm = 1.05` as proxy disaggregations, `(ii)` `ets_undamped`/`ets_damped`/`arima` as short-term historical CV leaders alongside `log_ols_smooth` as the monotone Stage 3–4 counterfactual surrogate, and `(iii)` MDP's `0.0%` post-clamp vs. `40.0%` pre-clamp audit alongside symmetric execution-clamped GA (`100.0%` feasible).
3. **Optional Post-Freeze Checks (If Needed Outside Sandbox):** Fetch the 3 World Bank JSONs when `api.worldbank.org` responds and open `http://localhost:8000` (`./site/start_site.sh`) in a local desktop browser to visually inspect Plotly charts.
