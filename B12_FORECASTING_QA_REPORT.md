# B-12: Forecasting Accuracy & Hierarchical Reconciliation QA Report

**Branch:** `b/b10-final-integration` | **Status:** Verified (`make lint` 0 errors, `make test` `65 passed, 3 skipped, 1 deselected`) | **Raw Stage 2 Exports:** Preserved Unmodified

---

## 1. Executive Summary

B-12 audited and strengthened Stage 2 forecasting validation, model-fitting integrity, clean-fuel counterfactual comparison, and full-precision hierarchical demographic reconciliation without altering original raw Stage 2 files (`forecasts_stage3.csv`, `baseline_unit_state.csv`, `stage2_uncertainty_draws.csv`, `s2_rolling_cv_metrics.csv`).

1. **Task 1 — Full-Precision Hierarchical Audit & Reconciliation Methods:** Enforcing `pop_both = pop_female + pop_male` prior to CSV rounding in [`src/optimise/reconcile_sex.py`](src/optimise/reconcile_sex.py) achieves full `float64` additivity: population max error `0.00e+00`, count max error `1.86e-09`, and population-weighted ASR consistency `2.27e-13` (post-CSV rounding `<= 0.01`). An 8-level demographic hierarchy check ([`outputs/tables/s2_hierarchical_audit_b12.csv`](outputs/tables/s2_hierarchical_audit_b12.csv)) confirms Stage 1 `urban + rural == country` (`2.91e-11`), consistent legacy WUP urban+rural sums vs WPP (`9.0` people / `5.5e-9`), strict separation of incompatible NPHC 2021 / WUP DEGURBA definitions, and absence of any regional aggregate (`H7: NOT_APPLICABLE`). Comparing Bottom-Up, OLS MinT, WLS, and Top-Down WHO GHE 2021 ([`outputs/tables/s2_reconciliation_methods_b12.csv`](outputs/tables/s2_reconciliation_methods_b12.csv)) proves Bottom-Up (`+50,254.13` DALYs) and OLS MinT (`+16,751.38` DALYs) distort authoritative `both` totals because raw GBDT sex splits lack empirical female-dominant WHO GHE 2021 shares.
2. **Task 2 — ETS `disp=False` Fix & Elimination of Silent Fallbacks:** In [`src/forecast/models.py`](src/forecast/models.py), `ExponentialSmoothing.fit(disp=False)` threw a `TypeError` in `statsmodels 0.14` and silently fell back to `fit_predict_arima`. Changing to `.fit(optimized=True)` and replacing silent `except Exception` blocks with explicit `ModelFitError` logging in [`src/forecast/cv.py`](src/forecast/cv.py) eliminates all silent fallbacks (`0` failures across `891` multi-horizon evaluations in [`outputs/tables/s2_model_failures_b12.csv`](outputs/tables/s2_model_failures_b12.csv)) and produces distinct ETS vs ARIMA predictions (`max |ETS - ARIMA| > 80/100k` per stratum).
3. **Tasks 3 & 4 — MASE Diagnosis, Horizon-Matched Skill Scores & Multi-Horizon Benchmark (`h = 1, 3, 5, 10`):** MASE exceeded `1.0` on `h = 5` solely because `compute_mase` divides multi-step `5`-year errors by strictly causal in-sample `1`-year random-walk differences (`~18.5/100k`). On matched `h = 1`, `arima` (`MASE = 0.67`, MAE `11.31`), `ets_damped` (`0.73`, `12.53`), and `ets_undamped` (`0.73`, `12.78`) all achieve `MASE < 1.0`. Relative to horizon-matched `h = 5` Naive persistence (`103.42` MAE), fixed `ets_undamped` (`47.73`, `+53.85%` pooled skill), `ets_damped` (`48.78`, `+52.83%`), and `arima` (`49.33`, `+52.30%`) strongly beat Naive, `drift` (`63.74`, `+38.37%`), `theta` (`79.28`, `+23.34%`), `log_ols_panel` (`140.96`, `−36.29%`), `poisson_rate_glm` (`147.55`, `−42.67%`), and `hist_gbdt_panel` (`172.23`, `−66.53%`).
4. **Tasks 5 & 6 — Clean-Fuel Counterfactual Comparison & Downstream Sensitivity:** Monotonic `HistGBDT` (`monotonic_cst=[0, -1, 0, 0]`) still plateaus identically (`slope = 0.0`) at `cf >= 0.90` because max training `cf` is `0.958` (`IND_urban` 2023). Smooth `log_ols_smooth` removes the plateau (`IND_urban` falls `251.56 -> 202.02` across `cf = 0.90..1.00`, improving 4-unit CV MAE `102.34` vs `151.00`), while `log_solid_saturating` achieves `46.20` MAE (`3.05M` 2040 BAU) subject to partial Stage 0 apportionment coupling (`w_k ∝ 1 - cf_k`). Substituting `log_ols` `IND_urban` slope (`-2.1933`) into Stage 3 NSGA-II changes DALY-Minimal by only `−1.08%` (`2,229,122.92 -> 2,204,995.92`, [`outputs/tables/s3_b12_counterfactual_sensitivity.csv`](outputs/tables/s3_b12_counterfactual_sensitivity.csv)), justifying retention of reconciled Stage 2 baseline contracts ([`data/processed/forecasts_stage3_reconciled.csv`](data/processed/forecasts_stage3_reconciled.csv)) without re-running full Stage 3/4.

---

## 2. Task 1 — Hierarchical Reconciliation Audit & Method Comparison

### 2.1 Full-Precision Checks (`H1`–`H8`)
From [`outputs/tables/s2_hierarchical_audit_b12.csv`](outputs/tables/s2_hierarchical_audit_b12.csv):

| Check ID | Level | Relationship Verified | Raw Max Abs Error | Reconciled Max Abs Error | Status |
| --- | --- | --- | ---: | ---: | --- |
| `H1_wpp_country_sex_pop` | Country Sex Pop | `pop_F + pop_M == pop_total` (IND/NPL 2025 & 2040) | `1.0` person | `1.0` person | `PASS` (WPP thousands rounding) |
| `H2_forecast_sex_pop` | Subnational Sex Pop | `pop_F + pop_M == pop_both` (full `float64`) | `11.30` | `0.00e+00` | `RECONCILED` (exact in derived contracts) |
| `H3_forecast_sex_counts` | DALY & Death Counts | `val_F + val_M == val_both` (full `float64`) | `223,505.17` | `1.86e-09` | `RECONCILED` (`<= 0.02` post-CSV 2-dec) |
| `H4_forecast_weighted_rates` | Sex ASRs (Pop-Weighted) | `(r_F * p_F + r_M * p_M) / p_both == r_both` | `34.9565` | `2.27e-13` | `RECONCILED` (never sums rates) |
| `H5_hist_urban_rural_to_country` | Stage 1 Counts (1990 & 2023) | `IND_u + IND_r == India`; `NPL_u + NPL_r == Nepal` | `2.91e-11` | `2.91e-11` | `PASS` (consistent apportionment) |
| `H6_forecast_urban_rural_pop_vs_country` | 2040 Subnational Pop | Legacy WUP linear-share sum vs WPP national | `9.0` | `9.0` | `PASS` (incompatible defs kept separate) |
| `H7_country_to_regional_aggregate` | Regional Total | Checked for South Asia aggregate row | — | — | `NOT_APPLICABLE` (none present; none invented) |
| `H8_mc_500_draws_sex_additivity` | 500 Uncertainty Draws | `draw_F + draw_M == draw_both` | `961,988.17` | `0.01` | `RECONCILED` (`<= 0.01` post-CSV 2-dec) |

### 2.2 Reconciliation Method Comparison (2040 BAU DALYs)
From [`outputs/tables/s2_reconciliation_methods_b12.csv`](outputs/tables/s2_reconciliation_methods_b12.csv), comparing unconstrained `raw_unreconciled`, `bottom_up_raw_gbdt`, `ols_mint_raw_gbdt` (`S (S^T S)^-1 S^T y`), `wls_dominant_both_raw_ratio`, and `top_down_who_ghe_2021` summed across the 4 target units:
- **`raw_unreconciled`:** Authoritative `both` sum = `2,699,837.26`, raw `female + male` = `2,750,091.39` (`+50,254.13` excess; `IND_urban` implied F share `49.64%` < WHO GHE `53.46%`).
- **`bottom_up_raw_gbdt`:** Forces `both = female + male`, biasing total 2040 BAU upward by **`+50,254.13` DALYs** (`IND_urban +15,221.14`, `IND_rural +28,075.23`, `NPL_urban +434.77`, `NPL_rural +6,522.99`) and preserving sex-inverted ratios.
- **`ols_mint_raw_gbdt`:** Splits discrepancy equally (`+16,751.37` bias in `both`; still lacks female-dominant `WHO_GHE_SEX_SPLITS`).
- **`top_down_who_ghe_2021` (Selected):** Zero bias vs authoritative `both` (`0.00`), exact `0.00` sex additivity error, and exact preservation of `WHO_GHE_SEX_SPLITS` (`53.46%` India, `58.34%` Nepal).

---

## 3. Tasks 2–4 — ETS Bug Fix, Causal MASE, Multi-Horizon CV (`h = 1, 3, 5, 10`)

From [`outputs/tables/s2_cv_summary_b12.csv`](outputs/tables/s2_cv_summary_b12.csv) across the non-redundant combined-sex (`both`) 9-location South Asia panel (`Bangladesh`, `Bhutan`, `India`, `Nepal`, `Pakistan`, `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`; origins `[2008, 2013, 2018]`; `n_failed = 0` across all `891` evaluations):

| Horizon | Model | `n_eval` | Mean MAE | Mean RMSE | Mean MASE | Macro Rel. MAE | Pooled Rel. MAE | Pooled Skill (`1 − RelMAE`) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h = 1` | `arima` | 27 | **11.3053** | 11.3053 | **0.6696** | **0.3075** | **0.3432** | **+65.68%** |
| `h = 1` | `ets_damped` | 27 | 12.5325 | 12.5325 | 0.7273 | 0.3764 | 0.3804 | +61.96% |
| `h = 1` | `ets_undamped` | 27 | 12.7787 | 12.7787 | 0.7307 | 0.4085 | 0.3879 | +61.21% |
| `h = 1` | `naive_baseline` | 27 | 32.9412 | 32.9412 | 1.7679 | 1.0000 | 1.0000 | 0.00% |
| `h = 1` | `hist_gbdt_panel` | 27 | 151.4511 | 151.4511 | 6.7241 | 5.1504 | 4.5976 | −359.76% |
| `h = 3` | `ets_undamped` | 27 | **39.3349** | **45.4104** | **2.1717** | 0.4946 | **0.4913** | **+50.87%** |
| `h = 3` | `arima` | 27 | 40.1193 | 46.6978 | 2.2764 | **0.4387** | 0.5011 | +49.89% |
| `h = 3` | `ets_damped` | 27 | 40.2651 | 46.4278 | 2.2399 | 0.4971 | 0.5029 | +49.71% |
| `h = 5` | `ets_undamped` | 27 | **47.7256** | **55.3607** | **2.6475** | 0.5176 | **0.4615** | **+53.85%** |
| `h = 5` | `ets_damped` | 27 | 48.7846 | 56.8739 | 2.7512 | 0.5184 | 0.4717 | +52.83% |
| `h = 5` | `arima` | 27 | 49.3343 | 57.0661 | 2.8693 | **0.4498** | 0.4770 | +52.30% |
| `h = 5` | `drift` | 27 | 63.7444 | 71.0683 | 3.5448 | 0.6766 | 0.6163 | +38.37% |
| `h = 5` | `theta` | 27 | 79.2823 | 87.3199 | 4.4736 | 0.7588 | 0.7666 | +23.34% |
| `h = 5` | `naive_baseline` | 27 | 103.4247 | 113.3443 | 5.5996 | 1.0000 | 1.0000 | 0.00% |
| `h = 5` | `log_ols_panel` | 27 | 140.9600 | 145.5965 | 6.1889 | 1.5746 | 1.3629 | −36.29% |
| `h = 5` | `poisson_rate_glm` | 27 | 147.5530 | 150.7191 | 6.1440 | 1.7088 | 1.4267 | −42.67% |
| `h = 5` | `hist_gbdt_panel` | 27 | 172.2318 | 178.6375 | 8.1433 | 1.7956 | 1.6653 | −66.53% |
| `h = 10` | `ets_undamped` | 18 | **86.7384** | **104.8991** | **5.1030** | 0.5323 | **0.5098** | **+49.02%** |
| `h = 10` | `ets_damped` | 18 | 89.0491 | 107.0723 | 5.3799 | 0.5169 | 0.5234 | +47.66% |
| `h = 10` | `arima` | 18 | 92.6112 | 112.2268 | 5.9059 | **0.5139** | 0.5443 | +45.57% |
| `h = 10` | `log_ols_panel` | 18 | 155.2030 | 162.2635 | 7.0562 | 0.9120 | 0.9122 | **+8.78%** |
| `h = 10` | `poisson_rate_glm` | 18 | 161.7423 | 168.8588 | 7.1188 | 0.9703 | 0.9507 | **+4.93%** |
| `h = 10` | `naive_baseline` | 18 | 170.1359 | 192.9823 | 9.8072 | 1.0000 | 1.0000 | 0.00% |
| `h = 10` | `hist_gbdt_panel` | 18 | 201.5083 | 215.0539 | 9.9368 | 1.2338 | 1.1844 | −18.44% |

- **Per-Series vs Global Selection (`s2_cv_per_series_winners_b12.csv`):** On `h = 5`, `arima` wins in `Bangladesh`, `Bhutan`, `IND_urban`, `NPL_urban`; `ets_undamped` wins in `NPL_rural`, `Nepal`, `Pakistan`; covariate panel models (`poisson_rate_glm` / `log_ols_panel`) beat univariate models on `IND_rural` (`83.79` vs `84.26`) and `India` (`33.40` vs `62.74`) where post-2008 Ujjwala clean-cooking acceleration bent rates faster than autoregressive momentum alone. Given only `3` CV origins, per-series model switching risks post-hoc selection variance; global `ets_undamped` or `ets_damped` provides a cleaner univariate trend benchmark.

---

## 4. Tasks 5 & 6 — Clean-Fuel Counterfactual Comparison & Downstream Impact

From [`outputs/tables/s2_counterfactual_comparison_b12.csv`](outputs/tables/s2_counterfactual_comparison_b12.csv) (evaluated across `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`, `both` sex):

| Counterfactual Model | 4-Unit `h=5` CV MAE | `IND_urban` `cf=0.90 -> 1.00` ASR | `cf >= 0.98` Plateau? | Implied 2040 BAU Total DALYs | Key Trade-Off |
| --- | ---: | ---: | --- | ---: | --- |
| `hist_gbdt_unconstrained` | 151.00 | `221.31 -> 221.31` (`0.00`) | **Yes** (`slope = 0.0`) | 4,870,271 *(on `both`-only)* / **2,699,837** *(full-panel Stage 2 export)* | Learns multi-country rural shrinkage; saturates beyond max training `cf = 0.958` |
| `hist_gbdt_monotonic` (`monotonic_cst=[0,-1,0,0]`) | 148.25 | `209.74 -> 209.74` (`0.00`) | **Yes** (`slope = 0.0`) | 4,626,329 *(both-only)* | Monotonic constraint does **not** solve boundary leaf saturation beyond `cf > 0.958` |
| `log_ols_smooth` | **102.34** | `251.56 -> 202.02` (`-49.54`) | **No** (`slope = -452.94/100k`) | 4,332,703 | Smooth response above `0.98`; linear `t` extrapolation remains higher than Stage 2 full-panel BAU |
| `log_solid_saturating` (`log(1-cf)`) | **46.20** | `168.48 -> 13.50` (`-154.98`) | **No** | 3,053,390 | Strongest parametric fit (`R^2=0.99`), but partially shares functional form with Stage 0 apportionment (`w_k ∝ 1 - cf_k`) |

- **Downstream Stage 3 Sensitivity (`s3_b12_counterfactual_sensitivity.csv`):** Testing non-saturating `IND_urban` counterfactual slope `-2.1933` (from `log_ols_smooth`) inside `Stage3Evaluator` + 5-seed `NSGA-II`:
  - **DALY-Minimal:** `2,229,122.92` (`$3,081.69M`) → **`2,204,995.92`** (`$3,096.07M`), a **`−1.08%`** burden shift.
  - **Knee Compromise:** `2,375,499.49` (`$1,667.18M`) → **`2,342,129.10`** (`$1,778.37M`), a **`−1.40%`** shift.
  - **Cost-Minimal:** `2,631,051.64` (`$63.11M`) → **`2,604,862.47`** (`$75.72M`).
  - Allocation priorities (`NPL_rural` biomass first, `IND_rural` + `NPL_urban` LPG/induction) remain invariant, validating retention of the reconciled B-11 Stage 3/4 outputs without invalidating prior deliverables.

---

## 5. Task 7 — Explicit Final Model Recommendation & Verification

**Final Recommendation:**
1. **Do NOT replace BAU 2040 forecasts directly with unconditioned `ARIMA(1,1,0)` or `ETS` (`h = 5` CV winners):** Although `ets_undamped` (`47.73` MAE, `+53.85%` skill), `ets_damped` (`48.78`, `+52.83%`), and `arima` (`49.33`, `+52.30%`) are clearly superior for short-term `1–5` year epidemiological trend prediction (`0` fit failures once `disp=False` is removed), they contain **no clean-fuel covariate** and extrapolate post-2021 rebounds in `IND_rural`, producing unconditioned 2040 BAU burdens near `9.0M` DALYs (`ARIMA` / `ETS_damped`) despite `IND_rural` clean-fuel expansion from `65.8%` to `85.0%`.
2. **Retain Reconciled Stage 2 Contracts (`forecasts_stage3_reconciled.csv`) with Dual-Model Discipline:** Keep `forecasts_stage3_reconciled.csv` (`2,699,837.26` DALYs) for Stage 3/4 decision modelling, report fixed `ETS` / `ARIMA` / `Drift` / `Theta` multi-horizon CV metrics (`s2_cv_summary_b12.csv`) honestly as the statistical trend benchmark, and report smooth `Log-OLS` sensitivity (`−1.08%` DALY-Minimal) alongside the known `HistGBDT` boundary plateau above `0.958`.

**Verification Run Summary (Post-Final Edit):**
- `make lint` (`ruff check src tests` + `ruff format --check src tests`): **Passed** (`47 files already formatted`, `0` errors).
- `make test` (`.venv/bin/pytest -k "not test_stage1_pipeline_run"`): **`65 passed, 3 skipped, 1 deselected`** in `28.83s` (including new `test_b12_full_precision_and_hierarchy_comparison` in [`tests/optimise/test_reconcile_sex.py`](tests/optimise/test_reconcile_sex.py) and 5 tests in [`tests/forecast/test_forecast_b12.py`](tests/forecast/test_forecast_b12.py)).
- Full `pytest -q`: `65 passed, 3 skipped, 1 failed` (`test_stage1_pipeline_run` fails on missing external `data/raw/` WPP file excluded per `.gitignore`).
- Stage 3 real + mock contract loading (`load_stage3_contracts`): **Verified valid** (`is_mock=False` and `is_mock=True`).
- Website build (`site/scripts/prepare_data.py` -> `8` JSON payloads, `55` downloads) + local HTTP check (`12` routes/assets): **`12/12 HTTP 200`**.
