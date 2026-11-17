# B-13: Side-by-Side HistGradientBoosting vs. Smooth Log-OLS Production Surrogate Comparison Report

**Branch:** `b/b10-final-integration` | **Preserved HistGBDT Snapshot:** [`exposure_response_surrogate_histgbdt.json`](data/processed/exposure_response_surrogate_histgbdt.json) & [`outputs/tables/histgbdt_baseline_snapshot/`](outputs/tables/histgbdt_baseline_snapshot/) | **Active Production Surrogate:** [`exposure_response_surrogate.json`](data/processed/exposure_response_surrogate.json) (`source_model = "log_ols_smooth"`)

---

## 1. Decision Summary

Following a complete side-by-side empirical comparison across identical validation observations, folds, response curves, extrapolation behaviour, analytical uncertainty, and full downstream Stage 3–4 regeneration ([`src/forecast/b13_surrogate.py`](src/forecast/b13_surrogate.py)), **`HistGradientBoostingRegressor` provides no modelling advantage over smooth `Log-OLS` as the Stage 3 counterfactual surrogate**. We therefore replaced Contract 3 (`exposure_response_surrogate.json`) with **pooled Smooth Log-OLS** (`log(ASR) ~ t + clean_fuel_share + C(location)`) across **all four intervention units** while keeping reconciled 2040 BAU (`2,699,837.26` DALYs) and HistGradientBoosting's historical benchmark untouched.

---

## 2. Production Trace of `HistGradientBoosting` (Tasks 1 & 2)

Before replacement, [`configs/default.yaml`](configs/default.yaml) pointed Stage 3 and Stage 4 directly to `data/processed/exposure_response_surrogate.json` (`"source_model": "hist_gbdt_panel"`):
1. [`src/forecast/projections.py`](src/forecast/projections.py) (`best_model_name = "hist_gbdt_panel"`) fit `HistGradientBoostingRegressor(max_iter=180, learning_rate=0.04, max_depth=4)` to predict `val_bau` and `val_100` in 2040, calibrating `slope_clean_fuel = ln(val_100 / val_bau) / (1 - cf_bau)`:
   - `IND_urban`: **`0.0000`** (leaf plateau above `cf >= 0.985`), `IND_rural`: **`-2.2794`**, `NPL_urban`: **`-2.8553`**, `NPL_rural`: **`-3.7254`**, with placeholder `uncertainty_sd = 0.08`.
2. [`src/optimise/evaluator.py`](src/optimise/evaluator.py) (`Stage3Evaluator`) loaded `slope_clean_fuel` and `uncertainty_sd` (`surr_hw = dalys_mean_us * 1.96 * uncertainty_sd * delta_c`), feeding `NSGA-II` (`run.py`, `scenario_runs.py`, `final_real_suite.py`) and all Stage 4 planners (`mdp.py`, `htn.py`, `pddl_runner.py`, `compare.py`).

---

## 3. Identical-Fold Quantitative Comparison (Tasks 3, 4 & 5)

Evaluated on identical 4-unit (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`, `both` sex, `1990–2023`) rolling-origin CV folds (`origins = [2008, 2013, 2018]`; [`outputs/tables/s2_counterfactual_cv_identical_b13.csv`](outputs/tables/s2_counterfactual_cv_identical_b13.csv)):

| Candidate Model | `h=1` MAE | `h=3` MAE | `h=5` MAE | `h=10` MAE | `cf >= 0.96` Behaviour | Analytical Slope SE |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| **`log_ols_smooth` (Pooled, Selected)** | `81.99` | `90.16` | **`102.34`** | **`106.60`** | Smooth strictly decreasing (`−2.1933`) | **`0.2557`** (`p = 2.51e-14`) |
| `log_ols_interacted` (`cf * C(loc)`) | `32.51` | `58.72` | `64.52` | `107.44` | Unit slopes `[−4.15, −1.12, −2.01, −1.83]` | `0.13–0.37` *(apportionment bias)* |
| `poisson_rate_glm` | — | — | `102.27` | — | Smooth strictly decreasing (`−1.7368`) | `Log` GLM SE |
| `naive_baseline` | `33.17` | `80.83` | `102.33` | `173.85` | Flat persistence (`0.0`) | None |
| `hist_gbdt_monotonic` (`[0,-1,0,0]`) | `116.64` | `130.56` | `148.25` | `162.54` | Still plateaus above `0.958` | None |
| `hist_gbdt_panel` (Unconstrained) | `123.93` | `133.88` | **`151.00`** | `160.43` | Step cliffs & `IND_urban = 0.0` plateau | Placeholder `0.08` |

### Why Pooled `log_ols_smooth` (`-2.1933`) Replaced `HistGBDT` Across All 4 Units Instead of `log_ols_interacted`
1. **Structural Smoothness & No Clamping Clutter:** `HistGBDT` clamped all four units onto the exact same terminal leaf (`124.08/100k`) at `cf = 1.00` ([`s2_surrogate_comparison_b13.csv`](outputs/tables/s2_surrogate_comparison_b13.csv)), creating an artificial `-3.7254` slope in `NPL_rural` alongside `0.0000` in `IND_urban`. Smooth Log-OLS produces smooth exponential trajectories (`cf = 0.25 -> 1.00`) across all 4 units (`IND_urban` `124.08 -> 120.06`, `NPL_rural` `861.02 -> 275.23`).
2. **Why Pooled (`-2.1933`) Over `log_ols_interacted` (`-4.1482` vs `-1.1162`):** Although unit interactions achieve lower CV error (`64.52` MAE), Stage 0 apportioned urban/rural rates proportionally to `(1 - cf)`. Letting `cf * C(location)` absorb that construction spuriously makes clean fuel look **3.7× more effective per percentage point in 96%-clean `IND_urban` (`-4.1482`) than in 65%-clean `IND_rural` (`-1.1162`)**, inverting diminishing-returns HAP exposure physics. Pooled `log_ols_smooth` estimates one consistent cross-panel elasticity (`−2.1933 ± 0.2557 SE`, `R^2 = 0.8938`).
3. **Genuine Slope Uncertainty:** Replacing HistGBDT's unverified `uncertainty_sd = 0.08` with the empirical OLS slope standard error **`0.2557`** propagates statistically grounded parameter uncertainty into `Stage3Evaluator` UIs.

---

## 4. Downstream Results Before vs. After Replacement (Task 6)

BAU 2040 (`2,699,837.26` DALYs, equity gap `31.41`/100k) and all Stage 1 / Stage 2 CV benchmarks ([`s2_rolling_cv_metrics.csv`](outputs/tables/s2_rolling_cv_metrics.csv), [`s2_cv_summary_b12.csv`](outputs/tables/s2_cv_summary_b12.csv)) are **unchanged**. Every downstream optimization, planning, MC, and sensitivity change is documented below:

| Downstream Metric | Preserved `HistGBDT` Surrogate | Active `Smooth Log-OLS` (`−2.1933 ± 0.2557`) | Delta / Interpretation |
| --- | ---: | ---: | --- |
| **Pareto Front Size / HV** | `265` pts / `2.9700e+10` | **`251` pts / `2.7965e+10`** | Slightly more conservative upper rural returns |
| **GA Cost-Minimal** (`DALYs` / `Cost` / `Eq`) | `2,615,798` (`84.0k` av.) / `$68.95M` / `30.01` | **`2,640,478` (`59.4k` av.) / `$70.75M` / `30.42`** | −29.4% averted DALYs at min cost (`NPL_rural` `-3.73 -> -2.19`) |
| **GA Knee Compromise** (`DALYs` / `Cost` / `Eq`) | `2,366,324` (`333.5k` av.) / `$1,750.92M` / `26.49` | **`2,388,130` (`311.7k` av.) / `$1,695.91M` / `26.85`** | `−6.5%` averted DALYs; `$55M` lower knee spend |
| **GA DALY-Min = Equity-Min** | `2,230,658` (`469.2k` av.) / `$3,076.02M` / `24.64` | **`2,246,364` (`453.5k` av.) / `$3,097.29M` / `24.91`** | Only `−3.3%` averted change; freed `IND_urban` offsets `NPL_rural` |
| **500-Draw Joint MC Knee `[p2.5–p97.5]`** | `2,380,646` `[1.95M–2.78M]` | **`2,401,031` `[1.97M–2.80M]`** | Full OLS slope SE propagated (`0.2557`) |
| **500-Draw Joint MC DALY-Min `[p2.5–p97.5]`** | `2,244,340` `[1.85M–2.61M]` | **`2,258,315` `[1.87M–2.63M]`** | Narrow 95% UI expansion from genuine SE |
| **Adaptive MDP (`λ=116`, `b=6/11`)** | `203,720` av. / `$1,078.8M` / `0.0%` viol. | **`127,463` av. / `$600.4M` / `0.0%` viol.** | Under `-2.1933`, `λ=116` selects lower-spend biomass recourse |
| **GA Matched (`1,000` rollouts)** | `217,147` av. / `$798.1M` / `40.0%` viol. | **`245,520` av. / `$1,176.9M` / `40.0%` viol.** | Scalarized return `90,698` vs MDP `46,750` |
| **PDDL (`7` steps) / HTN (`9` leaves)** | `412,448` av. (`26.9%` viol.) / `$807.4M` | **`410,256` av. (`26.9%` viol.) / `$807.4M`** | Same PDDL & HTN structures; minor DALY change |
| **Scenario B1 / B2 Proxy Infra** | `544,029` / `544,049` av. (`+74.9k`) | **`513,035` / `512,856` av. (`+59.6k`)** | Infra remains major rural unlock lever |
| **Alt B Recurring LPG Opex (`DALY-Min`)** | `$5,289.50M` (`1.72×`, `< $5,985.7M`) | **`$5,192.82M` (`1.68×`, `< $5,985.7M`)** | Still feasible within 50% fiscal budget cap |

---

## 5. Verification

- [`tests/forecast/test_surrogate_b13.py`](tests/forecast/test_surrogate_b13.py), [`tests/forecast/test_forecast_b12.py`](tests/forecast/test_forecast_b12.py), and all Stage 3–4 contract/optimisation/planning tests re-verified after regeneration.
- Both preserved `HistGBDT` artifacts (`exposure_response_surrogate_histgbdt.json`, `outputs/tables/histgbdt_baseline_snapshot/`) and active `log_ols_smooth` outputs remain cleanly archived for audit without altering historical BAU forecasts or MASE benchmarks.
