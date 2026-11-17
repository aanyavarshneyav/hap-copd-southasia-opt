/* Page 3: Forecasting & 2040 Projections (V2 Visual Storytelling) */
(function (global) {
  "use strict";

  async function render(container) {
    const data = await global.App.loadJSON("forecast_stage3.json");
    const { esc, term, accordion, fmtNum } = global.App;
    const status = data.real_stage2_status || {};
    const realFc = data.forecasts || [];
    const realSurr = data.surrogate || {};
    const cvMetrics = data.cv_metrics || [];
    const summary2040 = data.forecasts_2040_summary || [];

    const state = {
      activeView: "trajectories", // "trajectories" | "surrogate" | "cv_metrics"
      unit: "IND_rural",
      measure: "DALYs",
      metric: "Rate_per_100k",
      sex: "both",
    };

    const modelsHtml = `
      <div class="grid grid-auto" style="margin-bottom:1rem;">
        ${(status.models || [])
          .map(
            (m) => `
          <div class="card-elevated">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:0.4rem; margin-bottom:0.35rem;">
              <strong style="font-size:0.9rem;">${esc(m.name)}</strong>
              <span class="badge badge-derived">${esc(m.family)}</span>
            </div>
            <p style="margin:0; font-size:0.8rem; color:var(--text-secondary);">${esc(m.implementation)}</p>
          </div>
        `
          )
          .join("")}
      </div>
      <div class="takeaway-bar" style="margin-bottom:0;">
        <span class="takeaway-tag">Clarification</span>
        <div>
          The adapted ${term("Poisson Rate GLM")} models aggregated all-age burden with a <code>log(population)</code> offset rather than decomposing age, period, and cohort effects across 5-year age bins (though <code>configs/default.yaml</code> keeps the key <code>poisson_apc</code>).
        </div>
      </div>
    `;

    const cvHtml = `
      <div class="grid grid-2">
        <div>
          <h4 style="margin:0 0 0.5rem; font-size:0.9rem;">Expanding-Window ${term("Rolling-Origin CV")} Folds</h4>
          <div class="table-wrap">
            <table class="data-table">
              <thead><tr><th>Fold</th><th>Train</th><th>Test (h=5)</th><th>Metrics</th></tr></thead>
              <tbody>
                ${(status.cv_folds || [])
                  .map(
                    (f) => `
                  <tr>
                    <td>Fold ${esc(f.fold)}</td>
                    <td><code>${esc(f.train)}</code></td>
                    <td><code>${esc(f.test)}</code></td>
                    <td>MAE, RMSE, MASE</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h4 style="margin:0 0 0.5rem; font-size:0.9rem;">Uncertainty, Apportionment &amp; Proxy Sex Split</h4>
          <ul class="callout-list">
            <li><strong>Four Distinct Regimes:</strong> Stage 2 separates GBD historical estimates (1990–2023), in-sample fit, rolling-origin validation predictions, and 2024–2040 projections.</li>
            <li><strong>500 Uncertainty Draws:</strong> Combines GBD baseline 95% UI bounds with model residual variance and converts rates to counts via UN WPP 2024 populations.</li>
            <li><strong>Proxy Sex-Specific Projection:</strong> Sex-stratified BAU trajectories are paired with a proxy female-to-male exposure elasticity ratio (<code>sex_elasticity_ratio_f_to_m</code>) because clean-cooking rollout lacks sex disaggregation.</li>
          </ul>
        </div>
      </div>
    `;

    container.innerHTML = `
      <header class="page-header">
        <div class="page-header-top">
          <span class="page-eyebrow">03 · Stage 2 Statistical &amp; ML Forecasting</span>
          <div class="page-header-meta" style="margin-top:0;">
            <span class="badge badge-real">Real Stage 2 Forecasts: GBD 2023 + WPP 2024</span>
            <span class="badge badge-derived">Counterfactual Surrogate: HistGBDT Panel</span>
          </div>
        </div>
        <h1 class="page-title">Forecasting &amp; 2040 Projections</h1>
        <p class="page-lead">
          Compares five statistical and machine-learning models across South Asia to project 2040 COPD burden with 95% uncertainty intervals and calibrate a fast response curve.
        </p>
      </header>

      ${global.App.infoBanner(
        "VALIDATED & B-12/B-13 RECONCILED STAGE 2 REAL DATA (GBD 2023 + UN WPP 2024)",
        "Authoritative combined-sex ('both') 2040 BAU burden remains 2,699,837 DALYs (F+M = both within 1.86e-9). B-12 fixed ETS (47.73 MAE, +53.9% skill vs Naive) with 0 failures; B-13 replaced the production Contract 3 counterfactual surrogate with pooled Smooth Log-OLS (slope −2.1933 ± 0.2557 SE; 4-unit CV MAE 102.34 vs. 151.00 for HistGBDT), eliminating HistGBDT's artificial IND_urban plateau while preserving its historical benchmark."
      )}

      <!-- Level 1 & 2: Prominent Visual Stage -->
      <section class="section">
        <div class="tabs-bar" role="tablist" aria-label="Forecast exploration views">
          <button type="button" class="tab-btn active" role="tab" data-fc-view="trajectories">
            2025–2040 Real Forecast Explorer
          </button>
          <button type="button" class="tab-btn" role="tab" data-fc-view="surrogate">
            Exposure–Response Surrogate Curve
          </button>
          <button type="button" class="tab-btn" role="tab" data-fc-view="cv_metrics">
            Rolling-Origin CV &amp; 2040 Summary (Real Data)
          </button>
        </div>

        <div id="fc-controls-mount" class="controls-bar"></div>
        <div id="fc-stage-mount" class="card"></div>
      </section>

      <!-- Level 3: Expandable Methodology & CV Details -->
      ${accordion(
        "Technical Details: Five-Model Suite (ARIMA, ETS, Adapted Poisson GLM, RF, HistGBDT)",
        '<span class="badge badge-derived">Stage 2 Spec</span>',
        modelsHtml
      )}
      ${accordion(
        "Methodology &amp; Uncertainty: Rolling-Origin Validation, Apportionment &amp; Proxy Sex Projection",
        '<span class="badge badge-assumption">Proxy &amp; UI Propagation</span>',
        cvHtml
      )}
    `;

    function renderView() {
      const ctrlEl = document.getElementById("fc-controls-mount");
      const stageEl = document.getElementById("fc-stage-mount");

      if (state.activeView === "cv_metrics") {
        ctrlEl.style.display = "none";
        const cvB12H5 = (data.cv_summary_b12 || []).filter((r) => Number(r.horizon) === 5);
        const models = ["arima", "ets", "naive_baseline", "hist_gbdt_panel", "poisson_rate_glm"];
        const folds = ["2009-2013", "2014-2018", "2019-2023"];
        const foldRows = folds.map((fld) => {
          const stats = models.map((m) => {
            const sub = cvMetrics.filter((r) => r.test_window === fld && r.model === m);
            const mae = sub.reduce((a, b) => a + Number(b.mae || 0), 0) / Math.max(1, sub.length);
            const mase = sub.reduce((a, b) => a + Number(b.mase || 0), 0) / Math.max(1, sub.length);
            return { model: m, mae, mase };
          });
          return { fold: fld, stats };
        });

        stageEl.innerHTML = `
          <div class="card-header">
            <div>
              <h2 class="card-title">Stage 2 Rolling-Origin Cross-Validation &amp; B-12 Audit</h2>
              <p class="card-subtitle">B-12 fixed ETS, Drift, Theta &amp; horizon-matched Skill Scores (1 &minus; relMAE) across 9 South Asia locations (h=5 yrs)</p>
            </div>
            <span class="badge badge-real">Real Stage 2 + B-12 Validation</span>
          </div>
          <div id="chart-fc-cv" class="chart-box"></div>
          ${
            cvB12H5.length
              ? `<div class="table-wrap" style="margin-top:1rem;">
                  <table class="data-table">
                    <thead><tr><th>Model (B-12 Fixed, h=5 both)</th><th class="num-right">Mean MAE</th><th class="num-right">Mean RMSE</th><th class="num-right">MASE (vs 1-yr)</th><th class="num-right">Rel. MAE (vs h=5 Naive)</th><th class="num-right">Pooled Skill Score</th><th class="num-right">Failures</th></tr></thead>
                    <tbody>
                      ${cvB12H5
                        .map(
                          (r) => `
                        <tr>
                          <td><code>${esc(r.model)}</code></td>
                          <td class="num-right">${fmtNum(r.mean_mae, 2)}</td>
                          <td class="num-right">${fmtNum(r.mean_rmse, 2)}</td>
                          <td class="num-right">${fmtNum(r.mean_mase, 2)}</td>
                          <td class="num-right">${fmtNum(r.pooled_relative_mae, 3)}</td>
                          <td class="num-right"><strong>${(Number(r.pooled_skill_score) * 100).toFixed(1)}%</strong></td>
                          <td class="num-right">${r.n_failed}</td>
                        </tr>
                      `
                        )
                        .join("")}
                    </tbody>
                  </table>
                </div>`
              : ""
          }
          <div class="table-wrap" style="margin-top:1rem;">
            <table class="data-table">
              <thead><tr><th>Original Stage 2 Model</th><th>Mean MAE</th><th>Mean RMSE</th><th>Mean MASE</th><th>B-12 Audit Finding</th></tr></thead>
              <tbody>
                ${models
                  .map((m) => {
                    const sub = cvMetrics.filter((r) => r.model === m);
                    const n = Math.max(1, sub.length);
                    const mae = sub.reduce((a, b) => a + Number(b.mae || 0), 0) / n;
                    const rmse = sub.reduce((a, b) => a + Number(b.rmse || 0), 0) / n;
                    const mase = sub.reduce((a, b) => a + Number(b.mase || 0), 0) / n;
                    const role =
                      m === "hist_gbdt_panel"
                        ? "Counterfactual surrogate (cf_25..cf_100); IND_urban log-OLS sensitivity -1.08%"
                        : m === "arima"
                        ? "ARIMA(1,1,0); strong short-term univariate baseline (49.33 MAE both-sex)"
                        : m === "ets"
                        ? "Original fell back to ARIMA (disp=False); B-12 fix yields 47.73 undamped / 48.78 damped MAE"
                        : m === "poisson_rate_glm"
                        ? "Reduced-form covariate GLM; reaches positive skill (+4.9%) at h=10 yrs"
                        : "Persistence benchmark (re-anchored for horizon-matched skill score)";
                    return `<tr><td><code>${esc(m)}</code></td><td class="num-right">${mae.toFixed(2)}</td><td class="num-right">${rmse.toFixed(2)}</td><td class="num-right">${mase.toFixed(2)}</td><td>${esc(role)}</td></tr>`;
                  })
                  .join("")}
              </tbody>
            </table>
          </div>
          <div class="table-wrap" style="margin-top:1.1rem;">
            <table class="data-table">
              <thead><tr><th>Unit</th><th>Measure</th><th>Metric</th><th class="num-right">2040 Val</th><th class="num-right">Lower</th><th class="num-right">Upper</th><th class="num-right">Pop 2040</th><th class="num-right">Clean Fuel Share</th></tr></thead>
              <tbody>
                ${summary2040
                  .map(
                    (r) => `
                  <tr>
                    <td><strong>${esc(r.location)}</strong></td>
                    <td>${esc(r.measure)}</td>
                    <td><code>${esc(r.metric)}</code></td>
                    <td class="num-right">${fmtNum(r.val, 2)}</td>
                    <td class="num-right">${fmtNum(r.lower, 2)}</td>
                    <td class="num-right">${fmtNum(r.upper, 2)}</td>
                    <td class="num-right">${fmtNum(r.population, 0)}</td>
                    <td class="num-right">${(Number(r.clean_fuel_share) * 100).toFixed(1)}%</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `;

        const colors = { arima: "#83CDD1", ets: "#68BBC2", naive_baseline: "#A6AAA6", hist_gbdt_panel: "#DF8B6C", poisson_rate_glm: "#D6C7A9" };
        global.AppCharts.plot(
          "chart-fc-cv",
          models.map((m) => ({
            x: foldRows.map((fr) => fr.fold),
            y: foldRows.map((fr) => fr.stats.find((s) => s.model === m)?.mae || 0),
            name: m,
            type: "bar",
            marker: { color: colors[m] },
          })),
          {
            barmode: "group",
            xaxis: { title: "Rolling-Origin Test Window" },
            yaxis: { title: "Mean Absolute Error (MAE, ASR per 100k)" },
            legend: { orientation: "h", y: -0.22 },
            margin: { l: 64, r: 20, t: 18, b: 84 },
          }
        );
        return;
      }

      ctrlEl.style.display = "flex";
      const showMeasureMetricSex = state.activeView === "trajectories";
      ctrlEl.innerHTML = `
        <div class="control-group">
          <label for="fc-unit-sel">Allocation Unit</label>
          <select id="fc-unit-sel" class="control-select">
            <option value="IND_urban" ${state.unit === "IND_urban" ? "selected" : ""}>IND_urban (India Urban)</option>
            <option value="IND_rural" ${state.unit === "IND_rural" ? "selected" : ""}>IND_rural (India Rural)</option>
            <option value="NPL_urban" ${state.unit === "NPL_urban" ? "selected" : ""}>NPL_urban (Nepal Urban)</option>
            <option value="NPL_rural" ${state.unit === "NPL_rural" ? "selected" : ""}>NPL_rural (Nepal Rural)</option>
          </select>
        </div>
        ${
          showMeasureMetricSex
            ? `
          <div class="control-group">
            <label for="fc-measure-sel">Measure</label>
            <select id="fc-measure-sel" class="control-select">
              <option value="DALYs" ${state.measure === "DALYs" ? "selected" : ""}>DALYs</option>
              <option value="Deaths" ${state.measure === "Deaths" ? "selected" : ""}>Deaths</option>
            </select>
          </div>
          <div class="control-group">
            <label for="fc-metric-sel">Metric &amp; Unit</label>
            <select id="fc-metric-sel" class="control-select">
              <option value="Rate_per_100k" ${state.metric === "Rate_per_100k" ? "selected" : ""}>Age-Standardized Rate (per 100k)</option>
              <option value="Number" ${state.metric === "Number" ? "selected" : ""}>All-Age Count (Number)</option>
            </select>
          </div>
          <div class="control-group">
            <label for="fc-sex-sel">Sex</label>
            <select id="fc-sex-sel" class="control-select">
              <option value="both" ${state.sex === "both" ? "selected" : ""}>Both Sexes</option>
              <option value="female" ${state.sex === "female" ? "selected" : ""}>Female</option>
              <option value="male" ${state.sex === "male" ? "selected" : ""}>Male</option>
            </select>
          </div>
        `
            : ""
        }
      `;

      ctrlEl.querySelector("#fc-unit-sel").addEventListener("change", (e) => {
        state.unit = e.target.value;
        paintChart();
      });
      if (showMeasureMetricSex) {
        ctrlEl.querySelector("#fc-measure-sel").addEventListener("change", (e) => {
          state.measure = e.target.value;
          paintChart();
        });
        ctrlEl.querySelector("#fc-metric-sel").addEventListener("change", (e) => {
          state.metric = e.target.value;
          paintChart();
        });
        ctrlEl.querySelector("#fc-sex-sel").addEventListener("change", (e) => {
          state.sex = e.target.value;
          paintChart();
        });
      }

      if (state.activeView === "trajectories") {
        stageEl.innerHTML = `
          <div class="card-header">
            <div>
              <h2 class="card-title" id="fc-main-title">2025–2040 Real Contract 1 Forecasts</h2>
              <p class="card-subtitle">Business-as-Usual (BAU) 95% UI band vs. clean-fuel counterfactual targets (25%, 50%, 75%, 100%)</p>
            </div>
            <span class="badge badge-real">Real GBD 2023 + HistGBDT</span>
          </div>
          <div id="chart-fc-main" class="chart-box-tall"></div>
          <div class="chart-caption">
            <span id="fc-main-cap"></span>
            <span><strong>Source:</strong> <code>data/processed/forecasts_stage3.csv</code> (validated real Stage 2 contract).</span>
          </div>
        `;
      } else {
        stageEl.innerHTML = `
          <div class="card-header">
            <div>
              <h2 class="card-title" id="fc-main-title">Contract 3 Exposure–Response Surrogate</h2>
              <p class="card-subtitle">Log-linear burden multiplier M = exp(β · Δc) calibrated from real Stage 2 counterfactuals</p>
            </div>
            <span class="badge badge-real">Real Surrogate (r_fm = 1.05 Proxy)</span>
          </div>
          <div id="chart-fc-main" class="chart-box-tall"></div>
          <div class="chart-caption">
            <span id="fc-main-cap"></span>
            <span><strong>Source:</strong> <code>data/processed/exposure_response_surrogate.json</code> (female curve uses proxy elasticity).</span>
          </div>
        `;
      }
      paintChart();
    }

    function paintChart() {
      if (state.activeView === "trajectories") {
        const unitRows = realFc.filter(
          (r) => r.location === state.unit && r.measure === state.measure && r.metric === state.metric && r.sex === state.sex
        );
        const bauRows = unitRows.filter((r) => r.scenario === "bau").sort((a, b) => a.year - b.year);
        const years = bauRows.map((r) => r.year);
        const traces = [
          {
            x: [...years, ...years.slice().reverse()],
            y: [...bauRows.map((r) => r.upper), ...bauRows.map((r) => r.lower).reverse()],
            fill: "toself",
            fillcolor: "rgba(223, 139, 108, 0.16)",
            line: { color: "transparent" },
            hoverinfo: "skip",
            showlegend: false,
            type: "scatter",
          },
          {
            x: years,
            y: bauRows.map((r) => r.val),
            name: "BAU (Real Stage 2)",
            mode: "lines+markers",
            line: { color: "#DF8B6C", width: 2.6 },
            marker: { size: 5 },
            customdata: bauRows.map((r) => [r.lower, r.upper, r.clean_fuel_share]),
            hovertemplate:
              "<b>BAU (Real)</b><br>Year: %{x}<br>Value: %{y:,.1f}<br>95% UI: [%{customdata[0]:,.1f}, %{customdata[1]:,.1f}]<br>Clean Fuel: %{customdata[2]:.1%}<extra></extra>",
          },
        ];

        const cfMeta = [
          { key: "cf_25", label: "cf_25 (25% clean fuel)", color: "#A6AAA6" },
          { key: "cf_50", label: "cf_50 (50% clean fuel)", color: "#D6C7A9" },
          { key: "cf_75", label: "cf_75 (75% clean fuel)", color: "#83CDD1" },
          { key: "cf_100", label: "cf_100 (100% clean fuel)", color: "#68BBC2" },
        ];
        for (const sc of cfMeta) {
          const scRows = unitRows.filter((r) => r.scenario === sc.key).sort((a, b) => a.year - b.year);
          if (!scRows.length) continue;
          traces.push({
            x: scRows.map((r) => r.year),
            y: scRows.map((r) => r.val),
            name: sc.label,
            mode: "lines+markers",
            line: { color: sc.color, width: 2, dash: "dot" },
            marker: { size: 7 },
            hovertemplate: `<b>${sc.label}</b><br>Year: %{x}<br>Value: %{y:,.1f}<extra></extra>`,
          });
        }

        const unitLabel = state.metric === "Rate_per_100k" ? `${state.measure} ASR (per 100k)` : `${state.measure} (All-Age Count)`;
        global.AppCharts.plot("chart-fc-main", traces, {
          xaxis: { title: "Forecast Year (2025–2040)" },
          yaxis: { title: unitLabel },
          legend: { orientation: "h", y: -0.2 },
          margin: { l: 74, r: 20, t: 18, b: 84 },
        });

        const bau2040 = bauRows.find((r) => r.year === 2040);
        document.getElementById("fc-main-title").textContent = `${state.unit} — ${unitLabel} (${state.sex})`;
        document.getElementById("fc-main-cap").textContent = bau2040
          ? `Real 2040 BAU: ${fmtNum(bau2040.val, 1)} [95% UI: ${fmtNum(bau2040.lower, 1)}–${fmtNum(bau2040.upper, 1)}] at ${(Number(bau2040.clean_fuel_share) * 100).toFixed(1)}% clean fuel.`
          : "Real Contract 1 forecast.";
      } else if (state.activeView === "surrogate") {
        const uSurr = (realSurr.units || {})[state.unit] || {};
        const beta = Number(uSurr.slope_clean_fuel ?? 0.0);
        const rFm = Number(uSurr.sex_elasticity_ratio_f_to_m || 1.05);
        const baseCf = Number(uSurr.baseline_clean_fuel || 0.5);
        const sd = Number(uSurr.uncertainty_sd || 0.08);
        const maxDelta = Math.max(0.05, 1.0 - baseCf);
        const steps = Array.from({ length: 41 }, (_, i) => (i / 40) * maxDelta);

        global.AppCharts.plot(
          "chart-fc-main",
          [
            {
              x: steps.map((d) => d * 100),
              y: steps.map((d) => Math.exp(beta * d)),
              name: "Male multiplier exp(β · Δc)",
              mode: "lines",
              line: { color: "#D6C7A9", width: 2.6 },
              hovertemplate: "Δc: +%{x:.1f} pct pts<br>Male Multiplier: %{y:.3f}<extra></extra>",
            },
            {
              x: steps.map((d) => d * 100),
              y: steps.map((d) => Math.exp(rFm * beta * d)),
              name: `Female multiplier exp(${rFm}·β·Δc) [Proxy]`,
              mode: "lines",
              line: { color: "#68BBC2", width: 2.6 },
              hovertemplate: "Δc: +%{x:.1f} pct pts<br>Female Multiplier: %{y:.3f}<extra></extra>",
            },
          ],
          {
            xaxis: { title: "Additional Clean-Fuel Coverage Δc (% pts above 2040 BAU)" },
            yaxis: { title: "2040 Burden Multiplier (1.0 = BAU)", range: [0.15, 1.05] },
            legend: { orientation: "h", y: -0.2 },
            margin: { l: 64, r: 20, t: 18, b: 84 },
          }
        );

        document.getElementById("fc-main-title").textContent = `${state.unit} — Real Surrogate Dose–Response Curve`;
        document.getElementById("fc-main-cap").textContent =
          `slope_clean_fuel = ${beta.toFixed(4)} (${beta === 0 ? "saturated HistGBDT leaf >= 0.985" : "active slope"}), sex_elasticity_ratio_f_to_m = ${rFm} (proxy), baseline_clean_fuel = ${(baseCf * 100).toFixed(1)}%, uncertainty_sd = ${sd}.`;
      }
    }

    container.querySelectorAll("[data-fc-view]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeView = btn.getAttribute("data-fc-view");
        container.querySelectorAll("[data-fc-view]").forEach((b) => b.classList.toggle("active", b === btn));
        renderView();
      });
    });

    renderView();
  }

  global.PageForecast = { render };
})(window);
