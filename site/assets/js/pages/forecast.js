/* Page 3: Forecasting & 2040 Projections (V2 Visual Storytelling) */
(function (global) {
  "use strict";

  async function render(container) {
    const data = await global.App.loadJSON("forecast_stage3_mock.json");
    const { esc, term, accordion, fmtNum } = global.App;
    const status = data.real_stage2_status || {};
    const mockFc = data.mock_forecasts || [];
    const mockSurr = data.mock_surrogate || {};

    const state = {
      activeView: "trajectories", // "trajectories" | "surrogate" | "cv_pending"
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
            <span class="badge badge-pending">Real Stage 2 Forecasts: Pending make stage2</span>
            <span class="badge badge-mock">Preview: SYNTHETIC MOCK_V1</span>
          </div>
        </div>
        <h1 class="page-title">Forecasting &amp; 2040 Projections</h1>
        <p class="page-lead">
          Compares five statistical and machine-learning models across South Asia to project 2040 COPD burden with 95% uncertainty intervals and calibrate a fast response curve.
        </p>
      </header>

      ${global.App.mockBanner(
        "SYNTHETIC DATA PREVIEW (MOCK_V1) · REAL STAGE 2 PROJECTIONS PENDING",
        "Final 2024–2040 projections and rolling-origin CV scores (MAE, RMSE, MASE) await validated Stage 2 exports (make stage2). Interactive trajectories below display synthetic MOCK_V1 contract fixtures only."
      )}

      <!-- Level 1 & 2: Prominent Visual Stage -->
      <section class="section">
        <div class="tabs-bar" role="tablist" aria-label="Forecast exploration views">
          <button type="button" class="tab-btn active" role="tab" data-fc-view="trajectories">
            2025–2040 Contract Forecast Explorer (MOCK_V1)
          </button>
          <button type="button" class="tab-btn" role="tab" data-fc-view="surrogate">
            Exposure–Response Surrogate Curve (MOCK_V1)
          </button>
          <button type="button" class="tab-btn" role="tab" data-fc-view="cv_pending">
            Model Comparison &amp; Validation (Pending Real Data)
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

      if (state.activeView === "cv_pending") {
        ctrlEl.style.display = "none";
        stageEl.innerHTML = `
          <div class="pending-panel" style="border:none; background:transparent; padding:0.4rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
              <h3>Stage 2 Model Comparison &amp; Actual vs. Predicted — Awaiting Validated Exports</h3>
              <span class="badge badge-pending">Pending make stage2</span>
            </div>
            <p>
              No model accuracy metrics (MAE, RMSE, MASE) or real 2040 forecasts are invented here. Once Person A exports validated Stage 2 outputs, this view will compare ARIMA, ETS, Adapted Poisson Rate GLM, Random Forest, and HistGradientBoosting across three 5-year rolling-origin folds (2009–2013, 2014–2018, 2019–2023).
            </p>
            <div class="pending-checklist">
              <span class="pending-chip">outputs/tables/s2_rolling_cv_metrics.csv</span>
              <span class="pending-chip">outputs/tables/s2_forecasts_2040_summary.csv</span>
              <span class="pending-chip">data/processed/forecasts_stage3.csv</span>
              <span class="pending-chip">data/processed/exposure_response_surrogate.json</span>
            </div>
          </div>
        `;
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
              <h2 class="card-title" id="fc-main-title">2025–2040 Contract 1 Forecast Preview</h2>
              <p class="card-subtitle">Business-as-Usual (BAU) 95% UI band vs. clean-fuel counterfactual targets (25%, 50%, 75%, 100%)</p>
            </div>
            <span class="badge badge-mock">SYNTHETIC MOCK_V1</span>
          </div>
          <div id="chart-fc-main" class="chart-box-tall"></div>
          <div class="chart-caption">
            <span id="fc-main-cap"></span>
            <span><strong>Source:</strong> <code>data/processed/forecasts_stage3_mock.csv</code> (synthetic contract fixture).</span>
          </div>
        `;
      } else {
        stageEl.innerHTML = `
          <div class="card-header">
            <div>
              <h2 class="card-title" id="fc-main-title">Contract 3 Exposure–Response Surrogate</h2>
              <p class="card-subtitle">Log-linear burden multiplier M = exp(β · Δc) evaluating 2040 intervention gains in microseconds</p>
            </div>
            <span class="badge badge-mock">SYNTHETIC MOCK_V1</span>
          </div>
          <div id="chart-fc-main" class="chart-box-tall"></div>
          <div class="chart-caption">
            <span id="fc-main-cap"></span>
            <span><strong>Source:</strong> <code>data/processed/exposure_response_surrogate_mock.json</code> (female curve uses proxy elasticity).</span>
          </div>
        `;
      }
      paintChart();
    }

    function paintChart() {
      if (state.activeView === "trajectories") {
        const unitRows = mockFc.filter(
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
            name: "BAU (MOCK_V1)",
            mode: "lines+markers",
            line: { color: "#DF8B6C", width: 2.6 },
            marker: { size: 5 },
            customdata: bauRows.map((r) => [r.lower, r.upper, r.clean_fuel_share]),
            hovertemplate:
              "<b>BAU (MOCK_V1)</b><br>Year: %{x}<br>Value: %{y:,.1f}<br>95% UI: [%{customdata[0]:,.1f}, %{customdata[1]:,.1f}]<br>Clean Fuel: %{customdata[2]:.0%}<extra></extra>",
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
            hovertemplate: `<b>${sc.label} (MOCK_V1)</b><br>Year: %{x}<br>Value: %{y:,.1f}<extra></extra>`,
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
        document.getElementById("fc-main-title").textContent = `${state.unit} — ${unitLabel} (${state.sex}, MOCK_V1)`;
        document.getElementById("fc-main-cap").textContent = bau2040
          ? `Synthetic 2040 BAU: ${fmtNum(bau2040.val, 1)} [95% UI: ${fmtNum(bau2040.lower, 1)}–${fmtNum(bau2040.upper, 1)}].`
          : "Synthetic Contract 1 preview.";
      } else if (state.activeView === "surrogate") {
        const uSurr = (mockSurr.units || {})[state.unit] || {};
        const beta = Number(uSurr.slope_clean_fuel || -1.3);
        const rFm = Number(uSurr.sex_elasticity_ratio_f_to_m || 1.1);
        const baseCf = Number(uSurr.baseline_clean_fuel || 0.5);
        const sd = Number(uSurr.uncertainty_sd || 0.1);
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
            yaxis: { title: "2040 Burden Multiplier (1.0 = BAU)", range: [0.35, 1.05] },
            legend: { orientation: "h", y: -0.2 },
            margin: { l: 64, r: 20, t: 18, b: 84 },
          }
        );

        document.getElementById("fc-main-title").textContent = `${state.unit} — Surrogate Dose–Response Curve (MOCK_V1)`;
        document.getElementById("fc-main-cap").textContent =
          `slope_clean_fuel = ${beta}, sex_elasticity_ratio_f_to_m = ${rFm} (proxy), baseline_clean_fuel = ${(baseCf * 100).toFixed(0)}%, uncertainty_sd = ${sd}.`;
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
