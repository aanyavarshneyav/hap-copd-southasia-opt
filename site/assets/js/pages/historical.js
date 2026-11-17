/* Page 2: Historical Burden & Clean-Cooking Transitions (V2 Storytelling) */
(function (global) {
  "use strict";

  async function render(container) {
    const data = await global.App.loadJSON("historical_clean_fuel.json");
    const { esc, term, accordion, fmtNum, statusBadge } = global.App;
    const cfRecords = data.clean_fuel_records || [];
    const demoRecords = data.demographic_records || [];
    const gbd = data.gbd_burden_status || {};
    const gbdSummary = data.gbd_summary || [];
    const eapcRows = data.eapc_joinpoint || [];
    const pwlfRows = data.pwlf_breakpoints || [];
    const shareRows = data.attributable_share || [];

    const state = {
      activeView: "cf_access", // "cf_access" | "fuel_mix" | "gbd_burden"
      cfCountry: "BOTH",
      cfResidence: "urban_rural",
      showDhsOverlay: true,
      fuelCountry: "India",
      fuelResidence: "rural",
      gbdCountry: "ALL",
      gbdMetric: "DALYs_ASR",
    };

    const demoTableHtml = `
      <p style="font-size:0.86rem; color:var(--text-secondary); margin:0 0 0.85rem; max-width:70ch;">
        Stage 0 apportionment and Stage 3 costing use <strong>UN WUP 2025 National Definitions</strong> (not UN DEGURBA Level 1) because WHO SDG 7.1.2 and DHS/NFHS clean-cooking surveys stratify households using national urban/rural definitions.
      </p>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Unit</th>
              <th>Year</th>
              <th>Classification Frame</th>
              <th class="num-right">Population (Persons)</th>
              <th class="num-right">Derived / Obs. Households</th>
              <th class="num-right">Avg. HH Size</th>
              <th>Provenance</th>
            </tr>
          </thead>
          <tbody>
            ${demoRecords
              .filter((r) => ["2011", "2021", "2025", "2040"].includes(String(r.reference_year)) && r.residence !== "total")
              .map(
                (r) => `
              <tr>
                <td><strong>${esc(r.unit)}</strong></td>
                <td>${esc(r.reference_year)}</td>
                <td><code>${esc(r.urban_rural_definition)}</code></td>
                <td class="num-right">${fmtNum(r.population_total, 0)}</td>
                <td class="num-right">${r.households ? fmtNum(r.households, 0) : "—"}</td>
                <td class="num-right">${r.avg_household_size ? Number(r.avg_household_size).toFixed(2) : "—"}</td>
                <td>${statusBadge(r.status)} <span style="font-size:0.75rem; color:var(--text-muted);">${esc(r.source_name)}</span></td>
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;

    const methHtml = `
      <ul class="callout-list">
        <li><strong>All-Age Counts vs. ${term("ASR")} Separation:</strong> Per the repository's final GBD compromise, Stage 1 analyses aggregated all-age burden counts (<code>Number</code>) separately from age-standardized rates (<code>Rate per 100k</code>). No 5-year age-group HAP-COPD trajectories are implied or plotted.</li>
        <li><strong>${term("EAPC")} &amp; ${term("Joinpoint")} Trends:</strong> Fits <code>ln(ASR_t) = α + βt + ε_t</code> via OLS (1990–2023) for overall annual percentage change, plus piecewise linear regression (<code>pwlf</code>, 0–3 breakpoints via BIC/AIC) to detect inflection years.</li>
        <li><strong>People vs. Households Gap:</strong> Across DHS surveys, household clean-fuel coverage exceeds population coverage by +0.7 to +4.2 percentage points because larger households rely more heavily on solid fuels.</li>
        <li><strong>Descriptive Guardrail:</strong> Clean-cooking expansion and COPD burden trends are descriptive associations; causal impact cannot be inferred from historical trajectories alone.</li>
      </ul>
    `;

    container.innerHTML = `
      <header class="page-header">
        <div class="page-header-top">
          <span class="page-eyebrow">02 · Stage 1 Historical Epidemiology &amp; Exposure Drivers</span>
          <div class="page-header-meta" style="margin-top:0;">
            <span class="badge badge-real">Clean Cooking (1990–2024): Real Data</span>
            <span class="badge badge-real">GBD 2023 COPD Burden (1990–2023): Complete</span>
          </div>
        </div>
        <h1 class="page-title">Historical Burden &amp; Clean-Cooking Transitions</h1>
        <p class="page-lead">
          Tracks real clean-cooking fuel transitions across India and Nepal (1990–2024) alongside complete Stage 1 GBD 2023 estimates of South Asian COPD ${term("DALYs")} and deaths.
        </p>
      </header>

      <div class="takeaway-bar">
        <span class="takeaway-tag">Key Takeaway</span>
        <div>
          Across 1990–2023, HAP-COPD age-standardized DALY rates fell sharply across all five nations (Bhutan −94.2%, India −56.4% to 593.9/100k, Pakistan −44.0%, Nepal −40.0% to 1,033.9/100k, Bangladesh −32.2%), yet population ageing raised absolute DALYs in India (+24.8% to 6.96M) and Nepal (+75.7% to 254.0k).
        </div>
      </div>

      <!-- Level 1 & 2: Single Prominent Interactive Visual Stage -->
      <section class="section">
        <div class="tabs-bar" role="tablist" aria-label="Historical exploration views">
          <button type="button" class="tab-btn active" role="tab" data-hist-view="cf_access">
            Clean-Cooking Access (2000–2024 · Real)
          </button>
          <button type="button" class="tab-btn" role="tab" data-hist-view="fuel_mix">
            Primary Fuel Mix (1990–2023 · Real)
          </button>
          <button type="button" class="tab-btn" role="tab" data-hist-view="gbd_burden">
            GBD 2023 COPD Burden &amp; EAPC (1990–2023 · Real)
          </button>
        </div>

        <div id="hist-controls-mount" class="controls-bar"></div>

        <div id="hist-stage-mount" class="card"></div>
      </section>

      <!-- Compact 2024 Snapshot Strip -->
      <section class="section">
        <div class="grid grid-4">
          <div class="kpi-card">
            <div class="kpi-label">IND Urban (WHO 2024)</div>
            <div class="kpi-value">96.0%</div>
            <div class="kpi-detail">95% UI: 88%–100% (people) · NFHS-5: 89.7% HH</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">IND Rural (WHO 2024)</div>
            <div class="kpi-value">69.0%</div>
            <div class="kpi-detail">95% UI: 53%–82% (people) · NFHS-5: 43.2% HH</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">NPL Urban (WHO 2024)</div>
            <div class="kpi-value">64.0%</div>
            <div class="kpi-detail">95% UI: 34%–93% (people) · NPHC 2021: 58.5% HH</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">NPL Rural (WHO 2024)</div>
            <div class="kpi-value">25.0%</div>
            <div class="kpi-detail">95% UI: 14%–40% (people) · NPHC 2021: 20.4% HH</div>
          </div>
        </div>
      </section>

      <!-- Level 3: Expandable Technical Details -->
      ${accordion(
        "Technical Details: Urban/Rural Classification &amp; Household Framing (UN WUP 2025 vs. DEGURBA)",
        '<span class="badge badge-real">Census · WUP 2025 · WPP 2024</span>',
        demoTableHtml
      )}
      ${accordion(
        "Methodology &amp; Equations: All-Age vs. ASR, EAPC, Joinpoint &amp; Descriptive Guardrails",
        '<span class="badge badge-derived">Stage 1 Spec</span>',
        methHtml
      )}
    `;

    function renderView() {
      const ctrlEl = document.getElementById("hist-controls-mount");
      const stageEl = document.getElementById("hist-stage-mount");

      if (state.activeView === "cf_access") {
        ctrlEl.innerHTML = `
          <div class="control-group">
            <label for="cf-country-sel">Country</label>
            <select id="cf-country-sel" class="control-select">
              <option value="BOTH" ${state.cfCountry === "BOTH" ? "selected" : ""}>India &amp; Nepal</option>
              <option value="India" ${state.cfCountry === "India" ? "selected" : ""}>India Only</option>
              <option value="Nepal" ${state.cfCountry === "Nepal" ? "selected" : ""}>Nepal Only</option>
            </select>
          </div>
          <div class="control-group">
            <label for="cf-res-sel">Residence Stratum</label>
            <select id="cf-res-sel" class="control-select">
              <option value="urban_rural" ${state.cfResidence === "urban_rural" ? "selected" : ""}>Urban vs. Rural</option>
              <option value="total" ${state.cfResidence === "total" ? "selected" : ""}>National Total</option>
              <option value="urban" ${state.cfResidence === "urban" ? "selected" : ""}>Urban Only</option>
              <option value="rural" ${state.cfResidence === "rural" ? "selected" : ""}>Rural Only</option>
              <option value="all" ${state.cfResidence === "all" ? "selected" : ""}>All Strata</option>
            </select>
          </div>
          <div class="control-group">
            <label for="cf-dhs-toggle">DHS / NFHS Survey Markers</label>
            <select id="cf-dhs-toggle" class="control-select">
              <option value="show" ${state.showDhsOverlay ? "selected" : ""}>Overlay DHS Household Markers</option>
              <option value="hide" ${!state.showDhsOverlay ? "selected" : ""}>WHO People Series Only</option>
            </select>
          </div>
        `;

        stageEl.innerHTML = `
          <div class="card-header">
            <div>
              <h2 class="card-title">Clean-Cooking Access Trajectory (2000–2024)</h2>
              <p class="card-subtitle">Share of population primarily using clean cooking fuels &amp; technologies (%, 95% UI shaded)</p>
            </div>
            <span class="badge badge-real">Real Data · WHO SDG 7.1.2</span>
          </div>
          <div id="chart-hist-main" class="chart-box-tall"></div>
          <div class="chart-caption">
            <span><strong>Unit:</strong> % of population (<code>people</code> basis for WHO lines; diamond markers show <code>households</code> basis from DHS/NFHS).</span>
            <span><strong>Source:</strong> WHO GHO <code>PHE_HHAIR_PROP_POP_CLEAN_FUELS</code>, World Bank WDI (0.0 pt diff), DHS API.</span>
          </div>
        `;

        ctrlEl.querySelector("#cf-country-sel").addEventListener("change", (e) => {
          state.cfCountry = e.target.value;
          paintCfChart();
        });
        ctrlEl.querySelector("#cf-res-sel").addEventListener("change", (e) => {
          state.cfResidence = e.target.value;
          paintCfChart();
        });
        ctrlEl.querySelector("#cf-dhs-toggle").addEventListener("change", (e) => {
          state.showDhsOverlay = e.target.value === "show";
          paintCfChart();
        });

        paintCfChart();
      } else if (state.activeView === "fuel_mix") {
        ctrlEl.innerHTML = `
          <div class="control-group">
            <label for="fuel-country-sel">Country</label>
            <select id="fuel-country-sel" class="control-select">
              <option value="India" ${state.fuelCountry === "India" ? "selected" : ""}>India</option>
              <option value="Nepal" ${state.fuelCountry === "Nepal" ? "selected" : ""}>Nepal</option>
            </select>
          </div>
          <div class="control-group">
            <label for="fuel-res-sel">Residence Stratum</label>
            <select id="fuel-res-sel" class="control-select">
              <option value="rural" ${state.fuelResidence === "rural" ? "selected" : ""}>Rural</option>
              <option value="urban" ${state.fuelResidence === "urban" ? "selected" : ""}>Urban</option>
              <option value="total" ${state.fuelResidence === "total" ? "selected" : ""}>National Total</option>
            </select>
          </div>
        `;

        stageEl.innerHTML = `
          <div class="card-header">
            <div>
              <h2 class="card-title">Primary Cooking Fuel Composition (1990–2023)</h2>
              <p class="card-subtitle">Stacked population shares across biomass, gas, electricity, kerosene, coal, and charcoal</p>
            </div>
            <span class="badge badge-real">Real Data · WHO GHO</span>
          </div>
          <div id="chart-hist-main" class="chart-box-tall"></div>
          <div class="chart-caption">
            <span><strong>Unit:</strong> % of population (<code>people</code> basis, 1990–2023).</span>
            <span><strong>Source:</strong> WHO GHO <code>PHE_HHAIR_PROP_POP_CATEGORY_FUELS</code>.</span>
          </div>
        `;

        ctrlEl.querySelector("#fuel-country-sel").addEventListener("change", (e) => {
          state.fuelCountry = e.target.value;
          paintFuelChart();
        });
        ctrlEl.querySelector("#fuel-res-sel").addEventListener("change", (e) => {
          state.fuelResidence = e.target.value;
          paintFuelChart();
        });

        paintFuelChart();
      } else {
        ctrlEl.innerHTML = `
          <div class="control-group">
            <label for="gbd-country-sel">South Asia Panel Country</label>
            <select id="gbd-country-sel" class="control-select">
              <option value="ALL" ${state.gbdCountry === "ALL" ? "selected" : ""}>All 5 Countries (India &amp; Nepal Highlighted)</option>
              <option value="India" ${state.gbdCountry === "India" ? "selected" : ""}>India</option>
              <option value="Nepal" ${state.gbdCountry === "Nepal" ? "selected" : ""}>Nepal</option>
              <option value="Pakistan" ${state.gbdCountry === "Pakistan" ? "selected" : ""}>Pakistan</option>
              <option value="Bangladesh" ${state.gbdCountry === "Bangladesh" ? "selected" : ""}>Bangladesh</option>
              <option value="Bhutan" ${state.gbdCountry === "Bhutan" ? "selected" : ""}>Bhutan</option>
            </select>
          </div>
          <div class="control-group">
            <label for="gbd-metric-sel">Burden Measure</label>
            <select id="gbd-metric-sel" class="control-select">
              <option value="DALYs_ASR" ${state.gbdMetric === "DALYs_ASR" ? "selected" : ""}>DALYs Age-Standardized Rate (per 100k)</option>
              <option value="DALYs_Count" ${state.gbdMetric === "DALYs_Count" ? "selected" : ""}>DALYs All-Age Count (Number)</option>
              <option value="Deaths_ASR" ${state.gbdMetric === "Deaths_ASR" ? "selected" : ""}>Deaths Age-Standardized Rate (per 100k)</option>
              <option value="Deaths_Count" ${state.gbdMetric === "Deaths_Count" ? "selected" : ""}>Deaths All-Age Count (Number)</option>
              <option value="HAP_Share" ${state.gbdMetric === "HAP_Share" ? "selected" : ""}>HAP Share of Total COPD (%)</option>
            </select>
          </div>
        `;

        stageEl.innerHTML = `
          <div class="card-header">
            <div>
              <h2 class="card-title">GBD 2023 South Asia HAP-COPD Burden &amp; Structural Breakpoints (1990–2023)</h2>
              <p class="card-subtitle">All-age burden and age-standardized rates (ASRs per 100k), full-period EAPC, and PWLF breakpoints</p>
            </div>
            <span class="badge badge-real">Real GBD 2023 + UN WPP 2024</span>
          </div>
          <div id="chart-hist-gbd" class="chart-box-tall"></div>
          <div class="chart-caption">
            <span id="gbd-real-cap"></span>
            <span><strong>Source:</strong> <code>s1_attributable_share.csv</code>, <code>s1_historical_summary.csv</code>, <code>s1_eapc_joinpoint.csv</code>, <code>s1_pwlf_breakpoints.csv</code>.</span>
          </div>
          <div style="margin-top:1.25rem;" class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Location</th>
                  <th>Measure</th>
                  <th class="num-right">1990 Count</th>
                  <th class="num-right">2023 Count</th>
                  <th class="num-right">Count Δ%</th>
                  <th class="num-right">1990 ASR</th>
                  <th class="num-right">2023 ASR</th>
                  <th class="num-right">ASR Δ%</th>
                  <th class="num-right">EAPC (%/yr)</th>
                  <th>PWLF Breakpoint</th>
                </tr>
              </thead>
              <tbody id="gbd-summary-tbody"></tbody>
            </table>
          </div>
          <div style="margin-top:1.25rem;">
            <img src="assets/images/real/s1_country_period_heatmap.png" alt="Country by Period HAP-COPD DALY ASR Heatmap" style="width:100%; max-width:960px; border-radius:8px; border:1px solid var(--border-subtle);" />
          </div>
        `;

        const updateGbdView = () => {
          const wantMeasure = state.gbdMetric.startsWith("Deaths") ? "Deaths" : "DALYs";
          const isShare = state.gbdMetric === "HAP_Share";
          const locs = state.gbdCountry === "ALL"
            ? ["India", "Nepal", "Pakistan", "Bangladesh", "Bhutan", "IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]
            : [state.gbdCountry];
          const colorMap = {
            India: "#83CDD1",
            Nepal: "#DF8B6C",
            Pakistan: "#D6C7A9",
            Bangladesh: "#8A9A86",
            Bhutan: "#A6AAA6",
            IND_urban: "#68BBC2",
            IND_rural: "#54989E",
            NPL_urban: "#E5A58A",
            NPL_rural: "#C87152",
          };
          const traces = [];
          for (const loc of locs) {
            const rows = shareRows
              .filter((r) => r.location === loc && r.measure === wantMeasure)
              .sort((a, b) => a.year - b.year);
            if (!rows.length) continue;
            traces.push({
              x: rows.map((r) => r.year),
              y: rows.map((r) => (isShare ? r.hap_copd_share_of_total_copd_pct : r.hap_copd_asr)),
              name: loc,
              mode: "lines",
              line: { color: colorMap[loc] || "#83CDD1", width: loc.includes("_") ? 2 : 2.5, dash: loc.includes("_") ? "dot" : "solid" },
              hovertemplate: `<b>${loc}</b><br>Year: %{x}<br>${isShare ? "HAP Share: %{y:.1f}%" : "ASR: %{y:,.1f}/100k"}<extra></extra>`,
            });
          }
          const yTitle = isShare
            ? "HAP-COPD Share of Total COPD (%)"
            : `${wantMeasure} Age-Standardized Rate (per 100k)`;
          global.AppCharts.plot("chart-hist-gbd", traces, {
            xaxis: { title: "Year (1990–2023)" },
            yaxis: { title: yTitle },
            legend: { orientation: "h", y: -0.22 },
            margin: { l: 70, r: 20, t: 18, b: 92 },
          });

          const tbody = document.getElementById("gbd-summary-tbody");
          const sumFiltered = gbdSummary.filter(
            (r) =>
              r.sex === "both" &&
              (state.gbdCountry === "ALL" || r.location === state.gbdCountry) &&
              r.measure === wantMeasure
          );
          tbody.innerHTML = sumFiltered
            .map((r) => {
              const eRow = eapcRows.find(
                (e) => e.location === r.location && e.measure === r.measure && e.sex === "both" && e.period === "1990-2023"
              );
              const pRow = pwlfRows.find((p) => p.location === r.location && p.measure === r.measure);
              const bpText = pRow
                ? `${String(pRow.breakpoint_years || "").replace(/[\[\]np\.float64\(\)]/g, "")} (${Number(pRow.apc_segment_1).toFixed(2)}% → ${Number(pRow.apc_segment_2).toFixed(2)}%/yr)`
                : "—";
              return `
                <tr>
                  <td><strong>${esc(r.location)}</strong></td>
                  <td>${esc(r.measure)}</td>
                  <td class="num-right">${fmtNum(r.count_1990, 0)}</td>
                  <td class="num-right">${fmtNum(r.count_2023, 0)}</td>
                  <td class="num-right">${Number(r.count_pct_change) >= 0 ? "+" : ""}${Number(r.count_pct_change).toFixed(1)}%</td>
                  <td class="num-right">${fmtNum(r.asr_1990, 1)}</td>
                  <td class="num-right">${fmtNum(r.asr_2023, 1)}</td>
                  <td class="num-right">${Number(r.asr_pct_change).toFixed(1)}%</td>
                  <td class="num-right">${eRow ? `${Number(eRow.eapc).toFixed(2)}%` : "—"}</td>
                  <td style="font-size:0.78rem;">${esc(bpText)}</td>
                </tr>
              `;
            })
            .join("");

          const el = document.getElementById("gbd-real-cap");
          if (el) {
            el.textContent = `Showing ${yTitle} across ${state.gbdCountry === "ALL" ? "all 5 countries and 4 IND/NPL units" : state.gbdCountry}.`;
          }
        };

        ctrlEl.querySelector("#gbd-country-sel").addEventListener("change", (e) => {
          state.gbdCountry = e.target.value;
          updateGbdView();
        });
        ctrlEl.querySelector("#gbd-metric-sel").addEventListener("change", (e) => {
          state.gbdMetric = e.target.value;
          updateGbdView();
        });
        updateGbdView();
      }
    }

    function paintCfChart() {
      const who = cfRecords.filter((r) => r.source === "WHO GHO" && r.fuel_category === "clean_fuels_total");
      const dhs = cfRecords.filter((r) => String(r.source || "").startsWith("DHS/NFHS"));

      const countries = state.cfCountry === "BOTH" ? ["India", "Nepal"] : [state.cfCountry];
      const resMap = {
        urban_rural: ["urban", "rural"],
        total: ["total"],
        urban: ["urban"],
        rural: ["rural"],
        all: ["total", "urban", "rural"],
      };
      const residences = resMap[state.cfResidence] || ["urban", "rural"];
      const colors = {
        "India|urban": "#83CDD1",
        "India|rural": "#68BBC2",
        "India|total": "#EEEAE1",
        "Nepal|urban": "#D6C7A9",
        "Nepal|rural": "#DF8B6C",
        "Nepal|total": "#A6AAA6",
      };
      const fillColors = {
        "India|urban": "rgba(131, 205, 209, 0.13)",
        "India|rural": "rgba(104, 187, 194, 0.13)",
        "India|total": "rgba(238, 234, 225, 0.09)",
        "Nepal|urban": "rgba(214, 199, 169, 0.13)",
        "Nepal|rural": "rgba(223, 139, 108, 0.13)",
        "Nepal|total": "rgba(166, 170, 166, 0.1)",
      };

      const traces = [];
      for (const c of countries) {
        for (const res of residences) {
          const rows = who.filter((r) => r.country === c && r.residence === res).sort((a, b) => a.year - b.year);
          if (!rows.length) continue;
          const key = `${c}|${res}`;
          const years = rows.map((r) => r.year);
          traces.push({
            x: [...years, ...years.slice().reverse()],
            y: [...rows.map((r) => r.upper), ...rows.map((r) => r.lower).reverse()],
            fill: "toself",
            fillcolor: fillColors[key] || "rgba(166, 170, 166, 0.12)",
            line: { color: "transparent" },
            hoverinfo: "skip",
            showlegend: false,
            type: "scatter",
          });
          traces.push({
            x: years,
            y: rows.map((r) => r.val),
            name: `${c} · ${res} (WHO people)`,
            mode: "lines+markers",
            line: { color: colors[key] || "#EEEAE1", width: 2.4 },
            marker: { size: 4 },
            customdata: rows.map((r) => [r.lower, r.upper]),
            hovertemplate: `<b>${c} (${res})</b><br>Year: %{x}<br>Clean Fuel Access: %{y:.1f}% of pop<br>95% UI: [%{customdata[0]:.1f}%, %{customdata[1]:.1f}%]<extra></extra>`,
          });
        }
      }

      if (state.showDhsOverlay) {
        for (const c of countries) {
          for (const res of residences.filter((r) => r !== "total")) {
            const dhsHh = dhs
              .filter((r) => r.country === c && r.residence === res && r.unit_basis === "households")
              .sort((a, b) => a.year - b.year);
            if (dhsHh.length) {
              traces.push({
                x: dhsHh.map((r) => r.year),
                y: dhsHh.map((r) => r.val),
                name: `${c} · ${res} (DHS households)`,
                mode: "markers",
                marker: { symbol: "diamond", size: 8, color: colors[`${c}|${res}`] || "#EEEAE1", line: { color: "#171613", width: 1.5 } },
                hovertemplate: `<b>${c} (${res}) · DHS Survey</b><br>Year: %{x}<br>Clean Fuel: %{y:.1f}% of households<extra></extra>`,
              });
            }
          }
        }
      }

      global.AppCharts.plot("chart-hist-main", traces, {
        xaxis: { title: "Calendar Year (2000–2024)" },
        yaxis: { title: "Clean-Cooking Access (%)", range: [0, 104] },
        legend: { orientation: "h", y: -0.2 },
        margin: { l: 58, r: 20, t: 18, b: 84 },
      });
    }

    function paintFuelChart() {
      const rows = cfRecords.filter(
        (r) => r.source === "WHO GHO (Fuel Category)" && r.country === state.fuelCountry && r.residence === state.fuelResidence
      );
      const cats = [
        { key: "biomass", label: "Biomass", color: "#DF8B6C" },
        { key: "gas", label: "Gas (LPG/Natural)", color: "#68BBC2" },
        { key: "electricity", label: "Electricity", color: "#83CDD1" },
        { key: "kerosene", label: "Kerosene", color: "#B56A50" },
        { key: "coal", label: "Coal", color: "#928B80" },
        { key: "charcoal", label: "Charcoal", color: "#D6C7A9" },
      ];
      const traces = cats
        .map((cat) => {
          const sub = rows.filter((r) => r.fuel_category === cat.key).sort((a, b) => a.year - b.year);
          if (!sub.length) return null;
          return {
            x: sub.map((r) => r.year),
            y: sub.map((r) => r.val),
            name: cat.label,
            mode: "lines",
            stackgroup: "one",
            line: { width: 1.5, color: cat.color },
            hovertemplate: `<b>${cat.label}</b><br>Year: %{x}<br>Population Share: %{y:.1f}%<extra></extra>`,
          };
        })
        .filter(Boolean);

      global.AppCharts.plot("chart-hist-main", traces, {
        xaxis: { title: "Calendar Year (1990–2023)" },
        yaxis: { title: "Population Share (%)", range: [0, 105] },
        legend: { orientation: "h", y: -0.2 },
        margin: { l: 58, r: 20, t: 18, b: 84 },
      });
    }

    container.querySelectorAll("[data-hist-view]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeView = btn.getAttribute("data-hist-view");
        container.querySelectorAll("[data-hist-view]").forEach((b) => b.classList.toggle("active", b === btn));
        renderView();
      });
    });

    renderView();
  }

  global.PageHistorical = { render };
})(window);
