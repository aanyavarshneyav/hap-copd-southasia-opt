/* Page 2: Historical Burden & Clean-Cooking Transitions (V2 Storytelling) */
(function (global) {
  "use strict";

  async function render(container) {
    const data = await global.App.loadJSON("historical_clean_fuel.json");
    const { esc, term, accordion, fmtNum, statusBadge } = global.App;
    const cfRecords = data.clean_fuel_records || [];
    const demoRecords = data.demographic_records || [];
    const gbd = data.gbd_burden_status || {};

    const state = {
      activeView: "cf_access", // "cf_access" | "fuel_mix" | "gbd_pending"
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
            <span class="badge badge-pending">GBD 2023 COPD Burden: Pending make stage1</span>
          </div>
        </div>
        <h1 class="page-title">Historical Burden &amp; Clean-Cooking Transitions</h1>
        <p class="page-lead">
          Tracks real clean-cooking fuel transitions across India and Nepal (1990–2024) alongside the Stage 1 framework for South Asian COPD ${term("DALYs")} and deaths.
        </p>
      </header>

      <div class="takeaway-bar">
        <span class="takeaway-tag">Key Takeaway</span>
        <div>
          Verified WHO/World Bank data show India's rural clean-cooking access expanded from ~22% in 2010 to <strong>69.0%</strong> (95% UI: 53–82%) in 2024, while rural Nepal sits at <strong>25.0%</strong> (14–40%) and urban Nepal retains 39.3% firewood reliance. National GBD 2023 COPD burden series await raw GHDx export (<code>A-03</code>).
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
          <button type="button" class="tab-btn" role="tab" data-hist-view="gbd_pending">
            GBD 2023 COPD Burden (1990–2023 · Pending)
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
          <div class="pending-panel" style="border:none; background:transparent; padding:0.5rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
              <h3>GBD 2023 South Asia HAP-COPD Burden (1990–2023) — Pending Stage 1 Export</h3>
              <span class="badge badge-pending">Awaiting data/raw/gbd2023_*.csv</span>
            </div>
            <p id="gbd-pending-desc"></p>
            <div class="grid grid-3" style="margin-top:0.5rem;">
              <div class="card-elevated">
                <strong style="font-size:0.88rem;">Country Comparison</strong>
                <p style="margin:0.3rem 0 0; font-size:0.82rem; color:var(--text-secondary);">5-country South Asian panel (India &amp; Nepal highlighted vs. Pakistan, Bangladesh, Bhutan) across 1990 vs. 2023.</p>
              </div>
              <div class="card-elevated">
                <strong style="font-size:0.88rem;">EAPC &amp; Joinpoint Trends</strong>
                <p style="margin:0.3rem 0 0; font-size:0.82rem; color:var(--text-secondary);">1990–2023 annual trajectories with 95% UI ribbons, OLS EAPC, and <code>pwlf</code> inflection breakpoints.</p>
              </div>
              <div class="card-elevated">
                <strong style="font-size:0.88rem;">Country × Period Matrix</strong>
                <p style="margin:0.3rem 0 0; font-size:0.82rem; color:var(--text-secondary);">All-age counts and ASRs by country, year, and sex (no 5-year age-group trajectories per final methodology).</p>
              </div>
            </div>
            <div class="pending-checklist" style="margin-top:0.5rem;">
              ${(gbd.expected_files || []).map((f) => `<span class="pending-chip">${esc(f)}</span>`).join("")}
            </div>
          </div>
        `;

        const updateGbdText = () => {
          const mLabels = {
            DALYs_ASR: "Age-Standardized DALY Rate (per 100,000 population)",
            DALYs_Count: "All-Age Total DALYs (Number)",
            Deaths_ASR: "Age-Standardized Death Rate (per 100,000 population)",
            Deaths_Count: "All-Age Total Deaths (Number)",
            HAP_Share: "HAP-Attributable Share of Total COPD (%)",
          };
          const cText = state.gbdCountry === "ALL" ? "India, Nepal, Pakistan, Bangladesh, and Bhutan" : state.gbdCountry;
          const el = document.getElementById("gbd-pending-desc");
          if (el) {
            el.textContent = `Selected Stage 1 view: ${mLabels[state.gbdMetric]} across ${cText} (1990–2023). Raw IHME GBD 2023 CSVs require interactive GHDx export (Task A-03) before make stage1 can populate these charts—no synthetic GBD figures are shown.`;
          }
        };

        ctrlEl.querySelector("#gbd-country-sel").addEventListener("change", (e) => {
          state.gbdCountry = e.target.value;
          updateGbdText();
        });
        ctrlEl.querySelector("#gbd-metric-sel").addEventListener("change", (e) => {
          state.gbdMetric = e.target.value;
          updateGbdText();
        });
        updateGbdText();
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
