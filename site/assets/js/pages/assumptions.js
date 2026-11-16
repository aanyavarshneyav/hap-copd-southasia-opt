/* Page 4: Intervention & Assumptions Explorer (V2 Visual Storytelling) */
(function (global) {
  "use strict";

  async function render(container) {
    const data = await global.App.loadJSON("assumptions.json");
    const { esc, accordion, fmtNum, fmtUsdM, fmtPct, statusBadge } = global.App;

    const c4Rows = data.proposed_contract4_table || [];
    const baseComp = data.baseline_comparison || [];
    const budgetScen = data.budget_scenarios || [];
    const feasAudit = data.feasibility_audit || [];
    const gapMatrix = data.parameter_gap_matrix || [];
    const srcReg = data.source_register || [];
    const mockParams = data.mock_params || {};

    const state = {
      chartView: "baselines", // "baselines" | "budget_costs"
      tableTab: "params", // "params" | "feasibility" | "budgets"
      paramView: "real_proposed", // "real_proposed" | "mock_v1"
      unitFilter: "ALL",
      provFilter: "ALL",
      srcSearch: "",
    };

    const interventionsHtml = `
      <div class="grid grid-2">
        ${(data.interventions || [])
          .map(
            (iv) => `
          <div class="card-elevated">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
              <strong>${esc(iv.name)} <code style="font-size:0.76rem; color:var(--accent-blue-soft);">${esc(iv.id)}</code></strong>
              ${iv.id === "infra_lpg_grid" ? '<span class="badge badge-mock">Unresolved (null)</span>' : '<span class="badge badge-derived">5-Yr Cohort Package</span>'}
            </div>
            <p style="margin:0 0 0.4rem; font-size:0.83rem; color:var(--text-secondary);">${esc(iv.description)}</p>
            <div style="font-size:0.77rem; color:var(--accent-teal-soft);"><strong>Prerequisite:</strong> ${esc(iv.infra_required)}</div>
          </div>
        `
          )
          .join("")}
      </div>
    `;

    const gapMatrixAccordionBody = `
      <div class="controls-bar">
        <div class="control-group">
          <label for="prov-filter">Filter Provenance Status</label>
          <select id="prov-filter" class="control-select">
            <option value="ALL">All Parameter Rows (${gapMatrix.length})</option>
            <option value="acquired">Acquired / Sourced</option>
            <option value="derived">Derived</option>
            <option value="assumption">Assumption / Partial</option>
            <option value="unresolved">Unresolved Gap (null)</option>
            <option value="blocked">Blocked on Stage 2 (Person A)</option>
          </select>
        </div>
      </div>
      <div id="gap-matrix-mount"></div>
    `;

    const sourceRegAccordionBody = `
      <div class="controls-bar">
        <div class="control-group" style="flex:1; max-width:420px;">
          <label for="src-search">Search Source Register</label>
          <input id="src-search" class="control-input" type="search" placeholder="Filter by intervention, ASM ID, citation, or status…" />
        </div>
      </div>
      <div id="source-register-mount"></div>
    `;

    container.innerHTML = `
      <header class="page-header">
        <div class="page-header-top">
          <span class="page-eyebrow">04 · Contract 4 Parameters, Baselines &amp; Provenance</span>
          <div class="page-header-meta" style="margin-top:0;">
            <span class="badge badge-real">Sourced &amp; Derived: b/param-sourcing</span>
            <span class="badge badge-mock">Infra Capex &amp; Budget: Unresolved (null)</span>
            <span class="badge badge-assumption">Scenario Assumptions Only</span>
          </div>
        </div>
        <h1 class="page-title">Interventions &amp; Assumptions Explorer</h1>
        <p class="page-lead">
          Presents scenario unit costs, stove effectiveness, adoption ceilings, and the explicit bridge separating population-based clean-fuel access from household-based coverage.
        </p>
      </header>

      <div class="takeaway-bar">
        <span class="takeaway-tag">Key Takeaway</span>
        <div>
          Household clean-fuel coverage consistently exceeds population coverage by <strong>+0.7 to +4.2 pts</strong> in surveys, and rural India (209.2M households) drives ~87% of India–Nepal full-gap LPG scenario cost. All figures are <em>scenario assumptions</em>, never verified programme expenditures.
        </div>
      </div>

      <!-- Level 1: Primary Visual -->
      <section class="section">
        <div class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title" id="asm-chart-title">Population vs. Household Clean-Fuel Baselines</h2>
              <p class="card-subtitle" id="asm-chart-sub">WHO SDG 7.1.2 (people denominator) vs. observed NFHS-5 / NPHC 2021 (household denominator)</p>
            </div>
            <div class="btn-group">
              <button type="button" class="btn btn-sm active" data-asm-chart="baselines">Coverage Baselines (%)</button>
              <button type="button" class="btn btn-sm" data-asm-chart="budget_costs">Scenario Gap Costs ($M)</button>
            </div>
          </div>
          <div id="chart-asm-main" class="chart-box"></div>
          <div class="chart-caption" id="asm-chart-cap"></div>
        </div>
      </section>

      <!-- Level 2: Interactive Parameter Explorer -->
      <section class="section">
        <div class="section-header">
          <div>
            <h2 class="section-title">Parameter &amp; Scenario Explorer</h2>
            <p class="section-desc">Compare sourced/derived 2024 USD inputs against synthetic <code>MOCK_V1</code> placeholders and audit feasibility headroom.</p>
          </div>
          <div class="btn-group">
            <button type="button" class="btn btn-teal btn-sm active" id="btn-param-real">Sourced / Derived (2024 USD)</button>
            <button type="button" class="btn btn-sm" id="btn-param-mock">Synthetic Mock (MOCK_V1)</button>
          </div>
        </div>

        <div class="tabs-bar" role="tablist">
          <button type="button" class="tab-btn active" data-asm-tab="params">Cost, Effectiveness &amp; Caps</button>
          <button type="button" class="tab-btn" data-asm-tab="feasibility">Feasibility &amp; Headroom Audit</button>
          <button type="button" class="tab-btn" data-asm-tab="budgets">Reconciled Budget Scenarios</button>
        </div>

        <div id="asm-controls-mount" class="controls-bar">
          <div class="control-group">
            <label for="asm-unit-filter">Geographical Unit</label>
            <select id="asm-unit-filter" class="control-select">
              <option value="ALL">All 4 Units (IND &amp; NPL Urban/Rural)</option>
              <option value="IND_urban">India Urban (IND_urban)</option>
              <option value="IND_rural">India Rural (IND_rural)</option>
              <option value="NPL_urban">Nepal Urban (NPL_urban)</option>
              <option value="NPL_rural">Nepal Rural (NPL_rural)</option>
            </select>
          </div>
          <div id="param-mode-note" style="font-size:0.82rem; color:var(--text-secondary); align-self:center;"></div>
        </div>

        <div id="asm-table-mount"></div>
      </section>

      <!-- Level 3: Expandable Technical Details -->
      ${accordion(
        "Intervention Definitions, Alternative A Costing &amp; Infrastructure Prerequisites (q &lt; p)",
        '<span class="badge badge-derived">4 Levers · DEC-021</span>',
        interventionsHtml
      )}
      ${accordion(
        "Parameter Provenance &amp; Gap Matrix (Sourced vs. Derived vs. Assumption vs. Unresolved)",
        `<span class="badge badge-assumption">${gapMatrix.length} Parameter Rows</span>`,
        gapMatrixAccordionBody
      )}
      ${accordion(
        "Literature &amp; Programme Source Register (with Clickable URLs &amp; DOIs)",
        `<span class="badge badge-real">${srcReg.length} Citations</span>`,
        sourceRegAccordionBody
      )}
    `;

    function renderMainChart() {
      const titleEl = document.getElementById("asm-chart-title");
      const subEl = document.getElementById("asm-chart-sub");
      const capEl = document.getElementById("asm-chart-cap");

      if (state.chartView === "baselines") {
        titleEl.textContent = "Population vs. Household Clean-Fuel Baselines";
        subEl.textContent = "Separate denominators: WHO SDG 7.1.2 (people) drives exposure PAFs; NFHS-5 / NPHC 2021 (households) governs adoption headroom";
        capEl.innerHTML =
          '<span><strong>Unit:</strong> % clean-cooking share on respective denominator. Linked via <code>coverage_bridge_factor</code> (central 1.0).</span><span><strong>Sources:</strong> WHO GHO (2024), India NFHS-5 (2020), Nepal NPHC 2021.</span>';

        const units = baseComp.map((r) => r.unit);
        global.AppCharts.plot(
          "chart-asm-main",
          [
            {
              x: units,
              y: baseComp.map((r) => r.baseline_clean_fuel_population * 100),
              name: "WHO SDG 7.1.2 (2024, People %)",
              type: "bar",
              marker: { color: "#68BBC2" },
              hovertemplate: "<b>%{x}</b><br>WHO 2024 (People): %{y:.1f}%<extra></extra>",
            },
            {
              x: units,
              y: baseComp.map((r) => r.baseline_clean_fuel_households_obs * 100),
              name: "Observed Survey/Census (Households %)",
              type: "bar",
              marker: { color: "#D6C7A9" },
              hovertemplate: "<b>%{x}</b><br>Obs. Survey/Census (HH): %{y:.1f}%<extra></extra>",
            },
            {
              x: units,
              y: baseComp.map((r) => r.household_2024_sensitivity_derived * 100),
              name: "2024 Derived HH Sensitivity (%)",
              type: "bar",
              marker: { color: "#DF8B6C" },
              hovertemplate: "<b>%{x}</b><br>2024 Derived HH Sens.: %{y:.1f}%<extra></extra>",
            },
          ],
          {
            barmode: "group",
            xaxis: { title: "Geographical Allocation Unit" },
            yaxis: { title: "Clean-Cooking Share (%)", range: [0, 105] },
            legend: { orientation: "h", y: -0.2 },
            margin: { l: 58, r: 20, t: 18, b: 84 },
          }
        );
      } else {
        titleEl.textContent = "Reconciled 15-Year Full-Gap Fiscal Scenario Costs (Excl. Infra Capex)";
        subEl.textContent = "One-time 5-year cohort package costs (2024 USD Millions); regional infra_lpg_grid remains unresolved (null)";
        capEl.innerHTML =
          '<span><strong>Unit:</strong> 2024 USD Millions (scenario assumptions, NOT verified programme expenditure).</span><span><strong>Source:</strong> B-09 reconciled budget scenarios (<code>b09_budget_scenarios.csv</code>).</span>';

        const names = budgetScen.map((b) => b.scenario_id.replace("H_", "").replace("P_", ""));
        global.AppCharts.plot(
          "chart-asm-main",
          [
            {
              x: names,
              y: budgetScen.map((b) => b.IND_rural_cost_usd_m),
              name: "IND_rural ($M)",
              type: "bar",
              marker: { color: "#68BBC2" },
            },
            {
              x: names,
              y: budgetScen.map((b) => b.IND_urban_cost_usd_m),
              name: "IND_urban ($M)",
              type: "bar",
              marker: { color: "#83CDD1" },
            },
            {
              x: names,
              y: budgetScen.map((b) => b.NPL_urban_cost_usd_m),
              name: "NPL_urban ($M)",
              type: "bar",
              marker: { color: "#D6C7A9" },
            },
            {
              x: names,
              y: budgetScen.map((b) => b.NPL_rural_cost_usd_m),
              name: "NPL_rural ($M)",
              type: "bar",
              marker: { color: "#DF8B6C" },
            },
          ],
          {
            barmode: "stack",
            xaxis: { title: "Fiscal Scenario Basis" },
            yaxis: { title: "Full-Gap Cost (2024 USD Millions)" },
            legend: { orientation: "h", y: -0.22 },
            margin: { l: 68, r: 20, t: 18, b: 86 },
          }
        );
      }
    }

    function renderExplorerTable() {
      const mount = document.getElementById("asm-table-mount");
      const ctrlEl = document.getElementById("asm-controls-mount");
      const noteEl = document.getElementById("param-mode-note");

      if (state.tableTab === "feasibility") {
        ctrlEl.style.display = "none";
        mount.innerHTML = `
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Unit</th>
                  <th>Initial Infra</th>
                  <th class="num-right">WHO 2024 Pop</th>
                  <th class="num-right">Obs. HH Base</th>
                  <th class="num-right">Ceiling</th>
                  <th class="num-right">Pop Gap</th>
                  <th class="num-right">HH Gap</th>
                  <th class="num-right">Unlocked LPG Cap</th>
                  <th>Diagnosis</th>
                </tr>
              </thead>
              <tbody>
                ${feasAudit
                  .map(
                    (r) => `
                  <tr>
                    <td><strong>${esc(r.unit)}</strong></td>
                    <td>${r.infra_ready_init_assumed ? '<span class="badge badge-real">Ready</span>' : '<span class="badge badge-assumption">Unready</span>'}</td>
                    <td class="num-right">${fmtPct(r.pop_2024_baseline, 1)}</td>
                    <td class="num-right">${fmtPct(r.hh_obs_baseline, 1)}</td>
                    <td class="num-right">${fmtPct(r.coverage_ceiling, 0)}</td>
                    <td class="num-right">${fmtPct(r.gap_pop_2024, 1)}</td>
                    <td class="num-right">${fmtPct(r.gap_hh_obs, 1)}</td>
                    <td class="num-right">${fmtPct(r.lpg_unlocked_cap_sum, 0)}</td>
                    <td style="font-size:0.78rem; color:var(--text-secondary);">${esc(r.notes)}</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `;
        return;
      }

      if (state.tableTab === "budgets") {
        ctrlEl.style.display = "none";
        mount.innerHTML = `
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Basis</th>
                  <th>Status</th>
                  <th class="num-right">IND_urban</th>
                  <th class="num-right">IND_rural</th>
                  <th class="num-right">NPL_urban</th>
                  <th class="num-right">NPL_rural</th>
                  <th class="num-right">100% Gap ($M)</th>
                  <th class="num-right">50% ($M)</th>
                  <th class="num-right">25% ($M)</th>
                </tr>
              </thead>
              <tbody>
                ${budgetScen
                  .map(
                    (b) => `
                  <tr>
                    <td><strong>${esc(b.scenario_label)}</strong><div style="font-size:0.74rem; color:var(--text-muted);">${esc(b.note)}</div></td>
                    <td><code>${esc(b.denominator_basis)}</code></td>
                    <td>${statusBadge(b.baseline_classification)}</td>
                    <td class="num-right">${fmtUsdM(b.IND_urban_cost_usd_m, 1)}</td>
                    <td class="num-right">${fmtUsdM(b.IND_rural_cost_usd_m, 1)}</td>
                    <td class="num-right">${fmtUsdM(b.NPL_urban_cost_usd_m, 1)}</td>
                    <td class="num-right">${fmtUsdM(b.NPL_rural_cost_usd_m, 1)}</td>
                    <td class="num-right"><strong>${fmtUsdM(b.full_gap_100pct_usd_m, 1)}</strong></td>
                    <td class="num-right">${fmtUsdM(b.budget_50pct_usd_m, 1)}</td>
                    <td class="num-right">${fmtUsdM(b.budget_25pct_usd_m, 1)}</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `;
        return;
      }

      ctrlEl.style.display = "flex";
      if (state.paramView === "mock_v1") {
        noteEl.innerHTML = `<span class="badge badge-mock">SYNTHETIC MOCK_V1</span> Placeholder $M values (pooled cap: <strong>$500.0M</strong>) for pipeline testing only.`;
        const units = state.unitFilter === "ALL" ? data.units : [state.unitFilter];
        const rows = [];
        for (const u of units) {
          for (const iv of ["lpg_subsidy", "improved_biomass", "electric_induction", "infra_lpg_grid"]) {
            const cost = (mockParams.unit_costs_usd_millions?.[u] || {})[iv];
            const eff = (mockParams.effectiveness?.[u] || {})[iv];
            const ceil = (mockParams.coverage_ceiling || {})[u];
            const caps = (mockParams.max_adoption_per_period?.[u] || {})[iv] || {};
            rows.push(`
              <tr>
                <td><strong>${esc(u)}</strong></td>
                <td><code>${esc(iv)}</code></td>
                <td class="num-right">— (mock)</td>
                <td class="num-right">${fmtUsdM(cost, 1)}</td>
                <td class="num-right">${eff !== undefined ? Number(eff).toFixed(2) : "—"}</td>
                <td class="num-right">${ceil !== undefined ? fmtPct(ceil, 0) : "—"}</td>
                <td class="num-right">${caps.P1 ?? "—"} / ${caps.P2 ?? "—"} / ${caps.P3 ?? "—"}</td>
                <td><span class="badge badge-mock">Synthetic MOCK_V1</span></td>
              </tr>
            `);
          }
        }
        mount.innerHTML = `
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr><th>Unit</th><th>Intervention</th><th class="num-right">USD/HH</th><th class="num-right">Full +1.0 ($M)</th><th class="num-right">Eff. (η)</th><th class="num-right">Ceiling</th><th class="num-right">Caps (P1/P2/P3)</th><th>Status</th></tr>
              </thead>
              <tbody>${rows.join("")}</tbody>
            </table>
          </div>
        `;
        return;
      }

      noteEl.innerHTML = `<span class="badge badge-real">B-07/B-09 Sourced &amp; Derived</span> 2024 USD/HH scaled by 2025 derived households. <code>infra_lpg_grid</code> remains <code>null</code>.`;
      const filtered = c4Rows.filter((r) => state.unitFilter === "ALL" || r.unit === state.unitFilter);
      mount.innerHTML = `
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Unit</th>
                <th>Intervention</th>
                <th class="num-right">2025 HH</th>
                <th class="num-right">USD / HH (Low–Central–High)</th>
                <th class="num-right">Full +1.0 ($M Central)</th>
                <th class="num-right">Eff. (η)</th>
                <th class="num-right">Ceiling</th>
                <th class="num-right">Caps (P1/P2/P3)</th>
                <th>Provenance</th>
              </tr>
            </thead>
            <tbody>
              ${filtered
                .map((r) => {
                  const hhRange =
                    r.usd_per_hh_central !== null && r.usd_per_hh_central !== undefined
                      ? `$${Number(r.usd_per_hh_low).toFixed(2)} – <strong>$${Number(r.usd_per_hh_central).toFixed(2)}</strong> – $${Number(r.usd_per_hh_high).toFixed(2)}`
                      : `<span style="color:var(--accent-red-soft);">null (unresolved)</span>`;
                  return `
                  <tr>
                    <td><strong>${esc(r.unit)}</strong></td>
                    <td><code>${esc(r.intervention)}</code></td>
                    <td class="num-right">${fmtNum(r.households_2025_derived, 0)}</td>
                    <td class="num-right">${hhRange}</td>
                    <td class="num-right">${r.usd_millions_central !== null ? fmtUsdM(r.usd_millions_central, 1) : '<span style="color:var(--accent-red-soft);">null</span>'}</td>
                    <td class="num-right">${Number(r.effectiveness_central).toFixed(2)}</td>
                    <td class="num-right">${fmtPct(r.coverage_ceiling, 0)}</td>
                    <td class="num-right">${r.cap_P1} / ${r.cap_P2} / ${r.cap_P3}</td>
                    <td>${statusBadge(r.status)}<div style="font-size:0.73rem; color:var(--text-secondary); margin-top:0.15rem;">${esc(r.note)}</div></td>
                  </tr>
                `;
                })
                .join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderGapMatrix() {
      const mount = document.getElementById("gap-matrix-mount");
      const rows = gapMatrix.filter((r) => state.provFilter === "ALL" || String(r.real_data_status || "").toLowerCase().includes(state.provFilter));
      mount.innerHTML = `
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>Contract</th><th>Parameter</th><th>ASM</th><th>Status</th><th>Value / Range</th><th>Owner &amp; Note</th></tr></thead>
            <tbody>
              ${rows
                .map(
                  (r) => `
                <tr>
                  <td>${esc(r.contract_file)}</td>
                  <td><code>${esc(r.parameter_path)}</code></td>
                  <td><code>${esc(r.asm_id)}</code></td>
                  <td>${statusBadge(r.real_data_status)}</td>
                  <td style="font-size:0.78rem;">${esc(r.proposed_real_value_or_range)}</td>
                  <td style="font-size:0.77rem; color:var(--text-secondary);">${esc(r.blocking_reason)}</td>
                </tr>
              `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderSourceRegister() {
      const mount = document.getElementById("source-register-mount");
      const q = state.srcSearch.trim().toLowerCase();
      const rows = srcReg.filter((r) => !q || [r.param_id, r.parameter_name, r.intervention, r.source_citation, r.status, r.unit_scope].join(" ").toLowerCase().includes(q));
      function formatUrls(raw) {
        if (!raw) return "—";
        return String(raw)
          .split(";")
          .map((s) => s.trim())
          .filter(Boolean)
          .map((p) => {
            const m = p.match(/https?:\/\/[^\s)]+/);
            return m ? `<a href="${esc(m[0])}" target="_blank" rel="noopener noreferrer">${esc(m[0])}</a>` : esc(p);
          })
          .join("<br>");
      }
      mount.innerHTML = `
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>ASM</th><th>Parameter</th><th>Scope</th><th>Central Value</th><th>Status</th><th>Citation &amp; URLs</th></tr></thead>
            <tbody>
              ${rows
                .map(
                  (r) => `
                <tr>
                  <td><code>${esc(r.param_id)}</code></td>
                  <td><strong>${esc(r.parameter_name)}</strong><div style="font-size:0.73rem; color:var(--text-muted);">${esc(r.intervention)}</div></td>
                  <td><code>${esc(r.unit_scope)}</code></td>
                  <td>${r.usd_per_hh_central !== null ? `<strong>$${Number(r.usd_per_hh_central).toFixed(2)}/HH</strong>` : r.value_central !== null ? `<strong>${esc(r.value_central)}</strong>` : "—"}<div style="font-size:0.72rem; color:var(--text-muted);">${esc(r.original_currency_or_unit)}</div></td>
                  <td>${statusBadge(r.status)}</td>
                  <td style="font-size:0.78rem;"><div>${esc(r.source_citation)}</div><div style="margin-top:0.15rem;">${formatUrls(r.source_url_or_doi)}</div></td>
                </tr>
              `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    container.querySelectorAll("[data-asm-chart]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.chartView = btn.getAttribute("data-asm-chart");
        container.querySelectorAll("[data-asm-chart]").forEach((b) => b.classList.toggle("active", b === btn));
        renderMainChart();
      });
    });

    container.querySelectorAll("[data-asm-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.tableTab = btn.getAttribute("data-asm-tab");
        container.querySelectorAll("[data-asm-tab]").forEach((b) => b.classList.toggle("active", b === btn));
        renderExplorerTable();
      });
    });

    const btnReal = document.getElementById("btn-param-real");
    const btnMock = document.getElementById("btn-param-mock");
    btnReal.addEventListener("click", () => {
      state.paramView = "real_proposed";
      btnReal.classList.add("active");
      btnMock.classList.remove("active");
      renderExplorerTable();
    });
    btnMock.addEventListener("click", () => {
      state.paramView = "mock_v1";
      btnMock.classList.add("active");
      btnReal.classList.remove("active");
      renderExplorerTable();
    });

    document.getElementById("asm-unit-filter").addEventListener("change", (e) => {
      state.unitFilter = e.target.value;
      renderExplorerTable();
    });
    document.getElementById("prov-filter").addEventListener("change", (e) => {
      state.provFilter = e.target.value;
      renderGapMatrix();
    });
    document.getElementById("src-search").addEventListener("input", (e) => {
      state.srcSearch = e.target.value;
      renderSourceRegister();
    });

    renderMainChart();
    renderExplorerTable();
    renderGapMatrix();
    renderSourceRegister();
  }

  global.PageAssumptions = { render };
})(window);
