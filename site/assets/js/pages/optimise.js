/* Page 5: Multi-Objective Optimisation (NSGA-II) - V2 Visual Storytelling */
(function (global) {
  "use strict";

  async function render(container) {
    const data = await global.App.loadJSON("optimise.json");
    const { esc, term, accordion, fmtNum, fmtMillions, fmtUsdM } = global.App;

    const solutions = data.pareto_solutions || [];
    const selectedPlans = data.selected_plans || [];
    const baselines = data.baseline_comparison || [];
    const hvConv = data.hypervolume_convergence || [];
    const reevalRows = data.pareto_reeval || [];
    const sensReal = data.sensitivity_real || [];
    const opexSens = data.recurring_opex_sensitivity || [];
    const nsgaRow = baselines.find((b) => String(b.method || "").startsWith("nsga2")) || baselines[0] || {};
    const combinedHvB = Number(nsgaRow.hypervolume || 16540701192) / 1e9;
    const bauMean = Number(data.bau_2040?.dalys_2040_mean || 2750091.39);

    const planColors = {
      knee: "#DF8B6C",
      cost_minimal: "#D6C7A9",
      daly_minimal: "#68BBC2",
      equity_minimal: "#A6AAA6",
    };

    const planLabels = {
      knee: "Knee Point",
      cost_minimal: "Cost-Minimal",
      daly_minimal: "DALY-Minimal",
      equity_minimal: "Equity-Minimal",
    };

    function hasWebGL() {
      try {
        const c = document.createElement("canvas");
        return !!(c.getContext("webgl") || c.getContext("experimental-webgl"));
      } catch (_) {
        return false;
      }
    }

    const kneeSol = selectedPlans.find((p) => p.plan_name === "knee") || solutions[0];
    const webglOk = hasWebGL();
    const state = {
      paretoView: webglOk ? "3d" : "cost_daly", // "3d" | "cost_daly" | "cost_eq" | "daly_eq"
      activeSolutionId: kneeSol ? kneeSol.solution_id : 0,
      breakdownView: "heatmap", // "heatmap" | "bars"
      heatmapMode: "spend", // "spend" | "alloc"
      sweepTab: "mc_reeval", // "mc_reeval" | "sens_all" | "opex"
    };

    const mathAccordionBody = `
      <div class="grid grid-3" style="font-size:0.84rem; color:var(--text-secondary); margin-bottom:1rem;">
        <div><strong style="color:var(--text-primary);">f₁(x) · 2040 DALYs:</strong> Sum of projected 2040 female &amp; male ${term("DALYs")} across <code>IND_urban</code>, <code>IND_rural</code>, <code>NPL_urban</code>, <code>NPL_rural</code> evaluated via the Contract 3 log-linear surrogate (BAU = ${fmtMillions(bauMean, 2)}).</div>
        <div><strong style="color:var(--text-primary);">f₂(x) · Total Cost ($M):</strong> 5-year cohort package spend (Alt A 1-time programme capex) plus binary <code>infra_lpg_grid</code> capex (disabled / null in Scenario A), bounded by the <code>$${fmtNum(data.budget_cap_usd_millions, 2)}M</code> 50% fiscal cap.</div>
        <div><strong style="color:var(--text-primary);">f₃(x) · Equity Gap (Proxy):</strong> Population-weighted <code>|ASR_female - ASR_male|</code> per 100k using proxy female-to-male exposure elasticity (<code>r_fm = 1.05</code>).</div>
      </div>
      <ul class="callout-list">
        <li><strong>Decision Space (48 Variables):</strong> 4 units × 4 interventions × 3 five-year periods (<code>P1: 2026–30</code>, <code>P2: 2031–35</code>, <code>P3: 2036–40</code>).</li>
        <li><strong>Four Hard Constraints + Stage3Repair:</strong> Total budget cap ($5,985.73M), per-period adoption caps, coverage ceilings, and strictly-earlier infrastructure prerequisites (<code>q &lt; p</code>, Scenario A forces <code>infra_lpg_grid = 0</code>).</li>
        <li><strong>Multi-Seed Setup:</strong> 5 seeds (<code>42–46</code>), population <code>80</code>, <code>100</code> generations (40,000 evaluations) → <strong>${solutions.length}</strong> deduplicated non-dominated feasible plans.</li>
      </ul>
    `;

    const sweepsAndBaselinesBody = `
      <div class="tabs-bar" role="tablist">
        <button type="button" class="tab-btn active" data-sweep="mc_reeval">500-Draw Joint Uncertainty</button>
        <button type="button" class="tab-btn" data-sweep="sens_all">8-Factor Sensitivity Suite (18 Runs · Scen. B1/B2)</button>
        <button type="button" class="tab-btn" data-sweep="opex">Alt B Recurring LPG Opex</button>
      </div>
      <div id="sweep-content-mount" style="margin-bottom:1.5rem;"></div>

      <h4 style="margin:0 0 0.65rem; font-size:0.96rem;">NSGA-II vs. Baseline Heuristics &amp; 5-Seed ${term("Hypervolume")} Convergence</h4>
      <div class="grid grid-2" style="margin-bottom:1rem;">
        <div>
          <div id="chart-hv-conv" class="chart-box-compact"></div>
          <div class="chart-caption"><span>5-seed HV convergence (100 generations)</span><span>Combined HV: ${combinedHvB.toFixed(2)}B</span></div>
        </div>
        <div>
          <div id="chart-baselines" class="chart-box-compact"></div>
          <div class="chart-caption"><span>NSGA-II beats Random Search, Equal Budget &amp; Proportional Greedy</span><span>REAL_V1_2024USD</span></div>
        </div>
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Method</th>
              <th class="num-right">Evals</th>
              <th class="num-right">Pareto Pts</th>
              <th class="num-right">Hypervolume</th>
              <th class="num-right">Diff vs. NSGA-II</th>
              <th class="num-right">Min 2040 DALYs</th>
              <th class="num-right">Cost Range ($M)</th>
              <th class="num-right">Min ASR Gap</th>
            </tr>
          </thead>
          <tbody>
            ${baselines
        .map(
          (b) => `
              <tr>
                <td><strong>${esc(b.method)}</strong></td>
                <td class="num-right">${fmtNum(b.n_evaluations)}</td>
                <td class="num-right">${fmtNum(b.n_pareto_points)}</td>
                <td class="num-right">${(Number(b.hypervolume) / 1e9).toFixed(2)}B</td>
                <td class="num-right">${Number(b.hypervolume_diff_vs_nsga2_pct).toFixed(1)}%</td>
                <td class="num-right">${fmtMillions(b.min_dalys_2040_mean, 2)}</td>
                <td class="num-right">$${Number(b.min_cost_usd_millions).toFixed(1)}–$${Number(b.max_cost_usd_millions).toFixed(1)}M</td>
                <td class="num-right">${Number(b.min_equity_gap_asr).toFixed(2)}</td>
              </tr>
            `
        )
        .join("")}
          </tbody>
        </table>
      </div>
    `;

    container.innerHTML = `
      <header class="page-header">
        <div class="page-header-top">
          <span class="page-eyebrow">05 · Stage 3 Multi-Objective Resource Allocation</span>
          <div class="page-header-meta" style="margin-top:0;">
            <span class="badge badge-real">REAL DATA · GBD 2023 + REAL_V1_2024USD</span>
            <span class="badge badge-derived">${solutions.length} Pareto Solutions (Scenario A)</span>
          </div>
        </div>
        <h1 class="page-title">Multi-Objective Optimisation (NSGA-II)</h1>
        <p class="page-lead">
          Uses ${term("NSGA-II")} to uncover a ${term("Pareto front")} of clean-cooking budgets across India and Nepal (2026–2040) trading off 2040 COPD ${term("DALYs")}, total cost, and female–male equity.
        </p>
      </header>

      ${global.App.infoBanner(
      "VALIDATED & RECONCILED REAL-DATA OPTIMISATION (SCENARIO A · LOG-OLS SURROGATE)",
      "Uses REAL_V1_2024USD unit costs, 50% fiscal cap ($5,985.73M), sex-reconciled 2040 BAU = 2.70M DALYs (F+M = both), and pooled Smooth Log-OLS counterfactual surrogate (-2.1933 +/- 0.2557 SE). Scenario A disables new infra builds so unready NPL_rural relies on improved biomass; Scenario B1/B2 infra proxies unlock +59.6k more DALYs in sensitivity."
    )}

      <div class="takeaway-bar">
        <span class="takeaway-tag">Key Trade-Off</span>
        <div>
          Moving from <strong>Cost-Minimal</strong> ($70.8M, 59.4k averted) to the <strong>Knee</strong> compromise ($1,695.9M) averts <strong>311.7k DALYs</strong>. Reaching <strong>DALY-Minimal (= Equity-Minimal)</strong> ($3,097.3M) averts <strong>453.5k DALYs</strong> (16.8% reduction) and lowers the WHO GHE reconciled ASR gap from 31.41 to 24.91/100k, saturating below the $5,985.7M budget due to 2040-BAU headroom limits.
        </div>
      </div>

      <!-- Level 1: Single Large Interactive Pareto Visualisation -->
      <section class="section">
        <div class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title" id="pareto-title">3D Pareto Trade-Off Surface (${solutions.length} Feasible Plans)</h2>
              <p class="card-subtitle">Click any solution marker (or choose a representative plan below) to reveal its 4-unit allocation breakdown</p>
            </div>
            <div class="btn-group">
              ${webglOk ? `<button type="button" class="btn btn-sm ${state.paretoView === "3d" ? "active" : ""}" data-pview="3d">3D Surface</button>` : ""}
              <button type="button" class="btn btn-sm ${state.paretoView === "cost_daly" ? "active" : ""}" data-pview="cost_daly">Cost vs. DALYs</button>
              <button type="button" class="btn btn-sm ${state.paretoView === "cost_eq" ? "active" : ""}" data-pview="cost_eq">Cost vs. Equity</button>
              <button type="button" class="btn btn-sm ${state.paretoView === "daly_eq" ? "active" : ""}" data-pview="daly_eq">DALYs vs. Equity</button>
            </div>
          </div>

          <div id="chart-pareto-main" class="chart-box-tall"></div>

          <div class="chart-caption">
            <span><strong>Three Objectives (Minimise All):</strong> 2040 DALYs (count) · Total Cost (USD Millions) · Population-weighted Female–Male ${term("ASR")} Gap (per 100k, proxy).</span>
            <span><strong>Source:</strong> <code>outputs/tables/s3_pareto_front.csv</code> (5 seeds, REAL_V1_2024USD).</span>
          </div>
        </div>
      </section>

      <!-- Level 2: Four Selectable Plan Types + Allocation Breakdown -->
      <section class="section">
        <div class="section-header">
          <div>
            <h2 class="section-title">Select a Plan to Inspect Its Allocation Breakdown</h2>
            <p class="section-desc">Compare four anchor strategies from the Pareto front—or click any dot on the chart above.</p>
          </div>
        </div>

        <div class="grid grid-4" id="selected-plans-grid" style="margin-bottom:1.25rem;">
          ${selectedPlans
        .map(
          (p) => `
            <button type="button" class="plan-card ${p.solution_id === state.activeSolutionId ? "active" : ""}" data-sol-id="${p.solution_id}">
              <div class="plan-card-title">
                <span>${esc(planLabels[p.plan_name] || p.plan_name)}</span>
                <span class="mono" style="font-size:0.75rem; color:${planColors[p.plan_name] || "#928B80"};">#${p.solution_id}</span>
              </div>
              <div class="plan-metrics">
                <div><span class="plan-metric-k">2040 DALYs</span><span class="plan-metric-v">${fmtMillions(p.dalys_2040_mean, 2)}</span></div>
                <div><span class="plan-metric-k">Total Cost</span><span class="plan-metric-v">${fmtUsdM(p.total_cost_usd_millions, 1)}</span></div>
                <div><span class="plan-metric-k">DALYs Averted</span><span class="plan-metric-v">${fmtMillions(p.dalys_averted_mean, 2)}</span></div>
                <div><span class="plan-metric-k">ASR Gap (Proxy)</span><span class="plan-metric-v">${Number(p.equity_gap_asr).toFixed(1)} /100k</span></div>
              </div>
            </button>
          `
        )
        .join("")}
        </div>

        <div id="solution-inspector-summary" class="grid grid-4" style="margin-bottom:1rem;"></div>

        <div class="card">
          <div class="card-header">
            <div>
              <h3 class="card-title" id="breakdown-title">Allocation Matrix</h3>
              <p class="card-subtitle" id="breakdown-sub">4 Geographical Units × 4 Interventions × 3 Periods (P1: 2026–30, P2: 2031–35, P3: 2036–40)</p>
            </div>
            <div class="btn-group">
              <button type="button" class="btn btn-sm active" id="btn-bd-heatmap">Allocation Heatmap</button>
              <button type="button" class="btn btn-sm" id="btn-bd-bars">Spend Bars ($M)</button>
              <button type="button" class="btn btn-sm btn-teal active" id="btn-hm-spend">Units: $M</button>
              <button type="button" class="btn btn-sm btn-teal" id="btn-hm-alloc">Units: x ∈ [0, 1]</button>
            </div>
          </div>
          <div id="chart-alloc-main" class="chart-box"></div>
          <div class="chart-caption" id="breakdown-cap"></div>
        </div>
      </section>

      <!-- Level 3: Expandable Technical Details -->
      ${accordion(
          "Uncertainty Re-Evaluation (500 Draws), 8-Factor Sensitivity &amp; Baseline Benchmarks",
          '<span class="badge badge-real">18 Real Scenarios + 500 MC Draws</span>',
          sweepsAndBaselinesBody
        )}
      ${accordion(
          "Mathematical Formulation: Objectives, Constraints &amp; Stage3Repair",
          '<span class="badge badge-derived">48 Vars · q &lt; p</span>',
          mathAccordionBody
        )}
    `;

    function renderParetoChart() {
      const nonSelected = solutions.filter((s) => !s.selected_plans);
      const selected = selectedPlans.map((sp) => solutions.find((s) => s.solution_id === sp.solution_id) || sp);
      const titleEl = document.getElementById("pareto-title");

      if (state.paretoView === "3d") {
        titleEl.textContent = `3D Pareto Trade-Off Surface (${solutions.length} Feasible Plans)`;
        const cloud = {
          x: nonSelected.map((s) => s.total_cost_usd_millions),
          y: nonSelected.map((s) => s.dalys_2040_mean),
          z: nonSelected.map((s) => s.equity_gap_asr),
          customdata: nonSelected.map((s) => s.solution_id),
          mode: "markers",
          type: "scatter3d",
          name: `Pareto Front (${solutions.length} pts)`,
          marker: {
            size: 4,
            color: nonSelected.map((s) => s.total_cost_usd_millions),
            colorscale: [
              [0, "#68BBC2"],
              [0.5, "#D6C7A9"],
              [1, "#DF8B6C"],
            ],
            opacity: 0.86,
          },
          hovertemplate: "<b>Solution #%{customdata}</b><br>Cost: $%{x:.1f}M<br>2040 DALYs: %{y:,.0f}<br>ASR Gap: %{z:.2f}/100k<extra></extra>",
        };
        const selTraces = selected.map((s, idx) => {
          const sp = selectedPlans[idx];
          return {
            x: [s.total_cost_usd_millions],
            y: [s.dalys_2040_mean],
            z: [s.equity_gap_asr],
            customdata: [s.solution_id],
            mode: "markers",
            type: "scatter3d",
            name: planLabels[sp.plan_name] || sp.plan_name,
            marker: { size: 7, color: planColors[sp.plan_name] || "#EEEAE1", symbol: "diamond" },
            hovertemplate: `<b>${planLabels[sp.plan_name]} (#%{customdata})</b><br>Cost: $%{x:.1f}M<br>2040 DALYs: %{y:,.0f}<br>ASR Gap: %{z:.2f}/100k<extra></extra>`,
          };
        });
        global.AppCharts.plot("chart-pareto-main", [cloud, ...selTraces], {
          margin: { l: 0, r: 0, t: 10, b: 0 },
          scene: {
            xaxis: { title: "Cost ($M)", gridcolor: "rgba(185, 178, 166, 0.2)" },
            yaxis: { title: "2040 DALYs", gridcolor: "rgba(185, 178, 166, 0.2)" },
            zaxis: { title: "ASR Gap (/100k)", gridcolor: "rgba(185, 178, 166, 0.2)" },
            bgcolor: "rgba(23, 22, 19, 0)",
          },
          legend: { orientation: "h", y: -0.02 },
        });
      } else {
        const cfg = {
          cost_daly: ["total_cost_usd_millions", "dalys_2040_mean", "Total Cost (USD Millions)", "Projected 2040 DALYs (with 95% UI)", "Cost vs. 2040 DALYs Trade-Off"],
          cost_eq: ["total_cost_usd_millions", "equity_gap_asr", "Total Cost (USD Millions)", "Female–Male Equity Gap (ASR per 100k, Proxy)", "Cost vs. Sex-Equity Gap"],
          daly_eq: ["dalys_2040_mean", "equity_gap_asr", "Projected 2040 DALYs", "Female–Male Equity Gap (ASR per 100k, Proxy)", "2040 DALYs vs. Sex-Equity Gap"],
        }[state.paretoView];

        const [xKey, yKey, xTitle, yTitle, chartTitle] = cfg;
        titleEl.textContent = `${chartTitle} (${solutions.length} Feasible Plans)`;

        const cloud = {
          x: nonSelected.map((s) => s[xKey]),
          y: nonSelected.map((s) => s[yKey]),
          customdata: nonSelected.map((s) => s.solution_id),
          mode: "markers",
          type: "scatter",
          name: "Pareto Front",
          marker: { size: 7, color: "rgba(104, 187, 194, 0.52)", line: { width: 0.6, color: "#83CDD1" } },
          hovertemplate: `<b>Solution #%{customdata}</b><br>${xTitle}: %{x:,.2f}<br>${yTitle}: %{y:,.2f}<extra></extra>`,
        };

        const sel = selected.map((s, idx) => {
          const sp = selectedPlans[idx];
          const tr = {
            x: [s[xKey]],
            y: [s[yKey]],
            customdata: [s.solution_id],
            mode: "markers",
            type: "scatter",
            name: planLabels[sp.plan_name] || sp.plan_name,
            marker: {
              size: sp.plan_name === "knee" ? 14 : 11,
              color: planColors[sp.plan_name] || "#EEEAE1",
              symbol: sp.plan_name === "knee" ? "star" : "diamond",
              line: { color: "#171613", width: 1.4 },
            },
            hovertemplate: `<b>${planLabels[sp.plan_name]} (#%{customdata})</b><br>${xTitle}: %{x:,.2f}<br>${yTitle}: %{y:,.2f}<extra></extra>`,
          };
          if (yKey === "dalys_2040_mean") {
            tr.error_y = {
              type: "data",
              symmetric: false,
              array: [s.dalys_2040_upper - s.dalys_2040_mean],
              arrayminus: [s.dalys_2040_mean - s.dalys_2040_lower],
              color: planColors[sp.plan_name] || "#EEEAE1",
              thickness: 1.5,
            };
          }
          return tr;
        });

        global.AppCharts.plot("chart-pareto-main", [cloud, ...sel], {
          xaxis: { title: xTitle },
          yaxis: { title: yTitle },
          legend: { orientation: "h", y: -0.2 },
          margin: { l: 74, r: 22, t: 18, b: 84 },
        });
      }

      const domEl = document.getElementById("chart-pareto-main");
      if (domEl && domEl.on) {
        domEl.on("plotly_click", (evt) => {
          const pt = evt.points && evt.points[0];
          if (pt && pt.customdata !== undefined) {
            selectSolution(Number(pt.customdata));
          }
        });
      }
    }

    function selectSolution(solId) {
      state.activeSolutionId = solId;
      document.querySelectorAll("#selected-plans-grid .plan-card").forEach((card) => {
        card.classList.toggle("active", Number(card.getAttribute("data-sol-id")) === solId);
      });
      renderBreakdown();
    }

    function renderBreakdown() {
      const sol = solutions.find((s) => s.solution_id === state.activeSolutionId) || solutions[0];
      if (!sol) return;

      const summaryEl = document.getElementById("solution-inspector-summary");
      const planTag = sol.selected_plans ? ` (${sol.selected_plans})` : "";
      summaryEl.innerHTML = `
        <div class="kpi-card">
          <div class="kpi-label">Selected Solution</div>
          <div class="kpi-value">#${sol.solution_id}${esc(planTag)}</div>
          <div class="kpi-detail">Seed ${sol.seed} · Feasible: <strong>${sol.is_feasible ? "True (0 violations)" : "False"}</strong></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Total Cost</div>
          <div class="kpi-value">${fmtUsdM(sol.total_cost_usd_millions, 2)}</div>
          <div class="kpi-detail">${((sol.total_cost_usd_millions / data.budget_cap_usd_millions) * 100).toFixed(1)}% of $${fmtNum(data.budget_cap_usd_millions, 1)}M 50% cap</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">2040 DALYs [95% UI]</div>
          <div class="kpi-value">${fmtMillions(sol.dalys_2040_mean, 2)}</div>
          <div class="kpi-detail">[${fmtMillions(sol.dalys_2040_lower, 2)}, ${fmtMillions(sol.dalys_2040_upper, 2)}] · Averted ${fmtMillions(sol.dalys_averted_mean, 2)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Equity Gap (Proxy)</div>
          <div class="kpi-value">${Number(sol.equity_gap_asr).toFixed(2)} /100k</div>
          <div class="kpi-detail">Abs DALY diff: ${fmtNum(sol.equity_gap_dalys_abs, 0)}</div>
        </div>
      `;

      const units = data.units || ["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"];
      const ivs = data.interventions || ["lpg_subsidy", "improved_biomass", "electric_induction", "infra_lpg_grid"];
      const periods = data.periods || ["P1", "P2", "P3"];
      const titleEl = document.getElementById("breakdown-title");
      const capEl = document.getElementById("breakdown-cap");

      if (state.breakdownView === "heatmap") {
        const shortIv = { lpg_subsidy: "LPG", improved_biomass: "Bio", electric_induction: "Elec", infra_lpg_grid: "Infra" };
        const xLabels = [];
        for (const iv of ivs) {
          for (const p of periods) xLabels.push(`${shortIv[iv]}·${p}`);
        }
        const isSpend = state.heatmapMode === "spend";
        const zMatrix = units.map((u) => {
          const row = [];
          for (const iv of ivs) {
            for (const p of periods) {
              const cell = (sol.allocations || []).find((a) => a.unit === u && a.intervention === iv && a.period === p);
              row.push(cell ? (isSpend ? cell.cost_usd_millions : cell.allocation_value) : 0);
            }
          }
          return row;
        });

        titleEl.textContent = isSpend
          ? `Solution #${sol.solution_id} — Allocation Spend Heatmap ($M)`
          : `Solution #${sol.solution_id} — Decision Variable Heatmap (x ∈ [0, 1])`;
        capEl.innerHTML = isSpend
          ? "<span><strong>Unit:</strong> USD Millions per (unit, intervention, period) cell (REAL_V1_2024USD).</span><span><strong>Source:</strong> Decoded via <code>decode_allocation_spend()</code>.</span>"
          : "<span><strong>Unit:</strong> Decision variable x (coverage increment; infra_lpg_grid disabled in Scenario A).</span><span><strong>Source:</strong> <code>s3_pareto_front.csv</code>.</span>";

        global.AppCharts.plot(
          "chart-alloc-main",
          [
            {
              x: xLabels,
              y: units,
              z: zMatrix,
              type: "heatmap",
              colorscale: [
                [0, "#1D1B18"],
                [0.35, "#2F4A4B"],
                [0.7, "#4D8A90"],
                [1, "#83CDD1"],
              ],
              reversescale: false,
              hovertemplate: isSpend ? "<b>%{y} · %{x}</b><br>Spend: $%{z:.2f}M<extra></extra>" : "<b>%{y} · %{x}</b><br>Allocation x: %{z:.3f}<extra></extra>",
            },
          ],
          {
            xaxis: { title: "Intervention × Period", tickangle: -25 },
            yaxis: { title: "Allocation Unit", autorange: "reversed" },
            margin: { l: 92, r: 24, t: 18, b: 84 },
          }
        );
      } else {
        const upLabels = [];
        for (const u of units) {
          for (const p of periods) upLabels.push(`${u}·${p}`);
        }
        const ivColors = { lpg_subsidy: "#83CDD1", improved_biomass: "#D6C7A9", electric_induction: "#68BBC2", infra_lpg_grid: "#DF8B6C" };
        const barTraces = ivs.map((iv) => ({
          x: upLabels,
          y: upLabels.map((lbl) => {
            const [u, p] = lbl.split("·");
            const cell = (sol.allocations || []).find((a) => a.unit === u && a.intervention === iv && a.period === p);
            return cell ? cell.cost_usd_millions : 0;
          }),
          name: iv,
          type: "bar",
          marker: { color: ivColors[iv] },
          hovertemplate: `<b>%{x} · ${iv}</b><br>Spend: $%{y:.2f}M<extra></extra>`,
        }));

        titleEl.textContent = `Solution #${sol.solution_id} — Stacked Spend by Unit & Period ($M)`;
        capEl.innerHTML = "<span><strong>Unit:</strong> USD Millions (REAL_V1_2024USD).</span><span><strong>Scenario A:</strong> infra_lpg_grid fixed to 0.0 (unverified capex).</span>";

        global.AppCharts.plot("chart-alloc-main", barTraces, {
          barmode: "stack",
          xaxis: { title: "Unit × Period", tickangle: -30 },
          yaxis: { title: "Spend (USD Millions)" },
          legend: { orientation: "h", y: -0.26 },
          margin: { l: 64, r: 18, t: 18, b: 96 },
        });
      }
    }

    function renderSweepTab() {
      const mount = document.getElementById("sweep-content-mount");
      if (state.sweepTab === "mc_reeval") {
        mount.innerHTML = `
          <div class="table-wrap" style="margin-bottom:1rem;">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Plan / Policy</th>
                  <th class="num-right">Cost ($M)</th>
                  <th class="num-right">Analytical DALYs</th>
                  <th class="num-right">500-Draw Stage 2 Mean [2.5%–97.5%]</th>
                  <th class="num-right">500-Draw Joint Mean [2.5%–97.5%]</th>
                  <th class="num-right">Analytical Eq Gap</th>
                  <th class="num-right">Joint Eq Mean</th>
                </tr>
              </thead>
              <tbody>
                ${reevalRows
                  .map(
                    (r) => `
                  <tr>
                    <td><code>${esc(r.plan_name)}</code></td>
                    <td class="num-right">${fmtUsdM(r.total_cost_usd_millions, 1)}</td>
                    <td class="num-right">${fmtMillions(r.analytical_dalys_mean, 3)}</td>
                    <td class="num-right">${fmtMillions(r.stage2_draws_dalys_mean, 3)} [${fmtMillions(r.stage2_draws_dalys_p025, 2)}–${fmtMillions(r.stage2_draws_dalys_p975, 2)}]</td>
                    <td class="num-right"><strong>${fmtMillions(r.joint_draws_dalys_mean, 3)}</strong> [${fmtMillions(r.joint_draws_dalys_p025, 2)}–${fmtMillions(r.joint_draws_dalys_p975, 2)}]</td>
                    <td class="num-right">${Number(r.analytical_equity_gap_asr).toFixed(2)}</td>
                    <td class="num-right">${Number(r.joint_draws_equity_gap_mean).toFixed(2)}</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
          <div>
            <img src="assets/images/real/s3_pareto_uncertainty_reeval.png" alt="500-Draw Joint Monte Carlo Uncertainty Re-Evaluation" style="width:100%; max-width:920px; border-radius:8px; border:1px solid var(--border-subtle);" />
          </div>
        `;
      } else if (state.sweepTab === "sens_all") {
        mount.innerHTML = `
          <div class="table-wrap" style="margin-bottom:1rem;">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Factor</th>
                  <th>Scenario</th>
                  <th class="num-right">Budget ($M)</th>
                  <th class="num-right">Bridge</th>
                  <th class="num-right">η_bio</th>
                  <th class="num-right">Knee Averted</th>
                  <th class="num-right">Knee Cost ($M)</th>
                  <th class="num-right">DALY-Min Averted</th>
                  <th class="num-right">DALY-Min Cost ($M)</th>
                  <th class="num-right">MDP Averted</th>
                </tr>
              </thead>
              <tbody>
                ${sensReal
                  .map(
                    (r) => `
                  <tr>
                    <td><code>${esc(r.factor_group)}</code></td>
                    <td><strong>${esc(r.scenario_label)}</strong></td>
                    <td class="num-right">${fmtUsdM(r.budget_usd_millions, 1)}</td>
                    <td class="num-right">${Number(r.bridge_factor).toFixed(1)}</td>
                    <td class="num-right">${Number(r.eta_biomass).toFixed(2)}</td>
                    <td class="num-right">${fmtNum(r.ga_knee_dalys_averted, 0)}</td>
                    <td class="num-right">${fmtUsdM(r.ga_knee_cost_usd_m, 1)}</td>
                    <td class="num-right"><strong>${fmtNum(r.ga_daly_min_dalys_averted, 0)}</strong></td>
                    <td class="num-right">${fmtUsdM(r.ga_daly_min_cost_usd_m, 1)}</td>
                    <td class="num-right">${fmtNum(r.mdp_mean_dalys_averted, 0)}</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
          <div>
            <img src="assets/images/real/plan_sensitivity_tornado.png" alt="8-Factor Sensitivity Tornado Chart" style="width:100%; max-width:920px; border-radius:8px; border:1px solid var(--border-subtle);" />
          </div>
        `;
      } else {
        mount.innerHTML = `
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Selected Plan</th>
                  <th class="num-right">2040 DALYs</th>
                  <th class="num-right">Alt A One-Time ($M)</th>
                  <th class="num-right">Alt B Recurring LPG ($M)</th>
                  <th class="num-right">Extra LPG Opex ($M)</th>
                  <th class="num-right">Alt B / Alt A Ratio</th>
                  <th>Fits $5,985.7M Cap?</th>
                </tr>
              </thead>
              <tbody>
                ${opexSens
                  .map(
                    (r) => `
                  <tr>
                    <td><code>${esc(r.plan_name)}</code></td>
                    <td class="num-right">${fmtMillions(r.dalys_2040_mean, 3)}</td>
                    <td class="num-right">${fmtUsdM(r.alt_a_one_time_cost_usd_m, 1)}</td>
                    <td class="num-right"><strong>${fmtUsdM(r.alt_b_recurring_lpg_cost_usd_m, 1)}</strong></td>
                    <td class="num-right">+${fmtUsdM(r.alt_b_extra_lpg_opex_usd_m, 1)}</td>
                    <td class="num-right">${Number(r.alt_b_cost_ratio_vs_alt_a).toFixed(2)}×</td>
                    <td><span class="badge ${r.fits_base_budget_under_alt_b ? "badge-real" : "badge-mock"}">${r.fits_base_budget_under_alt_b ? "Yes" : "No"}</span></td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        `;
      }
    }

    function renderAccordionCharts() {
      if (hvConv.length) {
        const seedColors = ["#68BBC2", "#83CDD1", "#D6C7A9", "#DF8B6C", "#A6AAA6"];
        global.AppCharts.plot(
          "chart-hv-conv",
          hvConv.map((s, idx) => ({
            x: s.history.map((h) => h.generation ?? h.gen),
            y: s.history.map((h) => h.hypervolume / 1e9),
            name: `Seed ${s.seed}`,
            mode: "lines",
            line: { width: 2, color: seedColors[idx % seedColors.length] },
          })),
          { xaxis: { title: "Generation (1–100)" }, yaxis: { title: "Hypervolume (Billions)" }, legend: { orientation: "h", y: -0.25 }, margin: { l: 58, r: 16, t: 16, b: 78 } }
        );
      }

      global.AppCharts.plot(
        "chart-baselines",
        [
          {
            x: baselines.map((b) => b.method),
            y: baselines.map((b) => b.hypervolume / 1e9),
            type: "bar",
            marker: { color: baselines.map((b) => (b.method.startsWith("nsga2") ? "#68BBC2" : "#928B80")) },
          },
        ],
        { xaxis: { title: "Method", tickangle: -18 }, yaxis: { title: "Hypervolume (Billions)" }, margin: { l: 58, r: 16, t: 16, b: 86 } }
      );
    }

    container.querySelectorAll("[data-pview]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.paretoView = btn.getAttribute("data-pview");
        container.querySelectorAll("[data-pview]").forEach((b) => b.classList.toggle("active", b === btn));
        renderParetoChart();
      });
    });

    document.querySelectorAll("#selected-plans-grid .plan-card").forEach((btn) => {
      btn.addEventListener("click", () => selectSolution(Number(btn.getAttribute("data-sol-id"))));
    });

    const bHm = document.getElementById("btn-bd-heatmap");
    const bBars = document.getElementById("btn-bd-bars");
    const bSpend = document.getElementById("btn-hm-spend");
    const bAlloc = document.getElementById("btn-hm-alloc");

    bHm.addEventListener("click", () => {
      state.breakdownView = "heatmap";
      bHm.classList.add("active");
      bBars.classList.remove("active");
      bSpend.style.display = "inline-flex";
      bAlloc.style.display = "inline-flex";
      renderBreakdown();
    });
    bBars.addEventListener("click", () => {
      state.breakdownView = "bars";
      bBars.classList.add("active");
      bHm.classList.remove("active");
      bSpend.style.display = "none";
      bAlloc.style.display = "none";
      renderBreakdown();
    });
    bSpend.addEventListener("click", () => {
      state.heatmapMode = "spend";
      bSpend.classList.add("active");
      bAlloc.classList.remove("active");
      renderBreakdown();
    });
    bAlloc.addEventListener("click", () => {
      state.heatmapMode = "alloc";
      bAlloc.classList.add("active");
      bSpend.classList.remove("active");
      renderBreakdown();
    });

    container.querySelectorAll("[data-sweep]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.sweepTab = btn.getAttribute("data-sweep");
        container.querySelectorAll("[data-sweep]").forEach((b) => b.classList.toggle("active", b === btn));
        renderSweepTab();
      });
    });

    renderParetoChart();
    renderBreakdown();
    renderSweepTab();
    renderAccordionCharts();
  }

  global.PageOptimise = { render };
})(window);
