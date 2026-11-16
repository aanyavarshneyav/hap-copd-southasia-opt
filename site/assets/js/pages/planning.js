/* Page 6: Planning & Decision Intelligence (V2 Visual Storytelling) */
(function (global) {
  "use strict";

  async function render(container) {
    const data = await global.App.loadJSON("planning_mock.json");
    const { esc, term, accordion, fmtNum, fmtMillions, fmtUsdM } = global.App;

    const mdpCfg = data.mdp_config || {};
    const policyRows = data.mdp_policy_rows || [];
    const rollouts = data.mdp_rollouts || [];
    const mdpSens = data.mdp_sensitivity || [];
    const htn = data.htn_trace || {};
    const pddl = data.pddl || {};
    const comp = data.comparison || [];

    const state = {
      activeTab: "mdp", // "mdp" | "htn" | "pddl"
      compChartMode: "tradeoff", // "tradeoff" | "violations"
      mdpChartMode: "rollouts", // "rollouts" | "sensitivity"
      policyPeriod: "ALL",
      policyBudget: "ALL",
      policyQuery: "",
      htnExpanded: new Set(["root", "0", "1"]),
    };

    const fullCompTableBody = `
      <p style="font-size:0.84rem; color:var(--text-secondary); margin:0 0 0.75rem;">
        Evaluated across <code>N = 1,000</code> Common-Random-Numbers uptake shocks (<code>[0.5×, 1.0×, 1.25×]</code> with probabilities <code>[0.15, 0.70, 0.15]</code>) at <code>λ = 3,500</code> and <code>γ = 0.774</code>. Note: HTN is evaluated deterministically and is not included in the stochastic rollout table.
      </p>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Paradigm</th>
              <th class="num-right">Mean 2040 DALYs</th>
              <th class="num-right">SD DALYs</th>
              <th class="num-right">Mean Cost ($M)</th>
              <th class="num-right">ASR Gap</th>
              <th class="num-right">Scalarized Return</th>
              <th class="num-right">Violation Rate</th>
            </tr>
          </thead>
          <tbody>
            ${comp
        .map(
          (r) => `
              <tr>
                <td><strong>${esc(r.method)}</strong></td>
                <td>${esc(r.paradigm)}</td>
                <td class="num-right">${fmtMillions(r.mean_dalys_2040, 2)}</td>
                <td class="num-right">${fmtNum(r.sd_dalys_2040, 0)}</td>
                <td class="num-right">${fmtUsdM(r.mean_cost_usd_millions, 2)}</td>
                <td class="num-right">${Number(r.mean_equity_gap_asr).toFixed(2)}</td>
                <td class="num-right">${fmtNum(r.mean_scalarized_value, 0)}</td>
                <td class="num-right"><span class="badge ${Number(r.constraint_violation_rate) === 0 ? "badge-real" : "badge-mock"}">${(Number(r.constraint_violation_rate) * 100).toFixed(1)}%</span></td>
              </tr>
            `
        )
        .join("")}
            <tr>
              <td><strong>htn_forward_decomposition</strong></td>
              <td>HTN (deterministic)</td>
              <td class="num-right">${fmtMillions(htn.summary?.dalys_2040_mean, 2)}</td>
              <td class="num-right"><em>Unavailable</em></td>
              <td class="num-right">${fmtUsdM(htn.summary?.total_cost_usd_millions, 2)}</td>
              <td class="num-right">${Number(htn.summary?.equity_gap_asr || 0).toFixed(2)}</td>
              <td class="num-right"><em>Unavailable</em></td>
              <td class="num-right"><span class="badge badge-derived">Deterministic Feasible</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    `;

    container.innerHTML = `
      <header class="page-header">
        <div class="page-header-top">
          <span class="page-eyebrow">06 · Stage 4 Sequential &amp; Symbolic Planning</span>
          <div class="page-header-meta" style="margin-top:0;">
            <span class="badge badge-mock">SYNTHETIC DATA — MOCK_V1</span>
            <span class="badge badge-derived">MDP · HTN · STRIPS/PDDL</span>
          </div>
        </div>
        <h1 class="page-title">Planning &amp; Decision Intelligence</h1>
        <p class="page-lead">
          Compares adaptive sequential planning (${term("MDP")}), hierarchical task decomposition (${term("HTN")}), and classical logic planning (${term("PDDL")}) against static NSGA-II schedules.
        </p>
      </header>

      ${global.App.mockBanner(
      "SYNTHETIC DATA — DEMONSTRATION ONLY (MOCK_V1)",
      "MDP policies, 1,000-rollout distributions, HTN traces, PDDL cost checks, and method comparisons run on synthetic MOCK_V1 contracts."
    )}

      <div class="takeaway-bar">
        <span class="takeaway-tag">Key Finding</span>
        <div>
          On synthetic <code>MOCK_V1</code>, the adaptive <strong>MDP achieves 0.0% constraint violations</strong> across 1,000 uncertain-uptake rollouts ($66.0M spend, adapting 14.1% of P2 paths on low P1 uptake), whereas open-loop GA schedules achieve higher mean scalarized return (572k–621k vs. 390k) at the cost of <strong>30.9%–36.5% violations</strong> under 1.25× high-uptake shocks.
        </div>
      </div>

      <!-- Level 1: Concise Method Comparison Visual -->
      <section class="section">
        <div class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title" id="comp-title">Equal-Footing Comparison: Cost vs. Expected 2040 DALYs (1,000 Rollouts)</h2>
              <p class="card-subtitle">Compares static GA plans, adaptive MDP policies, and the satisficing PDDL plan under identical uptake shocks</p>
            </div>
            <div class="btn-group">
              <button type="button" class="btn btn-sm active" data-comp-view="tradeoff">Cost vs. DALYs (5th–95th UI)</button>
              <button type="button" class="btn btn-sm" data-comp-view="violations">Constraint Violation Rate (%)</button>
            </div>
          </div>
          <div id="chart-comp-main" class="chart-box"></div>
          <div class="chart-caption">
            <span><strong>Units:</strong> Cost in USD Millions; 2040 DALYs in Millions (error bars show 5th–95th rollout percentiles). HTN is deterministic ($395.6M, 3.45M DALYs) and excluded from stochastic rollouts.</span>
            <span><strong>Source:</strong> <code>comp_ga_vs_mdp_mock.csv</code> (MOCK_V1).</span>
          </div>
        </div>
      </section>

      <!-- Level 2: Tabbed Paradigm Explorer (MDP | HTN | STRIPS/PDDL) -->
      <section class="section">
        <div class="section-header">
          <div>
            <h2 class="section-title">Explore Each Planning Paradigm</h2>
            <p class="section-desc">Switch between the three formal planning implementations to inspect states, trees, and verified action sequences.</p>
          </div>
        </div>

        <div class="tabs-bar" role="tablist" aria-label="Planning paradigms">
          <button type="button" class="tab-btn active" role="tab" aria-selected="true" data-plan-tab="mdp">
            1. Markov Decision Process (MDP)
          </button>
          <button type="button" class="tab-btn" role="tab" aria-selected="false" data-plan-tab="htn">
            2. Hierarchical Task Network (HTN)
          </button>
          <button type="button" class="tab-btn" role="tab" aria-selected="false" data-plan-tab="pddl">
            3. STRIPS / PDDL (pyperplan)
          </button>
        </div>

        <div id="planning-tab-panel"></div>
      </section>

      <!-- Level 3: Full Comparison Table Accordion -->
      ${accordion(
      "Full Equal-Footing Comparison Table (All 12 Stochastic Strategies + HTN)",
      '<span class="badge badge-mock">N=1,000 CRN Rollouts</span>',
      fullCompTableBody
    )}
    `;

    function renderComparisonChart() {
      const titleEl = document.getElementById("comp-title");

      if (state.compChartMode === "tradeoff") {
        titleEl.textContent = "Equal-Footing Comparison: Cost vs. Expected 2040 DALYs (1,000 Rollouts)";

        // Deduplicate exact overlapping coordinates for clean visual labels
        const deduped = [];
        const seenCoords = new Map();
        for (const r of comp) {
          const key = `${Number(r.mean_cost_usd_millions).toFixed(2)}|${Number(r.mean_dalys_2040).toFixed(0)}`;
          if (seenCoords.has(key)) {
            const prev = seenCoords.get(key);
            if (!prev.label.includes("MDP") && r.method.includes("mdp")) {
              prev.label = r.method;
            } else if (prev.method === "mdp_adaptive_b6" && r.method === "mdp_adaptive_b11") {
              prev.label = "MDP adaptive (b6 & b11)";
            } else if (prev.method === "ga_daly_minimal" && r.method === "ga_equity_minimal") {
              prev.label = "GA DALY & Equity min";
            }
            continue;
          }
          const copy = { ...r, label: r.method };
          seenCoords.set(key, copy);
          deduped.push(copy);
        }

        const groups = [
          {
            name: "MDP (Adaptive)",
            color: "#68BBC2",
            symbol: "diamond",
            size: 13,
            rows: deduped.filter((r) => String(r.paradigm).includes("MDP")),
            textpos: "top right",
          },
          {
            name: "NSGA-II GA (Open-Loop)",
            color: "#D6C7A9",
            symbol: "circle",
            size: 11,
            rows: deduped.filter((r) => String(r.paradigm).includes("GA")),
            textpos: null,
          },
          {
            name: "STRIPS / PDDL (Classical)",
            color: "#A6AAA6",
            symbol: "square",
            size: 12,
            rows: deduped.filter((r) => String(r.paradigm).includes("PDDL")),
            textpos: "top right",
          },
        ];

        const gaPosMap = {
          ga_cost_minimal: "bottom right",
          "ga_pareto_s1 (id=33)": "top right",
          "ga_pareto_s2 (id=84)": "bottom left",
          "ga_matched_best (id=116)": "bottom left",
          ga_knee: "top right",
          "ga_pareto_s3 (id=134)": "bottom center",
          "ga_pareto_s4 (id=185)": "top right",
          ga_daly_minimal: "top left",
        };

        const traces = groups.map((g) => ({
          x: g.rows.map((r) => r.mean_cost_usd_millions),
          y: g.rows.map((r) => r.mean_dalys_2040 / 1e6),
          text: g.rows.map((r) => r.label),
          mode: "markers+text",
          name: g.name,
          textposition: g.textpos || g.rows.map((r) => gaPosMap[r.method] || "top center"),
          textfont: { size: 10, color: "#B9B2A6" },
          type: "scatter",
          marker: { size: g.size, color: g.color, symbol: g.symbol, line: { color: "#171613", width: 1.4 } },
          error_y: {
            type: "data",
            symmetric: false,
            array: g.rows.map((r) => (r.p95_dalys_2040 - r.mean_dalys_2040) / 1e6),
            arrayminus: g.rows.map((r) => (r.mean_dalys_2040 - r.p05_dalys_2040) / 1e6),
            color: "rgba(185, 178, 166, 0.45)",
            thickness: 1.4,
          },
          hovertemplate: "<b>%{text}</b><br>Mean Cost: $%{x:.1f}M<br>Mean 2040 DALYs: %{y:.2f}M<extra></extra>",
        }));

        global.AppCharts.plot("chart-comp-main", traces, {
          xaxis: { title: "Mean Planned Cost (USD Millions)", range: [-15, 430] },
          yaxis: { title: "Mean 2040 DALYs (Millions, 5th–95th pctile)" },
          legend: { orientation: "h", y: -0.2 },
          margin: { l: 64, r: 28, t: 24, b: 84 },
        });
      } else {
        titleEl.textContent = "Constraint Violation Rate Across 1,000 Uptake Shocks (%)";
        global.AppCharts.plot(
          "chart-comp-main",
          [
            {
              x: comp.map((r) => r.method),
              y: comp.map((r) => r.constraint_violation_rate * 100),
              type: "bar",
              marker: { color: comp.map((r) => (r.constraint_violation_rate === 0 ? "#68BBC2" : "#DF8B6C")) },
              hovertemplate: "<b>%{x}</b><br>Violation Rate: %{y:.1f}%<extra></extra>",
            },
          ],
          {
            xaxis: { title: "Evaluated Strategy", tickangle: -22 },
            yaxis: { title: "Rollouts Violating Cap/Ceiling (%)", range: [0, 44] },
            margin: { l: 60, r: 20, t: 18, b: 105 },
          }
        );
      }
    }

    function renderMdpTab(panel) {
      const policyExplorerBody = `
        <div class="controls-bar">
          <div class="control-group">
            <label for="mdp-period-sel">Period</label>
            <select id="mdp-period-sel" class="control-select">
              <option value="ALL">All Periods (P1–P3)</option>
              <option value="P1">P1 (2026–2030)</option>
              <option value="P2">P2 (2031–2035)</option>
              <option value="P3">P3 (2036–2040)</option>
            </select>
          </div>
          <div class="control-group">
            <label for="mdp-budget-sel">Budget Bin</label>
            <select id="mdp-budget-sel" class="control-select">
              <option value="ALL">All Tiers (0–5)</option>
              <option value="5">Bin 5 ($500M)</option>
              <option value="4">Bin 4 ($400M)</option>
              <option value="3">Bin 3 ($300M)</option>
              <option value="2">Bin 2 ($200M)</option>
              <option value="1">Bin 1 ($100M)</option>
              <option value="0">Bin 0 ($0M)</option>
            </select>
          </div>
          <div class="control-group" style="flex:1; max-width:340px;">
            <label for="mdp-search">Search Optimal Action</label>
            <input id="mdp-search" class="control-input" type="search" placeholder="e.g. infra, lpg, bio, IND_rural…" />
          </div>
        </div>
        <div id="mdp-policy-table-mount"></div>
      `;

      const formalTupleBody = `
        <ul class="callout-list">
          <li><strong>States (2,277 active):</strong> <code>(t, clean_bins[4], bio_bins[4], infra_flags[2], budget_bin)</code> tracks raw clean-fuel tiers separately from biomass (η = 0.45) exposure gains.</li>
          <li><strong>Actions (85 portfolios):</strong> Multi-unit bundles of <code>none</code>, <code>infra</code>, <code>lpg(+1/+2)</code>, <code>bio(+1)</code>, <code>elec(+1)</code>; rural LPG/induction blocked unless infra was built in a strictly earlier period (<code>q &lt; p</code>).</li>
          <li><strong>Transitions:</strong> Stochastic uptake multipliers <code>[0.5×, 1.0×, 1.25×]</code> with probabilities <code>[0.15, 0.70, 0.15]</code> (<code>ASM-15</code>); deterministic infra and budget transitions.</li>
          <li><strong>Solver:</strong> Vectorized NumPy finite-horizon backward induction (<code>P1..P3</code>) + infinite-horizon VI diagnostics (<code>tol=1e-6</code>).</li>
        </ul>
      `;

      panel.innerHTML = `
        <div class="grid grid-2" style="margin-bottom:1.1rem;">
          <div class="card">
            <div class="card-header">
              <div>
                <h3 class="card-title">How the Adaptive MDP Works</h3>
                <p class="card-subtitle">Reward = ΔDALYs Averted − λ_cost · Cost − λ_eq · ΔASR_Gap (λ = ${fmtNum(mdpCfg.cost_weight_lambda)}, γ = ${mdpCfg.discount_factor})</p>
              </div>
              <span class="badge badge-derived">2,277 States · 85 Actions</span>
            </div>
            <svg viewBox="0 0 580 250" style="width:100%; height:auto; display:block;" role="img" aria-label="MDP stochastic uptake and adaptive transition diagram">
              <defs>
                <marker id="arrowBlue2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#D6C7A9"></path></marker>
                <marker id="arrowTeal2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#68BBC2"></path></marker>
                <marker id="arrowAmber2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#DF8B6C"></path></marker>
              </defs>
              <rect x="12" y="88" width="112" height="72" rx="4" fill="#24221E" stroke="#68BBC2" stroke-width="1.4"/>
              <text x="68" y="114" text-anchor="middle" fill="#EEEAE1" font-size="12" font-weight="700">P1 (2026–30)</text>
              <text x="68" y="132" text-anchor="middle" fill="#B9B2A6" font-size="10">Initial State s₀</text>
              <text x="68" y="148" text-anchor="middle" fill="#68BBC2" font-size="9.5">Budget $500M</text>

              <rect x="164" y="18" width="144" height="50" rx="4" fill="#33251F" stroke="#DF8B6C" stroke-width="1.2"/>
              <text x="236" y="39" text-anchor="middle" fill="#DF8B6C" font-size="11" font-weight="600">Low Uptake (p=0.15)</text>
              <text x="236" y="55" text-anchor="middle" fill="#B9B2A6" font-size="10">0.5× step → P2 Recourse</text>

              <rect x="164" y="99" width="144" height="50" rx="4" fill="#24221E" stroke="#68BBC2" stroke-width="1.4"/>
              <text x="236" y="120" text-anchor="middle" fill="#68BBC2" font-size="11" font-weight="600">Nominal (p=0.70)</text>
              <text x="236" y="136" text-anchor="middle" fill="#EEEAE1" font-size="10">1.0× Expected Step</text>

              <rect x="164" y="180" width="144" height="50" rx="4" fill="#24221E" stroke="#D6C7A9" stroke-width="1.2"/>
              <text x="236" y="201" text-anchor="middle" fill="#D6C7A9" font-size="11" font-weight="600">High Uptake (p=0.15)</text>
              <text x="236" y="217" text-anchor="middle" fill="#B9B2A6" font-size="10">1.25× step (capped)</text>

              <path d="M124,110 L160,44" stroke="#DF8B6C" stroke-width="1.4" fill="none" marker-end="url(#arrowAmber2)"/>
              <path d="M124,124 L160,124" stroke="#68BBC2" stroke-width="1.6" fill="none" marker-end="url(#arrowTeal2)"/>
              <path d="M124,138 L160,204" stroke="#D6C7A9" stroke-width="1.4" fill="none" marker-end="url(#arrowBlue2)"/>

              <rect x="348" y="46" width="96" height="56" rx="4" fill="#24221E" stroke="#928B80" stroke-width="1.3"/>
              <text x="396" y="70" text-anchor="middle" fill="#EEEAE1" font-size="11.5" font-weight="700">P2 (2031–35)</text>
              <text x="396" y="88" text-anchor="middle" fill="#B9B2A6" font-size="9.5">q&lt;p unlocked</text>

              <rect x="348" y="148" width="96" height="56" rx="4" fill="#24221E" stroke="#928B80" stroke-width="1.3"/>
              <text x="396" y="172" text-anchor="middle" fill="#EEEAE1" font-size="11.5" font-weight="700">P3 (2036–40)</text>
              <text x="396" y="190" text-anchor="middle" fill="#B9B2A6" font-size="9.5">Final cohort</text>

              <rect x="476" y="92" width="92" height="66" rx="4" fill="rgba(104, 187, 194, 0.12)" stroke="#68BBC2" stroke-width="1.4"/>
              <text x="522" y="119" text-anchor="middle" fill="#68BBC2" font-size="11.5" font-weight="700">2040 Outcome</text>
              <text x="522" y="138" text-anchor="middle" fill="#EEEAE1" font-size="9.5">DALYs &amp; Cost</text>

              <path d="M308,44 L344,70" stroke="#A6AAA6" stroke-width="1.2" fill="none" marker-end="url(#arrowBlue2)"/>
              <path d="M308,124 L344,84" stroke="#68BBC2" stroke-width="1.4" fill="none" marker-end="url(#arrowTeal2)"/>
              <path d="M308,204 L344,180" stroke="#A6AAA6" stroke-width="1.2" fill="none" marker-end="url(#arrowBlue2)"/>
              <path d="M396,102 L396,144" stroke="#D6C7A9" stroke-width="1.4" fill="none" marker-end="url(#arrowBlue2)"/>
              <path d="M444,176 L472,134" stroke="#68BBC2" stroke-width="1.5" fill="none" marker-end="url(#arrowTeal2)"/>
            </svg>
            <div class="chart-caption">Unlike static GA schedules, the MDP observes realized uptake after each 5-year period before choosing the next investment.</div>
          </div>

          <div class="card">
            <div class="card-header">
              <div>
                <h3 class="card-title" id="mdp-chart-title">1,000-Rollout Outcome Distribution</h3>
                <p class="card-subtitle">Mean 4.85M DALYs (715k averted), Mean Cost $65.99M, 0.0% violations</p>
              </div>
              <div class="btn-group">
                <button type="button" class="btn btn-sm active" data-mdp-chart="rollouts">Rollout Histogram</button>
                <button type="button" class="btn btn-sm" data-mdp-chart="sensitivity">λ &amp; γ Sensitivity</button>
              </div>
            </div>
            <div id="chart-mdp-main" class="chart-box-compact"></div>
            <div class="chart-caption" id="mdp-chart-cap"></div>
          </div>
        </div>

        ${accordion(
        `MDP Optimal Policy Table Explorer (${fmtNum(policyRows.length)} Active States)`,
        '<span class="badge badge-mock">plan_mdp_policy_mock.csv</span>',
        policyExplorerBody
      )}
        ${accordion(
        "Formal MDP Tuple (S, A, P, R, γ, H) &amp; Discretization Safeguards",
        '<span class="badge badge-derived">src/planning/mdp.py</span>',
        formalTupleBody
      )}
      `;

      function paintMdpChart() {
        const titleEl = document.getElementById("mdp-chart-title");
        const capEl = document.getElementById("mdp-chart-cap");
        if (state.mdpChartMode === "rollouts") {
          titleEl.textContent = "1,000-Rollout Outcome Distribution (λ = 3,500, γ = 0.774)";
          capEl.textContent = "Units: Millions of projected 2040 DALYs across 1,000 seeded uptake shock paths.";
          global.AppCharts.plot(
            "chart-mdp-main",
            [
              {
                x: rollouts.map((r) => r.dalys_2040_mean / 1e6),
                type: "histogram",
                nbinsx: 18,
                marker: { color: "rgba(104, 187, 194, 0.78)", line: { color: "#171613", width: 1 } },
                hovertemplate: "2040 DALYs: %{x:.2f}M<br>Rollouts: %{y}<extra></extra>",
              },
            ],
            { xaxis: { title: "Projected 2040 DALYs (Millions)" }, yaxis: { title: "Rollouts (of 1,000)" }, margin: { l: 55, r: 16, t: 16, b: 56 } }
          );
        } else {
          titleEl.textContent = "MDP Sensitivity to Cost Weight λ_cost &amp; Discount γ";
          capEl.textContent = "Undiscounted valuation (γ = 1.0) values delayed clean-fuel gains fully, triggering P1 rural LPG infrastructure buildout.";
          const g774 = mdpSens.filter((r) => Number(r.discount_factor) === 0.774);
          const g100 = mdpSens.filter((r) => Number(r.discount_factor) === 1.0);
          global.AppCharts.plot(
            "chart-mdp-main",
            [
              { x: g774.map((r) => r.cost_weight_lambda), y: g774.map((r) => r.mean_dalys_averted / 1e6), name: "γ = 0.774 (discounted)", mode: "lines+markers", line: { color: "#68BBC2", width: 2.4 } },
              { x: g100.map((r) => r.cost_weight_lambda), y: g100.map((r) => r.mean_dalys_averted / 1e6), name: "γ = 1.0 (undiscounted)", mode: "lines+markers", line: { color: "#DF8B6C", width: 2.4, dash: "dash" } },
            ],
            { xaxis: { title: "Cost Weight λ_cost (DALYs/$1M)" }, yaxis: { title: "Mean DALYs Averted (Millions)" }, legend: { orientation: "h", y: -0.25 }, margin: { l: 58, r: 16, t: 16, b: 78 } }
          );
        }
      }

      function renderPolicyTable() {
        const q = state.policyQuery.trim().toLowerCase();
        const filtered = policyRows.filter((r) => {
          if (state.policyPeriod !== "ALL" && r.period !== state.policyPeriod) return false;
          if (state.policyBudget !== "ALL" && String(r.budget_bin) !== state.policyBudget) return false;
          if (q && !String(r.action_label || "").toLowerCase().includes(q)) return false;
          return true;
        });
        const shown = filtered.slice(0, 60);
        document.getElementById("mdp-policy-table-mount").innerHTML = `
          <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:0.45rem;">Showing ${shown.length} of ${fmtNum(filtered.length)} matching states.</div>
          <div class="table-wrap" style="max-height:340px; overflow-y:auto;">
            <table class="data-table">
              <thead><tr><th>Period</th><th>State</th><th>Budget</th><th>IND_u / IND_r / NPL_u / NPL_r</th><th>Optimal Action</th><th class="num-right">Cost</th><th class="num-right">V(s)</th></tr></thead>
              <tbody>
                ${shown
            .map(
              (r) => `
                  <tr>
                    <td><code>${esc(r.period)}</code></td>
                    <td class="mono">#${r.state_id}</td>
                    <td>Bin ${r.budget_bin} ($${Number(r.remaining_budget_tier_usd_millions).toFixed(0)}M)</td>
                    <td class="mono" style="font-size:0.75rem;">${(r.IND_urban__raw_cov * 100).toFixed(0)}% / ${(r.IND_rural__raw_cov * 100).toFixed(0)}% / ${(r.NPL_urban__raw_cov * 100).toFixed(0)}% / ${(r.NPL_rural__raw_cov * 100).toFixed(0)}%</td>
                    <td><strong>${esc(r.action_label)}</strong></td>
                    <td class="num-right">${fmtUsdM(r.action_cost_usd_millions, 2)}</td>
                    <td class="num-right">${fmtNum(r.state_value, 0)}</td>
                  </tr>
                `
            )
            .join("")}
              </tbody>
            </table>
          </div>
        `;
      }

      panel.querySelectorAll("[data-mdp-chart]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.mdpChartMode = btn.getAttribute("data-mdp-chart");
          panel.querySelectorAll("[data-mdp-chart]").forEach((b) => b.classList.toggle("active", b === btn));
          paintMdpChart();
        });
      });
      panel.querySelector("#mdp-period-sel").addEventListener("change", (e) => {
        state.policyPeriod = e.target.value;
        renderPolicyTable();
      });
      panel.querySelector("#mdp-budget-sel").addEventListener("change", (e) => {
        state.policyBudget = e.target.value;
        renderPolicyTable();
      });
      panel.querySelector("#mdp-search").addEventListener("input", (e) => {
        state.policyQuery = e.target.value;
        renderPolicyTable();
      });

      paintMdpChart();
      renderPolicyTable();
    }

    function renderHtnTab(panel) {
      const summary = htn.summary || {};
      const leaves = htn.leaves || [];

      const leafTableBody = `
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>#</th><th>Primitive Operator</th><th>Unit</th><th>Intervention</th><th>Period</th><th class="num-right">x</th><th class="num-right">Cost ($M)</th></tr></thead>
            <tbody>
              ${leaves
          .map(
            (lf, idx) => `
                <tr><td class="mono">${idx + 1}</td><td><code>${esc(lf.name)}</code></td><td><strong>${esc(lf.unit)}</strong></td><td><code>${esc(lf.intervention)}</code></td><td><code>${esc(lf.period)}</code></td><td class="num-right">${Number(lf.allocation_value).toFixed(2)}</td><td class="num-right">${fmtUsdM(lf.cost_usd_millions, 2)}</td></tr>
              `
          )
          .join("")}
            </tbody>
          </table>
        </div>
      `;

      panel.innerHTML = `
        <div class="grid grid-4" style="margin-bottom:1rem;">
          <div class="kpi-card"><div class="kpi-label">HTN 2040 DALYs (Mock)</div><div class="kpi-value">${fmtMillions(summary.dalys_2040_mean, 2)}</div><div class="kpi-detail">95% UI: [${fmtMillions(summary.dalys_2040_lower, 2)}, ${fmtMillions(summary.dalys_2040_upper, 2)}]</div></div>
          <div class="kpi-card"><div class="kpi-label">Total Plan Cost</div><div class="kpi-value">${fmtUsdM(summary.total_cost_usd_millions, 2)}</div><div class="kpi-detail">Feasible: <strong>${summary.is_feasible ? "True" : "False"}</strong></div></div>
          <div class="kpi-card"><div class="kpi-label">Equity Gap (ASR)</div><div class="kpi-value">${Number(summary.equity_gap_asr || 0).toFixed(2)} /100k</div><div class="kpi-detail">Deterministic evaluation</div></div>
          <div class="kpi-card"><div class="kpi-label">Primitive Actions</div><div class="kpi-value">${summary.n_primitive_leaves || leaves.length}</div><div class="kpi-detail">Top-down forward decomposition</div></div>
        </div>

        <div class="card" style="margin-bottom:1rem;">
          <div class="card-header">
            <div>
              <h3 class="card-title">Interactive HTN Task Decomposition Tree</h3>
              <p class="card-subtitle">Translates the national clean-cooking objective into country programmes, unit/period tasks, prerequisite checks, and primitive rollouts</p>
            </div>
            <div class="btn-group">
              <button type="button" class="btn btn-sm" id="htn-expand-all">Expand All</button>
              <button type="button" class="btn btn-sm" id="htn-collapse-all">Collapse All</button>
            </div>
          </div>
          <div id="htn-tree-mount" class="htn-tree"></div>
        </div>

        ${accordion(
        `Extracted Primitive Operator Schedule (${leaves.length} Steps)`,
        '<span class="badge badge-mock">plan_htn_trace_mock.json</span>',
        leafTableBody
      )}
      `;

      function collectAllIds(node, path = "root", out = []) {
        if (node.children && node.children.length) {
          out.push(path);
          node.children.forEach((ch, idx) => collectAllIds(ch, `${path}-${idx}`, out));
        }
        return out;
      }

      function renderTreeNode(node, path = "root", isRoot = true) {
        const hasChildren = Array.isArray(node.children) && node.children.length > 0;
        const isOpen = state.htnExpanded.has(path);
        const isPrim = node.node_type === "primitive";
        return `
          <div class="htn-node ${isRoot ? "htn-node-root" : ""}">
            <div class="htn-row ${isPrim ? "htn-row-primitive" : ""}">
              ${hasChildren
            ? `<button type="button" class="htn-toggle" data-htn-path="${esc(path)}" aria-expanded="${isOpen}">${isOpen ? "−" : "+"}</button>`
            : `<span style="width:22px; text-align:center; color:var(--accent-teal-soft);">•</span>`
          }
              <span class="htn-name">${esc(node.name)}</span>
              ${node.method ? `<span class="htn-method">${esc(node.method)}</span>` : ""}
              ${isPrim ? `<span class="htn-cost">x=${Number(node.allocation_value).toFixed(2)} · $${Number(node.cost_usd_millions).toFixed(2)}M</span>` : ""}
            </div>
            ${hasChildren && isOpen ? node.children.map((ch, idx) => renderTreeNode(ch, `${path}-${idx}`, false)).join("") : ""}
          </div>
        `;
      }

      function paintTree() {
        const mount = document.getElementById("htn-tree-mount");
        mount.innerHTML = renderTreeNode(htn.tree || {}, "root", true);
        mount.querySelectorAll("[data-htn-path]").forEach((btn) => {
          btn.addEventListener("click", () => {
            const p = btn.getAttribute("data-htn-path");
            if (state.htnExpanded.has(p)) state.htnExpanded.delete(p);
            else state.htnExpanded.add(p);
            paintTree();
          });
        });
      }

      document.getElementById("htn-expand-all").addEventListener("click", () => {
        state.htnExpanded = new Set(collectAllIds(htn.tree || {}));
        paintTree();
      });
      document.getElementById("htn-collapse-all").addEventListener("click", () => {
        state.htnExpanded = new Set(["root"]);
        paintTree();
      });

      paintTree();
    }

    function renderPddlTab(panel) {
      const sol = pddl.solution || {};
      const steps = sol.steps || [];

      const rawPddlBody = `
        <div class="grid grid-2">
          <div>
            <div style="font-size:0.78rem; font-weight:600; margin-bottom:0.35rem; color:var(--text-secondary);">pddl/domain.pddl</div>
            <pre class="code-block">${esc(pddl.domain_pddl || "")}</pre>
          </div>
          <div>
            <div style="font-size:0.78rem; font-weight:600; margin-bottom:0.35rem; color:var(--text-secondary);">pddl/problem_ind_npl.pddl &amp; Verified Output</div>
            <pre class="code-block">${esc((pddl.problem_pddl || "") + "\n\n; === SOLUTION ===\n" + (sol.raw_text || ""))}</pre>
          </div>
        </div>
      `;

      const predAndActionsBody = `
        <div class="table-wrap" style="margin-bottom:1rem;">
          <table class="data-table">
            <thead><tr><th>Predicate</th><th>Plain-English Meaning</th></tr></thead>
            <tbody>
              ${(pddl.predicates || []).map((pr) => `<tr><td><code>${esc(pr.name)}</code></td><td style="font-size:0.8rem;">${esc(pr.meaning)}</td></tr>`).join("")}
            </tbody>
          </table>
        </div>
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>Action</th><th>Preconditions</th><th>Effects</th><th>Operational Meaning</th></tr></thead>
            <tbody>
              ${(pddl.actions || [])
          .map(
            (a) => `
                <tr><td><code>${esc(a.name)}</code></td><td style="font-size:0.79rem;">${esc(a.preconditions)}</td><td style="font-size:0.79rem;">${esc(a.effects)}</td><td style="font-size:0.79rem; color:var(--text-secondary);">${esc(a.meaning)}</td></tr>
              `
          )
          .join("")}
            </tbody>
          </table>
        </div>
      `;

      panel.innerHTML = `
        <div class="grid grid-4" style="margin-bottom:1rem;">
          <div class="kpi-card"><div class="kpi-label">Classical Planner</div><div class="kpi-value" style="font-size:1.1rem;">pyperplan (A*+hFF)</div><div class="kpi-detail">Solved in ${pddl.solve_time_seconds}s</div></div>
          <div class="kpi-card"><div class="kpi-label">Replay Verification</div><div class="kpi-value" style="font-size:1.1rem; color:var(--accent-green-soft);">PASSED</div><div class="kpi-detail">39 final facts match goal</div></div>
          <div class="kpi-card"><div class="kpi-label">Evaluated Cost (Mock)</div><div class="kpi-value">$234.40M</div><div class="kpi-detail">is_feasible=True · 9 steps</div></div>
          <div class="kpi-card"><div class="kpi-label">Evaluated 2040 DALYs</div><div class="kpi-value">4.52M</div><div class="kpi-detail">95% UI [3.34M, 5.70M]</div></div>
        </div>

        <div class="card" style="margin-bottom:1rem;">
          <div class="card-header">
            <div>
              <h3 class="card-title">Verified 9-Step Action Sequence Timeline</h3>
              <p class="card-subtitle">Classical STRIPS planning enforces infrastructure prerequisites in P1 before rural LPG scale-up in P2</p>
            </div>
            <span class="badge badge-green">Replay Verified</span>
          </div>
          <div class="timeline">
            ${steps
          .map(
            (st) => `
              <div class="timeline-step ${st.operator === "advance-period" ? "timeline-step-advance" : ""}">
                <div class="step-num">STEP ${String(st.step).padStart(2, "0")}</div>
                <div class="step-period">${esc(st.period)}</div>
                <div>
                  <div class="step-action">${esc(st.raw)}</div>
                  <div class="step-desc">${esc(st.description)}</div>
                </div>
              </div>
            `
          )
          .join("")}
          </div>
        </div>

        ${accordion(
            "STRIPS Predicates, Preconditions &amp; Effects Explained",
            `<span class="badge badge-derived">${(pddl.actions || []).length} Action Schemas</span>`,
            predAndActionsBody
          )}
        ${accordion(
            "Raw PDDL Domain, Problem &amp; Planner Output",
            '<span class="badge badge-derived">pddl/*.pddl</span>',
            rawPddlBody
          )}
      `;
    }

    function renderActiveTab() {
      const panel = document.getElementById("planning-tab-panel");
      if (state.activeTab === "mdp") renderMdpTab(panel);
      else if (state.activeTab === "htn") renderHtnTab(panel);
      else renderPddlTab(panel);
      if (global.AppCharts) setTimeout(() => global.AppCharts.resizeAll(panel), 40);
    }

    container.querySelectorAll("[data-comp-view]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.compChartMode = btn.getAttribute("data-comp-view");
        container.querySelectorAll("[data-comp-view]").forEach((b) => b.classList.toggle("active", b === btn));
        renderComparisonChart();
      });
    });

    container.querySelectorAll("[data-plan-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.activeTab = btn.getAttribute("data-plan-tab");
        container.querySelectorAll("[data-plan-tab]").forEach((b) => {
          const isMatch = b === btn;
          b.classList.toggle("active", isMatch);
          b.setAttribute("aria-selected", String(isMatch));
        });
        renderActiveTab();
      });
    });

    renderComparisonChart();
    renderActiveTab();
  }

  global.PagePlanning = { render };
})(window);
