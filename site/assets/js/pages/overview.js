/* Page 1: Project Overview (Visual Storytelling Hero) */
(function (global) {
  "use strict";

  async function render(container) {
    const [overview, meta] = await Promise.all([
      global.App.loadJSON("overview.json"),
      global.App.loadJSON("metadata.json"),
    ]);
    const { esc, term, accordion } = global.App;

    const statusRows = Object.entries(meta.pipeline_status || {}).map(([, info]) => {
      let badge = '<span class="badge badge-pending">Pending</span>';
      if (info.status === "complete_mock") {
        badge = '<span class="badge badge-mock">Complete (Mock Data)</span>';
      } else if (info.status === "partial_real" || info.status === "partial_inputs_acquired") {
        badge = '<span class="badge badge-real">Real Inputs Acquired</span>';
      }
      return `
        <tr>
          <td><strong>${esc(info.label)}</strong></td>
          <td>${badge}</td>
          <td style="font-size:0.8rem; color:var(--text-secondary);">${esc(info.detail)}</td>
        </tr>
      `;
    });

    const techDetailsHtml = `
      <div class="grid grid-2" style="margin-bottom:1.25rem;">
        <div>
          <h4 style="margin:0 0 0.6rem; font-size:0.92rem;">Main Research Questions</h4>
          <ul class="callout-list">
            ${(overview.research_questions || [])
        .map(
          (rq) =>
            `<li><strong>${esc(rq.id)}:</strong> ${esc(rq.question)}</li>`
        )
        .join("")}
          </ul>
        </div>
        <div>
          <h4 style="margin:0 0 0.6rem; font-size:0.92rem;">Methodology Caveats</h4>
          <ul class="callout-list">
            <li><strong>Modelled Estimates:</strong> GBD 2023 outputs are modelled epidemiological estimates, not observed vital counts, and Stage 1–2 GBD exports remain pending.</li>
            <li><strong>All-Age vs. ${term("ASR")}:</strong> Stage 1–2 uses aggregated all-age burden and age-standardized rates rather than 5-year age-specific COPD bins.</li>
            <li><strong>Scenario Framing:</strong> Stage 3–4 runs use synthetic <code>MOCK_V1</code> contracts; intervention costs and effects are scenario assumptions.</li>
          </ul>
        </div>
      </div>
      <h4 style="margin:0 0 0.6rem; font-size:0.92rem;">Pipeline Data Integration Status</h4>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr><th>Workstream</th><th>Status</th><th>Evidence</th></tr>
          </thead>
          <tbody>${statusRows.join("")}</tbody>
        </table>
      </div>
    `;

    container.innerHTML = `
      <!-- Hero Section -->
      <section class="hero-grid" aria-labelledby="hero-headline">
        <div>
          <div class="page-eyebrow" style="margin-bottom:0.5rem;">South Asia Public-Health Decision Intelligence</div>
          <h1 id="hero-headline" class="hero-title">Household Air Pollution &amp; COPD</h1>
          <p class="hero-copy">
            Solid-fuel cooking smoke drives severe chronic lung disease across South Asia. This project forecasts COPD burden to 2040 and combines multi-objective optimisation with sequential AI planning to allocate clean-cooking budgets across India and Nepal.
          </p>
          <div class="hero-actions">
            <button type="button" class="btn btn-primary" data-jump="optimise">Explore Allocation Trade-Offs →</button>
            <button type="button" class="btn" data-jump="historical">View Clean-Cooking Gap</button>
          </div>
        </div>

        <div class="hero-visual-panel">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
            <strong style="font-size:0.88rem;">2024 Clean-Cooking Access Gap (India &amp; Nepal)</strong>
            <span class="badge badge-real">Real WHO SDG 7.1.2</span>
          </div>
          <div id="hero-gap-chart" style="width:100%; height:245px;"></div>
          <div class="hero-visual-caption">
            Share of population using clean fuels (%, 2024) vs. remaining transition gap targeted by 2026–2040 interventions. Source: WHO GHO &amp; UN WUP 2025.
          </div>
        </div>
      </section>

      <!-- 3-Stage Simple Research Pipeline -->
      <section class="section">
        <div class="section-header">
          <div>
            <h2 class="section-title">How the Research Fits Together</h2>
            <p class="section-desc">Three connected stages move from describing disease burden to prescribing feasible clean-cooking budgets.</p>
          </div>
        </div>
        <div class="pipeline-3">
          <article class="stage-step-card">
            <span class="stage-step-num">STAGE 01 · OBSERVE</span>
            <h3 class="stage-step-title">Historical Burden Analysis</h3>
            <p class="stage-step-desc">Measure 1990–2023 COPD ${term("DALYs")} and deaths alongside real urban/rural clean-cooking transitions across five South Asian nations.</p>
          </article>
          <article class="stage-step-card">
            <span class="stage-step-num">STAGE 02 · PROJECT</span>
            <h3 class="stage-step-title">Forecasting to 2040</h3>
            <p class="stage-step-desc">Benchmark five statistical and ML models with ${term("Rolling-Origin CV")} to project 2040 burden and calibrate a fast response curve.</p>
          </article>
          <article class="stage-step-card">
            <span class="stage-step-num">STAGE 03 · DECIDE</span>
            <h3 class="stage-step-title">Optimisation &amp; AI Planning</h3>
            <p class="stage-step-desc">Compare ${term("NSGA-II")} trade-off plans against adaptive ${term("MDP")}, ${term("HTN")}, and ${term("PDDL")} strategies under uncertainty.</p>
          </article>
        </div>
      </section>

      <!-- Verified Project Indicators -->
      <section class="section">
        <div class="indicator-strip" aria-label="Verified project scope indicators">
          <div class="indicator-item">
            <span class="indicator-val">5 Countries</span>
            <span class="indicator-lbl">South Asia panel (IND, NPL, PAK, BGD, BTN)</span>
          </div>
          <div class="indicator-item">
            <span class="indicator-val">1990–2040</span>
            <span class="indicator-lbl">Historical GBD (to 2023) &amp; 2040 forecast horizon</span>
          </div>
          <div class="indicator-item">
            <span class="indicator-val">4 Units × 4 Levers</span>
            <span class="indicator-lbl">India &amp; Nepal urban/rural × cooking &amp; infra</span>
          </div>
          <div class="indicator-item">
            <span class="indicator-val">4 Planning Methods</span>
            <span class="indicator-lbl">NSGA-II, NumPy MDP, HTN tree, STRIPS/PDDL</span>
          </div>
        </div>
      </section>

      <!-- Four Clear Entry Points -->
      <section class="section">
        <div class="section-header">
          <div>
            <h2 class="section-title">Explore the Research Sections</h2>
          </div>
        </div>
        <div class="entry-grid">
          <button type="button" class="entry-card" data-jump="historical">
            <div class="entry-card-top">
              <span class="entry-card-title">02 · Historical Burden &amp; Clean Cooking</span>
              <span class="badge badge-real">Real WHO/DHS Data</span>
            </div>
            <p class="entry-card-desc">Explore 1990–2024 clean-cooking trends and fuel transitions in India and Nepal, plus Stage 1 GBD 2023 readiness.</p>
          </button>

          <button type="button" class="entry-card" data-jump="forecast">
            <div class="entry-card-top">
              <span class="entry-card-title">03 · Forecasting &amp; 2040 Projections</span>
              <span class="badge badge-mock">Stage 2 Pending · MOCK_V1</span>
            </div>
            <p class="entry-card-desc">Inspect the 5-model benchmark design and interact with 2025–2040 counterfactual forecast contracts.</p>
          </button>

          <button type="button" class="entry-card" data-jump="optimise">
            <div class="entry-card-top">
              <span class="entry-card-title">05 · Multi-Objective Optimisation (NSGA-II)</span>
              <span class="badge badge-mock">Synthetic MOCK_V1</span>
            </div>
            <p class="entry-card-desc">Navigate the 3D Pareto front balancing 2040 DALYs, total cost, and sex equity across 219 feasible plans.</p>
          </button>

          <button type="button" class="entry-card" data-jump="planning">
            <div class="entry-card-top">
              <span class="entry-card-title">06 · Sequential &amp; Symbolic Planning</span>
              <span class="badge badge-mock">MDP · HTN · PDDL</span>
            </div>
            <p class="entry-card-desc">Compare adaptive MDP recourse, HTN task trees, and verified PDDL schedules against static GA plans.</p>
          </button>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:0.75rem; align-items:center; font-size:0.86rem; color:var(--text-secondary);">
          <span>Supporting sections:</span>
          <button type="button" class="btn btn-sm" data-jump="assumptions">04 · Intervention Costs &amp; Assumptions</button>
          <button type="button" class="btn btn-sm" data-jump="research">07 · Limitations, Sources &amp; Downloads</button>
        </div>
      </section>

      <!-- Level 3: Technical Scope & Live Data Status (Expandable) -->
      ${accordion(
      "Technical Details: Research Questions &amp; Live Data Integration Status",
      '<span class="badge badge-derived">Level 3 · Expandable</span>',
      techDetailsHtml,
      false
    )}
    `;

    // Render hero clean-cooking gap chart using real WHO SDG 7.1.2 2024 numbers
    global.AppCharts.plot(
      "hero-gap-chart",
      [
        {
          y: ["Nepal Rural<br>(9.8M pop)", "Nepal Urban<br>(19.8M pop)", "India Rural<br>(941.4M pop)", "India Urban<br>(522.4M pop)"],
          x: [25, 64, 69, 96],
          name: "2024 Clean Cooking (%)",
          type: "bar",
          orientation: "h",
          marker: { color: "#68BBC2" },
          hovertemplate: "<b>%{y}</b><br>Clean Cooking (2024): %{x}%<extra></extra>",
        },
        {
          y: ["Nepal Rural<br>(9.8M pop)", "Nepal Urban<br>(19.8M pop)", "India Rural<br>(941.4M pop)", "India Urban<br>(522.4M pop)"],
          x: [75, 36, 31, 4],
          name: "Solid-Fuel Gap (%)",
          type: "bar",
          orientation: "h",
          marker: { color: "rgba(223, 139, 108, 0.52)" },
          hovertemplate: "<b>%{y}</b><br>Remaining Gap: %{x}%<extra></extra>",
        },
      ],
      {
        barmode: "stack",
        xaxis: { title: "Population Share (%)", range: [0, 102] },
        yaxis: { automargin: true },
        legend: { orientation: "h", x: 0, y: 1.32 },
        margin: { l: 105, r: 12, t: 62, b: 44 },
      }
    );

    container.querySelectorAll("[data-jump]").forEach((btn) => {
      btn.addEventListener("click", () => {
        global.App.navigateTo(btn.getAttribute("data-jump"));
      });
    });
  }

  global.PageOverview = { render };
})(window);
