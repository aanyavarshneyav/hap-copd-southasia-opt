/* Main application router, data loader, glossary tooltips, and shared UI utilities */
(function (global) {
  "use strict";

  const ROUTES = {
    overview: { title: "Project Overview", module: "PageOverview" },
    historical: { title: "Historical Burden Analysis", module: "PageHistorical" },
    forecast: { title: "Forecasting & 2040 Projections", module: "PageForecast" },
    assumptions: { title: "Interventions & Assumptions", module: "PageAssumptions" },
    optimise: { title: "Multi-Objective Optimisation (NSGA-II)", module: "PageOptimise" },
    planning: { title: "Planning & Decision Intelligence", module: "PagePlanning" },
    research: { title: "Research, Limitations & Reproducibility", module: "PageResearch" },
  };

  const GLOSSARY = {
    DALYs: "Disability-Adjusted Life Years: a single health metric combining years of life lost to premature death and years lived with disability.",
    ASR: "Age-Standardized Rate (per 100,000): a weighted rate adjusting for differences in population age structure so countries and years can be compared fairly.",
    EAPC: "Estimated Annual Percentage Change: average yearly rate of change derived by fitting a log-linear trend ln(ASR) = α + βt over 1990–2023.",
    Joinpoint: "Piecewise linear regression (pwlf) identifying inflection years where an epidemiological trend speeds up or slows down.",
    PAF: "Population Attributable Fraction: the proportion of COPD burden estimated to be avoided if solid-fuel household air pollution exposure were eliminated.",
    "NSGA-II": "Non-dominated Sorting Genetic Algorithm II: an evolutionary algorithm that finds a diverse trade-off frontier across competing objectives simultaneously.",
    "Pareto front": "A collection of allocation plans where improving one objective (such as lowering 2040 DALYs) requires sacrificing at least one other (such as spending more budget).",
    Hypervolume: "A Pareto-front quality metric measuring the volume of objective space dominated by the solution set relative to a worst-case reference point (higher is better).",
    MDP: "Markov Decision Process: a sequential decision model that adapts choices over time (2026–2040) while accounting for uncertain intervention uptake.",
    HTN: "Hierarchical Task Network: a structured top-down planner decomposing national clean-cooking goals into regional tasks and concrete actions.",
    PDDL: "Planning Domain Definition Language (STRIPS): formal logic encoding prerequisites and effects to verify valid step-by-step action sequences.",
    "Rolling-Origin CV": "Time-series cross-validation that trains strictly on past years and tests on future 5-year windows (2009–13, 2014–18, 2019–23) without future leakage.",
    "Poisson Rate GLM": "Adapted generalized linear model predicting aggregated disease rates with a log-population offset (distinct from full age-period-cohort decomposition).",
  };

  const dataCache = new Map();

  async function loadJSON(filename) {
    if (dataCache.has(filename)) {
      return dataCache.get(filename);
    }
    const resp = await fetch(`data/${filename}`);
    if (!resp.ok) {
      throw new Error(`Failed to load data/${filename} (HTTP ${resp.status})`);
    }
    const parsed = await resp.json();
    dataCache.set(filename, parsed);
    return parsed;
  }

  function esc(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function term(key, label) {
    const def = GLOSSARY[key];
    const display = label || key;
    if (!def) return esc(display);
    return `<span class="term-tip" tabindex="0" aria-label="${esc(display)}: ${esc(def)}">${esc(display)}<span class="term-popover" role="tooltip"><strong>${esc(key)}:</strong> ${esc(def)}</span></span>`;
  }

  function accordion(title, badgeHtml, bodyHtml, openByDefault = false) {
    return `
      <details class="tech-accordion" ${openByDefault ? "open" : ""}>
        <summary class="tech-summary">
          <span class="tech-summary-left">
            <span class="tech-chevron" aria-hidden="true">▸</span>
            <span>${esc(title)}</span>
          </span>
          <span>${badgeHtml || ""}</span>
        </summary>
        <div class="tech-body">${bodyHtml}</div>
      </details>
    `;
  }

  function fmtNum(val, digits = 0) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return Number(val).toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function fmtMillions(val, digits = 2) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    return `${(Number(val) / 1e6).toFixed(digits)}M`;
  }

  function fmtUsdM(val, digits = 2) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "null (unresolved)";
    return `$${fmtNum(Number(val), digits)}M`;
  }

  function fmtPct(val, digits = 1, alreadyPct = false) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    const num = alreadyPct ? Number(val) : Number(val) * 100;
    return `${num.toFixed(digits)}%`;
  }

  function statusBadge(status) {
    const s = String(status || "").toLowerCase();
    if (s.includes("unresolved_scenario_a")) {
      return `<span class="badge badge-mock">Null (Disabled Scen. A)</span>`;
    }
    if (s.includes("unresolved") || s.includes("null")) {
      return `<span class="badge badge-mock">Unresolved Gap</span>`;
    }
    if (s.includes("active_in_real")) {
      return `<span class="badge badge-real">Active (REAL_V1)</span>`;
    }
    if (s.includes("mock")) {
      return `<span class="badge badge-mock">Synthetic Mock</span>`;
    }
    if (s.includes("sourced_partial") || s.includes("partial")) {
      return `<span class="badge badge-assumption">Partial / Assumption</span>`;
    }
    if (s === "sourced" || (s.includes("acquired") && !s.includes("proxy")) || s === "observed" || s === "real" || s === "verified" || s === "complete") {
      return `<span class="badge badge-real">${s === "verified" ? "Verified" : s === "complete" ? "Complete" : "Sourced / Real"}</span>`;
    }
    if (s.includes("derived")) {
      return `<span class="badge badge-derived">Derived</span>`;
    }
    if (s.includes("assumption")) {
      return `<span class="badge badge-assumption">Scenario Assumption</span>`;
    }
    if (s.includes("pending") || s.includes("blocked")) {
      return `<span class="badge badge-pending">Pending Stage 1–2</span>`;
    }
    return `<span class="badge badge-pending">${esc(status)}</span>`;
  }

  function mockBanner(title = "SYNTHETIC DATA — DEMONSTRATION ONLY", detail = "") {
    return `
      <div class="banner banner-mock" role="alert">
        <span class="banner-icon">SYNTHETIC DATA</span>
        <div class="banner-body">
          <strong>${esc(title)}</strong>
          ${detail ? `<p>${esc(detail)}</p>` : ""}
        </div>
      </div>
    `;
  }

  function navigateTo(route) {
    const clean = ROUTES[route] ? route : "overview";
    if (window.location.hash !== `#${clean}`) {
      window.location.hash = `#${clean}`;
    } else {
      renderCurrentRoute();
    }
  }

  function getActiveRoute() {
    const hash = (window.location.hash || "#overview").replace(/^#/, "");
    return ROUTES[hash] ? hash : "overview";
  }

  function closeMobileSidebar() {
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebar-backdrop");
    const toggle = document.getElementById("sidebar-toggle");
    if (sidebar) sidebar.classList.remove("open");
    if (backdrop) backdrop.hidden = true;
    if (toggle) toggle.setAttribute("aria-expanded", "false");
  }

  function openMobileSidebar() {
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebar-backdrop");
    const toggle = document.getElementById("sidebar-toggle");
    if (sidebar) sidebar.classList.add("open");
    if (backdrop) backdrop.hidden = false;
    if (toggle) toggle.setAttribute("aria-expanded", "true");
  }

  async function renderCurrentRoute() {
    const route = getActiveRoute();
    const routeInfo = ROUTES[route];
    const container = document.getElementById("page-content");
    const titleEl = document.getElementById("topbar-section-name");

    if (titleEl) titleEl.textContent = routeInfo.title;

    document.querySelectorAll(".nav-link").forEach((link) => {
      const isMatch = link.getAttribute("data-route") === route;
      link.classList.toggle("active", isMatch);
      if (isMatch) {
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    });

    closeMobileSidebar();

    container.innerHTML = `
      <div class="loading-state">
        <div class="spinner" aria-hidden="true"></div>
        <p>Loading ${esc(routeInfo.title)}…</p>
      </div>
    `;

    try {
      const mod = global[routeInfo.module];
      if (!mod || typeof mod.render !== "function") {
        throw new Error(`Page module ${routeInfo.module} is not registered.`);
      }
      await mod.render(container);
      container.focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: "auto" });

      // Ensure Plotly charts inside newly opened <details> accordions resize cleanly
      container.querySelectorAll("details.tech-accordion").forEach((det) => {
        det.addEventListener("toggle", () => {
          if (det.open && global.AppCharts) {
            setTimeout(() => global.AppCharts.resizeAll(det), 40);
          }
        });
      });

      if (global.AppCharts) {
        setTimeout(() => global.AppCharts.resizeAll(container), 60);
      }
    } catch (err) {
      container.innerHTML = `
        <div class="banner banner-mock">
          <span class="banner-icon">ERROR</span>
          <div class="banner-body">
            <strong>Unable to render ${esc(routeInfo.title)}</strong>
            <p>${esc(err.message || String(err))}</p>
          </div>
        </div>
      `;
    }
  }

  function infoBanner(title = "VALIDATED REAL-DATA RUN", detail = "") {
    return `
      <div class="banner banner-info" role="status">
        <span class="banner-icon">●</span>
        <div class="banner-body">
          <strong>${esc(title)}</strong>
          ${detail ? `<p>${esc(detail)}</p>` : ""}
        </div>
      </div>
    `;
  }

  async function init() {
    try {
      const meta = await loadJSON("metadata.json");
      const statusText = document.getElementById("global-data-status-text");
      if (statusText && meta) {
        statusText.textContent = meta.use_mock_forecasts
          ? "Mock Decision Demo (MOCK_V1) · Real Clean-Cooking Inputs"
          : `Real Decision Pipeline (${meta.data_version || "GBD 2023"} + ${meta.parameter_version || "REAL_V1_2024USD"}) · Scenario A`;
      }
    } catch (_) {
      /* keep fallback topbar text */
    }

    const toggle = document.getElementById("sidebar-toggle");
    const closeBtn = document.getElementById("sidebar-close");
    const backdrop = document.getElementById("sidebar-backdrop");

    if (toggle) toggle.addEventListener("click", openMobileSidebar);
    if (closeBtn) closeBtn.addEventListener("click", closeMobileSidebar);
    if (backdrop) backdrop.addEventListener("click", closeMobileSidebar);

    window.addEventListener("hashchange", renderCurrentRoute);
    window.addEventListener("resize", () => {
      if (global.AppCharts) global.AppCharts.resizeAll(document.getElementById("page-content"));
    });

    await renderCurrentRoute();
  }

  global.App = {
    ROUTES,
    GLOSSARY,
    loadJSON,
    esc,
    term,
    accordion,
    fmtNum,
    fmtMillions,
    fmtUsdM,
    fmtPct,
    statusBadge,
    mockBanner,
    infoBanner,
    navigateTo,
  };

  document.addEventListener("DOMContentLoaded", init);
})(window);
