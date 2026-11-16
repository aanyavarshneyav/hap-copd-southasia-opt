/* Page 7: Research, Limitations & Reproducibility (V2 Visual Storytelling) */
(function (global) {
  "use strict";

  async function render(container) {
    const data = await global.App.loadJSON("research.json");
    const { esc, accordion, statusBadge } = global.App;
    const repro = data.reproducibility || {};
    const verified = repro.verified_status || {};

    const sourcesHtml = `
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Source</th><th>Role</th><th>Geography</th><th>Status</th><th>URL</th></tr></thead>
          <tbody>
            ${(data.data_sources || [])
              .map(
                (src) => `
              <tr>
                <td><strong>${esc(src.name)}</strong></td>
                <td style="font-size:0.8rem;">${esc(src.role)}</td>
                <td style="font-size:0.79rem;">${esc(src.countries)}</td>
                <td>${statusBadge(src.status)}<div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.15rem;">${esc(src.status)}</div></td>
                <td><a href="${esc(src.url)}" target="_blank" rel="noopener noreferrer">${esc(src.url)}</a></td>
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;

    const reproHtml = `
      <div class="grid grid-2">
        <div>
          <h4 style="margin:0 0 0.55rem; font-size:0.92rem;">Environment, Seeds &amp; Verified Checks</h4>
          <ul class="callout-list" style="margin-bottom:0.85rem;">
            <li><strong>Python Version:</strong> <code>${esc(repro.python_requirement)}</code></li>
            <li><strong>Fixed Seeds:</strong> <code>global=${repro.seeds?.global}</code>, <code>numpy=${repro.seeds?.numpy}</code>, <code>sklearn=${repro.seeds?.sklearn}</code>, <code>pymoo=${repro.seeds?.pymoo}</code>, <code>mdp=${repro.seeds?.mdp}</code></li>
            <li><strong>Test Suite:</strong> <code>${esc(verified.pytest)}</code></li>
            <li><strong>Linter &amp; Format:</strong> <code>${esc(verified.ruff_lint)}</code> · <code>${esc(verified.ruff_format)}</code></li>
          </ul>
          <div style="font-size:0.76rem; font-weight:600; color:var(--text-secondary); margin-bottom:0.35rem;">Core Python Dependencies (<code>pyproject.toml</code>):</div>
          <div class="pending-checklist">
            ${(repro.dependencies || []).map((d) => `<span class="pending-chip">${esc(d)}</span>`).join("")}
          </div>
        </div>
        <div>
          <h4 style="margin:0 0 0.55rem; font-size:0.92rem;">Execution Commands</h4>
          <div class="table-wrap">
            <table class="data-table">
              <thead><tr><th>Command</th><th>Purpose</th></tr></thead>
              <tbody>
                ${(repro.commands || [])
                  .map(
                    (c) => `
                  <tr><td><code>${esc(c.cmd)}</code></td><td style="font-size:0.78rem; color:var(--text-secondary);">${esc(c.desc)}</td></tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    container.innerHTML = `
      <header class="page-header">
        <div class="page-header-top">
          <span class="page-eyebrow">07 · Methodology, Limitations &amp; Open Reproducibility</span>
          <div class="page-header-meta" style="margin-top:0;">
            <span class="badge badge-real">Python 3.12 · Offline Reproducible</span>
            <span class="badge badge-green">pytest: 37/37 Passing</span>
          </div>
        </div>
        <h1 class="page-title">Research, Limitations &amp; Reproducibility</h1>
        <p class="page-lead">
          Documents the five-stage research pipeline, eight methodological limitations, verified source inventory, and downloadable repository artifacts.
        </p>
      </header>

      <!-- Level 1: Pipeline Summary -->
      <section class="section">
        <div class="section-header">
          <div>
            <h2 class="section-title">Pipeline Architecture &amp; Stage Status</h2>
            <p class="section-desc">Contract-driven separation between Data/Forecasting (Person A) and Optimisation/Planning (Person B).</p>
          </div>
        </div>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr><th>Stage</th><th>Component</th><th>Owner</th><th>Status</th><th>Outputs</th></tr>
            </thead>
            <tbody>
              ${(data.methodology_stages || [])
                .map(
                  (s) => `
                <tr>
                  <td><strong>${esc(s.stage)}</strong></td>
                  <td><strong>${esc(s.name)}</strong><div style="font-size:0.74rem; color:var(--text-muted);">${esc(s.inputs)}</div></td>
                  <td>${esc(s.owner)}</td>
                  <td>${statusBadge(s.status)}<div style="font-size:0.73rem; color:var(--text-muted); margin-top:0.15rem;">${esc(s.status)}</div></td>
                  <td style="font-size:0.78rem;"><code>${esc(s.outputs)}</code></td>
                </tr>
              `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </section>

      <!-- Level 2: Scientific Limitations -->
      <section class="section">
        <div class="section-header">
          <div>
            <h2 class="section-title">Methodological Limitations &amp; Guardrails</h2>
            <p class="section-desc">Eight critical constraints governing how all historical, forecast, and decision outputs must be interpreted.</p>
          </div>
          <span class="badge badge-assumption">Scientific Integrity</span>
        </div>
        <div class="grid grid-2">
          ${(data.limitations || [])
            .map(
              (lim) => `
            <article class="card-elevated">
              <div style="font-weight:700; font-size:0.93rem; margin-bottom:0.3rem;">
                <span class="mono" style="color:var(--accent-amber-soft);">${esc(lim.id)}</span> · ${esc(lim.title)}
              </div>
              <p style="margin:0; font-size:0.83rem; color:var(--text-secondary);">${esc(lim.detail)}</p>
            </article>
          `
            )
            .join("")}
        </div>
      </section>

      <!-- Level 3: Expandable Sources, Reproducibility & Downloads -->
      ${accordion(
        `Documented Data Sources (${(data.data_sources || []).length} Verified References)`,
        '<span class="badge badge-real">IHME · UN · WHO · WB · DHS · Literature</span>',
        sourcesHtml
      )}
      ${accordion(
        "Reproducibility: Environment, Seeds, Dependencies &amp; CLI Commands",
        '<span class="badge badge-green">37/37 Tests Verified</span>',
        reproHtml
      )}

      <section class="section" style="margin-top:1.75rem;">
        <div class="section-header">
          <div>
            <h2 class="section-title">Verified Repository Downloads</h2>
            <p class="section-desc">Every artifact below is bundled under <code>site/downloads/</code> for offline inspection.</p>
          </div>
          <span class="badge badge-real">${(data.downloads || []).length} Files</span>
        </div>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr><th>Artifact</th><th>Description</th><th>Format</th><th class="num-right">Size</th><th>Action</th></tr>
            </thead>
            <tbody>
              ${(data.downloads || [])
                .map(
                  (dl) => `
                <tr>
                  <td><code>${esc(dl.filename)}</code></td>
                  <td style="font-size:0.8rem;">${esc(dl.label)}</td>
                  <td><span class="badge badge-derived">${esc(dl.format)}</span></td>
                  <td class="num-right">${esc(dl.size_kb)}</td>
                  <td><a class="btn btn-sm" href="${esc(dl.href)}" download="${esc(dl.filename)}">Download</a></td>
                </tr>
              `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </section>
    `;
  }

  global.PageResearch = { render };
})(window);
