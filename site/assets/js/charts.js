/* Reusable Plotly dark-observatory theme and chart helpers */
(function (global) {
  "use strict";

  const PALETTE = {
    blue: "#83CDD1",
    blueStrong: "#68BBC2",
    teal: "#68BBC2",
    tealStrong: "#54989E",
    amber: "#D6C7A9",
    amberStrong: "#DF8B6C",
    red: "#DF8B6C",
    redStrong: "#C87152",
    green: "#8A9A86",
    purple: "#A6AAA6",
    slate: "#928B80",
    grid: "rgba(185, 178, 166, 0.14)",
    bgPlot: "rgba(23, 22, 19, 0)",
    bgPaper: "rgba(23, 22, 19, 0)",
    text: "#EEEAE1",
    textMuted: "#B9B2A6",
  };

  function baseLayout(overrides = {}) {
    const base = {
      paper_bgcolor: PALETTE.bgPaper,
      plot_bgcolor: PALETTE.bgPlot,
      font: {
        family: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        color: PALETTE.text,
        size: 12,
      },
      margin: { l: 64, r: 24, t: 38, b: 54 },
      xaxis: {
        gridcolor: PALETTE.grid,
        zerolinecolor: "rgba(185, 178, 166, 0.24)",
        tickfont: { color: PALETTE.textMuted, size: 11 },
        titlefont: { color: PALETTE.text, size: 12 },
      },
      yaxis: {
        gridcolor: PALETTE.grid,
        zerolinecolor: "rgba(185, 178, 166, 0.24)",
        tickfont: { color: PALETTE.textMuted, size: 11 },
        titlefont: { color: PALETTE.text, size: 12 },
      },
      legend: {
        bgcolor: "rgba(29, 27, 24, 0.85)",
        bordercolor: "#403B33",
        borderwidth: 1,
        font: { size: 11, color: PALETTE.text },
      },
      hoverlabel: {
        bgcolor: "#24221E",
        bordercolor: "#403B33",
        font: { color: "#EEEAE1", size: 12 },
      },
    };

    return {
      ...base,
      ...overrides,
      xaxis: { ...base.xaxis, ...(overrides.xaxis || {}) },
      yaxis: { ...base.yaxis, ...(overrides.yaxis || {}) },
      legend: { ...base.legend, ...(overrides.legend || {}) },
      margin: { ...base.margin, ...(overrides.margin || {}) },
    };
  }

  function plot(elOrId, traces, layoutOverrides = {}, configOverrides = {}) {
    const el = typeof elOrId === "string" ? document.getElementById(elOrId) : elOrId;
    if (!el || typeof global.Plotly === "undefined") {
      return Promise.resolve(null);
    }
    const layout = baseLayout(layoutOverrides);
    const config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
      ...configOverrides,
    };
    return global.Plotly.react(el, traces, layout, config);
  }

  function resizeAll(container) {
    if (typeof global.Plotly === "undefined") return;
    const root = container || document;
    root.querySelectorAll(".js-plotly-plot").forEach((el) => {
      try {
        global.Plotly.Plots.resize(el);
      } catch (_) {
        /* ignore detached */
      }
    });
  }

  global.AppCharts = {
    PALETTE,
    baseLayout,
    plot,
    resizeAll,
  };
})(window);
