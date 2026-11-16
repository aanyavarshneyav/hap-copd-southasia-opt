# South Asia HAP–COPD Static Research Platform (`site/`)

Locally hosted, zero-backend static research observatory for the **Forecasting and Optimal Intervention Planning for COPD Burden Attributable to Household Air Pollution in South Asia** AI Lab project.

## UX Architecture (V2 Visual Storytelling)

Every section follows a progressive **Level 1 → Level 2 → Level 3** information hierarchy:
- **Level 1 (Immediate Understanding):** Concise title, one-sentence plain-English explanation, one prominent visualisation or diagram, and a supported takeaway.
- **Level 2 (Exploration):** Interactive selectors, chart mode switches, Pareto click-to-inspect, expandable HTN tree, and paradigm tabs.
- **Level 3 (Technical Details on Demand):** Expandable `<details>` accordions holding equations, constraint math, provenance matrices, source registers, and raw PDDL code, paired with inline hover/focus glossary tooltips (`DALYs`, `ASR`, `NSGA-II`, `Pareto front`, `MDP`, `HTN`, `PDDL`, etc.).

## Running Locally (Offline-Ready)

All charting (`assets/vendor/plotly-2.35.2.min.js`), styles, data snapshots (`data/*.json`), and downloadable artifacts (`downloads/*`) are bundled locally with zero runtime CDN or backend dependencies.

```bash
python3 -m http.server 8080 --directory site
```

Then open [http://localhost:8080](http://localhost:8080).

## Regenerating Static Data Snapshots

`site/scripts/prepare_data.py` reads existing repository configs, `outputs/tables/`, `pddl/`, `docs/`, and `b/param-sourcing:research/person_b_inputs/` **without modifying any file outside `site/`**:

```bash
.venv/bin/python site/scripts/prepare_data.py
```
