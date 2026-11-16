# Planning Component Specification (MDP, HTN, STRIPS/PDDL)

All three planning paradigms operate over the **same 4 units** (`IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`), **4 interventions** (`lpg_subsidy`, `improved_biomass`, `electric_induction`, `infra_lpg_grid`), and **3 five-year periods** (`P1: 2026-2030`, `P2: 2031-2035`, `P3: 2036-2040`), running on CPU-only laptops in seconds.

---

## 1. Markov Decision Process (MDP) — Custom NumPy Value Iteration

### 1.1 Formal Tuple $(S, A, P, R, \gamma, H)$
- **Horizon ($H = 3$ decision epochs):** $t \in \{0, 1, 2\}$ corresponding to `P1 (2026–30)`, `P2 (2031–35)`, `P3 (2036–40)`, plus terminal state at $t=3$ (`2040`).
- **State Space ($S$):**
  To prevent combinatorial blow-up while preserving sequential budget trade-offs and equal-footing comparison with GA:
  $$s_t = \big(t,\; c^{\text{bin}}_{1}, c^{\text{bin}}_{2}, c^{\text{bin}}_{3}, c^{\text{bin}}_{4},\; \text{infra}_{\text{IND\_r}},\; \text{infra}_{\text{NPL\_r}},\; b^{\text{bin}}_t\big)$$
  - $c^{\text{bin}}_u \in \{0, 1, 2, 3, 4, 5\}$: discretized clean-cooking coverage tier per unit (`{0%, 20%, 40%, 60%, 80%, 100%}`, **6 levels**; or 5 feasible levels above current baseline).
  - $\text{infra}_{\text{IND\_r}}, \text{infra}_{\text{NPL\_r}} \in \{0, 1\}$: rural LPG/grid prerequisite status (`1` = ready; urban units assumed `1` at $t=0$ unless verified otherwise in `ASSUMPTIONS.md`).
  - $b^{\text{bin}}_t \in \{0, 1, \dots, 5\}$: discretized remaining budget tier (**6 levels**, e.g., $0\%, 20\%, 40\%, 60\%, 80\%, 100\%$ of $B_{\text{total}}$).
  - **Joint state count per epoch:** $5^4 \times 2^2 \times 6 = 15,000$ states (or $6^4 \times 4 \times 6 = 31,104$ max). With finite-horizon backward induction ($t = 2, 1, 0$) and reachable-state pruning, actual active states are $< 10,000$.
- **Action Space ($A(s_t)$):**
  - Instead of an unrestricted Cartesian product across all interventions × units, each period selects from **discrete portfolio bundles** (e.g., allocate 1 budget tranche to `{no_op, infra_lpg_grid, lpg_subsidy, improved_biomass, electric_induction}` in a target unit, or bounded multi-unit bundles up to remaining $b^{\text{bin}}_t$, capped at ~25–50 feasible macro-actions per state).
  - **Precondition gating:** `lpg_subsidy` / `electric_induction` scale-up in rural units is blocked or capped unless $\text{infra}_u = 1$ (matching GA and PDDL rules).
- **Transition Function $P(s' \mid s, a)$:**
  - Sparse stochastic transition modeling **adoption / effectiveness uncertainty**:
    - Nominal outcome (prob $p_{\text{nom}}$, e.g. `0.7`): coverage advances by expected step $\Delta c(a)$ (capped at $100\%$).
    - Low-uptake outcome (prob $p_{\text{low}}$, e.g. `0.15`): advances by $0.5 \cdot \Delta c(a)$ (rounded to nearest bin) or stays in current bin.
    - High-uptake outcome (prob $p_{\text{high}}$, e.g. `0.15`): advances by $1.25 \cdot \Delta c(a)$ (capped at $100\%$).
    - *(Exact transition probabilities logged as scenario sensitivity in `docs/ASSUMPTIONS.md`.)*
  - Deterministic transitions for `infra` flags (`0 -> 1` upon `infra_lpg_grid` action) and `budget` depletion ($b' = \max(0, b - \text{cost\_tier}(a))$).
- **Reward Function $R(s, a, s')$:**
  - Per period: $R(s_t, a_t, s_{t+1}) = \text{DALYs\_averted}(s_t \to s_{t+1}) - \lambda_{\text{cost}} \cdot \text{Cost}(a_t) - \lambda_{\text{eq}} \cdot \text{ASR\_Gap}_{t+1}$
  - Uses the **exact same** `exposure_response_surrogate.json` and intervention cost parameters as Stage 3 NSGA-II.
  - $\lambda_{\text{cost}}$ and $\lambda_{\text{eq}}$ are swept across values corresponding to the GA Pareto trade-off frontier for equal-footing comparison.
- **Discount Factor ($\gamma$):**
  - Primary: $\gamma = 0.774$ ($\approx 0.95^5$ per 5-year period, equivalent to ~5%/yr health & cost discounting), with sensitivity run at $\gamma = 1.0$ (undiscounted, matching static GA cumulative accounting).
- **Solver (`src/planning/mdp.py`):**
  - Vectorized NumPy finite-horizon backward induction (exact Bellman optimality over $t=2,1,0$) plus infinite-horizon discounted value iteration option (`tol=1e-6`) for stationary policy diagnostics.

### 1.2 State-Space Complexity & Blow-Up Risks (Flagged)
| Design Choice | Why It Blows Up | Adopted Safeguard |
| --- | --- | --- |
| Fine coverage discretization (e.g., 5% steps = 21 bins) | $21^4 = 194,481$ states before adding infra/budget | Restrict to **5–6 coverage bins** (`20%` steps) → $5^4 = 625$ base combinations |
| Full Cartesian actions across 4 units × 4 interventions | Even 3 levels each = $3^{16} = 4.3 \times 10^7$ actions/state | Restrict to **budget-bounded discrete portfolio actions** ($\le 50$ valid actions/state) |
| Continuous remaining budget state | Infinite states | Coarsely bin remaining budget into **6 discrete tranches** |
| Dense transition matrix $P \in \mathbb{R}^{\|S\| \times \|A\| \times \|S\|}$ | Memory blow-up ($15\text{k} \times 50 \times 15\text{k} \times 8\text{B} \approx 90\text{ GB}$) | Sparse `(next_idx, prob)` indexing with 2–3 stochastic branches ($< 50\text{ MB}$ RAM) |

---

## 2. Hierarchical Task Network (HTN) — Plain Python Decomposition Tree
Implemented in `src/planning/htn.py` as a clean, inspectable Python task-network planner (total-order forward decomposition with method precondition checks).

### 2.1 Decomposition Hierarchy
```text
Goal: AchieveSouthAsiaCleanCookingPlan(units=[IND_u, IND_r, NPL_u, NPL_r], horizon=2026-2040, budget)
 └── Task: PlanCountryProgram(country) [for India, Nepal]
      ├── Task: PlanUnitProgram(unit, target_tier, period) [for urban, rural × P1..P3]
      │    ├── Method M1 (Urban, grid & LPG ready):
      │    │    └── [deploy_electric_induction OR deploy_lpg_subsidy]
      │    ├── Method M2 (Rural, infra NOT ready, high budget):
      │    │    ├── Task: BuildPrerequisites(unit) -> [!op_build_lpg_grid_infra(unit, period)]
      │    │    └── Task: ScaleCleanFuel(unit) -> [!op_deploy_lpg_subsidy OR !op_deploy_electric]
      │    └── Method M3 (Rural, infra NOT ready / constrained budget bridge):
      │         └── [!op_deploy_improved_biomass(unit, period)]
      └── Task: VerifyBudgetAndEquityGuard(plan_state)
```

### 2.2 Primitive Operators & Preconditions
- `!op_build_lpg_grid_infra(unit, period)`: requires `budget >= cost_infra`, `!infra_ready(unit)` → sets `infra_ready(unit) = True`.
- `!op_deploy_lpg_subsidy(unit, period)`: requires `infra_ready(unit)`, `budget >= cost_lpg`, `coverage(unit) < max_cov` → increases `coverage(unit)`.
- `!op_deploy_electric_induction(unit, period)`: requires `grid_ready(unit)`, `budget >= cost_elec`, `coverage(unit) < max_cov` → increases `coverage(unit)`.
- `!op_deploy_improved_biomass(unit, period)`: requires `budget >= cost_ics`, `coverage(unit) < max_cov` → gives interim exposure reduction without LPG/grid prerequisite.

---

## 3. STRIPS / PDDL Encoding (`pyperplan`)
Files live in `pddl/domain.pddl` and `pddl/problem_ind_npl.pddl`, solved via `pyperplan` (`astar` + `hff` heuristic) in `src/planning/pddl_runner.py`.

### 3.1 Critical `pyperplan` Constraint
`pyperplan` supports **classical STRIPS only** (`:requirements :strips :typing`). It does **not** support numeric fluents (`:fluents`), conditional effects, or action costs. Therefore:
- Coverage is modeled as ordered discrete tier predicates (`cov-low`, `cov-med`, `cov-high`).
- Budget is modeled via **consumable discrete budget tokens** (`(budget-avail ?b - budget-token)` consumed by each rollout action) and sequential period tokens (`(current-period ?p)`, `(next-period ?p1 ?p2)`).
- Keep object count tight (4 units, 3 periods, ~6–8 budget tokens) so `pyperplan` solves in **< 2 seconds**.

### 3.2 Core Types, Predicates & Actions
- **Types:** `unit period coverage-tier budget-token`
- **Key Predicates:**
  - `(has-lpg-infra ?u - unit)` / `(has-grid-infra ?u - unit)`
  - `(cov-tier ?u - unit ?c - coverage-tier)` and `(next-cov ?c1 ?c2 - coverage-tier)`
  - `(budget-avail ?b - budget-token)`
  - `(active-period ?p - period)` and `(next-period ?p1 ?p2 - period)`
- **Core Actions:**
  1. `build-lpg-infra(?u, ?p, ?b)`: pre `(active-period ?p) (budget-avail ?b)` → eff `(has-lpg-infra ?u) (not (budget-avail ?b))`
  2. `build-grid-infra(?u, ?p, ?b)`: pre `(active-period ?p) (budget-avail ?b)` → eff `(has-grid-infra ?u) (not (budget-avail ?b))`
  3. `rollout-lpg-subsidy(?u, ?p, ?c1, ?c2, ?b)`: pre `(has-lpg-infra ?u) (cov-tier ?u ?c1) (next-cov ?c1 ?c2) (active-period ?p) (budget-avail ?b)` → eff `(cov-tier ?u ?c2) (not (cov-tier ?u ?c1)) (not (budget-avail ?b))`
  4. `rollout-electric-cooking(?u, ?p, ?c1, ?c2, ?b)`: pre `(has-grid-infra ?u) (cov-tier ?u ?c1) (next-cov ?c1 ?c2) (active-period ?p) (budget-avail ?b)` → eff `(cov-tier ?u ?c2) (not (cov-tier ?u ?c1)) (not (budget-avail ?b))`
  5. `rollout-improved-biomass(?u, ?p, ?c1, ?c2, ?b)`: pre `(cov-tier ?u ?c1) (next-cov ?c1 ?c2) (active-period ?p) (budget-avail ?b)` *(no LPG/grid infra prerequisite)* → eff `(cov-tier ?u ?c2) (not (cov-tier ?u ?c1)) (not (budget-avail ?b))`
  6. `advance-period(?p1, ?p2)`: pre `(active-period ?p1) (next-period ?p1 ?p2)` → eff `(active-period ?p2) (not (active-period ?p1))`
- **Validation:** Verify `pyperplan` returns a valid plan reaching goal coverage tiers across `IND_urban`, `IND_rural`, `NPL_urban`, `NPL_rural`, confirming prerequisite ordering (`build-*-infra` precedes `rollout-*`).

---

## 4. Equal-Footing MDP vs. GA Comparison Protocol (RQ4)
To ensure a fair comparison between static NSGA-II and adaptive MDP:
1. Both use **identical** `baseline_unit_state`, `exposure_response_surrogate`, total budget cap $B_{\text{total}}$, adoption caps, and period costs.
2. Evaluate the GA open-loop schedule inside the **same stochastic transition simulator** (1,000 seeded rollouts) used by the MDP to measure **expected 2040 DALYs averted**, **95% rollout range**, **cost**, and **female–male ASR gap**.
3. Compare the MDP policy at matched scalarization weights $(\lambda_{\text{cost}}, \lambda_{\text{eq}})$ against the closest GA Pareto solutions in `outputs/tables/plan_mdp_vs_ga_comparison.csv` (Quantifying the **Value of Adaptive Recourse**).
