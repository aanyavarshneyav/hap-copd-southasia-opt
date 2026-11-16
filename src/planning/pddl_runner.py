"""STRIPS/PDDL runner, independent precondition/effect replay checker, and evaluator bridge.

Solves ``pddl/domain.pddl`` + ``pddl/problem_ind_npl.pddl`` with ``pyperplan``,
independently verifies every action's preconditions, effects, and goal satisfaction
via a custom STRIPS replay checker, and maps the plan into a 48-variable Stage 3
decision vector verified by :class:`src.optimise.evaluator.Stage3Evaluator`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pyperplan.planner import HEURISTICS, SEARCHES, search_plan

from src.data.contracts import load_config, load_stage3_contracts
from src.optimise.evaluator import EvaluationResult, Stage3Evaluator

_FLOAT_EPS: float = 1e-12
_DEFAULT_TIER_INCREMENT: float = 0.10


class PDDLVerificationError(ValueError):
    """Raised when a PDDL plan violates domain preconditions, types, or goal state."""


@dataclass(frozen=True)
class ActionSchema:
    """Parsed STRIPS action schema."""

    name: str
    params: tuple[tuple[str, str], ...]  # ((var_name, type_name), ...)
    pos_pre: tuple[tuple[str, ...], ...]
    neg_pre: tuple[tuple[str, ...], ...]
    add_eff: tuple[tuple[str, ...], ...]
    del_eff: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ParsedSTRIPS:
    """Parsed STRIPS domain + problem representation for independent replay checks."""

    domain_name: str
    problem_name: str
    objects_by_type: dict[str, set[str]]
    object_types: dict[str, str]
    actions: dict[str, ActionSchema]
    init_facts: frozenset[tuple[str, ...]]
    goal_facts: frozenset[tuple[str, ...]]


def _strip_comments(text: str) -> str:
    return "\n".join(line.split(";", maxsplit=1)[0] for line in text.splitlines())


def _tokenize_lisp(text: str) -> list[str]:
    clean = _strip_comments(text).lower()
    spaced = clean.replace("(", " ( ").replace(")", " ) ")
    return spaced.split()


def _parse_sexpr(tokens: list[str], pos: int = 0) -> tuple[Any, int]:
    if pos >= len(tokens):
        raise PDDLVerificationError("Unexpected end of PDDL input while parsing S-expression.")
    tok = tokens[pos]
    if tok == "(":
        items: list[Any] = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            item, pos = _parse_sexpr(tokens, pos)
            items.append(item)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise PDDLVerificationError("Unclosed '(' in PDDL input.")
        return items, pos + 1
    if tok == ")":
        raise PDDLVerificationError("Unexpected ')' in PDDL input.")
    return tok, pos + 1


def _parse_typed_list(items: list[Any]) -> list[tuple[str, str]]:
    """Parse ``a b - type1 c - type2`` into ``[(a, type1), (b, type1), (c, type2)]``."""
    out: list[tuple[str, str]] = []
    pending: list[str] = []
    idx = 0
    while idx < len(items):
        tok = items[idx]
        if tok == "-":
            if idx + 1 >= len(items):
                raise PDDLVerificationError("Dangling '-' in PDDL typed list.")
            t_name = str(items[idx + 1])
            for name in pending:
                out.append((name, t_name))
            pending.clear()
            idx += 2
        else:
            pending.append(str(tok))
            idx += 1
    for name in pending:
        out.append((name, "object"))
    return out


def _parse_literal_list(
    expr: list[Any],
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """Parse `(and ...)` or single predicate into `(positives, negatives)`."""
    if not expr:
        return (), ()
    clauses = expr[1:] if expr[0] == "and" else [expr]
    pos_list: list[tuple[str, ...]] = []
    neg_list: list[tuple[str, ...]] = []
    for cl in clauses:
        if not isinstance(cl, list) or not cl:
            raise PDDLVerificationError(f"Invalid PDDL literal clause: {cl!r}")
        if cl[0] == "not":
            if len(cl) != 2 or not isinstance(cl[1], list):
                raise PDDLVerificationError(f"Invalid negated PDDL clause: {cl!r}")
            neg_list.append(tuple(str(x) for x in cl[1]))
        else:
            pos_list.append(tuple(str(x) for x in cl))
    return tuple(pos_list), tuple(neg_list)


def parse_pddl_domain_and_problem(
    domain_path: str | Path,
    problem_path: str | Path,
) -> ParsedSTRIPS:
    """Parse STRIPS domain and problem files into an inspectable :class:`ParsedSTRIPS`."""
    d_tokens = _tokenize_lisp(Path(domain_path).read_text(encoding="utf-8"))
    d_sexpr, _ = _parse_sexpr(d_tokens)
    if not isinstance(d_sexpr, list) or d_sexpr[0] != "define":
        raise PDDLVerificationError("Domain file must start with `(define ...)`.")

    domain_name = str(d_sexpr[1][1])
    actions: dict[str, ActionSchema] = {}
    for block in d_sexpr[2:]:
        if not isinstance(block, list) or not block:
            continue
        if block[0] == ":action":
            act_name = str(block[1])
            kv: dict[str, Any] = {}
            idx = 2
            while idx < len(block):
                key = str(block[idx])
                kv[key] = block[idx + 1]
                idx += 2
            params = tuple(_parse_typed_list(kv.get(":parameters", [])))
            pos_pre, neg_pre = _parse_literal_list(kv.get(":precondition", []))
            add_eff, del_eff = _parse_literal_list(kv.get(":effect", []))
            actions[act_name] = ActionSchema(
                name=act_name,
                params=params,
                pos_pre=pos_pre,
                neg_pre=neg_pre,
                add_eff=add_eff,
                del_eff=del_eff,
            )

    p_tokens = _tokenize_lisp(Path(problem_path).read_text(encoding="utf-8"))
    p_sexpr, _ = _parse_sexpr(p_tokens)
    if not isinstance(p_sexpr, list) or p_sexpr[0] != "define":
        raise PDDLVerificationError("Problem file must start with `(define ...)`.")

    problem_name = str(p_sexpr[1][1])
    objects_by_type: dict[str, set[str]] = {}
    object_types: dict[str, str] = {}
    init_facts: set[tuple[str, ...]] = set()
    goal_facts: set[tuple[str, ...]] = set()

    for block in p_sexpr[2:]:
        if not isinstance(block, list) or not block:
            continue
        tag = str(block[0])
        if tag == ":domain" and str(block[1]) != domain_name:
            raise PDDLVerificationError(
                f"Problem domain {block[1]!r} does not match domain {domain_name!r}."
            )
        elif tag == ":objects":
            for obj_name, type_name in _parse_typed_list(block[1:]):
                objects_by_type.setdefault(type_name, set()).add(obj_name)
                object_types[obj_name] = type_name
        elif tag == ":init":
            for fact in block[1:]:
                init_facts.add(tuple(str(x) for x in fact))
        elif tag == ":goal":
            g_pos, g_neg = _parse_literal_list(block[1])
            if g_neg:
                raise PDDLVerificationError("Negative goals are not supported in STRIPS replay.")
            goal_facts.update(g_pos)

    return ParsedSTRIPS(
        domain_name=domain_name,
        problem_name=problem_name,
        objects_by_type=objects_by_type,
        object_types=object_types,
        actions=actions,
        init_facts=frozenset(init_facts),
        goal_facts=frozenset(goal_facts),
    )


def _parse_action_call(action_str: str) -> tuple[str, tuple[str, ...]]:
    cleaned = action_str.strip().lower()
    m = re.fullmatch(r"\(([^()]+)\)", cleaned)
    if not m:
        raise PDDLVerificationError(f"Malformed grounded action string: {action_str!r}")
    parts = m.group(1).split()
    return parts[0], tuple(parts[1:])


def _ground_literal(literal: tuple[str, ...], env: dict[str, str]) -> tuple[str, ...]:
    return tuple(env.get(tok, tok) for tok in literal)


def replay_and_verify_pddl_plan(
    domain_path: str | Path,
    problem_path: str | Path,
    action_sequence: list[str],
) -> set[tuple[str, ...]]:
    """Independently replay ``action_sequence`` against STRIPS preconditions and goals.

    Raises :class:`PDDLVerificationError` on unknown actions, type mismatches,
    unsatisfied preconditions, or unmet goal facts.
    """
    strips = parse_pddl_domain_and_problem(domain_path, problem_path)
    state: set[tuple[str, ...]] = set(strips.init_facts)

    for step_idx, action_str in enumerate(action_sequence, start=1):
        act_name, args = _parse_action_call(action_str)
        schema = strips.actions.get(act_name)
        if schema is None:
            raise PDDLVerificationError(f"Step {step_idx}: unknown action {act_name!r}.")
        if len(args) != len(schema.params):
            raise PDDLVerificationError(
                f"Step {step_idx} ({act_name}): expected {len(schema.params)} args, "
                f"got {len(args)}."
            )

        env: dict[str, str] = {}
        for (var_name, expected_type), arg_obj in zip(schema.params, args, strict=True):
            actual_type = strips.object_types.get(arg_obj)
            if actual_type != expected_type:
                raise PDDLVerificationError(
                    f"Step {step_idx} ({act_name}): arg {arg_obj!r} has type {actual_type!r}, "
                    f"expected {expected_type!r}."
                )
            env[var_name] = arg_obj

        for pre in schema.pos_pre:
            grounded = _ground_literal(pre, env)
            if grounded not in state:
                raise PDDLVerificationError(
                    f"Step {step_idx} {action_str}: missing precondition {grounded}."
                )
        for pre in schema.neg_pre:
            grounded = _ground_literal(pre, env)
            if grounded in state:
                raise PDDLVerificationError(
                    f"Step {step_idx} {action_str}: violated negative precondition {grounded}."
                )

        for eff in schema.del_eff:
            state.discard(_ground_literal(eff, env))
        for eff in schema.add_eff:
            state.add(_ground_literal(eff, env))

    missing_goals = sorted(strips.goal_facts - state)
    if missing_goals:
        raise PDDLVerificationError(f"Plan finished with unsatisfied goal(s): {missing_goals}.")
    return state


def plan_to_decision_vector(
    action_sequence: list[str],
    evaluator: Stage3Evaluator,
    *,
    tier_increment: float = _DEFAULT_TIER_INCREMENT,
) -> np.ndarray:
    """Convert a verified PDDL action sequence into a ``(48,)`` Stage 3 decision vector."""
    unit_map = {u.lower(): u for u in evaluator.units}
    period_map = {p.lower(): p for p in evaluator.periods}

    tensor = np.zeros(
        (evaluator.n_units, evaluator.n_interventions, evaluator.n_periods),
        dtype=float,
    )
    headroom = np.maximum(0.0, evaluator.effective_adoption_headroom() - _FLOAT_EPS)

    for action_str in action_sequence:
        act_name, args = _parse_action_call(action_str)
        if act_name == "advance-period":
            continue

        unit_key = args[0]
        if unit_key not in unit_map:
            raise ValueError(f"Unknown unit {unit_key!r} in PDDL action {action_str!r}.")
        u_idx = evaluator.units.index(unit_map[unit_key])

        if act_name in ("build-lpg-infra", "build-grid-infra"):
            p_idx = evaluator.periods.index(period_map[args[1]])
            tensor[u_idx, evaluator._idx_infra, p_idx] = evaluator.infra_build_threshold
        elif act_name in (
            "rollout-lpg-init-ready",
            "rollout-electric-init-ready",
            "rollout-improved-biomass",
        ):
            p_idx = evaluator.periods.index(period_map[args[1]])
            if act_name == "rollout-lpg-init-ready":
                i_idx = evaluator._idx_lpg
            elif act_name == "rollout-electric-init-ready":
                i_idx = evaluator._idx_elec
            else:
                i_idx = evaluator._idx_bio
            used = float(np.sum(tensor[u_idx, evaluator._non_infra_indices, :]))
            rem = max(0.0, float(headroom[u_idx]) - used)
            tensor[u_idx, i_idx, p_idx] += min(tier_increment, rem)
        elif act_name in ("rollout-lpg-after-infra", "rollout-electric-after-infra"):
            p_idx = evaluator.periods.index(period_map[args[2]])
            i_idx = (
                evaluator._idx_lpg if act_name == "rollout-lpg-after-infra" else evaluator._idx_elec
            )
            used = float(np.sum(tensor[u_idx, evaluator._non_infra_indices, :]))
            rem = max(0.0, float(headroom[u_idx]) - used)
            tensor[u_idx, i_idx, p_idx] += min(tier_increment, rem)
        else:
            raise ValueError(f"Unrecognized PDDL action {act_name!r} in evaluator mapping.")

    return tensor.reshape(-1)


def solve_pddl(
    domain_path: str | Path,
    problem_path: str | Path,
    *,
    search_name: str = "astar",
    heuristic_name: str = "hff",
) -> tuple[list[str], float]:
    """Run ``pyperplan`` and return ``(action_names, runtime_seconds)``."""
    if search_name not in SEARCHES:
        raise ValueError(f"Unknown pyperplan search {search_name!r}; allowed={list(SEARCHES)}")
    if heuristic_name not in HEURISTICS:
        raise ValueError(
            f"Unknown pyperplan heuristic {heuristic_name!r}; allowed={list(HEURISTICS)}"
        )

    t0 = time.perf_counter()
    solution = search_plan(
        str(domain_path),
        str(problem_path),
        SEARCHES[search_name],
        HEURISTICS[heuristic_name],
    )
    dt = time.perf_counter() - t0
    if solution is None:
        raise RuntimeError(f"pyperplan found no solution for {problem_path}.")
    return [str(op.name) for op in solution], float(dt)


def run_pddl_pipeline(
    config_path: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Solve STRIPS PDDL problem, replay-verify, check via Stage3Evaluator, and save trace."""
    root = Path(base_dir) if base_dir is not None else Path(".")
    config = load_config(config_path)
    contracts = load_stage3_contracts(config, base_dir=root)
    evaluator = Stage3Evaluator(config=config, contracts=contracts)

    pddl_cfg = config.get("planning", {}).get("pddl", {})
    search_name = str(pddl_cfg.get("search", "astar"))
    heuristic_name = str(pddl_cfg.get("heuristic", "hff"))

    pddl_dir = root / config["paths"]["pddl_dir"]
    domain_path = pddl_dir / "domain.pddl"
    problem_path = pddl_dir / "problem_ind_npl.pddl"

    tables_dir = root / config["paths"]["outputs_tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    actions, solve_s = solve_pddl(
        domain_path,
        problem_path,
        search_name=search_name,
        heuristic_name=heuristic_name,
    )
    final_state = replay_and_verify_pddl_plan(domain_path, problem_path, actions)
    x_vec = plan_to_decision_vector(actions, evaluator)
    ev_res: EvaluationResult = evaluator.evaluate(x_vec)

    if not bool(ev_res.is_feasible):
        raise ValueError(
            "Converted PDDL plan failed Stage3Evaluator constraints: "
            f"budget={ev_res.violation_budget}, cap={ev_res.violation_adoption_cap}, "
            f"ceiling={ev_res.violation_coverage_ceiling}, infra={ev_res.violation_infra_prereq}"
        )

    suffix = "_mock" if evaluator.is_mock else ""
    out_txt = tables_dir / f"plan_pddl_solution{suffix}.txt"
    out_meta = Path(f"{out_txt}.meta.json")
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"# PDDL Plan Solution ({'SYNTHETIC MOCK DATA' if evaluator.is_mock else 'SCENARIO DATA'})",
        (
            f"# Generated: {now_iso} | data_version={evaluator.data_version} "
            f"| is_mock={evaluator.is_mock}"
        ),
        f"# Planner: pyperplan ({search_name} + {heuristic_name}) | solve_time={solve_s:.4f}s",
        f"# Replay Verification: PASSED ({len(final_state)} final facts)",
        (
            "# Stage3Evaluator Check: "
            f"is_feasible={ev_res.is_feasible}, "
            f"dalys_2040_mean={float(ev_res.dalys_2040_mean):,.2f} "
            f"[{float(ev_res.dalys_2040_lower):,.2f}, {float(ev_res.dalys_2040_upper):,.2f}], "
            f"cost=${float(ev_res.total_cost_usd_millions):.2f}M, "
            f"equity_gap_asr={float(ev_res.equity_gap_asr):.2f}"
        ),
        "",
    ]
    for idx, act in enumerate(actions, start=1):
        lines.append(f"{idx:02d}: {act}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_meta.write_text(
        json.dumps(
            {
                "artifact": out_txt.name,
                "source": "src.planning.pddl_runner",
                "generated_at": now_iso,
                "is_mock": evaluator.is_mock,
                "warning_mock_inputs": evaluator.warning_mock_inputs,
                "data_version": evaluator.data_version,
                "planner": "pyperplan",
                "search": search_name,
                "heuristic": heuristic_name,
                "solve_seconds": solve_s,
                "n_steps": len(actions),
                "replay_verified": True,
                "evaluator_feasible": bool(ev_res.is_feasible),
                "dalys_2040_mean": float(ev_res.dalys_2040_mean),
                "total_cost_usd_millions": float(ev_res.total_cost_usd_millions),
                "equity_gap_asr": float(ev_res.equity_gap_asr),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "actions": actions,
        "solve_seconds": solve_s,
        "x_vec": x_vec,
        "evaluation": ev_res,
        "files": {"pddl_solution_txt": out_txt, "pddl_solution_meta": out_meta},
    }
