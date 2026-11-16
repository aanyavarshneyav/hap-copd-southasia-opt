"""Parameter audit tool and candidate parameter builder for Stage 3 & Stage 4.

Implements automated checks (a)-(f) for parameter sourcing consistency:
(a) Recompute unit costs as USD/HH x households from stated components and flag mismatches.
(b) Check Low <= Central <= High ordering across bounded parameters.
(c) Check probability triplets sum to 1.0 (with entries in [0, 1]).
(d) Check each per-period adoption cap against (ceiling - baseline) and report units
    where the sum of caps (overall and infra-unlocked) cannot reach the ceiling.
(e) Compute cost per effective household (cost / eta) for each unit and intervention,
    plus break-even eta_biomass against lpg_subsidy and electric_induction.
(f) Compute the cost of closing every unit's gap to coverage_ceiling with LPG (plus
    infra where not ready) and the ratio total_budget / full_gap_lpg_cost across
    Low / Central / High budgets.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.contracts import Stage3Contracts, load_config, load_stage3_contracts
from src.optimise.evaluator import NON_INFRA_INTERVENTIONS, Stage3Evaluator

COOKING_INTERVENTIONS: tuple[str, ...] = NON_INFRA_INTERVENTIONS
DEFAULT_PERIODS: tuple[str, ...] = ("P1", "P2", "P3")
ALLOCATION_UNITS: tuple[str, ...] = ("IND_urban", "IND_rural", "NPL_urban", "NPL_rural")
SCENARIO_TIERS: tuple[str, ...] = ("low", "central", "high")


@dataclass(frozen=True)
class UnitCostCheck:
    """Result of recomputing one (unit, intervention) cost from stated components."""

    unit: str
    intervention: str
    status: str  # "match", "mismatch", "missing_components", "missing_households"
    recomputed_usd_per_hh: float | None
    stated_usd_per_hh: float | None
    households: float | None
    expected_usd_millions: float | None
    actual_usd_millions: float | None
    abs_diff_usd_millions: float | None
    rel_diff: float | None
    note: str = ""


@dataclass(frozen=True)
class BoundsCheck:
    """Result of checking Low <= Central <= High for one parameter."""

    name: str
    low: float | None
    central: float | None
    high: float | None
    is_valid: bool
    status: str  # "ok", "order_violation", "missing_or_non_finite"


@dataclass(frozen=True)
class ProbTripletCheck:
    """Result of checking one probability triplet sums to 1.0."""

    name: str
    probs: tuple[float, ...]
    total: float
    is_valid: bool
    status: str


@dataclass(frozen=True)
class AdoptionCapCheck:
    """Per-(unit, intervention, period) adoption cap check vs remaining gap."""

    unit: str
    intervention: str
    period: str
    cap: float
    unit_gap: float
    unlocked: bool
    exceeds_unit_gap: bool
    in_unit_interval: bool


@dataclass(frozen=True)
class UnitInterventionCapSummary:
    """Sum of adoption caps vs unit coverage gap for one (unit, intervention)."""

    unit: str
    intervention: str
    unit_gap: float
    sum_all_periods: float
    sum_unlocked_periods: float
    can_reach_all_periods: bool
    can_reach_unlocked: bool


@dataclass(frozen=True)
class UnitCapSummary:
    """Combined portfolio reachability summary for one unit."""

    unit: str
    baseline: float
    ceiling: float
    unit_gap: float
    combined_all_periods: float
    combined_unlocked_periods: float
    combined_can_reach: bool
    unmeetable_single_interventions: tuple[str, ...]
    unmeetable_single_interventions_unlocked: tuple[str, ...]


@dataclass(frozen=True)
class CostEffectivenessRow:
    """Cost per effective unit of coverage and effective household."""

    unit: str
    intervention: str
    unit_cost_usd_millions: float
    eta: float
    cost_per_eff_coverage_usd_millions: float
    usd_per_hh: float | None
    cost_per_eff_hh_usd: float | None


@dataclass(frozen=True)
class BreakEvenRow:
    """Break-even eta_biomass at which improved_biomass loses to LPG or induction."""

    unit: str
    eta_biomass_current: float
    eta_lpg: float
    eta_elec: float
    cost_bio_usd_millions: float
    cost_lpg_usd_millions: float
    cost_elec_usd_millions: float
    breakeven_vs_lpg: float
    breakeven_vs_elec: float
    breakeven_vs_lpg_with_infra: float | None
    bio_loses_to_lpg: bool
    bio_loses_to_elec: bool


@dataclass(frozen=True)
class UnitFullGapLPGCost:
    """Cost breakdown to close one unit's gap to ceiling using LPG + infra."""

    unit: str
    baseline: float
    ceiling: float
    gap: float
    lpg_unit_cost_usd_millions: float
    lpg_cost_usd_millions: float
    lpg_infra_ready_init: bool
    infra_cost_usd_millions: float
    total_unit_cost_usd_millions: float
    lpg_unlocked_cap_sum: float
    lpg_caps_can_close_gap: bool


@dataclass(frozen=True)
class FullGapLPGSummary:
    """System-wide full-gap LPG cost and budget coverage ratios."""

    per_unit: tuple[UnitFullGapLPGCost, ...]
    total_full_gap_lpg_cost_usd_millions: float
    budget_ratios: dict[str, float | None]
    budgets_usd_millions: dict[str, float | None]


@dataclass(frozen=True)
class AuditReport:
    """Full parameter audit report covering checks (a)-(f)."""

    unit_cost_checks: tuple[UnitCostCheck, ...]
    bounds_checks: tuple[BoundsCheck, ...]
    prob_checks: tuple[ProbTripletCheck, ...]
    cap_checks: tuple[AdoptionCapCheck, ...]
    intervention_cap_summaries: tuple[UnitInterventionCapSummary, ...]
    unit_cap_summaries: tuple[UnitCapSummary, ...]
    cost_effectiveness: tuple[CostEffectivenessRow, ...]
    breakevens: tuple[BreakEvenRow, ...]
    full_gap_lpg: FullGapLPGSummary

    def to_dict(self) -> dict[str, Any]:
        """Serialize report into a JSON-compatible dictionary."""
        return asdict(self)


def compute_india_lpg_usd_per_hh(
    refills_per_yr: float,
    *,
    upfront_inr: float = 2050.0,
    cost_per_refill_inr: float = 300.0,
    years: float = 5.0,
    fx_inr_per_usd: float = 83.56,
) -> float:
    """Compute 5-year India LPG subsidy cost per household in USD."""
    return (
        float(upfront_inr) + float(refills_per_yr) * float(cost_per_refill_inr) * float(years)
    ) / float(fx_inr_per_usd)


def compute_nepal_lpg_usd_per_hh(
    noc_loss_per_cyl_npr: float,
    upfront_npr: float,
    *,
    cylinders_per_yr: float = 6.0,
    years: float = 5.0,
    fx_npr_per_usd: float = 133.70,
) -> float:
    """Compute 5-year Nepal LPG subsidy cost per household in USD."""
    return (
        float(noc_loss_per_cyl_npr) * float(cylinders_per_yr) * float(years) + float(upfront_npr)
    ) / float(fx_npr_per_usd)


def compute_component_usd_per_hh(spec: Mapping[str, Any]) -> float | None:
    """Recompute USD/HH from a structured component specification dictionary."""
    formula = spec.get("formula")
    if formula == "india_lpg":
        return compute_india_lpg_usd_per_hh(
            float(spec["refills_per_yr"]),
            upfront_inr=float(spec.get("upfront_inr", 2050.0)),
            cost_per_refill_inr=float(spec.get("cost_per_refill_inr", 300.0)),
            years=float(spec.get("years", 5.0)),
            fx_inr_per_usd=float(spec.get("fx_inr_per_usd", 83.56)),
        )
    if formula == "nepal_lpg":
        return compute_nepal_lpg_usd_per_hh(
            float(spec["noc_loss_per_cyl_npr"]),
            float(spec["upfront_npr"]),
            cylinders_per_yr=float(spec.get("cylinders_per_yr", 6.0)),
            years=float(spec.get("years", 5.0)),
            fx_npr_per_usd=float(spec.get("fx_npr_per_usd", 133.70)),
        )
    if formula == "linear_local":
        upfront = float(spec.get("upfront_local", 0.0))
        annual = float(spec.get("annual_local", 0.0))
        years = float(spec.get("years", 5.0))
        fx = float(spec["fx_local_per_usd"])
        return (upfront + annual * years) / fx
    if formula == "direct_usd_per_hh":
        val = spec.get("usd_per_hh")
        return None if val is None else float(val)
    return None


def _is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def validate_household_counts(
    household_counts: Mapping[str, float | int | None],
    *,
    units: Sequence[str] = ALLOCATION_UNITS,
) -> dict[str, float]:
    """Validate that positive, finite household counts are supplied for all units."""
    missing: list[str] = []
    invalid: list[str] = []
    validated: dict[str, float] = {}

    for unit in units:
        if unit not in household_counts or household_counts[unit] is None:
            missing.append(unit)
            continue
        raw = household_counts[unit]
        if not _is_finite_number(raw) or float(raw) <= 0.0:
            invalid.append(f"{unit}={raw!r}")
            continue
        validated[unit] = float(raw)

    if missing or invalid:
        parts: list[str] = []
        if missing:
            parts.append(f"missing household count(s) for {missing}")
        if invalid:
            parts.append(f"non-positive or non-finite household count(s): {invalid}")
        raise ValueError("Invalid household counts: " + "; ".join(parts) + ".")

    return validated


def derive_task3_lpg_usd_per_hh_scenarios() -> dict[str, dict[str, float]]:
    """Return Task 3 LPG 5-year USD/HH values separated by Low / Central / High scenario."""
    ind_low = compute_india_lpg_usd_per_hh(3.0)
    ind_cen = compute_india_lpg_usd_per_hh(4.0)
    ind_high = compute_india_lpg_usd_per_hh(9.0)
    return {
        "low": {
            "IND_urban": ind_low,
            "IND_rural": ind_low,
            "NPL_urban": compute_nepal_lpg_usd_per_hh(331.25, 2500.0),
            "NPL_rural": compute_nepal_lpg_usd_per_hh(331.25, 4500.0),
        },
        "central": {
            "IND_urban": ind_cen,
            "IND_rural": ind_cen,
            "NPL_urban": compute_nepal_lpg_usd_per_hh(450.0, 2500.0),
            "NPL_rural": compute_nepal_lpg_usd_per_hh(450.0, 4500.0),
        },
        "high": {
            "IND_urban": ind_high,
            "IND_rural": ind_high,
            "NPL_urban": compute_nepal_lpg_usd_per_hh(1174.0, 2500.0),
            "NPL_rural": compute_nepal_lpg_usd_per_hh(1174.0, 4500.0),
        },
    }


def convert_usd_per_hh_to_usd_millions(
    usd_per_hh_scenarios: Mapping[str, Mapping[str, float]],
    household_counts: Mapping[str, float | int | None],
    *,
    units: Sequence[str] = ALLOCATION_UNITS,
    tiers: Sequence[str] = SCENARIO_TIERS,
) -> dict[str, dict[str, float]]:
    """Convert per-household USD costs into USD millions keeping Low/Central/High separate.

    Formula per unit and tier: ``unit_cost_usd_millions = usd_per_hh * household_count / 1e6``.
    """
    validated_hh = validate_household_counts(household_counts, units=units)
    converted: dict[str, dict[str, float]] = {}

    for tier in tiers:
        if tier not in usd_per_hh_scenarios:
            raise ValueError(f"Missing scenario tier {tier!r} in usd_per_hh_scenarios.")
        tier_map = usd_per_hh_scenarios[tier]
        converted[tier] = {}
        for unit in units:
            if unit not in tier_map or not _is_finite_number(tier_map[unit]):
                raise ValueError(f"Missing or non-finite USD/HH for tier={tier!r}, unit={unit!r}.")
            usd_hh = float(tier_map[unit])
            if usd_hh < 0.0:
                raise ValueError(f"Negative USD/HH ({usd_hh}) for tier={tier!r}, unit={unit!r}.")
            converted[tier][unit] = usd_hh * validated_hh[unit] / 1_000_000.0

    return converted


def audit_unit_cost_recomputations(
    unit_costs_usd_millions: Mapping[str, Mapping[str, float]],
    cost_components: Mapping[str, Mapping[str, Mapping[str, Any]]] | None,
    households: Mapping[str, float | None] | None,
    units: Sequence[str],
    interventions: Sequence[str],
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-2,
) -> tuple[UnitCostCheck, ...]:
    """Check (a): recompute unit costs as USD/HH x households and flag mismatches."""
    comp_map = cost_components or {}
    hh_map = households or {}
    results: list[UnitCostCheck] = []

    for unit in units:
        hh_raw = hh_map.get(unit)
        hh_val = float(hh_raw) if _is_finite_number(hh_raw) and float(hh_raw) > 0.0 else None

        for interv in interventions:
            actual_raw = unit_costs_usd_millions.get(unit, {}).get(interv)
            actual_m = float(actual_raw) if _is_finite_number(actual_raw) else None
            spec = comp_map.get(unit, {}).get(interv)

            if not isinstance(spec, Mapping):
                results.append(
                    UnitCostCheck(
                        unit=unit,
                        intervention=interv,
                        status="missing_components",
                        recomputed_usd_per_hh=None,
                        stated_usd_per_hh=None,
                        households=hh_val,
                        expected_usd_millions=None,
                        actual_usd_millions=actual_m,
                        abs_diff_usd_millions=None,
                        rel_diff=None,
                        note="No stated cost components provided.",
                    )
                )
                continue

            # Fixed regional infra cost (not scaled by unit households unless capex/unserved HH)
            if spec.get("formula") == "fixed_usd_millions":
                expected_m = float(spec["fixed_usd_millions"])
                recomp_hh = None
                stated_hh = None
            elif spec.get("formula") == "crore_inr_to_usd_millions":
                fx = float(spec.get("fx_inr_per_usd", 83.56))
                expected_m = (float(spec["crore_inr"]) * 1e7) / fx / 1e6
                recomp_hh = None
                stated_hh = None
            elif spec.get("formula") == "infra_unserved_hh":
                unserved_raw = spec.get("unserved_households")
                unserved = float(unserved_raw) if _is_finite_number(unserved_raw) else None
                capex_hh = float(spec["capex_usd_per_hh"])
                recomp_hh = capex_hh
                stated_raw = spec.get("stated_usd_per_hh")
                stated_hh = float(stated_raw) if _is_finite_number(stated_raw) else None
                expected_m = (capex_hh * unserved / 1e6) if unserved is not None else None
            else:
                recomp_hh = compute_component_usd_per_hh(spec)
                stated_raw = spec.get("stated_usd_per_hh")
                stated_hh = float(stated_raw) if _is_finite_number(stated_raw) else None
                if recomp_hh is None:
                    results.append(
                        UnitCostCheck(
                            unit=unit,
                            intervention=interv,
                            status="missing_components",
                            recomputed_usd_per_hh=None,
                            stated_usd_per_hh=stated_hh,
                            households=hh_val,
                            expected_usd_millions=None,
                            actual_usd_millions=actual_m,
                            abs_diff_usd_millions=None,
                            rel_diff=None,
                            note="Unsupported or incomplete component formula.",
                        )
                    )
                    continue
                if hh_val is None:
                    note = "Missing unit households to scale USD/HH -> USD millions."
                    if stated_hh is not None and abs(recomp_hh - stated_hh) > max(
                        1e-4, 1e-4 * abs(recomp_hh)
                    ):
                        note += (
                            f" Stated USD/HH ({stated_hh:.4f}) != "
                            f"recomputed USD/HH ({recomp_hh:.4f})."
                        )
                    results.append(
                        UnitCostCheck(
                            unit=unit,
                            intervention=interv,
                            status="missing_households",
                            recomputed_usd_per_hh=recomp_hh,
                            stated_usd_per_hh=stated_hh,
                            households=None,
                            expected_usd_millions=None,
                            actual_usd_millions=actual_m,
                            abs_diff_usd_millions=None,
                            rel_diff=None,
                            note=note,
                        )
                    )
                    continue
                expected_m = recomp_hh * hh_val / 1_000_000.0

            if expected_m is None or actual_m is None:
                results.append(
                    UnitCostCheck(
                        unit=unit,
                        intervention=interv,
                        status="missing_households" if expected_m is None else "mismatch",
                        recomputed_usd_per_hh=recomp_hh,
                        stated_usd_per_hh=stated_hh,
                        households=hh_val,
                        expected_usd_millions=expected_m,
                        actual_usd_millions=actual_m,
                        abs_diff_usd_millions=None,
                        rel_diff=None,
                        note="Missing expected or actual USD millions value.",
                    )
                )
                continue

            abs_diff = abs(actual_m - expected_m)
            rel_diff = abs_diff / abs(expected_m) if expected_m != 0.0 else abs_diff
            hh_mismatch = (
                stated_hh is not None
                and recomp_hh is not None
                and abs(stated_hh - recomp_hh) > max(1e-3, 1e-3 * abs(recomp_hh))
            )
            is_mismatch = (abs_diff > max(abs_tol, rel_tol * abs(expected_m))) or hh_mismatch
            note = ""
            if hh_mismatch and stated_hh is not None and recomp_hh is not None:
                note = f"Stated USD/HH ({stated_hh:.4f}) != recomputed ({recomp_hh:.4f})."
            elif is_mismatch:
                note = f"Actual ${actual_m:.4f}M != recomputed ${expected_m:.4f}M."

            results.append(
                UnitCostCheck(
                    unit=unit,
                    intervention=interv,
                    status="mismatch" if is_mismatch else "match",
                    recomputed_usd_per_hh=recomp_hh,
                    stated_usd_per_hh=stated_hh,
                    households=hh_val,
                    expected_usd_millions=expected_m,
                    actual_usd_millions=actual_m,
                    abs_diff_usd_millions=abs_diff,
                    rel_diff=rel_diff,
                    note=note,
                )
            )

    return tuple(results)


def audit_low_central_high(
    bounds_map: Mapping[str, Mapping[str, Any] | Sequence[Any]],
    *,
    tol: float = 1e-12,
) -> tuple[BoundsCheck, ...]:
    """Check (b): verify Low <= Central <= High for every bounded parameter."""
    checks: list[BoundsCheck] = []
    for name, entry in bounds_map.items():
        if isinstance(entry, Mapping):
            raw_low, raw_cent, raw_high = (
                entry.get("low"),
                entry.get("central"),
                entry.get("high"),
            )
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            if len(entry) != 3:
                checks.append(
                    BoundsCheck(
                        name=name,
                        low=None,
                        central=None,
                        high=None,
                        is_valid=False,
                        status="missing_or_non_finite",
                    )
                )
                continue
            raw_low, raw_cent, raw_high = entry[0], entry[1], entry[2]
        else:
            checks.append(
                BoundsCheck(
                    name=name,
                    low=None,
                    central=None,
                    high=None,
                    is_valid=False,
                    status="missing_or_non_finite",
                )
            )
            continue

        if not (
            _is_finite_number(raw_low)
            and _is_finite_number(raw_cent)
            and _is_finite_number(raw_high)
        ):
            checks.append(
                BoundsCheck(
                    name=name,
                    low=float(raw_low) if _is_finite_number(raw_low) else None,
                    central=float(raw_cent) if _is_finite_number(raw_cent) else None,
                    high=float(raw_high) if _is_finite_number(raw_high) else None,
                    is_valid=False,
                    status="missing_or_non_finite",
                )
            )
            continue

        low, central, high = float(raw_low), float(raw_cent), float(raw_high)
        ok = (low <= central + tol) and (central <= high + tol)
        checks.append(
            BoundsCheck(
                name=name,
                low=low,
                central=central,
                high=high,
                is_valid=ok,
                status="ok" if ok else "order_violation",
            )
        )
    return tuple(checks)


def audit_probability_triplets(
    triplets_map: Mapping[str, Sequence[float]],
    *,
    tol: float = 1e-6,
) -> tuple[ProbTripletCheck, ...]:
    """Check (c): verify probability triplets contain 3 values in [0, 1] summing to 1."""
    checks: list[ProbTripletCheck] = []
    for name, raw_probs in triplets_map.items():
        probs_list = list(raw_probs)
        if len(probs_list) != 3 or not all(_is_finite_number(p) for p in probs_list):
            checks.append(
                ProbTripletCheck(
                    name=name,
                    probs=(),
                    total=float("nan"),
                    is_valid=False,
                    status="invalid_length_or_non_finite",
                )
            )
            continue
        probs = tuple(float(p) for p in probs_list)
        total = float(sum(probs))
        in_range = all(0.0 <= p <= 1.0 for p in probs)
        sums_to_one = abs(total - 1.0) <= tol
        ok = in_range and sums_to_one
        if not in_range:
            status = "out_of_range"
        elif not sums_to_one:
            status = "sum_mismatch"
        else:
            status = "ok"
        checks.append(
            ProbTripletCheck(
                name=name,
                probs=probs,
                total=total,
                is_valid=ok,
                status=status,
            )
        )
    return tuple(checks)


def audit_adoption_caps_vs_ceiling(
    units: Sequence[str],
    baseline_clean_fuel: Mapping[str, float],
    coverage_ceiling: Mapping[str, float],
    max_adoption_per_period: Mapping[str, Mapping[str, Mapping[str, float]]],
    lpg_ready_init: Mapping[str, bool],
    grid_ready_init: Mapping[str, bool],
    *,
    periods: Sequence[str] = DEFAULT_PERIODS,
    tol: float = 1e-12,
) -> tuple[
    tuple[AdoptionCapCheck, ...],
    tuple[UnitInterventionCapSummary, ...],
    tuple[UnitCapSummary, ...],
]:
    """Check (d): compare per-period caps and cap sums against (ceiling - baseline)."""
    cap_checks: list[AdoptionCapCheck] = []
    interv_summaries: list[UnitInterventionCapSummary] = []
    unit_summaries: list[UnitCapSummary] = []

    for unit in units:
        base_val = float(baseline_clean_fuel[unit])
        ceil_val = float(coverage_ceiling[unit])
        gap = max(0.0, ceil_val - base_val)

        combined_all = 0.0
        combined_unlocked = 0.0
        unmeetable_all: list[str] = []
        unmeetable_unlocked: list[str] = []

        for interv in COOKING_INTERVENTIONS:
            p_caps = max_adoption_per_period[unit][interv]
            sum_all = 0.0
            sum_unlocked = 0.0

            for p_idx, p_id in enumerate(periods):
                cap = float(p_caps[p_id])
                if p_idx == 0 and interv == "lpg_subsidy":
                    unlocked = bool(lpg_ready_init[unit])
                elif p_idx == 0 and interv == "electric_induction":
                    unlocked = bool(grid_ready_init[unit])
                else:
                    unlocked = True

                cap_checks.append(
                    AdoptionCapCheck(
                        unit=unit,
                        intervention=interv,
                        period=p_id,
                        cap=cap,
                        unit_gap=gap,
                        unlocked=unlocked,
                        exceeds_unit_gap=cap > gap + tol,
                        in_unit_interval=0.0 <= cap <= 1.0,
                    )
                )
                sum_all += cap
                if unlocked:
                    sum_unlocked += cap

            combined_all += sum_all
            combined_unlocked += sum_unlocked
            can_all = sum_all + tol >= gap
            can_unlocked = sum_unlocked + tol >= gap
            if not can_all:
                unmeetable_all.append(interv)
            if not can_unlocked:
                unmeetable_unlocked.append(interv)

            interv_summaries.append(
                UnitInterventionCapSummary(
                    unit=unit,
                    intervention=interv,
                    unit_gap=gap,
                    sum_all_periods=sum_all,
                    sum_unlocked_periods=sum_unlocked,
                    can_reach_all_periods=can_all,
                    can_reach_unlocked=can_unlocked,
                )
            )

        unit_summaries.append(
            UnitCapSummary(
                unit=unit,
                baseline=base_val,
                ceiling=ceil_val,
                unit_gap=gap,
                combined_all_periods=combined_all,
                combined_unlocked_periods=combined_unlocked,
                combined_can_reach=(combined_unlocked + tol >= gap),
                unmeetable_single_interventions=tuple(unmeetable_all),
                unmeetable_single_interventions_unlocked=tuple(unmeetable_unlocked),
            )
        )

    return tuple(cap_checks), tuple(interv_summaries), tuple(unit_summaries)


def compute_cost_effectiveness_and_breakeven(
    units: Sequence[str],
    unit_costs_usd_millions: Mapping[str, Mapping[str, float]],
    effectiveness: Mapping[str, Mapping[str, float]],
    baseline_clean_fuel: Mapping[str, float],
    coverage_ceiling: Mapping[str, float],
    lpg_ready_init: Mapping[str, bool],
    *,
    households: Mapping[str, float | None] | None = None,
    usd_per_hh: Mapping[str, Mapping[str, float | None]] | None = None,
) -> tuple[tuple[CostEffectivenessRow, ...], tuple[BreakEvenRow, ...]]:
    """Check (e): compute cost/eta and break-even eta_biomass vs LPG & induction."""
    hh_map = households or {}
    per_hh_map = usd_per_hh or {}
    ce_rows: list[CostEffectivenessRow] = []
    be_rows: list[BreakEvenRow] = []

    for unit in units:
        hh_raw = hh_map.get(unit)
        hh_val = float(hh_raw) if _is_finite_number(hh_raw) and float(hh_raw) > 0.0 else None

        for interv in COOKING_INTERVENTIONS:
            cost_m = float(unit_costs_usd_millions[unit][interv])
            eta = float(effectiveness[unit][interv])
            cost_per_eff_m = cost_m / eta if eta > 0.0 else math.inf

            direct_hh_raw = per_hh_map.get(unit, {}).get(interv)
            if _is_finite_number(direct_hh_raw):
                c_hh: float | None = float(direct_hh_raw)
            elif hh_val is not None:
                c_hh = cost_m * 1_000_000.0 / hh_val
            else:
                c_hh = None

            c_eff_hh = (c_hh / eta if eta > 0.0 else math.inf) if c_hh is not None else None
            ce_rows.append(
                CostEffectivenessRow(
                    unit=unit,
                    intervention=interv,
                    unit_cost_usd_millions=cost_m,
                    eta=eta,
                    cost_per_eff_coverage_usd_millions=cost_per_eff_m,
                    usd_per_hh=c_hh,
                    cost_per_eff_hh_usd=c_eff_hh,
                )
            )

        eta_bio = float(effectiveness[unit]["improved_biomass"])
        eta_lpg = float(effectiveness[unit]["lpg_subsidy"])
        eta_elec = float(effectiveness[unit]["electric_induction"])
        cost_bio = float(unit_costs_usd_millions[unit]["improved_biomass"])
        cost_lpg = float(unit_costs_usd_millions[unit]["lpg_subsidy"])
        cost_elec = float(unit_costs_usd_millions[unit]["electric_induction"])
        cost_infra = float(unit_costs_usd_millions[unit]["infra_lpg_grid"])

        be_lpg = eta_lpg * (cost_bio / cost_lpg) if cost_lpg > 0.0 else math.inf
        be_elec = eta_elec * (cost_bio / cost_elec) if cost_elec > 0.0 else math.inf

        gap = max(0.0, float(coverage_ceiling[unit]) - float(baseline_clean_fuel[unit]))
        if not bool(lpg_ready_init[unit]) and gap > 0.0:
            denom = cost_lpg * gap + cost_infra
            be_lpg_infra: float | None = eta_lpg * (cost_bio * gap) / denom if denom > 0.0 else None
        else:
            be_lpg_infra = None

        be_rows.append(
            BreakEvenRow(
                unit=unit,
                eta_biomass_current=eta_bio,
                eta_lpg=eta_lpg,
                eta_elec=eta_elec,
                cost_bio_usd_millions=cost_bio,
                cost_lpg_usd_millions=cost_lpg,
                cost_elec_usd_millions=cost_elec,
                breakeven_vs_lpg=be_lpg,
                breakeven_vs_elec=be_elec,
                breakeven_vs_lpg_with_infra=be_lpg_infra,
                bio_loses_to_lpg=eta_bio < be_lpg,
                bio_loses_to_elec=eta_bio < be_elec,
            )
        )

    return tuple(ce_rows), tuple(be_rows)


def compute_full_gap_lpg_summary(
    units: Sequence[str],
    baseline_clean_fuel: Mapping[str, float],
    coverage_ceiling: Mapping[str, float],
    unit_costs_usd_millions: Mapping[str, Mapping[str, float]],
    max_adoption_per_period: Mapping[str, Mapping[str, Mapping[str, float]]],
    lpg_ready_init: Mapping[str, bool],
    budgets_usd_millions: Mapping[str, float | None],
    *,
    periods: Sequence[str] = DEFAULT_PERIODS,
) -> FullGapLPGSummary:
    """Check (f): cost of closing every gap with LPG + infra and budget ratios."""
    rows: list[UnitFullGapLPGCost] = []
    total_cost = 0.0

    for unit in units:
        base_val = float(baseline_clean_fuel[unit])
        ceil_val = float(coverage_ceiling[unit])
        gap = max(0.0, ceil_val - base_val)
        lpg_uc = float(unit_costs_usd_millions[unit]["lpg_subsidy"])
        lpg_cost = gap * lpg_uc
        ready = bool(lpg_ready_init[unit])
        infra_cost = 0.0 if ready else float(unit_costs_usd_millions[unit]["infra_lpg_grid"])
        unit_total = lpg_cost + infra_cost
        total_cost += unit_total

        p_caps = max_adoption_per_period[unit]["lpg_subsidy"]
        unlocked_sum = sum(
            float(p_caps[p_id]) for p_idx, p_id in enumerate(periods) if (p_idx > 0 or ready)
        )
        rows.append(
            UnitFullGapLPGCost(
                unit=unit,
                baseline=base_val,
                ceiling=ceil_val,
                gap=gap,
                lpg_unit_cost_usd_millions=lpg_uc,
                lpg_cost_usd_millions=lpg_cost,
                lpg_infra_ready_init=ready,
                infra_cost_usd_millions=infra_cost,
                total_unit_cost_usd_millions=unit_total,
                lpg_unlocked_cap_sum=unlocked_sum,
                lpg_caps_can_close_gap=(unlocked_sum + 1e-12 >= gap),
            )
        )

    ratios: dict[str, float | None] = {}
    clean_budgets: dict[str, float | None] = {}
    for tier, b_raw in budgets_usd_millions.items():
        if _is_finite_number(b_raw):
            b_val = float(b_raw)
            clean_budgets[tier] = b_val
            ratios[tier] = (b_val / total_cost) if total_cost > 0.0 else math.inf
        else:
            clean_budgets[tier] = None
            ratios[tier] = None

    return FullGapLPGSummary(
        per_unit=tuple(rows),
        total_full_gap_lpg_cost_usd_millions=total_cost,
        budget_ratios=ratios,
        budgets_usd_millions=clean_budgets,
    )


def run_parameter_audit(spec: Mapping[str, Any]) -> AuditReport:
    """Run all checks (a)-(f) on a structured parameter audit specification."""
    units = [str(u) for u in spec["units"]]
    interventions = [str(i) for i in spec["interventions"]]
    periods = [str(p) for p in spec.get("periods", DEFAULT_PERIODS)]

    unit_cost_checks = audit_unit_cost_recomputations(
        spec["unit_costs_usd_millions"],
        spec.get("cost_components"),
        spec.get("households"),
        units,
        interventions,
    )
    bounds_checks = audit_low_central_high(spec.get("bounds", {}))
    prob_checks = audit_probability_triplets(spec.get("probability_triplets", {}))
    cap_checks, interv_caps, unit_caps = audit_adoption_caps_vs_ceiling(
        units,
        spec["baseline_clean_fuel"],
        spec["coverage_ceiling"],
        spec["max_adoption_per_period"],
        spec["lpg_ready_init"],
        spec["grid_ready_init"],
        periods=periods,
    )
    ce_rows, be_rows = compute_cost_effectiveness_and_breakeven(
        units,
        spec["unit_costs_usd_millions"],
        spec["effectiveness"],
        spec["baseline_clean_fuel"],
        spec["coverage_ceiling"],
        spec["lpg_ready_init"],
        households=spec.get("households"),
        usd_per_hh=spec.get("usd_per_hh"),
    )
    full_gap = compute_full_gap_lpg_summary(
        units,
        spec["baseline_clean_fuel"],
        spec["coverage_ceiling"],
        spec["unit_costs_usd_millions"],
        spec["max_adoption_per_period"],
        spec["lpg_ready_init"],
        spec.get("budgets_usd_millions", {}),
        periods=periods,
    )
    return AuditReport(
        unit_cost_checks=unit_cost_checks,
        bounds_checks=bounds_checks,
        prob_checks=prob_checks,
        cap_checks=cap_checks,
        intervention_cap_summaries=interv_caps,
        unit_cap_summaries=unit_caps,
        cost_effectiveness=ce_rows,
        breakevens=be_rows,
        full_gap_lpg=full_gap,
    )


def build_candidate_intervention_params(base_params: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the Task 3 candidate parameter overrides onto a base Contract 4 payload.

    Only applies the verified Task 3 changes:
    - Stack-adjusted Central effectiveness for ``lpg_subsidy`` and ``electric_induction``.
    - Preserves ``unit_costs_usd_millions``, ``coverage_ceiling``, ``max_adoption_per_period``,
      and ``total_budget_usd_millions`` from ``base_params`` (unit households and non-LPG
      USD/HH are not yet sourced to scale USD/HH into USD millions).
    """
    params = copy.deepcopy(dict(base_params))
    params["data_version"] = f"{base_params.get('data_version', 'MOCK_V1')}_CANDIDATE_T3"

    lpg_eta = {
        "IND_urban": 0.88,
        "IND_rural": 0.78,
        "NPL_urban": 0.86,
        "NPL_rural": 0.78,
    }
    elec_eta = {
        "IND_urban": 0.92,
        "IND_rural": 0.85,
        "NPL_urban": 0.90,
        "NPL_rural": 0.85,
    }
    for unit, val in lpg_eta.items():
        params["effectiveness"][unit]["lpg_subsidy"] = val
    for unit, val in elec_eta.items():
        params["effectiveness"][unit]["electric_induction"] = val

    return params


def build_task3_audit_spec(
    config: Mapping[str, Any],
    contracts: Stage3Contracts,
    *,
    use_candidate_overrides: bool = True,
) -> dict[str, Any]:
    """Assemble an audit specification from config, Stage 3 contracts, and Task 3 formulas."""
    evaluator = Stage3Evaluator(config=dict(config), contracts=contracts)
    params = (
        build_candidate_intervention_params(contracts.intervention_params)
        if use_candidate_overrides
        else copy.deepcopy(contracts.intervention_params)
    )

    baseline_cf = {u: float(evaluator._cf_bau_2040[idx]) for idx, u in enumerate(evaluator.units)}
    lpg_ready = {
        u: bool(evaluator._lpg_infra_init[idx] >= 1.0) for idx, u in enumerate(evaluator.units)
    }
    grid_ready = {
        u: bool(evaluator._grid_ready_init[idx] >= 1.0) for idx, u in enumerate(evaluator.units)
    }

    ind_lpg_low = compute_india_lpg_usd_per_hh(3.0)
    ind_lpg_cen = compute_india_lpg_usd_per_hh(4.0)
    ind_lpg_high = compute_india_lpg_usd_per_hh(9.0)

    npl_u_low = compute_nepal_lpg_usd_per_hh(331.25, 2500.0)
    npl_u_cen = compute_nepal_lpg_usd_per_hh(450.0, 2500.0)
    npl_u_high = compute_nepal_lpg_usd_per_hh(1174.0, 2500.0)

    npl_r_low = compute_nepal_lpg_usd_per_hh(331.25, 4500.0)
    npl_r_cen = compute_nepal_lpg_usd_per_hh(450.0, 4500.0)
    npl_r_high = compute_nepal_lpg_usd_per_hh(1174.0, 4500.0)

    cost_components: dict[str, dict[str, dict[str, Any]]] = {}
    usd_per_hh: dict[str, dict[str, float | None]] = {}
    bounds: dict[str, dict[str, float | None]] = {}

    if use_candidate_overrides:
        cost_components = {
            "IND_urban": {
                "lpg_subsidy": {
                    "formula": "india_lpg",
                    "refills_per_yr": 4.0,
                    "upfront_inr": 2050.0,
                    "cost_per_refill_inr": 300.0,
                    "years": 5.0,
                    "fx_inr_per_usd": 83.56,
                }
            },
            "IND_rural": {
                "lpg_subsidy": {
                    "formula": "india_lpg",
                    "refills_per_yr": 4.0,
                    "upfront_inr": 2050.0,
                    "cost_per_refill_inr": 300.0,
                    "years": 5.0,
                    "fx_inr_per_usd": 83.56,
                }
            },
            "NPL_urban": {
                "lpg_subsidy": {
                    "formula": "nepal_lpg",
                    "noc_loss_per_cyl_npr": 450.0,
                    "upfront_npr": 2500.0,
                    "cylinders_per_yr": 6.0,
                    "years": 5.0,
                    "fx_npr_per_usd": 133.70,
                }
            },
            "NPL_rural": {
                "lpg_subsidy": {
                    "formula": "nepal_lpg",
                    "noc_loss_per_cyl_npr": 450.0,
                    "upfront_npr": 4500.0,
                    "cylinders_per_yr": 6.0,
                    "years": 5.0,
                    "fx_npr_per_usd": 133.70,
                }
            },
        }
        usd_per_hh = {
            "IND_urban": {"lpg_subsidy": ind_lpg_cen},
            "IND_rural": {"lpg_subsidy": ind_lpg_cen},
            "NPL_urban": {"lpg_subsidy": npl_u_cen},
            "NPL_rural": {"lpg_subsidy": npl_r_cen},
        }
        bounds = {
            "usd_per_hh.IND_urban.lpg_subsidy": {
                "low": ind_lpg_low,
                "central": ind_lpg_cen,
                "high": ind_lpg_high,
            },
            "usd_per_hh.IND_rural.lpg_subsidy": {
                "low": ind_lpg_low,
                "central": ind_lpg_cen,
                "high": ind_lpg_high,
            },
            "usd_per_hh.NPL_urban.lpg_subsidy": {
                "low": npl_u_low,
                "central": npl_u_cen,
                "high": npl_u_high,
            },
            "usd_per_hh.NPL_rural.lpg_subsidy": {
                "low": npl_r_low,
                "central": npl_r_cen,
                "high": npl_r_high,
            },
            "india_refills_per_yr": {"low": 3.0, "central": 4.0, "high": 9.0},
            "nepal_noc_loss_per_cyl_npr": {"low": 331.25, "central": 450.0, "high": 1174.0},
            "crore_200_inr_usd_millions": {
                "low": None,
                "central": 23.9,
                "high": None,
            },
        }

    prob_triplets = {
        "planning.mdp.transition_probs": list(
            config.get("planning", {}).get("mdp", {}).get("transition_probs", [0.15, 0.70, 0.15])
        )
    }

    central_budget = float(params["total_budget_usd_millions"])
    budgets = {"low": None, "central": central_budget, "high": None}

    return {
        "units": evaluator.units,
        "interventions": evaluator.interventions,
        "periods": evaluator.periods,
        "unit_costs_usd_millions": params["unit_costs_usd_millions"],
        "cost_components": cost_components,
        "households": {u: None for u in evaluator.units},
        "usd_per_hh": usd_per_hh,
        "effectiveness": params["effectiveness"],
        "baseline_clean_fuel": baseline_cf,
        "coverage_ceiling": params["coverage_ceiling"],
        "max_adoption_per_period": params["max_adoption_per_period"],
        "lpg_ready_init": lpg_ready,
        "grid_ready_init": grid_ready,
        "bounds": bounds,
        "probability_triplets": prob_triplets,
        "budgets_usd_millions": budgets,
    }


def format_audit_report(report: AuditReport) -> str:
    """Format an :class:`AuditReport` into a concise human-readable text report."""
    lines: list[str] = []
    lines.append("=== PARAMETER AUDIT REPORT ===")

    lines.append("\n(a) Unit Cost Component Recomputations:")
    for c in report.unit_cost_checks:
        hh_str = f"{c.recomputed_usd_per_hh:.2f}" if c.recomputed_usd_per_hh is not None else "n/a"
        exp_str = (
            f"${c.expected_usd_millions:.2f}M" if c.expected_usd_millions is not None else "n/a"
        )
        act_str = f"${c.actual_usd_millions:.2f}M" if c.actual_usd_millions is not None else "n/a"
        lines.append(
            f"  - {c.unit:10s} | {c.intervention:18s} | {c.status:18s} | "
            f"USD/HH={hh_str} | exp={exp_str} vs act={act_str}" + (f" ({c.note})" if c.note else "")
        )

    lines.append("\n(b) Low <= Central <= High Checks:")
    if not report.bounds_checks:
        lines.append("  (none provided)")
    for b in report.bounds_checks:
        lines.append(f"  - {b.name}: {b.status} ({b.low}, {b.central}, {b.high})")

    lines.append("\n(c) Probability Triplets:")
    for p in report.prob_checks:
        lines.append(f"  - {p.name}: {p.status} probs={p.probs} sum={p.total:.6f}")

    lines.append("\n(d) Adoption Caps vs (Ceiling - Baseline):")
    for u in report.unit_cap_summaries:
        lines.append(
            f"  - {u.unit:10s}: base={u.baseline:.2f}, ceil={u.ceiling:.2f}, "
            f"gap={u.unit_gap:.2f} | combined_unlocked={u.combined_unlocked_periods:.2f} "
            f"(ok={u.combined_can_reach}) | "
            f"single_unmeetable_unlocked={list(u.unmeetable_single_interventions_unlocked)}"
        )

    lines.append("\n(e) Cost per Effective Unit & Biomass Break-Even eta:")
    for ce in report.cost_effectiveness:
        hh_eff = (
            f"${ce.cost_per_eff_hh_usd:.2f}/HH" if ce.cost_per_eff_hh_usd is not None else "n/a"
        )
        lines.append(
            f"  - {ce.unit:10s} | {ce.intervention:18s} | cost=${ce.unit_cost_usd_millions:.2f}M | "
            f"eta={ce.eta:.3f} | cost/eta=${ce.cost_per_eff_coverage_usd_millions:.2f}M | "
            f"USD/eff-HH={hh_eff}"
        )
    for be in report.breakevens:
        infra_str = (
            f"{be.breakeven_vs_lpg_with_infra:.4f}"
            if be.breakeven_vs_lpg_with_infra is not None
            else "n/a"
        )
        lines.append(
            f"  * Break-even {be.unit:10s}: eta_bio={be.eta_biomass_current:.3f} | "
            f"be_vs_lpg={be.breakeven_vs_lpg:.4f} (loses={be.bio_loses_to_lpg}) | "
            f"be_vs_elec={be.breakeven_vs_elec:.4f} (loses={be.bio_loses_to_elec}) | "
            f"be_vs_lpg_with_infra={infra_str}"
        )

    lines.append("\n(f) Full-Gap LPG Cost + Infra:")
    fg = report.full_gap_lpg
    for u in fg.per_unit:
        lines.append(
            f"  - {u.unit:10s}: gap={u.gap:.2f} * ${u.lpg_unit_cost_usd_millions:.1f}M "
            f"+ infra=${u.infra_cost_usd_millions:.1f}M = ${u.total_unit_cost_usd_millions:.2f}M "
            f"(lpg_caps_close={u.lpg_caps_can_close_gap})"
        )
    lines.append(
        f"  Total full-gap LPG+infra cost: ${fg.total_full_gap_lpg_cost_usd_millions:.2f}M"
    )
    for tier, ratio in fg.budget_ratios.items():
        b_val = fg.budgets_usd_millions.get(tier)
        r_str = f"{ratio:.4f}" if ratio is not None else "n/a"
        b_str = f"${b_val:.2f}M" if b_val is not None else "n/a"
        lines.append(f"  Budget [{tier}]: {b_str} -> ratio={r_str}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m src.optimise.param_audit``."""
    parser = argparse.ArgumentParser(description="Audit Stage 3/4 intervention parameter sets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Apply Task 3 candidate overrides before auditing.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path to save structured JSON audit report.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    contracts = load_stage3_contracts(config)
    spec = build_task3_audit_spec(
        config,
        contracts,
        use_candidate_overrides=bool(args.candidate),
    )
    report = run_parameter_audit(spec)
    print(format_audit_report(report))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
