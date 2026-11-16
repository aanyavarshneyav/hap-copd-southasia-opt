"""Unit tests for the parameter audit tool (``src/optimise/param_audit.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.data.contracts import load_config, load_stage3_contracts
from src.optimise.param_audit import (
    audit_adoption_caps_vs_ceiling,
    audit_low_central_high,
    audit_probability_triplets,
    audit_unit_cost_recomputations,
    build_task3_audit_spec,
    compute_cost_effectiveness_and_breakeven,
    compute_full_gap_lpg_summary,
    compute_india_lpg_usd_per_hh,
    compute_nepal_lpg_usd_per_hh,
    convert_usd_per_hh_to_usd_millions,
    derive_task3_lpg_usd_per_hh_scenarios,
    format_audit_report,
    run_parameter_audit,
    validate_household_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def test_task3_usd_per_hh_formulas() -> None:
    assert compute_india_lpg_usd_per_hh(3.0) == pytest.approx((2050.0 + 3 * 300 * 5) / 83.56)
    assert compute_india_lpg_usd_per_hh(4.0) == pytest.approx((2050.0 + 4 * 300 * 5) / 83.56)
    assert compute_india_lpg_usd_per_hh(9.0) == pytest.approx((2050.0 + 9 * 300 * 5) / 83.56)

    assert compute_nepal_lpg_usd_per_hh(331.25, 2500.0) == pytest.approx(
        (331.25 * 6 * 5 + 2500.0) / 133.70
    )
    assert compute_nepal_lpg_usd_per_hh(450.0, 2500.0) == pytest.approx(
        (450.0 * 6 * 5 + 2500.0) / 133.70
    )
    assert compute_nepal_lpg_usd_per_hh(1174.0, 4500.0) == pytest.approx(
        (1174.0 * 6 * 5 + 4500.0) / 133.70
    )


def test_unit_cost_recomputation_detects_match_and_mismatch() -> None:
    ind_hh = compute_india_lpg_usd_per_hh(4.0)  # 96.33796...
    npl_hh = compute_nepal_lpg_usd_per_hh(450.0, 2500.0)  # 119.6709...
    households = {"IND_urban": 10_000_000.0, "NPL_urban": 1_000_000.0}
    unit_costs = {
        "IND_urban": {
            "lpg_subsidy": ind_hh * 10.0,  # exact match in $M
            "improved_biomass": 50.0,  # mismatch (expected 20.0)
        },
        "NPL_urban": {
            "lpg_subsidy": 999.0,  # mismatch
            "improved_biomass": 10.0,  # missing components
        },
    }
    components = {
        "IND_urban": {
            "lpg_subsidy": {"formula": "india_lpg", "refills_per_yr": 4.0},
            "improved_biomass": {
                "formula": "direct_usd_per_hh",
                "usd_per_hh": 2.0,  # 2 * 10M / 1e6 = 20.0M != 50.0M
            },
        },
        "NPL_urban": {
            "lpg_subsidy": {
                "formula": "nepal_lpg",
                "noc_loss_per_cyl_npr": 450.0,
                "upfront_npr": 2500.0,
                "stated_usd_per_hh": npl_hh,
            },
        },
    }
    checks = audit_unit_cost_recomputations(
        unit_costs,
        components,
        households,
        units=["IND_urban", "NPL_urban"],
        interventions=["lpg_subsidy", "improved_biomass"],
    )
    by_key = {(c.unit, c.intervention): c for c in checks}
    assert by_key[("IND_urban", "lpg_subsidy")].status == "match"
    assert by_key[("IND_urban", "improved_biomass")].status == "mismatch"
    assert by_key[("NPL_urban", "lpg_subsidy")].status == "mismatch"
    assert by_key[("NPL_urban", "improved_biomass")].status == "missing_components"


def test_low_central_high_and_probability_triplet_checks() -> None:
    bounds = audit_low_central_high(
        {
            "valid": {"low": 1.0, "central": 2.0, "high": 3.0},
            "equal": [5.0, 5.0, 5.0],
            "inverted": {"low": 4.0, "central": 3.0, "high": 5.0},
            "incomplete": {"low": None, "central": 2.0, "high": 3.0},
        }
    )
    b_map = {b.name: b for b in bounds}
    assert b_map["valid"].is_valid is True
    assert b_map["equal"].is_valid is True
    assert b_map["inverted"].is_valid is False
    assert b_map["inverted"].status == "order_violation"
    assert b_map["incomplete"].is_valid is False
    assert b_map["incomplete"].status == "missing_or_non_finite"

    probs = audit_probability_triplets(
        {
            "ok": [0.15, 0.70, 0.15],
            "bad_sum": [0.20, 0.70, 0.20],
            "negative": [-0.10, 0.90, 0.20],
        }
    )
    p_map = {p.name: p for p in probs}
    assert p_map["ok"].is_valid is True
    assert p_map["bad_sum"].is_valid is False
    assert p_map["bad_sum"].status == "sum_mismatch"
    assert p_map["negative"].is_valid is False
    assert p_map["negative"].status == "out_of_range"


def test_adoption_caps_vs_ceiling_flags_unreachable_units() -> None:
    caps = {
        "IND_urban": {
            "lpg_subsidy": {"P1": 0.2, "P2": 0.2, "P3": 0.2},
            "improved_biomass": {"P1": 0.25, "P2": 0.25, "P3": 0.25},
            "electric_induction": {"P1": 0.15, "P2": 0.15, "P3": 0.15},
        },
        "NPL_rural": {
            "lpg_subsidy": {"P1": 0.2, "P2": 0.2, "P3": 0.2},
            "improved_biomass": {"P1": 0.03, "P2": 0.03, "P3": 0.03},
            "electric_induction": {"P1": 0.02, "P2": 0.02, "P3": 0.02},
        },
    }
    cap_checks, interv_sums, unit_sums = audit_adoption_caps_vs_ceiling(
        units=["IND_urban", "NPL_rural"],
        baseline_clean_fuel={"IND_urban": 0.88, "NPL_rural": 0.40},
        coverage_ceiling={"IND_urban": 0.98, "NPL_rural": 0.95},
        max_adoption_per_period=caps,
        lpg_ready_init={"IND_urban": True, "NPL_rural": False},
        grid_ready_init={"IND_urban": True, "NPL_rural": False},
    )
    assert any(c.exceeds_unit_gap for c in cap_checks if c.unit == "IND_urban")

    u_map = {u.unit: u for u in unit_sums}
    assert u_map["IND_urban"].combined_can_reach is True
    # NPL_rural gap = 0.55; unlocked LPG (P2+P3=0.40) + bio (0.09) + elec (0.04) = 0.53 < 0.55
    assert u_map["NPL_rural"].combined_can_reach is False
    assert "lpg_subsidy" in u_map["NPL_rural"].unmeetable_single_interventions_unlocked

    i_map = {(i.unit, i.intervention): i for i in interv_sums}
    assert i_map[("NPL_rural", "lpg_subsidy")].can_reach_all_periods is True  # 0.60 >= 0.55
    assert i_map[("NPL_rural", "lpg_subsidy")].can_reach_unlocked is False  # 0.40 < 0.55


def test_cost_effectiveness_breakeven_and_full_gap_lpg() -> None:
    unit_costs = {
        "IND_rural": {
            "lpg_subsidy": 620.0,
            "improved_biomass": 210.0,
            "electric_induction": 780.0,
            "infra_lpg_grid": 95.0,
        },
        "IND_urban": {
            "lpg_subsidy": 320.0,
            "improved_biomass": 120.0,
            "electric_induction": 410.0,
            "infra_lpg_grid": 40.0,
        },
    }
    effectiveness = {
        "IND_rural": {
            "lpg_subsidy": 0.78,
            "improved_biomass": 0.45,
            "electric_induction": 0.85,
            "infra_lpg_grid": 0.0,
        },
        "IND_urban": {
            "lpg_subsidy": 0.88,
            "improved_biomass": 0.45,
            "electric_induction": 0.92,
            "infra_lpg_grid": 0.0,
        },
    }
    ce_rows, be_rows = compute_cost_effectiveness_and_breakeven(
        units=["IND_urban", "IND_rural"],
        unit_costs_usd_millions=unit_costs,
        effectiveness=effectiveness,
        baseline_clean_fuel={"IND_urban": 0.88, "IND_rural": 0.58},
        coverage_ceiling={"IND_urban": 0.98, "IND_rural": 0.95},
        lpg_ready_init={"IND_urban": True, "IND_rural": False},
        households={"IND_urban": 100_000_000.0, "IND_rural": 200_000_000.0},
    )
    ce_map = {(r.unit, r.intervention): r for r in ce_rows}
    assert ce_map[("IND_rural", "improved_biomass")].cost_per_eff_coverage_usd_millions == (
        pytest.approx(210.0 / 0.45)
    )
    assert ce_map[("IND_rural", "lpg_subsidy")].cost_per_eff_hh_usd == pytest.approx(
        (620.0 * 1_000_000.0 / 200_000_000.0) / 0.78
    )

    be_map = {r.unit: r for r in be_rows}
    assert be_map["IND_rural"].breakeven_vs_lpg == pytest.approx(0.78 * (210.0 / 620.0))
    assert be_map["IND_rural"].breakeven_vs_elec == pytest.approx(0.85 * (210.0 / 780.0))
    assert be_map["IND_rural"].bio_loses_to_lpg is False

    caps = {
        u: {i: {"P1": 0.2, "P2": 0.2, "P3": 0.2} for i in ("lpg_subsidy",)}
        for u in ("IND_urban", "IND_rural")
    }
    fg = compute_full_gap_lpg_summary(
        units=["IND_urban", "IND_rural"],
        baseline_clean_fuel={"IND_urban": 0.88, "IND_rural": 0.58},
        coverage_ceiling={"IND_urban": 0.98, "IND_rural": 0.95},
        unit_costs_usd_millions=unit_costs,
        max_adoption_per_period=caps,
        lpg_ready_init={"IND_urban": True, "IND_rural": False},
        budgets_usd_millions={"low": 100.0, "central": 200.0, "high": 400.0},
    )
    expected_total = (0.10 * 320.0) + (0.37 * 620.0 + 95.0)
    assert fg.total_full_gap_lpg_cost_usd_millions == pytest.approx(expected_total)
    assert fg.budget_ratios["low"] == pytest.approx(100.0 / expected_total)
    assert fg.budget_ratios["central"] == pytest.approx(200.0 / expected_total)
    assert fg.budget_ratios["high"] == pytest.approx(400.0 / expected_total)


def test_task3_candidate_audit_end_to_end() -> None:
    config = load_config(CONFIG_PATH)
    contracts = load_stage3_contracts(config, base_dir=REPO_ROOT)
    spec = build_task3_audit_spec(config, contracts, use_candidate_overrides=True)
    report = run_parameter_audit(spec)
    text = format_audit_report(report)
    assert "PARAMETER AUDIT REPORT" in text
    assert report.full_gap_lpg.total_full_gap_lpg_cost_usd_millions == pytest.approx(402.76)
    assert all(p.is_valid for p in report.prob_checks)
    assert any(c.status == "missing_households" for c in report.unit_cost_checks)


def test_candidate_params_and_allocation_shares() -> None:
    from src.optimise.evaluator import Stage3Evaluator
    from src.optimise.param_audit import build_candidate_intervention_params
    from src.optimise.scenario_runs import _allocation_shares

    config = load_config(CONFIG_PATH)
    contracts = load_stage3_contracts(config, base_dir=REPO_ROOT)
    cand = build_candidate_intervention_params(contracts.intervention_params)
    assert cand["effectiveness"]["IND_urban"]["lpg_subsidy"] == pytest.approx(0.88)
    assert cand["effectiveness"]["IND_rural"]["lpg_subsidy"] == pytest.approx(0.78)
    assert cand["effectiveness"]["NPL_urban"]["electric_induction"] == pytest.approx(0.90)
    assert cand["effectiveness"]["NPL_rural"]["electric_induction"] == pytest.approx(0.85)

    ev = Stage3Evaluator(config=config, contracts=contracts)
    x = ev.evaluate([0.0] * ev.n_vars)
    assert x.is_feasible is True
    import numpy as np

    vec = np.zeros(ev.n_vars)
    vec[0] = 0.10  # IND_urban lpg P1
    shares = _allocation_shares(vec, ev)
    assert shares["lpg_cov_sum"] == pytest.approx(0.10)
    assert shares["bio_cov_sum"] == pytest.approx(0.0)


def test_convert_usd_per_hh_to_usd_millions_keeps_tiers_separate() -> None:
    lpg_scenarios = derive_task3_lpg_usd_per_hh_scenarios()
    # Synthetic unit-scale inputs purely to test exact arithmetic (never treated as real data)
    hh = {
        "IND_urban": 10_000_000.0,
        "IND_rural": 20_000_000.0,
        "NPL_urban": 1_000_000.0,
        "NPL_rural": 2_000_000.0,
    }
    converted = convert_usd_per_hh_to_usd_millions(lpg_scenarios, hh)
    assert set(converted.keys()) == {"low", "central", "high"}
    for tier in ("low", "central", "high"):
        for unit, count in hh.items():
            assert converted[tier][unit] == pytest.approx(
                lpg_scenarios[tier][unit] * count / 1_000_000.0
            )
    assert converted["low"]["IND_urban"] < converted["central"]["IND_urban"]
    assert converted["central"]["IND_urban"] < converted["high"]["IND_urban"]


def test_validate_household_counts_rejects_missing_or_non_positive() -> None:
    with pytest.raises(ValueError, match="missing household count"):
        validate_household_counts({"IND_urban": 100.0, "IND_rural": 200.0, "NPL_urban": 50.0})

    with pytest.raises(ValueError, match="missing household count"):
        validate_household_counts(
            {
                "IND_urban": 100.0,
                "IND_rural": None,
                "NPL_urban": 50.0,
                "NPL_rural": 75.0,
            }
        )

    for bad_val in (0.0, -5.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="non-positive or non-finite"):
            validate_household_counts(
                {
                    "IND_urban": bad_val,
                    "IND_rural": 200.0,
                    "NPL_urban": 50.0,
                    "NPL_rural": 75.0,
                }
            )


def test_task4_safeguard_metadata_stamps_mock_provenance(tmp_path: Path) -> None:
    import json

    import pandas as pd
    from src.optimise.scenario_runs import write_task4_safeguard_metadata

    csv_path = tmp_path / "task4_budget_sweep_mock.csv"
    pd.DataFrame([{"scenario": "budget_25pct", "eta_biomass": 0.45}]).to_csv(csv_path, index=False)

    written = write_task4_safeguard_metadata(tmp_path, stamp_existing_csvs=True)
    assert "summary_meta" in written
    assert "task4_budget_sweep_mock.csv.meta.json" in written

    stamped_df = pd.read_csv(csv_path)
    for col in (
        "is_mock",
        "warning_mock_inputs",
        "data_version",
        "forecast_basis",
        "cost_basis",
        "effectiveness_basis",
        "budget_basis",
    ):
        assert col in stamped_df.columns
    assert bool(stamped_df["is_mock"].iloc[0]) is True

    meta = json.loads(written["task4_budget_sweep_mock.csv.meta.json"].read_text(encoding="utf-8"))
    assert meta["is_mock"] is True
    assert meta["data_version"] == "MOCK_V1_CANDIDATE_T3"
    assert "MUST NOT be treated as real-world cost estimates" in meta["budget_basis"]["disclaimer"]
