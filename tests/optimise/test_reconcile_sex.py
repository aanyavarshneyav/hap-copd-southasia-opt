"""Tests for Stage 2 sex-specific forecast reconciliation (B-11 Task 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data.ingest import WHO_GHE_SEX_SPLITS
from src.optimise.reconcile_sex import (
    _reconcile_forecasts_full_precision,
    audit_demographic_hierarchy,
    audit_sex_discrepancies,
    build_reconciled_baseline_state,
    compare_reconciliation_methods,
    reconcile_forecasts_dataframe,
    reconcile_stage2_draws,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_FC_PATH = REPO_ROOT / "data" / "processed" / "forecasts_stage3.csv"
RAW_BASE_PATH = REPO_ROOT / "data" / "processed" / "baseline_unit_state.csv"
RAW_DRAWS_PATH = REPO_ROOT / "data" / "processed" / "stage2_uncertainty_draws.csv"


def test_reconciliation_enforces_exact_sex_additivity() -> None:
    """Reconciled female + male counts must equal combined-sex both within tolerance."""
    if not RAW_FC_PATH.exists():
        pytest.skip("Real forecasts_stage3.csv not present on disk.")

    raw_fc = pd.read_csv(RAW_FC_PATH)
    audit = audit_sex_discrepancies(raw_fc)
    assert not audit.empty
    # Verify raw file indeed had the documented non-additivity
    raw_2040_bau = audit[
        (audit["year"] == 2040) & (audit["scenario"] == "bau") & (audit["measure"] == "DALYs")
    ]
    assert abs(float(raw_2040_bau["raw_num_diff"].sum()) - 50254.13) < 1.0

    rec_fc = reconcile_forecasts_dataframe(raw_fc)
    # Combined-sex 'both' must remain byte-for-byte numerically identical
    orig_both = raw_fc[raw_fc["sex"] == "both"].sort_values(
        ["location", "year", "scenario", "measure", "metric"]
    )
    rec_both = rec_fc[rec_fc["sex"] == "both"].sort_values(
        ["location", "year", "scenario", "measure", "metric"]
    )
    assert np.allclose(
        orig_both["val"].to_numpy(dtype=float),
        rec_both["val"].to_numpy(dtype=float),
        atol=1e-9,
    )

    # Female + male Number must equal both within 2-decimal CSV rounding tolerance
    piv_num = rec_fc[rec_fc["metric"] == "Number"].pivot_table(
        index=["location", "year", "scenario", "measure"], columns="sex", values="val"
    )
    max_diff = float(np.max(np.abs(piv_num["female"] + piv_num["male"] - piv_num["both"])))
    assert max_diff <= 0.02

    # Rate consistency against WPP sex populations
    for unit, country in (("IND_urban", "India"), ("NPL_rural", "Nepal")):
        f_sub = piv_num.xs((unit, 2040, "bau", "DALYs"))
        ratio = float(f_sub["female"] / f_sub["both"])
        assert np.isclose(ratio, WHO_GHE_SEX_SPLITS["DALYs"][country]["female"], atol=1e-4)

    # Reconciled baseline unit state matches 2040 DALYs sums
    if RAW_BASE_PATH.exists():
        rec_base = build_reconciled_baseline_state(pd.read_csv(RAW_BASE_PATH), rec_fc)
        b_both = rec_fc[
            (rec_fc["year"] == 2040)
            & (rec_fc["scenario"] == "bau")
            & (rec_fc["measure"] == "DALYs")
            & (rec_fc["metric"] == "Number")
            & (rec_fc["sex"] == "both")
        ]["val"].sum()
        rec_base_sum = float(
            rec_base["bau_dalys_2040_female"].sum() + rec_base["bau_dalys_2040_male"].sum()
        )
        assert abs(rec_base_sum - float(b_both)) <= 0.05

    # Reconciled uncertainty draws satisfy female + male == both per draw
    if RAW_DRAWS_PATH.exists():
        rec_draws = reconcile_stage2_draws(pd.read_csv(RAW_DRAWS_PATH), rec_fc)
        piv_d = rec_draws.pivot_table(
            index=["location", "year", "draw_id"], columns="sex", values="dalys_count_draw"
        )
        assert float(np.max(np.abs(piv_d["female"] + piv_d["male"] - piv_d["both"]))) <= 0.02


def test_b12_full_precision_and_hierarchy_comparison() -> None:
    """B-12: verify exact float64 precision before rounding and MinT/Bottom-Up comparison."""
    if not RAW_FC_PATH.exists():
        pytest.skip("Real forecasts_stage3.csv not present on disk.")

    raw_fc = pd.read_csv(RAW_FC_PATH)
    _, prec = _reconcile_forecasts_full_precision(raw_fc)
    assert prec["max_float64_pop_err"] <= 1e-9
    assert prec["max_float64_num_err"] <= 1e-8
    assert prec["max_float64_rate_weighted_err"] <= 1e-9

    comp = compare_reconciliation_methods(raw_fc)
    assert not comp.empty
    methods = set(comp["method"].unique())
    assert {
        "raw_unreconciled",
        "bottom_up_raw_gbdt",
        "ols_mint_raw_gbdt",
        "wls_dominant_both_raw_ratio",
        "top_down_who_ghe_2021",
    }.issubset(methods)
    td = comp[comp["method"] == "top_down_who_ghe_2021"]
    assert float(td["both_bias_vs_authoritative"].abs().max()) == 0.0
    assert float(td["sex_additivity_err"].abs().max()) == 0.0

    rec_fc = reconcile_forecasts_dataframe(raw_fc)
    rec_draws = reconcile_stage2_draws(pd.read_csv(RAW_DRAWS_PATH), rec_fc)
    audit_df = audit_demographic_hierarchy(
        REPO_ROOT, raw_fc, rec_fc, pd.read_csv(RAW_DRAWS_PATH), rec_draws, prec
    )
    assert len(audit_df) == 8
    h7 = audit_df[audit_df["check_id"] == "H7_country_to_regional_aggregate"].iloc[0]
    assert "NOT_APPLICABLE" in str(h7["status"])
