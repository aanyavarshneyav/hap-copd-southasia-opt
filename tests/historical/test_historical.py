"""Smoke tests for Stage 1 Historical Analysis."""

from pathlib import Path

import pandas as pd
import pytest
from src.historical.metrics import (
    compute_eapc,
    compute_historical_summary,
    compute_pwlf_joinpoints,
)
from src.historical.run import run_stage1


@pytest.fixture
def sample_burden_data() -> pd.DataFrame:
    """Load cached interim burden parquet for testing."""
    p = Path("data/interim/historical_burden.parquet")
    if not p.exists():
        pytest.skip("Interim burden parquet not found")
    return pd.read_parquet(p)


def test_historical_summary_schema(sample_burden_data: pd.DataFrame) -> None:
    """Verify summary table structure and expected columns."""
    df_sum = compute_historical_summary(sample_burden_data)
    assert not df_sum.empty
    expected_cols = {
        "location",
        "measure",
        "sex",
        "count_1990",
        "count_2023",
        "count_pct_change",
        "asr_1990",
        "asr_2023",
        "asr_pct_change",
    }
    assert expected_cols.issubset(df_sum.columns)
    # Check that India and Nepal are present
    assert "India" in df_sum["location"].values
    assert "Nepal" in df_sum["location"].values


def test_eapc_computation(sample_burden_data: pd.DataFrame) -> None:
    """Verify EAPC calculation and 95% confidence intervals."""
    df_eapc = compute_eapc(sample_burden_data, 1990, 2023, "1990-2023")
    assert not df_eapc.empty
    assert "eapc" in df_eapc.columns
    assert "eapc_lower" in df_eapc.columns
    assert "eapc_upper" in df_eapc.columns

    # CI invariant: lower <= eapc <= upper
    assert (df_eapc["eapc_lower"] <= df_eapc["eapc"] + 1e-6).all()
    assert (df_eapc["eapc"] <= df_eapc["eapc_upper"] + 1e-6).all()


def test_pwlf_joinpoints(sample_burden_data: pd.DataFrame) -> None:
    """Verify piecewise linear joinpoint detection."""
    df_pwlf = compute_pwlf_joinpoints(sample_burden_data, locations=["India", "Nepal"])
    assert not df_pwlf.empty
    assert "breakpoint_years" in df_pwlf.columns
    assert "segment_apcs" in df_pwlf.columns


def test_stage1_pipeline_run() -> None:
    """Verify that Stage 1 pipeline runs end-to-end and outputs files."""
    results = run_stage1("configs/default.yaml")
    assert Path(results["summary_table"]).exists()
    assert Path(results["eapc_table"]).exists()
    assert Path(results["shares_table"]).exists()
    for fig in results["figures"]:
        assert Path(fig).exists()
        assert Path(fig).stat().st_size > 1000  # Non-trivial PNG
