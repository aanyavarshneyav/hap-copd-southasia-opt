"""Unit tests for deterministic mock fixture generation (``src/data/mock.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.data.contracts import load_config, load_stage3_contracts
from src.data.mock import generate_mock_fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "mock.yaml"


def test_mock_generator_is_byte_deterministic(tmp_path: Path) -> None:
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    out_a = generate_mock_fixtures(CONFIG_PATH, base_dir=dir_a)
    out_b = generate_mock_fixtures(CONFIG_PATH, base_dir=dir_b)

    assert set(out_a.keys()) == set(out_b.keys())
    for key in out_a:
        assert out_a[key].read_bytes() == out_b[key].read_bytes()


def test_generated_mocks_validate_and_carry_mock_flags(tmp_path: Path) -> None:
    out = generate_mock_fixtures(CONFIG_PATH, base_dir=tmp_path)
    config = load_config(CONFIG_PATH)
    bundle = load_stage3_contracts(config, base_dir=tmp_path)

    assert bundle.is_mock is True
    assert bool(bundle.forecasts_stage3["is_mock"].all()) is True
    assert bool(bundle.baseline_unit_state["is_mock"].all()) is True
    assert bundle.surrogate["is_mock"] is True
    assert bundle.intervention_params["is_mock"] is True

    for meta_key in ("forecasts_stage3_mock_meta", "baseline_unit_state_mock_meta"):
        meta = json.loads(out[meta_key].read_text(encoding="utf-8"))
        assert meta["is_mock"] is True
        assert "SYNTHETIC" in meta["warning"]


def test_mock_forecasts_years_and_sex_specific_population(tmp_path: Path) -> None:
    generate_mock_fixtures(CONFIG_PATH, base_dir=tmp_path)
    config = load_config(CONFIG_PATH)
    bundle = load_stage3_contracts(config, base_dir=tmp_path)
    df = bundle.forecasts_stage3

    bau_years = set(df.loc[df["scenario"] == "bau", "year"].unique().tolist())
    assert bau_years == set(range(2025, 2041))

    for cf in ("cf_25", "cf_50", "cf_75", "cf_100"):
        cf_years = set(df.loc[df["scenario"] == cf, "year"].unique().tolist())
        assert cf_years == {2025, 2030, 2035, 2040}

    sub = df[
        (df["year"] == 2040)
        & (df["scenario"] == "bau")
        & (df["measure"] == "DALYs")
        & (df["metric"] == "Number")
    ]
    piv = sub.pivot(index="location", columns="sex", values="population")
    assert ((piv["female"] + piv["male"] - piv["both"]).abs() < 1.0).all()

    # Committed mock files in repository must also validate cleanly
    repo_bundle = load_stage3_contracts(config, base_dir=REPO_ROOT)
    assert repo_bundle.is_mock is True
    assert isinstance(repo_bundle.forecasts_stage3, pd.DataFrame)
