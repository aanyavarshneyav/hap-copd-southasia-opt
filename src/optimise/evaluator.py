"""Vectorized NumPy surrogate evaluator for Stage 3 multi-objective optimisation.

Evaluates single ``(48,)`` or batched ``(N, 48)`` decision vectors over:
``4 units x 4 interventions x 3 periods = 48`` variables.

All parameters (costs, effectiveness, adoption caps, coverage ceilings, budget,
discounting, and surrogate elasticities) are loaded via ``src.data.contracts`` from
config and contract files — never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.contracts import (
    ContractValidationError,
    Stage3Contracts,
    load_config,
    load_stage3_contracts,
)

NON_INFRA_INTERVENTIONS: tuple[str, ...] = (
    "lpg_subsidy",
    "improved_biomass",
    "electric_induction",
)
INFRA_INTERVENTION: str = "infra_lpg_grid"


@dataclass(frozen=True)
class EvaluationResult:
    """Outputs and provenance flags from evaluating one or more decision vectors."""

    dalys_2040_mean: float | np.ndarray
    dalys_2040_lower: float | np.ndarray
    dalys_2040_upper: float | np.ndarray
    total_cost_usd_millions: float | np.ndarray
    equity_gap_asr: float | np.ndarray
    equity_gap_dalys_abs: float | np.ndarray
    violation_budget: float | np.ndarray
    violation_adoption_cap: float | np.ndarray
    violation_coverage_ceiling: float | np.ndarray
    violation_infra_prereq: float | np.ndarray
    is_feasible: bool | np.ndarray
    is_mock: bool
    warning_mock_inputs: bool
    data_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert scalar or batch result into a JSON-serializable dictionary."""

        def _conv(val: float | bool | np.ndarray) -> Any:
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.floating, np.bool_)):
                return val.item()
            return val

        return {
            "dalys_2040_mean": _conv(self.dalys_2040_mean),
            "dalys_2040_lower": _conv(self.dalys_2040_lower),
            "dalys_2040_upper": _conv(self.dalys_2040_upper),
            "total_cost_usd_millions": _conv(self.total_cost_usd_millions),
            "equity_gap_asr": _conv(self.equity_gap_asr),
            "equity_gap_dalys_abs": _conv(self.equity_gap_dalys_abs),
            "violation_budget": _conv(self.violation_budget),
            "violation_adoption_cap": _conv(self.violation_adoption_cap),
            "violation_coverage_ceiling": _conv(self.violation_coverage_ceiling),
            "violation_infra_prereq": _conv(self.violation_infra_prereq),
            "is_feasible": _conv(self.is_feasible),
            "is_mock": self.is_mock,
            "warning_mock_inputs": self.warning_mock_inputs,
            "data_version": self.data_version,
        }


def build_variable_names(
    units: list[str],
    interventions: list[str],
    periods: list[str],
) -> tuple[str, ...]:
    """Return ordered C-contiguous variable names for the flat ``U * I * P`` vector."""
    return tuple(f"{u}__{i}__{p}" for u in units for i in interventions for p in periods)


def vector_to_tensor(
    x: np.ndarray,
    n_units: int,
    n_interventions: int,
    n_periods: int,
) -> np.ndarray:
    """Map flat vector(s) ``(48,)`` or ``(N, 48)`` to ``(..., U, I, P)`` in C-order."""
    arr = np.asarray(x, dtype=float)
    expected_dim = n_units * n_interventions * n_periods
    if arr.ndim == 1:
        if arr.shape[0] != expected_dim:
            raise ValueError(
                f"Expected flat vector of length {expected_dim}, got shape {arr.shape}."
            )
        return arr.reshape(n_units, n_interventions, n_periods)
    if arr.ndim == 2:
        if arr.shape[1] != expected_dim:
            raise ValueError(f"Expected batch shape (N, {expected_dim}), got shape {arr.shape}.")
        return arr.reshape(arr.shape[0], n_units, n_interventions, n_periods)
    raise ValueError(f"Expected 1D or 2D decision array, got {arr.ndim}D shape {arr.shape}.")


def tensor_to_vector(tensor: np.ndarray) -> np.ndarray:
    """Flatten ``(U, I, P)`` or ``(N, U, I, P)`` back into ``(48,)`` or ``(N, 48)``."""
    arr = np.asarray(tensor, dtype=float)
    if arr.ndim == 3:
        return arr.reshape(-1)
    if arr.ndim == 4:
        return arr.reshape(arr.shape[0], -1)
    raise ValueError(f"Expected 3D or 4D tensor, got {arr.ndim}D shape {arr.shape}.")


def _extract_bau_2040_arrays(
    forecasts_df: pd.DataFrame,
    units: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract 2040 BAU DALYs (val, lower, upper), ASRs (val), and pop weights."""
    bau_2040 = forecasts_df[
        (forecasts_df["year"] == 2040)
        & (forecasts_df["scenario"] == "bau")
        & (forecasts_df["measure"] == "DALYs")
    ]
    n_u = len(units)
    dalys_val = np.zeros((n_u, 2), dtype=float)  # col 0=female, col 1=male
    dalys_low = np.zeros((n_u, 2), dtype=float)
    dalys_up = np.zeros((n_u, 2), dtype=float)
    asr_val = np.zeros((n_u, 2), dtype=float)
    pop_total_2040 = np.zeros(n_u, dtype=float)
    cf_bau_2040 = np.zeros(n_u, dtype=float)

    for u_idx, unit in enumerate(units):
        for s_idx, sex in enumerate(("female", "male")):
            num_row = bau_2040[
                (bau_2040["location"] == unit)
                & (bau_2040["sex"] == sex)
                & (bau_2040["metric"] == "Number")
            ]
            rate_row = bau_2040[
                (bau_2040["location"] == unit)
                & (bau_2040["sex"] == sex)
                & (bau_2040["metric"] == "Rate_per_100k")
            ]
            if num_row.empty or rate_row.empty:
                raise ContractValidationError(
                    f"Missing 2040 BAU DALYs rows for ({unit}, {sex}) in forecasts_stage3."
                )
            dalys_val[u_idx, s_idx] = float(num_row["val"].iloc[0])
            dalys_low[u_idx, s_idx] = float(num_row["lower"].iloc[0])
            dalys_up[u_idx, s_idx] = float(num_row["upper"].iloc[0])
            asr_val[u_idx, s_idx] = float(rate_row["val"].iloc[0])

        both_row = bau_2040[
            (bau_2040["location"] == unit)
            & (bau_2040["sex"] == "both")
            & (bau_2040["metric"] == "Number")
        ]
        if both_row.empty:
            raise ContractValidationError(
                f"Missing 2040 BAU 'both' population row for {unit} in forecasts_stage3."
            )
        pop_total_2040[u_idx] = float(both_row["population"].iloc[0])
        cf_bau_2040[u_idx] = float(both_row["clean_fuel_share"].iloc[0])

    pop_weights = pop_total_2040 / np.sum(pop_total_2040)
    return dalys_val, dalys_low, dalys_up, asr_val, pop_weights, cf_bau_2040


class Stage3Evaluator:
    """Vectorized evaluator for 2040 DALYs, cost, equity gap, and constraints."""

    def __init__(self, config: dict[str, Any], contracts: Stage3Contracts) -> None:
        self.units: list[str] = [str(u) for u in config["geography"]["allocation_units"]]
        self.interventions: list[str] = [str(i) for i in config["interventions"]]
        self.periods: list[str] = [str(p["id"]) for p in config["time"]["decision_periods"]]

        self.n_units = len(self.units)
        self.n_interventions = len(self.interventions)
        self.n_periods = len(self.periods)
        self.n_vars = self.n_units * self.n_interventions * self.n_periods
        self.variable_names = build_variable_names(self.units, self.interventions, self.periods)

        self.is_mock: bool = contracts.is_mock
        self.warning_mock_inputs: bool = contracts.is_mock
        self.data_version: str = str(contracts.surrogate["data_version"])

        # Intervention indices
        self._idx_lpg = self.interventions.index("lpg_subsidy")
        self._idx_bio = self.interventions.index("improved_biomass")
        self._idx_elec = self.interventions.index("electric_induction")
        self._idx_infra = self.interventions.index(INFRA_INTERVENTION)
        self._non_infra_indices = np.array(
            [self._idx_lpg, self._idx_bio, self._idx_elec], dtype=int
        )

        # Extract 2040 BAU burden & population weights
        (
            self._bau_dalys_val,
            self._bau_dalys_low,
            self._bau_dalys_up,
            self._bau_asr_val,
            self._pop_weights,
            self._cf_bau_2040,
        ) = _extract_bau_2040_arrays(contracts.forecasts_stage3, self.units)

        # Initial infra flags from Contract 2
        baseline_indexed = contracts.baseline_unit_state.set_index("location")
        self._lpg_infra_init = np.array(
            [bool(baseline_indexed.loc[u, "lpg_infra_ready_init"]) for u in self.units],
            dtype=float,
        )
        self._grid_ready_init = np.array(
            [bool(baseline_indexed.loc[u, "grid_ready_init"]) for u in self.units],
            dtype=float,
        )

        # Surrogate parameters from Contract 3
        surr_units = contracts.surrogate["units"]
        self._slope_male = np.array(
            [float(surr_units[u]["slope_clean_fuel"]) for u in self.units], dtype=float
        )
        sex_ratio = np.array(
            [float(surr_units[u]["sex_elasticity_ratio_f_to_m"]) for u in self.units],
            dtype=float,
        )
        self._slope_female = self._slope_male * sex_ratio
        self._surrogate_sd = np.array(
            [float(surr_units[u]["uncertainty_sd"]) for u in self.units], dtype=float
        )

        # Intervention & constraint parameters from Contract 4
        params = contracts.intervention_params
        self.total_budget_usd_millions = float(params["total_budget_usd_millions"])
        self.infra_build_threshold = float(params["infra_build_threshold"])
        self.apply_cost_discounting = bool(params["apply_cost_discounting"])
        self.discount_factor = float(params["discount_factor"])

        self._period_discount = np.array(
            [
                (self.discount_factor**p_idx) if self.apply_cost_discounting else 1.0
                for p_idx in range(self.n_periods)
            ],
            dtype=float,
        )
        self._coverage_ceiling = np.array(
            [float(params["coverage_ceiling"][u]) for u in self.units], dtype=float
        )

        self._effectiveness = np.zeros((self.n_units, self.n_interventions), dtype=float)
        self._unit_costs = np.zeros((self.n_units, self.n_interventions), dtype=float)
        self._max_adoption = np.zeros(
            (self.n_units, self.n_interventions, self.n_periods), dtype=float
        )
        for u_idx, u in enumerate(self.units):
            for i_idx, interv in enumerate(self.interventions):
                self._effectiveness[u_idx, i_idx] = float(params["effectiveness"][u][interv])
                self._unit_costs[u_idx, i_idx] = float(params["unit_costs_usd_millions"][u][interv])
                for p_idx, p_id in enumerate(self.periods):
                    self._max_adoption[u_idx, i_idx, p_idx] = float(
                        params["max_adoption_per_period"][u][interv][p_id]
                    )

        self._non_infra_effectiveness = self._effectiveness[:, self._non_infra_indices]
        self._non_infra_costs = self._unit_costs[:, self._non_infra_indices]
        self._non_infra_max_adoption = self._max_adoption[:, self._non_infra_indices, :]
        self._infra_costs = self._unit_costs[:, self._idx_infra]

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        base_dir: str | Path | None = None,
    ) -> Stage3Evaluator:
        """Instantiate evaluator by loading config and validating all Stage 3 contracts."""
        config = load_config(config_path)
        contracts = load_stage3_contracts(config, base_dir=base_dir)
        return cls(config=config, contracts=contracts)

    def evaluate_effective_gains(
        self,
        eff_gain_u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate 2040 DALYs (mean, lower, upper) and equity gaps from effective gains.

        Args:
            eff_gain_u: Array of shape ``(U,)`` or ``(N, U)`` containing effective
                clean-cooking exposure gains above 2040 BAU per unit.

        Returns:
            Tuple ``(dalys_mean, dalys_lower, dalys_upper, equity_gap_asr, equity_gap_dalys_abs)``
            each with shape ``(N,)`` (or ``(1,)`` if 1D input).
        """
        arr = np.asarray(eff_gain_u, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.shape[1] != self.n_units:
            raise ValueError(
                f"Expected eff_gain_u shape (N, {self.n_units}) or "
                f"({self.n_units},), got {arr.shape}."
            )

        max_headroom = np.maximum(0.0, 1.0 - self._cf_bau_2040)[np.newaxis, :]
        delta_c = np.minimum(np.maximum(0.0, arr), max_headroom)  # (N, U)

        mult_female = np.exp(self._slope_female[np.newaxis, :] * delta_c)
        mult_male = np.exp(self._slope_male[np.newaxis, :] * delta_c)
        mult = np.stack([mult_female, mult_male], axis=-1)  # (N, U, 2)

        bau_val = self._bau_dalys_val[np.newaxis, :, :]  # (1, U, 2)
        bau_low = self._bau_dalys_low[np.newaxis, :, :]
        bau_up = self._bau_dalys_up[np.newaxis, :, :]

        dalys_mean_us = mult * bau_val
        bau_lo_hw = mult * (bau_val - bau_low)
        bau_hi_hw = mult * (bau_up - bau_val)
        surr_hw = dalys_mean_us * (
            1.96 * self._surrogate_sd[np.newaxis, :, np.newaxis] * delta_c[:, :, np.newaxis]
        )
        dalys_low_us = np.maximum(0.0, dalys_mean_us - np.hypot(bau_lo_hw, surr_hw))
        dalys_up_us = dalys_mean_us + np.hypot(bau_hi_hw, surr_hw)

        dalys_2040_mean = np.sum(dalys_mean_us, axis=(1, 2))
        dalys_2040_lower = np.sum(dalys_low_us, axis=(1, 2))
        dalys_2040_upper = np.sum(dalys_up_us, axis=(1, 2))

        asr_us = mult * self._bau_asr_val[np.newaxis, :, :]  # (N, U, 2)
        asr_gap_u = np.abs(asr_us[:, :, 0] - asr_us[:, :, 1])  # (N, U)
        equity_gap_asr = np.sum(asr_gap_u * self._pop_weights[np.newaxis, :], axis=1)
        equity_gap_dalys_abs = np.abs(
            np.sum(dalys_mean_us[:, :, 0] - dalys_mean_us[:, :, 1], axis=1)
        )
        return (
            dalys_2040_mean,
            dalys_2040_lower,
            dalys_2040_upper,
            equity_gap_asr,
            equity_gap_dalys_abs,
        )

    def evaluate(self, x: np.ndarray) -> EvaluationResult:
        """Evaluate a single ``(48,)`` or batched ``(N, 48)`` decision array."""
        raw_arr = np.asarray(x, dtype=float)
        is_single = raw_arr.ndim == 1
        if not np.all(np.isfinite(raw_arr)):
            raise ValueError("Decision vector contains non-finite values (NaN or inf).")
        if np.any(raw_arr < 0.0):
            raise ValueError("Decision vector contains negative entries (< 0).")

        tensor = vector_to_tensor(raw_arr, self.n_units, self.n_interventions, self.n_periods)
        if is_single:
            tensor = tensor[np.newaxis, ...]  # (1, U, I, P)

        # Slice cooking interventions: (N, U, 3, P)
        x_cook = tensor[:, :, self._non_infra_indices, :]
        # Slice infra intervention: (N, U, P)
        x_infra = tensor[:, :, self._idx_infra, :]

        # Effective clean-cooking gain per unit, capped so total clean fuel <= 1.0
        raw_gain_u = np.sum(x_cook, axis=(2, 3))  # (N, U)
        eff_gain_u = np.sum(
            x_cook * self._non_infra_effectiveness[np.newaxis, :, :, np.newaxis],
            axis=(2, 3),
        )  # (N, U)

        (
            dalys_2040_mean,
            dalys_2040_lower,
            dalys_2040_upper,
            equity_gap_asr,
            equity_gap_dalys_abs,
        ) = self.evaluate_effective_gains(eff_gain_u)

        # Costs (cooking continuous + infra binary indicator >= threshold)
        disc = self._period_discount[np.newaxis, np.newaxis, np.newaxis, :]
        cook_cost = np.sum(
            x_cook * self._non_infra_costs[np.newaxis, :, :, np.newaxis] * disc,
            axis=(1, 2, 3),
        )
        infra_built = (x_infra >= self.infra_build_threshold).astype(float)  # (N, U, P)
        infra_cost = np.sum(
            infra_built
            * self._infra_costs[np.newaxis, :, np.newaxis]
            * self._period_discount[np.newaxis, np.newaxis, :],
            axis=(1, 2),
        )
        total_cost = cook_cost + infra_cost

        # Constraint violations (>= 0; 0.0 when feasible)
        violation_budget = np.maximum(0.0, total_cost - self.total_budget_usd_millions)
        violation_adoption_cap = np.sum(
            np.maximum(0.0, x_cook - self._non_infra_max_adoption[np.newaxis, :, :, :]),
            axis=(1, 2, 3),
        )
        violation_coverage_ceiling = np.sum(
            np.maximum(
                0.0,
                self._cf_bau_2040[np.newaxis, :]
                + raw_gain_u
                - self._coverage_ceiling[np.newaxis, :],
            ),
            axis=1,
        )

        # Data-driven infra prerequisites (same or earlier period q <= p, mirroring PDDL)
        cum_built = np.maximum.accumulate(infra_built, axis=2)  # (N, U, P)
        lpg_ready = np.maximum(self._lpg_infra_init[np.newaxis, :, np.newaxis], cum_built)
        grid_ready = np.maximum(self._grid_ready_init[np.newaxis, :, np.newaxis], cum_built)
        x_lpg = tensor[:, :, self._idx_lpg, :]
        x_elec = tensor[:, :, self._idx_elec, :]
        violation_infra_prereq = np.sum(
            x_lpg * (1.0 - lpg_ready) + x_elec * (1.0 - grid_ready),
            axis=(1, 2),
        )

        is_feasible = (
            (violation_budget == 0.0)
            & (violation_adoption_cap == 0.0)
            & (violation_coverage_ceiling == 0.0)
            & (violation_infra_prereq == 0.0)
        )

        if is_single:
            return EvaluationResult(
                dalys_2040_mean=float(dalys_2040_mean[0]),
                dalys_2040_lower=float(dalys_2040_lower[0]),
                dalys_2040_upper=float(dalys_2040_upper[0]),
                total_cost_usd_millions=float(total_cost[0]),
                equity_gap_asr=float(equity_gap_asr[0]),
                equity_gap_dalys_abs=float(equity_gap_dalys_abs[0]),
                violation_budget=float(violation_budget[0]),
                violation_adoption_cap=float(violation_adoption_cap[0]),
                violation_coverage_ceiling=float(violation_coverage_ceiling[0]),
                violation_infra_prereq=float(violation_infra_prereq[0]),
                is_feasible=bool(is_feasible[0]),
                is_mock=self.is_mock,
                warning_mock_inputs=self.warning_mock_inputs,
                data_version=self.data_version,
            )

        return EvaluationResult(
            dalys_2040_mean=dalys_2040_mean,
            dalys_2040_lower=dalys_2040_lower,
            dalys_2040_upper=dalys_2040_upper,
            total_cost_usd_millions=total_cost,
            equity_gap_asr=equity_gap_asr,
            equity_gap_dalys_abs=equity_gap_dalys_abs,
            violation_budget=violation_budget,
            violation_adoption_cap=violation_adoption_cap,
            violation_coverage_ceiling=violation_coverage_ceiling,
            violation_infra_prereq=violation_infra_prereq,
            is_feasible=is_feasible,
            is_mock=self.is_mock,
            warning_mock_inputs=self.warning_mock_inputs,
            data_version=self.data_version,
        )
