"""Historical trend metrics for AEGIS-GBD Stage 1.

Implements:
1. Historical summary (1990 vs 2023 counts, ASRs, relative shifts)
2. EAPC (Estimated Annual Percentage Change) with 95% confidence intervals
3. pwlf piecewise linear joinpoint detection and segment APCs
4. Attributable shares of total COPD and all-cause HAP
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pwlf
except ImportError:
    pwlf = None


def compute_historical_summary(
    df_hap: pd.DataFrame,
    df_tot: pd.DataFrame | None = None,
    df_all: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute 1990 vs 2023 summary table for all locations and sexes."""
    # Filter to 1990 and 2023
    sub = df_hap[df_hap["year"].isin([1990, 2023])].copy()

    records: list[dict[str, Any]] = []
    locations = sub["location"].unique()
    measures = ["DALYs", "Deaths"]
    sexes = ["both", "female", "male"]

    for loc in locations:
        for m in measures:
            for s in sexes:
                # Number (All ages)
                row_num_90 = sub[
                    (sub["location"] == loc)
                    & (sub["measure"] == m)
                    & (sub["sex"] == s)
                    & (sub["metric"] == "Number")
                    & (sub["age"] == "All ages")
                    & (sub["year"] == 1990)
                ]
                row_num_23 = sub[
                    (sub["location"] == loc)
                    & (sub["measure"] == m)
                    & (sub["sex"] == s)
                    & (sub["metric"] == "Number")
                    & (sub["age"] == "All ages")
                    & (sub["year"] == 2023)
                ]

                # Rate (Age-standardized)
                row_asr_90 = sub[
                    (sub["location"] == loc)
                    & (sub["measure"] == m)
                    & (sub["sex"] == s)
                    & (sub["metric"] == "Rate_per_100k")
                    & (sub["age"] == "Age-standardized")
                    & (sub["year"] == 1990)
                ]
                row_asr_23 = sub[
                    (sub["location"] == loc)
                    & (sub["measure"] == m)
                    & (sub["sex"] == s)
                    & (sub["metric"] == "Rate_per_100k")
                    & (sub["age"] == "Age-standardized")
                    & (sub["year"] == 2023)
                ]

                val_num_90 = float(row_num_90["val"].iloc[0]) if not row_num_90.empty else np.nan
                val_num_23 = float(row_num_23["val"].iloc[0]) if not row_num_23.empty else np.nan
                val_asr_90 = float(row_asr_90["val"].iloc[0]) if not row_asr_90.empty else np.nan
                val_asr_23 = float(row_asr_23["val"].iloc[0]) if not row_asr_23.empty else np.nan

                asr_pct_change = (
                    ((val_asr_23 - val_asr_90) / val_asr_90 * 100.0)
                    if (val_asr_90 and not np.isnan(val_asr_90))
                    else np.nan
                )
                num_pct_change = (
                    ((val_num_23 - val_num_90) / val_num_90 * 100.0)
                    if (val_num_90 and not np.isnan(val_num_90))
                    else np.nan
                )

                records.append(
                    {
                        "location": loc,
                        "measure": m,
                        "sex": s,
                        "count_1990": val_num_90,
                        "count_2023": val_num_23,
                        "count_pct_change": num_pct_change,
                        "asr_1990": val_asr_90,
                        "asr_2023": val_asr_23,
                        "asr_pct_change": asr_pct_change,
                    }
                )

    return pd.DataFrame(records)


def compute_eapc(
    df_hap: pd.DataFrame,
    start_year: int = 1990,
    end_year: int = 2023,
    period_label: str = "1990-2023",
) -> pd.DataFrame:
    """Compute Estimated Annual Percentage Change (EAPC) with 95% CIs.

    ln(ASR_t) = alpha + beta * t + eps
    EAPC = 100 * (exp(beta) - 1)
    95% CI from SE(beta).
    """
    sub = df_hap[
        (df_hap["metric"] == "Rate_per_100k")
        & (df_hap["age"] == "Age-standardized")
        & (df_hap["year"] >= start_year)
        & (df_hap["year"] <= end_year)
    ].copy()

    records: list[dict[str, Any]] = []
    locations = sub["location"].unique()
    measures = ["DALYs", "Deaths"]
    sexes = ["both", "female", "male"]

    for loc in locations:
        for m in measures:
            for s in sexes:
                series = sub[
                    (sub["location"] == loc) & (sub["measure"] == m) & (sub["sex"] == s)
                ].sort_values("year")

                if len(series) < 3:
                    continue

                years = series["year"].to_numpy(dtype=float)
                asrs = series["val"].to_numpy(dtype=float)

                # Avoid log of zero or negative
                valid_mask = asrs > 0
                if valid_mask.sum() < 3:
                    continue

                x = years[valid_mask]
                y = np.log(asrs[valid_mask])
                n = len(x)

                res = stats.linregress(x, y)
                beta = res.slope
                se_beta = res.stderr

                # t critical for 95% CI with n-2 degrees of freedom
                t_crit = stats.t.ppf(0.975, df=n - 2) if n > 2 else 1.96

                eapc = 100.0 * (np.exp(beta) - 1.0)
                eapc_lower = 100.0 * (np.exp(beta - t_crit * se_beta) - 1.0)
                eapc_upper = 100.0 * (np.exp(beta + t_crit * se_beta) - 1.0)

                records.append(
                    {
                        "location": loc,
                        "measure": m,
                        "sex": s,
                        "period": period_label,
                        "start_year": int(start_year),
                        "end_year": int(end_year),
                        "beta": beta,
                        "se_beta": se_beta,
                        "p_value": res.pvalue,
                        "r_squared": res.rvalue**2,
                        "eapc": eapc,
                        "eapc_lower": eapc_lower,
                        "eapc_upper": eapc_upper,
                    }
                )

    return pd.DataFrame(records)


def compute_pwlf_joinpoints(
    df_hap: pd.DataFrame,
    locations: list[str] | None = None,
    n_segments: int = 2,
) -> pd.DataFrame:
    """Fit piecewise linear regression via pwlf to detect inflection breakpoints and APCs."""
    if pwlf is None:
        warnings.warn("pwlf is not installed; returning empty joinpoint table.", stacklevel=2)
        return pd.DataFrame()

    sub = df_hap[
        (df_hap["metric"] == "Rate_per_100k")
        & (df_hap["age"] == "Age-standardized")
        & (df_hap["sex"] == "both")
    ].copy()

    if locations is None:
        locations = ["India", "Nepal", "IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]

    records: list[dict[str, Any]] = []

    for loc in locations:
        for m in ["DALYs", "Deaths"]:
            series = sub[(sub["location"] == loc) & (sub["measure"] == m)].sort_values("year")
            if len(series) < 6:
                continue

            x = series["year"].to_numpy(dtype=float)
            y = np.log(series["val"].to_numpy(dtype=float))

            try:
                # Suppress scipy optimize warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = pwlf.PiecewiseLinFit(x, y)
                    breaks = model.fit(n_segments)
                    slopes = model.calc_slopes()

                # Breakpoints (inner inflection years)
                inner_breaks = [round(b, 1) for b in breaks[1:-1]]
                segment_apcs = [round(100.0 * (np.exp(s) - 1.0), 2) for s in slopes]

                records.append(
                    {
                        "location": loc,
                        "measure": m,
                        "n_segments": n_segments,
                        "breakpoint_years": str(inner_breaks),
                        "segment_apcs": str(segment_apcs),
                        "slope_1": slopes[0] if len(slopes) > 0 else np.nan,
                        "apc_segment_1": segment_apcs[0] if len(segment_apcs) > 0 else np.nan,
                        "slope_2": slopes[1] if len(slopes) > 1 else np.nan,
                        "apc_segment_2": segment_apcs[1] if len(segment_apcs) > 1 else np.nan,
                    }
                )
            except Exception as e:
                warnings.warn(f"pwlf fit failed for {loc} {m}: {e}", stacklevel=2)
                continue

    return pd.DataFrame(records)


def compute_attributable_shares(
    df_hap: pd.DataFrame,
    df_tot: pd.DataFrame,
    df_all: pd.DataFrame,
) -> pd.DataFrame:
    """Compute HAP-COPD share of total COPD and COPD share of all-cause HAP."""
    # Filter to National panel countries, both sexes, Rate_per_100k, Age-standardized
    h_sub = df_hap[
        (df_hap["sex"] == "both")
        & (df_hap["age"] == "Age-standardized")
        & (df_hap["metric"] == "Rate_per_100k")
        & (df_hap["location"].isin(["India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"]))
    ][["location", "year", "measure", "val"]].rename(columns={"val": "hap_copd_asr"})

    t_sub = df_tot[(df_tot["age_name"] == "Age-standardized") & (df_tot["metric_name"] == "Rate")][
        ["location_name", "year", "measure_name", "val"]
    ].rename(
        columns={"location_name": "location", "measure_name": "measure", "val": "total_copd_asr"}
    )

    a_sub = df_all[(df_all["age_name"] == "Age-standardized") & (df_all["metric_name"] == "Rate")][
        ["location_name", "year", "measure_name", "val"]
    ].rename(
        columns={"location_name": "location", "measure_name": "measure", "val": "all_cause_hap_asr"}
    )

    merged = pd.merge(h_sub, t_sub, on=["location", "year", "measure"], how="left")
    merged = pd.merge(merged, a_sub, on=["location", "year", "measure"], how="left")

    merged["hap_copd_share_of_total_copd_pct"] = (
        merged["hap_copd_asr"] / merged["total_copd_asr"] * 100.0
    )
    merged["hap_copd_share_of_all_hap_pct"] = (
        merged["hap_copd_asr"] / merged["all_cause_hap_asr"] * 100.0
    )

    return merged.sort_values(["location", "measure", "year"])
