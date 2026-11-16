"""Stage 1 historical analysis visualizations for AEGIS-GBD.

Generates:
1. s1_trends_southasia.png: National ASR trajectories with 95% UI ribbons across 5 countries.
2. s1_ind_npl_comparison.png: Sex- and stratum-disaggregated comparison for India & Nepal.
3. s1_joinpoints.png: Piecewise linear regression fits with detected inflection break years.
4. s1_hap_copd_share.png: Trajectory of HAP-COPD share of total COPD and all-cause HAP.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import pwlf
except ImportError:
    pwlf = None

# Publication styling defaults
sns.set_theme(style="whitegrid", font="sans-serif")
PALETTE = {
    "India": "#D95F02",
    "Nepal": "#7570B3",
    "Pakistan": "#1B9E77",
    "Bangladesh": "#E7298A",
    "Bhutan": "#66A61E",
    "IND_urban": "#E6AB02",
    "IND_rural": "#D95F02",
    "NPL_urban": "#A6761D",
    "NPL_rural": "#7570B3",
    "female": "#D95F02",
    "male": "#1B9E77",
    "both": "#333333",
}


def plot_trends_southasia(df_hap: pd.DataFrame, output_path: Path) -> Path:
    """Plot 1990-2023 national ASR trajectories with 95% UI ribbons for the 5-country panel."""
    sub = df_hap[
        (df_hap["metric"] == "Rate_per_100k")
        & (df_hap["age"] == "Age-standardized")
        & (df_hap["sex"] == "both")
        & (df_hap["location"].isin(["India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"]))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for idx, measure in enumerate(["DALYs", "Deaths"]):
        ax = axes[idx]
        m_sub = sub[sub["measure"] == measure]
        for country in ["India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"]:
            c_data = m_sub[m_sub["location"] == country].sort_values("year")
            color = PALETTE.get(country, "#444444")
            ax.plot(
                c_data["year"],
                c_data["val"],
                label=country,
                color=color,
                linewidth=2.0,
            )
            ax.fill_between(
                c_data["year"],
                c_data["lower"],
                c_data["upper"],
                color=color,
                alpha=0.15,
            )

        ax.set_title(
            f"HAP-Attributable COPD {measure} ASR (1990–2023)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(f"Rate per 100,000 ({measure})", fontsize=11)
        ax.set_xlim(1990, 2023)
        ax.legend(title="Country", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_ind_npl_comparison(df_hap: pd.DataFrame, output_path: Path) -> Path:
    """Compare India and Nepal across sex and urban/rural strata."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: India vs Nepal by Sex (DALYs ASR)
    ax_a = axes[0, 0]
    sub_sex = df_hap[
        (df_hap["metric"] == "Rate_per_100k")
        & (df_hap["age"] == "Age-standardized")
        & (df_hap["measure"] == "DALYs")
        & (df_hap["location"].isin(["India", "Nepal"]))
        & (df_hap["sex"].isin(["female", "male"]))
    ]
    for loc, ls in [("India", "-"), ("Nepal", "--")]:
        for sex in ["female", "male"]:
            data = sub_sex[(sub_sex["location"] == loc) & (sub_sex["sex"] == sex)].sort_values(
                "year"
            )
            ax_a.plot(
                data["year"],
                data["val"],
                label=f"{loc} ({sex})",
                linestyle=ls,
                color=PALETTE.get(sex, "#333333"),
                linewidth=2.0,
            )
    ax_a.set_title("A: DALY ASR by Sex (India vs Nepal)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("DALYs per 100k")
    ax_a.legend(frameon=True)

    # Panel B: India vs Nepal by Sex (Deaths ASR)
    ax_b = axes[0, 1]
    sub_sex_d = df_hap[
        (df_hap["metric"] == "Rate_per_100k")
        & (df_hap["age"] == "Age-standardized")
        & (df_hap["measure"] == "Deaths")
        & (df_hap["location"].isin(["India", "Nepal"]))
        & (df_hap["sex"].isin(["female", "male"]))
    ]
    for loc, ls in [("India", "-"), ("Nepal", "--")]:
        for sex in ["female", "male"]:
            data = sub_sex_d[
                (sub_sex_d["location"] == loc) & (sub_sex_d["sex"] == sex)
            ].sort_values("year")
            ax_b.plot(
                data["year"],
                data["val"],
                label=f"{loc} ({sex})",
                linestyle=ls,
                color=PALETTE.get(sex, "#333333"),
                linewidth=2.0,
            )
    ax_b.set_title("B: Death ASR by Sex (India vs Nepal)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Deaths per 100k")
    ax_b.legend(frameon=True)

    # Panel C: Urban vs Rural Strata (DALYs ASR)
    ax_c = axes[1, 0]
    sub_strat = df_hap[
        (df_hap["metric"] == "Rate_per_100k")
        & (df_hap["age"] == "Age-standardized")
        & (df_hap["measure"] == "DALYs")
        & (df_hap["sex"] == "both")
        & (df_hap["location"].isin(["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]))
    ]
    for loc in ["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]:
        data = sub_strat[sub_strat["location"] == loc].sort_values("year")
        ax_c.plot(
            data["year"],
            data["val"],
            label=loc,
            color=PALETTE.get(loc, "#555555"),
            linewidth=2.0,
        )
    ax_c.set_title("C: DALY ASR by Urban/Rural Stratum (ASM-01)", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("DALYs per 100k")
    ax_c.set_xlabel("Year")
    ax_c.legend(frameon=True)

    # Panel D: Absolute DALY Burden (Millions)
    ax_d = axes[1, 1]
    sub_num = df_hap[
        (df_hap["metric"] == "Number")
        & (df_hap["age"] == "All ages")
        & (df_hap["measure"] == "DALYs")
        & (df_hap["sex"] == "both")
        & (df_hap["location"].isin(["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]))
    ]
    for loc in ["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]:
        data = sub_num[sub_num["location"] == loc].sort_values("year")
        ax_d.plot(
            data["year"],
            data["val"] / 1e6,
            label=loc,
            color=PALETTE.get(loc, "#555555"),
            linewidth=2.0,
        )
    ax_d.set_title("D: Absolute DALYs (Millions)", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("DALYs (Millions)")
    ax_d.set_xlabel("Year")
    ax_d.legend(frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_joinpoints(df_hap: pd.DataFrame, output_path: Path) -> Path:
    """Plot observed log-ASR and pwlf piecewise linear fits with inflection breakpoints."""
    if pwlf is None:
        warnings.warn("pwlf not installed; skipping joinpoint plot.", stacklevel=2)
        return output_path

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    sub = df_hap[
        (df_hap["metric"] == "Rate_per_100k")
        & (df_hap["age"] == "Age-standardized")
        & (df_hap["sex"] == "both")
        & (df_hap["measure"] == "DALYs")
    ]

    for idx, country in enumerate(["India", "Nepal"]):
        ax = axes[idx]
        data = sub[sub["location"] == country].sort_values("year")
        x = data["year"].to_numpy(dtype=float)
        y = np.log(data["val"].to_numpy(dtype=float))

        model = pwlf.PiecewiseLinFit(x, y)
        breaks = model.fit(2)
        x_pred = np.linspace(x.min(), x.max(), 100)
        y_pred = model.predict(x_pred)

        ax.scatter(x, y, color="#333333", alpha=0.7, label="Observed ln(ASR)", s=25)
        line_color = "#D95F02" if country == "India" else "#7570B3"
        ax.plot(
            x_pred,
            y_pred,
            color=line_color,
            linewidth=2.5,
            label="pwlf 2-Segment Fit",
        )

        # Mark breakpoint
        bp = breaks[1]
        ax.axvline(
            bp,
            color="#E7298A",
            linestyle="--",
            linewidth=1.5,
            label=f"Breakpoint ({bp:.1f})",
        )

        ax.set_title(
            f"{country} DALY ASR Joinpoint Fit (Breakpoint: {bp:.1f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("ln(DALY ASR)")
        ax.legend(frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_hap_copd_share(df_shares: pd.DataFrame, output_path: Path) -> Path:
    """Plot historical trajectory of HAP share of total COPD and all-cause HAP."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for idx, measure in enumerate(["DALYs", "Deaths"]):
        ax = axes[idx]
        sub = df_shares[df_shares["measure"] == measure]
        for country in ["India", "Nepal", "Pakistan", "Bangladesh", "Bhutan"]:
            c_data = sub[sub["location"] == country].sort_values("year")
            ax.plot(
                c_data["year"],
                c_data["hap_copd_share_of_total_copd_pct"],
                label=country,
                color=PALETTE.get(country, "#555555"),
                linewidth=2.0,
            )
        ax.set_title(f"HAP Share of Total COPD {measure} (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent of Total COPD (%)")
        ax.set_xlim(1990, 2023)
        ax.legend(title="Country", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
