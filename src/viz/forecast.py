"""Stage 2 forecasting visualizations for AEGIS-GBD.

Generates:
1. s2_forecasts_2040_ind_npl.png: 2024-2040 projected DALY and Death rates across 4 units.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")

UNIT_COLORS = {
    "IND_urban": "#E6AB02",
    "IND_rural": "#D95F02",
    "NPL_urban": "#A6761D",
    "NPL_rural": "#7570B3",
}


def plot_forecasts_2040(df_forecasts: pd.DataFrame, output_path: Path) -> Path:
    """Plot 2025-2040 projected DALY and Death rates for India and Nepal strata."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    sub = df_forecasts[
        (df_forecasts["metric"] == "Rate_per_100k")
        & (df_forecasts["sex"] == "both")
        & (df_forecasts["scenario"] == "bau")
    ]

    for idx, measure in enumerate(["DALYs", "Deaths"]):
        ax = axes[idx]
        m_sub = sub[sub["measure"] == measure]

        for unit in ["IND_urban", "IND_rural", "NPL_urban", "NPL_rural"]:
            u_data = m_sub[m_sub["location"] == unit].sort_values("year")
            color = UNIT_COLORS.get(unit, "#333333")
            ax.plot(
                u_data["year"],
                u_data["val"],
                label=unit,
                color=color,
                linewidth=2.2,
            )
            ax.fill_between(
                u_data["year"],
                u_data["lower"],
                u_data["upper"],
                color=color,
                alpha=0.15,
            )

        ax.set_title(f"Projected 2025–2040 BAU {measure} ASR", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(f"Rate per 100k ({measure})", fontsize=10)
        ax.set_xlim(2025, 2040)
        ax.legend(title="Unit", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
