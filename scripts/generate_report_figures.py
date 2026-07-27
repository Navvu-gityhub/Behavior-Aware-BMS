"""Generate final_report.md's figures from data already computed by the
calibration scripts (see docs/final_report.md Appendix for how each
reports/metrics/*.csv was produced). This script does no new modeling —
it only visualizes numbers already reported in prose/tables in the text,
so a reader can sanity-check a figure against the report's own claims.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444",
    "axes.grid": True,
    "grid.color": "#DDD",
    "grid.linewidth": 0.6,
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
})

# ---------------------------------------------------------------------
# Figure 1: the baseline health_index collapses a wide range of real
# fade rates onto a handful of repeated values (Section 4.1/4.2).
# ---------------------------------------------------------------------
fade = pd.read_csv("reports/metrics/nasa_ground_truth_fade.csv")
scores = pd.read_csv("reports/metrics/nasa_pipeline_scores.csv")
joined = fade.merge(scores[["battery_id", "health_index"]], on="battery_id", how="inner")

fig, ax = plt.subplots(figsize=(7, 4.2))
groups = joined.groupby("health_index")["fade_rate_ah_per_cycle"]
xs = sorted(joined["health_index"].unique())
for hi in xs:
    vals = joined.loc[joined.health_index == hi, "fade_rate_ah_per_cycle"]
    jitter = np.random.default_rng(0).uniform(-0.6, 0.6, size=len(vals))
    ax.scatter(np.full(len(vals), hi) + jitter, vals, s=34, alpha=0.75,
               color="#2A6F97", edgecolor="white", linewidth=0.4, zorder=3)
    ax.plot([hi - 1.2, hi + 1.2], [vals.mean()] * 2, color="#B23A48", lw=1.6, zorder=4)

ax.axhline(0, color="#999", lw=0.8, zorder=1)
ax.set_xlabel("Assigned health_index (bucketed rule)")
ax.set_ylabel("Real measured fade rate (Ah/cycle)")
ax.set_title("Fig. 1 — The baseline health_index does not resolve real fade-rate differences")
n_at_39 = (joined.health_index == 39).sum()
ax.annotate(
    f"{n_at_39} batteries all scored\nhealth_index = 39\n(fade rate spans 25×)",
    xy=(39, joined.loc[joined.health_index == 39, "fade_rate_ah_per_cycle"].max()),
    xytext=(46, 0.05), fontsize=9, color="#B23A48",
    arrowprops=dict(arrowstyle="->", color="#B23A48", lw=1),
)
fig.tight_layout()
fig.savefig(OUT / "fig1_health_index_collapse.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: trailing_avg_temp vs. capacity_loss for the two cohorts
# where the correlation is significant and correctly signed (Section 4.2).
# ---------------------------------------------------------------------
cyc = pd.read_csv("reports/metrics/continuous_model_training_data.csv")
cohort_info = {
    "RT_CC_mixed_current": ("ρ = 0.21, p < 0.0001", "#2A6F97"),
    "COLD4C_multiload": ("ρ = 0.22, p < 0.0001", "#B23A48"),
}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
for ax, (cohort, (label, color)) in zip(axes, cohort_info.items()):
    sub = cyc[cyc.cohort == cohort]
    ax.scatter(sub.trailing_avg_temp, sub.capacity_loss, s=10, alpha=0.35, color=color)
    z = np.polyfit(sub.trailing_avg_temp, sub.capacity_loss, 1)
    xs = np.linspace(sub.trailing_avg_temp.min(), sub.trailing_avg_temp.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="black", lw=1.6)
    ax.set_title(f"{cohort}\n{label}", fontsize=10)
    ax.set_xlabel("Trailing 5-cycle avg. temperature (°C)")
axes[0].set_ylabel("Per-cycle capacity loss (Ah)")
fig.suptitle("Fig. 2 — Temperature is a significant, correctly-signed, transferable signal", y=1.03, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig2_temperature_signal.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: horizon comparison (Section 4.5) -- rank signal improves,
# R^2 against a naive baseline gets worse.
# ---------------------------------------------------------------------
h = pd.read_csv("reports/metrics/horizon_regression_summary.csv")

fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
ax2 = ax1.twinx()

ax1.plot(h.horizon_cycles, h.lobo_median_r2_vs_own_mean, "o-", color="#B23A48", lw=2, label="LOBO median R² vs. own mean")
ax1.axhline(0, color="#B23A48", lw=0.7, ls="--", alpha=0.6)
ax1.set_ylabel("LOBO median R² vs. own mean", color="#B23A48")
ax1.tick_params(axis="y", labelcolor="#B23A48")

ax2.plot(h.horizon_cycles, h.lobo_median_spearman_rho, "s-", color="#2A6F97", lw=2, label="LOBO median Spearman ρ")
ax2.set_ylabel("LOBO median Spearman ρ", color="#2A6F97")
ax2.tick_params(axis="y", labelcolor="#2A6F97")

ax1.set_xlabel("Prediction horizon (cycles)")
ax1.set_xticks(h.horizon_cycles)
ax1.set_title("Fig. 3 — Longer horizons: rank signal improves, calibrated R² does not")
for x, n_drop in zip(h.horizon_cycles, h.n_batteries_dropped):
    if n_drop:
        ax1.annotate(f"−{n_drop} batteries", (x, ax1.get_ylim()[0]), fontsize=8, color="#666",
                     ha="center", va="bottom")
fig.tight_layout()
fig.savefig(OUT / "fig3_horizon_comparison.png", dpi=170)
plt.close(fig)

print("Wrote:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p, f"({p.stat().st_size // 1024} KB)")
