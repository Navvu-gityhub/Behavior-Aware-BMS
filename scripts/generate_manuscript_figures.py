"""Draw the figures for docs/manuscript/ress_draft.md from tracked artifacts.

Every number plotted here is read from a CSV under `reports/metrics/`. This
script does no modelling and fits nothing: it visualises results the manuscript
already states in prose, so a reader can check a figure against a claim.

The summary statistics annotated on Figure 1 are computed by importing the very
recipes `tests/test_reported_numbers.py` uses to pin those figures in the text.
That import is deliberate and is the point: if the figure and the prose computed
their numbers independently they could disagree, and this project has already
caught itself publishing a stale figure once (manuscript section 6.4). The
script asserts the recomputed values against the published ones and refuses to
write a figure that disagrees.

    python scripts/generate_manuscript_figures.py

Writes reports/figures/manuscript_fig{1,2,3}_*.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_reported_numbers import (  # noqa: E402
    CORRECTED_SWEEP_FILE,
    FULL_DRAW_COUNT,
    MATCHED_LEVELS,
    SWEEP_FILE,
    loco_iqr,
    paired_method_difference,
)

METRICS = REPO / "reports" / "metrics"
OUT = REPO / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

#: Published in ADR 0013 and quoted in section 5.2.1. The script refuses to draw
#: a figure whose recomputed statistics differ from these.
PUBLISHED = {
    "cycle_index": {"iqr": 0.892, "effect": -0.336, "ratio": 2.66},
    "full_discharge": {"iqr": 0.296, "effect": -0.110, "ratio": 2.70},
}
TOL = 0.006

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

NOISE = "#2B6CB0"
EFFECT = "#C05621"


def _gaps(rel: str, dataset: str) -> np.ndarray:
    """Per-draw LOCO gaps on the matched levels, fully-drawn levels only.

    Mirrors `loco_iqr`'s filtering exactly, so the plotted points are the same
    population the annotated IQR summarises.
    """
    table = pd.read_csv(METRICS / rel)
    table = table[table["dataset"] == dataset]
    full = table.groupby(["dataset", "n_cohorts"]).filter(
        lambda g: len(g) >= FULL_DRAW_COUNT
    )
    full = full[full["n_cohorts"].isin(MATCHED_LEVELS)]
    return full["gap"].to_numpy(dtype=float)


def _check(label: str, iqr: float, effect: float, ratio: float) -> None:
    want = PUBLISHED[label]
    for name, got, expected in (
        ("IQR", iqr, want["iqr"]),
        ("effect", effect, want["effect"]),
        ("ratio", ratio, want["ratio"]),
    ):
        if abs(got - expected) > TOL:
            raise SystemExit(
                f"{label} {name}: recomputed {got:.4f} but ADR 0013 publishes "
                f"{expected:.4f}. The figure and the prose disagree; fix the "
                f"discrepancy rather than the tolerance."
            )


def figure_1() -> Path:
    """Section 5.2's argument in one panel: noise shrinks, ratio does not."""
    panels = [
        ("cycle_index", SWEEP_FILE, "calce",
         "(a)  Cycle_Index target - the derivation section 5.4 refutes"),
        ("full_discharge", CORRECTED_SWEEP_FILE, "calce_full_discharge",
         "(b)  Full-discharge target - ADR 0012 correction"),
    ]

    stats = {}
    for label, rel, dataset, _ in panels:
        iqr = loco_iqr(rel, dataset=dataset, levels=MATCHED_LEVELS)
        effect = paired_method_difference(
            rel=rel, dataset=dataset, levels=MATCHED_LEVELS
        )
        ratio = iqr / abs(effect)
        _check(label, iqr, effect, ratio)
        stats[label] = (iqr, effect, ratio, _gaps(rel, dataset))

    span = max(
        float(np.abs(np.concatenate([s[3] for s in stats.values()])).max()), 1.0
    )
    limits = (-span * 1.12, span * 0.45)

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.4), sharex=True)

    for ax, (label, _rel, _ds, title) in zip(axes, panels, strict=True):
        iqr, effect, ratio, gaps = stats[label]
        q1, q3 = np.quantile(gaps, [0.25, 0.75])

        rng = np.random.default_rng(20260916)
        ax.scatter(
            gaps, rng.uniform(0.62, 0.98, gaps.size), s=14, color=NOISE,
            alpha=0.45, edgecolors="none",
            label=f"LOCO gap, one point per cohort draw (n={gaps.size})",
        )
        ax.add_patch(plt.Rectangle(
            (q1, 0.55), q3 - q1, 0.5, facecolor=NOISE, alpha=0.16,
            edgecolor=NOISE, linewidth=1.2,
        ))
        ax.annotate(
            f"IQR = {iqr:.3f} R2", xy=((q1 + q3) / 2, 1.13), ha="center",
            color=NOISE, fontweight="bold", fontsize=10,
        )

        ax.hlines(0.3, effect, 0, color=EFFECT, linewidth=6, alpha=0.85)
        ax.annotate(
            f"between-method effect = {effect:.3f} R2", xy=(effect / 2, 0.14),
            ha="center", color=EFFECT, fontweight="bold", fontsize=10,
        )

        ax.axvline(0.0, color="#666", linewidth=0.9, linestyle=":")
        ax.set_title(title, loc="left")
        ax.set_xlim(*limits)
        ax.set_ylim(0, 1.45)
        ax.set_yticks([])
        ax.text(
            0.015, 0.93, f"noise / effect = {ratio:.2f}x",
            transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#F2F5F9",
                      edgecolor="#9AA7B4"),
        )

    axes[0].legend(loc="lower left", fontsize=8.6, framealpha=0.95)
    axes[-1].set_xlabel(
        "LOCO minus LOBO gap in R2   (negative = skill lost under protocol shift)"
    )
    fig.suptitle(
        "The LOCO estimate is several times noisier than the difference it measures",
        fontsize=12, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.012,
        "CALCE, matched coverage levels k = 3, 4, 5.\n"
        "Correcting the target shrinks noise and effect ~3x; the ratio is unchanged.",
        ha="center", va="bottom", fontsize=8.4, color="#444", linespacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))

    path = OUT / "manuscript_fig1_loco_gap_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_2() -> Path:
    """Section 5.5: aggregate coverage certifies a system failing per cohort."""
    folds = pd.read_csv(METRICS / "coverage_folds.csv")
    loco = folds[folds["split"] == "LOCO"].copy()
    if loco.empty:
        raise SystemExit("coverage_folds.csv has no LOCO rows")

    nominal = float(loco["nominal"].iloc[0])
    methods = sorted(loco["method"].unique())
    order = sorted(loco["held_out"].unique())
    position = {name: index for index, name in enumerate(order)}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    palette = plt.get_cmap("tab10")

    for index, method in enumerate(methods):
        rows = loco[loco["method"] == method]
        offset = (index - (len(methods) - 1) / 2) * 0.12
        ax.scatter(
            [position[name] + offset for name in rows["held_out"]],
            rows["coverage"], s=42, color=palette(index % 10), label=method,
            edgecolors="white", linewidths=0.6, zorder=3,
        )

    ax.axhline(nominal, color="#C53030", linewidth=1.6, linestyle="--",
               label=f"nominal {nominal:.0%}", zorder=2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8.6)
    ax.set_ylabel("empirical coverage")
    ax.set_xlabel("held-out cohort")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Per-cohort conformal coverage under protocol shift, against the nominal guarantee",
        loc="left",
    )
    ax.legend(fontsize=8.4, ncol=2, loc="lower left", framealpha=0.95)

    worst = loco.loc[loco["coverage"].idxmin()]
    ax.annotate(
        f"worst: {worst['method']} on {worst['held_out']} - {worst['coverage']:.0%}",
        xy=(0.985, 0.04), xycoords="axes fraction", ha="right", fontsize=9,
        color="#C53030", fontweight="bold",
    )
    fig.tight_layout()

    path = OUT / "manuscript_fig2_per_cohort_coverage.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_3(cell_id: str = "CS2_5") -> Path:
    """Section 5.4: the same cell under two target derivations."""
    old = pd.read_csv(METRICS / "calce_cycle_level.csv")
    new = pd.read_csv(METRICS / "calce_full_discharge.csv")
    old = old[old["cell_id"] == cell_id]
    new = new[new["cell_id"] == cell_id]
    if old.empty or new.empty:
        raise SystemExit(f"{cell_id} missing from one of the CALCE frames")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))

    axes[0].scatter(old["cycle"], old["capacity_ah"], s=5, alpha=0.35,
                    color=NOISE, edgecolors="none")
    axes[0].set_title(
        f"(a)  Cycle_Index derivation - {len(old):,} points", loc="left"
    )

    axes[1].scatter(new["cycle"], new["capacity_ah"], s=26, alpha=0.85,
                    color=EFFECT, edgecolors="white", linewidths=0.5)
    axes[1].set_title(
        f"(b)  Full-discharge derivation - {len(new):,} points", loc="left"
    )

    top = max(float(old["capacity_ah"].max()), float(new["capacity_ah"].max())) * 1.08
    for ax in axes:
        ax.set_xlabel("cycle index")
        ax.set_ylim(0, top)
    axes[0].set_ylabel("discharge capacity (Ah)")

    fig.suptitle(
        f"Cell {cell_id}: alternating full and partial discharges vs graded full discharges",
        fontsize=11.5, fontweight="bold",
    )
    fig.text(
        0.5, 0.008,
        "(a) mixes full and partial discharges, so no smooth function of cycle "
        "number can track it; (b) admits only graded full discharges.",
        ha="center", fontsize=8.6, color="#444",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))

    path = OUT / "manuscript_fig3_target_derivations.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> int:
    for builder in (figure_1, figure_2, figure_3):
        path = builder()
        print(f"wrote {path.relative_to(REPO)}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
