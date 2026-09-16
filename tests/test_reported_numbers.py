"""Pin every figure quoted in the documentation to the artifact it came from.

This project's whole argument is that a claim must not drift away from its
evidence. The prose was the one place that rule was enforced by re-reading
rather than by anything structural, and re-reading is exactly the mechanism
this project exists to distrust. Figures went stale repeatedly and silently:
the SOH ceiling moved 0.612 -> 0.569 when B0041 left the screen, the CALCE
table in `final_report.md` survived a full re-run past its own expiry, and the
NASA benchmark table was superseded twice. Each was caught by a human noticing.

Each claim below names the document and section it protects, so a failure reads
"README's validation table claims 0.569; the artifact says X" rather than going
red with no explanation. There are two assertions per claim:

`test_document_still_quotes_the_figure` -- the number is still literally in the
prose. Catches an artifact being regenerated without the doc following.

`test_artifact_still_supports_the_figure` -- the artifact still yields it.
Catches the prose being edited away from the evidence.

**Several derived figures are recomputed here rather than read from a column**,
and their recipes were previously written down nowhere at all:

- The LOCO interquartile range (0.724) is the median within-level IQR of `gap`
  over levels carrying a full 24 draws. Pooling all levels gives 0.7102,
  because `calce` at 6 cohorts and `nasa` at 9 carry 2 draws each and an IQR
  over two points is not a spread. The exclusion is correct and was implicit.
- The paired between-method difference (-0.351) is the median of
  `svr_rbf - age_linear` on `loco_r2`, paired on identical subframes.
- ADR 0013's four figures apply those same two recipes to the corrected-target
  sweep, restricted to the coverage levels both runs draw fully, plus their
  ratio -- which is the quantity that survived the target correction and so is
  pinned rather than derived from the other two at reading time.

If a recipe is ever changed, these tests fail with the recipe stated, which is
the point: the number is reproducible or it is not a number.

The recipes pool across every row of the file they are handed, which is why
ADR 0013's re-run lives in its own artifact and why one test asserts the two
files never merge.

This module is deliberately pure ASCII. The documents it guards contain em
dashes, minus signs and superscript twos; every pattern below stops short of
them, so that an encoding accident in a regex cannot be mistaken for drift.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "reports" / "metrics"

# Documents quote 3 decimal places, so a rounded match must be allowed half a
# unit in the last place, plus a hair.
TOL = 0.0006

# Some figures are quoted as a whole percent ("56% of held-out cohorts"), which
# is half a point of slack, not half a thousandth. Using TOL for those fails on
# a correctly-rounded 55.56 -> 56, which is what happened on this file's first
# run.
TOL_WHOLE_PCT = 0.5

# Figures quoted to two decimals as a percent ("4.21%").
TOL_PCT_2DP = 0.006


# --------------------------------------------------------------------------
# artifact readers
# --------------------------------------------------------------------------


def _csv(rel: str) -> pd.DataFrame:
    path = METRICS / rel
    if not path.exists():
        pytest.fail(f"tracked artifact is missing: reports/metrics/{rel}")
    return pd.read_csv(path)


def signal(rel: str, target: str = "soh") -> float:
    """Noise ceiling for `target` from a benchmark signal report."""
    frame = _csv(rel)
    row = frame.loc[frame["target"] == target]
    assert len(row) == 1, f"{rel}: expected one row for target={target}"
    return float(row["max_attainable_r2"].iloc[0])


def bench(rel: str, method: str, column: str, target: str = "soh") -> float:
    """One cell of a benchmark results table."""
    frame = _csv(rel)
    row = frame.loc[(frame["target"] == target) & (frame["method"] == method)]
    assert len(row) == 1, f"{rel}: expected one row for {method}/{target}"
    return float(row[column].iloc[0])


def coverage(method: str, split: str, column: str) -> float:
    frame = _csv("coverage_summary.csv")
    row = frame.loc[(frame["method"] == method) & (frame["split"] == split)]
    assert len(row) == 1, f"coverage_summary.csv: expected one {method}/{split} row"
    return float(row[column].iloc[0])


# The sweep's two derived figures. Recipes documented in the module docstring.

FULL_DRAW_COUNT = 24


SWEEP_FILE = "coverage_sweep.csv"

# ADR 0013's re-run of the same sweep on the ADR 0012 full-discharge target.
# A separate file on purpose: every recipe below pools across whatever rows it
# is handed, so a third frame inside `coverage_sweep.csv` would have moved the
# three published figures with nothing failing on the substance.
CORRECTED_SWEEP_FILE = "coverage_sweep_full_discharge.csv"

# The coverage levels both runs draw fully. The corrected frame reaches k=6
# and the CALCE arm of the original does not, so a comparison of the two has
# to be restricted to the levels they share or it compares different k.
MATCHED_LEVELS = (3, 4, 5)


def _sweep(rel: str = SWEEP_FILE) -> pd.DataFrame:
    return _csv(rel)


def loco_iqr(
    rel: str = SWEEP_FILE,
    dataset: str | None = None,
    levels: tuple[int, ...] | None = None,
) -> float:
    """Median within-level IQR of the LOCO gap, over fully-drawn levels."""
    table = _sweep(rel)
    if dataset is not None:
        table = table[table["dataset"] == dataset]
    full = table.groupby(["dataset", "n_cohorts"]).filter(lambda g: len(g) >= FULL_DRAW_COUNT)
    if levels is not None:
        full = full[full["n_cohorts"].isin(levels)]
    per_level = full.groupby(["dataset", "n_cohorts"])["gap"].agg(
        lambda s: s.quantile(0.75) - s.quantile(0.25)
    )
    return float(per_level.median())


def _paired_difference(
    rel: str = SWEEP_FILE,
    dataset: str | None = None,
    levels: tuple[int, ...] | None = None,
) -> pd.Series:
    table = _sweep(rel)
    if dataset is not None:
        table = table[table["dataset"] == dataset]
    if levels is not None:
        table = table[table["n_cohorts"].isin(levels)]
    wide = table.pivot_table(
        index=["dataset", "n_cohorts", "repeat"], columns="method", values="loco_r2"
    )
    return (wide["svr_rbf"] - wide["age_linear"]).dropna()


def paired_method_difference(**kwargs: object) -> float:
    return float(_paired_difference(**kwargs).median())  # type: ignore[arg-type]


def nonlinear_win_rate(**kwargs: object) -> float:
    """Fraction of paired draws in which svr_rbf beats the straight line."""
    diff = _paired_difference(**kwargs)  # type: ignore[arg-type]
    return float((diff > 0).mean())


def noise_to_effect_ratio(rel: str, dataset: str | None = None) -> float:
    """ADR 0013's ratio: LOCO spread over the difference it must adjudicate.

    The claim that survived the target correction is this ratio, not either
    number in it, so the ratio is pinned in its own right rather than left to
    be divided out of two other pins.
    """
    spread = loco_iqr(rel, dataset=dataset, levels=MATCHED_LEVELS)
    effect = paired_method_difference(
        rel=rel, dataset=dataset, levels=MATCHED_LEVELS
    )
    return spread / abs(effect)


def detection(pattern: str) -> float:
    """Pull a figure out of the generated detection summary.

    `detection_agreement.csv` is a 10-byte header-only stub -- rule-flag
    agreement was not computable on a cycle-level frame -- so the detection
    numbers live in the generated markdown and are parsed from it here.
    See `test_detection_agreement_csv_is_the_known_empty_stub`.
    """
    text = (METRICS / "detection_summary.md").read_text(encoding="utf-8")
    found = re.search(pattern, text)
    assert found is not None, f"detection_summary.md no longer matches {pattern!r}"
    return float(found.group(1))


# --------------------------------------------------------------------------
# the claims
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """One figure, the prose that quotes it, and the evidence behind it."""

    id: str
    value: float
    extract: Callable[[], float]
    source: str
    quoted_in: tuple[tuple[str, str, str], ...] = field(default=())
    """(document, section, regex that must match the document)."""
    tol: float = TOL

    def __str__(self) -> str:  # pragma: no cover - test id only
        return self.id


def pct(x: float) -> float:
    return x * 100.0


CLAIMS: tuple[Claim, ...] = (
    # -- target noise ceilings -------------------------------------------
    Claim(
        id="nasa-capacity-loss-ceiling",
        value=0.044,
        extract=lambda: signal("benchmark_signal_report.csv", "capacity_loss"),
        source="benchmark_signal_report.csv[capacity_loss].max_attainable_r2",
        quoted_in=(
            ("docs/project_history.md", "Target noise ceilings", r"\*\*0\.044\*\*"),
            ("docs/final_report.md", "S4.10 target definition", r"\*\*0\.044\*\*"),
            ("docs/paper_outline.md", "Result 1", r"\*\*0\.044\*\*"),
            ("docs/manuscript/ress_draft.md", "S5 targets", r"0\.044"),
        ),
    ),
    Claim(
        id="nasa-soh-ceiling",
        value=0.569,
        extract=lambda: signal("benchmark_signal_report.csv", "soh"),
        source="benchmark_signal_report.csv[soh].max_attainable_r2",
        quoted_in=(
            ("docs/project_history.md", "Target noise ceilings", r"0\.569"),
            ("docs/final_report.md", "S4.10 / S4.11", r"0\.569"),
        ),
    ),
    Claim(
        id="calce-ceiling",
        value=0.870,
        extract=lambda: signal("calce/benchmark_signal_report.csv"),
        source="calce/benchmark_signal_report.csv[soh].max_attainable_r2",
        quoted_in=(
            ("docs/final_report.md", "S4.13 CALCE replication", r"0\.870"),
            ("docs/roadmap.md", "Specification defects", r"0\.870"),
            ("docs/paper_outline.md", "Result 3", r"0\.870"),
            ("docs/architecture_diagram_spec.md", "C2", r"0\.870"),
        ),
    ),
    Claim(
        id="calce-full-discharge-ceiling",
        value=0.907,
        extract=lambda: signal("calce_full_discharge/benchmark_signal_report.csv"),
        source="calce_full_discharge/benchmark_signal_report.csv[soh]",
        quoted_in=(
            ("docs/roadmap.md", "Item 0, ADR 0012", r"0\.907"),
            ("docs/paper_outline.md", "full-discharge", r"0\.907"),
            ("docs/manuscript/ress_draft.md", "S5 targets", r"\*\*0\.907\*\*"),
        ),
    ),
    Claim(
        id="calce-full-discharge-cell-count",
        value=22.0,
        extract=lambda: float(
            _csv("calce_full_discharge/benchmark_signal_report.csv")["n_cells"].iloc[0]
        ),
        source="calce_full_discharge/benchmark_signal_report.csv.n_cells",
        quoted_in=(
            ("docs/roadmap.md", "admissibility 19/23 -> 22/22", r"22/22"),
            ("docs/manuscript/ress_draft.md", "S5 admissibility", r"\*\*22/22\*\*"),
        ),
        tol=0.0,
    ),
    # -- NASA benchmark table --------------------------------------------
    Claim(
        id="nasa-xgboost-lobo-r2",
        value=0.732,
        extract=lambda: bench("benchmark_results.csv", "xgboost", "lobo_r2"),
        source="benchmark_results.csv[soh/xgboost].lobo_r2",
        quoted_in=(
            ("docs/project_history.md", "NASA method table", r"0\.732"),
            ("docs/final_report.md", "S4.11 NASA method table", r"0\.732"),
        ),
    ),
    Claim(
        id="nasa-xgboost-loco-r2",
        value=0.459,
        extract=lambda: bench("benchmark_results.csv", "xgboost", "loco_r2"),
        source="benchmark_results.csv[soh/xgboost].loco_r2",
        quoted_in=(
            ("docs/project_history.md", "NASA method table", r"\*\*0\.459\*\*"),
            ("docs/final_report.md", "S4.11 NASA method table", r"\*\*0\.459\*\*"),
        ),
    ),
    Claim(
        id="nasa-xgboost-lobo-mae-pct",
        value=4.21,
        extract=lambda: pct(bench("benchmark_results.csv", "xgboost", "lobo_mae")),
        source="benchmark_results.csv[soh/xgboost].lobo_mae x100",
        quoted_in=(
            ("docs/project_history.md", "NASA method table", r"\*\*4\.21%\*\*"),
            ("docs/final_report.md", "S4.11 NASA method table", r"\*\*4\.21%\*\*"),
        ),
        tol=TOL_PCT_2DP,
    ),
    Claim(
        id="nasa-xgboost-loco-mae-pct",
        value=7.62,
        extract=lambda: pct(bench("benchmark_results.csv", "xgboost", "loco_mae")),
        source="benchmark_results.csv[soh/xgboost].loco_mae x100",
        quoted_in=(
            ("docs/project_history.md", "NASA method table", r"\*\*7\.62%\*\*"),
            ("docs/final_report.md", "S4.11 NASA method table", r"\*\*7\.62%\*\*"),
        ),
        tol=TOL_PCT_2DP,
    ),
    Claim(
        id="nasa-age-linear-loco-r2",
        value=0.406,
        extract=lambda: bench("benchmark_results.csv", "age_linear", "loco_r2"),
        source="benchmark_results.csv[soh/age_linear].loco_r2",
        quoted_in=(
            ("docs/project_history.md", "behaviour adds ~0.05 R2 over cycle count", r"0\.406"),
            ("docs/final_report.md", "S4.11 commentary", r"0\.406"),
        ),
    ),
    # -- CALCE benchmark table -------------------------------------------
    Claim(
        id="calce-rf-lobo-r2",
        value=0.938,
        extract=lambda: bench("calce/benchmark_results.csv", "random_forest", "lobo_r2"),
        source="calce/benchmark_results.csv[soh/random_forest].lobo_r2",
        quoted_in=(("docs/final_report.md", "S4.13 CALCE table", r"0\.938"),),
    ),
    Claim(
        id="calce-rf-loco-r2",
        value=0.693,
        extract=lambda: bench("calce/benchmark_results.csv", "random_forest", "loco_r2"),
        source="calce/benchmark_results.csv[soh/random_forest].loco_r2",
        quoted_in=(
            ("docs/final_report.md", "S4.13 CALCE table", r"\*\*0\.693\*\*"),
            ("docs/project_history.md", "withdrawn claims", r"0\.693"),
        ),
    ),
    Claim(
        id="calce-rf-loco-mae-pct",
        value=8.59,
        extract=lambda: pct(bench("calce/benchmark_results.csv", "random_forest", "loco_mae")),
        source="calce/benchmark_results.csv[soh/random_forest].loco_mae x100",
        quoted_in=(
            ("docs/final_report.md", "S4.13 CALCE table", r"\*\*8\.59%\*\*"),
            ("docs/roadmap.md", "SOH < 3% spec defect", r"\*\*8\.59%\*\*"),
            ("docs/architecture_diagram_spec.md", "C2", r"\*\*8\.59%\*\*"),
        ),
        tol=TOL_PCT_2DP,
    ),
    # The LOBO/LOCO table that both the roadmap's spec defect and the diagram
    # spec's C2 argue from. It is the poster's `< 3%` target restated, quoted
    # in two documents, and must not go stale in only one of them.
    Claim(
        id="calce-lstm-lobo-mae-pct",
        value=2.26,
        extract=lambda: pct(bench("calce/benchmark_results.csv", "lstm", "lobo_mae")),
        source="calce/benchmark_results.csv[soh/lstm].lobo_mae x100",
        quoted_in=(
            ("docs/roadmap.md", "SOH < 3% spec defect", r"\*\*2\.26%\*\*"),
            ("docs/architecture_diagram_spec.md", "C2", r"\*\*2\.26%\*\*"),
        ),
        tol=TOL_PCT_2DP,
    ),
    Claim(
        id="calce-lstm-loco-mae-pct",
        value=15.68,
        extract=lambda: pct(bench("calce/benchmark_results.csv", "lstm", "loco_mae")),
        source="calce/benchmark_results.csv[soh/lstm].loco_mae x100",
        quoted_in=(
            ("docs/roadmap.md", "SOH < 3% spec defect", r"15\.68%"),
            ("docs/architecture_diagram_spec.md", "C2", r"15\.68%"),
        ),
        tol=TOL_PCT_2DP,
    ),
    Claim(
        id="calce-gpr-lobo-mae-pct",
        value=2.93,
        extract=lambda: pct(bench("calce/benchmark_results.csv", "gpr_matern", "lobo_mae")),
        source="calce/benchmark_results.csv[soh/gpr_matern].lobo_mae x100",
        quoted_in=(
            ("docs/roadmap.md", "SOH < 3% spec defect", r"\*\*2\.93%\*\*"),
            ("docs/architecture_diagram_spec.md", "C2", r"\*\*2\.93%\*\*"),
        ),
        tol=TOL_PCT_2DP,
    ),
    Claim(
        id="calce-gpr-loco-mae-pct",
        value=10.06,
        extract=lambda: pct(bench("calce/benchmark_results.csv", "gpr_matern", "loco_mae")),
        source="calce/benchmark_results.csv[soh/gpr_matern].loco_mae x100",
        quoted_in=(
            ("docs/roadmap.md", "SOH < 3% spec defect", r"10\.06%"),
            ("docs/architecture_diagram_spec.md", "C2", r"10\.06%"),
        ),
        tol=TOL_PCT_2DP,
    ),
    Claim(
        id="calce-rf-lobo-mae-pct",
        value=3.04,
        extract=lambda: pct(bench("calce/benchmark_results.csv", "random_forest", "lobo_mae")),
        source="calce/benchmark_results.csv[soh/random_forest].lobo_mae x100",
        quoted_in=(
            ("docs/roadmap.md", "SOH < 3% spec defect", r"3\.04%"),
            ("docs/architecture_diagram_spec.md", "C2", r"3\.04%"),
        ),
        tol=TOL_PCT_2DP,
    ),
    # -- conformal coverage ----------------------------------------------
    Claim(
        id="conformal-lobo-median-coverage",
        value=0.970,
        extract=lambda: coverage("elasticnet", "LOBO", "median_coverage"),
        source="coverage_summary.csv[elasticnet/LOBO].median_coverage",
        quoted_in=(
            ("docs/final_report.md", "S4.12 coverage table", r"0\.970"),
            ("docs/manuscript/ress_draft.md", "S7 coverage", r"0\.970"),
        ),
    ),
    Claim(
        id="conformal-loco-median-coverage",
        value=0.824,
        extract=lambda: coverage("elasticnet", "LOCO", "median_coverage"),
        source="coverage_summary.csv[elasticnet/LOCO].median_coverage",
        quoted_in=(
            ("docs/final_report.md", "S4.12 coverage table", r"0\.824"),
            ("docs/paper_outline.md", "Result 4", r"0\.824"),
            ("docs/manuscript/ress_draft.md", "S7 coverage", r"0\.824"),
        ),
    ),
    Claim(
        id="conformal-loco-worst-coverage",
        value=0.262,
        extract=lambda: coverage("elasticnet", "LOCO", "min_coverage"),
        source="coverage_summary.csv[elasticnet/LOCO].min_coverage",
        quoted_in=(
            ("docs/project_history.md", "conformal undercoverage", r"0\.262"),
            ("docs/final_report.md", "S4.12 coverage table", r"0\.262"),
            ("docs/paper_outline.md", "Result 4", r"0\.262"),
        ),
    ),
    Claim(
        id="conformal-loco-fraction-below-nominal-pct",
        value=56.0,
        extract=lambda: pct(coverage("elasticnet", "LOCO", "fraction_below_nominal")),
        source="coverage_summary.csv[elasticnet/LOCO].fraction_below_nominal x100",
        quoted_in=(
            ("docs/project_history.md", "conformal undercoverage", r"56% of held-out cohorts"),
            ("docs/final_report.md", "S4.12 coverage table", r"56%"),
            ("docs/paper_outline.md", "Result 4", r"\*\*56% of held-out cohorts"),
            ("docs/manuscript/ress_draft.md", "S7 coverage", r"56% of held-out"),
        ),
        tol=TOL_WHOLE_PCT,
    ),
    # -- the sweep's derived figures --------------------------------------
    Claim(
        id="loco-iqr",
        value=0.724,
        extract=loco_iqr,
        source="coverage_sweep.csv -> median within-level IQR of gap, levels n>=24",
        quoted_in=(
            ("docs/project_history.md", "LOCO noisier than the effect", r"\*\*0\.724 R"),
            ("docs/final_report.md", "S6 estimator variance", r"\*\*0\.724 R"),
            ("docs/paper_outline.md", "Result 5", r"0\.724 R"),
            ("docs/manuscript/ress_draft.md", "Abstract + S6", r"0\.724 R"),
            ("docs/architecture_diagram_spec.md", "C5", r"0\.724 R2"),
        ),
    ),
    Claim(
        id="paired-method-difference",
        value=-0.351,
        extract=paired_method_difference,
        source="coverage_sweep.csv -> median(svr_rbf - age_linear) on loco_r2",
        quoted_in=(
            ("docs/project_history.md", "LOCO noisier than the effect", r"\*\*0\.351 R"),
            ("docs/final_report.md", "S6 estimator variance", r"0\.351"),
            ("docs/manuscript/ress_draft.md", "Abstract + S6", r"0\.351"),
            ("docs/architecture_diagram_spec.md", "C5", r"0\.351 R2"),
        ),
    ),
    Claim(
        id="nonlinear-win-rate-pct",
        value=25.0,
        extract=lambda: pct(nonlinear_win_rate()),
        source="coverage_sweep.csv -> fraction of paired draws svr_rbf > age_linear",
        quoted_in=(("docs/project_history.md", "LOCO noisier than the effect", r"only 25% of draws"),),
        tol=TOL_WHOLE_PCT,
    ),
    # -- the same sweep on the ADR 0012 corrected target (ADR 0013) --------
    #
    # Same cells, same features, same two methods, same machinery; only the
    # target derivation differs. All four are restricted to the coverage
    # levels both runs draw fully, because the corrected frame reaches k=6
    # and the original's CALCE arm does not.
    Claim(
        id="corrected-target-loco-iqr",
        value=0.296,
        extract=lambda: loco_iqr(CORRECTED_SWEEP_FILE, levels=MATCHED_LEVELS),
        source=(
            "coverage_sweep_full_discharge.csv -> median within-level IQR of "
            "gap, levels n>=24, k in (3,4,5)"
        ),
        quoted_in=(
            (
                "docs/adr/0013-estimator-variance-on-the-corrected-target.md",
                "What the corrected target says",
                r"\*\*0\.296\*\*",
            ),
            ("docs/roadmap.md", "0a estimator variance", r"\*\*0\.296\*\*"),
        ),
    ),
    Claim(
        id="corrected-target-paired-difference",
        value=-0.110,
        extract=lambda: paired_method_difference(
            rel=CORRECTED_SWEEP_FILE, levels=MATCHED_LEVELS
        ),
        source=(
            "coverage_sweep_full_discharge.csv -> median(svr_rbf - age_linear) "
            "on loco_r2, k in (3,4,5)"
        ),
        quoted_in=((
            "docs/adr/0013-estimator-variance-on-the-corrected-target.md",
            "What the corrected target says",
            r"-0\.110",
        ),),
    ),
    # The ratio is the claim that survived the correction, so it is pinned in
    # its own right rather than left to be divided out of the two pins above.
    Claim(
        id="corrected-target-noise-to-effect-ratio",
        value=2.70,
        extract=lambda: noise_to_effect_ratio(CORRECTED_SWEEP_FILE),
        source="coverage_sweep_full_discharge.csv -> loco_iqr / |paired difference|",
        quoted_in=(
            (
                "docs/adr/0013-estimator-variance-on-the-corrected-target.md",
                "What the corrected target says",
                r"\*\*2\.70\*\*",
            ),
            ("docs/roadmap.md", "0a estimator variance", r"2\.66 versus 2\.70"),
        ),
        tol=0.006,
    ),
    Claim(
        id="artifact-target-noise-to-effect-ratio",
        value=2.66,
        extract=lambda: noise_to_effect_ratio(SWEEP_FILE, dataset="calce"),
        source="coverage_sweep.csv[calce] -> loco_iqr / |paired difference|",
        quoted_in=((
            "docs/adr/0013-estimator-variance-on-the-corrected-target.md",
            "What the corrected target says",
            r"\*\*2\.66\*\*",
        ),),
        tol=0.006,
    ),
    # -- unsupervised detection (ADR 0011) --------------------------------
    Claim(
        id="isolation-forest-fade-p",
        value=1.0000,
        extract=lambda: detection(r"U = \d+, p = ([\d.]+), 20 cells"),
        source="detection_summary.md isolation_forest Mann-Whitney p",
        quoted_in=(
            ("docs/project_history.md", "outliers are not harmful", r"p = 1\.00"),
            ("docs/architecture_diagram_spec.md", "C1 detector table", r"1\.0000"),
        ),
    ),
    Claim(
        id="lof-fade-p",
        value=0.8999,
        extract=lambda: detection(r"U = \d+, p = ([\d.]+), 29 cells"),
        source="detection_summary.md local_outlier_factor Mann-Whitney p",
        quoted_in=(
            ("docs/project_history.md", "outliers are not harmful", r"0\.90"),
            ("docs/architecture_diagram_spec.md", "C1 detector table", r"0\.8999"),
        ),
    ),
    Claim(
        id="cluster-silhouette",
        value=0.5414,
        extract=lambda: detection(r"best k = \d+, silhouette = ([\d.]+)"),
        source="detection_summary.md separability, real silhouette",
        quoted_in=(("docs/architecture_diagram_spec.md", "C1 validated list", r"0\.541"),),
        tol=0.0005,
    ),
    Claim(
        id="cluster-silhouette-null",
        value=0.1534,
        extract=lambda: detection(r"null \(uniform, same shape, p95\) = ([\d.]+)"),
        source="detection_summary.md separability, null p95",
        quoted_in=(("docs/architecture_diagram_spec.md", "C1 validated list", r"null 0\.153"),),
        tol=0.0005,
    ),
    Claim(
        id="detector-overlap",
        value=34.0,
        extract=lambda: detection(r"Detector overlap: (\d+) of \d+ in common"),
        source="detection_summary.md detector overlap",
        quoted_in=(
            ("docs/project_history.md", "outliers are not harmful", r"34 of 135 flags"),
            ("docs/architecture_diagram_spec.md", "C1", r"34 of their 135 flags"),
        ),
        tol=0.0,
    ),
    Claim(
        id="flagged-cycle-count",
        value=135.0,
        extract=lambda: detection(r"Detector overlap: \d+ of (\d+) in common"),
        source="detection_summary.md flag count",
        quoted_in=(
            ("docs/project_history.md", "outliers are not harmful", r"of 135 flags"),
            ("docs/architecture_diagram_spec.md", "C1 detector table", r"135 / 2682"),
        ),
        tol=0.0,
    ),
)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("claim", CLAIMS, ids=str)
def test_artifact_still_supports_the_figure(claim: Claim) -> None:
    """The evidence still yields the number the prose quotes."""
    actual = claim.extract()
    assert abs(actual - claim.value) <= claim.tol, (
        f"\n  claim   : {claim.id}\n"
        f"  docs say: {claim.value}\n"
        f"  artifact: {actual}  (from {claim.source})\n"
        f"  quoted  : "
        + "; ".join(f"{doc} - {section}" for doc, section, _ in claim.quoted_in)
        + "\n  Regenerate the doc figure or explain the change in an ADR."
    )


@pytest.mark.parametrize(
    ("claim", "doc", "section", "pattern"),
    [(c, d, s, p) for c in CLAIMS for d, s, p in c.quoted_in],
    ids=[f"{c.id}-{d}" for c in CLAIMS for d, _, _ in c.quoted_in],
)
def test_document_still_quotes_the_figure(
    claim: Claim, doc: str, section: str, pattern: str
) -> None:
    """The prose still carries the figure, in the section that argues from it."""
    path = REPO / doc
    assert path.exists(), f"{doc} is missing; {claim.id} has nothing to protect"
    text = path.read_text(encoding="utf-8")
    assert re.search(pattern, text) is not None, (
        f"\n  {doc} ({section}) no longer quotes {claim.id} = {claim.value}.\n"
        f"  Expected to match: {pattern}\n"
        f"  Evidence: {claim.source}\n"
        f"  Either the prose was edited away from its evidence, or this claim\n"
        f"  moved and the pin needs updating."
    )


def test_every_referenced_artifact_is_tracked() -> None:
    """A claim pointing at an untracked file protects nothing."""
    missing = [
        rel
        for rel in (
            "benchmark_signal_report.csv",
            "benchmark_results.csv",
            "calce/benchmark_signal_report.csv",
            "calce/benchmark_results.csv",
            "calce_full_discharge/benchmark_signal_report.csv",
            "coverage_summary.csv",
            "coverage_sweep.csv",
            "coverage_sweep_full_discharge.csv",
            "detection_summary.md",
        )
        if not (METRICS / rel).exists()
    ]
    assert not missing, f"tracked artifacts missing: {missing}"


def test_detection_agreement_csv_is_the_known_empty_stub() -> None:
    """Guard against anyone quoting a figure out of an empty file.

    Rule-flag agreement was not computable on a cycle-level frame -- the
    boolean flags live on row-level telemetry, which the frame aggregates
    away -- so the writer emitted a header and no rows. If it ever gains
    rows, that is a real result and the detection claims above should be
    re-sourced from it rather than parsed out of markdown.
    """
    path = METRICS / "detection_agreement.csv"
    if not path.exists():
        pytest.skip("detection_agreement.csv not present")
    frame = pd.read_csv(path)
    assert frame.empty, (
        "detection_agreement.csv now has rows. Rule-flag agreement became "
        "computable; re-source the detection claims from it and update ADR 0011."
    )


def test_the_two_sweeps_stay_in_separate_artifacts() -> None:
    """The published figures pool every row, so the frames must not mix.

    `loco_iqr`, `paired_method_difference` and `nonlinear_win_rate` pool
    across whatever `coverage_sweep.csv` holds. Adding the corrected-target
    frame to it would move all three published numbers while every claim
    above still "passed" against the moved artifact -- the exact silent drift
    this module exists to prevent. `run_coverage_sweep.py` refuses to do it;
    this asserts the outcome rather than trusting the guard.
    """
    original = set(_sweep(SWEEP_FILE)["dataset"].unique())
    assert original == {"nasa", "calce"}, (
        f"coverage_sweep.csv now holds {sorted(original)}. The published "
        "0.724 / -0.351 / 25% pool every row of it and have moved. Re-run "
        "with --out-prefix and restore the two-frame artifact."
    )
    corrected = set(_sweep(CORRECTED_SWEEP_FILE)["dataset"].unique())
    assert corrected == {"calce_full_discharge"}, (
        f"coverage_sweep_full_discharge.csv holds {sorted(corrected)}; "
        "ADR 0013 compares one frame against one frame."
    )


def test_the_target_correction_shrinks_the_estimator_spread() -> None:
    """ADR 0013's finding, asserted as a direction rather than a number.

    The magnitudes moved by roughly 3x between the two target derivations
    while the ratio between them did not. Both halves of that are the
    finding, so both are asserted: the spread must be substantially smaller
    on the corrected target, and the noise-to-effect ratio must not be.
    """
    original = loco_iqr(SWEEP_FILE, dataset="calce", levels=MATCHED_LEVELS)
    corrected = loco_iqr(CORRECTED_SWEEP_FILE, levels=MATCHED_LEVELS)
    assert corrected < original / 2, (
        f"the corrected-target LOCO IQR ({corrected:.4f}) is no longer much "
        f"smaller than the artifact target's ({original:.4f}). ADR 0013's "
        "first conclusion rests on that gap; re-check it."
    )
    ratios = (
        noise_to_effect_ratio(SWEEP_FILE, dataset="calce"),
        noise_to_effect_ratio(CORRECTED_SWEEP_FILE),
    )
    assert abs(ratios[0] - ratios[1]) < 0.25, (
        f"the noise-to-effect ratios have diverged: {ratios[0]:.2f} vs "
        f"{ratios[1]:.2f}. ADR 0013 claims the ratio is the part that "
        "survives the target correction; that claim now needs revisiting."
    )


def test_loco_iqr_recipe_excludes_degenerate_levels() -> None:
    """The exclusion behind 0.724 is load-bearing, so assert it explicitly.

    Pooling every level gives 0.7102, because two levels carry 2 draws each
    and an interquartile range over two points is not a spread. This test
    fails if someone "simplifies" the recipe back to pooling everything.
    """
    table = _sweep()
    sizes = table.groupby(["dataset", "n_cohorts"]).size()
    degenerate = sizes[sizes < FULL_DRAW_COUNT]
    assert not degenerate.empty, (
        "no under-drawn levels remain; if the sweep was re-run to full "
        "coverage, the exclusion is moot and loco_iqr() can be simplified."
    )

    pooled = table.groupby(["dataset", "n_cohorts"])["gap"].agg(
        lambda s: s.quantile(0.75) - s.quantile(0.25)
    )
    assert abs(float(pooled.median()) - loco_iqr()) > 0.01, (
        "the exclusion no longer changes the answer; re-check whether the "
        "published 0.724 still needs it."
    )
