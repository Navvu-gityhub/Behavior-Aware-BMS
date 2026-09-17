"""Expose what has actually been validated, read from the tracked artifacts.

The dashboard displays health indices and remaining-life figures. Without this
endpoint a viewer has no way to learn, from the product itself, that those
numbers were tested against real degradation and largely failed - they would
have to go and read `docs/final_report.md`. Most will not.

So the evidence is served beside the scores, and it is read from the same CSVs
under `reports/metrics/` that `tests/test_reported_numbers.py` pins the prose
to. Nothing here is hardcoded: if a benchmark is re-run and a figure moves, this
endpoint moves with it, and if an artifact is missing the endpoint says so
rather than substituting a plausible number.

One rule is enforced structurally rather than by convention: **no R2 is
returned without the noise ceiling of its target beside it.** A model R2 of
0.406 means something entirely different against a ceiling of 0.044 than
against one of 0.907, and reporting the first without the second is how this
project misread its own results for months (see final_report section 4.10).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[3]
METRICS = REPO / "reports" / "metrics"

router = APIRouter()


class TargetCeilingOut(BaseModel):
    """A target and the maximum R2 anything could attain against it."""

    target: str
    n_cells: int
    n_rows: int
    signal_fraction: float = Field(
        description="Isotonic signal fraction: the share of variance that is "
                    "not measurement noise."
    )
    max_attainable_r2: float = Field(
        description="The ceiling. A reported R2 above this is a bug, not a "
                    "result."
    )
    verdict: str


class MethodResultOut(BaseModel):
    method: str
    family: str
    target: str
    status: str
    lobo_r2: float | None = None
    loco_r2: float | None = None
    loco_minus_lobo: float | None = None
    r2_ceiling: float | None = Field(
        default=None,
        description="The target's noise ceiling. Never read an R2 without it.",
    )
    promoted: bool | None = None


class CoverageGroupOut(BaseModel):
    method: str
    split: str
    n_folds: int
    median_coverage: float
    min_coverage: float
    worst_group: str | None = None
    nominal: float
    fraction_below_nominal: float


class ValidationSummaryOut(BaseModel):
    """Everything the dashboard needs to state what has been tested."""

    available: bool
    missing_artifacts: list[str] = []
    target_ceilings: list[TargetCeilingOut] = []
    methods: list[MethodResultOut] = []
    coverage: list[CoverageGroupOut] = []
    split_types: list[str] = []
    metric: str = "R2 against the training-set mean, per held-out group"
    headline: str = ""
    limitations: list[str] = []


#: Stated here rather than derived, because they are editorial judgements about
#: what the evidence supports, not quantities in a CSV. Each one is traceable to
#: a section of `docs/final_report.md` or an ADR.
LIMITATIONS: tuple[str, ...] = (
    "The original heuristic scores showed no significant relationship with "
    "measured capacity fade (p > 0.13, n = 33).",
    "The per-cycle `capacity_loss` target has an isotonic signal fraction of "
    "0.044, so an R2 of 0.044 was the maximum obtainable against it. Earlier "
    "null results were substantially a property of the target.",
    "Measured capacity carries a reversible thermal offset that inverts the "
    "temperature signal across cohorts (apparent SOH 0.810 at 4 C vs 0.999 at "
    "44 C before degradation is possible).",
    "Every method loses skill under leave-one-cohort-out relative to "
    "leave-one-cell-out, in every frame tested, without exception.",
    "Six method-ranking claims were made from single-frame runs and all six "
    "were withdrawn. The instability is the finding; no ranking is asserted.",
    "The leave-one-cohort-out estimate's spread across cohort draws is about "
    "2.7x the between-method difference it is used to adjudicate, on both "
    "target derivations tested.",
    "No model in this system is promoted for prediction. The dashboard's "
    "health index, risk score and RUL are rule-based scores, not fitted "
    "predictors.",
)

HEADLINE = (
    "This project's validated result is methodological, not predictive: at the "
    "sample sizes standard battery datasets provide, a single "
    "leave-one-cohort-out estimate cannot distinguish between candidate "
    "methods and must be reported as an interval over cohort draws."
)


def _read(name: str) -> pd.DataFrame | None:
    path = METRICS / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _verdict(ceiling: float) -> str:
    """Plain-language reading of a noise ceiling.

    The bands are deliberately unflattering in the middle. A ceiling of 0.569
    means roughly 43% of the target's variance is measurement noise, and calling
    that "well-conditioned" is how a reader ends up treating an R2 of 0.46
    against it as a near-perfect fit.
    """
    if ceiling < 0.1:
        return f"{(1 - ceiling):.0%} measurement noise - not a usable target"
    if ceiling < 0.4:
        return f"weak: {(1 - ceiling):.0%} of variance is noise"
    if ceiling < 0.75:
        return f"moderate: {(1 - ceiling):.0%} of variance is noise"
    return f"well-conditioned: only {(1 - ceiling):.0%} noise"


@router.get(
    "/validation/summary",
    response_model=ValidationSummaryOut,
    tags=["validation"],
    summary="What has been validated, read from tracked artifacts",
)
def validation_summary() -> ValidationSummaryOut:
    signal = _read("benchmark_signal_report.csv")
    results = _read("benchmark_results.csv")
    coverage = _read("coverage_summary.csv")

    missing = [
        name
        for name, frame in (
            ("benchmark_signal_report.csv", signal),
            ("benchmark_results.csv", results),
            ("coverage_summary.csv", coverage),
        )
        if frame is None
    ]
    if signal is None or results is None:
        return ValidationSummaryOut(
            available=False,
            missing_artifacts=missing,
            headline=HEADLINE,
            limitations=list(LIMITATIONS),
        )

    ceilings = {
        str(row["target"]): float(row["max_attainable_r2"])
        for _, row in signal.iterrows()
    }

    target_ceilings = [
        TargetCeilingOut(
            target=str(row["target"]),
            n_cells=int(row["n_cells"]),
            n_rows=int(row["n_rows"]),
            signal_fraction=float(row["signal_fraction"]),
            max_attainable_r2=float(row["max_attainable_r2"]),
            verdict=_verdict(float(row["max_attainable_r2"])),
        )
        for _, row in signal.iterrows()
    ]

    methods: list[MethodResultOut] = []
    for _, row in results.iterrows():
        target = str(row["target"])
        lobo = row.get("lobo_r2")
        loco = row.get("loco_r2")
        methods.append(
            MethodResultOut(
                method=str(row["method"]),
                family=str(row.get("family", "")),
                target=target,
                status=str(row.get("status", "")),
                lobo_r2=None if pd.isna(lobo) else float(lobo),
                loco_r2=None if pd.isna(loco) else float(loco),
                loco_minus_lobo=(
                    None if pd.isna(lobo) or pd.isna(loco)
                    else float(loco) - float(lobo)
                ),
                # The endpoint's one structural rule: the ceiling travels with
                # the R2 it bounds, so a client cannot render one without the
                # other by accident.
                r2_ceiling=ceilings.get(target),
                promoted=(
                    None if pd.isna(row.get("promoted"))
                    else bool(row.get("promoted"))
                ),
            )
        )

    groups: list[CoverageGroupOut] = []
    if coverage is not None:
        for _, row in coverage.iterrows():
            worst = row.get("worst_group")
            groups.append(
                CoverageGroupOut(
                    method=str(row["method"]),
                    split=str(row["split"]),
                    n_folds=int(row["n_folds"]),
                    median_coverage=float(row["median_coverage"]),
                    min_coverage=float(row["min_coverage"]),
                    worst_group=None if pd.isna(worst) else str(worst),
                    nominal=float(row["nominal"]),
                    fraction_below_nominal=float(row["fraction_below_nominal"]),
                )
            )

    return ValidationSummaryOut(
        available=True,
        missing_artifacts=missing,
        target_ceilings=target_ceilings,
        methods=methods,
        coverage=groups,
        split_types=sorted({str(row["split"]) for _, row in coverage.iterrows()})
        if coverage is not None else [],
        headline=HEADLINE,
        limitations=list(LIMITATIONS),
    )


@router.get(
    "/validation/ceilings",
    response_model=list[TargetCeilingOut],
    tags=["validation"],
    summary="Target noise ceilings on their own",
)
def target_ceilings() -> list[TargetCeilingOut]:
    signal = _read("benchmark_signal_report.csv")
    if signal is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "benchmark_signal_report.csv is not present. Run "
                "`python scripts/run_benchmark_study.py` to produce it; this "
                "endpoint will not substitute a default ceiling."
            ),
        )
    return [
        TargetCeilingOut(
            target=str(row["target"]),
            n_cells=int(row["n_cells"]),
            n_rows=int(row["n_rows"]),
            signal_fraction=float(row["signal_fraction"]),
            max_attainable_r2=float(row["max_attainable_r2"]),
            verdict=_verdict(float(row["max_attainable_r2"])),
        )
        for _, row in signal.iterrows()
    ]
