"""Run the validation suite end to end and record what happened.

One entry point for the sequence that `docs/final_report.md` Appendix A
documents as a list of commands. This script **orchestrates** those commands; it
does not reimplement any of them. No evaluation protocol, cohort split, target
definition, feature set, model set or metric is defined here, deliberately: a
runner that could change a result would be a second place for the protocol to
live, and the point of this file is that there is only one.

    python scripts/run_validation_suite.py            # full suite
    python scripts/run_validation_suite.py --quick    # CI-sized subset
    python scripts/run_validation_suite.py --list     # show stages, run nothing

Most stages read the tracked CSVs under `reports/metrics/`, so they run from a
clean checkout with no raw telemetry. The few that need raw archives are
declared `needs_raw` and are **skipped with a reason** rather than failed, since
an absent dataset is not a broken pipeline.

Writes `reports/metrics/validation_manifest.json` (machine-readable: stage,
status, duration, exit code, artifacts and their hashes) and
`reports/metrics/validation_summary.md` beside it.

A non-zero exit means a stage that should have run did not succeed. Skips do
not fail the run; they are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "reports" / "metrics"
PYTHON = sys.executable

MANIFEST = METRICS / "validation_manifest.json"
SUMMARY = METRICS / "validation_summary.md"


@dataclass(frozen=True)
class Stage:
    """One documented reproduction command.

    `section` names the part of `docs/final_report.md` the stage supports, so a
    failure points at the prose it invalidates rather than only at a script.
    """

    name: str
    section: str
    command: tuple[str, ...]
    quick: bool = True
    needs_raw: bool = False
    raw_hint: str = ""
    artifacts: tuple[str, ...] = field(default=())


#: Ordered by dependency. Everything here is already documented in Appendix A;
#: this tuple is that appendix, made executable.
STAGES: tuple[Stage, ...] = (
    Stage(
        name="pipeline-smoke",
        section="S2 / S3 end-to-end pipeline",
        command=(PYTHON, "main.py", "--dashboard-out", "reports/validation_dashboard.html"),
    ),
    Stage(
        name="threshold-audit",
        section="S4.7 threshold reachability",
        command=(PYTHON, "scripts/audit_threshold_reachability.py"),
        artifacts=("risk_term_variability.csv",),
    ),
    Stage(
        name="health-index-versions",
        section="S4.8 health index v1 vs v2, LOBO + LOCO",
        command=(PYTHON, "scripts/validate_health_index_versions.py"),
        artifacts=(
            "health_version_task_a_ranking.csv",
            "health_version_task_b_cv.csv",
        ),
    ),
    Stage(
        name="horizon-regression",
        section="S4.5 longer prediction horizons",
        command=(PYTHON, "scripts/fit_horizon_regression_model.py"),
        artifacts=("horizon_regression_summary.csv",),
    ),
    Stage(
        name="mixed-effects",
        section="S4.6 mixed-effects identifiability",
        command=(PYTHON, "scripts/fit_mixed_effects_model.py"),
        artifacts=("mixed_effects_diagnostics.csv",),
    ),
    Stage(
        name="conformal-coverage",
        section="S4.12 / S5.5 coverage within vs across protocol",
        command=(
            PYTHON, "scripts/run_coverage_study.py",
            "--methods", "age_linear", "elasticnet",
        ),
        artifacts=("coverage_folds.csv", "coverage_summary.csv"),
    ),
    Stage(
        name="benchmark-study",
        section="S4.10-S4.12 target ceilings and the method benchmark",
        # --quick restricts to the reference methods. The full sweep refits six
        # estimators across 42 folds and takes ~14 min per target, which is why
        # it is excluded from the quick path rather than merely slower there.
        command=(
            PYTHON, "scripts/run_benchmark_study.py",
            "--targets", "capacity_loss", "cumulative_fade",
        ),
        quick=False,
        artifacts=(
            "benchmark_results.csv",
            "benchmark_signal_report.csv",
            "benchmark_cell_screen.csv",
        ),
    ),
    Stage(
        name="benchmark-study-quick",
        section="S4.10-S4.12 (reference methods only)",
        # `--out` is not optional here. Without it this writes to the same path
        # as the full study and replaces the twelve-method results with a
        # reference-methods-only subset, which silently destroys the artifact
        # the published xgboost and noise-ceiling figures are pinned to. That
        # happened once during development; the scratch path is the fix.
        command=(
            PYTHON, "scripts/run_benchmark_study.py", "--quick",
            "--out", "reports/metrics/_validation_scratch",
        ),
    ),
    Stage(
        name="report-figures",
        section="final_report.md figures 1-3",
        command=(PYTHON, "scripts/generate_report_figures.py"),
    ),
    Stage(
        name="manuscript-figures",
        section="ress_draft.md figures 1-3",
        command=(PYTHON, "scripts/generate_manuscript_figures.py"),
    ),
    Stage(
        name="survey-gate",
        section="ress_draft.md S2 literature survey",
        # Expected to refuse while the sample is incomplete. Recorded as a
        # reported status rather than a failure - see `_run`.
        command=(PYTHON, "scripts/render_survey_table.py", "--check"),
        quick=False,
    ),
    Stage(
        name="calce-full-discharge-frame",
        section="S4.15.1 / ADR 0012 corrected CALCE target",
        command=(PYTHON, "scripts/build_calce_full_discharge_frame.py"),
        quick=False,
        needs_raw=True,
        raw_hint=(
            "needs the raw CALCE CS2/CX2 archives under data/raw/; the tracked "
            "reports/metrics/calce_full_discharge.csv is its committed output"
        ),
        artifacts=("calce_full_discharge.csv",),
    ),
)

#: Stages whose non-zero exit is a *reported finding*, not a suite failure.
#: `survey-gate` refuses by design while the literature sample is incomplete,
#: and that refusal is the correct behaviour rather than a broken step.
EXPECTED_REFUSAL = {"survey-gate"}


def _digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _tracked_artifacts() -> list[Path]:
    """Every git-tracked file under reports/ that a stage could overwrite."""
    result = subprocess.run(
        ["git", "ls-files", "reports/"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    # Images are excluded: matplotlib stamps creation metadata into a PNG, so
    # regenerating an identical figure produces different bytes. Including them
    # would report drift on every run and drown the signal from the CSVs, where
    # a byte difference really is a value difference.
    return [
        REPO / line
        for line in result.stdout.split()
        if line and not line.lower().endswith((".png", ".jpg", ".svg", ".docx"))
    ]


def _snapshot(paths: list[Path]) -> dict[Path, str | None]:
    return {path: _digest(path) for path in paths}


def _restore(paths: list[Path]) -> None:
    """Return tracked artifacts to their committed content.

    The published record lives in these files and `tests/test_reported_numbers`
    pins quoted figures to them, so a run that rewrites one has invalidated the
    prose until someone decides otherwise. Restoring by default makes this
    runner a *detector* rather than a thing that can quietly rewrite history.
    """
    if not paths:
        return
    subprocess.run(
        ["git", "checkout", "--", *[str(p.relative_to(REPO)) for p in paths]],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


def _raw_available() -> bool:
    raw = REPO / "data" / "raw"
    return raw.exists() and any(raw.rglob("*"))


def _run(stage: Stage, raw_ok: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "stage": stage.name,
        "section": stage.section,
        "command": " ".join(stage.command),
    }

    if stage.needs_raw and not raw_ok:
        record.update(
            status="SKIPPED",
            reason=stage.raw_hint or "requires raw data not present in this checkout",
            duration_s=0.0,
        )
        return record

    before = {name: _digest(METRICS / name) for name in stage.artifacts}
    started = time.monotonic()
    completed = subprocess.run(
        list(stage.command), cwd=REPO, capture_output=True, text=True
    )
    elapsed = time.monotonic() - started
    after = {name: _digest(METRICS / name) for name in stage.artifacts}

    if completed.returncode == 0:
        status = "PASS"
    elif stage.name in EXPECTED_REFUSAL:
        status = "REFUSED"
    else:
        status = "FAIL"

    record.update(
        status=status,
        exit_code=completed.returncode,
        duration_s=round(elapsed, 2),
        artifacts={
            name: {
                "sha256_16": after[name],
                "changed": before[name] != after[name],
                "present": after[name] is not None,
            }
            for name in stage.artifacts
        },
    )
    if status in {"FAIL", "REFUSED"}:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        record["output_tail"] = tail[-12:]
    return record


def _write_summary(records: list[dict[str, object]], quick: bool) -> None:
    lines = [
        "# Validation suite",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"by `scripts/run_validation_suite.py`"
        f"{' --quick' if quick else ''}.",
        "",
        "Every stage below is a command from `docs/final_report.md` Appendix A.",
        "This runner orchestrates them and defines no protocol of its own.",
        "",
        "| Stage | Report section | Status | Seconds |",
        "|---|---|---|---:|",
    ]
    for record in records:
        lines.append(
            f"| `{record['stage']}` | {record['section']} | "
            f"**{record['status']}** | {record.get('duration_s', 0)} |"
        )

    changed = [
        f"`{name}`"
        for record in records
        for name, info in (record.get("artifacts") or {}).items()  # type: ignore[union-attr]
        if isinstance(info, dict) and info.get("changed")
    ]
    lines += ["", "## Artifacts rewritten with different content", ""]
    if changed:
        lines += [
            "A changed artifact is not automatically wrong, but it is also not "
            "automatically fine: `tests/test_reported_numbers.py` pins quoted "
            "figures to these files, so run it before accepting any change.",
            "",
        ] + [f"- {name}" for name in sorted(set(changed))]
    else:
        lines.append("None. Every regenerated artifact was byte-identical.")

    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only the CI-sized subset; skip the multi-minute sweeps.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List stages and exit."
    )
    parser.add_argument(
        "--accept-changes", action="store_true",
        help="Keep regenerated tracked artifacts instead of restoring them. "
             "Only use this when you have decided the new values are correct; "
             "re-run tests/test_reported_numbers.py afterwards.",
    )
    args = parser.parse_args(argv)

    selected = [s for s in STAGES if s.quick or not args.quick]

    if args.list:
        for stage in STAGES:
            mark = "quick" if stage.quick else "full"
            raw = " [needs raw data]" if stage.needs_raw else ""
            print(f"{mark:>5}  {stage.name:<28} {stage.section}{raw}")
        return 0

    raw_ok = _raw_available()
    tracked = _tracked_artifacts()
    before = _snapshot(tracked)
    print(f"validation suite: {len(selected)} stage(s), "
          f"{'quick' if args.quick else 'full'} mode; "
          f"raw data {'present' if raw_ok else 'absent'}; "
          f"watching {len(tracked)} tracked artifact(s)\n")

    records: list[dict[str, object]] = []
    for stage in selected:
        print(f"  {stage.name:<28} ...", end="", flush=True)
        record = _run(stage, raw_ok)
        records.append(record)
        print(f" {record['status']:<8} {record.get('duration_s', 0):>7}s")
        if record["status"] == "FAIL":
            for line in record.get("output_tail", []):  # type: ignore[union-attr]
                print(f"      | {line}")

    after = _snapshot(tracked)
    drifted = sorted(
        path.relative_to(REPO).as_posix()
        for path in tracked
        if before[path] != after[path]
    )
    if drifted and not args.accept_changes:
        _restore([REPO / name for name in drifted])

    METRICS.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "mode": "quick" if args.quick else "full",
                "raw_data_present": raw_ok,
                "drifted_tracked_artifacts": drifted,
                "drift_restored": bool(drifted) and not args.accept_changes,
                "stages": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_summary(records, args.quick)

    if drifted:
        verb = "kept (--accept-changes)" if args.accept_changes else "RESTORED"
        print(
            f"\n{len(drifted)} tracked artifact(s) were rewritten with "
            f"different content and have been {verb}:"
        )
        for name in drifted:
            print(f"  - {name}")
        if not args.accept_changes:
            print(
                "\nThese no longer reproduce from the current code. That is a "
                "reproducibility finding, not a test failure: the committed "
                "values are the published record and have been put back. "
                "Investigate before accepting new ones."
            )

    failed = [r for r in records if r["status"] == "FAIL"]
    tally = {
        status: sum(1 for r in records if r["status"] == status)
        for status in ("PASS", "REFUSED", "SKIPPED", "FAIL")
    }
    print(
        f"\n{tally['PASS']} passed, {tally['REFUSED']} refused (by design), "
        f"{tally['SKIPPED']} skipped, {tally['FAIL']} failed"
    )
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    print(f"wrote {SUMMARY.relative_to(REPO)}")

    if failed:
        print("\nFAILED: " + ", ".join(str(r["stage"]) for r in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
