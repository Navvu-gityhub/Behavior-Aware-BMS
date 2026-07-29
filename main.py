"""Behavior-Aware BMS — full pipeline entry point.

    python main.py                          # simulated fleet
    python main.py --data path/to/file.csv  # your own unified-schema CSV

Runs: simulate/load -> schema validate -> behavior flags -> stress score ->
rolling/age features -> per-battery summary -> risk assessment -> health
index -> RUL -> guardian reports -> dashboard.html.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Matches the import convention already used in tests/test_schema.py
# ("from src.bms.preprocessing.schema import ..."). Run as `python main.py`
# from the repository root.
sys.path.insert(0, str(Path(__file__).parent))

from src.bms.simulation.simulate_telemetry import SimulationConfig, simulate_fleet
from src.bms.preprocessing.schema import standardize_validate_bms_data
from src.bms.features.behavior_features import (
    compute_behavior_flags,
    add_rolling_features,
    add_age_features,
    summarize_batteries,
)
from src.bms.risk.stress_score import compute_stress_score, compute_risk_assessment
from src.bms.health.health_index import compute_health_index
from src.bms.rul.rul_estimation import compute_rul
from src.bms.guardian.guardian import generate_guardian_reports
from src.bms.dashboard.beacon import build_beacon_dashboard


@dataclass(frozen=True)
class PipelineResult:
    """Everything one pipeline run produces that a caller might need.

    `run_pipeline` previously returned the guardian table alone. The dashboard
    also needs the row-level feature table to draw real per-cycle series, and
    having the renderer re-read `behavior_features_v1.csv` from disk would
    couple it to a path and let it silently use stale data if a write failed.

    Returning a named result rather than a bare tuple is deliberate: a tuple
    would make `guardian, _ = run_pipeline(...)` the norm and would break
    silently the next time the pipeline gains an output. Attribute access
    keeps existing call sites readable and future additions non-breaking.
    """

    guardian: pd.DataFrame
    telemetry: pd.DataFrame
    summary: pd.DataFrame


def run_pipeline(
    raw: pd.DataFrame,
    output_dir: Path = Path("data/features"),
    reports_dir: Path = Path("reports/metrics"),
) -> PipelineResult:
    """Run the full pipeline and return its tables. See `PipelineResult`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    clean, issues = standardize_validate_bms_data(raw, dataset=raw.get("dataset", pd.Series(["unknown"])).iloc[0])
    if issues:
        print(f"[schema] {len(issues)} validation issue(s): {issues}", file=sys.stderr)

    clean = clean.sort_values(["cell_id", "cycle"]).reset_index(drop=True)

    flagged = compute_behavior_flags(clean)
    flagged["stress_score"] = compute_stress_score(flagged)
    featured = add_rolling_features(flagged)
    featured = add_age_features(featured)
    featured.to_csv(output_dir / "behavior_features_v1.csv", index=False)

    summary = summarize_batteries(featured)
    summary.to_csv(output_dir / "battery_summary_v1.csv", index=False)

    risk = compute_risk_assessment(summary)
    risk.to_csv(output_dir / "battery_risk_assessment_v1.csv", index=False)

    health = compute_health_index(summary)
    merged = risk.merge(health.drop(columns=["avg_stress", "avg_temp", "deep_discharge_duration", "fast_charge_duration", "aggressive_discharge_count", "avg_soc"]), on="battery_id")
    merged.to_csv(output_dir / "battery_health_index_v1.csv", index=False)

    rul = compute_rul(merged)
    rul.to_csv(output_dir / "battery_rul_estimation_v1.csv", index=False)

    guardian = generate_guardian_reports(rul)
    guardian.to_csv(output_dir / "battery_guardian_output_v1.csv", index=False)

    guardian["battery_state"].value_counts().to_csv(reports_dir / "battery_state_distribution.csv")
    guardian["risk_level"].value_counts().to_csv(reports_dir / "risk_distribution.csv")

    return PipelineResult(guardian=guardian, telemetry=featured, summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Behavior-Aware BMS pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to a unified-schema CSV. Omit to use simulated data.")
    parser.add_argument("--dashboard-out", type=str, default="dashboard.html")
    parser.add_argument("--n-batteries", type=int, default=20)
    parser.add_argument("--rows-per-battery", type=int, default=400)
    parser.add_argument("--source", type=str, default=None,
                        help="Provenance label: 'measured' or 'simulated'. Controls the "
                             "dashboard's data-provenance banner.")
    parser.add_argument("--dataset-label", type=str, default=None,
                        help="Human-readable dataset name shown on the dashboard.")
    args = parser.parse_args()

    if args.data:
        raw = pd.read_csv(args.data)
        # Provenance is tracked from here so the dashboard can state plainly
        # whether it is showing measurements or simulator output. Mislabelling
        # a simulated run as real is the most consequential error this tool
        # could make in a presentation, so the label is derived from the
        # invocation rather than left to the caller to remember.
        data_source = args.source or "measured"
        dataset_label = args.dataset_label or Path(args.data).name
        print(f"Loaded {len(raw)} rows from {args.data}")
    else:
        raw = simulate_fleet(SimulationConfig(n_batteries=args.n_batteries, rows_per_battery=args.rows_per_battery))
        data_source = "simulated"
        dataset_label = "Synthetic fleet telemetry (simulate_fleet)"
        print(f"Simulated {raw['cell_id'].nunique()} batteries, {len(raw)} rows")

    result = run_pipeline(raw)
    guardian = result.guardian

    dashboard_path = build_beacon_dashboard(
        guardian,
        args.dashboard_out,
        telemetry=result.telemetry,
        data_source=data_source,
        dataset_label=dataset_label,
    )
    print(f"Pipeline complete. {len(guardian)} batteries scored.")
    print(f"Dashboard written to {dashboard_path.resolve()}")


if __name__ == "__main__":
    main()
