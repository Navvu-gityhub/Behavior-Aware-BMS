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
from src.bms.dashboard.dashboard import build_dashboard


def run_pipeline(raw: pd.DataFrame, output_dir: Path = Path("data/features"), reports_dir: Path = Path("reports/metrics")) -> pd.DataFrame:
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

    return guardian


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Behavior-Aware BMS pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to a unified-schema CSV. Omit to use simulated data.")
    parser.add_argument("--dashboard-out", type=str, default="dashboard.html")
    parser.add_argument("--n-batteries", type=int, default=20)
    parser.add_argument("--rows-per-battery", type=int, default=400)
    args = parser.parse_args()

    if args.data:
        raw = pd.read_csv(args.data)
        print(f"Loaded {len(raw)} rows from {args.data}")
    else:
        raw = simulate_fleet(SimulationConfig(n_batteries=args.n_batteries, rows_per_battery=args.rows_per_battery))
        print(f"Simulated {raw['cell_id'].nunique()} batteries, {len(raw)} rows")

    guardian = run_pipeline(raw)

    dashboard_path = build_dashboard(guardian, args.dashboard_out)
    print(f"Pipeline complete. {len(guardian)} batteries scored.")
    print(f"Dashboard written to {dashboard_path.resolve()}")


if __name__ == "__main__":
    main()
