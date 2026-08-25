"""Run unsupervised segmentation and outlier detection, and test what they found.

    python scripts/run_detection_study.py

Writes to reports/metrics/:

    detection_cluster_profile.csv   each segment's rate of every rule flag
    detection_agreement.csv         anomaly-set precision/recall vs each rule
    detection_summary.md            rendered verdicts

WHAT THIS ANSWERS
-----------------
The architecture diagram files K-Means/DBSCAN and Isolation Forest/LOF under
"Harmful Behaviour Detection". Those methods find statistical structure and
statistical outliers; neither knows what degradation is. This script runs them
and then asks the two questions that decide whether the label is earned:

1. Are the segments real, or is the algorithm partitioning noise?
   (`assess_separability` — compared against a null model of the same shape,
   because uniform data scores a silhouette of ~0.29 and would pass any
   conventional absolute threshold.)

2. Do flagged cycles actually degrade faster than unflagged ones in the same
   cell? (`fade_association`.) If not, the outliers are not evidence of harm,
   and the report says so.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bms.benchmarks import add_targets  # noqa: E402
from src.bms.detect import (  # noqa: E402
    agreement_with_rules,
    assess_separability,
    cluster_flag_profile,
    dbscan_segments,
    fade_association,
    isolation_forest_scores,
    kmeans_segments,
    local_outlier_factor_scores,
)

REPO_ROOT = Path(__file__).parent.parent
METRICS = REPO_ROOT / "reports" / "metrics"
NASA = METRICS / "continuous_model_training_data.csv"

# Cycle-level behavioural features, matching what the benchmark suite uses.
CYCLE_FEATURES = ("avg_temp", "max_temp", "avg_stress",
                  "deep_discharge_duration", "aggressive_discharge_count", "avg_soc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=NASA)
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Assumed outlier fraction. Asserted, not learned — "
                             "it directly sets how many points come back flagged.")
    parser.add_argument("--out", type=Path, default=METRICS)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"No frame at {args.data}")

    args.out.mkdir(parents=True, exist_ok=True)
    data = add_targets(pd.read_csv(args.data))
    print(f"Loaded {len(data)} rows, {data['cell_id'].nunique()} cells "
          f"from {args.data.name}")

    lines: list[str] = ["# Unsupervised detection study", "",
                        f"Source: `{args.data.name}` — {len(data)} rows, "
                        f"{data['cell_id'].nunique()} cells.",
                        f"Contamination assumed: {args.contamination}.", ""]

    print("\n=== 1. Are the segments real? ===")
    report = assess_separability(data, CYCLE_FEATURES)
    print(report.render())
    lines += ["## Separability", "", "```", report.render(), "```", ""]

    if report.separable:
        labels, _ = kmeans_segments(data, CYCLE_FEATURES)
        profile = cluster_flag_profile(data, labels)
        profile.to_csv(args.out / "detection_cluster_profile.csv", index=False)
        print("\n=== 2. Do the segments match anything the rules name? ===")
        print(profile.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        lines += ["## Segment profile", "", "```",
                  profile.to_string(index=False), "```", ""]

        dlabels, eps = dbscan_segments(data, CYCLE_FEATURES)
        noise = float((dlabels == -1).mean())
        print(f"\nDBSCAN: eps={eps:.3f}, "
              f"{len(set(dlabels.tolist())) - (1 if -1 in dlabels else 0)} clusters, "
              f"noise share {noise:.3f}")
        lines.append(f"DBSCAN: eps={eps:.3f}, noise share {noise:.3f}\n")
    else:
        print("  segmentation not reported — see the verdict above")
        lines.append("Segmentation not reported: the gate refused.\n")

    print("\n=== 3. Do the outliers correspond to harm? ===")
    agreements = []
    for name, fn in (("isolation_forest", isolation_forest_scores),
                     ("local_outlier_factor", local_outlier_factor_scores)):
        out, _ = fn(data, CYCLE_FEATURES, contamination=args.contamination)
        agree = agreement_with_rules(data, out)
        agree.insert(0, "detector", name)
        agreements.append(agree)

        assoc = fade_association(data, out, fade_col="capacity_loss")
        print(f"\n--- {name}: flagged {int(out.sum())}/{len(out)} ---")

        # The boolean rule flags live on ROW-level telemetry; a cycle-level
        # frame carries their aggregates (`deep_discharge_duration`,
        # `aggressive_discharge_count`) instead. Say that rather than printing
        # an empty table, which reads as "no agreement" instead of
        # "not comparable on this frame".
        if agree.empty:
            note = ("rule-flag agreement not computable on a cycle-level frame: "
                    "the boolean flags exist on row-level telemetry, which this "
                    "frame aggregates away. Run against pipeline telemetry to "
                    "compare.")
            print(f"  {note}")
        else:
            note = ""
            print(agree.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(assoc.render())

        lines += [f"## {name}", ""]
        lines += ["```", note or agree.to_string(index=False), "",
                  assoc.render(), "```", ""]

    pd.concat(agreements, ignore_index=True).to_csv(
        args.out / "detection_agreement.csv", index=False)

    iso, _ = isolation_forest_scores(data, CYCLE_FEATURES,
                                     contamination=args.contamination)
    lof, _ = local_outlier_factor_scores(data, CYCLE_FEATURES,
                                         contamination=args.contamination)
    overlap = int(np.sum(iso & lof))
    print(f"\nDetector agreement with each other: {overlap} of "
          f"{int(iso.sum())} flagged in common")
    lines.append(f"Detector overlap: {overlap} of {int(iso.sum())} in common.\n")

    (args.out / "detection_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {args.out / 'detection_summary.md'}")


if __name__ == "__main__":
    main()
