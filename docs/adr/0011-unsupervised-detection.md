# ADR 0011: Unsupervised detection finds structure, not harm

**Status:** Accepted
**Date:** 2026-08-25

## Context

The architecture diagram's box 3.4, "Harmful Behaviour Detection", listed
K-Means/DBSCAN and Isolation Forest/LOF alongside four conditions to detect:
overcharging, deep discharge, high-rate fast charging, high-temperature usage.
None of the four algorithms existed; the box was ~25% complete, the emptiest
in the diagram, and what stood in for it were the rule-based flags in
`features.behavior_features`.

Building it surfaced a problem with the box itself. Its two halves are not the
same task:

- The four listed conditions are **already detected, by name**, by threshold
  rules a human wrote down.
- Clustering and outlier detection find **statistical** structure and
  statistical outliers. Neither algorithm knows what degradation is, and
  neither is given a fade target.

Nothing about a cluster boundary or an outlier score makes one side harmful.
Shipping the label the box name implies would repeat exactly the error ADR
0003 records for the old Guardian: presenting an output as evidence about
degradation when nothing tied it to degradation.

## Decision

**Build both halves, and gate each on the question the box name assumes has
already been answered.**

`src/bms/detect/` ships K-Means, DBSCAN, Isolation Forest and Local Outlier
Factor, plus two gates:

1. `assess_separability` — are the segments real, or is the algorithm
   partitioning noise?
2. `fade_association` — do flagged cycles actually degrade faster than
   unflagged cycles **in the same cell**?

Neither module emits the word "harmful". Segments are reported as usage
segments; outliers as statistical outliers.

## An absolute silhouette threshold certifies noise

The separability gate began as a silhouette floor at 0.25, the conventional
"weak structure" line. Testing it against uniformly random data refuted it:

> **Uniform noise in three dimensions scores a silhouette of 0.29.**

K-Means carves compact chunks out of structureless data and silhouette rewards
it for doing so. Any fixed threshold in the usual range passes noise.

The gate now compares against a **null model of the same shape** — reference
sets drawn uniformly over each feature's observed range, clustered identically
— and requires the real silhouette to beat the null's 95th percentile by a
margin. This is the reasoning behind the gap statistic, and it is the same
move `adaptive.validation` makes in scoring R² against a training-mean
baseline rather than against zero: the question is never "is this number big"
but "is it bigger than what the method produces on nothing".

`k` is likewise chosen by margin over its own null rather than by raw
silhouette, because raw silhouette rises with `k` on noise too and the maximum
would systematically prefer whichever `k` the null also favours.

Calibration is empirical rather than tuned: uniform noise gives a margin of
+0.006, three planted blobs give +0.40. The floor sits at 0.10, an order of
magnitude above the noise case; anything from 0.05 to 0.30 separates the two
populations identically.

## Results on NASA

**Segments are real.** k=6, silhouette 0.541 against a null of 0.153 — a
margin of +0.388, far clear of the floor. Six usage segments holding between
4.9% and 30.4% of cycles. DBSCAN independently finds structure and assigns
6.2% of cycles to its noise class.

On row-level pipeline telemetry the segments map cleanly onto the existing
rule flags — one segment is 100% deep-discharge, another combines
deep-discharge with high-temperature and aggressive-discharge at ~100% each, a
third is 100% fast-charge. **The clustering rediscovers the rules.** That is
useful corroboration and an argument for keeping the rules, which are cheaper
and interpretable; it is not new detection.

**The outliers are not harm.** Both detectors, at 5% assumed contamination:

| Detector | Flagged | Median fade (flagged) | Median fade (unflagged) | p | Verdict |
|---|---:|---:|---:|---:|---|
| Isolation Forest | 135 / 2682 | −0.005221 | +0.000101 | 1.0000 | NOT_ASSOCIATED |
| Local Outlier Factor | 135 / 2682 | −0.002545 | +0.000000 | 0.8999 | NOT_ASSOCIATED |

Flagged cycles are followed by *less* fade than unflagged ones, within cell.
Whatever these detectors are finding — calibration runs, rest periods, sensor
dropouts, protocol transitions are all statistically unusual and none is
harmful — it is not damage.

**The two detectors also disagree with each other**: 34 of 135 flagged cycles
in common, a quarter. Isolation Forest is global and LOF is local, so this is
expected rather than a defect, but it means "the anomaly detector flagged it"
is not even a well-defined statement without naming which one.

## Consequences

**Box 3.4 is implemented and reports a negative result.** The methods run, the
segments are real, and the outliers carry no demonstrated relationship to
degradation. Nothing from this module feeds the risk score, the Guardian, or
the dashboard, and it should not until an outlier set clears
`fade_association`.

**`contamination` is an assumption and is reported as one.** Both estimators
take the outlier fraction as a parameter; it is asserted, not learned, and it
directly sets how many points come back flagged. Every report states the value
used. A result that moves materially with it is a property of the assumption.

**The rules keep their place.** They detect the four named conditions
directly, they are interpretable, and the clustering corroborates them. The
honest summary of this box is that unsupervised methods added a validated
segmentation and a validated *absence* of an anomaly-harm link — not a new
detector.

## Limitations

- One dataset. The fade association was tested on NASA only; CALCE records no
  temperature or SOC, so the behavioural feature set does not exist there.
- `fade_association` uses `capacity_loss`, whose noise ceiling is 0.044
  (ADR 0007). A weak association could hide beneath that. The test is
  therefore evidence against a *strong* link, not proof of none.
- Rule-flag agreement is only computable on row-level telemetry; the
  cycle-level frames aggregate the booleans away, and the script says so
  rather than reporting an empty comparison.
- Contamination was not swept. A different assumed fraction changes the
  flagged set, though not plausibly the direction of a p = 1.0000 result.

## References

- `src/bms/detect/{clustering,anomaly}.py`, `tests/test_detect.py`
- `scripts/run_detection_study.py`
- `reports/metrics/detection_{summary.md,cluster_profile.csv,agreement.csv}`
- ADR 0003 (attribution must be tied to what it explains), ADR 0007 (target ceiling)
