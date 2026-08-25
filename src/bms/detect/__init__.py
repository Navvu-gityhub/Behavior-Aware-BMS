"""Unsupervised usage-pattern segmentation and outlier detection.

This is the architecture diagram's "Harmful Behaviour Detection" box, built
with the distinction the box name elides: these methods find statistical
structure and statistical outliers. Neither knows what degradation is.

The four conditions the diagram lists — overcharging, deep discharge,
high-rate fast charging, high-temperature usage — are detected by name, by the
rule flags in `features.behavior_features`. What is here complements those; it
does not replace them, and it does not license calling an outlier harmful.

Both halves therefore ship with a gate:

- `assess_separability` refuses to report segments when the best silhouette
  across candidate k falls below a floor. K-Means returns k clusters from
  uniform noise as readily as from structure.
- `fade_association` tests, within cell, whether flagged cycles are actually
  followed by faster fade. When they are not, the module says so rather than
  emitting an "anomaly" label that implies damage it has not demonstrated.
"""

from src.bms.detect.anomaly import (
    DEFAULT_CONTAMINATION,
    FadeAssociation,
    agreement_with_rules,
    fade_association,
    isolation_forest_scores,
    local_outlier_factor_scores,
)
from src.bms.detect.clustering import (
    MIN_MARGIN,
    MIN_SILHOUETTE,
    RULE_FLAGS,
    SeparabilityReport,
    assess_separability,
    cluster_flag_profile,
    dbscan_segments,
    kmeans_segments,
)

__all__ = [
    "DEFAULT_CONTAMINATION",
    "MIN_MARGIN",
    "MIN_SILHOUETTE",
    "RULE_FLAGS",
    "FadeAssociation",
    "SeparabilityReport",
    "agreement_with_rules",
    "assess_separability",
    "cluster_flag_profile",
    "dbscan_segments",
    "fade_association",
    "isolation_forest_scores",
    "kmeans_segments",
    "local_outlier_factor_scores",
]
