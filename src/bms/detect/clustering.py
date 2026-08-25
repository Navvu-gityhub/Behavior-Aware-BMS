"""Unsupervised usage-pattern segmentation, and a gate on whether the segments are real.

WHAT THIS BOX WAS SUPPOSED TO BE, AND WHAT IT ACTUALLY IS
----------------------------------------------------------
The architecture diagram calls this "Harmful Behaviour Detection" and lists
K-Means/DBSCAN alongside four things to detect: overcharging, deep discharge,
high-rate fast charging, high-temperature usage.

Those two halves are not the same task, and conflating them is the trap this
module is built to avoid. The four listed conditions are already detected, by
name, by the rule flags in `features.behavior_features.compute_behavior_flags`
— thresholds a human wrote down. Clustering does something different: it finds
groups that are *statistically* separable in feature space. Nothing about a
cluster boundary makes one side harmful.

So this module segments usage patterns and says so. It does not label a
cluster "harmful", because it has no ground truth with which to. What it can
do — and does — is ask two answerable questions:

1. **Are the clusters real at all?** `assess_separability` refuses when the
   best silhouette score across candidate k is below `MIN_SILHOUETTE`.
   K-Means always returns k clusters; it returns them from uniform noise just
   as willingly as from structure, so "we found three usage segments" is not a
   finding until the separability is measured.

2. **Do the clusters correspond to anything the rules already name?**
   `cluster_flag_profile` reports each cluster's mean rate of every existing
   behaviour flag. If a cluster is 90% high-temperature cycles, that is a real
   and interpretable segment. If flag rates are flat across clusters, the
   clustering has found variation the rules do not describe — which is
   interesting, but is emphatically not "detected harmful behaviour".

WHY THE GATE IS A NULL COMPARISON, NOT A SILHOUETTE THRESHOLD
--------------------------------------------------------------
The elbow method requires a human to look at a curve and decide where it
bends, which is exactly the kind of unfalsifiable judgement the rest of this
project has spent its effort removing. Silhouette is a number per candidate k,
so the choice is at least reproducible.

An absolute silhouette floor is not enough, though, and the reason was found
by testing it. **Uniformly random data in three dimensions scores a silhouette
of 0.313** — comfortably above the 0.25 "weak structure" threshold this module
originally used, and above the 0.25 that the literature's rule of thumb would
suggest. K-Means carves compact chunks out of structureless data and
silhouette rewards it for doing so. A fixed floor therefore certifies noise.

The gate instead compares against a **null model of the same shape**: reference
datasets drawn uniformly over each feature's observed range, clustered
identically. Structure is claimed only when the real silhouette exceeds the
null distribution's 95th percentile at the same k. This is the reasoning
behind the gap statistic (Tibshirani et al., 2001), and it is the same move
`adaptive.validation` makes by scoring R-squared against a training-mean
baseline rather than against zero: the question is never "is this number big"
but "is it bigger than what the method produces on nothing".

`MIN_SILHOUETTE` survives only as a degenerate-case floor beneath the null
comparison, not as the decision.

WHY DBSCAN IS OFFERED ALONGSIDE K-MEANS
----------------------------------------
They fail differently, which is the point of running both. K-Means partitions
everything, so every point lands in a cluster whether or not it belongs to
one. DBSCAN has a noise label, so it can decline to assign a point — the
closest thing to a refusal available in a clustering algorithm, and directly
useful here because battery telemetry contains genuinely unusual cycles.

DBSCAN's `eps` is chosen by the k-distance heuristic (the knee of sorted
distances to the k-th nearest neighbour) rather than hand-set, and the chosen
value is reported so it can be challenged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RANDOM_SEED = 20260821

# Candidate cluster counts. Two is the smallest partition that says anything;
# above six, per-cluster populations on these datasets get too small for the
# flag profile to mean much.
DEFAULT_K_RANGE: tuple[int, ...] = (2, 3, 4, 5, 6)

# Degenerate-case floor only. The decision is the null comparison below; this
# just stops a vanishing silhouette passing because the null was even worse.
MIN_SILHOUETTE = 0.15

# Reference datasets drawn per candidate k for the null model. Ten is enough
# for a 95th percentile to be meaningful without making the gate slower than
# the clustering it guards.
NULL_DRAWS = 10

# The real silhouette must beat this percentile of the null distribution.
NULL_PERCENTILE = 95

# ...and beat it by this much. "Greater than" alone is not enough: the null is
# drawn from the same family as the data being tested, so a structureless
# frame edges past its own null roughly as often as not. Measured on the
# fixtures this module is tested against:
#
#     uniform noise          margin ~ +0.006   (i.e. indistinguishable)
#     three planted blobs    margin ~ +0.40
#
# A floor of 0.10 sits an order of magnitude above the noise case and well
# below genuine structure, so it is not a tuned number — anything from about
# 0.05 to 0.30 separates these two populations identically.
MIN_MARGIN = 0.10

# Silhouette is O(n^2) in memory; sample rather than refuse on large frames.
SILHOUETTE_SAMPLE = 5000

# Behaviour flags the rules already name. Cluster profiles are reported
# against these so a segment can be checked for correspondence with something
# a human wrote down.
RULE_FLAGS: tuple[str, ...] = (
    "fast_charge_flag",
    "deep_discharge_flag",
    "high_temp_flag",
    "high_soc_flag",
    "aggressive_discharge_event",
)


@dataclass(frozen=True)
class SeparabilityReport:
    """Whether the data supports clustering at all, and at what k."""

    best_k: int
    best_silhouette: float
    scores: dict[int, float] = field(default_factory=dict)
    null_scores: dict[int, float] = field(default_factory=dict)
    n_samples: int = 0
    n_features: int = 0

    @property
    def null_threshold(self) -> float:
        """What uniform noise of this shape scores at the winning k."""
        return self.null_scores.get(self.best_k, float("nan"))

    @property
    def margin(self) -> float:
        """How far the real data beats structureless data of the same shape."""
        return self.best_silhouette - self.null_threshold

    @property
    def separable(self) -> bool:
        return bool(
            np.isfinite(self.best_silhouette)
            and np.isfinite(self.null_threshold)
            and self.best_silhouette >= MIN_SILHOUETTE
            and self.margin >= MIN_MARGIN
        )

    @property
    def status(self) -> str:
        return "SEPARABLE" if self.separable else "NOT_SEPARABLE"

    def render(self) -> str:
        lines = [
            f"Cluster separability: {self.status}",
            f"  best k = {self.best_k}, silhouette = {self.best_silhouette:.4f}",
            f"  null (uniform, same shape, p{NULL_PERCENTILE}) = "
            f"{self.null_threshold:.4f}   margin = {self.margin:+.4f} "
            f"(needs {MIN_MARGIN:+.2f})",
            f"  {self.n_samples} samples, {self.n_features} features",
        ]
        for k in sorted(self.scores):
            lines.append(
                f"    k={k}: real {self.scores[k]:+.4f}  "
                f"null {self.null_scores.get(k, float('nan')):+.4f}"
            )
        if not self.separable:
            lines.append(
                "  No structure beyond what this algorithm produces on noise. "
                "K-Means returns k clusters whether or not k clusters exist, "
                "and uniform data in low dimensions scores ~0.31 here — any "
                "segmentation reported from this frame would be a partition "
                "of nothing."
            )
        return "\n".join(lines)


def _matrix(data: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Standardised feature matrix. Distance-based methods require it."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    missing = [f for f in features if f not in data.columns]
    if missing:
        raise ValueError(f"clustering: missing feature columns {missing}")

    raw = data[list(features)].to_numpy(dtype=float)
    imputed = SimpleImputer(strategy="median").fit_transform(raw)
    return StandardScaler().fit_transform(imputed)


def assess_separability(
    data: pd.DataFrame,
    features: Sequence[str],
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    seed: int = RANDOM_SEED,
) -> SeparabilityReport:
    """Silhouette across candidate k. Answers "is there structure here?" first.

    Run before reporting any segmentation. A clustering result on data that
    fails this gate describes the algorithm's behaviour, not the fleet's.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    x = _matrix(data, features)
    if len(x) < max(k_range) + 1:
        return SeparabilityReport(
            best_k=0, best_silhouette=float("nan"),
            n_samples=len(x), n_features=x.shape[1],
        )

    rng = np.random.default_rng(seed)
    if len(x) > SILHOUETTE_SAMPLE:
        idx = rng.choice(len(x), size=SILHOUETTE_SAMPLE, replace=False)
        sample = x[idx]
    else:
        sample = x

    def silhouette_at(matrix: np.ndarray, k: int) -> float:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(matrix)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(matrix, labels))

    # Null model: uniform over each feature's observed range, same shape. This
    # is what the algorithm scores on data with no cluster structure at all,
    # and it is the number the real silhouette has to beat.
    lo, hi = sample.min(axis=0), sample.max(axis=0)

    scores: dict[int, float] = {}
    null_scores: dict[int, float] = {}
    for k in k_range:
        if k >= len(sample):
            continue
        real = silhouette_at(sample, int(k))
        if not np.isfinite(real):
            continue
        scores[int(k)] = real

        draws = [
            silhouette_at(rng.uniform(lo, hi, size=sample.shape), int(k))
            for _ in range(NULL_DRAWS)
        ]
        finite = [d for d in draws if np.isfinite(d)]
        null_scores[int(k)] = (
            float(np.percentile(finite, NULL_PERCENTILE)) if finite else float("nan")
        )

    if not scores:
        return SeparabilityReport(
            best_k=0, best_silhouette=float("nan"),
            n_samples=len(x), n_features=x.shape[1],
        )

    # Pick k by the margin over its own null, not by raw silhouette: raw
    # silhouette rises with k on noise too, so choosing the maximum would
    # systematically prefer the k where the null is also highest.
    best_k = max(
        scores,
        key=lambda k: scores[k] - null_scores.get(k, float("-inf")),
    )
    return SeparabilityReport(
        best_k=best_k, best_silhouette=scores[best_k], scores=scores,
        null_scores=null_scores, n_samples=len(x), n_features=x.shape[1],
    )


def kmeans_segments(
    data: pd.DataFrame,
    features: Sequence[str],
    n_clusters: int | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, SeparabilityReport]:
    """Partition into usage segments, with the separability verdict attached.

    `n_clusters=None` picks the best-silhouette k. The report travels back with
    the labels rather than being logged, because "these segments are not
    statistically separable" is a fact the caller must act on, not a
    diagnostic to discard — the same contract `io.load_calce_cycling` uses for
    its load report.
    """
    from sklearn.cluster import KMeans

    report = assess_separability(data, features, seed=seed)
    k = n_clusters if n_clusters is not None else report.best_k
    if k < 2:
        return np.full(len(data), -1), report

    x = _matrix(data, features)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)
    return labels, report


def dbscan_segments(
    data: pd.DataFrame,
    features: Sequence[str],
    eps: float | None = None,
    min_samples: int = 10,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, float]:
    """Density-based segments; returns labels and the `eps` actually used.

    Label -1 is DBSCAN's noise class — points it declines to assign. That is
    the closest thing to a refusal any clustering algorithm offers, and it is
    why DBSCAN is worth running next to K-Means, which assigns everything
    regardless.

    `eps=None` selects it by the k-distance knee: sort every point's distance
    to its `min_samples`-th nearest neighbour and take a high percentile. The
    value is returned so a reader can challenge it, since DBSCAN's output is
    far more sensitive to `eps` than K-Means' is to k.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    x = _matrix(data, features)
    if len(x) <= min_samples:
        return np.full(len(x), -1), float("nan")

    if eps is None:
        nn = NearestNeighbors(n_neighbors=min_samples).fit(x)
        distances, _ = nn.kneighbors(x)
        eps = float(np.percentile(distances[:, -1], 90))
        if not np.isfinite(eps) or eps <= 0:
            eps = 0.5

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    return labels, float(eps)


def cluster_flag_profile(
    data: pd.DataFrame,
    labels: np.ndarray,
    flags: Sequence[str] = RULE_FLAGS,
) -> pd.DataFrame:
    """Each cluster's rate of every rule flag — the interpretability check.

    A cluster is only meaningful to a reader if it corresponds to something
    nameable. Flat flag rates across clusters mean the segmentation has found
    variation the existing rules do not describe; that is a legitimate result
    and is emphatically not "harmful behaviour detected".
    """
    present = [f for f in flags if f in data.columns]
    frame = data.copy()
    frame["_cluster"] = labels

    aggregations = {f: (f, "mean") for f in present}
    aggregations["n_rows"] = ("_cluster", "size")

    profile = frame.groupby("_cluster", as_index=False).agg(**aggregations)
    profile["share"] = profile["n_rows"] / len(frame)
    return profile.rename(columns={"_cluster": "cluster"})
