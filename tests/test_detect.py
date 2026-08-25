"""Tests for unsupervised segmentation and outlier detection.

The load-bearing tests are the two gates, because without them this module
would emit labels that read as findings:

`test_uniform_noise_is_not_separable` — K-Means returns k clusters from
structureless data as readily as from structured data, so "we found three
usage segments" means nothing until separability is measured.

`test_random_flags_are_not_fade_associated` — an anomaly is a statistical
outlier. Calling it harmful requires evidence, and this is the test that
demands it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bms.detect import (
    MIN_MARGIN,
    agreement_with_rules,
    assess_separability,
    cluster_flag_profile,
    dbscan_segments,
    fade_association,
    isolation_forest_scores,
    kmeans_segments,
    local_outlier_factor_scores,
)

FEATURES = ("f0", "f1", "f2")


def clustered_frame(n_per: int = 120, seed: int = 0) -> pd.DataFrame:
    """Three well-separated blobs with a rule flag tied to one of them."""
    rng = np.random.default_rng(seed)
    parts = []
    for i, centre in enumerate([(0, 0, 0), (10, 10, 10), (-10, 8, -6)]):
        block = rng.normal(centre, 0.4, size=(n_per, 3))
        frame = pd.DataFrame(block, columns=list(FEATURES))
        frame["cell_id"] = [f"C{j % 6}" for j in range(n_per)]
        # The third blob is exactly the flagged population.
        frame["high_temp_flag"] = 1 if i == 2 else 0
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def noise_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.uniform(0, 1, size=(n, 3)), columns=list(FEATURES))
    frame["cell_id"] = [f"C{i % 8}" for i in range(n)]
    frame["high_temp_flag"] = 0
    return frame


class TestSeparabilityGate:
    def test_well_separated_blobs_are_separable(self):
        report = assess_separability(clustered_frame(), FEATURES)
        assert report.separable
        assert report.best_k == 3
        assert report.best_silhouette > 0.7

    def test_uniform_noise_is_not_separable(self):
        """K-Means will happily partition structureless data; the gate must not.

        Note what this data actually scores: silhouette ~0.29, comfortably
        above the 0.25 "weak structure" rule of thumb an absolute threshold
        would use. It is rejected because it fails to beat a null model of the
        same shape by the required margin, which is the only comparison that
        distinguishes structure from the algorithm's own behaviour on noise.
        """
        report = assess_separability(noise_frame(), FEATURES)
        assert not report.separable
        assert report.best_silhouette > 0.25, "the point: raw silhouette looks fine"
        assert report.margin < MIN_MARGIN
        assert "beyond what this algorithm produces on noise" in report.render()

    def test_planted_structure_clears_the_null_by_a_wide_margin(self):
        report = assess_separability(clustered_frame(), FEATURES)
        assert report.separable
        assert report.margin > 0.3, "real structure should not be marginal"

    def test_report_carries_every_candidate_k(self):
        report = assess_separability(clustered_frame(), FEATURES, k_range=(2, 3, 4))
        assert set(report.scores) == {2, 3, 4}

    def test_too_few_samples_returns_nan_not_a_guess(self):
        report = assess_separability(clustered_frame(n_per=1), FEATURES)
        assert not report.separable
        assert np.isnan(report.best_silhouette)

    def test_missing_feature_raises(self):
        with pytest.raises(ValueError, match="missing feature columns"):
            assess_separability(clustered_frame(), ("nope",))


class TestKMeans:
    def test_recovers_the_planted_groups(self):
        data = clustered_frame()
        labels, report = kmeans_segments(data, FEATURES)
        assert report.separable
        assert len(np.unique(labels)) == 3

    def test_report_travels_back_with_the_labels(self):
        """A caller must not be able to use segments without the verdict."""
        labels, report = kmeans_segments(noise_frame(), FEATURES)
        assert not report.separable
        assert len(labels) == len(noise_frame())

    def test_flag_profile_identifies_the_flagged_cluster(self):
        data = clustered_frame()
        labels, _ = kmeans_segments(data, FEATURES)
        profile = cluster_flag_profile(data, labels)

        # Exactly one cluster should be entirely flagged, and the rest not.
        rates = sorted(profile["high_temp_flag"].round(3).tolist())
        assert rates[-1] == pytest.approx(1.0)
        assert rates[0] == pytest.approx(0.0)
        assert profile["share"].sum() == pytest.approx(1.0)


class TestDbscan:
    def test_returns_the_eps_it_chose(self):
        _, eps = dbscan_segments(clustered_frame(), FEATURES, min_samples=10)
        assert np.isfinite(eps) and eps > 0

    def test_can_decline_to_assign_points(self):
        """DBSCAN's noise label is the only refusal a clusterer offers."""
        data = pd.concat([clustered_frame(), noise_frame(n=60, seed=5)],
                         ignore_index=True)
        labels, _ = dbscan_segments(data, FEATURES, min_samples=10)
        assert (labels == -1).any()

    def test_tiny_frame_returns_all_noise(self):
        labels, eps = dbscan_segments(clustered_frame(n_per=2), FEATURES,
                                      min_samples=50)
        assert (labels == -1).all()
        assert np.isnan(eps)


class TestAnomalyDetectors:
    def test_isolation_forest_flags_the_configured_fraction(self):
        data = clustered_frame()
        out, scores = isolation_forest_scores(data, FEATURES, contamination=0.1)
        assert 0.05 <= out.mean() <= 0.15
        assert len(scores) == len(data)

    def test_lof_flags_the_configured_fraction(self):
        data = clustered_frame()
        out, _ = local_outlier_factor_scores(data, FEATURES, contamination=0.1)
        assert 0.05 <= out.mean() <= 0.15

    def test_isolation_forest_is_deterministic(self):
        data = clustered_frame()
        a, _ = isolation_forest_scores(data, FEATURES)
        b, _ = isolation_forest_scores(data, FEATURES)
        assert np.array_equal(a, b)

    def test_agreement_reports_precision_and_recall_per_rule(self):
        data = clustered_frame()
        out, _ = isolation_forest_scores(data, FEATURES)
        table = agreement_with_rules(data, out, flags=("high_temp_flag",))
        assert list(table["rule_flag"]) == ["high_temp_flag"]
        assert 0.0 <= table["precision"].iloc[0] <= 1.0
        assert 0.0 <= table["recall"].iloc[0] <= 1.0

    def test_agreement_skips_absent_flags(self):
        data = clustered_frame()
        out, _ = isolation_forest_scores(data, FEATURES)
        assert agreement_with_rules(data, out, flags=("not_a_column",)).empty


class TestFadeAssociation:
    def _frame_with_fade(self, harmful: bool, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for cell in range(8):
            for i in range(60):
                flagged = i % 10 == 0
                base = rng.normal(0.001, 0.0002)
                # When `harmful`, flagged cycles really do fade faster.
                fade = base + (0.004 if (harmful and flagged) else 0.0)
                rows.append({"cell_id": f"C{cell}", "capacity_loss": fade,
                             "_flag": flagged})
        return pd.DataFrame(rows)

    def test_detects_a_real_association(self):
        data = self._frame_with_fade(harmful=True)
        assoc = fade_association(data, data["_flag"].to_numpy())
        assert assoc.verdict == "ASSOCIATED"
        assert assoc.flagged_fade_faster
        assert assoc.significant

    def test_random_flags_are_not_fade_associated(self):
        """The gate that stops an outlier being reported as damage."""
        data = self._frame_with_fade(harmful=False)
        assoc = fade_association(data, data["_flag"].to_numpy())
        assert assoc.verdict == "NOT_ASSOCIATED"
        assert "not evidence of damage" in assoc.render()

    def test_comparison_is_within_cell(self):
        """Pooling across cells would measure which cells hold outliers.

        Here every flag sits in one cell whose baseline fade is high. Pooled,
        the flagged group looks far worse; centred within cell, it does not.
        """
        rows = []
        for cell in range(8):
            fast = cell == 0
            for _ in range(60):
                rows.append({
                    "cell_id": f"C{cell}",
                    "capacity_loss": 0.02 if fast else 0.001,
                    "_flag": fast,
                })
        data = pd.DataFrame(rows)
        assoc = fade_association(data, data["_flag"].to_numpy())
        # Every flagged row is in one cell, so no cell has both groups.
        assert assoc.verdict == "UNDETERMINED"

    def test_too_few_cells_is_undetermined_not_a_verdict(self):
        rows = [{"cell_id": "C0", "capacity_loss": 0.001, "_flag": i % 2 == 0}
                for i in range(40)]
        data = pd.DataFrame(rows)
        assert fade_association(data, data["_flag"].to_numpy()).verdict == "UNDETERMINED"

    def test_missing_columns_are_undetermined(self):
        data = pd.DataFrame({"cell_id": ["a"], "x": [1]})
        assoc = fade_association(data, np.array([True]))
        assert assoc.verdict == "UNDETERMINED"
