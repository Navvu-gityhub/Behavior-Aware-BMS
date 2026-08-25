"""Unsupervised anomaly detection, and the test that stops it being mistaken for harm.

AN ANOMALY IS A STATISTICAL OUTLIER, NOT A DAMAGED CELL
--------------------------------------------------------
Isolation Forest and Local Outlier Factor find points that sit apart from the
bulk of the distribution. That is all they find. Nothing in either algorithm
knows what degradation is, and neither was given a fade target.

This distinction is the whole reason this module is written the way it is. The
architecture diagram files these methods under "Harmful Behaviour Detection",
and the natural reading of a returned anomaly is "this cycle damaged the
battery". That inference does not follow, and asserting it would repeat
precisely the error ADR 0003 documents for the old Guardian: presenting an
output as evidence about degradation when nothing tied it to degradation.

An unusual cycle may be a genuinely abusive one. It may equally be a
calibration run, a rest period, a sensor dropout, or the first cycle of a new
protocol. All four are statistically unusual and none is harmful.

THE TWO ANSWERABLE QUESTIONS
-----------------------------
Rather than assert the inference, this module measures whether it holds.

**Does the detector rediscover the rules?** `agreement_with_rules` reports
precision and recall of the anomaly set against each existing behaviour flag.
High agreement means the detector found what a human already wrote down —
useful corroboration, and a reason to prefer the rule, which is cheaper and
interpretable. Low agreement means it found something else, which is only
interesting if the next question has a good answer.

**Do flagged cycles actually degrade faster?** `fade_association` compares the
subsequent fade rate of flagged against unflagged cycles within the same cell,
with a Mann-Whitney U test. Within-cell is essential: cells differ enormously
in baseline fade, so a pooled comparison would mostly measure which cells
happened to contain more outliers.

If flagged cycles do not fade faster, the detector is not finding harm — and
this module reports that plainly rather than shipping an "anomaly" label that
implies otherwise. That is the same gate `scripts/fit_shap_attribution_model.py`
applies to SHAP: an attribution is computed, then barred from supporting
claims it has not earned.

CONTAMINATION IS AN ASSUMPTION, NOT A MEASUREMENT
--------------------------------------------------
Both estimators take a `contamination` parameter — the assumed fraction of
outliers. It is not learned; it is asserted, and it directly sets how many
points come back flagged. There is no way to choose it from the data without
already knowing the answer.

It is therefore an explicit argument with a documented default rather than a
tuned quantity, and every report states the value used. A result that changes
materially with contamination is a property of the assumption, not the fleet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bms.detect.clustering import RULE_FLAGS, _matrix

RANDOM_SEED = 20260821

# Asserted, not learned. See the module docstring.
DEFAULT_CONTAMINATION = 0.05

# Below this many cells with both flagged and unflagged cycles, the paired
# fade comparison has too little to say.
MIN_CELLS_FOR_ASSOCIATION = 5


@dataclass(frozen=True)
class FadeAssociation:
    """Whether flagged cycles are followed by faster fade than unflagged ones."""

    n_cells: int
    n_flagged: int
    n_unflagged: int
    median_fade_flagged: float
    median_fade_unflagged: float
    u_statistic: float
    p_value: float

    @property
    def flagged_fade_faster(self) -> bool:
        return bool(
            np.isfinite(self.median_fade_flagged)
            and np.isfinite(self.median_fade_unflagged)
            and self.median_fade_flagged > self.median_fade_unflagged
        )

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.p_value):
            return "UNDETERMINED"
        if self.significant and self.flagged_fade_faster:
            return "ASSOCIATED"
        return "NOT_ASSOCIATED"

    def render(self) -> str:
        lines = [
            f"Fade association: {self.verdict}",
            f"  flagged   median fade {self.median_fade_flagged:+.6f} "
            f"(n={self.n_flagged})",
            f"  unflagged median fade {self.median_fade_unflagged:+.6f} "
            f"(n={self.n_unflagged})",
            f"  Mann-Whitney U = {self.u_statistic:.0f}, p = {self.p_value:.4f}, "
            f"{self.n_cells} cells",
        ]
        if self.verdict != "ASSOCIATED":
            lines.append(
                "  Flagged cycles are not followed by faster fade. These are "
                "statistical outliers; on this evidence they are not evidence "
                "of damage, and must not be reported as harmful behaviour."
            )
        return "\n".join(lines)


def isolation_forest_scores(
    data: pd.DataFrame,
    features: Sequence[str],
    contamination: float = DEFAULT_CONTAMINATION,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (is_outlier, score). Lower score = more anomalous."""
    from sklearn.ensemble import IsolationForest

    x = _matrix(data, features)
    model = IsolationForest(
        contamination=contamination, random_state=seed, n_estimators=200,
        n_jobs=1,  # see classical.py: a result that depends on core count is not one
    ).fit(x)
    return model.predict(x) == -1, model.score_samples(x)


def local_outlier_factor_scores(
    data: pd.DataFrame,
    features: Sequence[str],
    contamination: float = DEFAULT_CONTAMINATION,
    n_neighbors: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (is_outlier, score). LOF is local; IsolationForest is global.

    Running both is deliberate: LOF flags points unusual relative to their own
    neighbourhood, so it can catch a cycle that is odd for its protocol while
    sitting inside the fleet-wide distribution. Isolation Forest would miss
    exactly that case.
    """
    from sklearn.neighbors import LocalOutlierFactor

    x = _matrix(data, features)
    n_neighbors = min(n_neighbors, max(2, len(x) - 1))
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, n_jobs=1,
    )
    return model.fit_predict(x) == -1, model.negative_outlier_factor_


def agreement_with_rules(
    data: pd.DataFrame,
    is_outlier: np.ndarray,
    flags: Sequence[str] = RULE_FLAGS,
) -> pd.DataFrame:
    """Precision and recall of the anomaly set against each existing rule flag.

    High agreement means the detector rediscovered a rule someone already
    wrote — which argues for keeping the rule, not for adding the detector.
    """
    rows = []
    for flag in flags:
        if flag not in data.columns:
            continue
        truth = data[flag].fillna(0).astype(float).to_numpy() > 0
        hits = int(np.sum(is_outlier & truth))
        precision = hits / max(1, int(is_outlier.sum()))
        recall = hits / max(1, int(truth.sum()))
        rows.append({
            "rule_flag": flag,
            "flag_rate": float(truth.mean()),
            "precision": precision,
            "recall": recall,
            "n_both": hits,
        })
    return pd.DataFrame(rows)


def fade_association(
    data: pd.DataFrame,
    is_outlier: np.ndarray,
    fade_col: str = "capacity_loss",
    cell_col: str = "cell_id",
) -> FadeAssociation:
    """Do flagged cycles show faster fade than unflagged ones in the same cell?

    Compared **within cell**, then pooled. Cells differ enormously in baseline
    fade rate, so a pooled raw comparison would mostly measure which cells
    happened to contain more outliers — the same pseudoreplication trap
    Section 4.2 of the final report guards against with its per-battery check.

    Fade values are centred on their own cell's median before pooling, so the
    test asks "is this cycle's fade high *for this cell*".
    """
    from scipy.stats import mannwhitneyu

    if fade_col not in data.columns or cell_col not in data.columns:
        return FadeAssociation(0, 0, 0, float("nan"), float("nan"),
                               float("nan"), float("nan"))

    frame = data.copy()
    frame["_outlier"] = is_outlier
    frame["_fade"] = pd.to_numeric(frame[fade_col], errors="coerce")
    frame = frame.dropna(subset=["_fade"])

    centred_flagged: list[float] = []
    centred_unflagged: list[float] = []
    n_cells = 0

    for _, group in frame.groupby(cell_col):
        flagged = group.loc[group["_outlier"], "_fade"]
        unflagged = group.loc[~group["_outlier"], "_fade"]
        if flagged.empty or unflagged.empty:
            continue
        centre = float(group["_fade"].median())
        centred_flagged.extend((flagged - centre).tolist())
        centred_unflagged.extend((unflagged - centre).tolist())
        n_cells += 1

    if n_cells < MIN_CELLS_FOR_ASSOCIATION:
        return FadeAssociation(
            n_cells, len(centred_flagged), len(centred_unflagged),
            float("nan"), float("nan"), float("nan"), float("nan"),
        )

    a = np.asarray(centred_flagged, dtype=float)
    b = np.asarray(centred_unflagged, dtype=float)
    stat, p = mannwhitneyu(a, b, alternative="greater")

    return FadeAssociation(
        n_cells=n_cells,
        n_flagged=len(a),
        n_unflagged=len(b),
        median_fade_flagged=float(np.median(a)),
        median_fade_unflagged=float(np.median(b)),
        u_statistic=float(stat),
        p_value=float(p),
    )
