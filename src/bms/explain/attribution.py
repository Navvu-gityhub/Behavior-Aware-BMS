"""Exact Shapley attribution for the pipeline's additive rule-based scores.

WHY THIS EXISTS
---------------
`guardian.guardian` previously explained a battery's state by re-deriving
causes from its own independent if/else thresholds, which had two problems:

1. The thresholds in `_primary_causes` (avg_temp > 35, fast_charge > 50,
   deep_discharge > 50) were *different numbers* from the thresholds the
   risk and health scores actually use (>= 30 / >= 40, >= 20 / >= 100,
   >= 20 / >= 100). Guardian could therefore name a cause that contributed
   nothing to the score, or stay silent about the term that dominated it.
2. It reported causes as an unordered set. A battery penalised 5 points for
   temperature and 30 for aggressive discharge got both listed with equal
   weight and no magnitude.

This module replaces that with an attribution that is, by construction,
consistent with the score being explained: it decomposes the *actual*
score into per-term contributions that sum exactly to it.

WHY SHAPLEY VALUES ARE EXACT HERE (AND WHY WE DON'T CALL `shap`)
----------------------------------------------------------------
The risk score and aging budget are additive by construction:

    f(x) = sum_i f_i(x_i)

For an additive model, the Shapley value has a closed form. Writing the
coalition value as the expected score with the features in S held at their
observed values,

    v(S) = E[ f(X) | X_S = x_S ] = sum_{i in S} f_i(x_i) + sum_{i not in S} E[ f_i(X_i) ]

the marginal contribution of feature i is

    v(S u {i}) - v(S) = f_i(x_i) - E[ f_i(X_i) ]

which does not depend on S at all. Every term in the Shapley weighted
average is therefore identical, and

    phi_i = f_i(x_i) - E[ f_i(X_i) ]

This is worth stating plainly because it has three practical consequences:

* **No approximation.** KernelSHAP samples coalitions and TreeSHAP relies on
  a tree structure; both return estimates. Here the closed form is exact,
  so there is no sampling error to report and no `nsamples` to tune.
* **No feature-independence assumption.** KernelSHAP's interventional
  estimate is only faithful when features are approximately independent —
  a bad assumption for battery telemetry, where temperature, C-rate and SOC
  exposure are strongly correlated. The derivation above never factorises a
  joint distribution: `v(S)` decomposes exactly because `f` is additive, not
  because the features are independent. So the correlation between our
  features, which is real and large, does not bias these attributions.
* **It is cheap and dependency-free.** Attribution costs one vectorised pass,
  so it runs inside the pipeline for every battery rather than being an
  offline analysis step.

The `shap` library IS used elsewhere in this project — see
`src/bms/explain/shap_model.py` — but for the genuinely non-additive
gradient-boosted model fitted against *measured* capacity fade, which is
where a sampling-based explainer is actually the right tool. Using
KernelSHAP here, on a model whose exact Shapley values are available in
closed form, would add a dependency, add sampling noise, and add an
unnecessary independence assumption, in exchange for nothing.

THE EFFICIENCY AXIOM IS OUR TEST ORACLE
---------------------------------------
Shapley values satisfy `sum_i phi_i = f(x) - E[f(X)]`. Because that must
hold to floating-point tolerance, it doubles as a self-check that the term
definitions used by the explainer are the same ones used by the scorer.
`verify_efficiency()` below asserts it, and `tests/test_explain.py` runs it
against real pipeline output — so if anyone edits a threshold in
`stress_score.py` without updating its term spec, the test fails rather than
the dashboard quietly showing an explanation that doesn't add up.

BASELINE CHOICE IS A REPORTING DECISION, NOT A DETAIL
-----------------------------------------------------
`phi_i` is defined relative to `E[f_i(X_i)]`, so every attribution is
"relative to a reference population". Two references are supported and the
choice changes what the number means:

* `fleet` (default): the mean over the batteries being scored. Reads as
  "why is this battery worse than the rest of this fleet?" — the right frame
  for a dashboard ranking cells against each other.
* `ideal`: a fixed hypothetical best-case battery (see `IDEAL_REFERENCE`).
  Reads as "why is this battery worse than perfect?" — the right frame for a
  single battery in isolation, where a fleet mean is undefined or would be
  computed over one row.

`fleet` is the default because the pipeline scores fleets, but the reference
is recorded in the output so an explanation is never presented without the
comparison it was computed against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

# Tolerance for the efficiency check. The score functions are sums of
# `np.where` selections over float64, so error is accumulation-level, not
# algorithmic; 1e-9 is loose enough to never flake and tight enough that a
# genuine term mismatch (which would be off by whole points) always trips it.
EFFICIENCY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ScoreTerm:
    """One additive term of a rule-based score.

    `fn` maps the raw feature column to that term's point contribution. It
    must be vectorised (accept and return a Series) and must depend on
    `feature` alone — a term reading two features would break the additive
    decomposition this module's exactness argument rests on.

    `label` is the human-readable cause name surfaced by Guardian and the
    dashboard. `higher_is_worse` records the term's intended direction, and
    is used only to sanity-check that a "protective" attribution is really
    protective before it is worded as one.
    """

    name: str
    feature: str
    label: str
    fn: Callable[[pd.Series], pd.Series]
    higher_is_worse: bool = True

    def contribution(self, values: pd.Series) -> pd.Series:
        return pd.Series(np.asarray(self.fn(values), dtype=float), index=values.index)


# A hypothetical best-case battery: moderate temperature, no abusive events,
# mid-band SOC. These are not measured values from any dataset — they are the
# input vector at which every penalty term in the pipeline takes its minimum,
# chosen so `reference="ideal"` means "compared to a battery that triggers no
# penalty at all". Documented as an assumption, not a calibration.
IDEAL_REFERENCE: Mapping[str, float] = {
    "avg_stress": 0.0,
    "avg_temp": 25.0,
    "deep_discharge_duration": 0.0,
    "fast_charge_duration": 0.0,
    "aggressive_discharge_count": 0.0,
    "avg_soc": 50.0,
}


def _reference_contributions(
    terms: Sequence[ScoreTerm],
    data: pd.DataFrame,
    reference: str,
) -> dict[str, float]:
    """Compute `E[f_i(X_i)]` for each term under the chosen reference."""
    if reference == "fleet":
        return {t.name: float(t.contribution(data[t.feature]).mean()) for t in terms}

    if reference == "ideal":
        missing = [t.feature for t in terms if t.feature not in IDEAL_REFERENCE]
        if missing:
            raise ValueError(
                f"reference='ideal' has no defined value for features {missing}. "
                "Add them to IDEAL_REFERENCE with a documented justification, or "
                "use reference='fleet'."
            )
        out: dict[str, float] = {}
        for t in terms:
            single = pd.Series([IDEAL_REFERENCE[t.feature]], index=[0])
            out[t.name] = float(t.contribution(single).iloc[0])
        return out

    raise ValueError(f"Unknown reference {reference!r}; expected 'fleet' or 'ideal'.")


def additive_shapley_values(
    data: pd.DataFrame,
    terms: Sequence[ScoreTerm],
    reference: str = "fleet",
) -> pd.DataFrame:
    """Exact per-term Shapley values for an additive score.

    Returns one column per term, indexed like `data`, in score points. A
    positive value means the term pushed this battery's score above the
    reference; negative means below.

    See the module docstring for why this closed form is exact rather than
    an approximation.
    """
    if not terms:
        raise ValueError("additive_shapley_values: no terms supplied")

    missing = sorted({t.feature for t in terms} - set(data.columns))
    if missing:
        raise ValueError(f"additive_shapley_values: missing feature columns {missing}")

    if reference == "fleet" and len(data) < 2:
        raise ValueError(
            "reference='fleet' needs at least 2 batteries to form a meaningful "
            "mean — with one row every attribution would be exactly zero. Use "
            "reference='ideal' for single-battery explanation."
        )

    ref = _reference_contributions(terms, data, reference)
    return pd.DataFrame(
        {t.name: t.contribution(data[t.feature]) - ref[t.name] for t in terms},
        index=data.index,
    )


def verify_efficiency(
    data: pd.DataFrame,
    terms: Sequence[ScoreTerm],
    scores: pd.Series,
    reference: str = "fleet",
    tolerance: float = EFFICIENCY_TOLERANCE,
) -> None:
    """Assert the Shapley efficiency axiom holds against the real score.

    Raises `AssertionError` if `sum_i phi_i != f(x) - E[f(X)]`, which in
    practice means the term specs have drifted from the scorer they claim to
    decompose. This is the guard that keeps explanations honest as thresholds
    change; see the module docstring.

    Note this checks the decomposition against the score *before* clipping.
    `compute_risk_assessment` clips to [0, 100]; for the pipeline's own term
    weights the unclipped sum cannot leave that range, but a caller supplying
    custom weights that can saturate should expect this check to fail — the
    clip is genuinely non-additive, and silently tolerating it would mean
    reporting attributions that don't sum to the score shown.
    """
    phi = additive_shapley_values(data, terms, reference=reference)
    ref = _reference_contributions(terms, data, reference)
    expected = np.asarray(scores, dtype=float) - sum(ref.values())
    actual = phi.sum(axis=1).to_numpy(dtype=float)

    worst = float(np.max(np.abs(actual - expected))) if len(actual) else 0.0
    if worst > tolerance:
        raise AssertionError(
            f"Shapley efficiency violated: max |sum(phi) - (score - E[score])| = "
            f"{worst:.3e} > {tolerance:.1e}. The term specification no longer "
            f"matches the score function it decomposes — a threshold or weight "
            f"was almost certainly changed in one place but not the other."
        )


def rank_attributions(
    phi: pd.DataFrame,
    terms: Sequence[ScoreTerm],
    top_n: int = 3,
    min_magnitude: float = 0.5,
) -> pd.DataFrame:
    """Reduce a Shapley matrix to a ranked, human-readable cause list.

    `min_magnitude` suppresses terms whose contribution rounds to noise; a
    0.4-point contribution is not a "primary cause" and listing it dilutes
    the ones that are. Batteries with no term above the floor are reported
    as driven by normal usage rather than given a spurious cause.
    """
    labels = {t.name: t.label for t in terms}
    rows = []

    for idx, row in phi.iterrows():
        drivers = row[row > min_magnitude].sort_values(ascending=False)
        protectors = row[row < -min_magnitude].sort_values()

        top = drivers.head(top_n)
        rows.append(
            {
                "top_causes": ", ".join(labels[k] for k in top.index) if len(top) else "normal usage",
                "top_cause_contributions": ", ".join(
                    f"{labels[k]} +{v:.1f}" for k, v in top.items()
                )
                if len(top)
                else "no term above attribution floor",
                "dominant_cause": labels[top.index[0]] if len(top) else "normal usage",
                "dominant_contribution": float(top.iloc[0]) if len(top) else 0.0,
                "strongest_protective_factor": labels[protectors.index[0]]
                if len(protectors)
                else "none",
                "n_significant_drivers": int(len(drivers)),
            }
        )

    return pd.DataFrame(rows, index=phi.index)


def explain_scores(
    data: pd.DataFrame,
    terms: Sequence[ScoreTerm],
    scores: pd.Series,
    reference: str = "fleet",
    top_n: int = 3,
    prefix: str = "",
    verify: bool = True,
) -> pd.DataFrame:
    """End-to-end attribution: Shapley values + ranked causes, with self-check.

    Returns the per-term contributions (prefixed `{prefix}shap_{term}`)
    alongside the ranked-cause columns, so downstream consumers can either
    read the plain-language summary or plot the full decomposition.

    `verify=True` runs the efficiency check on every call. It is one
    vectorised comparison over a per-battery table, so the cost is
    negligible relative to catching a silently wrong explanation.
    """
    if verify:
        verify_efficiency(data, terms, scores, reference=reference)

    phi = additive_shapley_values(data, terms, reference=reference)
    ranked = rank_attributions(phi, terms, top_n=top_n)

    out = phi.rename(columns={c: f"{prefix}shap_{c}" for c in phi.columns})
    ranked = ranked.rename(columns={c: f"{prefix}{c}" for c in ranked.columns})
    out = out.join(ranked)
    out[f"{prefix}attribution_reference"] = reference
    return out
