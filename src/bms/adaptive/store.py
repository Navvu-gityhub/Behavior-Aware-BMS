"""Versioned model store, promotion log, and rollback.

Every model this system deploys must be traceable to the evidence that
justified deploying it. That is the whole reason this module exists: a
promoted model whose justification was not recorded is indistinguishable, six
months later, from one that was never reviewed at all.

Three decisions shape the design.

**Parameters are stored as JSON, not pickles.** A governance system whose
artifacts cannot be read is not a governance system. JSON means `git diff`
shows exactly which coefficient moved between v3 and v4, a reviewer can
inspect a model without executing it, and there is no arbitrary-code-execution
surface on load. The cost is that a candidate must be able to express itself
as a JSON-serialisable mapping — and a model that cannot is, for this project's
purposes, too opaque to govern. That is a deliberate constraint rather than an
oversight.

**The decision log is append-only, and records rejections.** A store that only
kept promotions would present a history of unbroken success. The rejections
are the more informative half: they are the record of what the gate caught.
`scripts/fit_shap_attribution_model.py` firing its skill gate is a finding
worth keeping, not an embarrassment to discard.

**Rollback is a first-class operation.** Reverting is done by moving the active
pointer, never by deleting a version. Nothing is ever removed from the store,
so a bad promotion is always recoverable and always still visible in the log.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.bms.adaptive.validation import Verdict

# How far a coefficient may move between consecutive versions before the
# change is treated as suspicious, as a fraction of its previous magnitude.
#
# This is a stability guard, not an accuracy guard. A coefficient that doubles
# or flips sign on a refit is evidence the fit is being driven by whichever
# cells happened to arrive, and a model that unstable should not be promoted
# even in the rare case where its cross-validation numbers clear the gate.
# The temperature coefficient this project reports is 0.0038 Ah/degC with a
# 95% CI of 0.002-0.005 - roughly +/-40% - so 0.5 sits just outside the
# range ordinary resampling noise should produce.
DEFAULT_STABILITY_TOLERANCE = 0.5


@dataclass(frozen=True)
class ModelVersion:
    """One fitted model, plus the evidence that admitted it."""

    version_id: str
    cohort_id: str
    version: int
    params: Mapping[str, float]
    metrics: Mapping[str, float]
    reasons: tuple[str, ...]
    created_at: float
    parent_version: int | None = None
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["params"] = dict(self.params)
        payload["metrics"] = dict(self.metrics)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ModelVersion":
        return cls(
            version_id=payload["version_id"],
            cohort_id=payload["cohort_id"],
            version=int(payload["version"]),
            params=dict(payload["params"]),
            metrics=dict(payload["metrics"]),
            reasons=tuple(payload.get("reasons", ())),
            created_at=float(payload["created_at"]),
            parent_version=payload.get("parent_version"),
            note=payload.get("note", ""),
        )


@dataclass(frozen=True)
class Decision:
    """One entry in the append-only log. Covers rejections as well."""

    outcome: str  # PROMOTE | REJECT | ROLLBACK
    cohort_id: str
    version: int | None
    reasons: tuple[str, ...]
    metrics: Mapping[str, float]
    timestamp: float

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["metrics"] = dict(self.metrics)
        return payload


class StabilityError(ValueError):
    """A candidate's parameters moved too far from the version it replaces."""


class ModelStore:
    """Versioned per-cohort model storage with an audit trail.

    Layout on disk::

        <root>/
          active.json          cohort_id -> active version number
          versions/
            <cohort>__v1.json
            <cohort>__v2.json
          decisions.jsonl      append-only, one JSON object per line

    Models are stored per cohort rather than globally on purpose. ADR 0002
    found that the fitted model's skill lives in its per-cohort intercepts, so
    a single global artifact would bundle nine separate pieces of evidence into
    one object that can only be promoted or rejected as a unit.
    """

    def __init__(
        self,
        root: Path | str,
        stability_tolerance: float = DEFAULT_STABILITY_TOLERANCE,
    ) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.active_path = self.root / "active.json"
        self.log_path = self.root / "decisions.jsonl"
        self.stability_tolerance = stability_tolerance

        self.versions_dir.mkdir(parents=True, exist_ok=True)
        if not self.active_path.exists():
            self._write_active({})
        self.log_path.touch(exist_ok=True)

    # -- persistence helpers ------------------------------------------------

    def _read_active(self) -> dict[str, int]:
        return json.loads(self.active_path.read_text() or "{}")

    def _write_active(self, mapping: Mapping[str, int]) -> None:
        self.active_path.write_text(json.dumps(dict(mapping), indent=2, sort_keys=True))

    def _version_path(self, cohort_id: str, version: int) -> Path:
        # Cohort ids come from data and may contain characters that are awkward
        # in filenames; the version_id inside the file remains authoritative.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in cohort_id)
        return self.versions_dir / f"{safe}__v{version}.json"

    def _append_decision(self, decision: Decision) -> None:
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(decision.to_json(), sort_keys=True) + "\n")

    # -- queries ------------------------------------------------------------

    @property
    def cohorts(self) -> list[str]:
        return sorted(self._read_active())

    def active_version(self, cohort_id: str) -> int | None:
        return self._read_active().get(cohort_id)

    def active(self, cohort_id: str) -> ModelVersion | None:
        version = self.active_version(cohort_id)
        return None if version is None else self.load(cohort_id, version)

    def load(self, cohort_id: str, version: int) -> ModelVersion:
        path = self._version_path(cohort_id, version)
        if not path.exists():
            raise KeyError(f"No stored model for cohort '{cohort_id}' version {version}")
        return ModelVersion.from_json(json.loads(path.read_text()))

    def history(self, cohort_id: str) -> list[ModelVersion]:
        """Every stored version for a cohort, oldest first.

        Includes versions that are no longer active - a rolled-back model stays
        on disk so the decision remains reviewable.
        """
        versions = []
        for path in sorted(self.versions_dir.glob("*.json")):
            candidate = ModelVersion.from_json(json.loads(path.read_text()))
            if candidate.cohort_id == cohort_id:
                versions.append(candidate)
        return sorted(versions, key=lambda v: v.version)

    def decisions(self, cohort_id: str | None = None) -> Iterator[Decision]:
        """Replay the append-only log, optionally filtered to one cohort."""
        for line in self.log_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if cohort_id is not None and payload["cohort_id"] != cohort_id:
                continue
            yield Decision(
                outcome=payload["outcome"],
                cohort_id=payload["cohort_id"],
                version=payload.get("version"),
                reasons=tuple(payload.get("reasons", ())),
                metrics=payload.get("metrics", {}),
                timestamp=float(payload["timestamp"]),
            )

    # -- stability ----------------------------------------------------------

    def check_stability(
        self, cohort_id: str, params: Mapping[str, float]
    ) -> list[str]:
        """Compare a candidate's parameters against the active version.

        Returns a list of problems; empty means stable. A first version for a
        cohort has nothing to compare against and is therefore always stable -
        which is correct, but it means the first promotion into a cohort rests
        entirely on the cross-validation gate.
        """
        current = self.active(cohort_id)
        if current is None:
            return []

        problems: list[str] = []
        for key, old_value in current.params.items():
            if key not in params:
                problems.append(f"parameter '{key}' disappeared from the candidate")
                continue
            new_value = params[key]

            if old_value * new_value < 0:
                problems.append(
                    f"parameter '{key}' flipped sign ({old_value:.4g} -> {new_value:.4g})"
                )
                continue

            denominator = abs(old_value)
            if denominator < 1e-12:
                # A parameter that was effectively zero has no meaningful
                # relative scale, so relative change is not computable and
                # claiming instability would be an unsupported assertion.
                continue
            change = abs(new_value - old_value) / denominator
            if change > self.stability_tolerance:
                problems.append(
                    f"parameter '{key}' moved {change:.0%} "
                    f"({old_value:.4g} -> {new_value:.4g}), above the "
                    f"{self.stability_tolerance:.0%} tolerance"
                )

        for key in params:
            if key not in current.params:
                problems.append(f"parameter '{key}' is new in the candidate")
        return problems

    # -- mutation -----------------------------------------------------------

    def propose(
        self,
        cohort_id: str,
        params: Mapping[str, float],
        verdict: Verdict,
        note: str = "",
        enforce_stability: bool = True,
    ) -> ModelVersion | None:
        """Offer a candidate for promotion.

        Returns the stored `ModelVersion` on promotion and `None` on rejection.
        Either way the outcome is appended to the decision log, so a rejection
        is recorded evidence rather than a silent no-op.

        Promotion requires both that the gate passed and that the parameters
        are stable relative to the version being replaced. These are separate
        conditions: the gate asks whether the candidate generalises, stability
        asks whether the fit is being driven by which cells happened to arrive.
        """
        try:
            json.dumps(dict(params))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"ModelStore.propose: parameters for '{cohort_id}' are not "
                f"JSON-serialisable ({exc}). Models are stored as inspectable "
                f"JSON so that a reviewer can audit what changed between "
                f"versions without executing the artifact."
            ) from exc

        reasons = list(verdict.reasons)
        promote = bool(verdict.promote)

        if promote and enforce_stability:
            problems = self.check_stability(cohort_id, params)
            if problems:
                promote = False
                reasons.append(
                    "Parameter stability check failed: " + "; ".join(problems)
                )

        if not promote:
            self._append_decision(Decision(
                outcome="REJECT", cohort_id=cohort_id, version=None,
                reasons=tuple(reasons), metrics=dict(verdict.metrics),
                timestamp=time.time(),
            ))
            return None

        parent = self.active_version(cohort_id)
        version = (parent or 0) + 1
        record = ModelVersion(
            version_id=f"{cohort_id}__v{version}",
            cohort_id=cohort_id,
            version=version,
            params=dict(params),
            metrics=dict(verdict.metrics),
            reasons=tuple(reasons),
            created_at=time.time(),
            parent_version=parent,
            note=note,
        )
        self._version_path(cohort_id, version).write_text(
            json.dumps(record.to_json(), indent=2, sort_keys=True)
        )
        active = self._read_active()
        active[cohort_id] = version
        self._write_active(active)

        self._append_decision(Decision(
            outcome="PROMOTE", cohort_id=cohort_id, version=version,
            reasons=tuple(reasons), metrics=dict(verdict.metrics),
            timestamp=record.created_at,
        ))
        return record

    def rollback(self, cohort_id: str, reason: str = "") -> ModelVersion | None:
        """Revert the active pointer to the current version's parent.

        Nothing is deleted. The reverted version stays on disk and in the log,
        because a bad promotion that leaves no trace teaches nobody anything.
        """
        current = self.active(cohort_id)
        if current is None:
            raise KeyError(f"No active model for cohort '{cohort_id}' to roll back")
        if current.parent_version is None:
            raise ValueError(
                f"Cohort '{cohort_id}' is at v1, which has no parent to revert to. "
                f"Retiring a cohort entirely is a deliberate act - remove it from "
                f"active.json explicitly rather than rolling back into nothing."
            )

        parent = self.load(cohort_id, current.parent_version)
        active = self._read_active()
        active[cohort_id] = parent.version
        self._write_active(active)

        self._append_decision(Decision(
            outcome="ROLLBACK", cohort_id=cohort_id, version=parent.version,
            reasons=(reason or f"rolled back from v{current.version} to v{parent.version}",),
            metrics={}, timestamp=time.time(),
        ))
        return parent

    # -- reporting ----------------------------------------------------------

    def summary(self) -> list[dict[str, Any]]:
        """One row per cohort: what is live, and what the log says about it."""
        rows = []
        for cohort_id in self.cohorts:
            active = self.active(cohort_id)
            log = list(self.decisions(cohort_id))
            rows.append({
                "cohort_id": cohort_id,
                "active_version": None if active is None else active.version,
                "n_versions": len(self.history(cohort_id)),
                "n_promotions": sum(d.outcome == "PROMOTE" for d in log),
                "n_rejections": sum(d.outcome == "REJECT" for d in log),
                "n_rollbacks": sum(d.outcome == "ROLLBACK" for d in log),
                "loco_median_r2": (active.metrics.get("loco_median_r2") if active else None),
            })
        return rows
