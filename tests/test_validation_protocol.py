"""Regression tests that pin the validation METHODOLOGY, not its outcomes.

Nothing here asserts that a particular method wins, that an R2 exceeds a
threshold, or that a score is good. Those would be outcome tests, and this
project's central finding is that outcomes on this data are not stable enough to
assert (six ranking claims made and withdrawn - see ADR 0013). What is stable,
and what must not silently change, is *how* the evaluation is performed.

Each test below corresponds to a defect this project has actually shipped or
narrowly avoided:

- cohort leakage, which is how an in-sample rho of 0.870 survived review once;
- a fixed-amp fast-charge threshold that never fired on real telemetry;
- a C-rate computed against a constant instead of the cell's rated capacity;
- imputing a missing quantity rather than refusing.

Detailed serial-path capacity behaviour lives in `test_hardware_readiness.py`.
This module covers the batch validation path those defects also reach.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.validation import Validator  # noqa: E402
from src.bms.features.behavior_features import (  # noqa: E402
    DEFAULT_RATED_CAPACITY_AH,
    compute_behavior_flags,
)

REPO = Path(__file__).resolve().parents[1]


def _frame(n_cohorts: int = 3, cells_per_cohort: int = 2, rows: int = 12) -> pd.DataFrame:
    """A small frame with a known cohort/cell structure and a learnable target."""
    records = []
    rng = np.random.default_rng(20260916)
    for cohort in range(n_cohorts):
        for cell in range(cells_per_cohort):
            for cycle in range(rows):
                records.append({
                    "cell_id": f"C{cohort}_{cell}",
                    "cohort": f"COHORT_{cohort}",
                    "cycle": cycle,
                    "avg_temp": 20.0 + 5.0 * cohort + rng.normal(0, 0.1),
                    "capacity_loss": 0.001 * cycle * (1 + cohort) + rng.normal(0, 1e-5),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cohort separation: the leak that produced a rho of 0.870
# ---------------------------------------------------------------------------


def test_the_held_out_cohort_never_appears_in_its_own_training_set():
    """The property every LOCO claim in this project rests on.

    Asserted by capturing what each fold was actually trained on, rather than
    trusting the split's name.
    """
    data = _frame()
    validator = Validator(data, target="capacity_loss", cohort_col="cohort")
    seen: list[set[str]] = []

    def fit_fn(train: pd.DataFrame):
        seen.append(set(train["cohort"]))
        mean = float(train["capacity_loss"].mean())
        return lambda test: np.full(len(test), mean)

    result = validator.cross_validate(fit_fn, group_col="cohort", split="LOCO")

    held_out = [fold.held_out for fold in result.folds]
    assert set(held_out) == set(data["cohort"].unique())
    assert len(seen) == len(held_out), "a fold was scored without being refitted"

    for group, train_cohorts in zip(held_out, seen, strict=True):
        assert group not in train_cohorts, (
            f"cohort {group} was held out and still appeared in its own "
            f"training set - this is the leak that makes a LOCO number "
            f"indistinguishable from an in-sample one"
        )


def test_the_held_out_cell_never_appears_in_its_own_training_set():
    data = _frame()
    validator = Validator(data, target="capacity_loss", cohort_col="cohort")
    train_cells: list[set[str]] = []

    def fit_fn(train: pd.DataFrame):
        train_cells.append(set(train["cell_id"]))
        mean = float(train["capacity_loss"].mean())
        return lambda test: np.full(len(test), mean)

    result = validator.cross_validate(fit_fn, group_col="cell_id", split="LOBO")

    for fold, cells in zip(result.folds, train_cells, strict=True):
        assert fold.held_out not in cells


def test_every_fold_refits_rather_than_reusing_a_shipped_model():
    """Reusing coefficients across folds is leakage wearing a split's name."""
    data = _frame()
    validator = Validator(data, target="capacity_loss", cohort_col="cohort")
    calls = {"n": 0}

    def fit_fn(train: pd.DataFrame):
        calls["n"] += 1
        mean = float(train["capacity_loss"].mean())
        return lambda test: np.full(len(test), mean)

    result = validator.cross_validate(fit_fn, group_col="cohort", split="LOCO")
    scored = [fold for fold in result.folds if fold.error is None]
    assert calls["n"] == len(scored)
    assert calls["n"] > 1


def test_train_and_test_partition_the_frame_without_overlap():
    data = _frame()
    validator = Validator(data, target="capacity_loss", cohort_col="cohort")
    sizes: list[tuple[int, int]] = []

    def fit_fn(train: pd.DataFrame):
        sizes.append((len(train), 0))
        mean = float(train["capacity_loss"].mean())
        return lambda test: np.full(len(test), mean)

    result = validator.cross_validate(fit_fn, group_col="cohort", split="LOCO")
    for fold, (n_train, _) in zip(
        [f for f in result.folds if f.error is None], sizes, strict=True
    ):
        assert fold.n_train + fold.n_test == len(data), (
            "train and test do not partition the frame; rows are duplicated "
            "or dropped"
        )
        assert fold.n_train == n_train


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_cross_validation_is_deterministic_across_repeat_runs():
    """A validation number that moves between runs cannot be pinned to prose."""
    data = _frame()

    def run() -> list[float]:
        validator = Validator(data, target="capacity_loss", cohort_col="cohort")

        def fit_fn(train: pd.DataFrame):
            mean = float(train["capacity_loss"].mean())
            return lambda test: np.full(len(test), mean)

        result = validator.cross_validate(fit_fn, group_col="cohort", split="LOCO")
        return [fold.mae for fold in result.folds]

    first, second = run(), run()
    assert first == pytest.approx(second, nan_ok=True)


def test_fold_order_is_stable():
    """Folds are ordered by sorted group name, so a manifest diff is readable."""
    data = _frame()
    validator = Validator(data, target="capacity_loss", cohort_col="cohort")

    def fit_fn(train: pd.DataFrame):
        mean = float(train["capacity_loss"].mean())
        return lambda test: np.full(len(test), mean)

    result = validator.cross_validate(fit_fn, group_col="cohort", split="LOCO")
    held_out = [fold.held_out for fold in result.folds]
    assert held_out == sorted(held_out)


# ---------------------------------------------------------------------------
# C-rate: the threshold that never fired, and the constant that replaced it
# ---------------------------------------------------------------------------


def test_c_rate_flags_are_computed_against_the_supplied_capacity():
    """Same current, different cell, different verdict - or the feature is a
    fixed-amp threshold wearing a C-rate's name."""
    frame = pd.DataFrame({
        "current_a": [-2.5], "temperature_c": [25.0], "soc": [50.0],
    })
    small = compute_behavior_flags(frame, rated_capacity_ah=2.0)
    large = compute_behavior_flags(frame, rated_capacity_ah=3.4)

    assert small["aggressive_discharge_event"].iloc[0] == 1
    assert large["aggressive_discharge_event"].iloc[0] == 0


def test_the_flag_scales_with_capacity_across_a_range():
    """The threshold must move continuously with capacity, not step once."""
    frame = pd.DataFrame({
        "current_a": [-3.0], "temperature_c": [25.0], "soc": [50.0],
    })
    fired = [
        int(compute_behavior_flags(
            frame, rated_capacity_ah=capacity
        )["aggressive_discharge_event"].iloc[0])
        for capacity in (1.0, 2.0, 2.9, 3.1, 5.0)
    ]
    # 3.0 A is above 1 C for capacities below 3.0 Ah and below it above.
    assert fired == [1, 1, 1, 0, 0], f"flag did not track capacity: {fired}"


def test_a_fixed_amp_discharge_threshold_cannot_return():
    """Regression gate on a defect that shipped and was caught by real data.

    The original flag fired above a flat 2 A. Validated against NASA telemetry
    it never fired once - observed current peaked at 1.54 A - because 2 A had
    been chosen to match this project's own simulator. The replacement divides
    by rated capacity, and this test fails if a bare amp comparison returns to
    the flag definitions.
    """
    source = (REPO / "src" / "bms" / "features" / "behavior_features.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def compute_behavior_flags", 1)[1].split("\ndef ", 1)[0]
    code = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("#")
    )
    code = re.sub(r'""".*?"""', "", code, flags=re.S)

    assert "c_rate" in code, "the flags no longer reference a C-rate at all"
    assert "rated_capacity_ah" in code, (
        "compute_behavior_flags no longer divides by rated capacity"
    )

    current_comparison = re.search(
        r'out\["current_a"\]\s*[<>]=?\s*-?\d', code
    )
    assert current_comparison is None, (
        f"a raw current comparison is back in the flag definitions: "
        f"{current_comparison.group(0) if current_comparison else ''!r}. "
        f"A fixed-amp threshold does not generalise across cells of different "
        f"capacity, which is why it never fired on NASA."
    )


def test_the_batch_default_capacity_is_named_not_inlined():
    """The remaining 2.0 Ah assumption must stay greppable.

    It is legitimate for the CAN and dataset paths (ADR 0014), and it is
    dangerous the moment it stops being visible.
    """
    assert DEFAULT_RATED_CAPACITY_AH == 2.0
    source = (REPO / "src" / "bms" / "features" / "behavior_features.py").read_text(
        encoding="utf-8"
    )
    signature = source.split("def compute_behavior_flags", 1)[1].split(")", 1)[0]
    assert "rated_capacity_ah: float = DEFAULT_RATED_CAPACITY_AH" in signature, (
        "the default capacity is inlined as a literal in the signature rather "
        "than referring to the named constant"
    )


# ---------------------------------------------------------------------------
# Refusal rather than imputation
# ---------------------------------------------------------------------------


def test_scoring_refuses_rather_than_imputing_an_unknown_capacity():
    from src.bms.telemetry.pipeline import score_telemetry_frame
    from src.bms.telemetry.sources import coverage_from_channels

    telemetry = pd.DataFrame({
        "cell_id": ["X"] * 4,
        "test_time_s": [0.0, 60.0, 120.0, 180.0],
        "voltage_v": [4.0, 3.9, 3.8, 3.7],
        "current_a": [-1.5, -1.5, -1.5, -1.5],
        "temperature_c": [25.0, 26.0, 27.0, 28.0],
        "soc": [100.0, 70.0, 40.0, 10.0],
    })
    coverage = coverage_from_channels(
        ("test_time_s", "voltage_v", "current_a", "temperature_c", "soc"),
        source_label="unit-test", transport="serial",
    )

    result = score_telemetry_frame(
        source_name="unit-test", telemetry=telemetry, coverage=coverage,
        n_frames=4, n_decoded=4, cell_id="X", rated_capacity_ah=None,
    )

    assert result.guardian.empty, "scored a frame with no C-rate basis"
    assert any("rated capacity" in refusal.lower() for refusal in result.refusals)
    # Segmentation does not divide by capacity, so it must still have run:
    # refusing to score is not the same as refusing to measure.
    assert "segment_cycles" in result.stages_completed
    assert "score" not in result.stages_completed


def test_a_validator_refuses_a_frame_missing_its_target():
    data = _frame().drop(columns=["capacity_loss"])
    with pytest.raises(ValueError, match="capacity_loss"):
        Validator(data, target="capacity_loss", cohort_col="cohort")


def test_a_validator_refuses_a_group_column_it_does_not_have():
    data = _frame()
    validator = Validator(data, target="capacity_loss", cohort_col="cohort")

    def fit_fn(train: pd.DataFrame):
        return lambda test: np.zeros(len(test))

    with pytest.raises(ValueError, match="protocol"):
        validator.cross_validate(fit_fn, group_col="protocol", split="LOCO")
