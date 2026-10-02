"""Tests for RUL by fade extrapolation.

The failure modes here are not modelling ones - nothing is fitted across cells
- they are leakage and arithmetic:

`test_estimate_uses_only_history_up_to_the_cycle` — the one that would quietly
invalidate every number in the study. An estimate that saw cycle n+1 is being
scored against a future it was shown.

`test_a_flat_trajectory_is_refused` — a cell with no measurable fade
extrapolates to an end of life arbitrarily far away. The refusal is the
correct output; a number would not be.

`test_observed_eol_is_smoothed` — a single noisy discharge dipping below the
threshold would otherwise become the ground truth every prediction is scored
against, moving end of life hundreds of cycles early.

The fixtures are synthetic fade trajectories, not CALCE measurements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.rul.fade_extrapolation import (
    MAX_HORIZON_RATIO,
    MIN_HISTORY,
    estimate_rul,
    observed_eol,
    rul_error_table,
    rul_from_cycle_capacity,
)

THRESHOLD = 0.90


def _linear_cell(n: int = 400, start: float = 1.00,
                 per_cycle: float = 0.0005) -> tuple[np.ndarray, np.ndarray]:
    """A cell fading linearly. Crosses 0.90 at cycle 200 by construction."""
    cycles = np.arange(1, n + 1, dtype=float)
    return cycles, start - per_cycle * cycles


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_observed_eol_finds_the_construction_crossing():
    cycles, soh = _linear_cell()
    assert observed_eol(cycles, soh, THRESHOLD) == pytest.approx(200, abs=6)


def test_observed_eol_is_nan_when_the_cell_never_crosses():
    cycles, soh = _linear_cell(n=100)     # only reaches 0.95
    assert np.isnan(observed_eol(cycles, soh, THRESHOLD))


def test_observed_eol_is_smoothed():
    """One bad discharge must not declare end of life early."""
    cycles, soh = _linear_cell()
    soh = soh.copy()
    soh[40] = 0.80                        # a single anomalous reading
    assert observed_eol(cycles, soh, THRESHOLD) == pytest.approx(200, abs=6)


def test_rul_on_a_linear_cell_is_close_to_exact():
    cycles, soh = _linear_cell()
    est = estimate_rul(cycles, soh, at_cycle=150, threshold=THRESHOLD)
    assert est.refusal == ""
    assert est.rul_cycles == pytest.approx(50, abs=10)


def test_rul_shrinks_as_the_cell_ages():
    cycles, soh = _linear_cell()
    early = estimate_rul(cycles, soh, 100, THRESHOLD).rul_cycles
    late = estimate_rul(cycles, soh, 180, THRESHOLD).rul_cycles
    assert late < early


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_estimate_uses_only_history_up_to_the_cycle():
    """Changing the future must not change the estimate.

    If it does, the estimator has seen data it would not have at that cycle
    and every score in the study is optimistic by an unknown amount.
    """
    cycles, soh = _linear_cell()
    baseline = estimate_rul(cycles, soh, 150, THRESHOLD).rul_cycles

    tampered = soh.copy()
    tampered[cycles > 150] = 0.10          # future falls off a cliff
    after = estimate_rul(cycles, tampered, 150, THRESHOLD).rul_cycles

    assert after == pytest.approx(baseline, rel=1e-9)


def test_history_length_is_reported():
    cycles, soh = _linear_cell()
    assert estimate_rul(cycles, soh, 150, THRESHOLD).n_history > 0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_too_little_history_is_refused():
    cycles, soh = _linear_cell()
    est = estimate_rul(cycles, soh, at_cycle=MIN_HISTORY - 5,
                       threshold=THRESHOLD)
    assert np.isnan(est.rul_cycles)
    assert str(MIN_HISTORY) in est.refusal


def test_a_flat_trajectory_is_refused():
    """No measurable fade extrapolates to an arbitrary end of life."""
    cycles = np.arange(1, 300, dtype=float)
    soh = np.full_like(cycles, 0.99)
    est = estimate_rul(cycles, soh, 250, THRESHOLD)
    assert np.isnan(est.rul_cycles)
    assert "fade" in est.refusal


def test_a_rising_trajectory_is_refused_not_negated():
    """A cell reading healthier over time must not yield a negative slope
    solution that looks like a valid end of life."""
    cycles = np.arange(1, 300, dtype=float)
    soh = 0.90 + 0.0002 * cycles
    est = estimate_rul(cycles, soh, 250, THRESHOLD)
    assert np.isnan(est.rul_cycles)


def test_a_cell_already_past_the_threshold_reports_zero_not_negative():
    cycles, soh = _linear_cell()
    est = estimate_rul(cycles, soh, at_cycle=300, threshold=THRESHOLD)
    assert est.rul_cycles == 0.0
    assert est.refusal


# ---------------------------------------------------------------------------
# The scored table
# ---------------------------------------------------------------------------

def _frame(cells: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    blocks = [
        pd.DataFrame({"cell_id": cid, "cycle": c, "soh": s, "cohort": "T1"})
        for cid, (c, s) in cells.items()
    ]
    return pd.concat(blocks, ignore_index=True)


def test_error_table_scores_against_the_observed_crossing():
    frame = _frame({"A": _linear_cell()})
    table = rul_error_table(frame, threshold=THRESHOLD, stride=25)
    scored = table.dropna(subset=["error"])
    assert not scored.empty
    assert (scored["rul_true"] > 0).all()
    # A linear cell is the easy case; errors should be small.
    assert scored["error"].abs().median() < 25


def test_a_cell_that_never_crosses_is_excluded_with_a_reason():
    frame = _frame({"A": _linear_cell(), "B": _linear_cell(n=100)})
    table = rul_error_table(frame, threshold=THRESHOLD, stride=25)
    b_rows = table[table["cell_id"] == "B"]
    assert len(b_rows) == 1
    assert "never crosses" in b_rows.iloc[0]["refusal"]
    assert np.isnan(b_rows.iloc[0]["error"])


def test_missing_columns_are_named():
    with pytest.raises(ValueError) as excinfo:
        rul_error_table(pd.DataFrame({"cell_id": ["A"], "cycle": [1]}))
    assert "soh" in str(excinfo.value)


def test_error_sign_convention_is_prediction_minus_truth():
    """Negative error must mean the estimate ran early, not late.

    Every conclusion about conservative bias depends on this sign.
    """
    frame = _frame({"A": _linear_cell()})
    table = rul_error_table(frame, threshold=THRESHOLD, stride=25)
    scored = table.dropna(subset=["error"])
    recomputed = scored["rul_pred"] - scored["rul_true"]
    assert np.allclose(scored["error"], recomputed)


# ---------------------------------------------------------------------------
# The horizon bound
# ---------------------------------------------------------------------------

def test_an_estimate_reaching_far_beyond_its_history_is_refused():
    """The NASA failure: a nearly-flat early trend solves for a distant EOL.

    The arithmetic is sound and the slope is finite, so MIN_SLOPE does not
    catch it. What is wrong is that the estimate reaches further ahead than
    the data behind it can see.
    """
    cycles = np.arange(1, 81, dtype=float)
    soh = 1.00 - 0.00005 * cycles          # 0.90 is ~2000 cycles away
    est = estimate_rul(cycles, soh, at_cycle=80, threshold=THRESHOLD)
    assert np.isnan(est.rul_cycles)
    assert "cannot see that far" in est.refusal


def test_a_near_term_estimate_on_the_same_history_is_kept():
    """The bound must not refuse an estimate the history does support."""
    cycles = np.arange(1, 201, dtype=float)
    soh = 1.00 - 0.0005 * cycles           # crosses 0.90 at cycle 200
    est = estimate_rul(cycles, soh, at_cycle=150, threshold=THRESHOLD)
    assert est.refusal == ""
    assert est.rul_cycles == pytest.approx(50, abs=10)


def test_the_bound_scales_with_available_history():
    """Twice the history should permit roughly twice the reach."""
    slope = 0.0001                          # crosses 0.90 at cycle 1000
    short_c = np.arange(1, 101, dtype=float)
    long_c = np.arange(1, 401, dtype=float)
    short = estimate_rul(short_c, 1.00 - slope * short_c, 100, THRESHOLD)
    long = estimate_rul(long_c, 1.00 - slope * long_c, 400, THRESHOLD)
    assert np.isnan(short.rul_cycles), "99 cycles cannot see 900 ahead"
    assert np.isfinite(long.rul_cycles), "399 cycles can see 600 ahead"


def test_the_ratio_is_configurable_and_the_default_is_the_constant():
    cycles = np.arange(1, 81, dtype=float)
    soh = 1.00 - 0.00005 * cycles
    assert MAX_HORIZON_RATIO == 2.0
    tight = estimate_rul(cycles, soh, 80, THRESHOLD, max_horizon_ratio=0.5)
    loose = estimate_rul(cycles, soh, 80, THRESHOLD, max_horizon_ratio=100.0)
    assert np.isnan(tight.rul_cycles)
    assert np.isfinite(loose.rul_cycles)


# ---------------------------------------------------------------------------
# The pipeline adapter
# ---------------------------------------------------------------------------

def _cycle_frame(capacities: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "cell_id": "A",
        "cycle": range(1, len(capacities) + 1),
        "capacity_ah": capacities,
    })


def test_adapter_derives_soh_from_the_cells_own_early_cycles():
    """No nameplate: health is relative to this cell's first cycles."""
    # Ends at SOH 0.927 - still above the 0.90 threshold, so end of life is
    # genuinely ahead and a positive remaining life is the right answer.
    caps = [1.10] * 5 + list(np.linspace(1.10, 1.02, 195))
    est = rul_from_cycle_capacity(_cycle_frame(caps))
    assert est.refusal == ""
    assert est.rul_cycles > 0


def test_adapter_refuses_a_short_bench_capture():
    """A 3-cycle rig log cannot support an extrapolation, and must say so.

    This is the real serial path: `rig_stage_a.txt` segments into far fewer
    cycles than MIN_HISTORY, so the pipeline keeps the labelled heuristic
    rather than publishing a number the log cannot carry.
    """
    est = rul_from_cycle_capacity(_cycle_frame([1.10, 1.09, 1.08]))
    assert np.isnan(est.rul_cycles)
    assert est.refusal


def test_adapter_reports_zero_once_the_threshold_is_passed():
    """A cell already below the threshold has no remaining life to it."""
    caps = [1.10] * 5 + list(np.linspace(1.10, 0.80, 195))
    est = rul_from_cycle_capacity(_cycle_frame(caps))
    assert est.rul_cycles == 0.0


def test_adapter_refuses_without_a_usable_reference():
    est = rul_from_cycle_capacity(_cycle_frame([0.0] * 60))
    assert np.isnan(est.rul_cycles)
    assert "reference" in est.refusal


def test_adapter_names_a_missing_column():
    with pytest.raises(ValueError) as excinfo:
        rul_from_cycle_capacity(pd.DataFrame({"cycle": [1, 2, 3]}))
    assert "capacity_ah" in str(excinfo.value)
