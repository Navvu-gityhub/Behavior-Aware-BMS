"""Tests for the LSTM sequence model and its window construction.

The load-bearing test is `test_window_never_contains_a_future_cycle`. Every
claim this project makes about the LSTM rests on the windows being genuinely
causal, and that is checkable directly rather than by inspection: the fixture
sets each feature value equal to its own cycle index, so a window ending at
cycle k must contain exactly the values k-W+1 … k and nothing larger.

The model itself is only smoke-tested. Asserting that an LSTM reaches some
accuracy on a fixture would be testing the fixture.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

# Skip on absence WITHOUT importing torch here.
#
# `pytest.importorskip("torch")` would import it at collection time, before
# `sequence._import_torch` has set the single-threaded OpenMP variables — and
# on Windows that combination aborts the interpreter outright rather than
# raising (OMP Error #15; see that function's docstring). An abort during
# collection takes the whole suite down, so the import has to stay lazy.
if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("LSTM benchmark needs the torch extra", allow_module_level=True)

from src.bms.benchmarks import get, load_all  # noqa: E402
from src.bms.benchmarks.sequence import (  # noqa: E402
    DEFAULT_WINDOW,
    build_lstm,
    build_windows,
)

FEATURES = ("f_cycle", "f_const")
TARGET = "y"


@pytest.fixture(scope="module", autouse=True)
def _registry_loaded():
    load_all()


def traceable_frame(n_cells: int = 3, n_cycles: int = 25) -> pd.DataFrame:
    """Each row's feature equals its own cycle index, so windows are auditable."""
    rows = []
    for cell in range(n_cells):
        for cycle in range(1, n_cycles + 1):
            rows.append({
                "cell_id": f"C{cell}",
                "cohort": f"P{cell % 2}",
                "cycle": cycle,
                "f_cycle": float(cycle),
                "f_const": 1.0,
                TARGET: 1.0 - 0.005 * cycle,
            })
    return pd.DataFrame(rows)


class TestWindowConstruction:
    def test_one_window_per_row(self):
        data = traceable_frame(n_cells=3, n_cycles=25)
        windows = build_windows(data, FEATURES, TARGET, window=10)
        assert len(windows) == len(data)

    def test_window_shape_is_time_major(self):
        windows = build_windows(traceable_frame(), FEATURES, TARGET, window=10)
        assert windows.x.shape[1:] == (10, len(FEATURES))

    def test_window_never_contains_a_future_cycle(self):
        """The causality guarantee, checked rather than asserted."""
        window = 10
        data = traceable_frame(n_cells=2, n_cycles=25)
        windows = build_windows(data, FEATURES, TARGET, window=window)

        cycle_channel = FEATURES.index("f_cycle")
        for i in range(len(windows)):
            end = windows.end_cycles[i]
            values = windows.x[i, :, cycle_channel]
            assert values.max() <= end, (
                f"window ending at cycle {end} contains cycle {values.max()}"
            )

    def test_full_history_window_is_exactly_the_preceding_cycles(self):
        window = 10
        data = traceable_frame(n_cells=1, n_cycles=25)
        windows = build_windows(data, FEATURES, TARGET, window=window)
        cycle_channel = FEATURES.index("f_cycle")

        # Row at cycle 20 (index 19) has full history.
        i = int(np.where(windows.end_cycles == 20)[0][0])
        assert not windows.padded[i]
        assert list(windows.x[i, :, cycle_channel]) == [
            float(c) for c in range(11, 21)
        ]

    def test_window_is_chronological_not_frame_order(self):
        """Shuffling the input rows must not change the windows."""
        data = traceable_frame(n_cells=2, n_cycles=20)
        ordered = build_windows(data, FEATURES, TARGET, window=5)
        shuffled = build_windows(
            data.sample(frac=1.0, random_state=0), FEATURES, TARGET, window=5,
        )
        assert np.allclose(ordered.x, shuffled.x)
        assert np.allclose(ordered.end_cycles, shuffled.end_cycles)

    def test_windows_never_span_two_cells(self):
        window = 10
        data = traceable_frame(n_cells=3, n_cycles=25)
        windows = build_windows(data, FEATURES, TARGET, window=window)
        cycle_channel = FEATURES.index("f_cycle")

        # A window spanning cells would show a cycle index jumping back down
        # mid-window without being a pad repeat.
        for i in range(len(windows)):
            values = windows.x[i, :, cycle_channel]
            steps = np.diff(values)
            assert np.all(steps >= 0), f"non-monotonic window at index {i}"

    def test_early_cycles_are_padded_and_flagged(self):
        window = 10
        data = traceable_frame(n_cells=1, n_cycles=25)
        windows = build_windows(data, FEATURES, TARGET, window=window)
        assert windows.padded[:window - 1].all()
        assert not windows.padded[window - 1:].any()

    def test_padding_repeats_the_earliest_cycle_not_zeros(self):
        data = traceable_frame(n_cells=1, n_cycles=25)
        windows = build_windows(data, FEATURES, TARGET, window=10)
        cycle_channel = FEATURES.index("f_cycle")
        first = windows.x[0, :, cycle_channel]
        assert set(first.tolist()) == {1.0}

    def test_target_is_aligned_to_the_window_end(self):
        data = traceable_frame(n_cells=1, n_cycles=25)
        windows = build_windows(data, FEATURES, TARGET, window=5)
        for i in range(len(windows)):
            expected = 1.0 - 0.005 * windows.end_cycles[i]
            assert windows.y[i] == pytest.approx(expected)

    def test_missing_target_is_tolerated_at_predict_time(self):
        data = traceable_frame(n_cells=1, n_cycles=12).drop(columns=[TARGET])
        windows = build_windows(
            data, FEATURES, TARGET, window=5, require_target=False,
        )
        assert len(windows) == 12
        assert np.isnan(windows.y).all()

    def test_missing_feature_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            build_windows(traceable_frame(), ("nope",), TARGET, window=5)

    def test_missing_target_raises_when_required(self):
        data = traceable_frame().drop(columns=[TARGET])
        with pytest.raises(ValueError, match="missing target column"):
            build_windows(data, FEATURES, TARGET, window=5)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window must be"):
            build_windows(traceable_frame(), FEATURES, TARGET, window=0)


class TestLstmModel:
    def test_trains_and_predicts_with_correct_shape(self):
        data = traceable_frame(n_cells=4, n_cycles=20)
        fit = build_lstm(FEATURES, TARGET, window=5, epochs=3, hidden=8)

        train = data[data["cell_id"] != "C3"]
        test = data[data["cell_id"] == "C3"]

        predict = fit(train)
        out = predict(test)

        assert out.shape == (len(test),)
        assert np.isfinite(out).all()

    def test_predictions_map_back_to_test_row_order(self):
        """build_windows sorts by cell/cycle; predictions must be re-aligned.

        Assuming positional alignment would silently scramble every score, so
        a shuffled test frame must produce correspondingly shuffled output.
        """
        data = traceable_frame(n_cells=4, n_cycles=20)
        fit = build_lstm(FEATURES, TARGET, window=5, epochs=3, hidden=8)
        predict = fit(data[data["cell_id"] != "C3"])

        test = data[data["cell_id"] == "C3"].reset_index(drop=True)
        shuffled = test.sample(frac=1.0, random_state=7)

        ordered_out = predict(test)
        shuffled_out = predict(shuffled)

        assert np.allclose(ordered_out[shuffled.index.to_numpy()], shuffled_out)

    def test_is_deterministic_across_runs(self):
        data = traceable_frame(n_cells=4, n_cycles=20)
        train = data[data["cell_id"] != "C3"]
        test = data[data["cell_id"] == "C3"]

        first = build_lstm(FEATURES, TARGET, window=5, epochs=3, hidden=8)(train)(test)
        second = build_lstm(FEATURES, TARGET, window=5, epochs=3, hidden=8)(train)(test)
        assert np.allclose(first, second)

    def test_scaler_is_fitted_on_training_windows_only(self):
        """A scaler fitted on train+test is the classic quiet leak."""
        from src.bms.benchmarks.sequence import _LSTMRegressor

        data = traceable_frame(n_cells=3, n_cycles=20)
        windows = build_windows(data, FEATURES, TARGET, window=5)
        model = _LSTMRegressor(n_features=len(FEATURES), window=5,
                               epochs=2, hidden=8)
        model.fit(windows.x, windows.y, windows.cells)

        expected = windows.x.reshape(-1, len(FEATURES)).mean(axis=0)
        assert np.allclose(model._mean, expected)

    def test_empty_training_frame_raises(self):
        fit = build_lstm(FEATURES, TARGET, window=5, epochs=2, hidden=8)
        with pytest.raises(ValueError, match="no windows"):
            fit(traceable_frame().iloc[0:0])

    def test_predict_before_fit_raises(self):
        from src.bms.benchmarks.sequence import _LSTMRegressor

        model = _LSTMRegressor(n_features=2, window=5)
        with pytest.raises(RuntimeError, match="before fit"):
            model.predict(np.zeros((1, 5, 2)))


class TestRegistry:
    def test_lstm_is_findable(self):
        method = get("lstm")
        assert method.name == "lstm"
        assert method.requires_package == ("torch",)

    def test_lstm_declares_a_sequence_note(self):
        assert "sequence model" in get("lstm").note.lower()

    def test_lstm_is_distinct_from_mlp(self):
        """The tabular network must not be mistaken for the sequence model."""
        assert "NOT a sequence model" in get("mlp").note
