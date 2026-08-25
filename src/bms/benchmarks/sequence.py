"""A genuine sequence model: LSTM over consecutive-cycle windows.

WHY THIS IS A REAL SEQUENCE MODEL AND `mlp` IS NOT
---------------------------------------------------
`classical.mlp` is a neural network applied to one row — a flat feature vector
describing a single cycle. It has no notion of order and would score
identically if the cycles were shuffled. Calling that "the deep learning
baseline" would misdescribe what was tested.

This module builds windows of `W` **consecutive cycles** per cell and feeds
them to an LSTM, so the model sees a trajectory. Shuffling the cycles would
destroy its input, which is the operational definition of a sequence model.

IS A SEQUENCE FORMULATION DEFENSIBLE ON THIS DATA? YES — MEASURED, NOT ASSUMED
------------------------------------------------------------------------------
The check was run before this module was written, because the honest answer
might have been no:

* Cycle index is **strictly increasing within every one of the 32 cells** —
  zero violations. There is a real chronological order to exploit.
* **99.4% of consecutive-cycle gaps are exactly 1**, and the largest gap is 2.
  The series is effectively contiguous, so a window of W rows is a window of
  W cycles rather than an arbitrary span of unknown duration.
* The shortest cell has **21 cycles**, the median 67, the longest 193.

That last figure sets the window length. At W=10 every cell contributes at
least 12 windows and the corpus is 2,360 windows; at W=30 six cells drop out
entirely. W=10 is the default for two reasons: it keeps all 32 cells, and it
gives the LSTM twice the temporal context of the hand-engineered
`trailing_*` features (a 5-cycle trailing window) that it is competing
against. If the sequence model cannot beat those with strictly more history,
that is an informative result rather than a handicap.

WHAT THIS IS NOT
----------------
Each timestep is a **cycle-level aggregate**, not raw telemetry. The frame
carries one row per cycle with means and counts already computed, so this is a
sequence over summaries, not over the voltage/current trace. A sequence model
on raw within-cycle telemetry is a different and probably better experiment;
it is registered in `curves.py` as `sequence_model` and correctly reports
UNAVAILABLE, because this repository does not retain those traces.

The axis is also **cycle count, not wall-clock time**. Calendar effects are
invisible here. That is a property of the dataset, stated rather than
papered over.

LEAKAGE: THE THREE PLACES IT COULD ENTER, AND WHAT STOPS IT
------------------------------------------------------------
**1. Across the fold boundary.** Windows are built separately for the train
frame and the test frame; a window never spans both. Under LOBO the held-out
cell contributes no training window at all, and under LOCO neither does any
cell of the held-out protocol. The `Validator` supplies disjoint frames and
this module never sees the other side.

**2. Through the scaler.** `StandardScaler` is fitted on training windows
only and applied to test windows. Fitting on the union is the classic quiet
leak and would flatter every reported number.

**3. From the future.** A window for the row at cycle *k* contains cycles
*k−W+1 … k* and predicts the target at *k*. It never reads *k+1*. The
construction is tested directly (`tests/test_sequence.py`) rather than
asserted here.

Note what is *not* leakage: using the held-out cell's own **earlier cycles**
as input when predicting its later ones. That is precisely the deployment
situation — a battery's history is available to predict its present state —
and no held-out *target* is ever read. Refusing it would test a question
nobody asks.

EARLY CYCLES, AND WHY THEY ARE PADDED RATHER THAN DROPPED
----------------------------------------------------------
The first W−1 rows of each cell have insufficient history. Dropping them would
score the LSTM on an easier subset than every other method in the table,
making the comparison meaningless. They are instead left-padded by repeating
that cell's earliest observed cycle — the same thing a deployed system does
when a battery is new.

`predict` reports how many rows were padded, and
`scripts/run_sequence_study.py` additionally reports MAE restricted to
full-history rows, so the effect of the convention is visible rather than
buried in an aggregate.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import FitFn
from src.bms.benchmarks.classical import DEFAULT_FEATURES
from src.bms.benchmarks.registry import BenchmarkMethod, Family, register

RANDOM_SEED = 20260821

# See the module docstring: keeps all 32 cells, and gives the model twice the
# context of the `trailing_*` features it competes against.
DEFAULT_WINDOW = 10

# Deliberately small. 2,360 windows of 10x7 is not a regime where capacity
# helps, and a large network here would overfit the training cells and make
# the LOBO/LOCO gap a statement about model size rather than about protocol
# shift.
DEFAULT_HIDDEN = 32
DEFAULT_LAYERS = 1
DEFAULT_EPOCHS = 60
DEFAULT_BATCH = 64
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 8

# Fraction of *training cells* held out for early stopping. Split by cell, not
# by window: windows from one cell overlap heavily, so a random window split
# would put near-duplicates on both sides and the early-stopping signal would
# be measuring memorisation.
VALIDATION_CELL_FRACTION = 0.2

CELL_COLUMN = "cell_id"
CYCLE_COLUMN = "cycle"


def _import_torch():
    """Import torch through a single choke point that avoids an OpenMP abort.

    On Windows with an Anaconda numpy, importing torch into a process that has
    already loaded MKL aborts the interpreter outright::

        OMP: Error #15: Initializing libiomp5md.dll, but found
             libiomp5md.dll already initialized.

    torch ships its own Intel OpenMP runtime and MKL brings another; two
    copies in one process is unsupported. The failure is a hard `abort()`, not
    an exception, so it takes the whole test run down and cannot be caught.

    Forcing a single-threaded OpenMP before torch loads resolves it, because
    only one runtime ever spins up a thread pool. Verified to leave numerical
    results bit-identical — an `np.linalg.lstsq` fit returns the same
    coefficients with and without the setting.

    Deliberately NOT `KMP_DUPLICATE_LIB_OK=TRUE`, which is the workaround the
    error message itself suggests: Intel documents it as unsafe and warns it
    "may cause crashes or silently produce incorrect results". Silently
    incorrect results are exactly what this project cannot tolerate.

    `setdefault` throughout, so an operator who has deliberately configured
    threading keeps their choice.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

    import torch

    # Also pinned for reproducibility: multi-threaded reductions are not
    # deterministic in their summation order.
    torch.set_num_threads(1)
    return torch


@dataclass(frozen=True)
class WindowSet:
    """Windows built from one frame, with the provenance to audit them."""

    x: np.ndarray            # (n_windows, window, n_features)
    y: np.ndarray            # (n_windows,)
    cells: np.ndarray        # (n_windows,) cell id per window
    end_cycles: np.ndarray   # (n_windows,) cycle index the window ends on
    padded: np.ndarray       # (n_windows,) bool: needed left-padding

    def __len__(self) -> int:
        return len(self.y)


def build_windows(
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
    window: int = DEFAULT_WINDOW,
    cell_col: str = CELL_COLUMN,
    cycle_col: str = CYCLE_COLUMN,
    require_target: bool = True,
) -> WindowSet:
    """One window per row: cycles [k-W+1 .. k] predicting the target at k.

    Rows are sorted by cycle within each cell first, so the window is
    chronological regardless of the input frame's row order. Windows never
    span two cells.

    With `require_target=False` the target column may be absent or NaN and `y`
    is filled with NaN — used at predict time, where the target is exactly
    what is not known.
    """
    missing = [c for c in (*features, cell_col, cycle_col) if c not in data.columns]
    if missing:
        raise ValueError(f"build_windows: missing columns {missing}")
    if require_target and target not in data.columns:
        raise ValueError(f"build_windows: missing target column '{target}'")
    if window < 1:
        raise ValueError(f"build_windows: window must be >= 1, got {window}")

    columns = list(features)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    cells: list[object] = []
    ends: list[float] = []
    pads: list[bool] = []

    for cell_id, group in data.groupby(cell_col, sort=True):
        ordered = group.sort_values(cycle_col)
        matrix = ordered[columns].to_numpy(dtype=float)
        cycles = ordered[cycle_col].to_numpy(dtype=float)
        if target in ordered.columns:
            targets = ordered[target].to_numpy(dtype=float)
        else:
            targets = np.full(len(ordered), np.nan)

        for index in range(len(ordered)):
            start = index - window + 1
            if start >= 0:
                block = matrix[start:index + 1]
                padded = False
            else:
                # Left-pad by repeating this cell's earliest observed cycle.
                # See the module docstring for why padding beats dropping.
                pad = np.repeat(matrix[:1], -start, axis=0)
                block = np.vstack([pad, matrix[:index + 1]])
                padded = True

            xs.append(block)
            ys.append(float(targets[index]))
            cells.append(cell_id)
            ends.append(float(cycles[index]))
            pads.append(padded)

    if not xs:
        empty = np.empty((0, window, len(columns)))
        return WindowSet(empty, np.empty(0), np.empty(0, dtype=object),
                         np.empty(0), np.empty(0, dtype=bool))

    return WindowSet(
        x=np.stack(xs),
        y=np.asarray(ys, dtype=float),
        cells=np.asarray(cells, dtype=object),
        end_cycles=np.asarray(ends, dtype=float),
        padded=np.asarray(pads, dtype=bool),
    )


class _LSTMRegressor:
    """Minimal LSTM regressor over fixed-length windows.

    Kept as a plain class rather than a sklearn estimator: the harness needs
    only fit/predict, and a full estimator wrapper would add surface area
    without adding capability.
    """

    def __init__(
        self,
        n_features: int,
        window: int = DEFAULT_WINDOW,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH,
        learning_rate: float = DEFAULT_LR,
        patience: int = DEFAULT_PATIENCE,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.n_features = n_features
        self.window = window
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        # Annotated Any rather than nn.Module: torch is an optional dependency
        # and importing it at module scope for a type would defeat the point of
        # `requires_package` gating.
        self._model: Any = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.epochs_run = 0

    def _build(self):
        torch = _import_torch()
        from torch import nn

        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(self, n_features: int, hidden: int, layers: int):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features, hidden_size=hidden,
                    num_layers=layers, batch_first=True,
                )
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                # Last timestep only: the prediction is for the cycle the
                # window ends on.
                return self.head(out[:, -1, :]).squeeze(-1)

        return Net(self.n_features, self.hidden, self.layers)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        return (x - self._mean) / self._std

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "_LSTMRegressor":
        torch = _import_torch()
        from torch import nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Scaler fitted on training windows only. Per-feature over the flattened
        # (window, feature) axes, so a feature is scaled identically at every
        # timestep — scaling timesteps separately would make the same physical
        # quantity mean different things at different positions.
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        # A constant feature has zero spread; dividing by it yields inf/NaN and
        # poisons every window. Substituting 1.0 leaves it centred at zero,
        # which is the correct treatment of a column carrying no information
        # (NASA's `fast_charge_duration` is identically zero, hence the guard).
        std[std == 0] = 1.0
        self._mean, self._std = mean, std

        x_scaled = self._standardise(x)

        # Early-stopping split by CELL, not by window. Overlapping windows from
        # one cell are near-duplicates; splitting them randomly would let the
        # validation set contain near-copies of training rows.
        unique_cells = np.array(sorted({str(c) for c in groups}))
        rng = np.random.default_rng(self.seed)
        n_validation = max(1, int(round(len(unique_cells) * VALIDATION_CELL_FRACTION)))
        if len(unique_cells) <= 1:
            validation_cells: set[str] = set()
        else:
            n_validation = min(n_validation, len(unique_cells) - 1)
            validation_cells = set(
                rng.choice(unique_cells, size=n_validation, replace=False).tolist()
            )

        is_validation = np.array([str(c) in validation_cells for c in groups])
        keep = np.isfinite(y)
        train_mask = keep & ~is_validation
        validation_mask = keep & is_validation

        if train_mask.sum() == 0:
            train_mask = keep
            validation_mask = np.zeros_like(keep)

        device = torch.device("cpu")
        self._model = self._build().to(device)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        x_train = torch.tensor(x_scaled[train_mask], dtype=torch.float32)
        y_train = torch.tensor(y[train_mask], dtype=torch.float32)
        has_validation = bool(validation_mask.sum() > 0)
        if has_validation:
            x_validation = torch.tensor(x_scaled[validation_mask], dtype=torch.float32)
            y_validation = torch.tensor(y[validation_mask], dtype=torch.float32)

        generator = torch.Generator().manual_seed(self.seed)
        n = len(x_train)
        best = float("inf")
        best_state = None
        stale = 0

        for epoch in range(self.epochs):
            self._model.train()
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                optimiser.zero_grad()
                loss = loss_fn(self._model(x_train[idx]), y_train[idx])
                loss.backward()
                optimiser.step()

            self.epochs_run = epoch + 1
            if not has_validation:
                continue

            self._model.eval()
            with torch.no_grad():
                score = float(loss_fn(self._model(x_validation), y_validation))
            if score < best - 1e-6:
                best, stale = score, 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = _import_torch()

        if self._model is None:
            raise RuntimeError("_LSTMRegressor.predict called before fit()")
        if len(x) == 0:
            return np.empty(0, dtype=float)

        self._model.eval()
        tensor = torch.tensor(self._standardise(x), dtype=torch.float32)
        with torch.no_grad():
            return self._model(tensor).numpy().astype(float)


def build_lstm(
    features: Sequence[str],
    target: str,
    window: int = DEFAULT_WINDOW,
    **kwargs,
) -> FitFn:
    """A `FitFn` wrapping the LSTM, for the existing benchmark harness.

    Returns exactly one prediction per test row, as the `Validator` requires:
    every row gets a window ending on its own cycle, left-padded where the
    cell's history is shorter than `window`.
    """
    columns = list(features)

    def fit(train: pd.DataFrame):
        windows = build_windows(train, columns, target, window=window)
        if len(windows) == 0:
            raise ValueError("build_lstm: training frame produced no windows")

        model = _LSTMRegressor(
            n_features=len(columns), window=window, **kwargs
        ).fit(windows.x, windows.y, windows.cells)

        fallback = float(np.nanmean(windows.y))

        def predict(test: pd.DataFrame) -> np.ndarray:
            test_windows = build_windows(
                test, columns, target, window=window, require_target=False,
            )
            if len(test_windows) == 0:
                return np.full(len(test), fallback)

            values = model.predict(test_windows.x)

            # `build_windows` groups by cell and sorts by cycle, so the
            # prediction order is not the test frame's row order. Map back by
            # (cell, cycle) rather than assuming alignment — an assumption that
            # would silently scramble every score.
            keyed = {
                (str(c), float(k)): v
                for c, k, v in zip(
                    test_windows.cells, test_windows.end_cycles, values, strict=True,
                )
            }
            out = np.array([
                keyed.get((str(c), float(k)), fallback)
                for c, k in zip(
                    test[CELL_COLUMN].to_numpy(),
                    test[CYCLE_COLUMN].to_numpy(dtype=float),
                    strict=True,
                )
            ], dtype=float)
            return np.where(np.isfinite(out), out, fallback)

        return predict

    return fit


register(BenchmarkMethod(
    name="lstm",
    family=Family.CLASSICAL,
    citation=(
        "Hochreiter & Schmidhuber, Long Short-Term Memory, Neural Computation "
        "9(8) 1735-1780 (1997); applied to battery SOH/RUL by e.g. Zhang et "
        "al., IEEE Trans. Veh. Technol. 67(7) (2018)."
    ),
    build=lambda features, target: build_lstm(features, target),
    default_features=DEFAULT_FEATURES,
    requires_package=("torch",),
    note=(
        f"Genuine sequence model: LSTM over {DEFAULT_WINDOW} consecutive "
        f"cycles per cell, hidden={DEFAULT_HIDDEN}, 1 layer, early stopping on "
        f"a cell-disjoint split of the training fold. Timesteps are cycle-level "
        f"aggregates, not raw telemetry, and the axis is cycle count rather "
        f"than wall-clock time."
    ),
))
