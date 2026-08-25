"""Generate an Oxford-format .mat fixture for regression tests.

This is a FIXTURE, not Oxford measurement data. It reproduces the published
nested-struct layout of Oxford Battery Degradation Dataset 1 exactly —

    Cell<N> -> cyc<NNNN> -> {C1ch, C1dc, OCVch, OCVdc} -> {t, v, q, T}

— so `load_oxford_mat` is exercised against the real field names, the real
nesting depth, and the real sparse characterisation cadence (every 100 cycles).

Any number produced from these is a property of the fixture. Nothing here may
be reported as an Oxford result.

WHY A FIXTURE AND NOT THE REAL ARCHIVE
--------------------------------------
The archive is ~1 GB and is not redistributable in this repository, so the
loader cannot be tested against it here. Writing a parser against a documented
format without ever executing it is how a silently wrong loader ships, and
this project already has one instance of exactly that (the CALCE "initial vs
post-storage" columns that turned out to be the same measurement twice).

A fixture does not prove the loader handles the real file. It proves the
loader handles the format as documented, which is the strongest claim
available without the data, and it means the day the archive lands the only
open question is whether the file matches its own documentation.

Deliberate awkwardness reproduced here, because each one broke something:

* `q` is written in **mAh**, as the published dataset does, so the unit
  detection in `_detect_capacity_scale` is exercised rather than bypassed.
* Cycle keys are zero-padded to four digits (`cyc0000`, `cyc0100`, `cyc1000`),
  so lexical sorting and numeric sorting agree for these but the loader still
  parses the integer rather than trusting the string order.
* One cell is given a missing measurement block, so the missing-block
  reporting path has coverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _block(
    n_samples: int,
    capacity_mah: float,
    discharge: bool,
    temperature_c: float,
    start_time_s: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """One measurement block: parallel t / v / q / T arrays."""
    fraction = np.linspace(0.0, 1.0, n_samples)

    # Accumulated charge over the block, which is what Oxford records.
    q = capacity_mah * fraction
    if discharge:
        # Sign conventions differ between exports of this dataset; the loader
        # takes an absolute value for exactly this reason, so the fixture
        # writes the negative convention to exercise it.
        q = -q

    v = (4.2 - 1.5 * fraction) if discharge else (2.7 + 1.5 * fraction)

    return {
        "t": start_time_s + fraction * 3600.0,
        "v": v + rng.normal(0, 0.004, n_samples),
        "q": q,
        "T": temperature_c + rng.normal(0, 0.3, n_samples),
    }


def make_oxford_mat(
    path: Path,
    n_cells: int = 3,
    n_characterisations: int = 5,
    cycle_stride: int = 100,
    n_samples: int = 40,
    initial_capacity_mah: float = 740.0,
    fade_per_characterisation_mah: float = 12.0,
    ambient_c: float = 40.0,
    drop_block: tuple[str, str] | None = ("Cell2", "OCVdc"),
    seed: int = 0,
) -> None:
    """Write a .mat file with Oxford's nested layout.

    `drop_block` omits one measurement block from every characterisation of
    one cell, so the loader's missing-block reporting is covered. Pass None
    for a complete file.
    """
    from scipy.io import savemat

    rng = np.random.default_rng(seed)
    payload: dict[str, dict] = {}

    for cell_index in range(1, n_cells + 1):
        cell_name = f"Cell{cell_index}"
        cell: dict[str, dict] = {}

        # A little between-cell spread in fade rate, so a per-cell trajectory
        # is distinguishable rather than every cell being identical.
        cell_fade = fade_per_characterisation_mah * (1.0 + 0.25 * (cell_index - 1))

        for step in range(n_characterisations):
            cycle = step * cycle_stride
            capacity = initial_capacity_mah - cell_fade * step
            start = float(cycle) * 3600.0

            blocks = {
                "C1ch": _block(n_samples, capacity, False, ambient_c, start, rng),
                "C1dc": _block(n_samples, capacity, True, ambient_c, start + 3600, rng),
                # The OCV sweeps run at C/18 and take far longer.
                "OCVch": _block(n_samples, capacity, False, ambient_c, start + 7200, rng),
                "OCVdc": _block(n_samples, capacity, True, ambient_c, start + 25200, rng),
            }

            if drop_block is not None and cell_name == drop_block[0]:
                blocks.pop(drop_block[1], None)

            cell[f"cyc{cycle:04d}"] = blocks

        payload[cell_name] = cell

    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), payload, do_compression=True)
