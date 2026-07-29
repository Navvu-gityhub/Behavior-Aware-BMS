"""CAN/DBC ingestion adapter.

Decodes CAN frames described by a DBC file into the unified BMS schema
(`src/bms/preprocessing/schema.py`), the same role `load_nasa.py` and
`load_calce.py` play for their respective datasets. This is the
concrete "CAN Bus / adapter layer" piece of connecting the pipeline to
real EV telemetry -- see docs/can_dbc_adapter.md for the honest scope of
what this does and doesn't prove.

Uses `cantools` (an established, actively-maintained DBC-parsing
library) for the actual bit-decoding rather than hand-rolling it. DBC's
big-endian ("Motorola") bit-numbering scheme is a well-known source of
subtle, hard-to-spot correctness bugs if reimplemented from scratch --
exactly the kind of silent-wrongness this project has caught and fixed
before (see the NaN-as-healthy bug in health/health_index.py). Using a
purpose-built, widely-used library here is the same engineering judgment
call as using pandas/scikit-learn/statsmodels elsewhere in this codebase
rather than reimplementing statistics from scratch.

The bundled example DBC (`dbc_examples/twizy_bms_1.dbc`) is real,
publicly-documented vehicle protocol data -- see
`dbc_examples/SOURCE.md` for provenance and honest scope. It covers
exactly one real vehicle's one real BMS status message (current, SoC).
It is not broad multi-OEM coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import cantools
import pandas as pd

DEFAULT_DBC_PATH = Path(__file__).parent / "dbc_examples" / "twizy_bms_1.dbc"

# Maps DBC signal names (vehicle/message-specific) to unified-schema
# column names. Deliberately small and explicit rather than reusing
# schema.py's FIELD_ALIASES table: DBC signal names are per-vehicle and
# arbitrary (this one happens to follow OVMS's `v_b_*` convention), so
# guessing at a broader automatic mapping would risk silently mapping the
# wrong signal to the wrong field for some other vehicle's DBC. Extend
# this per DBC file, don't assume it generalizes.
TWIZY_BMS_1_SIGNAL_MAP: Mapping[str, str] = {
    "v_b_current": "current_a",
    "v_b_soc": "soc",
    # v_c_climit (charge current limit) has no unified-schema equivalent;
    # kept as-is in the output rather than silently dropped.
}


def load_dbc(path: str | Path = DEFAULT_DBC_PATH) -> "cantools.database.can.Database":
    """Parse a DBC file. Thin wrapper so callers don't need to import
    cantools directly, and so the default example DBC has one canonical
    load path other modules/tests can rely on."""
    return cantools.database.load_file(str(path))


def decode_can_frame(db: "cantools.database.can.Database", can_id: int, data: bytes) -> dict:
    """Decode one CAN frame's raw bytes into named signal values.

    Raises cantools' own KeyError (not swallowed) if `can_id` isn't in
    the loaded DBC -- an unrecognized frame ID should be a loud failure,
    not silently skipped or returned as an empty dict. Consistent with
    this project's general stance on missing/unrecognized data (see the
    NaN-as-healthy fix): fail explicitly rather than let it look like
    "no signal" when it's actually "we don't know how to decode this."
    """
    try:
        return db.decode_message(can_id, data)
    except KeyError as exc:
        raise KeyError(
            f"CAN ID {can_id} (0x{can_id:x}) is not defined in this DBC. "
            f"Known IDs: {[m.frame_id for m in db.messages]}"
        ) from exc


def can_frames_to_unified_schema(
    frames: Iterable[Mapping],
    *,
    db: Optional["cantools.database.can.Database"] = None,
    signal_map: Mapping[str, str] = TWIZY_BMS_1_SIGNAL_MAP,
    cell_id: str,
    dataset: str = "obd_can",
) -> pd.DataFrame:
    """Decode a sequence of raw CAN frames into a unified-schema-shaped
    DataFrame, the same output shape `load_nasa.py`/`load_calce.py`
    produce for their datasets.

    Each item in `frames` is a mapping with at least `can_id` (int) and
    `data` (bytes); an optional `timestamp` is passed through if present.

    Honesty note (read before assuming this output is pipeline-ready):
    this does NOT guarantee the result passes
    `schema.standardize_validate_bms_data`'s validation. If the DBC's
    messages don't cover a required field (e.g. this bundled example has
    no voltage or temperature signal), the resulting DataFrame will be
    missing that column, and validation will correctly flag it rather
    than the pipeline silently proceeding on incomplete data. That's the
    intended, honest behavior -- see tests/test_can_dbc.py for a test
    that specifically confirms validation catches this, rather than
    treating an all-tests-pass result as this adapter being "done."
    """
    db = db or load_dbc()
    rows = []
    for frame in frames:
        decoded = decode_can_frame(db, frame["can_id"], frame["data"])
        row = {signal_map.get(k, k): v for k, v in decoded.items()}
        row["cell_id"] = cell_id
        row["dataset"] = dataset
        if "timestamp" in frame:
            row["timestamp"] = frame["timestamp"]
        rows.append(row)

    return pd.DataFrame(rows)
