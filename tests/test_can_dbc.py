"""Tests for src.bms.io.load_can_dbc.

The critical test here is test_decode_matches_ovms_documented_example:
it decodes the exact raw bytes OVMS's own DBC primer uses as its worked
example, and asserts the output matches the exact values OVMS states
that decode to. That's the correctness anchor for the whole module --
if this test passes, the DBC parsing/decoding is verifiably correct
against real, independently-published vehicle protocol data, not just
internally self-consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.bms.io.load_can_dbc import (
    DEFAULT_DBC_PATH,
    can_frames_to_unified_schema,
    decode_can_frame,
    load_dbc,
)
from src.bms.preprocessing.schema import standardize_validate_bms_data


# The exact message OVMS's DBC primer uses as its worked example:
# https://docs.openvehicles.com/en/latest/components/vehicle_dbc/docs/dbc-primer.html
TWIZY_BMS_1_CAN_ID = 341  # 0x155
TWIZY_BMS_1_EXAMPLE_BYTES = bytes.fromhex("0596E7546D58006F")

# Values OVMS's documentation states this exact message decodes to.
OVMS_EXPECTED = {
    "v_c_climit": 25.0,
    "v_b_current": 58.25,
    "v_b_soc": 69.98,
}


def test_decode_matches_ovms_documented_example():
    """The correctness anchor: decode real bytes, compare against the
    real source's own stated result. Not a self-consistency check."""
    db = load_dbc()
    decoded = decode_can_frame(db, TWIZY_BMS_1_CAN_ID, TWIZY_BMS_1_EXAMPLE_BYTES)

    for signal, expected_value in OVMS_EXPECTED.items():
        assert signal in decoded, f"expected signal {signal!r} missing from decoded output"
        assert decoded[signal] == pytest.approx(expected_value, abs=0.001), (
            f"{signal}: decoded {decoded[signal]}, OVMS docs say {expected_value}"
        )


def test_unknown_can_id_raises_with_useful_message():
    db = load_dbc()
    with pytest.raises(KeyError, match="not defined in this DBC"):
        decode_can_frame(db, 0xDEAD, TWIZY_BMS_1_EXAMPLE_BYTES)


def test_dbc_example_file_actually_bundled_and_loadable():
    """Guards against the bundled DBC file silently going missing or
    unparseable -- separate from the correctness test above, which would
    also fail in that case but with a less specific signal."""
    assert DEFAULT_DBC_PATH.exists(), f"bundled DBC missing: {DEFAULT_DBC_PATH}"
    db = load_dbc()
    assert len(db.messages) >= 1


def test_can_frames_to_unified_schema_produces_expected_columns():
    frames = [
        {"can_id": TWIZY_BMS_1_CAN_ID, "data": TWIZY_BMS_1_EXAMPLE_BYTES, "timestamp": "2026-01-01T00:00:00"},
        {"can_id": TWIZY_BMS_1_CAN_ID, "data": bytes.fromhex("0000000000000000")},
    ]
    df = can_frames_to_unified_schema(frames, cell_id="TWIZY_TEST_01")

    assert len(df) == 2
    assert set(["current_a", "soc", "cell_id", "dataset"]).issubset(df.columns)
    assert df.iloc[0]["current_a"] == pytest.approx(58.25, abs=0.001)
    assert df.iloc[0]["soc"] == pytest.approx(69.98, abs=0.001)
    assert (df["cell_id"] == "TWIZY_TEST_01").all()
    assert (df["dataset"] == "obd_can").all()


def test_honest_integration_with_schema_validator_flags_missing_voltage_and_temp():
    """This is the important integration test, not a formality.

    The Twizy BMS_1 message has no voltage or temperature signal, and
    the unified schema marks both as required, non-nullable fields.
    This test asserts the EXISTING schema validator correctly catches
    that gap -- proving the adapter's output honestly fails validation
    when it's genuinely incomplete, rather than the adapter (or this
    test) papering over it. A green test suite for this module should
    mean "the adapter is correct AND the system correctly recognizes its
    limits," not "the limits were hidden well enough to pass."
    """
    frames = [{"can_id": TWIZY_BMS_1_CAN_ID, "data": TWIZY_BMS_1_EXAMPLE_BYTES}]
    df = can_frames_to_unified_schema(frames, cell_id="TWIZY_TEST_01")

    _, issues = standardize_validate_bms_data(df, dataset="obd_can", cell_id="TWIZY_TEST_01")

    issues_text = " ".join(issues)
    assert "voltage_v" in issues_text, (
        f"expected the validator to flag missing voltage_v; issues were: {issues}"
    )
    assert "temperature_c" in issues_text, (
        f"expected the validator to flag missing temperature_c; issues were: {issues}"
    )


if __name__ == "__main__":
    test_decode_matches_ovms_documented_example()
    test_unknown_can_id_raises_with_useful_message()
    test_dbc_example_file_actually_bundled_and_loadable()
    test_can_frames_to_unified_schema_produces_expected_columns()
    test_honest_integration_with_schema_validator_flags_missing_voltage_and_temp()
    print("All CAN/DBC adapter tests passed.")
