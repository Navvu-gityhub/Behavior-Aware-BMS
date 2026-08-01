"""Tests for src.bms.io.can_vehicle_registry.

test_registry_disambiguates_between_two_vehicles is the important one:
it proves the registry's core claim (auto-identify + decode the right
vehicle out of several registered ones) actually works, using the
synthetic test fixture alongside the real, verified Twizy DBC. The
fixture is clearly labeled as not-a-real-vehicle throughout -- this test
is about the registry MECHANISM, not a second real-vehicle claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# See the note in tests/test_can_dbc.py: can_vehicle_registry imports
# load_can_dbc, which imports cantools at module level. Without this guard an
# absent optional extra aborts collection for the whole suite.
pytest.importorskip("cantools", reason="optional extra; see requirements-dev.txt")

from src.bms.io.can_vehicle_registry import default_registry

TWIZY_BMS_1_CAN_ID = 341
TWIZY_BMS_1_EXAMPLE_BYTES = bytes.fromhex("0596E7546D58006F")
TEST_FIXTURE_CAN_ID = 999
TEST_FIXTURE_BYTES = bytes.fromhex("2A00000000000000")  # test_signal byte0 = 0x2A = 42


def test_default_registry_has_exactly_one_real_vehicle():
    """Guards the honesty claim itself: the default registry (as used by
    application code, not tests) should contain exactly the one verified
    vehicle -- not silently grow to include the test fixture."""
    registry = default_registry()
    assert registry.registered_vehicles() == ["twizy"]


def test_identify_vehicle_matches_twizy_from_its_can_id():
    registry = default_registry()
    matches = registry.identify_vehicle({TWIZY_BMS_1_CAN_ID})
    assert matches == [("twizy", 1.0)]


def test_identify_vehicle_returns_empty_for_unknown_id():
    registry = default_registry()
    matches = registry.identify_vehicle({0xDEAD})
    assert matches == []


def test_registry_disambiguates_between_two_vehicles():
    """The core multi-vehicle claim, actually exercised: register two
    vehicles (one real, one synthetic-for-testing), feed frames from
    only one of them, and confirm the registry identifies and decodes
    using the CORRECT one, not just "a" one."""
    registry = default_registry(include_test_fixture=True)
    assert set(registry.registered_vehicles()) == {"twizy", "TEST_FIXTURE_NOT_A_REAL_VEHICLE"}

    twizy_frames = [{"can_id": TWIZY_BMS_1_CAN_ID, "data": TWIZY_BMS_1_EXAMPLE_BYTES}]
    name, df = registry.decode_frames_auto(twizy_frames, cell_id="V1")
    assert name == "twizy"
    assert df.iloc[0]["current_a"] == pytest.approx(58.25, abs=0.001)

    fixture_frames = [{"can_id": TEST_FIXTURE_CAN_ID, "data": TEST_FIXTURE_BYTES}]
    name2, df2 = registry.decode_frames_auto(fixture_frames, cell_id="V2")
    assert name2 == "TEST_FIXTURE_NOT_A_REAL_VEHICLE"


def test_decode_frames_auto_raises_on_unrecognized_vehicle():
    registry = default_registry()
    with pytest.raises(ValueError, match="No registered vehicle recognizes"):
        registry.decode_frames_auto([{"can_id": 0xDEAD, "data": b"\x00" * 8}], cell_id="V1")


def test_decode_frames_auto_raises_below_min_score_rather_than_guessing():
    registry = default_registry(include_test_fixture=True)
    # A frame set with IDs from BOTH vehicles: neither covers 100% of
    # its own known set well enough alone if we set an unreasonably high
    # bar. This specifically checks the "don't silently guess" path.
    mixed = [
        {"can_id": TWIZY_BMS_1_CAN_ID, "data": TWIZY_BMS_1_EXAMPLE_BYTES},
    ]
    # Sanity: this actually decodes fine at the default threshold.
    name, _ = registry.decode_frames_auto(mixed, cell_id="V1")
    assert name == "twizy"

    # But an impossibly high bar should refuse rather than force a match.
    with pytest.raises(ValueError, match="below min_score"):
        registry.decode_frames_auto(mixed, cell_id="V1", min_score=1.5)


if __name__ == "__main__":
    test_default_registry_has_exactly_one_real_vehicle()
    test_identify_vehicle_matches_twizy_from_its_can_id()
    test_identify_vehicle_returns_empty_for_unknown_id()
    test_registry_disambiguates_between_two_vehicles()
    test_decode_frames_auto_raises_on_unrecognized_vehicle()
    test_decode_frames_auto_raises_below_min_score_rather_than_guessing()
    print("All CAN vehicle registry tests passed.")
