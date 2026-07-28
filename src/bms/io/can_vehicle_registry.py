"""Multi-vehicle CAN/DBC registry.

Generalizes load_can_dbc.py's single-hardcoded-vehicle decoding into a
pluggable registry: multiple vehicles' DBC files can be registered, and
given a batch of raw CAN frames, the registry identifies which
registered vehicle they came from (by matching observed CAN IDs against
each vehicle's known message IDs) and decodes accordingly.

Read this before assuming "generalized" means "broad real-vehicle
coverage" -- it doesn't, and that's a researched, deliberate limitation,
not an oversight. See docs/can_dbc_adapter.md's "Why not all OVMS
vehicles" section for what was actually checked:

- Most of OVMS's 30+ supported vehicles use hand-written C++ decoders,
  not DBC files -- there's no portable spec to register for them.
- Community reverse-engineering for vehicles like the Nissan Leaf or Kia
  Soul EV exists, but as scattered, sometimes-contradictory forum posts
  ("my car doesn't always agree with this spreadsheet") -- not something
  to build verified support on top of.
- A real DBC for the Kia/Hyundai EV platform exists (CSS Electronics),
  but the file itself isn't published (gated behind a lead-gen
  download), it uses UDS multi-frame diagnostics (a materially more
  complex protocol than the plain broadcast decoding this module does),
  and there's no independently-published worked example to verify
  against the way the OVMS Twizy primer provided.

So what's registered by default is exactly one real, independently
verified vehicle (Twizy) plus one clearly-synthetic test fixture that
exists ONLY to prove the multi-vehicle registry logic itself works,
not to claim a second real vehicle is supported. Extending real coverage
means finding (or producing, with real hardware and OVMS's own RE
toolkit) another vehicle with the same quality bar Twizy had: a public,
independently-stated worked example to verify against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from src.bms.io.load_can_dbc import can_frames_to_unified_schema, load_dbc, TWIZY_BMS_1_SIGNAL_MAP, DEFAULT_DBC_PATH


@dataclass
class RegisteredVehicle:
    name: str
    dbc: "cantools.database.can.Database"  # noqa: F821 - forward ref, cantools imported in load_can_dbc
    signal_map: Mapping[str, str]
    source_url: str
    verified: bool
    notes: str = ""

    @property
    def known_can_ids(self) -> set[int]:
        return {m.frame_id for m in self.dbc.messages}


class VehicleDbcRegistry:
    """Holds N vehicles' DBC files and auto-identifies which one a batch
    of CAN frames came from.

    Identification heuristic: for each registered vehicle, score =
    (fraction of THAT VEHICLE'S OWN known message IDs that appear in the
    observed frames) -- not fraction of all observed traffic explained.
    That distinction matters: a real vehicle bus carries hundreds of
    message types from many ECUs, and a DBC that only defines one BMS
    message (like the bundled Twizy example) would score near-zero under
    a "how much of everything did we explain" metric even on a perfect
    match. Scoring against the vehicle's own known set avoids that.

    Known limitation, stated rather than hidden: this is a simple
    heuristic (ID-set overlap), not real vehicle fingerprinting. Two
    vehicles registered with overlapping ID ranges (a real possibility --
    CAN IDs aren't globally unique across manufacturers) could be
    ambiguous; this registry doesn't attempt byte-level plausibility
    checks to disambiguate. Fine for the one-real-vehicle case this
    ships with; would need hardening before trusting it on a large,
    noisy multi-vehicle fleet.
    """

    def __init__(self) -> None:
        self._vehicles: dict[str, RegisteredVehicle] = {}

    def register(
        self,
        name: str,
        dbc_path: str | Path,
        signal_map: Mapping[str, str],
        *,
        source_url: str,
        verified: bool,
        notes: str = "",
    ) -> None:
        db = load_dbc(dbc_path)
        self._vehicles[name] = RegisteredVehicle(
            name=name, dbc=db, signal_map=signal_map, source_url=source_url, verified=verified, notes=notes
        )

    def registered_vehicles(self) -> list[str]:
        return sorted(self._vehicles.keys())

    def identify_vehicle(self, observed_can_ids: Iterable[int]) -> list[tuple[str, float]]:
        """Returns [(vehicle_name, score)] for every registered vehicle
        with at least one matching CAN ID, sorted by score descending.
        Empty list means no registered vehicle recognizes any of the
        given IDs -- a real, informative result, not an error."""
        observed = set(observed_can_ids)
        results = []
        for name, v in self._vehicles.items():
            known = v.known_can_ids
            overlap = observed & known
            if overlap and known:
                score = len(overlap) / len(known)
                results.append((name, score))
        return sorted(results, key=lambda pair: -pair[1])

    def decode_frames_auto(
        self, frames: list[Mapping], *, cell_id: str, min_score: float = 0.5
    ) -> tuple[str, pd.DataFrame]:
        """Identify the vehicle from `frames`' CAN IDs, then decode using
        that vehicle's DBC and signal map.

        Raises ValueError -- doesn't silently guess -- if no vehicle
        matches, or if the best match's score is below `min_score`. An
        ambiguous identification should be a loud failure, the same
        stance this project takes on missing/unrecognized data elsewhere
        (see the NaN-as-healthy fix in health/health_index.py).
        """
        observed_ids = {f["can_id"] for f in frames}
        candidates = self.identify_vehicle(observed_ids)

        if not candidates:
            raise ValueError(
                f"No registered vehicle recognizes any of these CAN IDs: {sorted(observed_ids)}. "
                f"Registered vehicles: {self.registered_vehicles()}"
            )

        best_name, best_score = candidates[0]
        if best_score < min_score:
            raise ValueError(
                f"Best match ({best_name}, score={best_score:.2f}) is below min_score={min_score}. "
                f"All candidates: {candidates}. Refusing to guess."
            )

        vehicle = self._vehicles[best_name]
        df = can_frames_to_unified_schema(
            frames,
            db=vehicle.dbc,
            signal_map=vehicle.signal_map,
            cell_id=cell_id,
            dataset=f"obd_can_{best_name}",
        )
        return best_name, df


# --- Synthetic test-fixture DBC ---
# Exists ONLY to prove the multi-vehicle registry logic (identification,
# disambiguation, decode-by-best-match) actually works with N>1 vehicles
# registered. It is NOT a real vehicle and must never be presented as
# one -- `verified=False` and the name make that explicit everywhere
# this shows up (registry listing, dataset column, docs).
_TEST_FIXTURE_DBC_TEXT = """VERSION "Synthetic test fixture -- NOT a real vehicle"

NS_ :

BS_:

BU_: Vector__XXX

BO_ 999 TEST_MSG: 8 Vector__XXX
 SG_ test_signal : 7|8@0+ (1,0) [0|255] "unit" Vector__XXX
"""


def _write_test_fixture_dbc(dest: Path) -> Path:
    dest.write_text(_TEST_FIXTURE_DBC_TEXT)
    return dest


def default_registry(*, include_test_fixture: bool = False) -> VehicleDbcRegistry:
    """The registry this project actually ships: one real, verified
    vehicle. `include_test_fixture=True` additionally registers the
    synthetic fixture above, for tests that need >=2 vehicles to
    exercise disambiguation -- off by default so nothing importing this
    in application code accidentally treats the fixture as real."""
    registry = VehicleDbcRegistry()
    registry.register(
        "twizy",
        DEFAULT_DBC_PATH,
        TWIZY_BMS_1_SIGNAL_MAP,
        source_url="https://docs.openvehicles.com/en/latest/components/vehicle_dbc/docs/dbc-primer.html",
        verified=True,
        notes="Verified against OVMS's own documented worked example (see tests/test_can_dbc.py).",
    )

    if include_test_fixture:
        import tempfile

        fixture_path = Path(tempfile.gettempdir()) / "_beacon_test_fixture.dbc"
        _write_test_fixture_dbc(fixture_path)
        registry.register(
            "TEST_FIXTURE_NOT_A_REAL_VEHICLE",
            fixture_path,
            {"test_signal": "notes"},
            source_url="(synthetic, generated in-process for testing only)",
            verified=False,
            notes="Exists only to test multi-vehicle registry logic. Not a real vehicle.",
        )

    return registry
