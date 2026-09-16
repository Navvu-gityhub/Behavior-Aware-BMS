"""Generate the committed BEACON serial capture fixture.

This is a FIXTURE, not a physical measurement. It is produced by the
deterministic emulator, so every number derived from it is a property of the
emulator and nothing here may be reported as a hardware result. The file exists
so that replay determinism is tested against bytes that are checked in and do
not change between runs, rather than against a source regenerated inside the
test — a fixture regenerated on the fly proves the generator is deterministic,
not that the parser is.

Two properties are deliberate:

* **The rig declares 3.4 Ah, not 2.0 Ah.** The fixture is therefore also a
  regression test against reintroducing the silent 2.0 Ah C-rate default
  (ADR 0014): if that default ever came back, the flags computed from this
  capture would change and `test_hardware_readiness.py` would fail.
* **The boot banner is included.** Every real board prints one, so a fixture
  without noise would not exercise the sentinel filter.

Regenerate with:

    python tests/fixtures/make_serial_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bms.telemetry import EmulatedRigSource, RigProfile  # noqa: E402

#: Kept small on purpose. Two cycles is the minimum that lets `measure_cycles`
#: judge completeness against a peer rather than trivially, and a 300 s period
#: keeps the committed file to a few hundred lines.
FIXTURE_PROFILE = RigProfile(
    cell_id="FIXTURE_RIG",
    n_cycles=2,
    sample_period_s=300.0,
    capacity_ah=3.4,
)

CAPTURE = Path(__file__).with_name("beacon_rig_capture.txt")


def main() -> int:
    source = EmulatedRigSource(profile=FIXTURE_PROFILE)
    written = source.write_capture(CAPTURE)
    lines = CAPTURE.read_text(encoding="utf-8").splitlines()
    data = sum(1 for line in lines if line.startswith("BEACON1 D"))
    print(f"Wrote {written}")
    print(f"  {len(lines)} lines, {data} data records, {CAPTURE.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
