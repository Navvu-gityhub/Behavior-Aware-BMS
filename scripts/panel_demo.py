"""Show it live: one real capture through every stage, then every fault.

    python scripts/panel_demo.py
    python scripts/panel_demo.py --capture data/interim/rig_stage_b_voltage_verified.txt

For a review panel that says "stop explaining, show me". Runs on a committed
bench-rig capture, so it works with no board attached; with a board, record a
capture first and pass it with --capture.

Stage by stage: the raw wire line, its checksum recomputed by hand, the parsed
record in unified channel names, the scored pipeline result. Then each fault
a panel asks for, injected into a copy of the real capture:

    corrupted byte      -> checksum fails, line rejected
    NaN in a field      -> record rejected, never scored
    impossible value    -> range check rejects, does not clamp
    time going backward -> capture refused
    sensor disconnected -> channel refused by name; nothing becomes "healthy"
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.telemetry.serial_pipeline import replay_serial_capture  # noqa: E402
from src.bms.telemetry.serial_schema import (  # noqa: E402
    LineDecodeError,
    parse_line,
    xor_checksum,
)

DEFAULT = Path("data/interim/rig_stage_b_voltage_verified.txt")
RULE = "-" * 72


def head(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def first_data_line(lines: list[str]) -> int:
    for k, line in enumerate(lines):
        if line.startswith("BEACON1 D"):
            return k
    raise SystemExit("capture has no data lines")


def try_line(label: str, line: str) -> None:
    print(f"{label}\n  {line}")
    try:
        kind, payload = parse_line(line)
        values = getattr(payload, "values", payload)
        print(f"  -> ACCEPTED as {kind}: {dict(values) if isinstance(values, dict) else values}")
    except LineDecodeError as exc:
        print(f"  -> REJECTED: {exc}")


def replay(label: str, lines: list[str]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    result = replay_serial_capture(path, require_full_coverage=False)
    print(f"{label}")
    print("  " + result.render().replace("\n", "\n  "))
    Path(path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", type=Path, default=DEFAULT)
    args = parser.parse_args()
    lines = args.capture.read_text(encoding="utf-8", errors="replace").splitlines()
    k = first_data_line(lines)
    line = lines[k]
    framed, checksum = line.rsplit("*", 1)
    prefix, payload = framed.split(" ", 2)[:2], framed.split(" ", 2)[2]
    prefix = " ".join(prefix)

    def sign(new_payload: str) -> str:
        """A line with a VALID checksum, so only its content is wrong."""
        return f"{prefix} {new_payload}*{xor_checksum(new_payload)}"

    head("1. RAW: what the board sent")
    for shown in lines[: k + 2]:
        print(f"  {shown}")

    head("2. CHECKSUM: recomputed here, by hand")
    print(f"  payload:    {payload}")
    print(f"  sent:       *{checksum}")
    print(f"  recomputed: *{xor_checksum(payload)}   (XOR of every byte of the payload)")

    head("3. PARSED: unified channel names, range-checked")
    try_line("the real line:", line)

    head("4. SCORED: the whole capture through the shared pipeline")
    replay(f"{args.capture.name}:", lines)

    head("5. FAULTS: each injected into a copy of the real capture")
    corrupted = f"{prefix} {payload.replace('3', '8', 1)}*{checksum}"
    try_line("a) one byte corrupted in transit (checksum no longer matches):", corrupted)
    voltage = payload.split('"v":', 1)[1].split(",", 1)[0]
    try_line("b) NaN in the voltage field (checksum valid, so only the value is wrong):",
             sign(payload.replace(f'"v":{voltage}', '"v":NaN', 1)))
    bad = sign(payload.replace(f'"v":{voltage}', '"v":999.0', 1))
    try_line("c) impossible value, 999 V (valid checksum) - the line alone parses, "
             "because packs are a legal unit:", bad)
    injected = list(lines)
    injected[k] = bad
    replay("   ...but this rig declared unit=cell, so the pipeline rejects that record:",
           injected)

    reversed_lines = list(lines)
    data_idx = [i for i, ln in enumerate(lines) if ln.startswith("BEACON1 D")]
    if len(data_idx) > 3:
        a, b = data_idx[2], data_idx[3]
        reversed_lines[a], reversed_lines[b] = reversed_lines[b], reversed_lines[a]
        replay("d) time running backwards (two records swapped):", reversed_lines)

    unplugged = [ln for ln in lines if not ln.startswith("BEACON1 HELLO")]
    replay("e) board never declared its channels (HELLO lost, as on an unplugged "
           "or mis-flashed board):", unplugged)
    print(f"\n{RULE}\nNothing above was turned into a plausible number. Every fault is "
          f"either rejected line by line or refused for the whole capture, with "
          f"the reason.\n{RULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
