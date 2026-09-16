"""Run the serial telemetry path end to end - emulated, live, or replayed.

The demo entry point for the hardware integration. With no arguments it runs the
complete rig-to-Guardian path against the deterministic emulator, so the whole
route is demonstrable with nothing plugged in.

    python scripts/run_serial_demo.py                    # emulated rig
    python scripts/run_serial_demo.py --ports            # list serial ports
    python scripts/run_serial_demo.py --port COM5        # a real board
    python scripts/run_serial_demo.py --capture out.txt  # record while scoring
    python scripts/run_serial_demo.py --replay out.txt   # replay a recording

`--capture` records whatever source the run is using - a live port, or the
emulator - so a physical bring-up produces a file that replays to the same
numbers. That is what makes a hardware result checkable by someone who was not
in the room.

`--port` never defaults and is never inferred: on a laptop with a Bluetooth
serial device or two boards attached, auto-selecting would silently read the
wrong one. `--ports` lists candidates for a human to choose from.

The exit code is 0 when the capture was scored and 1 when it was refused, so
this is usable as a bench check. A refusal is a correct outcome, not a crash -
the reasons are printed either way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.telemetry import (  # noqa: E402
    DEFAULT_BAUDRATE,
    DEFAULT_MIN_ACCEPTED_FRACTION,
    EmulatedRigSource,
    LineSource,
    RecordingLineSource,
    RigProfile,
    SerialPortSource,
    available_ports,
    replay_serial_capture,
    run_serial_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--port",
        help="Serial port of a live rig, e.g. COM5 or /dev/ttyUSB0. "
             "Never inferred; use --ports to list candidates.",
    )
    mode.add_argument(
        "--replay", type=Path, help="Replay a previously recorded capture file."
    )
    mode.add_argument(
        "--ports", action="store_true",
        help="List available serial ports and exit.",
    )

    parser.add_argument(
        "--baudrate", type=int, default=DEFAULT_BAUDRATE,
        help=f"Serial baud rate (default {DEFAULT_BAUDRATE}).",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Seconds to capture from a live port (default 60).",
    )
    parser.add_argument(
        "--cell-id", default=None,
        help="Override the identity the rig declares.",
    )
    parser.add_argument(
        "--cycles", type=int, default=3,
        help="Emulated rig: charge/discharge cycles to generate (default 3).",
    )
    parser.add_argument(
        "--period", type=float, default=60.0,
        help="Emulated rig: sample period in seconds (default 60).",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Emulated rig: use the compact key=value codec.",
    )
    parser.add_argument(
        "--corrupt-every", type=int, default=0,
        help="Emulated rig: truncate every Nth line, to exercise the "
             "rejection counters and the accepted-fraction refusal.",
    )
    parser.add_argument(
        "--capture", type=Path, default=None,
        help="Record every line this run receives to a file, then score it as "
             "normal. Works for a live port, a replay and the emulator alike, "
             "so a hardware bring-up can be committed as a replayable fixture.",
    )
    parser.add_argument(
        "--capacity-ah", type=float, default=None,
        help="Rated capacity of the cell under test, in amp-hours. Overrides "
             "whatever the rig declares in its HELLO line, and is required for "
             "firmware that declares nothing: C-rate is current divided by "
             "this, and the run refuses rather than assuming a cell size.",
    )
    parser.add_argument(
        "--min-accepted", type=float, default=DEFAULT_MIN_ACCEPTED_FRACTION,
        help="Fraction of data records that must parse before the capture is "
             f"scored (default {DEFAULT_MIN_ACCEPTED_FRACTION}).",
    )
    parser.add_argument(
        "--show-telemetry", action="store_true",
        help="Print the head of the decoded unified-schema frame.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.ports:
        ports = available_ports()
        if not ports:
            print(
                "No serial ports found.\n"
                "If a board is attached, check its driver, and confirm "
                "pyserial is installed (pip install pyserial) - this command "
                "reports no ports when the dependency is absent too."
            )
            return 1
        print("Available serial ports:")
        for device, description in ports:
            print(f"  {device:<20} {description}")
        print("\nPass one with --port. It is never chosen automatically.")
        return 0

    # `--replay` reaches the same pipeline through `replay_serial_capture`,
    # which builds its own source, so recording it would mean copying a file to
    # itself. Every other mode goes through one `run_serial_pipeline` call with
    # an optionally-recorded source.
    if args.replay is not None:
        if args.capture is not None:
            print(
                "--capture has no effect with --replay: the capture already "
                "exists. Copy the file instead.\n"
            )
        result = replay_serial_capture(
            args.replay,
            cell_id=args.cell_id,
            min_accepted_fraction=args.min_accepted,
            rated_capacity_ah=args.capacity_ah,
        )
    else:
        source: LineSource
        if args.port:
            source = SerialPortSource(
                name=f"serial:{args.port}",
                port=args.port,
                baudrate=args.baudrate,
                duration_s=args.duration,
            )
        else:
            print(
                "No --port given: running the deterministic emulated rig.\n"
                "This exercises the identical parser, coverage gate and scoring "
                "stages a physical board would.\n"
            )
            source = _emulated_source(args)

        if args.capture is not None:
            # Recording wraps the source rather than replacing the run, so what
            # lands on disk is exactly the bytes this run scored. Writing the
            # capture in a separate pass - as this script used to - meant the
            # file and the result came from two different sessions, and on a
            # live port it meant the file came from the emulator entirely.
            source = RecordingLineSource(inner=source, path=args.capture)

        result = run_serial_pipeline(
            source,
            cell_id=args.cell_id,
            min_accepted_fraction=args.min_accepted,
            rated_capacity_ah=args.capacity_ah,
        )

    print(result.render())

    if args.capture is not None and args.replay is None:
        print(f"\nRecorded to {args.capture}")
        print(
            f"Replay it with:  python {Path(__file__).name} "
            f"--replay {args.capture}"
        )

    if args.show_telemetry and not result.telemetry.empty:
        print("\nDecoded telemetry (head):")
        print(result.telemetry.head(10).to_string(index=False))

    if not result.cycles.empty:
        print("\nCycle measurements:")
        columns = [
            column for column in
            ("cycle", "capacity_ah", "is_complete", "avg_temp", "depth_of_discharge")
            if column in result.cycles.columns
        ]
        print(result.cycles[columns].to_string(index=False))

    if not result.guardian.empty:
        print("\nGuardian report:")
        columns = [
            column for column in
            ("battery_id", "battery_state", "risk_level", "risk_score",
             "health_index", "rul_cycles")
            if column in result.guardian.columns
        ]
        print(result.guardian[columns].to_string(index=False))
        print(f"\n{result.guardian.iloc[0]['guardian_caveat']}")

    # A refusal is a correct outcome, so it is reported rather than raised - but
    # it is not a success, so the exit code distinguishes it for a bench check.
    return 0 if result.scored else 1


def _emulated_source(args: argparse.Namespace) -> EmulatedRigSource:
    return EmulatedRigSource(
        profile=RigProfile(
            cell_id=args.cell_id or "RIG_01",
            n_cycles=args.cycles,
            sample_period_s=args.period,
        ),
        compact=args.compact,
        corrupt_every=args.corrupt_every,
    )


if __name__ == "__main__":
    raise SystemExit(main())
