"""One bench session on the rig: record, watch, then let BEACON judge it.

    python scripts/bench_session.py --port COM5 --hours 6
    python scripts/bench_session.py --report data/interim/bench/<capture>.txt --reference-mah 2150
    python scripts/bench_session.py --emulate          # dry run, no board

WHAT IT DOES
------------
1. Records every line the board sends to data/interim/bench/<time>_<cell>.txt.
   It writes as it goes, so a yanked cable keeps everything received.
2. Every 10 s it prints voltage, current, temperature, the phase (charge /
   discharge / rest) and the charge counted in that phase. It ALARMS:
   - while discharging, if the voltage falls below --min-voltage
     (disconnect the load);
   - at any time, if the temperature passes --max-temp (stop and disconnect).
   The firmware cannot switch the load itself, so these are warnings to you.
3. On Ctrl+C, or at --hours, it replays the capture through BEACON's own
   pipeline and writes <capture>_report.md beside it. The report holds:
   - the health card;
   - every discharge and charge with its measured charge (mAh);
   - the acceptance checklist from docs/hardware_integration.md, judged on
     this capture.

ACCEPTANCE (docs/hardware_integration.md, "Physical acceptance checklist")
---------------------------------------------------------------------------
- current is negative under load (a discharge phase exists);
- the largest complete discharge is within 5% of an independent mAh
  reference (a USB capacity tester or a hobby charger that measures);
- the capture replays to the same numbers.

The checklist states which channels the board declared in its HELLO. A board
with stub sensors produces well-formed records that score (see "The trap" in
that document), so a result is a hardware result only for the channels the
board really measured.

SAFETY: use a protected 18650, a load that keeps current at or below 1C, a
fire-safe surface, and stay near the cell during a discharge.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.guardian.health_card import render_health_card  # noqa: E402
from src.bms.telemetry.cycles import segment_phases  # noqa: E402
from src.bms.telemetry.serial_pipeline import replay_serial_capture  # noqa: E402
from src.bms.telemetry.serial_schema import LineDecodeError, parse_line  # noqa: E402
from src.bms.telemetry.serial_source import (  # noqa: E402
    EmulatedRigSource,
    RecordingLineSource,
    RigProfile,
    SerialPortSource,
)

OUT = Path("data/interim/bench")
REST_A = 0.02
TOLERANCE = 0.05


def fmt(x, pattern: str) -> str:
    return pattern.format(x) if x is not None else "--"


class Watch:
    """Running phase and charge count, with the two safety alarms."""

    def __init__(self, min_voltage: float, max_temp: float):
        self.min_voltage, self.max_temp = min_voltage, max_temp
        self.phase, self.phase_mah, self.last_t = "rest", 0.0, None
        self.last_print = 0.0
        self.hello_fields: tuple[str, ...] = ()

    def feed(self, line: str) -> None:
        try:
            kind, payload = parse_line(line)
        except LineDecodeError:
            return
        if kind == "HELLO":
            self.hello_fields = tuple(getattr(payload, "fields", ()))
            print(f"board declares: {', '.join(self.hello_fields)}", flush=True)
            return
        if kind == "S":
            print(f"board status: {payload}", flush=True)
            return
        if kind != "D":
            return
        values = payload.values
        t, i = values.get("test_time_s"), values.get("current_a")
        v, tc = values.get("voltage_v"), values.get("temperature_c")
        phase = "rest" if i is None or abs(i) < REST_A else ("discharge" if i < 0 else "charge")
        if phase != self.phase:
            if self.phase != "rest":
                print(f"--- {self.phase} ended: {self.phase_mah:.0f} mAh", flush=True)
            self.phase, self.phase_mah = phase, 0.0
        if self.last_t is not None and t is not None and i is not None:
            self.phase_mah += abs(i) * max(t - self.last_t, 0.0) / 3.6
        self.last_t = t
        if v is not None and phase == "discharge" and v < self.min_voltage:
            print(f"!!! ALARM: {v:.3f} V under load, below {self.min_voltage} V - DISCONNECT THE LOAD", flush=True)
        if tc is not None and tc > self.max_temp:
            print(f"!!! ALARM: cell at {tc:.1f} C, above {self.max_temp} C - STOP AND DISCONNECT", flush=True)
        now = time.monotonic()
        if now - self.last_print >= 10:
            self.last_print = now
            print(f"t={fmt(t, '{:8.0f}')}s  V={fmt(v, '{:.3f}')}  I={fmt(i, '{:+.3f}')}A  "
                  f"T={fmt(tc, '{:.1f}')}C  {phase:9s} {self.phase_mah:7.0f} mAh", flush=True)


def phase_table(tel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in segment_phases(tel, rest_threshold_a=REST_A):
        if p.kind == "rest":
            continue
        rows.append({"phase": p.kind, "start_s": round(p.start_time_s), "duration_min": round(
            (p.end_time_s - p.start_time_s) / 60, 1), "mean_current_a": round(p.mean_current_a, 3),
            "charge_mah": round(abs(p.charge_moved_ah) * 1000, 1)})
    return pd.DataFrame(rows)


def report(capture: Path, rated_ah: float, reference_mah: float | None) -> Path:
    result = replay_serial_capture(capture, rated_capacity_ah=rated_ah, require_full_coverage=False)
    tel = result.telemetry
    phases = phase_table(tel) if not tel.empty else pd.DataFrame()
    dis = phases[phases["phase"] == "discharge"] if len(phases) else phases
    measured = float(dis["charge_mah"].max()) if len(dis) else float("nan")
    channels = [c for c in ("voltage_v", "current_a", "temperature_c", "soc")
                if c in tel.columns and pd.to_numeric(tel[c], errors="coerce").notna().any()]
    checks = [("Current negative under load (a discharge phase exists)", len(dis) > 0,
               f"{len(dis)} discharge phase(s)")]
    if reference_mah:
        ok = np.isfinite(measured) and abs(measured - reference_mah) / reference_mah <= TOLERANCE
        checks.append((f"Largest discharge within {TOLERANCE:.0%} of the reference", bool(ok),
                       f"{measured:.0f} mAh measured vs {reference_mah:.0f} mAh reference"
                       + (f" ({(measured - reference_mah) / reference_mah:+.1%})" if np.isfinite(measured) else "")))
    else:
        checks.append(("Largest discharge vs an independent reference", None,
                       "no --reference-mah given: rerun with --report and your tester's reading"))
    again = replay_serial_capture(capture, rated_capacity_ah=rated_ah, require_full_coverage=False)
    same = (again.telemetry.shape == tel.shape) and again.telemetry.equals(tel)
    checks.append(("Capture replays to the same numbers", same, "replayed twice, frames compared"))
    lines = [f"# Bench session: {capture.name}", "",
             f"Replayed through BEACON's pipeline on {datetime.now():%Y-%m-%d %H:%M}. "
             f"Rated capacity {rated_ah} Ah.", "",
             f"**Channels with real data in this capture:** {', '.join(channels) or 'none'}. "
             "A result is a hardware result only for these.", "",
             "## Acceptance checklist", "",
             *[f"- {'PASS' if ok else ('-' if ok is None else 'FAIL')} - {name}: {detail}" for name, ok, detail in checks],
             "", "## Phases", "", "```", phases.to_string(index=False) if len(phases) else "(no phases)", "```",
             "", "## BEACON's health card", "", "```", render_health_card(result, capture.stem), "```"]
    out = capture.with_name(capture.stem + "_report.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--port", help="serial port, e.g. COM5")
    mode.add_argument("--report", type=Path, help="write the report for an existing capture")
    mode.add_argument("--emulate", action="store_true", help="dry run with the emulated rig")
    ap.add_argument("--baud", type=int, default=9600, help="9600 for this project's NodeMCU (CH340)")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--capacity-ah", type=float, default=2.2)
    ap.add_argument("--reference-mah", type=float, default=None)
    ap.add_argument("--cell-id", default="RIG_01")
    ap.add_argument("--min-voltage", type=float, default=3.0)
    ap.add_argument("--max-temp", type=float, default=45.0)
    a = ap.parse_args()

    if a.report:
        print(report(a.report, a.capacity_ah, a.reference_mah).read_text(encoding="utf-8"))
        return 0
    capture = OUT / f"{datetime.now():%Y%m%d_%H%M%S}_{a.cell_id}.txt"
    if a.emulate:
        source = EmulatedRigSource(profile=RigProfile(cell_id=a.cell_id, n_cycles=6, sample_period_s=30.0))
    else:
        source = SerialPortSource(name=f"bench:{a.port}", port=a.port, baudrate=a.baud,
                                  duration_s=a.hours * 3600)
    watch = Watch(a.min_voltage, a.max_temp)
    print(f"recording to {capture}  (Ctrl+C to stop)", flush=True)
    try:
        for line in RecordingLineSource(inner=source, path=capture).lines():
            watch.feed(line)
    except KeyboardInterrupt:
        print("\nstopped by you", flush=True)
    if not capture.exists():
        print("nothing was received; check the port, the baud rate and the cable")
        return 1
    print(f"\ncapture saved: {capture}\n")
    print(report(capture, a.capacity_ah, a.reference_mah).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
