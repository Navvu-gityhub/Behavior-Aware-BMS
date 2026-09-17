"""Flash the BEACON rig firmware onto a board whose USB-serial bridge is flaky.

`arduino-cli upload` assumes a well-behaved bridge: it opens the port at 115200,
sets a write timeout, and toggles DTR/RTS to drop the ESP8266 into its
bootloader. A counterfeit CH340 paired with WCH's driver 3.5+ can refuse all
three, and the upload then fails at "Cannot configure port" before any protocol
starts.

This script does the same job with each of those assumptions relaxed:

* **Descending baud rates.** A chip that refuses 115200 often accepts 57600,
  38400 or 9600. Each is tried in turn, fastest first.
* **A rejected write timeout is survivable.** esptool sets `write_timeout`
  unconditionally and dies if the driver refuses it. Here the setter is wrapped
  so a refusal is noted and the flash proceeds - the timeout only bounds how
  long a stuck write blocks; it is not needed to write flash.
* **Optional manual bootloader.** `--manual-bootloader` uses `--before
  no_reset`, so a board whose DTR/RTS lines are refused can still be flashed
  after you enter the bootloader by hand.
* **The port is touched as late as possible.** Compilation happens first,
  because these bridges commonly permit exactly one open per enumeration and
  spending it on anything but the flash wastes it. For the same reason probing
  is opt-in, not automatic.

    python scripts/flash_rig.py --port COM5
    python scripts/flash_rig.py --port COM5 --manual-bootloader
    python scripts/flash_rig.py --port COM5 --probe-only

This is a workaround, not a repair. The fault is the bridge chip or its driver;
see docs/hardware_integration.md. If `--probe-only` reports DTR/RTS OK and an
open at 115200, this script is unnecessary - use `arduino-cli upload`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKETCH = REPO / "firmware" / "beacon_rig"
FQBN = "esp8266:esp8266:nodemcuv2"

#: Fastest first, so a healthy board takes the quick path.
BAUD_LADDER = (115200, 57600, 38400, 19200, 9600)


def probe(port: str) -> dict[str, object]:
    """Report what this port supports.

    Consumes at least one open, which on a one-open-per-enumeration bridge is
    the one the flash needed. Never called before a flash attempt.
    """
    import serial

    findings: dict[str, object] = {
        "opens": [], "dtr_rts": False, "write_timeout": False,
    }

    for baud in BAUD_LADDER:
        try:
            handle = serial.Serial(port, baud, timeout=0.3)
        except Exception:
            continue
        findings["opens"].append(baud)  # type: ignore[union-attr]

        if not findings["write_timeout"]:
            try:
                handle.write_timeout = 10
                findings["write_timeout"] = True
            except Exception:
                pass
        if not findings["dtr_rts"]:
            try:
                handle.dtr = False
                handle.rts = True
                handle.dtr = True
                handle.rts = False
                findings["dtr_rts"] = True
            except Exception:
                pass
        try:
            handle.close()
        except Exception:
            pass
        time.sleep(0.25)

    return findings


def _report(findings: dict[str, object]) -> None:
    opens = findings["opens"]
    print(f"  opens at:      {opens or 'NOTHING - replug the board'}")
    print(f"  DTR/RTS:       {'OK' if findings['dtr_rts'] else 'REFUSED'}")
    print(f"  write_timeout: {'OK' if findings['write_timeout'] else 'REFUSED'}")


def _patch_pyserial() -> None:
    """Make rejected timeout changes non-fatal.

    pyserial reconfigures the whole port on every `timeout` or `write_timeout`
    assignment, and a driver that refuses `SetCommState` on an open handle
    raises. esptool changes both repeatedly during its bootloader sync, so an
    otherwise usable link dies mid-connect.

    Both are conveniences: they bound how long a read or write blocks. Losing
    them costs an unresponsive board a longer wait, and buys the ability to
    flash at all. The values are still applied when the driver accepts them.
    """
    from serial import serialutil

    for attribute in ("write_timeout", "timeout"):
        original = getattr(serialutil.SerialBase, attribute)

        def tolerant(self, value, _original=original):  # type: ignore[no-untyped-def]
            try:
                _original.fset(self, value)
            except Exception:
                pass

        setattr(
            serialutil.SerialBase,
            attribute,
            property(original.fget, tolerant),
        )


def _patch_esptool_open(esptool, baud: int) -> None:
    """Open the port AT the target baud instead of changing it afterwards.

    This is the fix, and it is worth stating precisely because it is not
    obvious. esptool opens the port and *then* sets the rate:

        self._port = serial.serial_for_url(port)   # opens at pyserial default
        self._set_port_baudrate(baud)              # reconfigures an OPEN handle

    The second step is a `SetCommState` on an already-open handle, and some
    CH340 drivers refuse exactly that while accepting the identical rate
    supplied at open time - which is why `serial.Serial(port, 115200)` succeeds
    on the same port, in the same second, that esptool fails on.

    esptool's own comment says the split exists as a workaround for a CH341
    driver on Linux. On this Windows driver the workaround is the bug, so the
    two steps are collapsed back into one.
    """
    import serial

    original_open = serial.serial_for_url

    def open_at_target(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("baudrate", baud)
        return original_open(url, *args, **kwargs)

    serial.serial_for_url = open_at_target
    esptool.serial.serial_for_url = open_at_target

    original_set = esptool.ESPLoader._set_port_baudrate

    def set_baudrate(self, requested):  # type: ignore[no-untyped-def]
        # Already there, thanks to the patch above. pyserial reconfigures on
        # every assignment even when the value is unchanged, so skipping the
        # no-op avoids re-triggering the call that fails.
        if getattr(self._port, "baudrate", None) == requested:
            return
        original_set(self, requested)

    esptool.ESPLoader._set_port_baudrate = set_baudrate


def _binary() -> Path:
    """Compile and return the built image, so a flash is never stale."""
    print("compiling ...")
    build = REPO / "build" / "flash_rig"
    build.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["arduino-cli", "compile", "--fqbn", FQBN,
         "--output-dir", str(build), str(SKETCH)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-1200:])
        print(result.stderr[-1200:])
        raise SystemExit("compile failed - fix that before flashing")

    binaries = sorted(build.glob("*.ino.bin"))
    if not binaries:
        raise SystemExit(f"no .ino.bin produced in {build}")
    print(f"  built {binaries[0].name} ({binaries[0].stat().st_size:,} bytes)")
    return binaries[0]


def _esptool_path() -> Path:
    root = Path.home() / "AppData" / "Local" / "Arduino15" / "packages" / "esp8266"
    found = sorted(root.rglob("esptool/esptool.py"))
    if not found:
        raise SystemExit(
            "esptool.py not found under the esp8266 core. Install it with:\n"
            "  arduino-cli core install esp8266:esp8266 --additional-urls "
            "https://arduino.esp8266.com/stable/package_esp8266com_index.json"
        )
    return found[-1]


def flash(port: str, binary: Path, esptool_py: Path, before: str,
          rounds: int = 1) -> bool:
    """Try the baud ladder, optionally several times over.

    Repetition is not superstition here: this bridge has been observed to
    refuse every rate in one pass and then accept 115200 moments later, so
    the failure is a window rather than a capability limit. Retrying is the
    correct response to an intermittent link, and costs only time.
    """
    _patch_pyserial()
    sys.path.insert(0, str(esptool_py.parent))
    import esptool  # type: ignore[import-not-found]

    ladder = [(r, b) for r in range(1, rounds + 1) for b in BAUD_LADDER]
    for round_index, baud in ladder:
        print(f"\n--- round {round_index}/{rounds}: {baud} baud ({before}) ---")
        argv = [
            "--chip", "esp8266", "--port", port, "--baud", str(baud),
            "--before", before, "--after", "hard_reset",
            "write_flash", "0x0", str(binary),
        ]
        _patch_esptool_open(esptool, baud)
        try:
            esptool.main(argv)
            print(f"\nFLASHED at {baud} baud.")
            return True
        except SystemExit as exc:
            if exc.code in (0, None):
                print(f"\nFLASHED at {baud} baud.")
                return True
            print(f"  failed at {baud} (exit {exc.code})")
        except Exception as exc:
            print(f"  failed at {baud}: {type(exc).__name__}: {str(exc)[:110]}")
        time.sleep(0.6)

    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument(
        "--probe-only", action="store_true",
        help="Report what the port supports and exit without flashing.",
    )
    parser.add_argument(
        "--retries", type=int, default=1, metavar="N",
        help="Passes over the baud ladder. This bridge is intermittent, "
             "so several passes materially improve the odds. Try 10.",
    )
    parser.add_argument(
        "--manual-bootloader", action="store_true",
        help="Skip auto-reset. Use when DTR/RTS is refused: hold FLASH, tap "
             "RST, release FLASH, then run this.",
    )
    args = parser.parse_args(argv)

    if args.probe_only:
        print(f"probing {args.port} ...")
        _report(probe(args.port))
        print(
            "\nNOTE: probing consumes an open. On a bridge that allows one open "
            "per enumeration, replug the board before flashing."
        )
        return 0

    # Compile before touching the port: the first open after a replug is often
    # the only one that works, and it must be spent on the flash.
    binary = _binary()
    esptool_py = _esptool_path()
    print(f"using {esptool_py.name}")

    if args.manual_bootloader:
        print(
            "\nPut the board into its bootloader now:\n"
            "    1. hold FLASH down\n"
            "    2. tap RST once\n"
            "    3. release FLASH\n"
            "Then press Enter."
        )
        try:
            input()
        except EOFError:
            print("  (no console attached; continuing)")

    before = "no_reset" if args.manual_bootloader else "default_reset"

    if flash(args.port, binary, esptool_py, before, rounds=args.retries):
        print(
            "\nTap RST to leave the bootloader, then run:\n"
            f"  python scripts/run_serial_demo.py --port {args.port} --duration 120"
        )
        return 0

    print("\nEvery baud rate failed. Diagnosing what the port can do ...")
    findings = probe(args.port)
    _report(findings)

    if not findings["opens"]:
        print(
            "\nThe port would not open at all. Unplug the board, wait two "
            "seconds, plug it back in, and re-run this IMMEDIATELY - these "
            "bridges commonly allow one open per enumeration."
        )
    elif not findings["dtr_rts"] and not args.manual_bootloader:
        print(
            "\nDTR/RTS is refused, so auto-reset into the bootloader cannot "
            "work. Replug the board, then re-run with --manual-bootloader."
        )
    else:
        print(
            "\nThe bridge chip cannot complete a flash, and no software change "
            "will alter that. See docs/hardware_integration.md for the driver "
            "rollback, the external USB-TTL adapter, and the replacement-board "
            "options."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
