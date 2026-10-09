# Ten-minute demo

Rehearsed on 2026-10-09: every command below ran on the development laptop.
On a machine where Python is not on PATH, replace `python` with its full path,
e.g. `C:\Users\Navee\anaconda3\python.exe`.

## Before (once, ~15 minutes)

1. Open PowerShell in the repository folder.
2. Build the battery memories (needs the downloaded NASA, CALCE and LG M50T data):
   ```powershell
   python scripts\demo_battery_memory.py
   ```
3. Start the service and leave the window open:
   ```powershell
   $env:BEACON_PROFILE_DIR="data\interim\profiles"
   python -m uvicorn src.bms.api.app:app --port 8000
   ```
4. Open a second PowerShell window in the same folder.
5. Open `http://127.0.0.1:8000/memory` and `http://127.0.0.1:8000/docs` in the browser.

## 1. It measures health from raw signals (2 min)

```powershell
python scripts\health_report.py --calce data\raw\calce\CS2\Type2\CS2_35.zip --upto-cycle 90
```

Point out:

- **91.0% capacity health, +/-3.4 points.** The +/- is measured error from lab cells, not a guess.
- **Resistance 150 -> 154 mOhm,** from the voltage drop when the load switches on.
- **Section 0 lists what the result rests on,** including what is missing ("no temperature channel").

## 2. Each battery has its own memory (4 min)

In the browser, at `/memory`:

- **CALCE_CS2_35:** tracked within about 1% of its measured capacity over 783 discharges, down to 53%. The red note explains why the 136 later discharges are refused.
- **NASA_B0047_4C:** LOW confidence, with the reason (cold, where 11-14 point errors were measured). BEACON knows where it is weak.
- **M50T_25C_D:** a cell type held out of all development, fed one ageing set at a time. It reads 88.4%, within the validated error.
- **The temperature chart:** the measured cell temperature for each reading.

## 3. It refuses rather than guesses (2 min)

```powershell
python scripts\panel_demo.py
```

A real rig capture runs through every stage. Then a corrupted byte, an
impossible value, time going backwards and a missing sensor are each injected.
Each is rejected or refused, with the reason.

## 4. It is a working service (1 min)

At `/docs`, show the endpoints. `POST /profiles/{id}/ingest` feeds a real CSV
log into a battery's memory.

## Close (30 s)

Every claim is in `docs/validation_matrix.md`, including the failures: cold
accuracy, confidence that cannot see a steady error, and early remaining life.
Each has a next test written in the paper's Future work section.

## Questions to expect

- **"NASA_B0005 is 16 points off and only MEDIUM?"** Yes: a known gap. It is neither cold nor temperature-shifted, so the warning does not catch it (validation matrix row 33).
- **"Is this live hardware?"** Part 3 replays real captures from the rig. The rig has not yet recorded a full discharge; that is the next experiment.
