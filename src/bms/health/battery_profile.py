"""A battery's own memory: what it has learned about itself, one discharge at a time.

WHY THIS EXISTS
---------------
`current_field_soh` replays a whole telemetry log to answer one question. A
BMS cannot do that: it does not keep years of 1 Hz samples, and the answer
should not depend on whether it still does. What the estimate actually needs
from the past is small - for each discharge, the charge drawn inside the
voltage window, the total charge, the current, and the step resistance - and
the reference this cell formed when it was new.

`BatteryProfile` keeps exactly that, per battery, and updates it as each new
piece of telemetry arrives. Its answers are the shipped estimator's answers:
it rebuilds the same per-discharge table from its memory
(`voltage_window.soh_table_from_charges`, `field_soh.apply_field_gates`) and
reports through the same function (`field_soh.field_soh_from_table`), so every
gate and refusal is shared, and a test holds it equal to the whole-log path.

It also records the conditions of each reading - mean current and, when the
telemetry carries it, temperature - which the estimate does not use today.
They are kept because the cross-dataset work found accuracy depends on them
(reports/metrics/cross_dataset/), and a profile that forgot them could never
learn from them.

WHAT IT REQUIRES
----------------
Each `update` must contain whole discharges: split telemetry at rest, not in
the middle of a discharge. A discharge cut at a chunk boundary is two
truncated discharges, and the truncation gate will treat them as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.bms.health.field_soh import (
    DEFAULT_REFERENCE_CYCLES,
    RESISTANCE_WINDOW_CYCLES,
    FieldSOH,
    apply_field_gates,
    curves_from_telemetry,
    field_soh_from_table,
    screen_step_resistances,
)
from src.bms.health.voltage_window import (
    WindowSpec,
    soh_table_from_charges,
    window_charge,
)
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles

PROFILE_VERSION = 1
_ROW_FIELDS = ("cycle", "capacity_ah", "r_step_ohm", "mean_current_a", "r_trailing_ohm",
               "ir_drop_v", "window_charge_ah", "cycle_charge_ah", "temperature_c")


def fingerprint(telemetry: pd.DataFrame) -> str:
    """A content key for one piece of telemetry: its size, span and sums."""
    import hashlib

    parts = [str(len(telemetry))]
    for col in ("test_time_s", "current_a", "voltage_v"):
        if col in telemetry.columns and len(telemetry):
            v = pd.to_numeric(telemetry[col], errors="coerce").to_numpy(float)
            parts += [f"{np.nanmin(v):.6g}", f"{np.nanmax(v):.6g}", f"{np.nansum(v):.9g}"]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


@dataclass
class BatteryProfile:
    battery_id: str
    rest_threshold_a: float = 0.02
    v_high: float = WindowSpec().v_high
    v_low: float = WindowSpec().v_low
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES
    rows: list[dict] = field(default_factory=list)
    # Fingerprints of telemetry already absorbed: replaying the same log twice
    # must not count its discharges twice.
    sources: list[str] = field(default_factory=list)

    @property
    def spec(self) -> WindowSpec:
        return WindowSpec(self.v_high, self.v_low)

    @property
    def n_discharges(self) -> int:
        return len(self.rows)

    # -- learning ---------------------------------------------------------
    def update(self, telemetry: pd.DataFrame) -> int:
        """Absorb the discharges in a new piece of telemetry; return how many.

        Telemetry this profile has already absorbed is recognised and skipped.
        """
        key = fingerprint(telemetry)
        if key in self.sources:
            return 0
        self.sources.append(key)
        d = cycles_to_frame(measure_cycles(telemetry, self.battery_id,
                                           rest_threshold_a=self.rest_threshold_a),
                            complete_only=False)
        if d.empty:
            return 0
        curves, steps = curves_from_telemetry(telemetry, d, self.battery_id, self.rest_threshold_a)
        offset = self.n_discharges
        known = {int(c) for c in steps["cycle"]} if len(steps) else set()
        temps = self._temperatures(telemetry, d)

        # Discharges without a usable curve still count toward the delivered
        # charge the reference gate needs, so they are remembered too.
        #
        # Indexed once per update rather than searched per discharge, and the
        # step screen run once over the whole history: it is causal (each step
        # is judged only against steps before it), so one pass gives every
        # position the value a pass over its own prefix would. A full-life
        # CALCE log went from 14 s to well under 2 s with identical answers.
        ordered = d.sort_values("cycle")
        step_of = {int(r.cycle): r for r in steps.itertuples()} if len(steps) else {}
        blocks = {int(c): (g["voltage_v"].to_numpy(float), g["capacity_ah_curve"].to_numpy(float))
                  for c, g in curves.groupby("cycle")} if len(curves) else {}
        r_history = [r["r_step_ohm"] for r in self.rows if r.get("has_step_row", True)]
        first_new = len(r_history)
        r_history += [float(step_of[int(c)].r_step_ohm) for c in ordered["cycle"] if int(c) in known]
        screened = screen_step_resistances(r_history)
        position = first_new
        new_rows = []
        for row in ordered.itertuples():
            local = int(row.cycle)
            entry = {k: float("nan") for k in _ROW_FIELDS}
            entry.update({"cycle": offset + len(new_rows) + 1, "capacity_ah": float(row.capacity_ah),
                          "temperature_c": temps.get(local, float("nan")), "has_step_row": local in known})
            if local in known:
                st = step_of[local]
                window = screened[max(0, position - RESISTANCE_WINDOW_CYCLES + 1):position + 1]
                position += 1
                finite = window[np.isfinite(window)]
                trailing = float(np.median(finite)) if len(finite) else self._last_trailing(new_rows)
                entry.update({"r_step_ohm": float(st.r_step_ohm),
                              "mean_current_a": float(st.mean_current_a),
                              "r_trailing_ohm": float(trailing)})
                if np.isfinite(trailing) and local in blocks:
                    drop = abs(entry["mean_current_a"]) * float(trailing)
                    entry["ir_drop_v"] = drop
                    v, q = blocks[local]
                    spec = self.spec if drop == 0.0 else WindowSpec(self.v_high - drop, self.v_low - drop)
                    entry["window_charge_ah"] = float(window_charge(v, q, spec))
                    entry["cycle_charge_ah"] = float(q.max())
            new_rows.append(entry)
        self.rows.extend(new_rows)
        return len(new_rows)

    def _last_trailing(self, pending: list[dict]) -> float:
        for r in reversed(self.rows + pending):
            if np.isfinite(r.get("r_trailing_ohm", np.nan)):
                return float(r["r_trailing_ohm"])
        return float("nan")

    @staticmethod
    def _temperatures(telemetry: pd.DataFrame, d: pd.DataFrame) -> dict[int, float]:
        if "avg_temp" not in d.columns:
            return {}
        return {int(c): float(t) for c, t in zip(d["cycle"], d["avg_temp"], strict=True)
                if t is not None and np.isfinite(float(t))}

    # -- answering --------------------------------------------------------
    def table(self) -> pd.DataFrame:
        """The gated per-discharge table, rebuilt from memory alone."""
        rows = pd.DataFrame(self.rows, columns=[*_ROW_FIELDS, "has_step_row"])
        measured = rows[np.isfinite(rows["r_trailing_ohm"].astype(float))]
        if measured.empty:
            return pd.DataFrame()
        charges = measured.assign(cell_id=self.battery_id)[
            ["cell_id", "cycle", "ir_drop_v", "window_charge_ah", "cycle_charge_ah"]]
        table = soh_table_from_charges(charges, self.spec, self.reference_cycles)
        table = table.merge(measured[["cycle", "r_trailing_ohm"]], on="cycle", how="left")
        return apply_field_gates(table, rows[["cycle", "capacity_ah"]], self.reference_cycles)

    def current(self) -> FieldSOH:
        return field_soh_from_table(self.table(), self.n_discharges, self.spec, True,
                                    self.reference_cycles)

    def readings(self) -> pd.DataFrame:
        """Each discharge with its conditions and accepted SOH, for display or learning."""
        rows = pd.DataFrame(self.rows, columns=[*_ROW_FIELDS, "has_step_row"])
        t = self.table()
        if t.empty:
            return rows.assign(soh=np.nan)
        return rows.merge(t[["cycle", "soh_window_accepted"]].rename(
            columns={"soh_window_accepted": "soh"}), on="cycle", how="left")

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        def clean(v):
            return None if isinstance(v, float) and not np.isfinite(v) else v
        return {"version": PROFILE_VERSION, "battery_id": self.battery_id,
                "rest_threshold_a": self.rest_threshold_a, "v_high": self.v_high, "v_low": self.v_low,
                "reference_cycles": self.reference_cycles, "sources": list(self.sources),
                "rows": [{k: clean(v) for k, v in r.items()} for r in self.rows]}

    @classmethod
    def from_dict(cls, data: dict) -> BatteryProfile:
        if data.get("version") != PROFILE_VERSION:
            raise ValueError(f"BatteryProfile: unsupported profile version {data.get('version')!r}")
        rows = [{k: (float("nan") if v is None else v) for k, v in r.items()} for r in data["rows"]]
        return cls(data["battery_id"], data["rest_threshold_a"], data["v_high"], data["v_low"],
                   data["reference_cycles"], rows, list(data.get("sources", [])))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BatteryProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
