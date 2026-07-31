import { useEffect, useState } from 'react';
import { api, ApiError } from '../api';

// Thermal visualisation over cycle and discharge progress.
//
// Both axes are measured. There is deliberately no per-cell axis: the unified
// schema carries a single pack-aggregate temperature channel, so a per-cell
// heatmap would mean interpolating dozens of values from one (ADR 0004). The
// backend states this in `resolution_note` and it is rendered verbatim rather
// than paraphrased, so the constraint travels with the view.

function colourFor(value, min, max) {
  if (max <= min) return 'var(--soc-color)';
  const t = (value - min) / (max - min);
  // Blue through amber to red. Interpolating in RGB is adequate here because
  // the endpoints are far apart in hue and the band count is low.
  const r = Math.round(91 + t * (242 - 91));
  const g = Math.round(157 + t * (70 - 157));
  const b = Math.round(240 + t * (60 - 240));
  return `rgb(${r}, ${g}, ${b})`;
}

export function ThermalMap({ batteryId }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!batteryId) return;
    let cancelled = false;
    setError(null);
    api
      .thermal(batteryId)
      .then((d) => !cancelled && setData(d))
      .catch((err) => {
        if (cancelled) return;
        setData(null);
        setError(err instanceof ApiError ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [batteryId]);

  if (!batteryId) {
    return <div className="empty-state">Replay a CAN log to see its thermal profile.</div>;
  }
  if (error) {
    return <div className="empty-state">{error}</div>;
  }
  if (!data || data.points.length === 0) {
    return <div className="empty-state">No temperature channel in the last run for {batteryId}.</div>;
  }

  const { points, temperature_min_c: min, temperature_max_c: max } = data;
  const cycles = [...new Set(points.map((p) => p.cycle))].sort((a, b) => a - b);

  // Bucket each cycle's samples into fixed columns so rows align regardless of
  // how many samples a cycle happened to contain.
  const COLUMNS = 24;
  const grid = cycles.map((cycle) => {
    const inCycle = points.filter((p) => p.cycle === cycle);
    return Array.from({ length: COLUMNS }, (_, col) => {
      const lo = col / COLUMNS;
      const hi = (col + 1) / COLUMNS;
      const bucket = inCycle.filter((p) => p.phase_fraction >= lo && p.phase_fraction < hi);
      if (bucket.length === 0) return null;
      return bucket.reduce((sum, p) => sum + p.temperature_c, 0) / bucket.length;
    });
  });

  return (
    <div className="thermal">
      <div className="thermal-scale">
        <span className="mono">{min.toFixed(1)}°C</span>
        <span className="thermal-gradient" />
        <span className="mono">{max.toFixed(1)}°C</span>
      </div>

      <div className="thermal-grid">
        {grid.map((row, i) => (
          <div className="thermal-row" key={cycles[i]}>
            <span className="thermal-label mono">cyc {cycles[i]}</span>
            <div className="thermal-cells">
              {row.map((value, j) => (
                <span
                  key={j}
                  className="thermal-cell"
                  style={{ background: value == null ? 'var(--panel-2)' : colourFor(value, min, max) }}
                  title={value == null ? 'no sample' : `${value.toFixed(1)}°C`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="thermal-axis mono dim">discharge start → discharge end</div>
      <p className="dim small">{data.resolution_note}</p>
    </div>
  );
}
