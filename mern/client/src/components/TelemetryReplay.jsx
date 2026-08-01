import { useState } from 'react';
import { api, ApiError } from '../api';

// Telemetry replay: submit a CAN log path, show what the pipeline made of it.
//
// The design constraint that shapes this component is that a replay can
// legitimately produce no scores. A DBC without a temperature signal, or a log
// with no complete discharge, are both ordinary outcomes rather than errors.
// So the refusal list is rendered as a first-class result, not as an error
// state tucked away in a corner: it is often the most informative thing the
// run produced.

const CHANNELS = [
  { key: 'voltage_v', label: 'Pack voltage' },
  { key: 'current_a', label: 'Pack current' },
  { key: 'temperature_c', label: 'Pack temperature' },
  { key: 'soc', label: 'State of charge' },
];

// Channels a viewer might expect from a battery pack but which no dataset or
// DBC in this project supplies. Listed explicitly rather than omitted, so the
// pack view cannot imply per-cell resolution it does not have (ADR 0004).
const NEVER_INSTRUMENTED = [
  'Individual cell voltages',
  'Individual cell temperatures',
  'Cell balancing currents',
];

function PackInstrumentation({ coverage }) {
  if (!coverage) return null;
  const mapped = new Set(coverage.mapped_channels || []);

  return (
    <div className="pack-view">
      <div className="pack-figure" aria-label="Battery pack, aggregate instrumentation">
        {Array.from({ length: 24 }, (_, i) => (
          <span key={i} className="pack-cell" title="Not individually instrumented" />
        ))}
      </div>
      <div className="pack-legend">
        <div className="pack-legend-title">Instrumented</div>
        {CHANNELS.map(({ key, label }) => (
          <div key={key} className={`chan ${mapped.has(key) ? 'on' : 'off'}`}>
            <span className="chan-dot" />
            {label}
            {!mapped.has(key) && <span className="chan-note">not on this bus</span>}
          </div>
        ))}
        <div className="pack-legend-title dim">Not instrumented</div>
        {NEVER_INSTRUMENTED.map((label) => (
          <div key={label} className="chan off">
            <span className="chan-dot" />
            {label}
          </div>
        ))}
        <p className="pack-note">
          The pack above is drawn as a single measured unit. Cells are shown greyed because this
          schema carries one aggregate temperature channel — colouring them individually would
          mean inventing values that were never measured.
        </p>
      </div>
    </div>
  );
}

export function TelemetryReplay({ onRun }) {
  const [logPath, setLogPath] = useState('');
  const [dbcPath, setDbcPath] = useState('');
  const [batteryId, setBatteryId] = useState('VEH_01');
  const [allowPartial, setAllowPartial] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleReplay() {
    setRunning(true);
    setError(null);
    try {
      const payload = {
        log_path: logPath,
        battery_id: batteryId,
        require_full_coverage: !allowPartial,
      };
      if (dbcPath) payload.dbc_path = dbcPath;
      const run = await api.replay(payload);
      setResult(run);
      if (onRun) onRun(batteryId, run);
    } catch (err) {
      setError(err instanceof ApiError ? `${err.status} — ${err.message}` : String(err));
      setResult(null);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="replay">
      <div className="replay-form">
        <label>
          CAN log path
          <input
            type="text"
            value={logPath}
            placeholder="/path/to/drive.asc"
            onChange={(e) => setLogPath(e.target.value)}
          />
        </label>
        <label>
          DBC (optional)
          <input
            type="text"
            value={dbcPath}
            placeholder="defaults to bundled reference pack"
            onChange={(e) => setDbcPath(e.target.value)}
          />
        </label>
        <label className="narrow">
          Battery ID
          <input type="text" value={batteryId} onChange={(e) => setBatteryId(e.target.value)} />
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={allowPartial}
            onChange={(e) => setAllowPartial(e.target.checked)}
          />
          Decode past a missing channel
        </label>
        <button onClick={handleReplay} disabled={running || !logPath}>
          {running ? 'Replaying…' : 'Replay log'}
        </button>
      </div>

      {error && <div className="replay-error">{error}</div>}

      {result && (
        <div className="replay-result">
          <div className="replay-status">
            <span className={`badge status-${result.status}`}>{result.status.replace(/_/g, ' ')}</span>
            <span className="mono dim">
              {result.n_decoded.toLocaleString()} / {result.n_frames.toLocaleString()} frames decoded
            </span>
            <span className="stages">
              {result.stages_completed.map((s) => (
                <span key={s} className="stage-chip">{s}</span>
              ))}
            </span>
          </div>

          <PackInstrumentation coverage={result.coverage} />

          {result.capacity_yield && (
            <div className="yield-box">
              <div className="yield-headline">
                <strong>{result.capacity_yield.n_complete}</strong> of{' '}
                <strong>{result.capacity_yield.n_discharges}</strong> discharges usable for SOH
              </div>
              <div className="yield-bar">
                <span
                  style={{ width: `${Math.round(result.capacity_yield.usable_fraction * 100)}%` }}
                />
              </div>
              <p className="dim small">
                Partial discharges are excluded rather than scaled: capacity only compares across
                equal depths of discharge, and the BMS's own state of charge is derived from a
                capacity estimate, so normalising by it would be circular.
              </p>
            </div>
          )}

          {result.cycles.length > 0 && (
            <table className="cycle-table">
              <thead>
                <tr>
                  <th>Cycle</th>
                  <th>Capacity (Ah)</th>
                  <th>Mean current (A)</th>
                  <th>Mean temp (°C)</th>
                  <th>Complete</th>
                </tr>
              </thead>
              <tbody>
                {result.cycles.map((c) => (
                  <tr key={c.cycle}>
                    <td className="mono">{c.cycle}</td>
                    <td className="mono">{c.capacity_ah.toFixed(4)}</td>
                    <td className="mono">{c.mean_current_a.toFixed(2)}</td>
                    <td className="mono">{c.avg_temp == null ? '—' : c.avg_temp.toFixed(1)}</td>
                    <td>{c.is_complete ? '✓' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {result.refusals.length > 0 && (
            <div className="refusals">
              <div className="refusals-title">Pipeline refused to compute</div>
              {result.refusals.map((r, i) => (
                <p key={i}>{r}</p>
              ))}
            </div>
          )}

          <div className="fade-refusal">
            <div className="fade-title">
              Fade prediction: <span className="mono">unavailable</span>
            </div>
            <p>{result.fade_prediction_refusal}</p>
          </div>
        </div>
      )}
    </div>
  );
}
