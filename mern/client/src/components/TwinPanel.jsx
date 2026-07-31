import { useEffect, useState } from 'react';
import { api, ApiError } from '../api';

// Digital twin state and its history.
//
// A single snapshot says what condition a pack is in; the sequence says whether
// it is getting worse. Transitions are therefore given equal weight to the
// current state rather than being relegated to a footnote.

export function TwinPanel({ batteryId, refreshKey }) {
  const [history, setHistory] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!batteryId) return;
    let cancelled = false;
    setError(null);
    api
      .twinHistory(batteryId)
      .then((d) => !cancelled && setHistory(d))
      .catch((err) => {
        if (cancelled) return;
        setHistory(null);
        setError(err instanceof ApiError ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [batteryId, refreshKey]);

  if (!batteryId) {
    return <div className="empty-state">Replay a CAN log to build twin history.</div>;
  }
  if (error) return <div className="empty-state">{error}</div>;
  if (!history) return <div className="empty-state">Loading twin history…</div>;

  const latest = history.snapshots[history.snapshots.length - 1];

  return (
    <div className="twin-panel">
      <div className="twin-current">
        <span className={`badge ${latest.twin_state}`}>{latest.twin_state.replace(/_/g, ' ')}</span>
        <div className="detail-stats">
          <div className="stat">
            <div className="label">Health index</div>
            <div className="value">{latest.health_index.toFixed(0)}</div>
          </div>
          <div className="stat">
            <div className="label">Failure likelihood</div>
            <div className="value">{(latest.failure_likelihood * 100).toFixed(0)}%</div>
          </div>
          <div className="stat">
            <div className="label">RUL cycles</div>
            <div className="value">{latest.rul_cycles.toLocaleString()}</div>
          </div>
          <div className="stat">
            <div className="label">Snapshots</div>
            <div className="value">{history.n_snapshots}</div>
          </div>
        </div>
        <div className="twin-policy">{latest.replacement_policy}</div>
      </div>

      <div className="twin-history">
        <div className="panel-title">State history</div>
        <div className="twin-track">
          {history.snapshots.map((s, i) => (
            <span
              key={i}
              className={`twin-tick ${s.twin_state}`}
              title={`${s.twin_state} — health ${s.health_index.toFixed(0)} — ${s.evaluated_at}`}
            />
          ))}
        </div>

        {history.transitions.length > 0 ? (
          <div className="transitions">
            {history.transitions.map((t, i) => (
              <div className="t-row" key={i}>
                <span className="mono dim">{t.at.slice(11, 19)}</span>
                <span>
                  {t.from_state ? t.from_state.replace(/_/g, ' ') : 'first observation'}
                  {' → '}
                  <strong>{t.to_state.replace(/_/g, ' ')}</strong>
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="dim small">No state change across {history.n_snapshots} snapshot(s).</div>
        )}
      </div>

      <p className="dim small">
        Twin state is a relabel of the rule-based health index thresholds, not an independent
        model — see docs/digital_twin.md.
      </p>
    </div>
  );
}
