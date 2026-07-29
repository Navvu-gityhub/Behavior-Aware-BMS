import { Sparkline } from './Sparkline.jsx';

export function BatteryDetail({ detail, timeline }) {
  if (!detail) {
    return (
      <div className="empty-state">
        <p>Select a battery from the fleet table to see its twin state, Guardian report, and health traces.</p>
      </div>
    );
  }

  const { twin } = detail;
  const transitions = [...detail.transitions].reverse();

  return (
    <>
      <div className="detail-header">
        <div className="detail-id">{detail.battery_id}</div>
        <span className={`badge ${twin.twin_state}`}>{twin.twin_state.replace('_', ' ')}</span>
      </div>

      <div className="detail-stats">
        <div className="stat">
          <div className="label">Health Index</div>
          <div className="value">{twin.health_index.toFixed(0)}</div>
        </div>
        <div className="stat">
          <div className="label">Failure Likelihood</div>
          <div className="value">{(twin.failure_likelihood * 100).toFixed(0)}%</div>
        </div>
        <div className="stat">
          <div className="label">RUL</div>
          <div className="value">{twin.rul_cycles.toLocaleString()}</div>
        </div>
        <div className="stat">
          <div className="label">Risk</div>
          <div className="value">{detail.risk_level}</div>
        </div>
        <div className="stat">
          <div className="label">Policy</div>
          <div className="value" style={{ fontSize: 13 }}>{twin.replacement_policy}</div>
        </div>
      </div>

      <div className="report-box">{detail.guardian_report}</div>
      <div className="report-meta">
        <b>Primary causes:</b> {detail.primary_causes}
        <br />
        <b>Recommendation:</b> {detail.recommendation}
      </div>

      <div className="panel-title divider">Health Trace (by cycle)</div>
      <div className="traces">
        {timeline.length ? (
          <>
            <div className="trace">
              <div className="trace-label"><span>Stress score</span><span>0–100</span></div>
              <Sparkline points={timeline} dataKey="stress_score" color="var(--accent)" />
            </div>
            <div className="trace">
              <div className="trace-label"><span>State of charge</span><span>%</span></div>
              <Sparkline points={timeline} dataKey="soc" color="var(--soc-color)" unit="%" />
            </div>
            <div className="trace">
              <div className="trace-label"><span>Temperature</span><span>°C</span></div>
              <Sparkline points={timeline} dataKey="temperature_c" color="var(--temp-color)" unit="°" />
            </div>
          </>
        ) : (
          <div style={{ padding: '8px 0', color: 'var(--text-faint)', fontSize: 12 }}>
            No timeline for this battery in the most recent run.
          </div>
        )}
      </div>

      <div className="panel-title divider">Twin State Transitions</div>
      <div className="transitions">
        {transitions.length ? (
          transitions.map((t, i) => (
            <div className="t-row" key={i}>
              <span>{new Date(t.at).toLocaleString()}</span>
              <span>{t.from_state || 'first seen'}</span>
              <span className="arrow">→</span>
              <span>{t.to_state}</span>
            </div>
          ))
        ) : (
          <div className="t-row">no transitions recorded</div>
        )}
      </div>
    </>
  );
}
