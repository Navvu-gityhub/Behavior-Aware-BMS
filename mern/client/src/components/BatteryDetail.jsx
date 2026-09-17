import { Sparkline } from './Sparkline.jsx';
import { ProvenanceStat, ProvenanceTag, ProvenanceLegend } from './Provenance.jsx';

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
        <ProvenanceTag field="twin_state" />
      </div>

      <ProvenanceLegend />

      <div className="detail-stats">
        <ProvenanceStat field="health_index" value={twin.health_index.toFixed(0)} />
        <ProvenanceStat
          field="failure_likelihood"
          value={`${(twin.failure_likelihood * 100).toFixed(0)}%`}
        />
        <ProvenanceStat field="rul_cycles" value={twin.rul_cycles.toLocaleString()} />
        <ProvenanceStat field="risk_level" value={detail.risk_level} />
        <ProvenanceStat field="replacement_policy" value={twin.replacement_policy} />
      </div>

      <div className="report-box">{detail.guardian_report}</div>
      <div className="report-meta">
        <b>Primary causes:</b> {detail.primary_causes}
        <br />
        <b>Recommendation:</b> {detail.recommendation}
      </div>
      <div className="evidence-row">
        <span className={`evidence-badge ${detail.evidence_confidence === 'N/A' ? 'NA' : detail.evidence_confidence}`}>
          {detail.evidence_confidence}
        </span>
        <span className="evidence-text">{detail.evidence_note}</span>
      </div>

      <div className="panel-title divider">Health Trace (by cycle)</div>
      <div className="traces">
        {timeline.length ? (
          <>
            <div className="trace">
              <div className="trace-label">
                <span>Stress score <ProvenanceTag field="stress_score" /></span>
                <span>0–100</span>
              </div>
              <Sparkline points={timeline} dataKey="stress_score" color="var(--accent)" />
            </div>
            <div className="trace">
              <div className="trace-label">
                <span>State of charge <ProvenanceTag field="soc" /></span>
                <span>%</span>
              </div>
              <Sparkline points={timeline} dataKey="soc" color="var(--soc-color)" unit="%" />
            </div>
            <div className="trace">
              <div className="trace-label">
                <span>Temperature <ProvenanceTag field="temperature_c" /></span>
                <span>°C</span>
              </div>
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
