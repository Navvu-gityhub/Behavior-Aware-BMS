// Guardian explanation panel.
//
// Renders the exact Shapley decomposition the pipeline produced, along with the
// caveat that travels with every Guardian row. The caveat is shown rather than
// summarised: it records that the attribution is exact with respect to the
// score while the score itself is not a validated predictor of capacity fade,
// and losing that distinction is precisely how a rule-based triage number gets
// mistaken for a measurement.

const EVIDENCE_TONE = {
  VALIDATED: 'NORMAL',
  MIXED: 'MODERATE_RISK',
  HEURISTIC: 'HIGH_RISK',
};

export function GuardianPanel({ guardian }) {
  if (!guardian || guardian.length === 0) {
    return (
      <div className="empty-state">
        No Guardian output. The run either refused or produced no complete cycle.
      </div>
    );
  }

  const row = guardian[0];
  const shapKeys = Object.keys(row)
    .filter((k) => k.startsWith('health_shap_'))
    .sort((a, b) => Math.abs(row[b]) - Math.abs(row[a]));
  const largest = shapKeys.length ? Math.abs(row[shapKeys[0]]) : 1;

  return (
    <div className="guardian">
      <div className="guardian-head">
        <span className={`badge ${row.battery_state}`}>{row.battery_state}</span>
        {row.evidence_confidence && (
          <span className={`evidence-badge ${EVIDENCE_TONE[row.evidence_confidence] || ''}`}>
            {row.evidence_confidence}
          </span>
        )}
        <span className="mono dim">{row.attribution_method}</span>
      </div>

      <div className="report-box">{row.guardian_report}</div>

      {shapKeys.length > 0 && (
        <div className="shap">
          <div className="panel-title">Contribution to health index</div>
          {shapKeys.map((key) => {
            const value = row[key];
            const width = Math.round((Math.abs(value) / largest) * 100);
            return (
              <div className="shap-row" key={key}>
                <span className="shap-label mono">{key.replace('health_shap_', '')}</span>
                <div className="shap-bar">
                  <span
                    className={value >= 0 ? 'pos' : 'neg'}
                    style={{ width: `${width}%` }}
                  />
                </div>
                <span className="shap-value mono">{value >= 0 ? '+' : ''}{value.toFixed(2)}</span>
              </div>
            );
          })}
          <p className="dim small">
            Exact Shapley values for an additive score, not a sampled approximation. They sum to the
            difference between this battery's score and the fleet mean.
          </p>
        </div>
      )}

      {row.evidence_note && <p className="dim small">{row.evidence_note}</p>}

      {row.guardian_caveat && <div className="caveat">{row.guardian_caveat}</div>}
    </div>
  );
}
