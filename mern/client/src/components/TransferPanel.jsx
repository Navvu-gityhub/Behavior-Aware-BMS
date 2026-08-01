import { Fragment, useEffect, useState } from 'react';
import { api, ApiError } from '../api';

// Transfer feasibility and dataset comparison.
//
// This view exists to show which cross-dataset experiments are scientifically
// admissible, and — more usefully — which are not and why. A blocked axis is
// rendered with its reason attached, because "NASA to Stanford is marginal" is
// only actionable alongside "Severson holds ambient temperature at 30 C in a
// chamber, so a fitted coefficient has nothing to act on".

function StatusBadge({ status }) {
  const tone =
    status === 'PREDICTED_FEASIBLE' ? 'NORMAL'
      : status === 'PREDICTED_MARGINAL' ? 'MODERATE_RISK'
        : 'HIGH_RISK';
  return <span className={`badge ${tone}`}>{status.replace('PREDICTED_', '').toLowerCase()}</span>;
}

export function TransferPanel() {
  const [rows, setRows] = useState(null);
  const [specs, setSpecs] = useState(null);
  const [expanded, setExpanded] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.feasibility('nasa'), api.datasets()])
      .then(([f, d]) => {
        if (cancelled) return;
        setRows(f);
        setSpecs(d);
      })
      .catch((err) => !cancelled && setError(err instanceof ApiError ? err.message : String(err)));
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <div className="empty-state">{error}</div>;
  if (!rows || !specs) return <div className="empty-state">Loading feasibility…</div>;

  const specByName = Object.fromEntries(specs.map((s) => [s.name, s]));

  return (
    <div className="transfer">
      <p className="dim small">
        Predicted from published dataset metadata, before any file is loaded. A transfer needs a
        feature that varies in <em>both</em> datasets: constant in the source and no coefficient
        can be fitted, constant in the target and the coefficient has nothing to act on.
      </p>

      <table className="transfer-table">
        <thead>
          <tr>
            <th>Target</th>
            <th>Cells</th>
            <th>Status</th>
            <th>Usable axes</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const spec = specByName[row.target];
            const open = expanded === row.target;
            return (
              <Fragment key={row.target}>
                <tr
                  className={`transfer-row ${open ? 'open' : ''}`}
                  onClick={() => setExpanded(open ? null : row.target)}
                >
                  <td className="mono">{row.target}</td>
                  <td className="mono">{spec ? spec.n_cells : '—'}</td>
                  <td><StatusBadge status={row.status} /></td>
                  <td className="mono small">
                    {row.usable_axes.length ? row.usable_axes.join(', ') : 'none'}
                  </td>
                  <td className="dim">{open ? '−' : '+'}</td>
                </tr>
                {open && (
                  <tr>
                    <td colSpan={5} className="transfer-detail">
                      <div className="axis-list">
                        {row.verdicts.map((v) => (
                          <div key={v.axis} className={`axis ${v.usable ? 'usable' : 'blocked'}`}>
                            <div className="axis-head">
                              <span className="mono">{v.axis.replace(/_/g, ' ')}</span>
                              <span className="mono dim">
                                {v.source_variation} → {v.target_variation}
                              </span>
                              {v.marginal && v.usable && <span className="axis-tag">marginal</span>}
                            </div>
                            <p>{v.reason}</p>
                          </div>
                        ))}
                      </div>
                      {spec && (
                        <div className="spec-box">
                          <div className="spec-head">
                            {spec.description}
                            {spec.chemistry && <span className="dim"> · {spec.chemistry}</span>}
                          </div>
                          <ul>
                            {spec.caveats.map((c, i) => <li key={i}>{c}</li>)}
                          </ul>
                          <div className="dim small">{spec.citation}</div>
                        </div>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
