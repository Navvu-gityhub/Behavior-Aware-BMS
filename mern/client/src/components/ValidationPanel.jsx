import { useEffect, useState } from 'react';
import { api } from '../api.js';

/**
 * What has actually been tested, beside the scores the rest of the app shows.
 *
 * Every figure is read from the tracked artifacts under reports/metrics/ via
 * the Python service. Nothing is hardcoded here: if a benchmark is re-run and a
 * number moves, this panel moves with it, and if an artifact is missing the
 * panel says which rather than rendering a plausible default.
 *
 * The one rule it enforces visually: an R2 is never shown without the noise
 * ceiling of its target next to it. That pairing is what turns "0.46" from an
 * impressive-looking number into "0.46 against a maximum attainable 0.57".
 */
export function ValidationPanel() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .validationSummary()
      .then((data) => !cancelled && setSummary(data))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="empty-state">
        <p>Could not load validation evidence: {error}</p>
      </div>
    );
  }
  if (!summary) return <div className="empty-state"><p>Loading evidence…</p></div>;

  if (!summary.available) {
    return (
      <div className="empty-state">
        <p>
          The validation artifacts are not present in this checkout, so nothing
          can be reported. Missing:{' '}
          <code>{summary.missing_artifacts.join(', ')}</code>
        </p>
        <p>
          Run <code>python scripts/run_validation_suite.py</code> to produce
          them. This panel will not substitute defaults.
        </p>
      </div>
    );
  }

  const scored = summary.methods
    .filter((m) => m.loco_r2 !== null && m.lobo_r2 !== null)
    .sort((a, b) => (b.loco_r2 ?? -9) - (a.loco_r2 ?? -9));

  return (
    <div className="validation">
      <div className="headline-claim">
        <div className="headline-label">What this project actually established</div>
        <p>{summary.headline}</p>
      </div>

      <div className="panel-title divider">Target noise ceilings</div>
      <p className="vnote">
        The maximum R² anything could attain against each target. A model score
        means nothing without the ceiling beside it.
      </p>
      <table className="vtable">
        <thead>
          <tr>
            <th>Target</th><th>Cells</th><th>Rows</th>
            <th>Signal fraction</th><th>Max attainable R²</th><th>Reading</th>
          </tr>
        </thead>
        <tbody>
          {summary.target_ceilings.map((t) => (
            <tr key={t.target} className={t.max_attainable_r2 < 0.1 ? 'row-bad' : ''}>
              <td className="bid">{t.target}</td>
              <td>{t.n_cells}</td>
              <td>{t.n_rows.toLocaleString()}</td>
              <td>{t.signal_fraction.toFixed(3)}</td>
              <td><b>{t.max_attainable_r2.toFixed(3)}</b></td>
              <td className="vverdict">{t.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="panel-title divider">Methods under protocol shift</div>
      <p className="vnote">
        LOBO holds out a cell; LOCO holds out an entire operating protocol. The
        gap is what the method loses when the duty cycle changes.{' '}
        <b>No ranking is asserted:</b> six ranking claims were made from
        single-frame runs during this work and all six were withdrawn.
      </p>
      <table className="vtable">
        <thead>
          <tr>
            <th>Method</th><th>Target</th>
            <th>LOBO R²</th><th>LOCO R²</th><th>Gap</th>
            <th>Ceiling</th><th>Promoted</th>
          </tr>
        </thead>
        <tbody>
          {scored.map((m) => (
            <tr key={`${m.method}-${m.target}`}>
              <td className="bid">{m.method}</td>
              <td>{m.target}</td>
              <td>{m.lobo_r2.toFixed(3)}</td>
              <td>{m.loco_r2.toFixed(3)}</td>
              <td className={m.loco_minus_lobo < 0 ? 'neg' : ''}>
                {m.loco_minus_lobo.toFixed(3)}
              </td>
              <td className="vceiling">
                {m.r2_ceiling === null ? '—' : m.r2_ceiling.toFixed(3)}
              </td>
              <td>{m.promoted ? 'yes' : 'no'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {summary.coverage.length > 0 && (
        <>
          <div className="panel-title divider">Conformal coverage</div>
          <p className="vnote">
            Read <b>min</b> coverage and the worst group, not the median. An
            aggregate that meets the nominal guarantee can hide a cohort where
            it fails badly.
          </p>
          <table className="vtable">
            <thead>
              <tr>
                <th>Method</th><th>Split</th><th>Folds</th>
                <th>Median</th><th>Min</th><th>Worst group</th>
                <th>Nominal</th><th>Below nominal</th>
              </tr>
            </thead>
            <tbody>
              {summary.coverage.map((c) => (
                <tr key={`${c.method}-${c.split}`}>
                  <td className="bid">{c.method}</td>
                  <td>{c.split}</td>
                  <td>{c.n_folds}</td>
                  <td>{(c.median_coverage * 100).toFixed(0)}%</td>
                  <td className={c.min_coverage < c.nominal ? 'neg' : ''}>
                    <b>{(c.min_coverage * 100).toFixed(0)}%</b>
                  </td>
                  <td>{c.worst_group ?? '—'}</td>
                  <td>{(c.nominal * 100).toFixed(0)}%</td>
                  <td>{(c.fraction_below_nominal * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <div className="panel-title divider">Limitations</div>
      <ul className="vlimits">
        {summary.limitations.map((limit) => (
          <li key={limit}>{limit}</li>
        ))}
      </ul>
    </div>
  );
}
