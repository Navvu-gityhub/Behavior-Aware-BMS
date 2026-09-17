import { ORIGINS, describe, kindTone } from '../provenance.js';

/**
 * A small badge saying what kind of number sits next to it.
 *
 * Deliberately not a hover-only tooltip: the kind is always visible, and only
 * the longer definition is on hover. A caveat that appears only on hover is a
 * caveat most readers never see.
 */
export function ProvenanceTag({ field }) {
  const entry = describe(field);
  const title = entry.caveat
    ? `${entry.definition}\n\nCAVEAT: ${entry.caveat}`
    : entry.definition;

  return (
    <abbr className={`prov prov-${kindTone(entry.kind)}`} title={title}>
      {entry.kind}
      {entry.caveat ? <span className="prov-flag" aria-hidden="true">!</span> : null}
    </abbr>
  );
}

/**
 * A labelled value with its provenance attached.
 *
 * `value` is rendered as given; formatting stays with the caller, because the
 * number of decimal places is a display decision and the provenance is not.
 */
export function ProvenanceStat({ field, value, hint }) {
  const entry = describe(field);
  return (
    <div className="stat">
      <div className="label">
        {entry.label}
        {entry.unit ? <span className="stat-unit"> ({entry.unit})</span> : null}
      </div>
      <div className="value">{value}</div>
      <ProvenanceTag field={field} />
      {hint ? <div className="stat-hint">{hint}</div> : null}
    </div>
  );
}

/**
 * The banner that says where the data on screen came from.
 *
 * This is the label that matters most. A `MEASURED` tag on synthetic telemetry
 * is not a measurement, so the origin has to be impossible to miss rather than
 * inferred from which tab the user happens to be on.
 */
export function DataSourceBanner({ origin, detail }) {
  const meta = ORIGINS[origin];
  if (!meta) return null;

  return (
    <div className={`origin origin-${meta.tone}`} role="status">
      <span className="origin-chip">{origin}</span>
      <span className="origin-blurb">
        {meta.blurb}
        {detail ? ` ${detail}` : ''}
      </span>
      {!meta.reachable ? (
        <span className="origin-unreachable">never produced by this project</span>
      ) : null}
    </div>
  );
}

/** Legend explaining the three kinds, shown once per view rather than per value. */
export function ProvenanceLegend() {
  return (
    <div className="prov-legend">
      <span className="prov prov-measured">MEASURED</span>
      <span>read from a sensor or dataset record</span>
      <span className="prov prov-calculated">CALCULATED</span>
      <span>deterministic arithmetic on measured values</span>
      <span className="prov prov-model">MODEL-DERIVED</span>
      <span>output of hand-tuned weights; not validated against measured fade</span>
      <span className="prov-flag-legend">!</span>
      <span>carries a documented caveat — hover the badge</span>
    </div>
  );
}
