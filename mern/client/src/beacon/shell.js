/**
 * The BEACON dashboard shell — the exact DOM the renderer expects.
 *
 * Transcribed from `render_beacon_html()` in the design prototype's
 * `beacon.py`. Every element id here is one `renderer.js` reaches for
 * (`#kpis`, `#batStage`, `#fleetGrid`, `#guardianMount`, `#twinMount`, …), so
 * this file and the renderer must change together or not at all.
 *
 * It is built as a string rather than as JSX on purpose. The renderer owns this
 * subtree and rewrites it imperatively; giving React a second opinion about the
 * same nodes would mean two things writing the same DOM, which is how the
 * battery figure ends up half-rendered.
 */

const NAV_ITEMS = [
  ['home', 'Overview', 'M3 11.2 12 4l9 7.2V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z'],
  ['health', 'Battery health', 'M12 3l8 4v6c0 4.5-3.4 7.7-8 8-4.6-.3-8-3.5-8-8V7zM9 12l2 2 4-4'],
  ['analytics', 'Analytics', 'M4 19V9M10 19V5M16 19v-7M22 19H2'],
  ['evidence', 'Evidence', 'M6 3h9l4 4v14H6zM15 3v4h4M9 13h6M9 17h4'],
];

const esc = (value) =>
  String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

function navHtml() {
  return NAV_ITEMS.map(
    ([key, label, path], index) =>
      `<button class="dbtn" data-view="${key}" aria-current="${index === 0}" aria-label="${label}">` +
      `<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" ` +
      `stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="${path}"/></svg>` +
      `<span class="tt">${label}</span></button>`
  ).join('\n    ');
}

function evidenceHtml(fleet) {
  return `
<section class="sec">
  <div class="sechead"><div><h2>Validation status</h2>
    <div class="sub">What these numbers are validated to support, and what they are not.</div></div></div>
  <div class="g12">
    <div class="c7"><div class="kv">
      <div class="row"><div class="k"><span>Health index vs measured NASA fade (n=33)</span></div>
        <div class="v critical">rho -0.27 &middot; p 0.12</div></div>
      <div class="row"><div class="k"><span>Fitted v2 model, unseen cell in a known protocol</span></div>
        <div class="v good">rho 0.84 &middot; p &lt; 0.001</div></div>
      <div class="row"><div class="k"><span>Fitted v2 model, unseen protocol</span></div>
        <div class="v critical">rho -0.30 &middot; p 0.10</div></div>
      <div class="row"><div class="k"><span>Risk score points from terms constant across the NASA fleet</span></div>
        <div class="v warn">61%</div></div>
      <div class="row"><div class="k"><span>Distinct health index values across this fleet</span></div>
        <div class="v">${fleet.distinct_health_values}<u> of ${fleet.n_batteries}</u></div></div>
      <div class="row"><div class="k"><span>Guardian attribution against its own score</span></div>
        <div class="v good">exact</div></div>
    </div></div>
    <div class="c5">
      <div class="steps">
        <div class="step"><span class="i">01</span><span>Attribution is exact with respect to the score it
          decomposes. The score itself is not a validated predictor of capacity fade.</span></div>
        <div class="step"><span class="i">02</span><span>Risk, health and RUL are rule-based heuristics with
          hand-chosen weights. Treat them as transparent triage, not as measurement.</span></div>
        <div class="step"><span class="i">03</span><span>Any metric this dataset cannot support renders as
          unavailable with its reason, never as a substituted value (ADR-0004).</span></div>
      </div>
      <div class="caveat">Full derivations: <code>docs/final_report.md</code>.</div>
    </div>
  </div>
</section>`;
}



/**
 * The measured-hardware panel for the home view.
 *
 * Sits beneath the provenance banner so the two are read together: the banner
 * describes where the FLEET came from, this panel describes where these four
 * numbers came from, and they are usually not the same place. It is styled
 * unlike the fleet tiles on purpose - see the note in beacon.css.
 *
 * Returns '' when no rig capture was replayed, so the home view is unchanged
 * for a run that has no hardware behind it.
 */
function rigHomeHtml(rig) {
  if (!rig) return '';

  const by = {};
  (rig.measured || []).forEach((m) => { by[m.channel] = m; });

  const fmt = (m, decimals, unit) => {
    if (!m) return null;
    const range = m.minimum === m.maximum
      ? 'constant'
      : `${m.minimum.toFixed(decimals)} to ${m.maximum.toFixed(decimals)}`;
    return {
      value: `${m.mean.toFixed(decimals)}<small>${unit}</small>`,
      note: `n=${m.n} · ${range}`,
    };
  };

  const v = fmt(by.voltage_v, 4, ' V');
  const i = fmt(by.current_a, 4, ' A');
  const missing = (rig.coverage && rig.coverage.missing_channels) || [];
  const tempMissing = missing.indexOf('temperature_c') >= 0;
  const t = tempMissing ? null : fmt(by.temperature_c, 1, ' °C');
  const pct = rig.serial ? Math.round(rig.serial.accepted_fraction * 100) : null;
  const dur = by.test_time_s;

  const tile = (label, cell) => `<div class="rt"><div class="rk">${esc(label)}</div>` +
    (cell
      ? `<div class="rv">${cell.value}</div><div class="ru">${esc(cell.note)}</div>`
      : '<div class="rv na">not measured</div><div class="ru">no sensor fitted</div>') +
    '</div>';

  const scored = rig.status === 'SCORED' || rig.status === 'SCORED_WITH_REFUSALS';

  return `
<div class="rig">
  <div class="rhead">
    <span class="rorigin">MEASURED · LIVE HARDWARE</span>
    <span class="rtitle">Bench rig</span>
    <span class="rid">${esc(rig.battery_id)}</span>
    <span class="rstat">${pct !== null ? `${pct}% OF RECORDS ACCEPTED` : esc(rig.status)}</span>
  </div>
  <div class="rgrid">
    ${tile('Terminal voltage', v)}
    ${tile('Current, signed', i)}
    ${tile('Cell temperature', t)}
    <div class="rt"><div class="rk">Health index &middot; RUL</div>
      <div class="rv ${scored ? '' : 'na'}">${scored ? 'scored' : 'refused'}</div>
      <div class="ru">${dur ? `over ${dur.maximum.toFixed(0)} s` : ''}</div></div>
  </div>
  <div class="rnote">${scored
    ? 'Coverage is complete, so this capture was scored through the same pipeline as the fleet.'
    : `<b>Not scored.</b> Coverage is incomplete without ${esc(missing.join(', ') || 'a required channel')}, so no health index, RUL or state of health exists for this rig. The measurements above are unaffected — they were made, and they stand.`}</div>
</div>`;
}

/**
 * The bench rig's own capture, rendered beside the dataset fleet.
 *
 * This section exists because a rig can measure some channels honestly and
 * still be unscoreable for want of another. Showing only the refusal would
 * throw away real measurements; showing only the measurements would imply a
 * health score the coverage gate declined to produce. Both appear, labelled.
 *
 * `rig` is null when no capture has been replayed, and the section is then
 * omitted entirely rather than rendered empty.
 */
function rigHtml(rig) {
  if (!rig) return '';

  const scored = rig.status === 'SCORED' || rig.status === 'SCORED_WITH_REFUSALS';
  const accepted = rig.serial
    ? `${rig.serial.n_accepted}/${rig.serial.n_accepted + rig.serial.n_rejected}`
    : `${rig.n_decoded}/${rig.n_frames}`;
  const acceptPct = rig.serial ? Math.round(rig.serial.accepted_fraction * 100) : null;

  const unitFor = { voltage_v: 'V', current_a: 'A', temperature_c: '°C', soc: '%' };
  const labelFor = {
    voltage_v: 'Terminal voltage', current_a: 'Current, signed',
    temperature_c: 'Cell temperature', soc: 'State of charge',
    test_time_s: 'Capture duration',
  };

  const measuredRows = (rig.measured || [])
    .filter((m) => m.channel !== 'test_time_s' && m.channel !== 'soc')
    .map((m) => {
      const u = unitFor[m.channel] || m.unit || '';
      const range = m.minimum === m.maximum
        ? `${m.mean.toFixed(4)} ${u}`
        : `${m.mean.toFixed(4)} ${u} <u>${m.minimum.toFixed(4)} – ${m.maximum.toFixed(4)}</u>`;
      return `<div class="row"><div class="k"><span>${esc(labelFor[m.channel] || m.channel)}</span></div>
        <div class="v good">${range}</div></div>`;
    }).join('');

  const missingRows = ((rig.coverage && rig.coverage.missing_channels) || [])
    .map((c) => `<div class="row"><div class="k"><span>${esc(labelFor[c] || c)}</span></div>
      <div class="v critical">not supplied by this rig</div></div>`).join('');

  const duration = (rig.measured || []).find((m) => m.channel === 'test_time_s');

  return `
<section class="sec">
  <div class="sechead"><div><h2>Bench rig &mdash; ${esc(rig.battery_id)}</h2>
    <div class="sub">Measured on physical hardware over USB serial. Unlike the fleet above, these are
      instrument readings, not dataset replay or simulation.</div></div>
    <div class="right">${esc(rig.status)}</div></div>
  <div class="g12">
    <div class="c7"><div class="kv">
      <div class="row"><div class="k"><span>Records accepted</span></div>
        <div class="v good">${accepted}${acceptPct !== null ? ` <u>${acceptPct}%</u>` : ''}</div></div>
      ${measuredRows}
      ${missingRows}
      ${duration ? `<div class="row"><div class="k"><span>Capture duration</span></div>
        <div class="v">${duration.maximum.toFixed(1)}<u> s</u></div></div>` : ''}
      <div class="row"><div class="k"><span>Health index, RUL, state of health</span></div>
        <div class="v ${scored ? 'good' : 'critical'}">${scored ? 'scored' : 'refused'}</div></div>
    </div></div>
    <div class="c5">
      <div class="steps">
        <div class="step"><span class="i">01</span><span>Every value on the left was measured by an
          instrument on a real cell and carried over a checksummed wire. Nothing here is replayed or
          simulated.</span></div>
        <div class="step"><span class="i">02</span><span>The rig declares its sensor complement in its
          HELLO line. A channel it cannot measure is not declared and not sent, rather than being
          filled with a plausible constant.</span></div>
        <div class="step"><span class="i">03</span><span>${scored
          ? 'Coverage is complete, so the capture was scored through the same pipeline the fleet uses.'
          : 'Coverage is incomplete, so no health index, RUL or state of health is produced for this rig. The measurements above stand; the scores do not exist.'}</span></div>
      </div>
      ${(rig.refusals || []).length
        ? `<div class="caveat">${esc(String(rig.refusals[0]).split(String.fromCharCode(10))[0])}</div>` : ''}
    </div>
  </div>
</section>`;
}

/**
 * Build the shell for one payload.
 *
 * `title` stays "BEACON" so the brand mark matches the prototype exactly.
 */
export function shellHtml(data, title = 'BEACON', rig = null) {
  const { provenance: prov, fleet } = data;

  const need = fleet.n_needing_action;
  const n = fleet.n_batteries || 1;
  // A statement about the fleet this run scored, not about any external
  // service: nothing else is being monitored.
  const share = need / n;
  const sysTone = need === 0 ? 'good' : share > 0.3 ? 'alert' : 'warn';
  const sysWord = need === 0 ? 'nominal' : share > 0.3 ? 'attention' : 'watch';

  const generated = new Date()
    .toISOString()
    .replace('T', ' ')
    .slice(0, 16) + ' UTC';

  const notice = !prov.is_measured
    ? `<div class="notice"><b>Simulated data.</b><span>${esc(prov.warning)}</span></div>`
    : `<div class="notice measured"><b>Measured data.</b><span>Source: ${esc(prov.dataset_label)}. ` +
      `Scores remain rule-based and are not validated against capacity fade &mdash; see Evidence.</span></div>`;

  return `
<nav class="dock" aria-label="Sections">
  <div class="dmark">
    <svg width="17" height="17" viewBox="0 0 24 24" fill="#34d399"><path d="M13 2L4.5 13.5H11L10 22l8.5-11.5H12L13 2z"/></svg>
  </div>
    ${navHtml()}
  <button class="dbtn themebtn" id="themeToggle" aria-label="Toggle light and dark theme">
    <svg class="ic-moon" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.8 6.8 0 0 0 10.5 10.5z"/></svg>
    <svg class="ic-sun" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="4.2"/>
      <path d="M12 2.4v2.3M12 19.3v2.3M4.2 4.2l1.7 1.7M18.1 18.1l1.7 1.7M2.4 12h2.3M19.3 12h2.3M4.2 19.8l1.7-1.7M18.1 5.9l1.7-1.7"/></svg>
    <span class="tt">Theme</span>
  </button>
  <div class="dver">v1.0</div>
</nav>

<div class="shell">
  <header class="hdr">
    <div class="hdrL">
      <div>
        <div class="lbl">ACTIVE BATTERY</div>
        <div class="selwrap">
          <button class="selbtn" id="switchBtn" aria-haspopup="listbox" aria-label="Select battery">
            <span id="switchId">&mdash;</span>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"
              stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
          </button>
          <div class="pop" id="pop" data-open="false">
            <input id="cellSearch" type="search" placeholder="Search battery" aria-label="Search battery" autocomplete="off">
            <div class="poplist"><div class="cells" id="cells" role="group" aria-label="Select a battery"></div></div>
            <div class="popfoot"><span id="packid" class="mono" style="color:var(--ink)">&mdash;</span>
              <span id="packstate" style="display:inline-flex;align-items:center;gap:7px"></span></div>
          </div>
        </div>
        <div class="statechip" id="stateChip"></div>
      </div>
    </div>

    <div class="brand"><h1>${esc(title)}</h1><div class="tag">BATTERY INTELLIGENCE PLATFORM</div></div>

    <div class="hdrR">
      <span class="hpill"><span class="sd ${sysTone} live"></span>
        <span class="st"><b>Fleet ${sysWord}</b><i>${need} of ${fleet.n_batteries} need action</i></span></span>
      <span class="hpill">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
          stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19a4.5 4.5 0 0 0 .5-8.97A6 6 0 0 0 6.2 9.2 4.5 4.5 0 0 0 7 19z"/></svg>
        <span class="st"><b>${esc(prov.dataset_label)}</b><i>generated ${generated}</i></span></span>
    </div>
  </header>

  <main class="wrap">

    <section class="view on" id="view-home">
      <div class="hero">
        <div class="socline">
          <div class="socl" id="socLabel">STATE OF HEALTH</div>
          <div class="socv" id="socValue">&mdash;</div>
          <div class="socna" id="socNote" style="display:none"></div>
        </div>

        <div style="position:relative;width:100%;max-width:880px">
          <button class="navarrow prev" id="prevBat" aria-label="Previous battery">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
              stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg></button>
          <div class="batstage" id="batStage" role="button" tabindex="0"
            aria-label="Open the Digital Twin for this battery"></div>
          <button class="navarrow next" id="nextBat" aria-label="Next battery">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
              stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg></button>
        </div>

        <button class="battap" id="openTwin">
          Tap the battery to open the Digital Twin
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
        </button>

        <div class="metrics" id="kpis"></div>

        <button class="cta" data-view="analytics">View analytics
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"
            stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M6 13l6 6 6-6"/></svg>
        </button>

        ${notice}
        ${rigHomeHtml(rig)}
      </div>
    </section>

    <section class="view" id="view-health"><div id="rowA"></div></section>

    <section class="view" id="view-analytics">
      <section class="sec">
        <div class="sechead"><div><h2>Fleet</h2>
          <div class="sub">Every battery scored by this run.</div></div>
          <div class="right">health index &times; rul (cycles)</div></div>
        <div id="fleetBand"></div>
        <div id="scatterMount"></div>
      </section>
      <section class="sec">
        <div class="sechead"><div><h2>Battery health table</h2>
          <div class="sub">Sort by any column; select a row to scope the battery views to that cell.</div></div></div>
        <div id="fleetGrid"></div>
      </section>
      <div id="rowB"></div>
      <div id="riskMount"></div>
    </section>

    <section class="view" id="view-evidence">${rigHtml(rig)}${evidenceHtml(fleet)}</section>

    <footer class="foot">
      <p>${esc(title)} renders the output of the BEACON pipeline. Every value is computed by the pipeline; no
        placeholder readings are shown. Metrics the input dataset cannot support are rendered as
        unavailable rather than substituted.</p>
      <div class="fm">
        <span>${prov.n_batteries} batteries</span>
        <span>${Number(prov.n_telemetry_rows).toLocaleString('en-US')} rows</span>
        <span>${esc(String(prov.dataset_label).toLowerCase())}</span>
        <span>${generated}</span>
      </div>
    </footer>
  </main>
</div>

<button class="fab" id="fab" aria-label="Open Battery Guardian">
  <span class="badge" id="fabDot"></span>
  <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#7ff0c4" stroke-width="1.5"
    stroke-linecap="round" stroke-linejoin="round">
    <rect x="4" y="7.5" width="16" height="11.5" rx="4.5"/><path d="M12 3.4v4.1M8.4 3.9h7.2"/>
    <circle cx="9.4" cy="13.2" r="1.5" fill="#7ff0c4" stroke="none"/>
    <circle cx="14.6" cy="13.2" r="1.5" fill="#7ff0c4" stroke="none"/></svg>
  <span class="flabel">Battery Guardian</span>
</button>

<aside class="gpanel" id="gpanel" data-open="false" aria-label="Battery Guardian">
  <div class="gph">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="1.9"
      stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h3l2.5-7 4 14 2.5-7h6"/></svg>
    <span class="t">BATTERY GUARDIAN</span>
    <button class="x" id="closePanel" aria-label="Close">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
  </div>
  <div id="guardianMount"></div>
</aside>

<div class="scrim" id="scrim" data-open="false"></div>
<aside class="sheet" id="sheet" data-open="false" aria-label="Digital Twin">
  <div class="shh">
    <div><h2 id="twinTitle">Digital Twin</h2>
      <div class="s">Model-simulated state alongside what the pipeline measured</div></div>
    <button class="x" id="closeSheet" aria-label="Close">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
  </div>
  <div id="twinMount"></div>
</aside>`;
}
