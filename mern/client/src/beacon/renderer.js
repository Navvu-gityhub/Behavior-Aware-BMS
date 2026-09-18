/* eslint-disable */
/**
 * BEACON dashboard renderer - VERBATIM from the design prototype.
 *
 * Source: redesign/beacon.js in the dashboard redesign bundle, reproduced
 * unchanged except for the two lines marked PORT below. Do not restyle or
 * "improve" anything here: this file is the design, and the React layer exists
 * only to fetch the payload and mount it.
 *
 * PORT 1: `D` and `current` are bound inside boot() rather than at module
 *         scope. In the original this file was a classic script appended
 *         after the payload, so window.__BEACON__ already existed when the
 *         module body ran. As an ES module it is imported first and the
 *         payload arrives later, so reading it at module scope threw
 *         `Cannot read properties of undefined` and blanked the page.
 * PORT 2: the DOMContentLoaded listener is replaced by a named export, because
 *         in a SPA the document has long since loaded by the time this mounts.
 */

let D;  /* PORT 1: bound in boot() */
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => (
  {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const toneClass = t => ["good","warn","alert","critical"].includes(t) ? t : "neutral";
const TONE = {good:"#34d399", warn:"#fbbf24", alert:"#fb923c", critical:"#f87171", neutral:"#93a0b7"};
const toneHex = t => TONE[toneClass(t)];
const ACC = "#34d399";
const num = (v,d=0) => Number(v).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});

let current;  /* PORT 1: bound in boot() */
let sortKey = "health_index", sortDir = -1;
let uid = 0;
const CHARTS = {};

/* pipeline parameters arrive pre-formatted; read them back by label so nothing
   here invents a quantity beacon_data.py did not emit. */
function param(b, label){
  const p = (b.parameters||[]).find(x => x.label === label);
  if(!p) return null;
  const n = parseFloat(String(p.value).replace(/,/g,""));
  return isNaN(n) ? null : {n, unit:p.unit || "", text:p.value};
}

/* ===================================================================
   hero battery
   =================================================================== */
function batterySVG(pct, tone, label){
  const col = toneHex(tone);
  const W=760, H=340;
  const bx=146, bw=470, by=68, bh=204;          // glass body
  const wx=bx+16, wy=by+16, ww=bw-32, wh=bh-32; // inner window
  const fx=wx+8, fy=wy+8, fw=ww-16, fh=wh-16;   // cell fill
  const known = pct !== null && pct !== undefined;
  const w = known ? Math.max(8, Math.min(fw, fw*(pct/100))) : 0;

  let segs="";
  for(let i=1;i<6;i++){
    const x=fx+(fw/6)*i;
    segs+=`<line x1="${x.toFixed(1)}" y1="${fy+5}" x2="${x.toFixed(1)}" y2="${fy+fh-5}"
      stroke="rgba(0,0,0,.20)" stroke-width="2.5"/>`;
  }

  return `<svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Battery ${esc(current.id)}: ${known?`${esc(label)} ${pct.toFixed(0)} percent`:esc(label)+" not available"}, state ${esc(current.state)}">
  <defs>
    <linearGradient id="mtl" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4a5670"/><stop offset="34%" stop-color="#232c40"/>
      <stop offset="70%" stop-color="#161d2d"/><stop offset="100%" stop-color="#303b52"/></linearGradient>
    <linearGradient id="glass" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3d4a63"/><stop offset="26%" stop-color="#1a2233"/>
      <stop offset="76%" stop-color="#121927"/><stop offset="100%" stop-color="#2b3549"/></linearGradient>
    <linearGradient id="cellf" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${col}" stop-opacity="1"/>
      <stop offset="34%" stop-color="${col}" stop-opacity=".78"/>
      <stop offset="72%" stop-color="${col}" stop-opacity=".86"/>
      <stop offset="100%" stop-color="${col}" stop-opacity="1"/></linearGradient>
    <radialGradient id="halo"><stop offset="0%" stop-color="${col}" stop-opacity=".22"/>
      <stop offset="100%" stop-color="${col}" stop-opacity="0"/></radialGradient>
    <radialGradient id="floor"><stop offset="0%" stop-color="${col}" stop-opacity=".30"/>
      <stop offset="100%" stop-color="${col}" stop-opacity="0"/></radialGradient>
    <clipPath id="fillclip"><rect x="${fx}" y="${fy}" height="${fh}" rx="17" width="${w.toFixed(1)}"/></clipPath>
  </defs>

  <ellipse cx="${W/2}" cy="302" rx="248" ry="24" fill="url(#floor)"/>
  <ellipse cx="${W/2}" cy="172" rx="300" ry="132" fill="url(#halo)" opacity="${known?1:.3}"/>

  <rect x="656" y="150" width="28" height="40" rx="11" fill="url(#mtl)" stroke="rgba(255,255,255,.12)"/>
  <rect x="100" y="112" width="58" height="116" rx="21" fill="url(#mtl)" stroke="rgba(255,255,255,.12)"/>
  <rect x="604" y="112" width="58" height="116" rx="21" fill="url(#mtl)" stroke="rgba(255,255,255,.12)"/>
  <rect x="112" y="126" width="16" height="88" rx="8" fill="rgba(255,255,255,.09)"/>
  <rect x="616" y="126" width="16" height="88" rx="8" fill="rgba(255,255,255,.09)"/>

  <rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="34" fill="url(#glass)" stroke="rgba(255,255,255,.13)"/>
  <rect x="${wx}" y="${wy}" width="${ww}" height="${wh}" rx="24" fill="#080d18" stroke="rgba(0,0,0,.5)"/>

  <g clip-path="url(#fillclip)" class="cellfill">
    <rect x="${fx}" y="${fy}" width="${fw}" height="${fh}" rx="17" fill="url(#cellf)"/>
    ${segs}
    <rect x="${fx}" y="${fy}" width="${fw}" height="30" rx="15" fill="rgba(255,255,255,.26)"/>
    <rect x="${fx}" y="${fy+fh-16}" width="${fw}" height="10" rx="5" fill="rgba(255,255,255,.13)"/>
  </g>
  ${known?`<rect x="${(fx+w-3).toFixed(1)}" y="${fy}" width="3" height="${fh}" rx="1.5" fill="#fff" opacity=".55"/>`:""}

  <rect x="${bx+30}" y="${by+12}" width="${bw-60}" height="17" rx="8.5" fill="rgba(255,255,255,.15)" pointer-events="none"/>
  <rect x="${bx+52}" y="${by+bh-22}" width="${bw-104}" height="8" rx="4" fill="rgba(255,255,255,.06)" pointer-events="none"/>
  <rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="34" fill="none"
    stroke="${col}" stroke-opacity="${known?.30:.10}" stroke-width="1.2" pointer-events="none"/>
</svg>`;
}

/* ===================================================================
   charts
   =================================================================== */
function empty(t, b){ return `<div class="empty"><div class="et">${esc(t)}</div><div class="eb">${b}</div></div>`; }
function sec(title, sub, right, body){
  return `<section class="sec"><div class="sechead">
    <div><h2>${esc(title)}</h2>${sub?`<div class="sub">${sub}</div>`:""}</div>
    ${right?`<div class="right">${right}</div>`:""}</div>${body}</section>`;
}
function statusText(tone, label){ return `<span class="sd ${toneClass(tone)}"></span>${esc(label)}`; }

function lineChart(series, color, o){
  o = Object.assign({h:280, digits:2, unit:""}, o||{});
  if(!series || series.length < 2){
    return `<div class="emptybox" style="min-height:${Math.round(o.h*.62)}px">${empty("No per-cycle series",
      "This view needs cycle-indexed telemetry. Re-run the pipeline with <code>--data</code> pointing at a dataset that carries per-cycle records.")}</div>`;
  }
  const id="c"+(++uid), w=1000, h=o.h, pad={t:14,r:12,b:26,l:46};
  const min=Math.min(...series), max=Math.max(...series), span=(max-min)||1;
  const lo=min-span*.10, hi=max+span*.10, range=hi-lo;
  const X=i=>pad.l+(i/(series.length-1))*(w-pad.l-pad.r);
  const Y=v=>pad.t+(1-(v-lo)/range)*(h-pad.t-pad.b);
  const line=series.map((v,i)=>`${i?"L":"M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  const area=`${line}L${X(series.length-1).toFixed(1)},${(h-pad.b).toFixed(1)}L${pad.l},${(h-pad.b).toFixed(1)}Z`;
  let g="";
  for(let i=0;i<=4;i++){
    const v=lo+range*i/4, y=Y(v);
    g+=`<line x1="${pad.l}" x2="${w-pad.r}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,.055)"/>
      <text x="${pad.l-9}" y="${(y+3.6).toFixed(1)}" fill="#5d6a80" font-size="10.5" text-anchor="end"
        font-family="ui-monospace,Menlo,monospace">${v.toFixed(Math.abs(v)>=100?0:o.digits)}</text>`;
  }
  for(let i=0;i<=5;i++){
    const idx=Math.round((series.length-1)*i/5);
    g+=`<text x="${X(idx).toFixed(1)}" y="${h-8}" fill="#5d6a80" font-size="10.5" text-anchor="middle"
      font-family="ui-monospace,Menlo,monospace">${idx}</text>`;
  }
  CHARTS[id]={series,X,Y,w,h,color,unit:o.unit,digits:o.digits};
  return `<div class="chart" data-chart="${id}"><svg viewBox="0 0 ${w} ${h}" role="img"
      aria-label="Series of ${series.length} points from ${min.toFixed(o.digits)} to ${max.toFixed(o.digits)}">
    <defs><linearGradient id="f${id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${color}" stop-opacity=".16"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>
    ${g}<path d="${area}" fill="url(#f${id})"/>
    <path d="${line}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
    <g class="cx" opacity="0"><line y1="${pad.t}" y2="${h-pad.b}" stroke="rgba(255,255,255,.3)"/>
      <circle r="3.4" fill="${color}" stroke="#050a14" stroke-width="1.8"/></g>
  </svg><div class="tip"></div></div>`;
}

function sparkline(series, color, w=170, h=24){
  if(!series || series.length<2) return "";
  const min=Math.min(...series), max=Math.max(...series), span=(max-min)||1;
  const d=series.map((v,i)=>`${i?"L":"M"}${(i/(series.length-1)*w).toFixed(1)},${(h-1.5-((v-min)/span)*(h-3)).toFixed(1)}`).join("");
  const id="s"+(++uid);
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none" aria-hidden="true">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${color}" stop-opacity=".22"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>
    <path d="${d}L${w},${h}L0,${h}Z" fill="url(#${id})"/>
    <path d="${d}" fill="none" stroke="${color}" stroke-width="1.3" vector-effect="non-scaling-stroke"
      stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}

function scatter(){
  const bs=D.batteries, w=1000, h=330, pad={t:16,r:14,b:38,l:48};
  const xs=bs.map(b=>b.rul_cycles), ys=bs.map(b=>b.health_index);
  const xlo=Math.min(...xs), xhi=Math.max(...xs), ylo=Math.min(...ys), yhi=Math.max(...ys);
  const x0=xlo-((xhi-xlo)||1)*.1, x1=xhi+((xhi-xlo)||1)*.1;
  const y0=ylo-((yhi-ylo)||1)*.14, y1=yhi+((yhi-ylo)||1)*.14;
  const X=v=>pad.l+((v-x0)/(x1-x0))*(w-pad.l-pad.r);
  const Y=v=>pad.t+(1-(v-y0)/(y1-y0))*(h-pad.t-pad.b);
  let g="";
  for(let i=0;i<=4;i++){
    const v=y0+(y1-y0)*i/4, y=Y(v);
    g+=`<line x1="${pad.l}" x2="${w-pad.r}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,.055)"/>
      <text x="${pad.l-9}" y="${(y+3.6).toFixed(1)}" fill="#5d6a80" font-size="10.5" text-anchor="end"
        font-family="ui-monospace,Menlo,monospace">${v.toFixed(0)}</text>`;
  }
  for(let i=0;i<=5;i++){
    const v=x0+(x1-x0)*i/5, x=X(v);
    g+=`<text x="${x.toFixed(1)}" y="${h-8}" fill="#5d6a80" font-size="10.5" text-anchor="middle"
      font-family="ui-monospace,Menlo,monospace">${num(v)}</text>`;
  }
  const pts=bs.map(b=>{
    const sel=b.id===current.id, c=toneHex(b.state_tone);
    return `<g class="dot" data-id="${esc(b.id)}">
      ${sel?`<circle cx="${X(b.rul_cycles).toFixed(1)}" cy="${Y(b.health_index).toFixed(1)}" r="10" fill="none"
        stroke="${ACC}" stroke-width="1"/>`:""}
      <circle cx="${X(b.rul_cycles).toFixed(1)}" cy="${Y(b.health_index).toFixed(1)}" r="${sel?4.8:3.6}"
        fill="${sel?ACC:c}" fill-opacity="${sel?1:.72}"/>
      <title>${esc(b.id)} — index ${b.health_index.toFixed(0)}, RUL ${num(b.rul_cycles)}, ${esc(b.state)}</title></g>`;
  }).join("");
  return `<div class="chart"><svg viewBox="0 0 ${w} ${h}" role="img"
    aria-label="Health index against remaining useful life for every battery">${g}${pts}</svg></div>`;
}

function rankedBars(items){
  const max=Math.max(...items.map(i=>i.value),1);
  return `<div class="ranked">`+items.map(i=>`<div class="rank"><div>${esc(i.label)}</div>
    <div class="track"><i style="width:${(i.value/max*100).toFixed(1)}%;background:${i.color}"></i></div>
    <div class="amt">${i.display}</div></div>`).join("")+`</div>`;
}

function diverging(rows, note){
  if(!rows||!rows.length) return `<div class="na-inline">No attribution available for this battery.</div>`;
  const max=Math.max(...rows.map(r=>Math.abs(r.value)),1);
  return `<div class="diverge">`+rows.map(r=>{
    const pos=r.value>=0, f=Math.abs(r.value)/max, c=pos?"#fb923c":"#34d399";
    return `<div class="dv"><div>${esc(r.label)}</div>
      <div class="track"><span class="zero"></span>
        <i style="left:${pos?50:50-f*50}%;width:${(f*50).toFixed(1)}%;background:${c}"></i></div>
      <div class="amt" style="color:${c}">${pos?"+":""}${r.value.toFixed(1)}</div></div>`;
  }).join("")+`</div><div class="dvkey">
    <span><i style="background:#fb923c"></i>${esc(note[0])}</span>
    <span><i style="background:#34d399"></i>${esc(note[1])}</span></div>`;
}

/* ===================================================================
   hero metrics
   =================================================================== */
const IC = {
  soh:"M12 3l8 4v6c0 4.5-3.4 7.7-8 8-4.6-.3-8-3.5-8-8V7z",
  idx:"M12 3l8 4v6c0 4.5-3.4 7.7-8 8-4.6-.3-8-3.5-8-8V7zM9 12l2 2 4-4",
  temp:"M14 14.8V5a2 2 0 10-4 0v9.8a4 4 0 104 0z",
  rul:"M12 7v5l3 2M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
  risk:"M12 3l9 16H3zM12 9v5M12 17h.01",
  soc:"M13 2L4.5 13.5H11L10 22l8.5-11.5H12L13 2z",
};
function metric(label, icon, color, body){
  return `<div class="metric"><div class="mh">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="1.9"
      stroke-linecap="round" stroke-linejoin="round"><path d="${icon}"/></svg>${esc(label)}</div>${body}</div>`;
}
function naMetric(reason){
  return `<div class="mna">Not available</div><div class="mf">${reason}</div>`;
}

function renderMetrics(b){
  const temp = param(b, "Mean temperature");

  const soc = param(b, "Mean state of charge");

  return [
    metric("STATE OF CHARGE", IC.soc, "#22d3ee", soc
      ? `<div class="mv">${soc.n.toFixed(1)}<u>%</u></div>
         <div class="msp">${sparkline(b.soc_series,"#22d3ee")}</div>
         <div class="mf">mean across this cell's telemetry</div>`
      : naMetric("No <code>soc</code> column in this dataset.")),
    metric("HEALTH INDEX", IC.idx, "#fbbf24",
      `<div class="mv ${toneClass(b.state_tone)}">${b.health_index.toFixed(0)}<u>/100</u></div>
       <div class="msp">${sparkline(b.stress_series,toneHex(b.state_tone))}</div>
       <div class="mf">${esc(b.state.toLowerCase())} &middot; severity, not fade</div>`),
    metric("TEMPERATURE", IC.temp, "#fb923c", temp
      ? `<div class="mv">${temp.n.toFixed(1)}<u>${esc(temp.unit)}</u></div>
         <div class="msp">${sparkline(b.temp_series,"#fb923c")}</div>
         <div class="mf">mean across this cell's telemetry</div>`
      : naMetric("No temperature column in this dataset.")),
    metric("REMAINING LIFE", IC.rul, "#a78bfa",
      `<div class="mv">${num(b.rul_cycles)}<u>cycles</u></div>
       <div class="msp">${sparkline(b.soc_series&&b.soc_series.length?b.soc_series:b.stress_series,"#a78bfa")}</div>
       <div class="mf">policy &middot; ${esc((b.replacement_policy||"n/a").toLowerCase())}</div>`),
    metric("RISK LEVEL", IC.risk, "#22d3ee", b.risk_score==null
      ? naMetric("Risk scoring did not run for this battery.")
      : `<div class="mv ${toneClass(b.risk_tone)}">${esc((b.risk_level||"n/a").toLowerCase())}</div>
         <div class="msp"><div class="rank" style="padding:0;border:0;grid-template-columns:1fr">
           <div class="track"><i style="width:${b.risk_score.toFixed(0)}%;background:${toneHex(b.risk_tone)}"></i></div></div></div>
         <div class="mf">score ${b.risk_score.toFixed(0)} / 100</div>`),
  ].join("");
}

/* ===================================================================
   guardian panel
   =================================================================== */
function renderGuardian(b){
  const tone = toneClass(b.state_tone), col = toneHex(b.state_tone);
  const top = (b.health_attribution||[]).slice(0,4);
  const evid = top.length
    ? top.map(r=>{
        const worse = r.value > 0;
        return `<div><span class="sd ${worse?"alert":"good"}"></span>
          <span>${esc(r.label)} <span style="color:${worse?"#fb923c":"#34d399"};font-family:var(--mono)"
            >${worse?"+":""}${r.value.toFixed(1)}</span> vs fleet mean</span></div>`;
      }).join("")
    : `<div><span class="sd neutral"></span><span>No attribution terms were produced for this battery.</span></div>`;

  const actions = [b.targeted_action, b.recommendation].filter(Boolean);

  return `<div class="gdx">
      <div class="ic" style="background:${col}1f;border:1px solid ${col}44">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="${col}" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 4v6c0 4.5-3.4 7.7-8 8-4.6-.3-8-3.5-8-8V7z"/></svg></div>
      <div><div class="tl ${tone}">Battery ${esc(b.state.toLowerCase())}</div>
        <div class="cf">Health index ${b.health_index.toFixed(0)} / 100 &middot; ${esc(b.id)}</div></div>
    </div>
    <div class="gbar"><i style="width:${Math.min(100,b.health_index).toFixed(0)}%;background:${col}"></i></div>

    <div class="gsec">MAIN EVIDENCE</div>
    <div class="gev">${evid}</div>

    <div class="gsec">DIAGNOSIS</div>
    <div class="gnote">${esc(b.guardian_report) || "No guardian narrative was produced for this battery."}</div>

    <div class="gsec">RECOMMENDATION</div>
    <div id="recs">${actions.length
      ? `<div class="steps">${actions.map((t,i)=>
          `<div class="step" style="padding:9px 0;font-size:12.5px"><span class="i">${String(i+1).padStart(2,"0")}</span>
            <span>${esc(t)}</span></div>`).join("")}
         <div class="step" style="padding:9px 0;font-size:12.5px"><span class="i">${String(actions.length+1).padStart(2,"0")}</span>
           <span>Replacement policy ${esc(b.replacement_policy||"n/a")} &middot; ${num(b.rul_cycles)} cycles remaining.</span></div></div>`
      : `<div class="na-inline">No recommendation was produced.</div>`}</div>

    <div class="gcav">${esc(b.guardian_caveat) ||
      "Attribution is exact with respect to the score it decomposes; the score is not a validated predictor of capacity fade."}</div>
    <button class="gbtn" data-goto="health">Open full analysis
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button>`;
}

/* ===================================================================
   data views
   =================================================================== */
function renderHealth(b){
  const hasSOH = b.soh_available && b.soh_series.length > 1;
  const series = hasSOH ? b.soh_series : b.temp_series;
  const color  = hasSOH ? "#34d399" : "#22d3ee";
  const title  = hasSOH ? "State of health over cycles" : "Rolling mean temperature over cycles";
  const subt   = hasSOH
    ? "Measured capacity relative to this cell's own initial capacity."
    : "This dataset carries no measured capacity, so the thermal exposure driving the score is shown instead.";
  const params = (b.parameters||[]).map(p=>
    `<div class="row"><div class="k"><span>${esc(p.label)}</span></div>
     <div class="v">${esc(p.value)}${p.unit?`<u> ${esc(p.unit)}</u>`:""}</div></div>`).join("");

  return sec(title, subt, `${hasSOH?"measured":"derived"} &middot; ${series.length} points`,
      lineChart(series, color, {h:290, digits:hasSOH?2:1, unit:hasSOH?"%":"\u00b0C"}))
    + `<div class="g12">
        <div class="c7">${sec("Health index attribution",
          "Shapley contribution of each behaviour term to this cell's index.", "",
          diverging(b.health_attribution, ["above fleet mean","below fleet mean"]))}</div>
        <div class="c5">${sec("Aggregate parameters","Averaged over this cell's telemetry.","",
          `<div class="kv">${params || `<div class="na-inline">No aggregate parameters available.</div>`}</div>`)}</div>
      </div>`;
}

function renderFleetLine(){
  const f = D.fleet;
  const st = f.state_counts || {};
  const named = [["HEALTHY","good"],["WARNING","warn"],["DEGRADED","alert"],["CRITICAL","critical"]]
    .filter(([s])=>st[s]).map(([s,t])=>`<span class="${t}">${st[s]}</span> ${s.toLowerCase()}`).join(", ");
  return `<div class="kv" style="margin-bottom:4px"><div class="row">
      <div class="k"><span><b style="color:var(--ink)">${f.n_needing_action}</b> of ${f.n_batteries} need action &mdash; ${named}</span></div>
      <div class="v">mean index ${f.mean_health_index.toFixed(1)} <u>&middot; mean rul ${num(f.mean_rul_cycles)} &middot; ${f.distinct_health_values}/${f.n_batteries} distinct</u></div>
    </div></div>`;
}

const COLS=[{k:"id",t:"Battery"},{k:"health_index",t:"Health index",r:true},{k:"state",t:"State"},
  {k:"risk_score",t:"Risk",r:true},{k:"risk_level",t:"Risk level"},{k:"rul_cycles",t:"RUL",r:true},
  {k:"replacement_policy",t:"Policy"}];

function renderGrid(){
  const rows=D.batteries.slice().sort((a,b)=>{
    const x=a[sortKey], y=b[sortKey];
    if(x===y) return 0;
    if(x==null) return 1;
    if(y==null) return -1;
    return (typeof x==="number"?x-y:String(x).localeCompare(String(y)))*sortDir;
  });
  const maxH=Math.max(...D.batteries.map(b=>b.health_index),1);
  const th=COLS.map(c=>{
    const on=c.k===sortKey;
    return `<th data-sort="${c.k}" class="${c.r?"r":""}"${on?` aria-sort="${sortDir<0?"descending":"ascending"}"`:""}
      >${esc(c.t)}<span class="arw">${on&&sortDir>0?"&#9650;":"&#9660;"}</span></th>`;
  }).join("");
  const tb=rows.map(b=>`<tr data-id="${esc(b.id)}" aria-selected="${b.id===current.id}">
    <td class="id">${esc(b.id)}</td>
    <td class="r"><span class="hbar"><i style="width:${(b.health_index/maxH*100).toFixed(0)}%;background:${toneHex(b.state_tone)}"></i></span><span class="n">${b.health_index.toFixed(1)}</span></td>
    <td><span class="st">${statusText(b.state_tone,b.state.toLowerCase())}</span></td>
    <td class="r n">${b.risk_score==null?"&mdash;":b.risk_score.toFixed(1)}</td>
    <td class="muted">${esc((b.risk_level||"n/a").toLowerCase())}</td>
    <td class="r n">${num(b.rul_cycles)}</td>
    <td class="muted">${esc((b.replacement_policy||"—").toLowerCase())}</td></tr>`).join("");
  return `<div class="tblwrap"><table class="tbl"><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table></div>`;
}

function renderUsage(b){
  const flags=(b.usage||[]).filter(u=>u.label!=="Nominal").slice().sort((x,y)=>y.share-x.share)
    .map(u=>({label:u.label, value:u.share, display:u.share.toFixed(1)+"%", color:u.color}));
  const nom=(b.usage||[]).find(u=>u.label==="Nominal");
  const body = flags.length
    ? rankedBars(flags)+`<div class="caveat">Incidence rates, not a partition: one telemetry row can carry
        several flags at once, so these do not sum to 100%.${nom?` Rows carrying no flag: ${nom.share.toFixed(1)}%.`:""}</div>`
    : empty("Not available","Behaviour flag incidence needs the row-level feature table.");
  return `<div class="g12">
    <div class="c5">${sec("Behaviour flag incidence",`Share of ${esc(b.id)}'s telemetry rows carrying each flag.`,"",body)}</div>
    <div class="c7">${sec("Stress score over cycles","Rolling mean of the composite stress term.","",
      lineChart(b.stress_series,"#22d3ee",{h:250,digits:1}))}</div></div>`;
}

function renderRisk(b){
  const rows=(b.risk_attribution||[]).map(r=>{
    const tone=r.value>2?"alert":r.value>0?"warn":"good";
    return `<div class="row"><div class="k">${statusText(tone,r.label)}</div>
      <div class="v ${tone}">${r.value>=0?"+":""}${r.value.toFixed(1)}</div></div>`;
  }).join("");
  const rc=D.fleet.risk_counts||{};
  const dist=[["LOW","good"],["MEDIUM","warn"],["HIGH","alert"],["CRITICAL","critical"]]
    .map(([l,t])=>({label:l.toLowerCase(), value:rc[l]||0, display:String(rc[l]||0), color:TONE[t]}));
  return `<div class="g12">
    <div class="c7">${sec("Risk score decomposition",`Per-term contribution for ${esc(b.id)}.`,
      `<span>${statusText(b.risk_tone,(b.risk_level||"n/a").toLowerCase())}</span>`,
      `<div class="kv">${rows||`<div class="na-inline">No attribution available.</div>`}</div>`)}</div>
    <div class="c5">${sec("Fleet risk distribution","Batteries per risk level across this run.","",rankedBars(dist))}</div></div>`;
}

function renderTwin(b){
  const t=b.digital_twin;
  const chart=(t&&t.predicted_series&&t.predicted_series.length>1)
    ? lineChart(t.predicted_series,"#a78bfa",{h:240,digits:2})
    : `<div class="emptybox" style="min-height:170px">${empty("Model trajectory not available",
        "This panel reads <code>battery.digital_twin</code>, which <code>src/bms/digital_twin/twin.py</code> does not yet emit into the dashboard payload. Per ADR-0004 no substitute trajectory is drawn.")}</div>`;
  const field=(l,v)=>`<div class="row"><div class="k"><span>${esc(l)}</span></div>
    <div class="v">${v!=null?esc(v):`<span class="na-inline">not available</span>`}</div></div>`;
  const obs=(b.parameters||[]).map(p=>
    `<div class="row"><div class="k"><span>${esc(p.label)}</span></div>
     <div class="v">${esc(p.value)}${p.unit?`<u> ${esc(p.unit)}</u>`:""}</div></div>`).join("");

  return sec("Predicted trajectory",`Model-simulated degradation path for ${esc(b.id)}.`,"",chart)
    + sec("Twin state","Reconciliation between the model and observed telemetry.","",
        `<div class="kv">
          ${field("Model version", t&&t.model_version)}
          ${field("Predicted vs observed divergence", t&&t.divergence)}
          ${field("Simulated cycles ahead", t&&t.horizon_cycles)}
          ${field("Calibration source", t&&t.calibration_source)}
        </div>
        <div class="caveat">Every field states that it is unavailable rather than showing an estimated value.
          Populate <code>battery["digital_twin"]</code> in <code>beacon_data.py</code> to fill this panel.</div>`)
    + sec("Observed state",`What the pipeline actually measured for ${esc(b.id)}.`,"",
        `<div class="kv">
          <div class="row"><div class="k"><span>Health index</span></div><div class="v">${b.health_index.toFixed(1)}<u> /100</u></div></div>
          <div class="row"><div class="k"><span>Battery state</span></div><div class="v">${esc(b.state)}</div></div>
          <div class="row"><div class="k"><span>State of health</span></div><div class="v">${
            b.soh_available?b.soh_latest.toFixed(1)+"<u> %</u>":`<span class="na-inline">not available</span>`}</div></div>
          <div class="row"><div class="k"><span>Remaining useful life</span></div><div class="v">${num(b.rul_cycles)}<u> cycles</u></div></div>
          ${obs}
        </div>`);
}

/* ===================================================================
   interaction
   =================================================================== */
function wireCharts(){
  $$(".chart[data-chart]").forEach(el=>{
    const c=CHARTS[el.dataset.chart]; if(!c || el.dataset.wired) return;
    el.dataset.wired="1";
    const svg=el.querySelector("svg"), cx=el.querySelector(".cx"), tip=el.querySelector(".tip");
    const ln=cx.querySelector("line"), dt=cx.querySelector("circle");
    el.addEventListener("pointermove", e=>{
      const r=svg.getBoundingClientRect(), k=r.width/c.w;
      const t=((e.clientX-r.left)/k-46)/(c.w-58);
      const i=Math.max(0,Math.min(c.series.length-1,Math.round(t*(c.series.length-1))));
      const px=c.X(i), py=c.Y(c.series[i]);
      ln.setAttribute("x1",px); ln.setAttribute("x2",px);
      dt.setAttribute("cx",px); dt.setAttribute("cy",py);
      cx.setAttribute("opacity","1"); tip.classList.add("on");
      tip.style.left=(px*k)+"px"; tip.style.top=(py*k)+"px";
      tip.innerHTML=`<span class="tl">cycle ${i}</span><b>${c.series[i].toFixed(c.digits)}${c.unit?" "+c.unit:""}</b>`;
    });
    el.addEventListener("pointerleave",()=>{cx.setAttribute("opacity","0");tip.classList.remove("on")});
  });
}

function selectBattery(id){
  if(isRig(id)){ current={id:id,__rig:true}; render(); return; }
  const b=D.batteries.find(x=>x.id===id);
  if(!b||b.id===current.id) return;
  current=b; render();
}

/* ================= bench rig =================
 * The rig is a real cell on a bench, and it belongs in the battery picker for
 * the same reason the fleet does. It is deliberately NOT added to D.batteries:
 * every fleet statistic, the scatter, the grid and the mean health index
 * iterate that array, and a member with no health index would either crash
 * them or silently skew them. It gets its own entry and its own render path
 * instead, so the fleet numbers stay exactly what they were.
 */
function rigId(){ return D.rig ? D.rig.battery_id : null; }
function isRig(id){ return !!rigId() && id === rigId(); }

function rigChannel(name){
  return (D.rig && D.rig.measured || []).find(m => m.channel === name) || null;
}

function rigMetric(label, channel, decimals, unit, icon, color){
  const m = rigChannel(channel);
  if(!m){
    return metric(label, icon, color,
      naMetric("No sensor fitted. The rig declares the channels it has and omits the rest."));
  }
  const range = m.minimum === m.maximum ? "constant"
    : `${m.minimum.toFixed(decimals)} to ${m.maximum.toFixed(decimals)}`;
  return metric(label, icon, color,
    `<div class="mv">${m.mean.toFixed(decimals)}<u>${esc(unit)}</u></div>
     <div class="mf">measured &middot; n=${m.n} &middot; ${esc(range)}</div>`);
}

function renderRigView(){
  const r = D.rig;
  const v = rigChannel("voltage_v");
  const missing = (r.coverage && r.coverage.missing_channels) || [];
  const dur = rigChannel("test_time_s");

  $("#socLabel").textContent = v ? "TERMINAL VOLTAGE · MEASURED" : "STATE OF HEALTH";
  $("#socValue").innerHTML = v
    ? `${v.mean.toFixed(4)}<u>V</u>`
    : `<span style="font-size:22px;font-weight:400;color:var(--ink-3)">Not available</span>`;
  $("#socNote").innerHTML = "State of health needs measured per-cycle capacity, which needs a complete channel set. This rig does not supply "
    + `<code>${esc(missing.join(", ") || "every required channel")}</code>, so the cell is drawn empty and no health index, RUL or SOH is produced. The voltage above was measured.`;
  $("#socNote").style.display = "block";

  /* Drawn empty: there is no state of health to fill it with, and filling it
     from the voltage would imply a conversion nobody has validated. */
  $("#batStage").innerHTML = batterySVG(null, "warn", "state of health");

  $("#kpis").innerHTML = [
    rigMetric("TERMINAL VOLTAGE", "voltage_v", 4, "V", IC.soc, "#22d3ee"),
    rigMetric("CURRENT, SIGNED", "current_a", 4, "A", IC.risk, "#22d3ee"),
    rigMetric("TEMPERATURE", "temperature_c", 1, "°C", IC.temp, "#fb923c"),
    metric("HEALTH INDEX", IC.idx, "#fbbf24",
      naMetric("Coverage is incomplete, so no health index was produced.")),
    metric("RECORDS ACCEPTED", IC.soh, "#34d399",
      `<div class="mv good">${r.serial ? Math.round(r.serial.accepted_fraction*100) : 100}<u>%</u></div>
       <div class="mf">${r.serial ? esc(r.serial.summary) : ""}</div>`),
  ].join("");

  const refusal = (r.refusals && r.refusals[0]) ? String(r.refusals[0]).split(String.fromCharCode(10))[0] : "";
  $("#guardianMount").innerHTML = `<section class="sec"><div class="sechead"><div><h2>Guardian</h2>
    <div class="sub">Attribution needs a score to decompose.</div></div></div>
    <div class="caveat">No risk score exists for this rig, so there is nothing to attribute. ${esc(refusal)}</div></section>`;

  $("#rowA").innerHTML = `<section class="sec"><div class="sechead"><div><h2>Bench rig &mdash; ${esc(r.battery_id)}</h2>
    <div class="sub">Physical hardware over USB serial${dur?`, ${dur.maximum.toFixed(0)} s capture`:""}. Instrument readings, not dataset replay.</div></div>
    <div class="right">${esc(r.status)}</div></div>
    <div class="kv">
      <div class="row"><div class="k"><span>Transport</span></div><div class="v">${esc((r.rig||"").split("  ")[0]||"serial")}</div></div>
      <div class="row"><div class="k"><span>Records accepted</span></div><div class="v good">${r.serial?esc(r.serial.summary):"—"}</div></div>
      <div class="row"><div class="k"><span>Channels missing</span></div><div class="v critical">${esc(missing.join(", ")||"none")}</div></div>
      <div class="row"><div class="k"><span>Health index &middot; RUL &middot; SOH</span></div><div class="v critical">refused</div></div>
    </div>
    <div class="caveat">Every value this rig measured is shown. Every value it could not measure is named, not substituted.</div></section>`;

  $("#switchId").textContent = r.battery_id;
  $$(".cellbtn").forEach(el => el.setAttribute("aria-pressed", String(el.dataset.id === r.battery_id)));
  $("#stateChip").innerHTML = `<span class="sd warn"></span><span>measured &middot; not scored</span>`;
}

function stepBattery(dir){
  // Stepping walks the fleet. From the rig it re-enters at the first battery,
  // because the rig has no position in a fleet it is not part of.
  if(current.__rig){ selectBattery(D.batteries[0].id); return; }
  const i=D.batteries.findIndex(x=>x.id===current.id);
  selectBattery(D.batteries[(i+dir+D.batteries.length)%D.batteries.length].id);
}

function render(){
  if(current.__rig){ renderRigView(); return; }
  const b=current, soc=param(b,"Mean state of charge");

  // The hero reads state of health when the dataset supports it, and mean state
  // of charge otherwise. Both are labelled for what they are; neither is a stand-in
  // for the other, and when neither exists the cell is drawn empty (ADR-0004).
  const hero = b.soh_available
    ? {pct:b.soh_latest, label:"STATE OF HEALTH", digits:1}
    : soc ? {pct:soc.n, label:"MEAN STATE OF CHARGE", digits:0} : null;

  $("#socLabel").textContent = hero ? hero.label : "STATE OF HEALTH";
  $("#socValue").innerHTML = hero
    ? `${hero.pct.toFixed(hero.digits)}<u>%</u>`
    : `<span style="font-size:22px;font-weight:400;color:var(--ink-3)">Not available</span>`;
  $("#socNote").innerHTML = hero ? ""
    : "State of health needs measured per-cycle capacity and state of charge needs an <code>soc</code> column. This dataset carries neither, so the cell is drawn empty rather than filled with a substituted value.";
  $("#socNote").style.display = hero ? "none" : "block";

  $("#batStage").innerHTML = batterySVG(hero?hero.pct:null, b.state_tone, hero?hero.label.toLowerCase():"state of health");
  $("#kpis").innerHTML = renderMetrics(b);
  $("#guardianMount").innerHTML = renderGuardian(b);
  $("#rowA").innerHTML = renderHealth(b);
  $("#fleetBand").innerHTML = renderFleetLine();
  $("#scatterMount").innerHTML = scatter();
  $("#fleetGrid").innerHTML = renderGrid();
  $("#rowB").innerHTML = renderUsage(b);
  $("#riskMount").innerHTML = renderRisk(b);
  $("#twinMount").innerHTML = renderTwin(b);

  $("#switchId").textContent = b.id;
  $("#packid").textContent = b.id;
  $("#packstate").innerHTML = statusText(b.state_tone, b.state.toLowerCase());
  $("#stateChip").innerHTML = statusText(b.state_tone, "Pack " + b.state.toLowerCase());
  $("#twinTitle").textContent = b.id + " — Digital Twin";
  $("#fabDot").style.background = toneHex(b.state_tone);

  $$(".cellbtn").forEach(el=>el.setAttribute("aria-pressed", String(el.dataset.id===b.id)));
  wireCharts();
}

const VIEWS = ["home","health","analytics","evidence"];
function goto(v){
  if(!VIEWS.includes(v)) v="home";
  $$(".view").forEach(x=>x.classList.toggle("on", x.id==="view-"+v));
  $$(".dbtn[data-view]").forEach(x=>x.setAttribute("aria-current", String(x.dataset.view===v)));
  if(location.hash.slice(1)!==v) history.replaceState(null,"","#"+v);
  window.scrollTo(0,0);
  wireCharts();
}

function setSheet(open){
  $("#sheet").setAttribute("data-open", String(open));
  $("#scrim").setAttribute("data-open", String(open));
  if(open) wireCharts();
}
function setPanel(open){ $("#gpanel").setAttribute("data-open", String(open)); }
function closePop(){ $("#pop").setAttribute("data-open","false"); }

function boot(){
  /* PORT 1: the payload is read here, not at module scope. */
  D = window.__BEACON__;
  if(!D || !Array.isArray(D.batteries) || !D.batteries.length){
    throw new Error("BEACON: window.__BEACON__ is missing or has no batteries");
  }
  current = D.batteries[0];

  $("#cells").innerHTML = D.batteries.map(b=>
    `<button class="cellbtn" data-id="${esc(b.id)}" aria-pressed="false">
      <span class="sd ${toneClass(b.state_tone)}"></span>
      <span class="cid">${esc(b.id)}</span>
      <span class="cst">${esc(b.state.toLowerCase())}</span>
      <span class="hi">${b.health_index.toFixed(0)}</span></button>`).join("")
    + (D.rig ? `<button class="cellbtn rigcell" data-id="${esc(D.rig.battery_id)}" aria-pressed="false">
      <span class="sd warn"></span>
      <span class="cid">${esc(D.rig.battery_id)}</span>
      <span class="cst">bench rig &middot; measured</span>
      <span class="hi">&mdash;</span></button>` : "");

  document.addEventListener("click", e=>{
    const cell=e.target.closest(".cellbtn");
    if(cell){ selectBattery(cell.dataset.id); closePop(); return; }
    const row=e.target.closest(".tbl tbody tr");
    if(row){ selectBattery(row.dataset.id); return; }
    const pt=e.target.closest(".chart .dot");
    if(pt){ selectBattery(pt.dataset.id); return; }
    const th=e.target.closest(".tbl th[data-sort]");
    if(th){
      const k=th.dataset.sort;
      if(k===sortKey) sortDir=-sortDir;
      else { sortKey=k; sortDir = typeof D.batteries[0][k]==="number" ? -1 : 1; }
      $("#fleetGrid").innerHTML=renderGrid(); return;
    }
    const nav=e.target.closest("[data-view]");
    if(nav){ goto(nav.dataset.view); return; }
    const jump=e.target.closest("[data-goto]");
    if(jump){ setPanel(false); goto(jump.dataset.goto); return; }
    if(e.target.closest("#prevBat")){ stepBattery(-1); return; }
    if(e.target.closest("#nextBat")){ stepBattery(1); return; }
    if(e.target.closest("#batStage") || e.target.closest("#openTwin")){ setSheet(true); return; }
    if(e.target.closest("#closeSheet") || e.target.closest("#scrim")){ setSheet(false); return; }
    if(e.target.closest("#fab")){ setPanel($("#gpanel").getAttribute("data-open")!=="true"); return; }
    if(e.target.closest("#closePanel")){ setPanel(false); return; }
    if(e.target.closest("#switchBtn")){
      const open=$("#pop").getAttribute("data-open")==="true";
      $("#pop").setAttribute("data-open",String(!open));
      if(!open) setTimeout(()=>$("#cellSearch").focus(),30);
      return;
    }
    if(!e.target.closest("#pop")) closePop();
  });

  $("#cellSearch").addEventListener("input", e=>{
    const q=e.target.value.trim().toLowerCase();
    $$("#cells .cellbtn").forEach(el=>{
      el.style.display = !q || el.dataset.id.toLowerCase().includes(q) ? "" : "none";
    });
  });

  document.addEventListener("keydown", e=>{
    if(e.key==="Escape"){ closePop(); setSheet(false); setPanel(false); }
    if(e.target.matches("input")) return;
    if(e.key==="[") stepBattery(-1);
    if(e.key==="]") stepBattery(1);
  });

  window.addEventListener("hashchange",()=>goto(location.hash.slice(1)));
  render();
  goto(location.hash.slice(1)||"home");
}
export { boot };
