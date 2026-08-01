"""BEACON dashboard renderer: self-contained interactive HTML.

Implements the approved BEACON visual design. Output is a single .html file
with no external requests — fonts fall back to system stacks, charts are
inline SVG drawn from embedded JSON, and interaction is vanilla JS. It opens
from the filesystem with no server, which is the stated V1 constraint.

Rendering is deliberately split from data preparation (`beacon_data.py`):
the mapping from pipeline output to displayed value is where correctness
lives and is unit-tested there, while this module only formats.

Charts are inline SVG rather than the base64 matplotlib PNGs the previous
dashboard embedded. That is not a cosmetic preference: SVG stays crisp at any
zoom, the resulting file is roughly an order of magnitude smaller, and the
series can be re-drawn client-side when the user switches battery, which a
baked PNG cannot do without regenerating the whole page.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.bms.dashboard.beacon_data import BeaconData, build_beacon_data

_NAV_ITEMS = [
    ("dashboard", "Dashboard", "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z"),
    ("health", "Battery health", "M12 3l8 4v6c0 4.5-3.4 7.7-8 8-4.6-.3-8-3.5-8-8V7z"),
    ("usage", "Usage analytics", "M3 17l5-6 4 4 5-7 4 5M3 21h18"),
    ("risk", "Risk monitor", "M12 3l9 16H3zM12 9v5M12 17h.01"),
    ("guardian", "Guardian", "M12 3a9 9 0 019 9v4a5 5 0 01-5 5H8a5 5 0 01-5-5v-4a9 9 0 019-9z"),
    ("actions", "Recommendations", "M9 18h6M10 21h4M12 3a6 6 0 00-4 10.5V16h8v-2.5A6 6 0 0012 3z"),
    ("evidence", "Evidence", "M6 3h9l4 4v14H6zM15 3v4h4"),
]

_CSS = """
:root{
  --bg:#060c17; --bg-2:#0a1424; --panel:#0d1a2d; --panel-2:#112038;
  --line:#1b2f4c; --line-soft:#152744;
  --ink:#e8eefb; --ink-2:#9db0cc; --ink-3:#63789a;
  --blue:#3b82f6; --cyan:#22d3ee; --green:#22c55e;
  --purple:#a855f7; --amber:#f59e0b; --red:#ef4444;
  --r:16px;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:
    radial-gradient(1100px 620px at 12% -8%, rgba(59,130,246,.16), transparent 60%),
    radial-gradient(900px 520px at 88% 4%, rgba(168,85,247,.10), transparent 62%),
    var(--bg);
  color:var(--ink);
  font-family:"Inter","Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif;
  font-size:14px; line-height:1.5; min-height:100vh;
  -webkit-font-smoothing:antialiased;
}
.shell{display:grid;grid-template-columns:264px 1fr;gap:20px;padding:20px;max-width:1560px;margin:0 auto}

/* ---------- sidebar ---------- */
.side{position:sticky;top:20px;align-self:start;display:flex;flex-direction:column;gap:18px}
.brand{display:flex;gap:13px;align-items:flex-start;padding:4px 6px 2px}
.mark{width:46px;height:46px;border-radius:14px;flex:none;display:grid;place-items:center;
  background:linear-gradient(150deg,#2563eb,#22d3ee);box-shadow:0 8px 26px rgba(37,99,235,.42)}
.brandname{font-size:25px;font-weight:800;letter-spacing:.16em;line-height:1.05}
.brandsub{font-size:10.5px;color:var(--ink-3);letter-spacing:.03em;margin-top:5px;max-width:172px}
.nav{display:flex;flex-direction:column;gap:3px}
.nav button{
  display:flex;align-items:center;gap:12px;width:100%;text-align:left;cursor:pointer;
  background:transparent;border:1px solid transparent;border-radius:12px;
  padding:11px 13px;color:var(--ink-2);font:inherit;font-size:13.5px;transition:.16s
}
.nav button:hover{background:var(--panel);color:var(--ink)}
.nav button[aria-current="true"]{
  background:linear-gradient(96deg,rgba(37,99,235,.94),rgba(37,99,235,.55));
  color:#fff;border-color:rgba(96,165,250,.5);box-shadow:0 8px 22px rgba(37,99,235,.3)
}
.nav svg{flex:none;opacity:.92}
.nav button:focus-visible,.chip:focus-visible,.cellbtn:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}

.packcard{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:16px}
.packcard h4{margin:0 0 3px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
.packid{font-size:19px;font-weight:700;letter-spacing:.02em}
.packstate{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--ink-2);margin-top:5px}
.cells{display:flex;flex-direction:column;gap:4px;margin-top:14px;max-height:224px;overflow:auto}
.cellbtn{display:flex;justify-content:space-between;align-items:center;gap:8px;cursor:pointer;
  background:var(--bg-2);border:1px solid var(--line-soft);border-radius:10px;padding:8px 10px;
  color:var(--ink-2);font:inherit;font-size:12.5px;transition:.14s}
.cellbtn:hover{background:var(--panel-2);color:var(--ink);border-color:#274769}
.cellbtn[aria-pressed="true"]{border-color:var(--blue);background:rgba(37,99,235,.16);color:#fff}
.cellbtn .hi{font-variant-numeric:tabular-nums;font-weight:600}

/* ---------- top bar ---------- */
.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:18px}
.pill{display:inline-flex;align-items:center;gap:8px;background:var(--panel);border:1px solid var(--line);
  border-radius:999px;padding:8px 15px;font-size:12.5px;color:var(--ink-2)}
.dot{width:8px;height:8px;border-radius:50%;flex:none}
.dot.good{background:var(--green);box-shadow:0 0 0 4px rgba(34,197,94,.16)}
.dot.warn{background:var(--amber);box-shadow:0 0 0 4px rgba(245,158,11,.16)}
.dot.alert{background:#fb923c;box-shadow:0 0 0 4px rgba(251,146,60,.16)}
.dot.critical{background:var(--red);box-shadow:0 0 0 4px rgba(239,68,68,.18)}
.dot.neutral{background:var(--ink-3)}
.spacer{flex:1}

.banner{display:flex;gap:11px;align-items:flex-start;border-radius:14px;padding:13px 16px;margin-bottom:18px;
  background:rgba(245,158,11,.09);border:1px solid rgba(245,158,11,.36);color:#fde68a;font-size:12.5px}
.banner.measured{background:rgba(34,197,94,.08);border-color:rgba(34,197,94,.32);color:#bbf7d0}
.banner b{color:#fff}

/* ---------- panels ---------- */
.grid{display:grid;gap:16px}
.kpis{grid-template-columns:repeat(4,1fr)}
.row-a{grid-template-columns:1.62fr 1fr}
.row-b{grid-template-columns:1fr 1fr 1fr}
.card{background:linear-gradient(180deg,var(--panel),var(--bg-2));border:1px solid var(--line);
  border-radius:var(--r);padding:18px}
.card h3{margin:0;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-2);font-weight:650}
.card .sub{font-size:11.5px;color:var(--ink-3);margin-top:3px}
.cardhead{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:14px}
.icon{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;flex:none}

.kpi .big{font-size:38px;font-weight:750;line-height:1.05;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kpi .unit{font-size:15px;font-weight:600;color:var(--ink-2);margin-left:4px}
.kpi .qual{font-size:12.5px;margin-top:5px;font-weight:600}
.kpi .foot{font-size:11.5px;color:var(--ink-3);margin-top:9px}
.kpitop{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.na{color:var(--ink-3);font-size:15px;font-weight:600;padding:10px 0 4px}
.na small{display:block;font-weight:400;font-size:11.5px;margin-top:5px;max-width:230px;line-height:1.45}

.good{color:var(--green)} .warn{color:var(--amber)} .alert{color:#fb923c}
.critical{color:var(--red)} .neutral{color:var(--ink-2)}

.guard{display:flex;flex-direction:column;gap:13px}
.bot{display:flex;gap:13px;align-items:flex-start}
.botface{width:62px;height:62px;border-radius:18px;flex:none;display:grid;place-items:center;
  background:radial-gradient(circle at 50% 34%,rgba(34,211,238,.34),rgba(37,99,235,.16));
  border:1px solid rgba(34,211,238,.42);box-shadow:0 0 26px rgba(34,211,238,.22)}
.bubble{background:var(--panel-2);border:1px solid var(--line);border-radius:14px;padding:13px 15px;
  font-size:12.8px;color:var(--ink);line-height:1.62}
.caveat{font-size:11px;color:var(--ink-3);line-height:1.55;border-top:1px dashed var(--line);padding-top:11px}

.rows{display:flex;flex-direction:column;gap:2px}
.r{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:9.5px 0;
  border-bottom:1px solid var(--line-soft);font-size:13px}
.r:last-child{border-bottom:0}
.r .k{color:var(--ink-2);display:flex;align-items:center;gap:9px;min-width:0}
.r .k span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.r .v{font-variant-numeric:tabular-nums;font-weight:600;flex:none}
.chip{border-radius:999px;padding:3.5px 12px;font-size:11.5px;font-weight:650;border:1px solid transparent}
.chip.good{background:rgba(34,197,94,.14);border-color:rgba(34,197,94,.34);color:#86efac}
.chip.warn{background:rgba(245,158,11,.14);border-color:rgba(245,158,11,.34);color:#fcd34d}
.chip.alert{background:rgba(251,146,60,.14);border-color:rgba(251,146,60,.34);color:#fdba74}
.chip.critical{background:rgba(239,68,68,.14);border-color:rgba(239,68,68,.36);color:#fca5a5}
.chip.neutral{background:rgba(148,163,184,.12);border-color:rgba(148,163,184,.28);color:var(--ink-2)}

.bars{display:flex;flex-direction:column;gap:11px;margin-top:4px}
.bar{display:grid;grid-template-columns:132px 1fr 58px;align-items:center;gap:11px;font-size:12.3px}
.bar .track{height:8px;border-radius:999px;background:var(--line-soft);position:relative;overflow:hidden}
.bar .fill{position:absolute;top:0;bottom:0;border-radius:999px}
.bar .amt{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink-2)}
.zero{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--ink-3);opacity:.55}

.legend{display:flex;flex-direction:column;gap:8px;font-size:12.3px}
.legend div{display:flex;align-items:center;gap:9px;justify-content:space-between}
.legend .lbl{display:flex;align-items:center;gap:9px;color:var(--ink-2)}
.sw{width:9px;height:9px;border-radius:3px;flex:none}
.donutwrap{display:grid;grid-template-columns:132px 1fr;gap:16px;align-items:center}

.recs{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}
.rec{flex:1 1 216px;background:var(--panel-2);border:1px solid var(--line);border-radius:13px;
  padding:13px 15px;font-size:12.5px;color:var(--ink);line-height:1.55;display:flex;gap:11px;align-items:flex-start}
.rec .idx{width:22px;height:22px;border-radius:7px;flex:none;display:grid;place-items:center;
  background:rgba(34,197,94,.16);color:#86efac;font-size:11px;font-weight:700}

.foot{color:var(--ink-3);font-size:11.5px;margin:22px 4px 8px;line-height:1.7}
.foot code{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:11px}

@media (max-width:1240px){
  .kpis{grid-template-columns:repeat(2,1fr)} .row-a,.row-b{grid-template-columns:1fr}
}
@media (max-width:900px){
  .shell{grid-template-columns:1fr;padding:14px} .side{position:static} .kpis{grid-template-columns:1fr}
  .bar{grid-template-columns:104px 1fr 50px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

_JS = r"""
const D = window.__BEACON__;
let current = D.batteries[0];

const $ = (s, r=document) => r.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => (
  {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const toneClass = t => ["good","warn","alert","critical"].includes(t) ? t : "neutral";

/* ---- inline SVG charts ---------------------------------------------- */
function areaChart(series, color, h=190){
  if(!series || series.length < 2){
    return `<div class="na" style="height:${h}px;display:grid;place-items:center;text-align:center">
      No per-cycle series<small>This view needs cycle-indexed telemetry. Re-run the pipeline
      with <code>--data</code> pointing at a dataset that carries per-cycle records.</small></div>`;
  }
  const w = 720, pad = {t:14, r:16, b:26, l:44};
  const min = Math.min(...series), max = Math.max(...series);
  const span = (max - min) || 1;
  const lo = min - span*0.12, hi = max + span*0.12, range = hi - lo;
  const x = i => pad.l + (i/(series.length-1)) * (w - pad.l - pad.r);
  const y = v => pad.t + (1 - (v-lo)/range) * (h - pad.t - pad.b);

  const line = series.map((v,i)=>`${i?"L":"M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const area = `${line}L${x(series.length-1).toFixed(1)},${(h-pad.b).toFixed(1)}L${pad.l},${(h-pad.b).toFixed(1)}Z`;

  let ticks = "";
  for(let i=0;i<=3;i++){
    const v = lo + (range*i/3), yy = y(v);
    ticks += `<line x1="${pad.l}" x2="${w-pad.r}" y1="${yy.toFixed(1)}" y2="${yy.toFixed(1)}"
      stroke="#152744" stroke-width="1"/>
      <text x="${pad.l-9}" y="${(yy+4).toFixed(1)}" fill="#63789a" font-size="10.5"
      text-anchor="end">${v.toFixed(v>=100?0:2)}</text>`;
  }
  const gid = "g"+Math.random().toString(36).slice(2,8);
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" role="img"
    aria-label="Series of ${series.length} points, from ${min.toFixed(2)} to ${max.toFixed(2)}">
    <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${color}" stop-opacity=".42"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>
    ${ticks}
    <path d="${area}" fill="url(#${gid})"/>
    <path d="${line}" fill="none" stroke="${color}" stroke-width="2.1"
      stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${x(series.length-1).toFixed(1)}" cy="${y(series[series.length-1]).toFixed(1)}"
      r="4.2" fill="${color}" stroke="#0a1424" stroke-width="2.4"/>
  </svg>`;
}

function sparkline(series, color){
  if(!series || series.length < 2) return `<div style="height:34px"></div>`;
  const w=118,h=34,min=Math.min(...series),max=Math.max(...series),span=(max-min)||1;
  const pts = series.map((v,i)=>
    `${(i/(series.length-1)*w).toFixed(1)},${(h-2-((v-min)/span)*(h-6)).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
    <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.9"
      stroke-linejoin="round" stroke-linecap="round" opacity=".92"/></svg>`;
}

function gauge(value, color){
  const v = Math.max(0, Math.min(100, value));
  const r=52, cx=64, cy=68, tau=Math.PI;
  const a = tau - (v/100)*tau;
  const pt = (ang,rad)=>[cx+rad*Math.cos(ang), cy-rad*Math.sin(ang)];
  const [sx,sy]=pt(tau,r), [ex,ey]=pt(0,r), [nx,ny]=pt(a,r);
  const [hx,hy]=pt(a, r-13);
  return `<svg viewBox="0 0 128 84" width="128" height="84" role="img"
      aria-label="Gauge at ${v.toFixed(0)} of 100">
    <path d="M${sx},${sy} A${r},${r} 0 0 1 ${ex},${ey}" fill="none" stroke="#152744"
      stroke-width="9" stroke-linecap="round"/>
    <path d="M${sx},${sy} A${r},${r} 0 ${v>50?1:0} 1 ${nx},${ny}" fill="none" stroke="${color}"
      stroke-width="9" stroke-linecap="round"/>
    <line x1="${cx}" y1="${cy}" x2="${hx.toFixed(1)}" y2="${hy.toFixed(1)}" stroke="#e8eefb"
      stroke-width="2.4" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="4.2" fill="#e8eefb"/></svg>`;
}

function donut(items){
  if(!items || !items.length) return "";
  const total = items.reduce((s,i)=>s+i.share,0) || 1;
  const r=52, c=2*Math.PI*r; let off=0, segs="";
  items.forEach(it=>{
    const frac = it.share/total, len = frac*c;
    segs += `<circle cx="64" cy="64" r="${r}" fill="none" stroke="${it.color}" stroke-width="19"
      stroke-dasharray="${len.toFixed(2)} ${(c-len).toFixed(2)}"
      stroke-dashoffset="${(-off).toFixed(2)}" transform="rotate(-90 64 64)"/>`;
    off += len;
  });
  return `<svg viewBox="0 0 128 128" width="132" height="132" role="img"
      aria-label="Share of telemetry rows carrying each behaviour flag">
    <circle cx="64" cy="64" r="${r}" fill="none" stroke="#152744" stroke-width="19"/>
    ${segs}
    <text x="64" y="60" text-anchor="middle" fill="#9db0cc" font-size="10">flagged</text>
    <text x="64" y="77" text-anchor="middle" fill="#e8eefb" font-size="18" font-weight="700"
      >${(100-(items.find(i=>i.label==="Nominal")?.share ?? 0)).toFixed(0)}%</text></svg>`;
}

/* ---- rendering ------------------------------------------------------- */
function kpi(label, iconPath, iconColor, body){
  return `<section class="card kpi">
    <div class="kpitop">
      <div><h3>${esc(label)}</h3></div>
      <div class="icon" style="background:${iconColor}22;border:1px solid ${iconColor}44">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="${iconColor}"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${iconPath}"/></svg>
      </div>
    </div>${body}</section>`;
}

function renderKPIs(b){
  const soh = b.soh_available
    ? `<div class="big">${b.soh_latest.toFixed(1)}<span class="unit">%</span></div>
       <div class="qual ${b.soh_latest>=90?"good":b.soh_latest>=80?"warn":"critical"}">
         ${b.soh_latest>=90?"Excellent":b.soh_latest>=80?"Serviceable":"Below end-of-life threshold"}</div>
       <div class="foot">Measured capacity vs this cell's own initial capacity</div>
       ${sparkline(b.soh_series, "#22c55e")}`
    : `<div class="na">Not available<small>State of health needs measured per-cycle capacity.
       This dataset has none, so no value is shown rather than a substitute.</small></div>`;

  const risk = b.risk_score === null
    ? `<div class="na">Not available<small>Risk scoring did not run for this battery.</small></div>`
    : `<div class="big">${b.risk_score.toFixed(0)}<span class="unit">/100</span></div>
       <div class="qual ${toneClass(b.risk_tone)}">${esc(b.risk_level)}</div>
       <div class="foot">Rule-based degradation risk</div>${sparkline(b.stress_series,"#38bdf8")}`;

  return [
    kpi("State of health","M12 3l8 4v6c0 4.5-3.4 7.7-8 8-4.6-.3-8-3.5-8-8V7z","#22c55e",soh),
    kpi("Degradation risk","M12 3l9 16H3zM12 9v5M12 17h.01","#38bdf8",risk),
    kpi("Remaining useful life","M6 2h12M6 22h12M8 2v4l4 4 4-4V2M8 22v-4l4-4 4 4v4","#a855f7",
      `<div class="big">${b.rul_cycles.toLocaleString()}</div>
       <div class="qual neutral">cycles</div>
       <div class="foot">Policy: ${esc(b.replacement_policy||"n/a")}</div>`),
    kpi("Health index","M12 3l8 4v6c0 4.5-3.4 7.7-8 8-4.6-.3-8-3.5-8-8V7zM9 12l2 2 4-4","#f59e0b",
      `<div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
        <div><div class="big">${b.health_index.toFixed(0)}<span class="unit">/100</span></div>
        <div class="qual ${toneClass(b.state_tone)}">${esc(b.state)}</div></div>
        ${gauge(b.health_index, b.state_tone==="good"?"#22c55e":b.state_tone==="warn"?"#f59e0b":
          b.state_tone==="alert"?"#fb923c":"#ef4444")}</div>
      <div class="foot">Higher means more consumed life. Severity score, not a fade measurement.</div>`),
  ].join("");
}

function renderTrend(b){
  const hasSOH = b.soh_available && b.soh_series.length>1;
  const series = hasSOH ? b.soh_series : b.temp_series;
  const color  = hasSOH ? "#22c55e" : "#38bdf8";
  const title  = hasSOH ? "State of health over cycles" : "Rolling mean temperature over cycles";
  const sub    = hasSOH
    ? "Measured capacity relative to this cell's initial capacity."
    : "No measured capacity in this dataset, so the thermal exposure driving the score is shown instead.";
  return `<section class="card">
    <div class="cardhead"><div><h3>${esc(title)}</h3><div class="sub">${esc(sub)}</div></div>
      <span class="chip neutral">${esc(current.id)}</span></div>
    ${areaChart(series, color)}</section>`;
}

function renderGuardian(b){
  return `<section class="card">
    <div class="cardhead"><div><h3>Guardian</h3>
      <div class="sub">Cause attribution for the health index</div></div>
      <div class="icon" style="background:#22d3ee22;border:1px solid #22d3ee44">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="2"
          stroke-linecap="round"><path d="M12 3a9 9 0 019 9v4a5 5 0 01-5 5H8a5 5 0 01-5-5v-4a9 9 0 019-9z"/>
          <circle cx="9" cy="13" r="1.3" fill="#22d3ee"/><circle cx="15" cy="13" r="1.3" fill="#22d3ee"/></svg></div></div>
    <div class="guard">
      <div class="bot"><div class="botface">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#67e8f9" stroke-width="1.6"
          stroke-linecap="round"><rect x="4" y="7" width="16" height="12" rx="4"/>
          <path d="M12 3v4M8 3.5h8"/><circle cx="9.5" cy="13" r="1.5" fill="#67e8f9"/>
          <circle cx="14.5" cy="13" r="1.5" fill="#67e8f9"/></svg></div>
        <div class="bubble">${esc(b.guardian_report)}</div></div>
      <div>
        <h3 style="margin-bottom:10px">Contribution to health index</h3>
        ${attributionBars(b.health_attribution)}</div>
      <div class="caveat">${esc(b.guardian_caveat)}</div>
    </div></section>`;
}

function attributionBars(rows){
  if(!rows || !rows.length) return `<div class="sub">No attribution available.</div>`;
  const max = Math.max(...rows.map(r=>Math.abs(r.value)), 1);
  return `<div class="bars">` + rows.map(r=>{
    const frac = Math.abs(r.value)/max, pos = r.value >= 0;
    const color = pos ? "#fb923c" : "#22c55e";
    const half = 50;
    const width = (frac*half).toFixed(1);
    const left = pos ? half : (half - frac*half);
    return `<div class="bar">
      <div style="color:var(--ink-2)">${esc(r.label)}</div>
      <div class="track"><span class="zero" style="left:50%"></span>
        <span class="fill" style="left:${left}%;width:${width}%;background:${color}"></span></div>
      <div class="amt" style="color:${color}">${pos?"+":""}${r.value.toFixed(1)}</div></div>`;
  }).join("") + `</div>
  <div class="sub" style="margin-top:11px">Points above (orange) or below (green) the fleet average.
    These sum exactly to this battery's health index minus the fleet mean.</div>`;
}

function renderUsage(b){
  const body = (b.usage && b.usage.length)
    ? `<div class="donutwrap">${donut(b.usage)}
        <div class="legend">${b.usage.map(u=>
          `<div><span class="lbl"><span class="sw" style="background:${u.color}"></span>${esc(u.label)}</span>
           <b>${u.share.toFixed(1)}%</b></div>`).join("")}</div></div>
       <div class="sub" style="margin-top:12px">Share of telemetry rows carrying each flag. Flags overlap,
         so these are incidence rates, not slices of a whole.</div>`
    : `<div class="na">Not available<small>Needs the row-level feature table.</small></div>`;
  return `<section class="card"><div class="cardhead"><div><h3>Usage analytics</h3>
    <div class="sub">Behaviour flag incidence</div></div></div>${body}</section>`;
}

function renderRisk(b){
  const rows = (b.risk_attribution||[]).map(r=>{
    const tone = r.value > 2 ? "alert" : r.value > 0 ? "warn" : "good";
    return `<div class="r"><div class="k"><span class="dot ${tone}"></span><span>${esc(r.label)}</span></div>
      <div class="v"><span class="chip ${tone}">${r.value>=0?"+":""}${r.value.toFixed(1)}</span></div></div>`;
  }).join("");
  return `<section class="card"><div class="cardhead"><div><h3>Risk assessment</h3>
    <div class="sub">Per-term contribution to the risk score</div></div>
    <span class="chip ${toneClass(b.risk_tone)}">${esc(b.risk_level||"n/a")}</span></div>
    <div class="rows">${rows || '<div class="sub">No attribution available.</div>'}</div></section>`;
}

function renderParams(b){
  const rows = (b.parameters||[]).map(p=>
    `<div class="r"><div class="k"><span>${esc(p.label)}</span></div>
     <div class="v">${esc(p.value)}<span style="color:var(--ink-3);font-weight:400"> ${esc(p.unit)}</span></div></div>`
  ).join("");
  return `<section class="card"><div class="cardhead"><div><h3>Battery parameters</h3>
    <div class="sub">Aggregated over this cell's telemetry</div></div></div>
    <div class="rows">${rows}</div></section>`;
}

function renderRecs(b){
  const items = [b.targeted_action, b.recommendation,
    `Replacement policy: ${b.replacement_policy}. Estimated remaining life ${b.rul_cycles.toLocaleString()} cycles.`]
    .filter(Boolean);
  return `<section class="card"><div class="cardhead"><div><h3>Recommendations</h3>
    <div class="sub">Targeted at the dominant contributor: ${esc(b.dominant_cause)}</div></div></div>
    <div class="recs">${items.map((t,i)=>
      `<div class="rec"><span class="idx">${i+1}</span><span>${esc(t)}</span></div>`).join("")}</div></section>`;
}

function render(){
  const b = current;
  $("#kpis").innerHTML = renderKPIs(b);
  $("#rowA").innerHTML = renderTrend(b) + renderGuardian(b);
  $("#rowB").innerHTML = renderUsage(b) + renderRisk(b) + renderParams(b);
  $("#recs").innerHTML = renderRecs(b);
  $("#packid").textContent = b.id;
  $("#packstate").innerHTML =
    `<span class="dot ${toneClass(b.state_tone)}"></span>${esc(b.state)} &middot; health index ${b.health_index.toFixed(0)}`;
  document.querySelectorAll(".cellbtn").forEach(el =>
    el.setAttribute("aria-pressed", String(el.dataset.id === b.id)));
}

function boot(){
  $("#cells").innerHTML = D.batteries.map(b =>
    `<button class="cellbtn" data-id="${esc(b.id)}" aria-pressed="false">
      <span>${esc(b.id)}</span>
      <span class="hi ${toneClass(b.state_tone)}">${b.health_index.toFixed(0)}</span></button>`).join("");
  $("#cells").addEventListener("click", e => {
    const btn = e.target.closest(".cellbtn"); if(!btn) return;
    current = D.batteries.find(x => x.id === btn.dataset.id) || current;
    render();
  });
  document.querySelectorAll(".nav button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav button").forEach(b2 => b2.setAttribute("aria-current","false"));
      btn.setAttribute("aria-current","true");
      const target = document.getElementById(btn.dataset.target);
      if(target) target.scrollIntoView({behavior:"smooth", block:"start"});
    });
  });
  render();
}
document.addEventListener("DOMContentLoaded", boot);
"""


def _nav_html() -> str:
    targets = {
        "dashboard": "kpis", "health": "rowA", "usage": "rowB", "risk": "rowB",
        "guardian": "rowA", "actions": "recs", "evidence": "evidence",
    }
    out = []
    for i, (key, label, path) in enumerate(_NAV_ITEMS):
        out.append(
            f'<button data-target="{targets[key]}" aria-current="{"true" if i == 0 else "false"}">'
            f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="{path}"/></svg>{label}</button>'
        )
    return "\n".join(out)


def render_beacon_html(data: BeaconData, title: str = "BEACON") -> str:
    prov = data.provenance
    fleet = data.fleet
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    banner = (
        f'<div class="banner"><b>Simulated data.</b>&nbsp;{prov["warning"]}</div>'
        if not prov["is_measured"]
        else f'<div class="banner measured"><b>Measured data.</b>&nbsp;Source: '
             f'{prov["dataset_label"]}. Scores remain rule-based and are not '
             f'validated against capacity fade — see the evidence panel below.</div>'
    )

    state_chips = " ".join(
        f'<span class="chip {("good" if s == "HEALTHY" else "warn" if s == "WARNING" else "alert" if s == "DEGRADED" else "critical")}">'
        f'{s.title()} {n}</span>'
        for s, n in fleet["state_counts"].items() if n
    )

    evidence = f"""
<section class="card" id="evidence" style="margin-top:16px">
  <div class="cardhead"><div><h3>Evidence and limitations</h3>
  <div class="sub">What these numbers are validated to support, and what they are not</div></div></div>
  <div class="rows">
    <div class="r"><div class="k"><span>Health index vs measured NASA fade (n=33)</span></div>
      <div class="v"><span class="chip critical">rho = -0.27, p = 0.12</span></div></div>
    <div class="r"><div class="k"><span>Fitted v2 model, unseen cell in a known protocol</span></div>
      <div class="v"><span class="chip good">rho = 0.84, p &lt; 0.001</span></div></div>
    <div class="r"><div class="k"><span>Fitted v2 model, unseen protocol</span></div>
      <div class="v"><span class="chip critical">rho = -0.30, p = 0.10</span></div></div>
    <div class="r"><div class="k"><span>Risk score points from terms constant across the NASA fleet</span></div>
      <div class="v"><span class="chip warn">61%</span></div></div>
    <div class="r"><div class="k"><span>Distinct health index values across this fleet</span></div>
      <div class="v">{fleet["distinct_health_values"]} of {fleet["n_batteries"]}</div></div>
    <div class="r"><div class="k"><span>Guardian attribution against its own score</span></div>
      <div class="v"><span class="chip good">exact</span></div></div>
  </div>
  <div class="sub" style="margin-top:13px;line-height:1.65">
    Attribution is exact with respect to the score it decomposes, and the score is not
    a validated predictor of capacity fade. Treat these outputs as transparent triage,
    not as a measurement. Full derivations: <code>docs/final_report.md</code>.
  </div>
</section>"""

    payload = json.dumps(
        {"provenance": prov, "fleet": fleet, "batteries": data.batteries},
        separators=(",", ":"),
    ).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Behaviour-aware EV battery analytics</title>
<style>{_CSS}</style>
</head>
<body>
<div class="shell">
  <aside class="side">
    <div class="brand">
      <div class="mark">
        <svg width="23" height="23" viewBox="0 0 24 24" fill="#fff"><path d="M13 2L4.5 13.5H11L10 22l8.5-11.5H12L13 2z"/></svg>
      </div>
      <div>
        <div class="brandname">BEACON</div>
        <div class="brandsub">Behaviour-aware EV analytics and condition optimisation network</div>
      </div>
    </div>
    <nav class="nav" aria-label="Sections">{_nav_html()}</nav>
    <div class="packcard">
      <h4>Battery pack</h4>
      <div class="packid" id="packid">—</div>
      <div class="packstate" id="packstate"></div>
      <div class="cells" id="cells" role="group" aria-label="Select a battery"></div>
    </div>
  </aside>

  <main>
    <div class="top">
      <span class="pill"><span class="dot {"critical" if fleet["n_needing_action"] else "good"}"></span>
        Fleet status: {fleet["n_needing_action"]} of {fleet["n_batteries"]} need action</span>
      {state_chips}
      <span class="spacer"></span>
      <span class="pill">Generated {generated}</span>
    </div>
    {banner}
    <div class="grid kpis" id="kpis"></div>
    <div class="grid row-a" id="rowA" style="margin-top:16px"></div>
    <div class="grid row-b" id="rowB" style="margin-top:16px"></div>
    <div id="recs" style="margin-top:16px"></div>
    {evidence}
    <p class="foot">
      BEACON renders the output of <code>main.py</code>. Every value is computed by the pipeline;
      no placeholder readings are shown. Metrics the input dataset cannot support are
      rendered as unavailable rather than substituted.<br>
      {prov["n_batteries"]} batteries &middot; {prov["n_telemetry_rows"]:,} telemetry rows &middot;
      source: {prov["dataset_label"]}
    </p>
  </main>
</div>
<script>window.__BEACON__ = {payload};</script>
<script>{_JS}</script>
</body>
</html>"""


def build_beacon_dashboard(
    guardian: pd.DataFrame,
    output_path: str | Path = "dashboard.html",
    telemetry: pd.DataFrame | None = None,
    data_source: str = "simulated",
    dataset_label: str = "Synthetic fleet telemetry",
) -> Path:
    """Build the BEACON dashboard and write it to `output_path`."""
    data = build_beacon_data(
        guardian, telemetry=telemetry, data_source=data_source, dataset_label=dataset_label
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_beacon_html(data), encoding="utf-8")
    return path
