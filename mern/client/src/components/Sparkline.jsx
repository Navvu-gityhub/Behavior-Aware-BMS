const STATE_ORDER = ['NORMAL', 'MODERATE_RISK', 'HIGH_RISK', 'FAILURE_IMMINENT'];

export function Sparkline({ points, dataKey, color, unit = '' }) {
  if (!points.length) {
    return <div style={{ color: 'var(--text-faint)', fontSize: 11, padding: '8px 0' }}>no data</div>;
  }
  const vals = points.map((p) => p[dataKey]);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const pad = (max - min) * 0.1 || 1;
  const lo = min - pad;
  const hi = max + pad;
  const W = 600;
  const H = 56;
  const marginY = 6;
  const xStep = W / Math.max(points.length - 1, 1);
  const yFor = (v) => H - marginY - ((v - lo) / (hi - lo)) * (H - 2 * marginY);
  const path = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${(i * xStep).toFixed(1)} ${yFor(p[dataKey]).toFixed(1)}`)
    .join(' ');
  const last = points[points.length - 1];

  return (
    <>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} className="grid-line" x1="0" x2={W} y1={H * f} y2={H * f} />
        ))}
        <path
          className="trace-line"
          d={path}
          style={{ stroke: color, filter: `drop-shadow(0 0 3px ${color}90)` }}
        />
        <circle cx={W} cy={yFor(last[dataKey])} r="2.5" fill={color} />
      </svg>
      <div className="trace-range">
        <span>min {min.toFixed(1)}{unit}</span>
        <span>latest {last[dataKey].toFixed(1)}{unit}</span>
        <span>max {max.toFixed(1)}{unit}</span>
      </div>
    </>
  );
}

export { STATE_ORDER };
