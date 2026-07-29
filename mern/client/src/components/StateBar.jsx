import { STATE_ORDER } from './Sparkline.jsx';

export function StateBar({ batteries }) {
  const counts = { NORMAL: 0, MODERATE_RISK: 0, HIGH_RISK: 0, FAILURE_IMMINENT: 0 };
  batteries.forEach((b) => {
    counts[b.twin_state] = (counts[b.twin_state] || 0) + 1;
  });
  return (
    <div className="state-bar">
      {STATE_ORDER.map((s) => (
        <div key={s} className={`state-chip ${s}`}>
          <span>{s.replace('_', ' ')}</span>
          <span className="n">{counts[s]}</span>
        </div>
      ))}
    </div>
  );
}
