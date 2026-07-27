const STATE_COLOR_VAR = {
  NORMAL: '--normal',
  MODERATE_RISK: '--moderate',
  HIGH_RISK: '--high',
  FAILURE_IMMINENT: '--failure',
};

export function FleetTable({ batteries, selectedId, onSelect }) {
  if (!batteries.length) {
    return (
      <div className="empty-state">
        <p>No batteries tracked yet. Run a simulation to populate the fleet.</p>
      </div>
    );
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Battery</th>
          <th>Twin State</th>
          <th>Health Idx</th>
          <th>RUL (cycles)</th>
          <th>Policy</th>
        </tr>
      </thead>
      <tbody>
        {batteries.map((b) => (
          <tr
            key={b.battery_id}
            className={b.battery_id === selectedId ? 'selected' : ''}
            onClick={() => onSelect(b.battery_id)}
          >
            <td className="bid">{b.battery_id}</td>
            <td>
              <span className={`badge ${b.twin_state}`}>{b.twin_state.replace('_', ' ')}</span>
            </td>
            <td>
              <span className="health-bar">
                <div
                  style={{
                    width: `${b.health_index}%`,
                    background: `var(${STATE_COLOR_VAR[b.twin_state]})`,
                  }}
                />
              </span>{' '}
              {b.health_index.toFixed(0)}
            </td>
            <td>{b.rul_cycles.toLocaleString()}</td>
            <td>{b.replacement_policy}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
