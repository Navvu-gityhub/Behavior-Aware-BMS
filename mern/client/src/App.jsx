import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import { api, ApiError } from './api.js';
import { StateBar } from './components/StateBar.jsx';
import { FleetTable } from './components/FleetTable.jsx';
import { BatteryDetail } from './components/BatteryDetail.jsx';

export default function App() {
  const [batteries, setBatteries] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [connected, setConnected] = useState(null); // null = unknown yet
  const [nBatteries, setNBatteries] = useState(12);
  const [simulating, setSimulating] = useState(false);
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);

  const showToast = useCallback((msg) => {
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 5000);
  }, []);

  const selectBattery = useCallback(
    async (id) => {
      setSelectedId(id);
      try {
        const [d, t] = await Promise.all([
          api.getBattery(id),
          api.getTimeline(id).catch(() => []),
        ]);
        setDetail(d);
        setTimeline(t);
      } catch (e) {
        showToast(`Could not load ${id}: ${e.message}`);
      }
    },
    [showToast]
  );

  const loadFleet = useCallback(
    async (preserveSelection) => {
      try {
        const list = await api.listBatteries();
        setConnected(true);
        setBatteries(list);
        if (list.length) {
          const keep = preserveSelection && list.some((b) => b.battery_id === selectedId);
          await selectBattery(keep ? selectedId : list[0].battery_id);
        }
      } catch (e) {
        setConnected(false);
        const reason = e instanceof ApiError ? e.message : e.message;
        showToast(`Could not reach the gateway: ${reason}`);
      }
    },
    [selectBattery, selectedId, showToast]
  );

  useEffect(() => {
    loadFleet(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleSimulate() {
    setSimulating(true);
    try {
      await api.simulate({ n_batteries: nBatteries, rows_per_battery: 300 });
      await loadFleet(true);
    } catch (e) {
      showToast(`Simulation failed: ${e.message}`);
    } finally {
      setSimulating(false);
    }
  }

  return (
    <>
      <header>
        <div className="brand">
          <span className="mark">⚡</span>
          <div>
            <h1>BEACON — Live Fleet (MERN)</h1>
            <div className="tagline">rule-based diagnostics, not a validated predictive model</div>
          </div>
        </div>
        <div className="header-right">
          <div className="live-status">
            <span className={`pulse ${connected ? 'on' : ''}`} />
            <span>
              {connected === null ? 'connecting…' : connected ? `connected · ${batteries.length} tracked` : 'gateway unreachable'}
            </span>
          </div>
          <label style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-dim)' }}>
            batteries{' '}
            <input
              type="number"
              min={1}
              max={200}
              value={nBatteries}
              onChange={(e) => setNBatteries(parseInt(e.target.value, 10) || 1)}
            />
          </label>
          <button onClick={handleSimulate} disabled={simulating}>
            {simulating ? 'Running…' : 'Run Simulation'}
          </button>
        </div>
      </header>

      <StateBar batteries={batteries} />

      <main>
        <div className="panel">
          <div className="panel-title">Fleet — click a battery for detail</div>
          <FleetTable batteries={batteries} selectedId={selectedId} onSelect={selectBattery} />
        </div>

        <div className="panel">
          <div className="panel-title">Battery Detail</div>
          <BatteryDetail detail={detail} timeline={timeline} />
        </div>
      </main>

      <footer>
        Every score here (health_index, risk_score, rul_cycles, failure_likelihood) is a hand-tuned heuristic,
        not validated against measured capacity fade — see docs/final_report.md for what's actually been tested.
        Twin state is a 1:1 relabel of health_index's own thresholds — see docs/digital_twin.md.
        <br />
        This client talks to the Express gateway, which proxies to the unmodified Python pipeline — see docs/mern.md.
      </footer>

      {toast && <div className="toast">{toast}</div>}
    </>
  );
}
