import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';
import { shellHtml } from './shell.js';
import { boot } from './renderer.js';
import { initTheme, bindThemeToggle } from './theme.js';

/**
 * The bench rig capture the Evidence view reports.
 *
 * A repository path, resolved by the API against its root allowlist
 * (src/bms/api/paths.py) - the browser never names a filesystem location the
 * service has not already permitted.
 */
const RIG_CAPTURE = 'data/interim/rig_stage_b_voltage_verified.txt';
import './beacon.css';
import './gate.css';

/**
 * Mounts the BEACON dashboard.
 *
 * React's job here is narrow and deliberately so: fetch the payload, write the
 * shell, hand control to the prototype's own renderer. It does not re-implement
 * any part of the design, and it does not touch the subtree afterwards - the
 * renderer owns those nodes and rewrites them imperatively, so a second writer
 * would fight it.
 *
 * `dangerouslySetInnerHTML` is the right tool for exactly that reason. The
 * markup is a module constant in this repository, not user input, and the
 * payload it interpolates is escaped in `shell.js`.
 */
export function BeaconDashboard() {
  const host = useRef(null);
  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let unbindTheme = () => {};

    // Before the first paint of the shell, so the page never renders dark and
    // then flips to light a frame later.
    initTheme();

    // The rig capture is fetched beside the fleet, and a failure to find one is
    // not an error: most runs have no bench capture, and the section is simply
    // omitted. Only the fleet payload can fail the whole view.
    Promise.all([
      api.beacon(),
      api
        .serialReplay(RIG_CAPTURE)
        .catch(() => null),
    ])
      .then(([data, rig]) => {
        if (cancelled || !host.current) return;

        // The renderer reads this at boot. Setting it before writing the shell
        // keeps the original contract intact.
        // The renderer reads the rig off the same object it reads the fleet
        // off, so the picker and the rig view need no second data path. It is
        // attached rather than merged into `batteries`: every fleet statistic
        // iterates that array, and the rig is not a member of the fleet.
        window.__BEACON__ = { ...data, rig };
        host.current.innerHTML = shellHtml(data, 'BEACON', rig);
        boot();
        unbindTheme = bindThemeToggle(host.current);
        setReady(true);
      })
      .catch((e) => !cancelled && setError(e.message));

    return () => {
      cancelled = true;
      unbindTheme();
    };
  }, []);

  if (error) {
    return (
      <div className="beacon-gate">
        <h1>BEACON</h1>
        <p>{error}</p>
        <p className="sub">
          No pipeline run has been ingested yet, so there is no fleet to render.
          The dashboard does not fabricate one to fill the page.
        </p>
        <button className="cta" onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <>
      {!ready && (
        <div className="beacon-gate">
          <h1>BEACON</h1>
          <p className="sub">Assembling the fleet…</p>
        </div>
      )}
      <div ref={host} />
    </>
  );
}
