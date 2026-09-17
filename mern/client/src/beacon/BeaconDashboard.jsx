import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';
import { shellHtml } from './shell.js';
import { boot } from './renderer.js';
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

    api
      .beacon()
      .then((data) => {
        if (cancelled || !host.current) return;

        // The renderer reads this at boot. Setting it before writing the shell
        // keeps the original contract intact.
        window.__BEACON__ = data;
        host.current.innerHTML = shellHtml(data);
        boot();
        setReady(true);
      })
      .catch((e) => !cancelled && setError(e.message));

    return () => {
      cancelled = true;
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
