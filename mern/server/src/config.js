export const config = {
  port: parseInt(process.env.PORT || '5000', 10),
  // A getter, not a captured value: reads the env var fresh on every
  // access. Matters for tests (can point at a fake backend per-test
  // without re-importing modules to bust Node's ESM cache) and means a
  // process manager that changes env vars without a full restart would
  // actually take effect, not just appear to.
  get pythonApiBaseUrl() {
    return process.env.PYTHON_API_BASE_URL || 'http://127.0.0.1:8000';
  },
  // Which origins the browser may call this gateway from.
  //
  // The default is permissive because the default deployment is a laptop: the
  // Vite dev server runs on a different port, and pinning an origin list here
  // would break the moment someone ran it on 5174. That is a reasonable
  // default for a bench tool and a poor one for anything reachable, so it is
  // configurable rather than fixed:
  //
  //   CORS_ORIGIN=https://beacon.internal.example.com
  //
  // Note this gateway exposes no authenticated endpoint and sets no cookie, so
  // a permissive origin grants a caller nothing they could not get by calling
  // the gateway directly. It is worth tightening on a shared host, not worth
  // pretending is a security boundary.
  get corsOrigin() {
    return process.env.CORS_ORIGIN || '*';
  },
};
