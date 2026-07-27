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
};
