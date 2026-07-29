import { config } from './config.js';

/** Thrown when the Python service responds but with a non-2xx status
 * (e.g. its own 404 for an unknown battery, its own 422 for a bad
 * request body). Carries the upstream status through unchanged. */
export class UpstreamError extends Error {
  constructor(status, detail) {
    super(detail || `Python service returned ${status}`);
    this.name = 'UpstreamError';
    this.status = status;
  }
}

/** Thrown when the Python service can't be reached at all (not running,
 * wrong port, network issue) -- distinct from UpstreamError because this
 * is a gateway problem, not something the underlying pipeline decided. */
export class ServiceUnavailableError extends Error {
  constructor(cause) {
    super(`Python pipeline service unreachable at ${config.pythonApiBaseUrl}: ${cause.message}`);
    this.name = 'ServiceUnavailableError';
  }
}

/**
 * Call the existing FastAPI service and return parsed JSON.
 *
 * This function is the entire point of the "lighter" MERN scope: it does
 * not compute a health index, a risk score, or a twin state -- it forwards
 * to the Python service that already does, and already has 25 passing
 * tests behind it (see src/bms/api/, tests/test_api.py,
 * tests/test_digital_twin.py in the Python package). If that logic is
 * ever wrong, fixing it belongs in the Python package, not here --
 * duplicating it here would be exactly the "two independently-tuned
 * scoring systems" problem the digital twin module's docstring already
 * warns against for the risk score vs. health index overlap.
 */
export async function callPython(path, { method = 'GET', body } = {}) {
  let res;
  try {
    res = await fetch(`${config.pythonApiBaseUrl}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (cause) {
    throw new ServiceUnavailableError(cause);
  }

  const text = await res.text();
  const data = text ? JSON.parse(text) : null;

  if (!res.ok) {
    throw new UpstreamError(res.status, data && data.detail);
  }
  return data;
}
