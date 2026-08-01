const BASE = '/api';

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    throw new ApiError(res.status, (data && data.error) || res.statusText);
  }
  return data;
}

export const api = {
  healthz: () => request('/healthz'),
  listBatteries: () => request('/batteries'),
  getBattery: (id) => request(`/batteries/${encodeURIComponent(id)}`),
  getTimeline: (id) => request(`/batteries/${encodeURIComponent(id)}/timeline`),
  simulate: (payload) =>
    request('/pipeline/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  // --- telemetry -----------------------------------------------------------
  //
  // These reach the same Express gateway as the fleet calls above; the gateway
  // forwards each to the Python service (mern/server/src/routes/telemetry.js).
  // Nothing here is mocked: every view built on these renders values the
  // pipeline computed, or renders nothing and says why.

  coverage: (dbcPath) =>
    request(`/telemetry/coverage${dbcPath ? `?dbc_path=${encodeURIComponent(dbcPath)}` : ''}`),

  replay: (payload) =>
    request('/telemetry/replay', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  liveCapture: (payload) =>
    request('/telemetry/live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  latestRun: (id) => request(`/telemetry/latest/${encodeURIComponent(id)}`),

  liveState: (id, window) =>
    request(`/telemetry/live/${encodeURIComponent(id)}${window ? `?window=${window}` : ''}`),

  thermal: (id) => request(`/telemetry/thermal/${encodeURIComponent(id)}`),

  twinHistory: (id) => request(`/telemetry/twin/${encodeURIComponent(id)}`),

  // --- transfer validation --------------------------------------------------

  feasibility: (source) =>
    request(`/transfer/feasibility${source ? `?source=${encodeURIComponent(source)}` : ''}`),

  datasets: () => request('/datasets'),
};

export { ApiError };
