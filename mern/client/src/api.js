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
};

export { ApiError };
