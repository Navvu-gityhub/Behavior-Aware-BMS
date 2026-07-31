import { Router } from 'express';
import { callPython } from '../pythonClient.js';

// Telemetry, digital-twin and transfer-validation routes.
//
// Kept in their own router rather than appended to batteries.js so the
// pre-existing fleet routes are untouched -- this file only adds forwarding.
// Like that router, nothing here reimplements pipeline logic: each handler
// forwards to the Python service and lets the centralised error middleware in
// app.js decide the status a caller sees.
//
// Why the gateway needs explicit routes at all: the React client reaches the
// backend only through /api/*, and app.js 404s anything that does not match a
// route. A new Python endpoint is therefore invisible to the UI until it is
// forwarded here. The alternative -- a blanket proxy -- would expose every
// future Python route to the browser by default, which is a wider surface than
// this gateway should have.
export const telemetryRouter = Router();

// Query strings are not forwarded wholesale. Each route names the parameters it
// passes through, so a client cannot reach an upstream parameter this gateway
// has not considered.
function query(params) {
  const pairs = Object.entries(params).filter(([, value]) => value !== undefined && value !== '');
  if (pairs.length === 0) return '';
  return `?${pairs.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join('&')}`;
}

// -- signal coverage ---------------------------------------------------------

telemetryRouter.get('/telemetry/coverage', async (req, res, next) => {
  try {
    const result = await callPython(`/telemetry/coverage${query({ dbc_path: req.query.dbc_path })}`);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

// -- replay and live capture -------------------------------------------------

telemetryRouter.post('/telemetry/replay', async (req, res, next) => {
  try {
    const result = await callPython('/telemetry/replay', { method: 'POST', body: req.body });
    res.json(result);
  } catch (err) {
    next(err);
  }
});

telemetryRouter.post('/telemetry/live', async (req, res, next) => {
  try {
    const result = await callPython('/telemetry/live', { method: 'POST', body: req.body });
    res.json(result);
  } catch (err) {
    next(err);
  }
});

telemetryRouter.get('/telemetry/latest/:id', async (req, res, next) => {
  try {
    const result = await callPython(`/telemetry/latest/${encodeURIComponent(req.params.id)}`);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

telemetryRouter.get('/telemetry/live/:id', async (req, res, next) => {
  try {
    const path = `/telemetry/live/${encodeURIComponent(req.params.id)}${query({ window: req.query.window })}`;
    const result = await callPython(path);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

telemetryRouter.get('/telemetry/thermal/:id', async (req, res, next) => {
  try {
    const result = await callPython(`/telemetry/thermal/${encodeURIComponent(req.params.id)}`);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

// -- digital twin ------------------------------------------------------------

telemetryRouter.get('/telemetry/twin/:id', async (req, res, next) => {
  try {
    const result = await callPython(`/telemetry/twin/${encodeURIComponent(req.params.id)}`);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

// -- transfer validation -----------------------------------------------------

telemetryRouter.get('/transfer/feasibility', async (req, res, next) => {
  try {
    const result = await callPython(`/transfer/feasibility${query({ source: req.query.source })}`);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

telemetryRouter.get('/datasets', async (req, res, next) => {
  try {
    const result = await callPython('/datasets');
    res.json(result);
  } catch (err) {
    next(err);
  }
});
