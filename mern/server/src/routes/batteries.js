import { Router } from 'express';
import { callPython } from '../pythonClient.js';

export const batteriesRouter = Router();

// GET /api/healthz -- gateway's own liveness PLUS the upstream Python
// service's, so a caller can tell "gateway is up but Python isn't" apart
// from "everything is fine" in one request.
batteriesRouter.get('/healthz', async (req, res, next) => {
  try {
    const upstream = await callPython('/healthz');
    res.json({ gateway: 'ok', pipeline: upstream });
  } catch (err) {
    next(err);
  }
});

batteriesRouter.post('/pipeline/simulate', async (req, res, next) => {
  try {
    const result = await callPython('/pipeline/simulate', { method: 'POST', body: req.body });
    res.json(result);
  } catch (err) {
    next(err);
  }
});

// GET /api/validation/summary -- what has actually been validated, read by the
// Python service from the tracked artifacts under reports/metrics/. Proxied
// rather than recomputed here for the same reason every other route is: the
// gateway must not become a second place where a number can be produced.
// GET /api/dashboard/beacon -- the full dashboard payload, assembled by the
// Python service from the same builder the static renderer used. Proxied, not
// recomputed: the client renders, it does not decide what is available.
batteriesRouter.get('/dashboard/beacon', async (req, res, next) => {
  try {
    res.json(await callPython('/dashboard/beacon'));
  } catch (err) {
    next(err);
  }
});

batteriesRouter.get('/validation/summary', async (req, res, next) => {
  try {
    res.json(await callPython('/validation/summary'));
  } catch (err) {
    next(err);
  }
});

batteriesRouter.get('/validation/ceilings', async (req, res, next) => {
  try {
    res.json(await callPython('/validation/ceilings'));
  } catch (err) {
    next(err);
  }
});

batteriesRouter.get('/batteries', async (req, res, next) => {
  try {
    const result = await callPython('/batteries');
    res.json(result);
  } catch (err) {
    next(err);
  }
});

batteriesRouter.get('/batteries/:id', async (req, res, next) => {
  try {
    const result = await callPython(`/batteries/${encodeURIComponent(req.params.id)}`);
    res.json(result);
  } catch (err) {
    next(err);
  }
});

batteriesRouter.get('/batteries/:id/timeline', async (req, res, next) => {
  try {
    const result = await callPython(`/batteries/${encodeURIComponent(req.params.id)}/timeline`);
    res.json(result);
  } catch (err) {
    next(err);
  }
});
