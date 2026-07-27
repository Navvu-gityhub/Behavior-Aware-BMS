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
