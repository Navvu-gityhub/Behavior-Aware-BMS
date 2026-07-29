import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import request from 'supertest';

// A fake Python backend, not the real one -- these tests verify the
// gateway's OWN behavior (forwarding, status/error mapping), which is the
// only new logic this layer adds. Re-testing health_index/risk_score/twin
// correctness here would duplicate tests/test_api.py and
// tests/test_digital_twin.py in the Python package for no benefit; if
// those change, these tests shouldn't need to.
let fakePython;
let fakePythonUrl;
let app;

before(async () => {
  fakePython = http.createServer((req, res) => {
    res.setHeader('Content-Type', 'application/json');
    if (req.url === '/healthz') {
      res.end(JSON.stringify({ status: 'ok', n_runs: 3, n_batteries_tracked: 7 }));
    } else if (req.url === '/pipeline/simulate' && req.method === 'POST') {
      let body = '';
      req.on('data', (chunk) => (body += chunk));
      req.on('end', () => {
        const parsed = JSON.parse(body);
        res.end(JSON.stringify({ n_batteries_scored: parsed.n_batteries, transitions: [], battery_ids: [] }));
      });
    } else if (req.url === '/batteries') {
      res.end(JSON.stringify([{ battery_id: 'SIM000', twin_state: 'NORMAL' }]));
    } else if (req.url === '/batteries/KNOWN') {
      res.end(JSON.stringify({ battery_id: 'KNOWN', twin: { twin_state: 'NORMAL' } }));
    } else if (req.url === '/batteries/UNKNOWN') {
      res.statusCode = 404;
      res.end(JSON.stringify({ detail: "No battery 'UNKNOWN' in the fleet store yet." }));
    } else {
      res.statusCode = 404;
      res.end(JSON.stringify({ detail: 'not found' }));
    }
  });
  await new Promise((resolve) => fakePython.listen(0, resolve));
  fakePythonUrl = `http://127.0.0.1:${fakePython.address().port}`;
  process.env.PYTHON_API_BASE_URL = fakePythonUrl;

  const { createApp } = await import('../app.js');
  app = createApp();
});

after(async () => {
  await new Promise((resolve) => fakePython.close(resolve));
});

test('GET /api/healthz aggregates gateway + pipeline status', async () => {
  const res = await request(app).get('/api/healthz');
  assert.equal(res.status, 200);
  assert.equal(res.body.gateway, 'ok');
  assert.equal(res.body.pipeline.n_batteries_tracked, 7);
});

test('POST /api/pipeline/simulate forwards the request body and returns the response unchanged', async () => {
  const res = await request(app).post('/api/pipeline/simulate').send({ n_batteries: 9, rows_per_battery: 100 });
  assert.equal(res.status, 200);
  assert.equal(res.body.n_batteries_scored, 9);
});

test('GET /api/batteries forwards the fleet list', async () => {
  const res = await request(app).get('/api/batteries');
  assert.equal(res.status, 200);
  assert.deepEqual(res.body, [{ battery_id: 'SIM000', twin_state: 'NORMAL' }]);
});

test('GET /api/batteries/:id forwards a known battery', async () => {
  const res = await request(app).get('/api/batteries/KNOWN');
  assert.equal(res.status, 200);
  assert.equal(res.body.battery_id, 'KNOWN');
});

test("GET /api/batteries/:id forwards Python's 404 and its detail message, not a generic one", async () => {
  const res = await request(app).get('/api/batteries/UNKNOWN');
  assert.equal(res.status, 404);
  assert.match(res.body.error, /UNKNOWN/);
});

test('unknown gateway route returns 404 without reaching Python', async () => {
  const res = await request(app).get('/api/not-a-real-route');
  assert.equal(res.status, 404);
});

test('Python service unreachable returns 502, not a crash', async () => {
  process.env.PYTHON_API_BASE_URL = 'http://127.0.0.1:1'; // nothing listens here
  try {
    const res = await request(app).get('/api/batteries');
    assert.equal(res.status, 502);
    assert.match(res.body.error, /unreachable/);
  } finally {
    process.env.PYTHON_API_BASE_URL = fakePythonUrl; // restore for any later tests
  }
});
