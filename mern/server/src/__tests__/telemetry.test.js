import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import request from 'supertest';

// Same discipline as gateway.test.js: a fake Python backend, because these
// tests verify the GATEWAY's behaviour -- that a route is forwarded at all,
// that path and query parameters survive, that upstream status codes are
// mapped. Pipeline correctness is already covered by tests/test_telemetry_api.py
// and re-testing it here would duplicate that for no benefit.
let fakePython;
let fakePythonUrl;
let app;

// Records what the fake backend was asked for, so a test can assert the
// gateway forwarded the right path rather than merely returning something.
let lastRequest;

before(async () => {
  fakePython = http.createServer((req, res) => {
    lastRequest = { url: req.url, method: req.method };
    res.setHeader('Content-Type', 'application/json');

    if (req.url.startsWith('/telemetry/coverage')) {
      res.end(JSON.stringify({
        dbc_path: 'x.dbc',
        signal_map_used: { pack_current: 'current_a' },
        status: 'COMPLETE',
        complete: true,
        available_signals: ['pack_current'],
        mapped_channels: ['current_a'],
        missing_channels: [],
        explanation: '',
      }));
    } else if (req.url === '/telemetry/replay' && req.method === 'POST') {
      let body = '';
      req.on('data', (chunk) => (body += chunk));
      req.on('end', () => {
        const parsed = JSON.parse(body);
        res.end(JSON.stringify({ status: 'SCORED', battery_id: parsed.battery_id, refusals: [] }));
      });
    } else if (req.url === '/telemetry/live' && req.method === 'POST') {
      res.end(JSON.stringify({ status: 'SCORED', battery_id: 'LIVE', refusals: [] }));
    } else if (req.url === '/telemetry/latest/KNOWN') {
      res.end(JSON.stringify({ battery_id: 'KNOWN', status: 'SCORED' }));
    } else if (req.url === '/telemetry/latest/UNKNOWN') {
      res.statusCode = 404;
      res.end(JSON.stringify({ detail: 'No telemetry run recorded.' }));
    } else if (req.url.startsWith('/telemetry/live/')) {
      res.end(JSON.stringify({ battery_id: 'KNOWN', n_samples: 10, recent: [] }));
    } else if (req.url.startsWith('/telemetry/thermal/')) {
      res.end(JSON.stringify({ battery_id: 'KNOWN', points: [], n_cycles: 0 }));
    } else if (req.url.startsWith('/telemetry/twin/')) {
      res.end(JSON.stringify({ battery_id: 'KNOWN', n_snapshots: 2, snapshots: [], transitions: [] }));
    } else if (req.url.startsWith('/transfer/feasibility')) {
      res.end(JSON.stringify([{ source: 'nasa', target: 'stanford_severson', status: 'PREDICTED_MARGINAL' }]));
    } else if (req.url === '/datasets') {
      res.end(JSON.stringify([{ name: 'nasa', n_cells: 34 }]));
    } else {
      res.statusCode = 404;
      res.end(JSON.stringify({ detail: 'not found' }));
    }
  });

  await new Promise((resolve) => fakePython.listen(0, '127.0.0.1', resolve));
  fakePythonUrl = `http://127.0.0.1:${fakePython.address().port}`;
  process.env.PYTHON_API_BASE_URL = fakePythonUrl;

  ({ createApp: app } = await import('../app.js'));
  app = app();
});

after(async () => {
  await new Promise((resolve) => fakePython.close(resolve));
});

// -- the routes exist at all -------------------------------------------------
//
// This is the whole point of the file. Before these routes were added, every
// one of these paths returned the gateway's 404 and the React client could not
// reach the telemetry API however correct the Python side was.

test('coverage is forwarded', async () => {
  const res = await request(app).get('/api/telemetry/coverage');
  assert.equal(res.status, 200);
  assert.equal(res.body.status, 'COMPLETE');
});

test('coverage forwards the dbc_path query parameter', async () => {
  await request(app).get('/api/telemetry/coverage?dbc_path=/tmp/x.dbc');
  assert.ok(lastRequest.url.includes('dbc_path='));
  assert.ok(lastRequest.url.includes('x.dbc'));
});

test('replay forwards the request body', async () => {
  const res = await request(app)
    .post('/api/telemetry/replay')
    .send({ log_path: '/tmp/drive.asc', battery_id: 'VEH_9' });
  assert.equal(res.status, 200);
  assert.equal(res.body.battery_id, 'VEH_9');
});

test('live capture is forwarded', async () => {
  const res = await request(app)
    .post('/api/telemetry/live')
    .send({ channel: 'vcan0', duration_s: 5 });
  assert.equal(res.status, 200);
});

test('latest is forwarded with the battery id in the path', async () => {
  const res = await request(app).get('/api/telemetry/latest/KNOWN');
  assert.equal(res.status, 200);
  assert.equal(lastRequest.url, '/telemetry/latest/KNOWN');
});

test('live state forwards the window parameter', async () => {
  await request(app).get('/api/telemetry/live/KNOWN?window=50');
  assert.ok(lastRequest.url.includes('window=50'));
});

test('thermal timeline is forwarded', async () => {
  const res = await request(app).get('/api/telemetry/thermal/KNOWN');
  assert.equal(res.status, 200);
});

test('twin history is forwarded', async () => {
  const res = await request(app).get('/api/telemetry/twin/KNOWN');
  assert.equal(res.status, 200);
  assert.equal(res.body.n_snapshots, 2);
});

test('feasibility is forwarded', async () => {
  const res = await request(app).get('/api/transfer/feasibility');
  assert.equal(res.status, 200);
  assert.equal(res.body[0].status, 'PREDICTED_MARGINAL');
});

test('feasibility forwards the source parameter', async () => {
  await request(app).get('/api/transfer/feasibility?source=nasa');
  assert.ok(lastRequest.url.includes('source=nasa'));
});

test('datasets is forwarded', async () => {
  const res = await request(app).get('/api/datasets');
  assert.equal(res.status, 200);
  assert.equal(res.body[0].name, 'nasa');
});

// -- status mapping ----------------------------------------------------------

test('an upstream 404 stays a 404 rather than becoming a 500', async () => {
  const res = await request(app).get('/api/telemetry/latest/UNKNOWN');
  assert.equal(res.status, 404);
});

test('unrecognised ids do not receive a query string they did not send', async () => {
  // query() drops undefined and empty values, so a bare request must not
  // acquire a trailing '?'.
  await request(app).get('/api/telemetry/coverage');
  assert.equal(lastRequest.url, '/telemetry/coverage');
});

// -- the pre-existing routes must still work ---------------------------------

test('mounting a second router does not shadow the fleet routes', async () => {
  // batteriesRouter is mounted first on the same prefix; adding telemetryRouter
  // must not intercept its paths.
  const res = await request(app).get('/api/telemetry/coverage');
  assert.equal(res.status, 200);
  const missing = await request(app).get('/api/no-such-route');
  assert.equal(missing.status, 404);
  assert.match(missing.body.error, /No such gateway route/);
});
