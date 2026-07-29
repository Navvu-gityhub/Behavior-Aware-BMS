import express from 'express';
import cors from 'cors';
import morgan from 'morgan';

import { batteriesRouter } from './routes/batteries.js';
import { UpstreamError, ServiceUnavailableError } from './pythonClient.js';

export function createApp() {
  const app = express();

  app.use(morgan('dev'));
  app.use(cors()); // React dev server runs on a different port -- needs this for local dev.
  app.use(express.json());

  app.use('/api', batteriesRouter);

  // 404 for anything under /api that didn't match a route above.
  app.use('/api', (req, res) => {
    res.status(404).json({ error: `No such gateway route: ${req.method} ${req.originalUrl}` });
  });

  // Centralized error handling: this is the one place that decides what
  // HTTP status a caller sees, so individual routes don't each reinvent
  // the UpstreamError/ServiceUnavailableError -> status code mapping.
  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    if (err instanceof UpstreamError) {
      res.status(err.status).json({ error: err.message });
    } else if (err instanceof ServiceUnavailableError) {
      // 502 Bad Gateway is the correct code here: the gateway itself is
      // fine, the service it depends on isn't reachable.
      res.status(502).json({ error: err.message });
    } else {
      console.error(err);
      res.status(500).json({ error: 'Internal gateway error' });
    }
  });

  return app;
}
