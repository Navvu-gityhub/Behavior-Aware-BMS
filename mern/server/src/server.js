import { createApp } from './app.js';
import { config } from './config.js';

const app = createApp();

app.listen(config.port, () => {
  console.log(`beacon-gateway listening on http://localhost:${config.port}`);
  console.log(`proxying pipeline calls to ${config.pythonApiBaseUrl}`);
});
