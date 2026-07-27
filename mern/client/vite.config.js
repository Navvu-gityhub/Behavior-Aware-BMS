import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-time reverse proxy to the Express gateway, mirroring how a real
    // deployment would sit nginx/similar in front of both -- means the
    // browser only ever talks to one origin and CORS is a non-issue, not
    // something worked around with permissive headers.
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
