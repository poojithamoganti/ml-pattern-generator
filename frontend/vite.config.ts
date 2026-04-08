import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backend = env.VITE_BACKEND_ORIGIN || 'http://127.0.0.1:8001'
  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': backend,
        '/health': backend,
      },
    },
  }
})
