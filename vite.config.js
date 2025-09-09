import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [react()],
    preview: {
      host: '0.0.0.0',
      port: env.PORT || 4173,
      strictPort: true
    }
  }
})
