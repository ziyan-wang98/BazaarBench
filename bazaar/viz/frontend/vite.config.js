import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// The Vue SPA is built as a relative-path bundle so the resulting
// `dist/` can be copied into any output directory (e.g.
// `runs/site/`) and served straight from the filesystem — no
// dev server, no base-href fiddling. The Python exporter drops
// `data.json` next to `index.html`; the app fetches it relative
// at runtime.
export default defineConfig({
  plugins: [vue()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Keep the bundle readable so a reviewer can peek. Minification
    // on a research demo buys us nothing; debuggability buys a lot.
    minify: false,
    sourcemap: true,
  },
  server: {
    port: 5173,
    strictPort: false,
  },
})
