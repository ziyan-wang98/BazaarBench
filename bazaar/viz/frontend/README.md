# bazaar/viz/frontend — Vue 3 run inspector

Multi-surface Vue SPA that reads a BazaarBench run's exported
`data.json` and renders its five+ inspection surfaces.

## Dev

```bash
# from the BazaarBench root, export a data snapshot into this
# frontend's public/ so `npm run dev` serves it:
bazaar export-data runs/demo.db --out bazaar/viz/frontend/public/data.json

cd bazaar/viz/frontend
npm install
npm run dev
# → http://localhost:5173
```

Changes to `.vue` / `.js` / `.css` hot-reload. The `data.json` is
loaded once at boot; re-run `bazaar export-data` and hard-reload the
page to pick up a new snapshot.

## Production (static assembly + serve)

```bash
# 1. build the SPA and drop it next to an inlined data snapshot:
bazaar dashboard-vue runs/demo.db --out runs/site

# 2. serve it over http:// (browsers block ES-module imports
#    from file:// URLs, so double-clicking index.html goes blank):
bazaar serve runs/site
# → http://localhost:8765/ opens in your browser
```

Stop the server with Ctrl-C. The built `runs/site/` directory is
self-contained — deploy it to any static host (GitHub Pages,
Netlify, plain nginx) without rebuilding.

## Architecture

```
src/
  main.js                       entry, wires router
  App.vue                       shell: sticky navbar + <RouterView>
  router/index.js               hash routes, code-split per view
  composables/useData.js        fetches data.json once, exposes
                                 reactive snapshot + helper lookups
  views/
    Home.vue                    KPI grid + per-surface entry tiles
    Feed.vue                    listing grid
    Threads.vue                 chat bubbles + photo cards
    Agents.vue                  agent directory grid
    AgentProfile.vue            persona / ledger / narratives
    SocialGraph.vue             d3-force network
    MapView.vue                 interactive SVG map
    Metrics.vue                 event-log dashboard
    Timeline.vue                tick-scroll event list
  styles/app.css                shared theme
```

## Stack

Vue 3 + Vite + vue-router + d3.
