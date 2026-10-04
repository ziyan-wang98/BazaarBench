// Single source of truth for the Vue SPA.
//
// Loads `data.json` (written by bazaar.viz.exporter) once at app
// boot and exposes a reactive `snapshot` ref plus `loading` /
// `error` flags. Views pull fields off `snapshot.value` — no per-
// view fetching, no duplicate parses.
//
// We deliberately don't use Pinia here. A single reactive ref
// behind a `useSnapshot()` composable gives us ~50 lines of
// dependency-free state management and maps 1:1 to a single
// JSON file.
import { ref } from 'vue'

const snapshot = ref(null)
const loading = ref(false)
const error = ref(null)

let started = false

export function useSnapshot() {
  if (!started) {
    started = true

    // Preferred path: a <script id="bazaar-snapshot"
    // type="application/json"> block inlined into index.html by the
    // Python CLI. This makes the built SPA work over a `file://`
    // URL — the browser blocks `fetch('./data.json')` as a
    // cross-origin request under file:// in every major browser,
    // so the prod build must not depend on fetch.
    const inlined = typeof document !== 'undefined'
      ? document.getElementById('bazaar-snapshot')
      : null
    if (inlined && inlined.textContent && inlined.textContent.trim()) {
      try {
        snapshot.value = JSON.parse(inlined.textContent)
        return { snapshot, loading, error }
      } catch (e) {
        // Malformed inline payload — fall through to fetch as a
        // best-effort recovery.
        // eslint-disable-next-line no-console
        console.error('bazaar-snapshot inline JSON failed to parse', e)
      }
    }

    // Fallback: dev mode (`npm run dev`). Vite serves
    // `public/data.json` at `/data.json`.
    loading.value = true
    fetch('./data.json', { cache: 'no-cache' })
      .then((r) => {
        if (!r.ok) throw new Error(`data.json fetch failed: ${r.status}`)
        return r.json()
      })
      .then((d) => {
        snapshot.value = d
        loading.value = false
      })
      .catch((e) => {
        error.value = e
        loading.value = false
      })
  }
  return { snapshot, loading, error }
}

// Convenience lookup helpers — views use these instead of repeated
// finds, keeping the templates small.

export function agentById(snap, id) {
  if (!snap) return null
  return snap.agents.find((a) => a.agent_id === Number(id)) || null
}

export function listingById(snap, id) {
  if (!snap) return null
  return snap.listings.find((l) => l.listing_id === Number(id)) || null
}

export function photoById(snap, id) {
  if (!snap) return null
  return snap.photos.find((p) => p.photo_id === Number(id)) || null
}

export function threadsForAgent(snap, id) {
  if (!snap) return []
  return snap.threads.filter(
    (t) => t.buyer_agent_id === id || t.seller_agent_id === id,
  )
}

export function narrativesForAgent(snap, id) {
  if (!snap) return []
  return snap.narratives
    .filter((n) => n.agent_id === id && !n.decayed)
    .sort((a, b) => b.created_tick - a.created_tick)
}

export function ledgerForAgent(snap, id) {
  if (!snap) return []
  const found = snap.ledger.find((x) => x.agent_id === id)
  return found ? found.entries : []
}

export function photosForAgent(snap, id) {
  if (!snap) return []
  return snap.photos
    .filter((p) => p.sender_agent_id === id)
    .sort((a, b) => b.photo_id - a.photo_id)
}
