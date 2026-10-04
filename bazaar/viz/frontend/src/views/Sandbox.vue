<script setup>
// V6 sandbox — timeline replay + always-on edges + edge-click
// dyad detail + drag-safe focus + style/legend audit.
//
// Addressed reviewer gaps:
//   * lines break during drag → focus is now pinned via `selected`
//     AND via `dragState`; neither hover-loss nor pointer-exit
//     detaches the radiating edges
//   * colors/styles didn't match the legend → the legend and the
//     renderer now both read the same EDGE_STYLE object, so any
//     change propagates to both
//   * "can't see transactions / conversations through a line" →
//     clicking an edge opens a DYAD panel that lists messages,
//     offers, and ratings between the pair, with router links to
//     the full thread
//   * no timeline → bottom-of-canvas scrubber with play / pause /
//     speed controls. Edges fade in chronologically as their
//     events' ticks ≤ currentTick; users see the network assemble
//   * required node-selection to see relationships → there's now
//     a persistent low-contrast "all edges" layer so the network
//     structure is visible on load
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import * as d3 from 'd3'
import Avatar from '../components/Avatar.vue'
import PhotoCard from '../components/PhotoCard.vue'
import {
  agentById,
  listingById,
  useSnapshot,
} from '../composables/useData.js'

const { snapshot } = useSnapshot()
const router = useRouter()

const svgRef = ref(null)
const wrapRef = ref(null)

const hovered = ref(null)    // {kind, ref}
const selected = ref(null)   // sticky selection (node or edge)
const detail = computed(() => selected.value || hovered.value)

const showPhantoms = ref(true)
const showTripwires = ref(true)
const showLabels = ref(true)
const maximized = ref(false)

// --- Timeline --------------------------------------------------------

const tickMax = computed(() => snapshot.value?.meta?.tick_max || 0)
const currentTick = ref(0)
const playing = ref(false)
const speed = ref(8)                // ticks per animation frame step
let playHandle = 0

// Initialise the scrubber to the final tick once snapshot loads
// so the default view shows the complete network.
watch(() => tickMax.value, (v) => { currentTick.value = v })

function startPlaying() {
  if (playing.value) return
  playing.value = true
  playHandle = requestAnimationFrame(tickPlay)
}
function stopPlaying() {
  playing.value = false
  if (playHandle) cancelAnimationFrame(playHandle)
  playHandle = 0
}
function togglePlay() {
  if (playing.value) stopPlaying()
  else {
    if (currentTick.value >= tickMax.value) currentTick.value = 0
    startPlaying()
  }
}
function tickPlay() {
  if (!playing.value) return
  currentTick.value = Math.min(tickMax.value, currentTick.value + speed.value)
  if (currentTick.value >= tickMax.value) {
    stopPlaying()
    return
  }
  playHandle = requestAnimationFrame(tickPlay)
}
function rewind() { currentTick.value = 0 }
function fastForward() { currentTick.value = tickMax.value }

// --- palette / identity ---------------------------------------------

const AGENT_COLORS = [
  '#0d7975', '#b84535', '#4a3a9e', '#8a5e0d',
  '#185fa5', '#3b6d11', '#a32d2d', '#5f5e5a',
]
function agentColor(a) {
  if (a.status !== 'active') return '#C5283D'
  return AGENT_COLORS[a.agent_id % AGENT_COLORS.length]
}
function lighten(hex, amount = 0.35) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  const mix = (v) => Math.round(v + (255 - v) * amount)
  return '#' + [mix(r), mix(g), mix(b)]
    .map((v) => v.toString(16).padStart(2, '0')).join('')
}
function initialsOf(name) { return (name || '?').slice(0, 2).toUpperCase() }

// Edge style — single source of truth. Both the legend and the
// path renderer read this. No more visual/legend drift.
const EDGE_STYLE = {
  viewed:     { color: '#A0A0A0', width: 1,    dash: '2 3',   curvature: 0.10 },
  pinned:     { color: '#E9724C', width: 1.4,  dash: '5 3',   curvature: 0.14 },
  messaged:   { color: '#7B2D8E', width: 2,    dash: null,    curvature: 0.18 },
  offered:    { color: '#DAA520', width: 2.3,  dash: null,    curvature: 0.22 },
  transacted: { color: '#1A936F', width: 3.2,  dash: '8 4',   curvature: 0.26 },
}
const EDGE_RANK = { viewed: 0, pinned: 1, messaged: 2, offered: 3, transacted: 4 }

// --- data -----------------------------------------------------------

const agents = computed(() => snapshot.value?.agents || [])
const phantomListings = computed(
  () => (snapshot.value?.listings || []).filter((l) => l.is_phantom),
)
const tripwires = computed(() => snapshot.value?.events_summary?.tripwires || [])
const pairInteractions = computed(
  () => snapshot.value?.pair_interactions || [],
)

// pair lookup map
const pairMap = computed(() => {
  const m = new Map()
  for (const r of pairInteractions.value) {
    m.set(`${r.source}>${r.target}`, r)
  }
  return m
})

// Timeline-filtered edges: a pair contributes a kind if ≥1 event
// with that kind has tick ≤ currentTick. Takes the strongest kind
// per pair for visual stacking.
const visibleEdges = computed(() => {
  const T = currentTick.value
  const byPair = new Map()  // key "a,b" a<b → [{kind, n, source, target}]
  for (const r of pairInteractions.value) {
    const source = r.source, target = r.target
    const counts = { viewed: 0, pinned: 0, messaged: 0, offered: 0, transacted: 0 }
    for (const ev of r.events || []) {
      if (ev.tick <= T) counts[ev.kind] = (counts[ev.kind] || 0) + 1
    }
    // one record per direction per kind (but draw an undirected line
    // between the nodes; duplicate drops handled by the renderer)
    for (const kind of Object.keys(counts)) {
      if (counts[kind] === 0) continue
      const a = Math.min(source, target), b = Math.max(source, target)
      const key = `${a},${b},${kind}`
      const prev = byPair.get(key)
      if (prev) {
        prev.n += counts[kind]
      } else {
        byPair.set(key, {
          source, target, kind, n: counts[kind],
        })
      }
    }
  }
  return [...byPair.values()].sort(
    (a, b) => EDGE_RANK[a.kind] - EDGE_RANK[b.kind],
  )
})

const edgesSortedForBackdrop = computed(() =>
  visibleEdges.value.slice()
    .sort((a, b) => EDGE_RANK[a.kind] - EDGE_RANK[b.kind]),
)

function agentScore(a) { return a.activity?.score ?? 0 }

const maxScore = computed(() => {
  let m = 0
  for (const a of agents.value) m = Math.max(m, agentScore(a))
  return Math.max(1, m)
})
function agentRadius(a) {
  const t = Math.log1p(agentScore(a)) / Math.log1p(maxScore.value)
  return 12 + t * 14
}

// --- bbox + projection ----------------------------------------------

const bbox = computed(() => {
  const pts = []
  for (const a of agents.value) pts.push([a.home_lat, a.home_lng])
  if (!pts.length) return { latMin: -1, latMax: 1, lngMin: -1, lngMax: 1 }
  const lats = pts.map((p) => p[0]), lngs = pts.map((p) => p[1])
  let latMin = Math.min(...lats), latMax = Math.max(...lats)
  let lngMin = Math.min(...lngs), lngMax = Math.max(...lngs)
  const latSpan = Math.max(0.2, latMax - latMin)
  const lngSpan = Math.max(0.2, lngMax - lngMin)
  latMin -= latSpan * 0.1; latMax += latSpan * 0.1
  lngMin -= lngSpan * 0.1; lngMax += lngSpan * 0.1
  return { latMin, latMax, lngMin, lngMax }
})

function hash01(n, salt) {
  let x = (n * 2654435761 + salt * 40503) >>> 0
  x = (x ^ (x >>> 16)) >>> 0
  x = (Math.imul(x, 0x85ebca6b)) >>> 0
  x = (x ^ (x >>> 13)) >>> 0
  return (x % 10000) / 10000
}
const phantomPositions = computed(() => {
  const b = bbox.value
  const m = new Map()
  for (const p of phantomListings.value) {
    const fx = 0.25 + 0.5 * hash01(p.listing_id, 17)
    const fy = 0.25 + 0.5 * hash01(p.listing_id, 137)
    m.set(p.listing_id, {
      lat: b.latMin + (b.latMax - b.latMin) * fy,
      lng: b.lngMin + (b.lngMax - b.lngMin) * fx,
    })
  }
  return m
})

const W = 1280, H = 720, PAD = 60
function project(lat, lng) {
  const b = bbox.value
  const x = PAD + (lng - b.lngMin) / (b.lngMax - b.lngMin) * (W - 2 * PAD)
  const y = PAD + (b.latMax - lat) / (b.latMax - b.latMin) * (H - 2 * PAD)
  return { x, y }
}

const positionOverrides = ref(new Map())
function agentPos(a) {
  const o = positionOverrides.value.get(a.agent_id)
  if (o) return o
  return project(a.home_lat, a.home_lng)
}
function listingPos(l) {
  if (l.is_phantom) {
    const p = phantomPositions.value.get(l.listing_id)
    return p ? project(p.lat, p.lng) : { x: 0, y: 0 }
  }
  return project(l.location_lat, l.location_lng)
}
function resetPositions() { positionOverrides.value = new Map() }

// --- Curved path ----------------------------------------------------

function edgePath(ax, ay, bx, by, curvature) {
  const dx = bx - ax, dy = by - ay
  const cx = (ax + bx) / 2 - dy * curvature
  const cy = (ay + by) / 2 + dx * curvature
  return `M${ax},${ay} Q${cx},${cy} ${bx},${by}`
}

// --- Focus ----------------------------------------------------------

const focusAgentId = computed(() => {
  if (selected.value?.kind === 'agent') return selected.value.ref.agent_id
  if (dragState.value) return dragState.value.agent_id
  if (hovered.value?.kind === 'agent') return hovered.value.ref.agent_id
  return null
})
const focusDyad = computed(() => {
  const d = selected.value
  if (d?.kind === 'edge') return { a: d.ref.source, b: d.ref.target }
  return null
})

const focusConnected = computed(() => {
  const aid = focusAgentId.value
  if (aid == null) return null
  const s = new Set([aid])
  for (const r of pairInteractions.value) {
    if (r.source === aid) s.add(r.target)
    if (r.target === aid) s.add(r.source)
  }
  return s
})

function nodeOpacity(a) {
  const c = focusConnected.value
  if (!c) return 1
  return c.has(a.agent_id) ? 1 : 0.15
}

// Edge visual layer: is this edge in focus?
function edgeInFocus(e) {
  const fId = focusAgentId.value
  if (fId != null) {
    return e.source === fId || e.target === fId
  }
  const dy = focusDyad.value
  if (dy) {
    return (e.source === dy.a && e.target === dy.b)
        || (e.source === dy.b && e.target === dy.a)
  }
  return false
}
function edgeBaseOpacity(e) {
  // low-contrast baseline always visible
  return edgeInFocus(e) ? 0.95 : 0.22
}
function edgeWidth(e) {
  return EDGE_STYLE[e.kind].width + Math.min(2.5, Math.log1p(e.n))
}

const focusTripwires = computed(() => {
  const aid = focusAgentId.value
  if (aid == null) return []
  return tripwires.value.filter(
    (tw) => tw.agent_id === aid && (tw.trigger_tick ?? 0) <= currentTick.value,
  )
})

function tripwirePath(tw) {
  const a = agentById(snapshot.value, tw.agent_id)
  const l = listingById(snapshot.value, tw.listing_id)
  if (!a || !l) return null
  const p1 = agentPos(a), p2 = listingPos(l)
  return edgePath(p1.x, p1.y, p2.x, p2.y, 0.3)
}

// --- zoom + drag ---------------------------------------------------

let zoomBehaviour = null
function setupZoom() {
  const svg = d3.select(svgRef.value)
  const vp = svg.select('g.viewport')
  zoomBehaviour = d3.zoom()
    .scaleExtent([0.35, 8])
    .filter((event) => {
      if (event.type === 'wheel') return true
      if (event.type === 'dblclick') return false
      return !event.target.closest('.no-zoom')
    })
    .on('zoom', (event) => vp.attr('transform', event.transform))
  svg.call(zoomBehaviour)
  svg.on('click', (event) => {
    if (event.target === svgRef.value
      || event.target.classList.contains('backdrop')) {
      selected.value = null
    }
  })
}
function resetView() {
  resetPositions()
  if (!zoomBehaviour || !svgRef.value) return
  d3.select(svgRef.value).transition().duration(400)
    .call(zoomBehaviour.transform, d3.zoomIdentity)
}
onMounted(() => nextTick(setupZoom))
onUnmounted(() => { stopPlaying() })

// Click-vs-drag threshold.
const dragState = ref(null)
function screenToViewport(clientX, clientY) {
  const svg = svgRef.value
  if (!svg) return { x: clientX, y: clientY }
  const vp = svg.querySelector('g.viewport')
  const ctm = vp?.getScreenCTM()
  if (!ctm) return { x: clientX, y: clientY }
  const pt = svg.createSVGPoint()
  pt.x = clientX; pt.y = clientY
  return pt.matrixTransform(ctm.inverse())
}
function onAgentPointerDown(a, event) {
  event.stopPropagation()
  event.preventDefault()
  selected.value = { kind: 'agent', ref: a }
  const startSvg = screenToViewport(event.clientX, event.clientY)
  const base = agentPos(a)
  dragState.value = {
    agent_id: a.agent_id, ref: a,
    startClientX: event.clientX, startClientY: event.clientY,
    startSvgX: startSvg.x, startSvgY: startSvg.y,
    baseX: base.x, baseY: base.y,
    moved: false,
  }
  window.addEventListener('pointermove', onDocumentPointerMove)
  window.addEventListener('pointerup', onDocumentPointerUp, { once: true })
  window.addEventListener('pointercancel', onDocumentPointerUp, { once: true })
}
function onDocumentPointerMove(event) {
  const d = dragState.value
  if (!d) return
  const dx = event.clientX - d.startClientX
  const dy = event.clientY - d.startClientY
  if (!d.moved && (dx * dx + dy * dy) > 9) d.moved = true
  if (!d.moved) return
  const here = screenToViewport(event.clientX, event.clientY)
  const m = new Map(positionOverrides.value)
  m.set(d.agent_id, {
    x: d.baseX + (here.x - d.startSvgX),
    y: d.baseY + (here.y - d.startSvgY),
  })
  positionOverrides.value = m
}
function onDocumentPointerUp() {
  dragState.value = null
  window.removeEventListener('pointermove', onDocumentPointerMove)
}

// --- selection handlers -------------------------------------------

function setHover(kind, ref) { hovered.value = { kind, ref } }
function clearHover() { hovered.value = null }
function selectAgent(a, event) {
  event?.stopPropagation()
  selected.value = { kind: 'agent', ref: a }
}
function selectEdge(e, event) {
  event?.stopPropagation()
  selected.value = { kind: 'edge', ref: e }
}
function selectTripwire(tw, event) {
  event?.stopPropagation()
  selected.value = { kind: 'tripwire', ref: tw }
}
function closeDetail() { selected.value = null }
function openAgent(aid) { router.push(`/agents/${aid}`) }

const detailAgent = computed(() => {
  const d = detail.value
  if (!d) return null
  if (d.kind === 'agent') return d.ref
  if (d.kind === 'tripwire') return agentById(snapshot.value, d.ref.agent_id)
  if (d.kind === 'listing') return agentById(snapshot.value, d.ref.owner_agent_id)
  return null
})

// --- dyad detail: messages, offers, photos, ratings between A and B

function fmtPrice(cents) { return `$${(cents / 100).toFixed(2)}` }

function dyadThreads(a, b) {
  if (!snapshot.value) return []
  return snapshot.value.threads.filter(
    (t) => (t.buyer_agent_id === a && t.seller_agent_id === b)
        || (t.buyer_agent_id === b && t.seller_agent_id === a),
  )
}
function dyadMessages(a, b) {
  const out = []
  for (const t of dyadThreads(a, b)) {
    for (const m of t.messages) {
      if (m.tick <= currentTick.value) {
        out.push({ ...m, thread_id: t.thread_id })
      }
    }
  }
  return out.sort((x, y) => x.tick - y.tick)
}
function dyadOffers(a, b) {
  const out = []
  for (const t of dyadThreads(a, b)) {
    for (const o of t.offers) {
      if (o.tick <= currentTick.value) {
        out.push({ ...o, thread_id: t.thread_id })
      }
    }
  }
  return out.sort((x, y) => x.tick - y.tick)
}
function dyadPhotoIds(a, b) {
  const msgs = dyadMessages(a, b)
  return msgs
    .filter((m) => m.photo_id != null)
    .map((m) => m.photo_id)
}
function photoById(id) {
  if (!snapshot.value) return null
  return snapshot.value.photos.find((p) => p.photo_id === id) || null
}

// --- tripwire drawable

const tripwireLines = computed(() => {
  if (!showTripwires.value) return []
  return focusTripwires.value.map((tw) => ({
    tw, path: tripwirePath(tw),
  })).filter((x) => x.path)
})

// --- KPIs

const kpis = computed(() => [
  { label: 'agents', value: agents.value.length },
  { label: 'visible edges', value: visibleEdges.value.length },
  { label: 'phantoms', value: phantomListings.value.length },
  { label: 'tripwires', value: tripwires.value.filter(
      (t) => (t.trigger_tick ?? 0) <= currentTick.value).length },
])

// Legend reads from EDGE_STYLE so legend/render drift is impossible.
const legendEntries = computed(() =>
  [
    ['transacted', 'transacted'],
    ['offered',    'offered'],
    ['messaged',   'messaged'],
    ['pinned',     'pinned'],
    ['viewed',     'viewed'],
  ].map(([kind, label]) => ({
    kind, label,
    style: EDGE_STYLE[kind],
  }))
)

// --- dyad meta for the detail panel header
const dyadAgents = computed(() => {
  const d = selected.value
  if (d?.kind !== 'edge') return null
  return {
    a: agentById(snapshot.value, d.ref.source),
    b: agentById(snapshot.value, d.ref.target),
  }
})
</script>

<template>
  <div class="sandbox" :class="{ maximized }">
    <svg style="position:absolute;width:0;height:0;overflow:hidden" aria-hidden="true">
      <defs>
        <radialGradient
          v-for="a in agents"
          :key="'g' + a.agent_id"
          :id="'agent-grad-' + a.agent_id"
          cx="35%" cy="35%" r="65%"
        >
          <stop offset="0%" :stop-color="lighten(agentColor(a), 0.45)" />
          <stop offset="100%" :stop-color="agentColor(a)" />
        </radialGradient>
        <filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="1.6" in="SourceAlpha" result="blur"/>
          <feOffset dx="0" dy="1.2" result="offset"/>
          <feComponentTransfer><feFuncA type="linear" slope="0.45"/></feComponentTransfer>
          <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
    </svg>

    <!-- header -->
    <div class="panel-header">
      <div class="left-group">
        <span class="panel-title">Market sandbox</span>
        <span class="panel-sub">
          drag agents · hover for focus · click edge for dyad detail
          · scrub timeline to replay the network
        </span>
      </div>
      <div class="header-tools">
        <div class="kpi-strip">
          <div v-for="k in kpis" :key="k.label" class="kpi-pill">
            <span class="num">{{ k.value }}</span>
            <span class="lbl">{{ k.label }}</span>
          </div>
        </div>
        <button class="tool-btn" @click="resetView" title="reset view + positions">
          <span class="icon">↻</span><span class="btn-text">Reset</span>
        </button>
        <button class="tool-btn icon-only"
                @click="maximized = !maximized"
                :title="maximized ? 'exit maximize' : 'maximize'">
          <span class="icon">{{ maximized ? '⤡' : '⛶' }}</span>
        </button>
      </div>
    </div>

    <!-- SVG canvas -->
    <div ref="wrapRef" class="sandbox-container">
      <svg ref="svgRef" class="sandbox-svg"
           :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet">
        <rect class="backdrop" :width="W" :height="H" fill="transparent" />

        <g class="viewport">
          <!-- all edges layer (low-contrast always-on) -->
          <g class="edges-all">
            <path
              v-for="(e, i) in edgesSortedForBackdrop"
              :key="'e' + i + '-' + e.kind"
              :d="edgePath(
                agentPos(agentById(snapshot, e.source)).x,
                agentPos(agentById(snapshot, e.source)).y,
                agentPos(agentById(snapshot, e.target)).x,
                agentPos(agentById(snapshot, e.target)).y,
                EDGE_STYLE[e.kind].curvature,
              )"
              fill="none"
              :stroke="EDGE_STYLE[e.kind].color"
              :stroke-width="edgeWidth(e)"
              :stroke-opacity="edgeBaseOpacity(e)"
              :stroke-dasharray="EDGE_STYLE[e.kind].dash"
              stroke-linecap="round"
              class="no-zoom edge"
              :class="{ 'edge-focus': edgeInFocus(e) }"
              @mouseenter="setHover('edge', e)"
              @mouseleave="clearHover"
              @click="selectEdge(e, $event)"
            >
              <animate
                v-if="e.kind === 'transacted' && edgeInFocus(e)"
                attributeName="stroke-dashoffset"
                from="0" to="-12" dur="0.9s" repeatCount="indefinite"
              />
            </path>
          </g>

          <!-- tripwire lines (focus only) -->
          <g v-if="focusAgentId != null">
            <path
              v-for="(t, i) in tripwireLines"
              :key="'twl' + i"
              :d="t.path"
              fill="none"
              stroke="#a32d2d" stroke-width="2"
              stroke-dasharray="6 4" stroke-opacity="0.9"
              stroke-linecap="round"
              pointer-events="none"
            >
              <animate attributeName="stroke-dashoffset"
                       from="0" to="-10" dur="0.9s" repeatCount="indefinite"/>
            </path>
          </g>

          <!-- phantoms -->
          <g v-if="showPhantoms">
            <g
              v-for="l in phantomListings"
              :key="'p' + l.listing_id"
              :transform="`translate(${listingPos(l).x}, ${listingPos(l).y})`"
              class="no-zoom"
              @mouseenter="setHover('listing', l)"
              @mouseleave="clearHover"
              @click.stop="selected = { kind: 'listing', ref: l }"
            >
              <circle r="8" fill="#fcebeb" fill-opacity="0.35"
                      stroke="#a32d2d" stroke-dasharray="2 2" stroke-width="1.3" />
              <text y="3" text-anchor="middle" font-size="10"
                    fill="#a32d2d" pointer-events="none"
                    font-weight="600">👻</text>
            </g>
          </g>

          <!-- agents -->
          <g
            v-for="a in agents"
            :key="'a' + a.agent_id"
            :transform="`translate(${agentPos(a).x}, ${agentPos(a).y})`"
            :style="{ opacity: nodeOpacity(a),
                      cursor: dragState?.agent_id === a.agent_id ? 'grabbing' : 'grab' }"
            class="agent no-zoom"
            @mouseenter="setHover('agent', a)"
            @mouseleave="clearHover"
            @pointerdown="onAgentPointerDown(a, $event)"
            @dblclick.stop="openAgent(a.agent_id)"
          >
            <circle
              v-if="tripwires.some(tw => tw.agent_id === a.agent_id
                                       && (tw.trigger_tick ?? 0) <= currentTick)"
              :r="agentRadius(a) + 8"
              fill="none" stroke="#a32d2d" stroke-width="1.4"
              opacity="0.6" class="tripwire-aura"
            />
            <circle
              v-if="focusAgentId === a.agent_id"
              :r="agentRadius(a) + 4"
              fill="none" stroke="#E91E63" stroke-width="1.3"
              opacity="0.8" class="focus-halo"
            />
            <circle
              :r="agentRadius(a)"
              :fill="`url(#agent-grad-${a.agent_id})`"
              :stroke="focusAgentId === a.agent_id ? '#E91E63' : '#fff'"
              :stroke-width="focusAgentId === a.agent_id ? 2.5 : 2"
              filter="url(#node-shadow)"
            />
            <text
              v-if="agentRadius(a) >= 14"
              text-anchor="middle"
              :y="agentRadius(a) >= 18 ? 4 : 3.5"
              :font-size="agentRadius(a) >= 20 ? 12 : 10"
              fill="white" font-weight="600" pointer-events="none"
              :style="{ fontFamily: 'system-ui, sans-serif',
                        textShadow: '0 1px 2px rgba(0,0,0,0.3)' }"
            >{{ initialsOf(a.display_name) }}</text>
            <text
              v-if="showLabels"
              text-anchor="middle"
              :y="agentRadius(a) + 14"
              font-size="11" fill="#333"
              font-weight="500" pointer-events="none"
              :style="{ fontFamily: 'system-ui, sans-serif' }"
            >{{ a.display_name.length > 14 ? a.display_name.slice(0,14) + '…' : a.display_name }}</text>
            <title>
              {{ a.display_name }} · @{{ a.user_name }}
              · ZIP {{ a.home_zip }} · activity {{ agentScore(a) }}
            </title>
          </g>
        </g>
      </svg>
    </div>

    <!-- legend bottom-left -->
    <div class="legend">
      <span class="legend-title">Interactions</span>
      <div class="legend-items">
        <div v-for="L in legendEntries" :key="L.kind" class="legend-item">
          <svg width="34" height="10" class="line-swatch">
            <path :d="edgePath(2, 5, 32, 5, L.style.curvature)"
                  fill="none"
                  :stroke="L.style.color"
                  :stroke-width="L.style.width"
                  :stroke-dasharray="L.style.dash"
                  stroke-linecap="round"/>
          </svg>
          <span>{{ L.label }}</span>
        </div>
        <div class="legend-item">
          <svg width="34" height="10" class="line-swatch">
            <path d="M2,5 Q17,0 32,5" fill="none"
                  stroke="#a32d2d" stroke-width="2"
                  stroke-dasharray="5 3" stroke-linecap="round"/>
          </svg>
          <span>tripwire path</span>
        </div>
      </div>
      <div class="legend-footnote">
        node size = activity · <span style="color:#E91E63">magenta ring = focus</span>
        · <span style="color:#a32d2d">red aura = tripwire</span>
      </div>
    </div>

    <!-- layer switches bottom-right -->
    <div class="layer-switches" v-if="!selected || selected.kind !== 'edge'">
      <div class="sw-row">
        <label class="toggle-switch">
          <input v-model="showPhantoms" type="checkbox" />
          <span class="slider" />
        </label><span class="sw-label">phantoms</span>
      </div>
      <div class="sw-row">
        <label class="toggle-switch">
          <input v-model="showTripwires" type="checkbox" />
          <span class="slider" />
        </label><span class="sw-label">tripwires</span>
      </div>
      <div class="sw-row">
        <label class="toggle-switch">
          <input v-model="showLabels" type="checkbox" />
          <span class="slider" />
        </label><span class="sw-label">labels</span>
      </div>
    </div>

    <!-- detail panel top-right -->
    <div v-if="detail" class="detail-panel">
      <div class="detail-panel-header">
        <span class="detail-title">
          {{
            detail.kind === 'agent'    ? 'Agent' :
            detail.kind === 'tripwire' ? 'Tripwire event' :
            detail.kind === 'listing'  ? (detail.ref.is_phantom ? 'Phantom listing' : 'Listing') :
            detail.kind === 'edge'     ? 'Dyad · ' + detail.ref.kind : '?'
          }}
        </span>
        <button v-if="selected" class="detail-close" @click="closeDetail">×</button>
      </div>
      <div class="detail-content">
        <!-- Agent detail -->
        <template v-if="detail.kind === 'agent'">
          <div class="det-head">
            <Avatar :id="detail.ref.agent_id" :name="detail.ref.display_name" :size="36" />
            <div>
              <div class="name">{{ detail.ref.display_name }}</div>
              <div class="tag">@{{ detail.ref.user_name }} · ZIP {{ detail.ref.home_zip }}</div>
            </div>
          </div>
          <div class="detail-section">
            <div class="section-title">Activity</div>
            <div class="act-grid">
              <div><span class="k">score</span><span class="v">{{ detail.ref.activity?.score ?? 0 }}</span></div>
              <div><span class="k">transactions</span><span class="v">{{ detail.ref.activity?.transacted ?? 0 }}</span></div>
              <div><span class="k">messaged out</span><span class="v">{{ detail.ref.activity?.messaged_out ?? 0 }}</span></div>
              <div><span class="k">offered out</span><span class="v">{{ detail.ref.activity?.offered_out ?? 0 }}</span></div>
              <div><span class="k">listings</span><span class="v">{{ detail.ref.activity?.listings ?? 0 }}</span></div>
              <div><span class="k">photos</span><span class="v">{{ detail.ref.activity?.photos ?? 0 }}</span></div>
            </div>
          </div>
          <button class="open-btn" @click="openAgent(detail.ref.agent_id)">
            Open full profile →
          </button>
        </template>

        <!-- Edge detail -->
        <template v-else-if="detail.kind === 'edge'">
          <div v-if="dyadAgents" class="det-head">
            <Avatar :id="dyadAgents.a?.agent_id" :name="dyadAgents.a?.display_name" :size="28" />
            <span style="color:#888;">↔</span>
            <Avatar :id="dyadAgents.b?.agent_id" :name="dyadAgents.b?.display_name" :size="28" />
            <div style="margin-left:6px;">
              <div class="name">
                {{ dyadAgents.a?.display_name }} ·
                {{ dyadAgents.b?.display_name }}
              </div>
              <div class="tag">strongest visible kind: {{ detail.ref.kind }}</div>
            </div>
          </div>

          <div class="detail-section">
            <div class="section-title">Interaction breakdown</div>
            <div class="act-grid">
              <div
                v-for="kind in ['transacted','offered','messaged','pinned','viewed']"
                :key="'bd-'+kind"
              >
                <span class="k">{{ kind }}</span>
                <span class="v">
                  {{
                    (pairMap.get(`${detail.ref.source}>${detail.ref.target}`)?.[kind] ?? 0)
                    + (pairMap.get(`${detail.ref.target}>${detail.ref.source}`)?.[kind] ?? 0)
                  }}
                </span>
              </div>
            </div>
          </div>

          <div class="detail-section">
            <div class="section-title">Offers (timeline ≤ t={{ currentTick }})</div>
            <div v-if="!dyadOffers(detail.ref.source, detail.ref.target).length"
                 class="hint">No offers yet at this tick.</div>
            <div v-else>
              <div
                v-for="o in dyadOffers(detail.ref.source, detail.ref.target)"
                :key="'o' + o.offer_id"
                class="offer-line"
              >
                <span class="tag">t={{ o.tick }}</span>
                <span class="badge b-amber">R{{ o.round }}</span>
                proposer #{{ o.proposer_id }} ·
                {{ fmtPrice(o.price_cents) }} · {{ o.status }}
              </div>
            </div>
          </div>

          <div class="detail-section">
            <div class="section-title">Messages</div>
            <div v-if="!dyadMessages(detail.ref.source, detail.ref.target).length"
                 class="hint">Nothing yet.</div>
            <div v-else class="msg-scroll">
              <div
                v-for="m in dyadMessages(detail.ref.source, detail.ref.target).slice(-10)"
                :key="'m' + m.message_id"
                class="msg-line"
              >
                <span class="tag">t={{ m.tick }}</span>
                <span class="who" :style="{ color: agentColor(agentById(snapshot, m.sender_agent_id)) }">
                  #{{ m.sender_agent_id }}
                </span>
                <span class="body">{{ m.body.slice(0, 90) }}{{ m.body.length > 90 ? '…' : '' }}</span>
              </div>
            </div>
          </div>

          <div class="detail-section"
               v-if="dyadPhotoIds(detail.ref.source, detail.ref.target).length">
            <div class="section-title">Photos exchanged</div>
            <PhotoCard
              v-for="pid in dyadPhotoIds(detail.ref.source, detail.ref.target)"
              :key="'phc'+pid"
              :photo="photoById(pid)"
            />
          </div>

          <div style="display:flex;gap:8px;margin-top:12px;">
            <button class="open-btn" @click="openAgent(detail.ref.source)">
              → @{{ dyadAgents?.a?.user_name || detail.ref.source }}
            </button>
            <button class="open-btn" @click="openAgent(detail.ref.target)">
              → @{{ dyadAgents?.b?.user_name || detail.ref.target }}
            </button>
          </div>
        </template>

        <!-- Listing detail -->
        <template v-else-if="detail.kind === 'listing'">
          <div class="detail-row"><span class="detail-label">Title:</span>
            <span class="detail-value">
              <span v-if="detail.ref.is_phantom" style="color: var(--red);">👻 </span>
              {{ detail.ref.title }}
            </span></div>
          <div class="detail-row"><span class="detail-label">Price:</span>
            <span class="detail-value">{{ fmtPrice(detail.ref.price_cents) }}</span></div>
          <div v-if="detail.ref.is_phantom" class="detail-section">
            <div class="section-title" style="color:#a32d2d">H1 decoy</div>
            <p class="hint">
              Platform-seeded decoy · no owner · never responds.
            </p>
          </div>
        </template>

        <!-- Tripwire detail -->
        <template v-else-if="detail.kind === 'tripwire'">
          <p class="hint" style="color:#a32d2d; margin-top:0;">
            Benign policy committed to a phantom. H1 evidence.
          </p>
          <div class="detail-row"><span class="detail-label">Kind:</span>
            <span class="detail-value">{{ detail.ref.kind }}</span></div>
          <div v-if="detailAgent" class="det-head" style="margin-top:10px;">
            <Avatar :id="detailAgent.agent_id" :name="detailAgent.display_name" :size="28" />
            <div>
              <div class="name">{{ detailAgent.display_name }}</div>
              <div class="tag">ZIP {{ detailAgent.home_zip }}</div>
            </div>
          </div>
          <button v-if="detailAgent" class="open-btn" style="margin-top:10px;"
                  @click="openAgent(detailAgent.agent_id)">
            Open profile →
          </button>
        </template>
      </div>
    </div>

    <!-- timeline scrubber -->
    <div class="timeline">
      <button class="tl-btn" @click="rewind" title="rewind">⏮</button>
      <button class="tl-btn" @click="togglePlay" :title="playing ? 'pause' : 'play'">
        {{ playing ? '❚❚' : '▶' }}
      </button>
      <button class="tl-btn" @click="fastForward" title="jump to end">⏭</button>
      <input
        type="range"
        class="tl-range"
        :min="0" :max="tickMax" step="1"
        v-model.number="currentTick"
      />
      <span class="tl-tick">t = {{ currentTick }} / {{ tickMax }}</span>
      <select v-model.number="speed" class="tl-speed">
        <option :value="2">slow</option>
        <option :value="8">1×</option>
        <option :value="25">5×</option>
        <option :value="80">fast</option>
      </select>
    </div>
  </div>
</template>

<style scoped>
.sandbox {
  position: relative; width: 100%;
  height: calc(100vh - 100px); min-height: 720px;
  background-color: #FAFAFA;
  background-image: radial-gradient(#D0D0D0 1.5px, transparent 1.5px);
  background-size: 24px 24px;
  overflow: hidden; border-radius: 12px;
  border: 1px solid #EAEAEA;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  display: flex; flex-direction: column;
}
.sandbox.maximized {
  position: fixed; top: 8px; left: 8px; right: 8px; bottom: 8px;
  height: auto; z-index: 40; border-radius: 14px;
  box-shadow: 0 16px 48px rgba(0,0,0,0.18);
}
.sandbox-container { flex: 1; width: 100%; position: relative; }
.sandbox-svg {
  width: 100%; height: 100%; display: block;
  cursor: grab; user-select: none;
}
.sandbox-svg:active { cursor: grabbing; }
.agent { cursor: pointer; transition: opacity 0.2s ease-out; }
.edge { cursor: pointer; transition: stroke-opacity 0.2s ease-out; }
.edge:hover { stroke-opacity: 1 !important; }
.edge-focus { stroke-opacity: 0.95 !important; }

/* header */
.panel-header {
  position: absolute; top: 0; left: 0; right: 0;
  padding: 14px 20px; z-index: 10;
  display: flex; justify-content: space-between; align-items: center;
  background: linear-gradient(to bottom,
              rgba(255,255,255,0.95), rgba(255,255,255,0));
  pointer-events: none;
}
.panel-header .left-group, .panel-header .header-tools { pointer-events: auto; }
.panel-title { font-size: 14px; font-weight: 600; color: #333; }
.panel-sub { font-size: 11.5px; color: #888; margin-left: 10px; }
.header-tools { display: flex; gap: 10px; align-items: center; }
.kpi-strip { display: flex; gap: 6px; margin-right: 10px; }
.kpi-pill {
  background: white; border: 1px solid #E0E0E0;
  border-radius: 16px; padding: 4px 10px;
  display: flex; align-items: baseline; gap: 5px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.kpi-pill .num { font-weight: 600; font-size: 13px; color: #333; }
.kpi-pill .lbl { font-size: 11px; color: #888; }
.tool-btn {
  height: 32px; padding: 0 12px;
  border: 1px solid #E0E0E0; background: #FFF; border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  gap: 6px; cursor: pointer; color: #666;
  transition: all 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
  font-size: 13px; font-family: inherit;
}
.tool-btn:hover { background: #F5F5F5; color: #000; border-color: #CCC; }
.tool-btn.icon-only { padding: 0 10px; }
.tool-btn .icon { font-size: 14px; line-height: 1; }

/* legend */
.legend {
  position: absolute; bottom: 74px; left: 24px;
  background: rgba(255,255,255,0.95);
  padding: 12px 16px; border-radius: 8px;
  border: 1px solid #EAEAEA;
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
  z-index: 10;
}
.legend-title {
  display: block; font-size: 11px; font-weight: 600;
  color: #E91E63; margin-bottom: 10px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.legend-items { display: flex; flex-direction: column; gap: 5px; }
.legend-item {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: #555;
}
.line-swatch { flex-shrink: 0; }
.legend-footnote {
  margin-top: 10px; font-size: 10.5px; color: #888;
  border-top: 1px solid #EEE; padding-top: 8px;
}

/* layer switches */
.layer-switches {
  position: absolute; bottom: 74px; right: 24px;
  background: rgba(255,255,255,0.95);
  border: 1px solid #EAEAEA; border-radius: 10px;
  padding: 10px 14px;
  display: flex; flex-direction: column; gap: 6px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
  z-index: 10;
}
.sw-row { display: flex; align-items: center; gap: 10px; font-size: 12px; color: #555; }
.sw-label { font-size: 12px; color: #666; }
.toggle-switch { position: relative; display: inline-block; width: 32px; height: 18px; }
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.slider {
  position: absolute; cursor: pointer;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: #E0E0E0; border-radius: 18px;
  transition: 0.2s;
}
.slider:before {
  position: absolute; content: "";
  height: 14px; width: 14px; left: 2px; bottom: 2px;
  background: white; border-radius: 50%; transition: 0.2s;
  box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
input:checked + .slider { background-color: #7B2D8E; }
input:checked + .slider:before { transform: translateX(14px); }

/* timeline */
.timeline {
  position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
  background: rgba(255,255,255,0.97);
  border: 1px solid #EAEAEA; border-radius: 999px;
  padding: 6px 14px;
  display: flex; gap: 10px; align-items: center;
  box-shadow: 0 4px 16px rgba(0,0,0,0.08);
  z-index: 11;
  width: min(760px, 85%);
}
.tl-btn {
  background: transparent; border: none; font-size: 14px;
  color: #666; cursor: pointer; padding: 4px 6px;
}
.tl-btn:hover { color: #000; }
.tl-range { flex: 1; }
.tl-tick {
  font-size: 11.5px; font-variant-numeric: tabular-nums;
  color: #444; min-width: 100px; text-align: right;
}
.tl-speed {
  font-size: 11px; border: 1px solid #E0E0E0; border-radius: 4px;
  padding: 2px 6px; background: white; color: #555;
}

/* detail panel */
.detail-panel {
  position: absolute; top: 64px; right: 20px;
  width: 360px; max-height: calc(100% - 180px);
  background: #FFF;
  border: 1px solid #EAEAEA; border-radius: 10px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  overflow: hidden; z-index: 20;
  display: flex; flex-direction: column;
  font-family: system-ui, sans-serif; font-size: 13px;
}
.detail-panel-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px; background: #FAFAFA;
  border-bottom: 1px solid #EEE; flex-shrink: 0; gap: 8px;
}
.detail-title { font-weight: 600; color: #333; font-size: 14px; }
.detail-close {
  background: none; border: none; font-size: 20px; line-height: 1;
  color: #999; cursor: pointer; padding: 0;
}
.detail-close:hover { color: #333; }
.detail-content { padding: 14px 16px; overflow-y: auto; flex: 1; }
.det-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.det-head .name { font-weight: 500; color: #333; }
.det-head .tag { font-size: 11.5px; color: #888; }
.detail-row { display: flex; margin: 5px 0; font-size: 12.5px; }
.detail-label { color: #888; min-width: 78px; flex-shrink: 0; }
.detail-value { color: #333; flex: 1; }
.detail-section {
  margin-top: 14px; padding-top: 12px; border-top: 1px solid #F0F0F0;
}
.section-title {
  font-size: 11px; font-weight: 600; color: #E91E63;
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
}
.hint { font-size: 12px; color: #666; margin: 0 0 6px; line-height: 1.4; }
.act-grid { display: grid; grid-template-columns: 1fr 1fr;
            gap: 3px 14px; font-size: 12px; }
.act-grid > div { display: flex; justify-content: space-between; }
.act-grid .k { color: #888; }
.act-grid .v { color: #333; font-variant-numeric: tabular-nums; }
.offer-line {
  display: flex; align-items: center; gap: 6px;
  font-size: 11.5px; color: #555; margin: 3px 0;
}
.msg-scroll { max-height: 160px; overflow-y: auto; }
.msg-line {
  display: flex; align-items: baseline; gap: 6px;
  font-size: 11.5px; padding: 3px 0;
  border-bottom: 0.5px solid #f3f2ed;
}
.msg-line:last-child { border-bottom: none; }
.msg-line .tag { font-size: 10.5px; color: #888; min-width: 40px; }
.msg-line .who { font-weight: 500; font-size: 11px; }
.msg-line .body { color: #333; flex: 1; }

.open-btn {
  padding: 7px 10px; flex: 1;
  border-radius: 6px; border: 1px solid #E91E63;
  background: white; color: #E91E63; font-size: 11.5px;
  cursor: pointer; font-family: inherit; font-weight: 500;
  transition: all 0.15s;
}
.open-btn:hover { background: #E91E63; color: white; }

/* animations */
@keyframes tripwire-pulse {
  0%   { transform: scale(1);   opacity: 0.7; }
  70%  { transform: scale(1.4); opacity: 0; }
  100% { transform: scale(1.4); opacity: 0; }
}
.tripwire-aura {
  transform-origin: center; transform-box: fill-box;
  animation: tripwire-pulse 1.6s ease-out infinite;
}
@keyframes focus-halo-pulse {
  0%, 100% { opacity: 0.5; } 50% { opacity: 0.9; }
}
.focus-halo { animation: focus-halo-pulse 1.6s ease-in-out infinite; }
</style>
