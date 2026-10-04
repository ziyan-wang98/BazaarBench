<script setup>
// Geographic operations console. Hallmarks:
//
//   * dot-grid "blueprint" background
//   * floating panel header — title left, tool buttons right,
//     white→transparent gradient so markers can sit underneath
//   * bottom-left legend card with colored dots
//   * top-right detail panel (320px) with X close, type badge,
//     section-structured content
//   * iOS-style toggle switches for layer controls
//   * magenta ring (#E91E63) on selected agent, thread edges
//     around selection highlight in magenta too
//   * 3px click-vs-drag threshold so light clicks don't nudge
//
// Distinct from MiroFish: we keep the geographic projection;
// their GraphPanel is a knowledge graph without geography.
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import * as d3 from 'd3'
import Avatar from '../components/Avatar.vue'
import {
  agentById,
  listingById,
  useSnapshot,
} from '../composables/useData.js'

const { snapshot } = useSnapshot()
const router = useRouter()

const svgRef = ref(null)
const wrapRef = ref(null)

const hovered = ref(null)
const selected = ref(null)
const detail = computed(() => selected.value || hovered.value)

// Layer toggles.
const showListings = ref(true)
const showPhantoms = ref(true)
const showTripwires = ref(true)
const showLabels = ref(true)

const maximized = ref(false)

// ---- data ---------------------------------------------------------------

const agents = computed(() => snapshot.value?.agents || [])
const listings = computed(() => snapshot.value?.listings || [])
const realListings = computed(() =>
  listings.value.filter(
    (l) => !l.is_phantom && !(l.location_lat === 0 && l.location_lng === 0),
  ),
)
const phantomListings = computed(
  () => listings.value.filter((l) => l.is_phantom),
)
const tripwires = computed(() => snapshot.value?.events_summary?.tripwires || [])

const threadEdges = computed(() => {
  if (!snapshot.value) return []
  return snapshot.value.graph.edges.filter((e) => e.kind === 'thread')
})

const connected = computed(() => {
  const m = new Map()
  for (const e of threadEdges.value) {
    if (!m.has(e.source)) m.set(e.source, new Set())
    if (!m.has(e.target)) m.set(e.target, new Set())
    m.get(e.source).add(e.target)
    m.get(e.target).add(e.source)
  }
  return m
})

const tripwiresByAgent = computed(() => {
  const m = new Map()
  for (const tw of tripwires.value) {
    if (!m.has(tw.agent_id)) m.set(tw.agent_id, [])
    m.get(tw.agent_id).push(tw)
  }
  return m
})

// ---- bbox & projection --------------------------------------------------

const bbox = computed(() => {
  const pts = []
  for (const a of agents.value) pts.push([a.home_lat, a.home_lng])
  for (const l of realListings.value) pts.push([l.location_lat, l.location_lng])
  if (!pts.length) return { latMin: -1, latMax: 1, lngMin: -1, lngMax: 1 }
  const lats = pts.map((p) => p[0])
  const lngs = pts.map((p) => p[1])
  let latMin = Math.min(...lats), latMax = Math.max(...lats)
  let lngMin = Math.min(...lngs), lngMax = Math.max(...lngs)
  const latSpan = Math.max(0.2, latMax - latMin)
  const lngSpan = Math.max(0.2, lngMax - lngMin)
  latMin -= latSpan * 0.08; latMax += latSpan * 0.08
  lngMin -= lngSpan * 0.08; lngMax += lngSpan * 0.08
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
    const fx = 0.2 + 0.6 * hash01(p.listing_id, 17)
    const fy = 0.2 + 0.6 * hash01(p.listing_id, 137)
    m.set(p.listing_id, {
      lat: b.latMin + (b.latMax - b.latMin) * fy,
      lng: b.lngMin + (b.lngMax - b.lngMin) * fx,
    })
  }
  return m
})

const W = 1200
const H = 700
const PAD = 44

function project(lat, lng) {
  const b = bbox.value
  const x = PAD + (lng - b.lngMin) / (b.lngMax - b.lngMin) * (W - 2 * PAD)
  const y = PAD + (b.latMax - lat) / (b.latMax - b.latMin) * (H - 2 * PAD)
  return { x, y }
}

function agentPos(a) { return project(a.home_lat, a.home_lng) }
function listingPos(l) {
  if (l.is_phantom) {
    const p = phantomPositions.value.get(l.listing_id)
    return p ? project(p.lat, p.lng) : { x: 0, y: 0 }
  }
  return project(l.location_lat, l.location_lng)
}

// ---- focus highlighting -------------------------------------------------

const focusAgentId = computed(() => {
  if (selected.value?.kind === 'agent') return selected.value.ref.agent_id
  if (hovered.value?.kind === 'agent')  return hovered.value.ref.agent_id
  return null
})
const hasFocus = computed(() => focusAgentId.value != null)

function isFocusConnected(agentId) {
  if (!hasFocus.value) return true
  if (agentId === focusAgentId.value) return true
  return connected.value.get(focusAgentId.value)?.has(agentId) ?? false
}

function edgeOpacity(e) {
  if (!hasFocus.value) return 0
  return (e.source === focusAgentId.value || e.target === focusAgentId.value)
    ? 0.85 : 0
}

function nodeOpacity(agentId) {
  if (!hasFocus.value) return 1
  return isFocusConnected(agentId) ? 1 : 0.18
}

const focusTripwires = computed(() => {
  if (!hasFocus.value) return []
  return tripwiresByAgent.value.get(focusAgentId.value) || []
})

function tripwirePath(tw) {
  const a = agentById(snapshot.value, tw.agent_id)
  const l = listingById(snapshot.value, tw.listing_id)
  if (!a || !l) return null
  const p1 = agentPos(a)
  const p2 = listingPos(l)
  return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
}

// ---- d3.zoom ------------------------------------------------------------

let zoomBehaviour = null

function setupZoom() {
  const svg = d3.select(svgRef.value)
  const vp = svg.select('g.viewport')
  zoomBehaviour = d3.zoom()
    .scaleExtent([0.4, 8])
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

function resetZoom() {
  if (!zoomBehaviour || !svgRef.value) return
  d3.select(svgRef.value)
    .transition().duration(400)
    .call(zoomBehaviour.transform, d3.zoomIdentity)
}

onMounted(() => {
  // Wait a tick for SVG to mount.
  setTimeout(setupZoom, 0)
})
onUnmounted(() => {})

// ---- interaction --------------------------------------------------------

// 3-px click-vs-drag threshold on agent groups — if a user
// mousedown-up-moves < 3 px total we treat it as a click.
const mouseDown = ref(null)

function onAgentDown(agent, event) {
  mouseDown.value = { x: event.clientX, y: event.clientY, agent }
}
function onAgentUp(agent, event) {
  const md = mouseDown.value
  mouseDown.value = null
  if (!md || md.agent !== agent) return
  const dx = event.clientX - md.x
  const dy = event.clientY - md.y
  if (dx * dx + dy * dy < 9) {
    // click, not drag
    event.stopPropagation()
    selected.value = { kind: 'agent', ref: agent }
  }
}

function setHover(kind, ref) { hovered.value = { kind, ref } }
function clearHover() { hovered.value = null }
function setSelected(kind, ref, event) {
  event?.stopPropagation()
  selected.value = { kind, ref }
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

function fmtPrice(cents) { return `$${(cents / 100).toFixed(2)}` }

const kpis = computed(() => [
  { label: 'agents',    value: agents.value.length },
  { label: 'listings',  value: realListings.value.length },
  { label: 'phantoms',  value: phantomListings.value.length },
  { label: 'tripwires', value: tripwires.value.length },
])
</script>

<template>
  <div class="graph-panel" :class="{ maximized }">
    <!-- floating panel header -->
    <div class="panel-header">
      <div class="left-group">
        <span class="panel-title">Geographic operations map</span>
        <span class="panel-sub">
          drag to pan · scroll to zoom · hover to focus
        </span>
      </div>
      <div class="header-tools">
        <div class="kpi-strip">
          <div v-for="k in kpis" :key="k.label" class="kpi-pill">
            <span class="num">{{ k.value }}</span>
            <span class="lbl">{{ k.label }}</span>
          </div>
        </div>
        <button class="tool-btn" @click="resetZoom" title="reset view">
          <span class="icon">↻</span>
          <span class="btn-text">Reset</span>
        </button>
        <button
          class="tool-btn icon-only"
          @click="maximized = !maximized"
          :title="maximized ? 'exit maximize' : 'maximize'"
        >
          <span class="icon">{{ maximized ? '⤡' : '⛶' }}</span>
        </button>
      </div>
    </div>

    <!-- SVG canvas -->
    <div ref="wrapRef" class="graph-container">
      <svg ref="svgRef" class="graph-svg" :viewBox="`0 0 ${W} ${H}`"
           preserveAspectRatio="xMidYMid meet">
        <rect class="backdrop" :width="W" :height="H" fill="transparent" />
        <g class="viewport">
          <!-- thread edges — only visible when an agent is focused -->
          <g v-if="hasFocus">
            <line
              v-for="(e, i) in threadEdges"
              :key="'te' + i"
              :x1="agentPos(agentById(snapshot, e.source) || {home_lat:0,home_lng:0}).x"
              :y1="agentPos(agentById(snapshot, e.source) || {home_lat:0,home_lng:0}).y"
              :x2="agentPos(agentById(snapshot, e.target) || {home_lat:0,home_lng:0}).x"
              :y2="agentPos(agentById(snapshot, e.target) || {home_lat:0,home_lng:0}).y"
              stroke="#E91E63"
              :stroke-width="Math.min(4, 1.5 + Math.sqrt(e.weight))"
              :stroke-opacity="edgeOpacity(e)"
              stroke-linecap="round"
              pointer-events="none"
            />
          </g>

          <!-- tripwire "attack paths" — red dashed, animated -->
          <g v-if="hasFocus && showTripwires">
            <template v-for="(tw, i) in focusTripwires" :key="'twline'+i">
              <line
                v-if="tripwirePath(tw)"
                v-bind="tripwirePath(tw)"
                stroke="#a32d2d" stroke-width="1.8"
                stroke-dasharray="6 4" stroke-opacity="0.9"
                pointer-events="none"
              >
                <animate attributeName="stroke-dashoffset"
                         from="0" to="-10" dur="0.9s" repeatCount="indefinite"/>
              </line>
            </template>
          </g>

          <!-- real listings -->
          <g v-if="showListings">
            <circle
              v-for="l in realListings"
              :key="'l' + l.listing_id"
              :cx="listingPos(l).x" :cy="listingPos(l).y"
              r="3.5" fill="#C5283D" fill-opacity="0.7"
              class="no-zoom"
              @mouseenter="setHover('listing', l)"
              @mouseleave="clearHover"
              @click="setSelected('listing', l, $event)"
            />
          </g>

          <!-- phantoms — deterministically placed inside bbox -->
          <g v-if="showPhantoms">
            <circle
              v-for="l in phantomListings"
              :key="'p' + l.listing_id"
              :cx="listingPos(l).x" :cy="listingPos(l).y"
              r="6"
              fill="#fcebeb" fill-opacity="0.35"
              stroke="#a32d2d" stroke-dasharray="2 2" stroke-width="1.2"
              class="no-zoom"
              @mouseenter="setHover('listing', l)"
              @mouseleave="clearHover"
              @click="setSelected('listing', l, $event)"
            />
          </g>

          <!-- agents -->
          <g
            v-for="a in agents"
            :key="'a' + a.agent_id"
            :transform="`translate(${agentPos(a).x}, ${agentPos(a).y})`"
            :style="{ opacity: nodeOpacity(a.agent_id) }"
            class="agent no-zoom"
            @mouseenter="setHover('agent', a)"
            @mouseleave="clearHover"
            @mousedown="onAgentDown(a, $event)"
            @mouseup="onAgentUp(a, $event)"
            @dblclick.stop="openAgent(a.agent_id)"
          >
            <circle
              :r="selected?.kind === 'agent' && selected.ref.agent_id === a.agent_id ? 10 : 8"
              :fill="a.status === 'active' ? '#1A936F' : '#C5283D'"
              fill-opacity="0.9"
              :stroke="selected?.kind === 'agent' && selected.ref.agent_id === a.agent_id
                      ? '#E91E63'
                      : (tripwiresByAgent.get(a.agent_id)?.length ? '#a32d2d' : '#fff')"
              :stroke-width="selected?.kind === 'agent' && selected.ref.agent_id === a.agent_id
                            ? 3.5
                            : (tripwiresByAgent.get(a.agent_id)?.length ? 2.5 : 2)"
            />
            <text
              v-if="showLabels"
              :x="12" y="4"
              font-size="11" fill="#333"
              font-weight="500"
              pointer-events="none"
              :style="{ fontFamily: 'system-ui, sans-serif' }"
            >{{ a.display_name.length > 12 ? a.display_name.slice(0,12) + '…' : a.display_name }}</text>
          </g>
        </g>
      </svg>
    </div>

    <!-- bottom-left legend -->
    <div class="graph-legend">
      <span class="legend-title">Markers</span>
      <div class="legend-items">
        <div class="legend-item">
          <span class="legend-dot" style="background: #1A936F"></span>
          <span class="legend-label">agent · active</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot" style="background: #C5283D"></span>
          <span class="legend-label">agent · banned</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot" style="background: #C5283D; opacity: 0.6"></span>
          <span class="legend-label">listing</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot phantom-dot"></span>
          <span class="legend-label">phantom</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot" style="background: #E91E63"></span>
          <span class="legend-label">focused · selected</span>
        </div>
      </div>
    </div>

    <!-- layer controls (top-right below detail) -->
    <div class="layer-switches" v-if="!selected">
      <div class="sw-row">
        <label class="toggle-switch">
          <input v-model="showListings" type="checkbox" />
          <span class="slider" />
        </label>
        <span class="sw-label">listings</span>
      </div>
      <div class="sw-row">
        <label class="toggle-switch">
          <input v-model="showPhantoms" type="checkbox" />
          <span class="slider" />
        </label>
        <span class="sw-label">phantoms</span>
      </div>
      <div class="sw-row">
        <label class="toggle-switch">
          <input v-model="showTripwires" type="checkbox" />
          <span class="slider" />
        </label>
        <span class="sw-label">tripwires</span>
      </div>
      <div class="sw-row">
        <label class="toggle-switch">
          <input v-model="showLabels" type="checkbox" />
          <span class="slider" />
        </label>
        <span class="sw-label">labels</span>
      </div>
    </div>

    <!-- top-right detail panel -->
    <div v-if="detail" class="detail-panel">
      <div class="detail-panel-header">
        <span class="detail-title">
          {{
            detail.kind === 'agent' ? 'Agent' :
            detail.kind === 'tripwire' ? 'Tripwire event' :
            'Listing'
          }}
        </span>
        <span v-if="detail.kind === 'agent'" class="detail-type-badge"
              :style="{ background: detail.ref.status === 'active' ? '#1A936F' : '#C5283D' }">
          {{ detail.ref.status }}
        </span>
        <span v-else-if="detail.kind === 'listing' && detail.ref.is_phantom"
              class="detail-type-badge" style="background: #a32d2d">
          phantom
        </span>
        <span v-else-if="detail.kind === 'tripwire'"
              class="detail-type-badge" style="background: #a32d2d">
          H1 evidence
        </span>
        <button v-if="selected" class="detail-close" @click="closeDetail">×</button>
      </div>
      <div class="detail-content">
        <template v-if="detail.kind === 'agent'">
          <div class="det-head">
            <Avatar :id="detail.ref.agent_id" :name="detail.ref.display_name" :size="36" />
            <div>
              <div class="name">{{ detail.ref.display_name }}</div>
              <div class="tag">@{{ detail.ref.user_name }}</div>
            </div>
          </div>
          <div class="detail-row"><span class="detail-label">ZIP:</span>
            <span class="detail-value">{{ detail.ref.home_zip }}</span></div>
          <div class="detail-row"><span class="detail-label">Lat/Lng:</span>
            <span class="detail-value">{{ detail.ref.home_lat.toFixed(3) }}, {{ detail.ref.home_lng.toFixed(3) }}</span></div>
          <div class="detail-row"><span class="detail-label">Device:</span>
            <span class="detail-value">{{ detail.ref.device }}</span></div>

          <div class="detail-section">
            <div class="section-title">Behaviour</div>
            <div class="kv-mini">
              <span class="k">activity</span>
              <div class="bar"><span :style="{width: (detail.ref.activity_rate*100)+'%'}"/></div>
              <span class="v">{{ detail.ref.activity_rate.toFixed(2) }}</span>
              <span class="k">privacy</span>
              <div class="bar"><span :style="{width: (detail.ref.privacy_awareness*100)+'%'}"/></div>
              <span class="v">{{ detail.ref.privacy_awareness.toFixed(2) }}</span>
            </div>
          </div>

          <div class="detail-section">
            <div class="section-title">Network</div>
            <div class="detail-row"><span class="detail-label">Threads:</span>
              <span class="detail-value">{{ connected.get(detail.ref.agent_id)?.size ?? 0 }}</span></div>
            <div class="detail-row">
              <span class="detail-label">Tripwires:</span>
              <span class="detail-value"
                :style="{color: tripwiresByAgent.get(detail.ref.agent_id)?.length ? '#a32d2d' : 'inherit'}">
                {{ tripwiresByAgent.get(detail.ref.agent_id)?.length ?? 0 }}
              </span>
            </div>
          </div>
          <button class="open-btn" @click="openAgent(detail.ref.agent_id)">
            Open full profile →
          </button>
        </template>

        <template v-else-if="detail.kind === 'listing'">
          <div class="detail-row"><span class="detail-label">Title:</span>
            <span class="detail-value">
              <span v-if="detail.ref.is_phantom" style="color: var(--red);">👻 </span>
              {{ detail.ref.title }}
            </span></div>
          <div class="detail-row"><span class="detail-label">Price:</span>
            <span class="detail-value">{{ fmtPrice(detail.ref.price_cents) }}</span></div>
          <div class="detail-row"><span class="detail-label">Category:</span>
            <span class="detail-value">{{ detail.ref.category }}</span></div>
          <div class="detail-row"><span class="detail-label">ZIP:</span>
            <span class="detail-value">{{ detail.ref.location_zip }}</span></div>
          <div class="detail-row"><span class="detail-label">Condition:</span>
            <span class="detail-value">{{ detail.ref.condition }}</span></div>
          <div v-if="detail.ref.is_phantom" class="detail-section">
            <div class="section-title" style="color: #a32d2d">H1 decoy</div>
            <p class="hint">Platform-seeded decoy listing · no owner ·
              never responds. Agents engaging this are the phantom-
              tripwire evidence for spontaneous drift.</p>
          </div>
          <div v-else-if="detailAgent" class="detail-section">
            <div class="section-title">Seller</div>
            <div class="det-head">
              <Avatar :id="detailAgent.agent_id" :name="detailAgent.display_name" :size="28" />
              <div>
                <div class="name">{{ detailAgent.display_name }}</div>
                <div class="tag">ZIP {{ detailAgent.home_zip }}</div>
              </div>
            </div>
            <button class="open-btn" @click="openAgent(detailAgent.agent_id)">
              Open seller profile →
            </button>
          </div>
        </template>

        <template v-else-if="detail.kind === 'tripwire'">
          <p class="hint" style="color: #a32d2d; margin-top: 0;">
            Benign policy committed to a phantom listing. This is H1
            evidence: spontaneous drift without any adversary.
          </p>
          <div class="detail-row"><span class="detail-label">Kind:</span>
            <span class="detail-value">{{ detail.ref.kind }}</span></div>
          <div class="detail-row"><span class="detail-label">Listing:</span>
            <span class="detail-value">#{{ detail.ref.listing_id }}</span></div>
          <div v-if="detailAgent" class="detail-section">
            <div class="section-title">Drifting agent</div>
            <div class="det-head">
              <Avatar :id="detailAgent.agent_id" :name="detailAgent.display_name" :size="28" />
              <div>
                <div class="name">{{ detailAgent.display_name }}</div>
                <div class="tag">ZIP {{ detailAgent.home_zip }}</div>
              </div>
            </div>
            <button class="open-btn" @click="openAgent(detailAgent.agent_id)">
              Open profile →
            </button>
          </div>
        </template>
      </div>
    </div>

    <!-- focus hud (bottom right) -->
    <div v-if="hasFocus" class="focus-hud">
      showing connections for
      <b>{{ agentById(snapshot, focusAgentId)?.display_name }}</b>
      · click empty canvas to clear
    </div>
  </div>
</template>

<style scoped>
/* ---- full-bleed panel with dot-grid blueprint background -------------- */
.graph-panel {
  position: relative;
  width: 100%;
  /* tall enough to feel like a workspace, grows when maximized */
  height: calc(100vh - 120px);
  min-height: 640px;
  background-color: #FAFAFA;
  background-image: radial-gradient(#D0D0D0 1.5px, transparent 1.5px);
  background-size: 24px 24px;
  overflow: hidden;
  border-radius: 12px;
  border: 1px solid #EAEAEA;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.graph-panel.maximized {
  position: fixed; top: 8px; left: 8px; right: 8px; bottom: 8px;
  height: auto; z-index: 40; border-radius: 14px;
  box-shadow: 0 16px 48px rgba(0,0,0,0.18);
}

.graph-container { width: 100%; height: 100%; }
.graph-svg {
  width: 100%; height: 100%; display: block;
  cursor: grab; user-select: none;
}
.graph-svg:active { cursor: grabbing; }

.agent { cursor: pointer; transition: opacity 0.15s ease-out; }

/* ---- floating panel header -------------------------------------------- */
.panel-header {
  position: absolute;
  top: 0; left: 0; right: 0;
  padding: 14px 20px;
  z-index: 10;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(to bottom,
              rgba(255,255,255,0.95), rgba(255,255,255,0));
  pointer-events: none;
}
.panel-header .left-group,
.panel-header .header-tools { pointer-events: auto; }
.panel-title { font-size: 14px; font-weight: 600; color: #333; }
.panel-sub   { font-size: 11.5px; color: #888; margin-left: 10px; }
.header-tools { display: flex; gap: 10px; align-items: center; }

.kpi-strip { display: flex; gap: 6px; margin-right: 10px; }
.kpi-pill {
  background: white; border: 1px solid #E0E0E0;
  border-radius: 16px; padding: 4px 10px;
  display: flex; align-items: baseline; gap: 5px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.kpi-pill .num  { font-weight: 600; font-size: 13px; color: #333; }
.kpi-pill .lbl  { font-size: 11px; color: #888; }

.tool-btn {
  height: 32px; padding: 0 12px;
  border: 1px solid #E0E0E0; background: #FFF;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  gap: 6px; cursor: pointer; color: #666;
  transition: all 0.2s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.02);
  font-size: 13px; font-family: inherit;
}
.tool-btn:hover { background: #F5F5F5; color: #000; border-color: #CCC; }
.tool-btn.icon-only { padding: 0 10px; }
.tool-btn .btn-text { font-size: 12px; }
.tool-btn .icon { font-size: 14px; line-height: 1; }

/* ---- legend bottom-left ----------------------------------------------- */
.graph-legend {
  position: absolute;
  bottom: 24px; left: 24px;
  background: rgba(255,255,255,0.95);
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid #EAEAEA;
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
  z-index: 10;
}
.legend-title {
  display: block; font-size: 11px; font-weight: 600;
  color: #E91E63; margin-bottom: 10px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.legend-items {
  display: flex; flex-wrap: wrap; gap: 8px 14px; max-width: 320px;
}
.legend-item {
  display: flex; align-items: center; gap: 6px;
  font-size: 12px; color: #555;
}
.legend-dot {
  width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
}
.legend-dot.phantom-dot {
  background: transparent;
  border: 1.2px dashed #a32d2d;
}

/* ---- iOS layer switches bottom-right --------------------------------- */
.layer-switches {
  position: absolute;
  bottom: 24px; right: 24px;
  background: rgba(255,255,255,0.95);
  border: 1px solid #EAEAEA;
  border-radius: 10px;
  padding: 10px 14px;
  display: flex; flex-direction: column; gap: 6px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
  z-index: 10;
}
.sw-row { display: flex; align-items: center; gap: 10px; font-size: 12px; color: #555; }
.sw-label { font-size: 12px; color: #666; }

.toggle-switch {
  position: relative; display: inline-block; width: 32px; height: 18px;
}
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.slider {
  position: absolute; cursor: pointer;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: #E0E0E0; border-radius: 18px;
  transition: 0.2s;
}
.slider:before {
  position: absolute; content: "";
  height: 14px; width: 14px;
  left: 2px; bottom: 2px;
  background: white; border-radius: 50%;
  transition: 0.2s;
  box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
input:checked + .slider { background-color: #7B2D8E; }
input:checked + .slider:before { transform: translateX(14px); }

/* ---- detail panel top-right ------------------------------------------ */
.detail-panel {
  position: absolute;
  top: 64px; right: 20px;
  width: 320px; max-height: calc(100% - 130px);
  background: #FFF;
  border: 1px solid #EAEAEA;
  border-radius: 10px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  overflow: hidden;
  z-index: 20;
  display: flex; flex-direction: column;
  font-family: system-ui, sans-serif; font-size: 13px;
}
.detail-panel-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px;
  background: #FAFAFA;
  border-bottom: 1px solid #EEE;
  flex-shrink: 0; gap: 8px;
}
.detail-title { font-weight: 600; color: #333; font-size: 14px; }
.detail-type-badge {
  padding: 3px 10px; border-radius: 12px;
  font-size: 11px; font-weight: 500; color: #fff;
  margin-left: auto; margin-right: 4px;
}
.detail-close {
  background: none; border: none; font-size: 20px; line-height: 1;
  color: #999; cursor: pointer; padding: 0; margin-left: auto;
  transition: color 0.2s;
}
.detail-close:hover { color: #333; }
.detail-content { padding: 14px 16px; overflow-y: auto; flex: 1; }
.det-head {
  display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
}
.det-head .name { font-weight: 500; color: #333; }
.det-head .tag  { font-size: 11.5px; color: #888; }
.detail-row {
  display: flex; margin: 5px 0; font-size: 12.5px;
}
.detail-label { color: #888; min-width: 78px; flex-shrink: 0; }
.detail-value { color: #333; flex: 1; }
.detail-section { margin-top: 14px; padding-top: 12px;
                  border-top: 1px solid #F0F0F0; }
.section-title {
  font-size: 11px; font-weight: 600; color: #E91E63;
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
}
.hint { font-size: 12px; color: #666; margin: 0 0 8px; line-height: 1.4; }

.kv-mini {
  display: grid;
  grid-template-columns: 70px 1fr 38px;
  gap: 5px 10px; font-size: 12px; align-items: center;
}
.kv-mini .k { color: #888; }
.kv-mini .v { color: #333; text-align: right; font-variant-numeric: tabular-nums; }
.kv-mini .bar {
  height: 6px; background: #ececea; border-radius: 3px; overflow: hidden;
}
.kv-mini .bar > span {
  display: block; height: 100%;
  background: linear-gradient(to right, #1A936F, #7B2D8E);
}

.open-btn {
  margin-top: 14px; padding: 8px 10px; width: 100%;
  border-radius: 6px; border: 1px solid #E91E63;
  background: white; color: #E91E63; font-size: 12px;
  cursor: pointer; font-family: inherit; font-weight: 500;
  transition: all 0.15s;
}
.open-btn:hover { background: #E91E63; color: white; }

/* ---- focus hud -------------------------------------------------------- */
.focus-hud {
  position: absolute;
  left: 50%; bottom: 24px; transform: translateX(-50%);
  background: rgba(233, 30, 99, 0.95); color: white;
  border-radius: 16px; padding: 6px 14px;
  font-size: 12px; pointer-events: none;
  box-shadow: 0 4px 12px rgba(233, 30, 99, 0.3);
  z-index: 11;
}
.focus-hud b { font-weight: 600; }
</style>
