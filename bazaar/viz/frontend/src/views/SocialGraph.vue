<script setup>
// V3 — d3-force social graph.
//
// Nodes = agents, sized by activity_rate, coloured by status
// (active / banned) with a red ring when the agent has tripped
// any phantom-tripwire event.
//
// Edges:
//   thread   teal, thickness ∝ message count (buyer ↔ seller pair)
//   rating   amber (≥4 stars) / red (≤2 stars)
//   block    dashed red
//
// Interactions:
//   drag a node to reposition
//   scroll to zoom · click-drag empty space to pan
//   click a node → router push to /agents/:id
//   hover a node → side panel shows agent + pair stats
//   edge-kind checkboxes filter the network
//
// Clean-room: the algorithm (forceSimulation + zoom + drag) comes
// straight from the d3-force public docs; no MiroFish source
// copied.
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import * as d3 from 'd3'
import { useSnapshot } from '../composables/useData.js'

const { snapshot } = useSnapshot()
const router = useRouter()

const svgRef = ref(null)
const container = ref(null)
const hovered = ref(null)           // agent node data
const selectedEdge = ref(null)      // edge data

// Edge-kind filters.
const showThread = ref(true)
const showRating = ref(true)
const showBlock = ref(true)

let simulation = null
let zoomBehaviour = null
let resizeObs = null

const PALETTE = {
  active:    '#0d7975',  // teal
  banned:    '#a32d2d',  // red
  tripwired: '#b84535',  // coral outline
  thread:    '#0d7975',
  good:      '#3b6d11',
  bad:       '#a32d2d',
  neutral:   '#8a5e0d',
  block:     '#a32d2d',
}

// ---- snapshot → graph model ---------------------------------------------

const tripwireCounts = computed(() => {
  const m = {}
  for (const tw of snapshot.value?.events_summary?.tripwires || []) {
    m[tw.agent_id] = (m[tw.agent_id] || 0) + 1
  }
  return m
})

const graphModel = computed(() => {
  if (!snapshot.value) return { nodes: [], edges: [] }
  const nodes = snapshot.value.graph.nodes.map((n) => ({
    id: n.id,
    name: n.display_name,
    user_name: n.user_name,
    status: n.status,
    home_zip: n.home_zip,
    privacy: n.privacy_awareness,
    tripwires: tripwireCounts.value[n.id] || 0,
  }))
  const edges = snapshot.value.graph.edges.map((e) => ({
    source: e.source,
    target: e.target,
    kind: e.kind,
    weight: e.weight,
    avg_stars: e.avg_stars ?? null,
  }))
  return { nodes, edges }
})

// Per-node activity count = total edge weight incident on this
// node. Node radius is a log scale of activity so super-connected
// nodes don't blow up.
function nodeRadius(n) {
  const r = 5 + Math.sqrt((n.activity || 0)) * 2.5
  return Math.max(5, Math.min(r, 18))
}

// ---- rendering -----------------------------------------------------------

onMounted(() => {
  renderGraph()
  resizeObs = new ResizeObserver(() => renderGraph())
  if (container.value) resizeObs.observe(container.value)
})

onUnmounted(() => {
  if (simulation) simulation.stop()
  if (resizeObs) resizeObs.disconnect()
})

watch(
  () => [graphModel.value, showThread.value, showRating.value, showBlock.value],
  () => renderGraph(),
)

function renderGraph() {
  if (!svgRef.value || !graphModel.value.nodes.length) return
  const { nodes, edges } = graphModel.value

  const filteredEdges = edges.filter((e) => {
    if (e.kind === 'thread' && !showThread.value) return false
    if (e.kind === 'rating' && !showRating.value) return false
    if (e.kind === 'block' && !showBlock.value) return false
    return true
  })

  // Pre-compute activity per node for sizing.
  const activity = {}
  for (const n of nodes) activity[n.id] = 0
  for (const e of filteredEdges) {
    activity[e.source] = (activity[e.source] || 0) + e.weight
    activity[e.target] = (activity[e.target] || 0) + e.weight
  }
  for (const n of nodes) n.activity = activity[n.id] || 0

  const rect = container.value.getBoundingClientRect()
  const width = Math.max(600, rect.width)
  const height = Math.max(420, rect.height)

  const svg = d3.select(svgRef.value)
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
  svg.selectAll('*').remove()

  const root = svg.append('g').attr('class', 'zoom-layer')

  // Link color / dasharray helpers.
  const linkColor = (e) => {
    if (e.kind === 'thread') return PALETTE.thread
    if (e.kind === 'block') return PALETTE.block
    if (e.kind === 'rating') {
      if (e.avg_stars != null && e.avg_stars >= 4) return PALETTE.good
      if (e.avg_stars != null && e.avg_stars <= 2) return PALETTE.bad
      return PALETTE.neutral
    }
    return '#999'
  }
  const linkDash = (e) => (e.kind === 'block' ? '4 3' : null)
  const linkWidth = (e) => {
    if (e.kind === 'thread') return Math.min(6, 1 + Math.sqrt(e.weight))
    if (e.kind === 'rating') return 1.5
    if (e.kind === 'block') return 1.4
    return 1
  }

  // The edges array is rebuilt each call — we must clone it so
  // d3 mutating {source, target} into object refs doesn't corrupt
  // the reactive source.
  const simEdges = filteredEdges.map((e) => ({ ...e }))
  const simNodes = nodes.map((n) => ({ ...n }))

  const linkSel = root.append('g')
    .attr('class', 'links')
    .selectAll('line')
    .data(simEdges)
    .join('line')
    .attr('stroke', linkColor)
    .attr('stroke-opacity', 0.55)
    .attr('stroke-width', linkWidth)
    .attr('stroke-dasharray', linkDash)
    .on('mouseenter', (_, d) => { selectedEdge.value = d })
    .on('mouseleave', () => { selectedEdge.value = null })

  const nodeSel = root.append('g')
    .attr('class', 'nodes')
    .selectAll('g')
    .data(simNodes)
    .join('g')
    .attr('cursor', 'pointer')
    .on('mouseenter', (_, d) => { hovered.value = d })
    .on('mouseleave', () => { hovered.value = null })
    .on('click', (_, d) => { router.push(`/agents/${d.id}`) })

  nodeSel.append('circle')
    .attr('r', nodeRadius)
    .attr('fill', (d) => d.status === 'active' ? PALETTE.active : PALETTE.banned)
    .attr('fill-opacity', 0.85)
    .attr('stroke', (d) => d.tripwires ? PALETTE.tripwired : '#fff')
    .attr('stroke-width', (d) => d.tripwires ? 2.5 : 1.5)

  nodeSel.append('text')
    .text((d) => d.name)
    .attr('x', (d) => nodeRadius(d) + 4)
    .attr('y', 3)
    .attr('font-size', '10.5px')
    .attr('fill', '#2c2c2a')
    .attr('pointer-events', 'none')

  // ---- drag handler ---------------------------------------------------

  const drag = d3.drag()
    .on('start', (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      d.fx = d.x; d.fy = d.y
    })
    .on('drag', (event, d) => {
      d.fx = event.x; d.fy = event.y
    })
    .on('end', (event, d) => {
      if (!event.active) simulation.alphaTarget(0)
      d.fx = null; d.fy = null
    })
  nodeSel.call(drag)

  // ---- zoom ------------------------------------------------------------

  zoomBehaviour = d3.zoom()
    .extent([[0, 0], [width, height]])
    .scaleExtent([0.25, 5])
    .on('zoom', (event) => root.attr('transform', event.transform))
  svg.call(zoomBehaviour)

  // ---- simulation ------------------------------------------------------

  if (simulation) simulation.stop()
  simulation = d3.forceSimulation(simNodes)
    .force('link', d3.forceLink(simEdges)
      .id((d) => d.id)
      .distance((e) => e.kind === 'thread' ? 60 : 100)
      .strength((e) => 0.2 + Math.min(0.6, 0.08 * (e.weight || 1))))
    .force('charge', d3.forceManyBody().strength(-260))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collide', d3.forceCollide().radius((d) => nodeRadius(d) + 6))
    .on('tick', () => {
      linkSel
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y)
      nodeSel.attr('transform', (d) => `translate(${d.x}, ${d.y})`)
    })
}

// Pair stats panel — from hovered node or selected edge.
const detail = computed(() => {
  if (selectedEdge.value) {
    const e = selectedEdge.value
    const a = graphModel.value.nodes.find(
      (n) => n.id === (typeof e.source === 'object' ? e.source.id : e.source),
    )
    const b = graphModel.value.nodes.find(
      (n) => n.id === (typeof e.target === 'object' ? e.target.id : e.target),
    )
    return {
      kind: 'edge',
      text: `${a?.name ?? '?'} ↔ ${b?.name ?? '?'} · ${e.kind} · weight ${e.weight}`
          + (e.avg_stars != null ? ` · avg ${e.avg_stars}★` : ''),
    }
  }
  if (hovered.value) {
    const n = hovered.value
    return {
      kind: 'node',
      text: `${n.name} · @${n.user_name} · ZIP ${n.home_zip}`
          + ` · privacy ${n.privacy.toFixed(2)}`
          + (n.tripwires ? ` · 👻 ${n.tripwires}` : '')
          + ` · ${n.activity} incident edges`,
    }
  }
  return null
})
</script>

<template>
  <h1>Social graph</h1>
  <div class="sub">
    {{ graphModel.nodes.length }} agents · {{ graphModel.edges.length }} edges ·
    drag nodes · scroll to zoom · click a node to open the profile
  </div>

  <div class="graph-toolbar card">
    <label>
      <input v-model="showThread" type="checkbox" />
      <span class="legend-dot" style="background: #0d7975" />
      <span>threads (weight = messages)</span>
    </label>
    <label>
      <input v-model="showRating" type="checkbox" />
      <span class="legend-dot" style="background: #8a5e0d" />
      <span>ratings (colour = avg star; green ≥4, red ≤2)</span>
    </label>
    <label>
      <input v-model="showBlock" type="checkbox" />
      <span class="legend-dot dashed" style="background: #a32d2d" />
      <span>blocks (dashed)</span>
    </label>
    <span class="spacer" />
    <span class="legend-note">
      <span class="node-swatch" style="background: #0d7975" />active ·
      <span class="node-swatch" style="background: #a32d2d" />banned ·
      <span class="node-swatch ring" />👻 tripwire ring
    </span>
  </div>

  <div ref="container" class="graph-wrap">
    <svg ref="svgRef" class="graph-svg" />
  </div>

  <div class="detail-panel card" :class="{ visible: !!detail }">
    <span v-if="detail">{{ detail.text }}</span>
    <span v-else class="empty">Hover a node or edge for details · click a node to open its profile</span>
  </div>
</template>

<style scoped>
.graph-toolbar {
  display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
  font-size: 12.5px; color: var(--ink-dim); margin-bottom: 12px;
}
.graph-toolbar label {
  display: inline-flex; align-items: center; gap: 6px; cursor: default;
}
.graph-toolbar .spacer { flex: 1; }
.legend-dot {
  width: 16px; height: 3px; border-radius: 1px;
  display: inline-block;
}
.legend-dot.dashed {
  background-image: linear-gradient(to right, #a32d2d 50%, transparent 50%);
  background-size: 5px 100%;
}
.node-swatch {
  display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  vertical-align: middle; margin: 0 3px;
}
.node-swatch.ring {
  background: #0d7975;
  box-shadow: 0 0 0 2px var(--coral) inset;
}
.legend-note { font-size: 11.5px; color: var(--ink-mute); }

.graph-wrap {
  background: white; border: 0.5px solid var(--border);
  border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  min-height: 560px; height: calc(100vh - 320px);
  overflow: hidden;
}
.graph-svg {
  width: 100%; height: 100%; display: block;
  background: linear-gradient(180deg, #fbfaf6 0%, #ffffff 100%);
  cursor: grab;
}
.graph-svg:active { cursor: grabbing; }

.detail-panel {
  margin-top: 10px; padding: 10px 14px; font-size: 12.5px;
  min-height: 42px; display: flex; align-items: center;
}
.detail-panel .empty { color: var(--ink-mute); font-style: italic; }
</style>
