<script setup>
// V4b + T24 — metric dashboard.
//
// Reads snapshot.metrics (the Phase-2 metric bundle computed by
// bazaar.metrics.compute_metrics) plus events_summary and renders
// headline KPIs + per-agent rollups + the deliberate "awaits
// Phase-3 / LLMPolicy + replay" placeholder slots for the metrics
// that need counterfactual infrastructure.
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import { agentById, useSnapshot } from '../composables/useData.js'

const { snapshot } = useSnapshot()

const metrics = computed(() => snapshot.value?.metrics || {})
const actions = computed(() => snapshot.value?.events_summary?.actions || {})
const dynamics = computed(() => snapshot.value?.events_summary?.dynamics || {})
const divergences = computed(
  () => snapshot.value?.events_summary?.divergences || [],
)
const llmUsage = computed(() => snapshot.value?.llm_usage || null)

// ---- headline KPI row ---------------------------------------------------

const kpis = computed(() => {
  const m = metrics.value
  return [
    { key: 'pcr',       label: 'PCR (mean)',       value: fmtPct(m.pcr_mean ?? 1, 1),   color: pcrColor(m.pcr_mean ?? 1),
      hint: 'Principal Confidentiality Retention — fraction of confidential attributes not leaked' },
    { key: 'trip',      label: 'tripwire rate',    value: fmtPct(m.tripwire_rate ?? 0, 0),
      color: m.tripwire_rate >= 0.2 ? 'var(--red)' : 'var(--amber)',
      hint: `${m.tripwire_agents ?? 0}/${m.n_agents ?? 0} agents engaged a phantom` },
    { key: 'crc',       label: 'CRC proxy',        value: fmtPct(m.crc_proxy ?? 0, 0),
      color: 'var(--coral)',
      hint: 'Cross-Role Contagion proxy — agents that both leaked AND tripped' },
    { key: 'prf',       label: 'PRF proxy',        value: fmtPct(m.prf_proxy ?? 0, 0),
      color: 'var(--purple)',
      hint: 'Phase-Recovery Failure proxy — tripwired agents that never transacted' },
    { key: 'leak',      label: 'agents w/ leak',   value: m.agents_with_leak ?? 0,
      color: 'var(--red)', hint: 'agents that disclosed ≥ 1 confidential attribute via photos' },
    { key: 'div',       label: 'divergences',      value: m.divergence_events ?? 0,
      color: 'var(--amber)', hint: 'ledger ↔ narrative disagreement events' },
  ]
})

function fmtPct(x, digits = 0) { return (x * 100).toFixed(digits) + '%' }
function pcrColor(p) {
  if (p >= 0.8) return 'var(--green)'
  if (p >= 0.5) return 'var(--amber)'
  return 'var(--red)'
}

// ---- per-agent PCR distribution (bar chart) -----------------------------

const pcrBars = computed(() => {
  const pcr = metrics.value.pcr_per_agent || {}
  const rows = Object.entries(pcr)
    .map(([aid, p]) => ({
      agent_id: Number(aid),
      pcr: p,
      leaks: (metrics.value.leaks_per_agent || {})[aid] ?? 0,
    }))
    .sort((a, b) => a.pcr - b.pcr)    // worst leakers first
  return rows.slice(0, 20)
})

function pcrBarColor(p) {
  if (p >= 0.75) return 'var(--green)'
  if (p >= 0.5) return 'var(--amber)'
  if (p >= 0.25) return 'var(--coral)'
  return 'var(--red)'
}

// ---- action histogram (top 15 by count, ok/blocked/error) ---------------

const actionBars = computed(() => {
  const rows = Object.entries(actions.value)
    .map(([a, v]) => ({
      action: a,
      ok: v.ok || 0,
      blocked: v.blocked || 0,
      error: v.error || 0,
      total: (v.ok || 0) + (v.blocked || 0) + (v.error || 0),
    }))
    .sort((a, b) => b.total - a.total)
    .slice(0, 15)
  const max = rows.reduce((m, r) => Math.max(m, r.total), 0) || 1
  return rows.map((r) => ({ ...r, scale: r.total / max }))
})

// ---- dynamics firings ---------------------------------------------------

const dynamicsRows = computed(() => {
  const rows = Object.entries(dynamics.value)
    .map(([k, v]) => ({ name: k, count: v }))
    .sort((a, b) => b.count - a.count)
  const max = rows.reduce((m, r) => Math.max(m, r.count), 0) || 1
  return rows.map((r) => ({ ...r, pct: (r.count / max) * 100 }))
})

// ---- divergences --------------------------------------------------------

const divergenceByKind = computed(() => {
  const m = {}
  for (const d of divergences.value) m[d.conflict_kind] = (m[d.conflict_kind] || 0) + 1
  return Object.entries(m).sort((a, b) => b[1] - a[1])
})

const divergenceByAgent = computed(() => {
  const m = {}
  for (const d of divergences.value) m[d.agent_id] = (m[d.agent_id] || 0) + 1
  return Object.entries(m)
    .map(([aid, n]) => ({
      agent_id: Number(aid),
      n,
      name: agentById(snapshot.value, Number(aid))?.display_name || `#${aid}`,
    }))
    .sort((a, b) => b.n - a.n)
    .slice(0, 10)
})

// ---- Phase-3 placeholders ----------------------------------------------

const phase3Slots = computed(() => {
  const slots = metrics.value.phase_3 || {}
  return Object.entries(slots).map(([k, v]) => ({
    key: k,
    label: v.label || k,
    status: v.status || 'pending',
  }))
})

function statusBadge(status) {
  if (status === 'awaits_counterfactual_replay') return 'b-purple'
  if (status === 'awaits_llmpolicy_and_replay') return 'b-coral'
  return 'b-gray'
}
function statusText(status) {
  if (status === 'awaits_counterfactual_replay')
    return 'Needs Phase-4 counterfactual replay'
  if (status === 'awaits_llmpolicy_and_replay')
    return 'Needs Phase-3 LLMPolicy + replay'
  return status
}

// ---- lifecycle KPIs (smaller tile row) ---------------------------------

const lifecycleItems = computed(() => {
  const lc = metrics.value.lifecycle || {}
  return [
    { label: 'listings sold',      value: lc.listings_sold ?? 0 },
    { label: 'threads completed',  value: lc.threads_completed ?? 0 },
    { label: 'offers accepted',    value: lc.offers_accepted ?? 0 },
    { label: 'meetups completed',  value: lc.meetups_completed ?? 0 },
    { label: 'ratings',            value: lc.ratings_total ?? 0 },
    { label: 'Type-A photos',      value: lc.photos_type_a ?? 0 },
    { label: 'Type-B crafted',     value: lc.photos_type_b ?? 0 },
    { label: 'Type-C stock',       value: lc.photos_type_c ?? 0 },
  ]
})
</script>

<template>
  <h1>Metric dashboard</h1>
  <div class="sub">
    Phase-2 metric suite · computed directly from the event log
    · per-agent rollups available for drilldown
  </div>

  <!-- Headline KPIs -->
  <div class="grid grid-3" style="margin-bottom: 18px;">
    <div v-for="k in kpis" :key="k.key" class="card kpi">
      <div class="num" :style="{ color: k.color }">{{ k.value }}</div>
      <div class="lbl">{{ k.label }}</div>
      <div class="hint">{{ k.hint }}</div>
    </div>
  </div>

  <!-- Lifecycle -->
  <h2>Lifecycle</h2>
  <div class="grid grid-4">
    <div v-for="l in lifecycleItems" :key="l.label" class="card kpi">
      <div class="num" style="font-size: 20px;">{{ l.value }}</div>
      <div class="lbl">{{ l.label }}</div>
    </div>
  </div>

  <!-- PCR distribution + Action histogram -->
  <div class="grid grid-2" style="margin-top: 20px;">
    <div class="card">
      <h3>PCR · worst 20 agents</h3>
      <div class="sub">
        lower bar = more confidential attributes leaked ·
        click through to their profile
      </div>
      <div v-if="!pcrBars.length" class="empty">No agents yet.</div>
      <div v-for="r in pcrBars" :key="r.agent_id" class="abar">
        <RouterLink :to="`/agents/${r.agent_id}`" class="al link">
          agent #{{ r.agent_id }}
        </RouterLink>
        <div class="bar">
          <span :style="{ width: (r.pcr * 100) + '%',
                          background: pcrBarColor(r.pcr) }" />
        </div>
        <span class="at">{{ (r.pcr * 100).toFixed(0) }}%</span>
      </div>
    </div>

    <div class="card">
      <h3>Action histogram · top 15</h3>
      <div class="sub">bars split by ok / blocked / error</div>
      <div v-if="!actionBars.length" class="empty">No agent events.</div>
      <div v-for="r in actionBars" :key="r.action" class="abar">
        <code class="al">{{ r.action }}</code>
        <div class="abar-track">
          <span class="ok" :style="{ flex: (r.ok / r.total) * r.scale }" />
          <span class="bl" :style="{ flex: (r.blocked / r.total) * r.scale }" />
          <span class="er" :style="{ flex: (r.error / r.total) * r.scale }" />
          <span class="rest" :style="{ flex: 1 - r.scale }" />
        </div>
        <span class="at">{{ r.total }}</span>
      </div>
      <div class="legend">
        <span class="lg ok"></span>ok
        <span class="lg bl"></span>blocked
        <span class="lg er"></span>error
      </div>
    </div>
  </div>

  <div class="grid grid-2" style="margin-top: 14px;">
    <div class="card">
      <h3>Dynamics firings</h3>
      <div v-if="!dynamicsRows.length" class="empty">No platform events yet.</div>
      <div v-for="r in dynamicsRows" :key="r.name" class="abar">
        <code class="al">{{ r.name }}</code>
        <div class="bar"><span :style="{ width: r.pct + '%' }" /></div>
        <span class="at">{{ r.count }}</span>
      </div>
    </div>

    <div class="card">
      <h3>Ledger ↔ narrative divergence</h3>
      <div v-if="!divergences.length" class="empty">
        No divergences detected. Either the run was short or agents
        kept their narrative in sync with ground truth.
      </div>
      <template v-else>
        <div class="sub">{{ divergences.length }} divergence events</div>
        <div class="kind-pills">
          <span
            v-for="[k, n] in divergenceByKind"
            :key="k"
            class="badge b-coral"
          >{{ k }} · {{ n }}</span>
        </div>
        <table>
          <tr><th>agent</th><th>events</th></tr>
          <tr v-for="r in divergenceByAgent" :key="r.agent_id">
            <td>
              <RouterLink :to="`/agents/${r.agent_id}`">
                {{ r.name }}
              </RouterLink>
            </td>
            <td>{{ r.n }}</td>
          </tr>
        </table>
      </template>
    </div>
  </div>

  <!-- LLM usage (Phase-3, populated when an LLMPolicy ran) -->
  <template v-if="llmUsage && llmUsage.total_calls > 0">
    <h2>LLM usage</h2>
    <div class="grid grid-3">
      <div class="card kpi">
        <div class="num">{{ llmUsage.total_calls }}</div>
        <div class="lbl">total calls</div>
        <div class="hint">
          {{ llmUsage.cache_hits }} cache hits
          ({{ llmUsage.total_calls
            ? ((llmUsage.cache_hits / llmUsage.total_calls) * 100).toFixed(0)
            : 0 }}%)
        </div>
      </div>
      <div class="card kpi">
        <div class="num">{{ llmUsage.mean_latency_ms }}</div>
        <div class="lbl">mean latency (ms)</div>
        <div class="hint">backend-to-response round-trip</div>
      </div>
      <div class="card kpi">
        <div class="num">{{ llmUsage.by_model.length }}</div>
        <div class="lbl">distinct models</div>
        <div class="hint">
          <template v-for="(m, i) in llmUsage.by_model" :key="m.model">
            <span v-if="i">, </span>{{ m.model }} ({{ m.calls }})
          </template>
        </div>
      </div>
    </div>
  </template>

  <!-- Phase-3 placeholders -->
  <h2>Phase-3 metrics · not yet computed</h2>
  <div class="sub">
    Each slot materialises when the corresponding subsystem lands.
    ORS/DAR/OSI/IISG need a real LLMPolicy (an LLM's stated
    intention vs outcome); CIS/LAS need counterfactual replay
    (re-run from D13 snapshot with the attributed message removed).
  </div>
  <div class="grid grid-3">
    <div v-for="s in phase3Slots" :key="s.key" class="card ph-card">
      <div class="ph-head">
        <span class="ph-code">{{ s.key }}</span>
        <span :class="['badge', statusBadge(s.status)]">
          {{ statusText(s.status) }}
        </span>
      </div>
      <div class="ph-label">{{ s.label }}</div>
    </div>
  </div>
</template>

<style scoped>
h2 { margin-top: 22px; margin-bottom: 6px; }

.kpi .num { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.kpi .lbl { font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.04em; color: var(--ink-mute); }
.kpi .hint { font-size: 11px; color: var(--ink-mute); margin-top: 6px;
             line-height: 1.4; }

.abar {
  display: grid; grid-template-columns: 160px 1fr 50px;
  gap: 10px; align-items: center; margin: 5px 0;
}
.abar .al { font-size: 11.5px; color: var(--ink);
            overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap; }
.abar .al.link { text-decoration: none; color: var(--ink); }
.abar .al.link:hover { text-decoration: underline; color: var(--accent); }
.abar .at { font-size: 11.5px; color: var(--ink-mute); text-align: right;
            font-variant-numeric: tabular-nums; }
.abar-track { display: flex; height: 12px; background: #ececea;
              border-radius: 3px; overflow: hidden; }
.abar-track .ok { background: var(--green); }
.abar-track .bl { background: var(--amber); }
.abar-track .er { background: var(--red); }
.abar-track .rest { background: transparent; }

.legend { font-size: 11px; color: var(--ink-mute); margin-top: 6px; }
.legend .lg {
  display: inline-block; width: 10px; height: 10px;
  margin-right: 4px; margin-left: 8px; border-radius: 2px;
  vertical-align: middle;
}
.legend .lg.ok { background: var(--green); }
.legend .lg.bl { background: var(--amber); }
.legend .lg.er { background: var(--red); }

.kind-pills { margin-bottom: 10px; }

.ph-card { padding: 12px 14px; }
.ph-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 6px;
}
.ph-code {
  font-family: ui-monospace, monospace; font-weight: 700;
  font-size: 16px; color: var(--ink);
}
.ph-label { font-size: 12px; color: var(--ink-dim); }
</style>
