<script setup>
// V4b — tick-by-tick event timeline.
//
// The snapshot doesn't carry the raw events list (that would bloat
// the JSON and most events are redundant with denormalised fields
// anyway). We reconstruct a useful timeline from the data we do
// have: messages, offers, listings, narratives, photos, tripwires,
// divergences. Ordered by tick descending. Filter sidebar supports
// agent picker + kind multi-select.
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { agentById, useSnapshot } from '../composables/useData.js'

const { snapshot } = useSnapshot()

const allKinds = [
  'create_listing', 'message', 'send_photo', 'make_offer',
  'summarize_session', 'phantom_tripwire', 'memory_divergence',
]
const activeKinds = ref(new Set(allKinds))
const selectedAgent = ref(null)
const limit = ref(120)

function toggleKind(k) {
  if (activeKinds.value.has(k)) activeKinds.value.delete(k)
  else activeKinds.value.add(k)
  activeKinds.value = new Set(activeKinds.value)
}

const events = computed(() => {
  if (!snapshot.value) return []
  const out = []
  for (const t of snapshot.value.threads) {
    for (const m of t.messages) {
      out.push({
        tick: m.tick,
        agent_id: m.sender_agent_id,
        kind: m.photo_id != null ? 'send_photo' : 'message',
        summary: `thread #${t.thread_id}: ${m.body}`,
        link: { name: 'threads', hash: `#${t.thread_id}` },
      })
    }
    for (const o of t.offers) {
      out.push({
        tick: o.tick,
        agent_id: o.proposer_id,
        kind: 'make_offer',
        summary: `thread #${t.thread_id} R${o.round} · $${(o.price_cents / 100).toFixed(2)} · ${o.status}`,
        link: { name: 'threads', hash: `#${t.thread_id}` },
      })
    }
  }
  for (const l of snapshot.value.listings) {
    if (l.owner_agent_id == null) continue
    out.push({
      tick: l.created_at_tick,
      agent_id: l.owner_agent_id,
      kind: 'create_listing',
      summary: `${l.title} @ $${(l.price_cents / 100).toFixed(2)} · ${l.category}`,
      link: { name: 'feed' },
    })
  }
  for (const n of snapshot.value.narratives) {
    if (n.decayed) continue
    out.push({
      tick: n.created_tick,
      agent_id: n.agent_id,
      kind: 'summarize_session',
      summary: `[${n.scope}${n.scope_ref_id != null ? '#' + n.scope_ref_id : ''}] ${n.content}`,
    })
  }
  for (const tw of snapshot.value.events_summary?.tripwires || []) {
    out.push({
      tick: tw.trigger_tick ?? 0,
      agent_id: tw.agent_id,
      kind: 'phantom_tripwire',
      summary: `engaged phantom #${tw.listing_id} (${tw.kind})`,
      link: { name: 'map' },
    })
  }
  for (const d of snapshot.value.events_summary?.divergences || []) {
    out.push({
      tick: d.narrative_tick ?? 0,
      agent_id: d.agent_id,
      kind: 'memory_divergence',
      summary: `${d.conflict_kind} · vs agent#${d.counterparty_id}`,
    })
  }
  return out.sort((a, b) => b.tick - a.tick || b.agent_id - a.agent_id)
})

const filtered = computed(() => events.value.filter((e) => {
  if (!activeKinds.value.has(e.kind)) return false
  if (selectedAgent.value != null && e.agent_id !== selectedAgent.value)
    return false
  return true
}))

const shown = computed(() => filtered.value.slice(0, limit.value))
const hiddenCount = computed(
  () => Math.max(0, filtered.value.length - shown.value.length),
)

const allAgents = computed(() =>
  (snapshot.value?.agents || []).map((a) => ({
    id: a.agent_id,
    label: `#${a.agent_id} ${a.display_name}`,
  }))
)

function nameOf(aid) {
  if (aid == null) return 'platform'
  const a = agentById(snapshot.value, aid)
  return a ? a.display_name : `#${aid}`
}

const KIND_CLASS = {
  create_listing:    'b-green',
  message:           'b-blue',
  send_photo:        'b-purple',
  make_offer:        'b-amber',
  summarize_session: 'b-coral',
  phantom_tripwire:  'b-red',
  memory_divergence: 'b-coral',
}
</script>

<template>
  <h1>Timeline</h1>
  <div class="sub">
    {{ filtered.length }} events in the current filter ·
    showing newest {{ shown.length }}{{ hiddenCount ? ' (' + hiddenCount + ' more hidden)' : '' }}
  </div>

  <div class="timeline-layout">
    <aside class="side">
      <div class="card">
        <h3>Kind</h3>
        <label v-for="k in allKinds" :key="k" class="row">
          <input
            type="checkbox"
            :checked="activeKinds.has(k)"
            @change="toggleKind(k)"
          />
          <span :class="['badge', KIND_CLASS[k]]">{{ k }}</span>
        </label>
      </div>

      <div class="card">
        <h3>Agent</h3>
        <select v-model="selectedAgent" class="select">
          <option :value="null">all agents</option>
          <option
            v-for="a in allAgents"
            :key="a.id"
            :value="a.id"
          >{{ a.label }}</option>
        </select>
      </div>

      <div class="card">
        <h3>Window</h3>
        <label class="row">
          <span class="lab">show</span>
          <select v-model.number="limit" class="select">
            <option :value="60">60 events</option>
            <option :value="120">120 events</option>
            <option :value="300">300 events</option>
            <option :value="10000">all</option>
          </select>
        </label>
      </div>
    </aside>

    <section class="card events-card">
      <div v-if="!shown.length" class="empty">
        No events match the current filters.
      </div>
      <ol v-else class="events">
        <li
          v-for="(ev, i) in shown"
          :key="i"
          class="event"
        >
          <span class="tick">t={{ ev.tick }}</span>
          <span :class="['badge', KIND_CLASS[ev.kind]]">{{ ev.kind }}</span>
          <RouterLink
            v-if="ev.agent_id != null"
            :to="`/agents/${ev.agent_id}`"
            class="who"
          >{{ nameOf(ev.agent_id) }}</RouterLink>
          <span v-else class="who platform">platform</span>
          <span class="summary">{{ ev.summary }}</span>
        </li>
      </ol>
    </section>
  </div>
</template>

<style scoped>
.timeline-layout {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 16px;
  align-items: start;
}
.side {
  display: flex; flex-direction: column; gap: 12px;
  position: sticky; top: 66px;
}
.row { display: flex; align-items: center; gap: 8px;
       padding: 4px 0; font-size: 12.5px; }
.row .lab { color: var(--ink-mute); min-width: 60px; }
.select {
  flex: 1; width: 100%; padding: 5px 8px; border-radius: 6px;
  border: 0.5px solid var(--border); font-size: 12.5px;
  background: white; font-family: inherit; color: var(--ink);
}

.events-card { padding: 0; }
.events { list-style: none; margin: 0; padding: 0; }
.event {
  display: grid;
  grid-template-columns: 60px 150px 140px 1fr;
  gap: 10px;
  padding: 8px 14px;
  border-bottom: 0.5px solid var(--border);
  font-size: 12.5px;
  align-items: baseline;
}
.event:hover { background: #fbfaf6; }
.event:last-child { border-bottom: none; }
.event .tick {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 11.5px; color: var(--ink-mute);
}
.event .who { color: var(--accent); }
.event .who.platform { color: var(--ink-mute); font-style: italic; }
.event .summary {
  color: var(--ink-dim); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 820px) {
  .timeline-layout { grid-template-columns: 1fr; }
  .side { position: static; }
  .event { grid-template-columns: 50px 130px 1fr; }
  .event .who { grid-column: 3; }
  .event .summary { grid-column: 1 / -1; white-space: normal; }
}
</style>
