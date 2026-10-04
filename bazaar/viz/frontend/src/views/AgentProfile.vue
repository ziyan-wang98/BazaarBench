<script setup>
// V2 (part 2) — per-agent profile.
//
// Sections: header · persona card · ledger · narrative memory ·
// photos authored · action log. Mirrors the static HTML version in
// profile.py but driven by the inlined snapshot and with router
// links back out.
import { computed } from 'vue'
import { useRoute, RouterLink } from 'vue-router'
import Avatar from '../components/Avatar.vue'
import PhotoCard from '../components/PhotoCard.vue'
import {
  agentById,
  ledgerForAgent,
  narrativesForAgent,
  photosForAgent,
  threadsForAgent,
  useSnapshot,
} from '../composables/useData.js'

const props = defineProps({ id: { type: String, required: true } })
const route = useRoute()
const { snapshot } = useSnapshot()
const aid = computed(() => Number(props.id ?? route.params.id))

const agent = computed(() => agentById(snapshot.value, aid.value))
const persona = computed(() => agent.value?.persona || {})
const big5 = computed(() => persona.value?.big_five || {})

const ledger = computed(() => ledgerForAgent(snapshot.value, aid.value))
const narratives = computed(() => narrativesForAgent(snapshot.value, aid.value))
const photos = computed(() => photosForAgent(snapshot.value, aid.value))
const threads = computed(() => threadsForAgent(snapshot.value, aid.value))

const actionLog = computed(() => {
  if (!snapshot.value) return []
  // The snapshot doesn't include the raw events list (would bloat
  // JSON); instead we reconstruct a per-agent event approximation
  // from messages, offers, photos, narratives, ledger entries —
  // the things that correspond to ok-status actions this agent
  // produced. Sorted newest first.
  const events = []
  for (const t of snapshot.value.threads) {
    for (const m of t.messages) {
      if (m.sender_agent_id === aid.value) {
        events.push({
          tick: m.tick,
          kind: m.photo_id ? 'send_photo' : 'message',
          summary: `thread #${t.thread_id} · ${m.body}`,
        })
      }
    }
    for (const o of t.offers) {
      if (o.proposer_id === aid.value) {
        events.push({
          tick: o.tick,
          kind: 'make_offer',
          summary: `thread #${t.thread_id} · $${(o.price_cents / 100).toFixed(2)} · ${o.status}`,
        })
      }
    }
  }
  for (const l of snapshot.value.listings) {
    if (l.owner_agent_id === aid.value) {
      events.push({
        tick: l.created_at_tick,
        kind: 'create_listing',
        summary: `${l.title} @ $${(l.price_cents / 100).toFixed(2)}`,
      })
    }
  }
  for (const n of narratives.value) {
    events.push({
      tick: n.created_tick,
      kind: 'summarize_session',
      summary: n.content,
    })
  }
  return events
    .sort((a, b) => b.tick - a.tick)
    .slice(0, 50)
})

const LEDGER_KIND_CLASS = {
  transaction: 'b-teal',
  rating:      'b-amber',
  block:       'b-red',
  report:      'b-coral',
}

const ACTION_KIND_CLASS = {
  create_listing:    'b-green',
  message:           'b-blue',
  send_photo:        'b-purple',
  make_offer:        'b-amber',
  summarize_session: 'b-coral',
}
</script>

<template>
  <div v-if="!agent" class="card">
    <p class="empty">Agent #{{ id }} not found in this snapshot.</p>
  </div>
  <template v-else>
    <!-- header -->
    <div class="profile-hdr">
      <Avatar :id="agent.agent_id" :name="agent.display_name" :size="56" />
      <div>
        <div style="font-size:18px;font-weight:500;">{{ agent.display_name }}</div>
        <div class="tag">
          @{{ agent.user_name }} · ZIP {{ agent.home_zip }}
          · {{ agent.device }}
        </div>
      </div>
      <span class="spacer" />
      <span
        :class="['badge', agent.status === 'active' ? 'b-green' : 'b-red']"
      >{{ agent.status }}</span>
    </div>

    <!-- persona -->
    <h2>Persona card</h2>
    <div class="grid grid-2">
      <div class="card">
        <h3>Identity + synthetic PII</h3>
        <div class="kv">
          <span class="k">age</span><span>{{ persona.age }}</span>
          <span class="k">gender</span><span>{{ persona.gender }}</span>
          <span class="k">profession</span><span>{{ persona.profession }}</span>
          <span class="k">home street</span><span>{{ persona.home_street }}</span>
          <span class="k">phone</span><span>{{ persona.phone_number }}</span>
          <span class="k">email</span><span>{{ persona.email }}</span>
          <span class="k">Venmo</span><span>{{ persona.venmo_handle }}</span>
          <span class="k">Zelle</span><span>{{ persona.zelle_handle }}</span>
          <span class="k">activity rate</span>
          <span>{{ persona.activity_rate?.toFixed(2) }}</span>
          <span class="k">privacy awareness</span>
          <span>{{ persona.privacy_awareness?.toFixed(2) }}</span>
          <span class="k">trust default</span>
          <span>{{ persona.trust_default?.toFixed(2) }}</span>
          <span class="k">haggle tendency</span>
          <span>{{ persona.haggle_tendency?.toFixed(2) }}</span>
          <span class="k">disposable income</span>
          <span>${{ (persona.disposable_income_cents / 100).toFixed(2) }}</span>
          <span class="k">monthly budget</span>
          <span>${{ (persona.monthly_budget_cents / 100).toFixed(2) }}</span>
        </div>
        <div class="interests">
          <span class="k">interests</span>
          <div>
            <span
              v-for="i in (persona.interests || [])"
              :key="i"
              class="badge b-teal"
            >{{ i }}</span>
            <span v-if="!(persona.interests || []).length" class="empty">(none)</span>
          </div>
        </div>
      </div>
      <div class="card">
        <h3>Big Five personality</h3>
        <div class="big5">
          <div
            v-for="k in ['openness','conscientiousness','extraversion','agreeableness','neuroticism']"
            :key="k"
          >
            <div class="big5-row">
              <span>{{ k }}</span>
              <span>{{ (big5[k] ?? 0.5).toFixed(2) }}</span>
            </div>
            <div class="bar">
              <span :style="{ width: ((big5[k] ?? 0.5) * 100) + '%' }" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ledger -->
    <h2>Structured ledger</h2>
    <div class="sub">
      Platform-maintained · agent cannot edit · auto-injected into every LLM call
    </div>
    <div class="card">
      <div v-if="!ledger.length" class="empty">No ledger entries yet.</div>
      <table v-else>
        <tr><th>tick</th><th>kind</th><th>counterparty</th><th>summary</th></tr>
        <tr v-for="e in ledger" :key="e.ref_table + '-' + e.ref_id + '-' + e.kind">
          <td>t={{ e.tick }}</td>
          <td>
            <span :class="['badge', LEDGER_KIND_CLASS[e.kind] || 'b-gray']">
              {{ e.kind }}
            </span>
          </td>
          <td>
            <RouterLink
              v-if="e.counterparty_id != null"
              :to="`/agents/${e.counterparty_id}`"
            >#{{ e.counterparty_id }}</RouterLink>
            <span v-else>—</span>
          </td>
          <td>{{ e.summary }}</td>
        </tr>
      </table>
    </div>

    <!-- narrative -->
    <h2>Narrative memory</h2>
    <div class="sub">
      Free-text impressions · subject to drift ·
      divergences from the ledger are the H1 inherited-drift signal
    </div>
    <div v-if="!narratives.length" class="card">
      <p class="empty">No narrative memories.</p>
    </div>
    <div v-else class="grid grid-2">
      <div
        v-for="n in narratives"
        :key="n.memory_id"
        class="card"
        style="padding:10px 12px;"
      >
        <div class="tag">
          t={{ n.created_tick }} · {{ n.scope }}
          <RouterLink
            v-if="n.scope_ref_id != null && n.scope === 'counterparty'"
            :to="`/agents/${n.scope_ref_id}`"
          >#{{ n.scope_ref_id }}</RouterLink>
          <span v-else-if="n.scope_ref_id != null">#{{ n.scope_ref_id }}</span>
        </div>
        <div style="margin-top:4px;">{{ n.content }}</div>
      </div>
    </div>

    <!-- photos -->
    <h2>Photos authored</h2>
    <div class="sub">
      Type A honest · Type B crafted · Type C stock ·
      background / EXIF leak fields in red
    </div>
    <div v-if="!photos.length" class="card">
      <p class="empty">No photos sent.</p>
    </div>
    <div v-else class="grid grid-3">
      <PhotoCard
        v-for="p in photos"
        :key="p.photo_id"
        :photo="p"
      />
    </div>

    <!-- threads -->
    <h2>Conversations</h2>
    <div v-if="!threads.length" class="card">
      <p class="empty">No threads.</p>
    </div>
    <div v-else class="card">
      <table>
        <tr><th>thread</th><th>role</th><th>counterparty</th><th>status</th><th>last tick</th></tr>
        <tr v-for="t in threads" :key="t.thread_id">
          <td><RouterLink :to="`/threads#${t.thread_id}`">#{{ t.thread_id }}</RouterLink></td>
          <td>{{ t.buyer_agent_id === aid ? 'buyer' : 'seller' }}</td>
          <td>
            <RouterLink
              v-if="(t.buyer_agent_id === aid ? t.seller_agent_id : t.buyer_agent_id) != null"
              :to="`/agents/${t.buyer_agent_id === aid ? t.seller_agent_id : t.buyer_agent_id}`"
            >#{{ t.buyer_agent_id === aid ? t.seller_agent_id : t.buyer_agent_id }}</RouterLink>
            <span v-else>Phantom</span>
          </td>
          <td>{{ t.status }}</td>
          <td>{{ t.last_msg_tick ?? '—' }}</td>
        </tr>
      </table>
    </div>

    <!-- action log -->
    <h2>Recent actions</h2>
    <div class="sub">Reconstructed from the snapshot (newest 50)</div>
    <div class="card">
      <div v-if="!actionLog.length" class="empty">Nothing to show.</div>
      <table v-else>
        <tr><th>tick</th><th>kind</th><th>what</th></tr>
        <tr v-for="(ev, i) in actionLog" :key="i">
          <td>t={{ ev.tick }}</td>
          <td>
            <span :class="['badge', ACTION_KIND_CLASS[ev.kind] || 'b-gray']">
              {{ ev.kind }}
            </span>
          </td>
          <td>{{ ev.summary }}</td>
        </tr>
      </table>
    </div>
  </template>
</template>

<style scoped>
.profile-hdr {
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 16px;
}
.profile-hdr .spacer { flex: 1; }
.profile-hdr .tag { color: var(--ink-mute); font-size: 12px; }

.kv { display: grid; grid-template-columns: 160px 1fr;
      gap: 4px 14px; font-size: 12.5px; }
.kv .k { color: var(--ink-mute); }

.interests { margin-top: 10px; }
.interests .k { font-size: 11.5px; color: var(--ink-mute); }

.big5 { display: grid; gap: 10px; }
.big5-row {
  display: flex; justify-content: space-between; font-size: 11.5px;
  color: var(--ink-dim); margin-bottom: 3px;
}

h2 { margin-top: 20px; }
</style>
