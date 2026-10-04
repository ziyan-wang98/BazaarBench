<script setup>
// V2 (part 2) — agent directory.
// Card grid; click a card to open the per-agent profile.
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import Avatar from '../components/Avatar.vue'
import { useSnapshot } from '../composables/useData.js'

const { snapshot } = useSnapshot()

const search = ref('')
const showBanned = ref(true)

// Per-agent counters derived once — cheap; re-derive on snapshot change.
const counters = computed(() => {
  if (!snapshot.value) return {}
  const out = {}
  for (const l of snapshot.value.listings) {
    if (l.owner_agent_id == null) continue
    const c = out[l.owner_agent_id] ||= {
      listings: 0, threads: 0, photos: 0, narratives: 0, tripwires: 0,
    }
    c.listings += 1
  }
  for (const t of snapshot.value.threads) {
    for (const aid of [t.buyer_agent_id, t.seller_agent_id]) {
      if (aid == null) continue
      const c = out[aid] ||= {
        listings: 0, threads: 0, photos: 0, narratives: 0, tripwires: 0,
      }
      c.threads += 1
    }
  }
  for (const p of snapshot.value.photos) {
    const c = out[p.sender_agent_id] ||= {
      listings: 0, threads: 0, photos: 0, narratives: 0, tripwires: 0,
    }
    c.photos += 1
  }
  for (const n of snapshot.value.narratives) {
    if (n.decayed) continue
    const c = out[n.agent_id] ||= {
      listings: 0, threads: 0, photos: 0, narratives: 0, tripwires: 0,
    }
    c.narratives += 1
  }
  for (const tw of snapshot.value.events_summary?.tripwires || []) {
    const c = out[tw.agent_id] ||= {
      listings: 0, threads: 0, photos: 0, narratives: 0, tripwires: 0,
    }
    c.tripwires += 1
  }
  return out
})

const filtered = computed(() => {
  if (!snapshot.value) return []
  const needle = search.value.trim().toLowerCase()
  return snapshot.value.agents.filter((a) => {
    if (!showBanned.value && a.status !== 'active') return false
    if (!needle) return true
    return (a.user_name.toLowerCase().includes(needle)
         || a.display_name.toLowerCase().includes(needle)
         || a.home_zip.includes(needle))
  })
})

function counter(aid) {
  return counters.value[aid] || {
    listings: 0, threads: 0, photos: 0, narratives: 0, tripwires: 0,
  }
}
</script>

<template>
  <h1>Agents</h1>
  <div class="sub">
    {{ filtered.length }} of {{ snapshot?.agents.length || 0 }} agents ·
    click a card for the full profile · 👻 column = phantom tripwires
  </div>

  <div class="controls card">
    <label class="row">
      <span class="lab">search</span>
      <input
        v-model="search"
        type="text"
        placeholder="name, @handle, or ZIP"
        class="search-input"
      />
    </label>
    <label class="row">
      <input v-model="showBanned" type="checkbox" />
      <span>include banned</span>
    </label>
  </div>

  <div class="grid grid-3">
    <RouterLink
      v-for="a in filtered"
      :key="a.agent_id"
      :to="`/agents/${a.agent_id}`"
      class="agent-card"
    >
      <div class="head">
        <Avatar :id="a.agent_id" :name="a.display_name" :size="36" />
        <div>
          <div class="name">{{ a.display_name }}</div>
          <div class="tag">@{{ a.user_name }} · ZIP {{ a.home_zip }}</div>
        </div>
        <span class="spacer" />
        <span
          :class="['badge', a.status === 'active' ? 'b-green' : 'b-red']"
        >{{ a.status }}</span>
      </div>

      <div class="row meters">
        <div><span class="k">activity</span>
          <div class="bar"><span :style="{ width: (a.activity_rate * 100) + '%' }"></span></div>
        </div>
        <div><span class="k">privacy</span>
          <div class="bar"><span :style="{ width: (a.privacy_awareness * 100) + '%' }"></span></div>
        </div>
      </div>

      <div class="counters">
        <span>{{ counter(a.agent_id).listings }} listings</span>
        <span>·</span>
        <span>{{ counter(a.agent_id).threads }} threads</span>
        <span>·</span>
        <span>{{ counter(a.agent_id).photos }} photos</span>
        <span>·</span>
        <span>{{ counter(a.agent_id).narratives }} narratives</span>
        <span
          v-if="counter(a.agent_id).tripwires"
          class="tripwires"
        >· 👻 {{ counter(a.agent_id).tripwires }}</span>
      </div>
    </RouterLink>
  </div>
</template>

<style scoped>
.controls {
  display: flex; gap: 16px; align-items: center;
  flex-wrap: wrap; margin-bottom: 14px;
}
.row { display: flex; align-items: center; gap: 8px; font-size: 12.5px; }
.row .lab { color: var(--ink-mute); }
.search-input {
  flex: 1; min-width: 220px; padding: 5px 8px; border-radius: 6px;
  border: 0.5px solid var(--border); font-size: 12.5px;
  font-family: inherit; color: var(--ink);
}

.agent-card {
  background: white; border: 0.5px solid var(--border);
  border-radius: 10px; padding: 12px 14px;
  display: flex; flex-direction: column; gap: 10px;
  color: inherit;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.agent-card:hover {
  text-decoration: none; border-color: var(--accent);
  box-shadow: 0 2px 4px rgba(13, 121, 117, 0.1);
}
.head { display: flex; align-items: center; gap: 10px; }
.head .name { font-weight: 500; color: var(--ink); }
.head .tag { font-size: 11.5px; color: var(--ink-mute); }
.head .spacer { flex: 1; }

.meters { gap: 14px; align-items: start; }
.meters > div { flex: 1; }
.meters .k { display: block; color: var(--ink-mute);
             font-size: 10.5px; margin-bottom: 3px; }

.counters {
  font-size: 11.5px; color: var(--ink-mute);
  display: flex; flex-wrap: wrap; gap: 4px;
}
.counters .tripwires { color: var(--red); font-weight: 500; }
</style>
