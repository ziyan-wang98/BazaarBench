<script setup>
// V2 — message thread viewer.
//
// Each thread is a card: header (listing, status badge, buyer ↔
// seller with avatars), message bubbles with sender avatar + tick
// annotation + inline PhotoCard when a message carries a photo,
// then a row of offer pills at the bottom.
//
// Left sidebar filters by status and participant; right pane is
// the scrollable list.
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import Avatar from '../components/Avatar.vue'
import PhotoCard from '../components/PhotoCard.vue'
import { agentById, photoById, useSnapshot } from '../composables/useData.js'

const { snapshot } = useSnapshot()

// ---- filter state --------------------------------------------------------

const STATUS_OPTIONS = ['open', 'committed', 'completed', 'cancelled', 'ghosted']
const activeStatuses = ref(new Set(STATUS_OPTIONS))
const participantFilter = ref('')   // matches user_name or display_name

function toggleStatus(s) {
  if (activeStatuses.value.has(s)) {
    activeStatuses.value.delete(s)
  } else {
    activeStatuses.value.add(s)
  }
  // Force reactivity — Set mutation doesn't trigger watchers.
  activeStatuses.value = new Set(activeStatuses.value)
}

// ---- derivations ---------------------------------------------------------

function nameOf(aid) {
  if (aid == null) return 'Phantom'
  const a = agentById(snapshot.value, aid)
  return a ? a.display_name : `#${aid}`
}

function listingFor(t) {
  if (!snapshot.value) return null
  return snapshot.value.listings.find(
    (l) => l.listing_id === t.listing_id,
  )
}

const filteredThreads = computed(() => {
  if (!snapshot.value) return []
  const needle = participantFilter.value.trim().toLowerCase()
  return snapshot.value.threads.filter((t) => {
    if (!activeStatuses.value.has(t.status)) return false
    if (!needle) return true
    const buyer = agentById(snapshot.value, t.buyer_agent_id)
    const seller = t.seller_agent_id != null
      ? agentById(snapshot.value, t.seller_agent_id) : null
    return (buyer && (buyer.user_name.toLowerCase().includes(needle)
                   || buyer.display_name.toLowerCase().includes(needle)))
        || (seller && (seller.user_name.toLowerCase().includes(needle)
                   || seller.display_name.toLowerCase().includes(needle)))
  })
})

const STATUS_CLASS = {
  open:      'b-green',
  committed: 'b-blue',
  completed: 'b-teal',
  cancelled: 'b-gray',
  ghosted:   'b-red',
}

function photoFor(messagePhotoId) {
  if (messagePhotoId == null) return null
  return photoById(snapshot.value, messagePhotoId)
}

function fmtPrice(cents) {
  return `$${(cents / 100).toFixed(2)}`
}
</script>

<template>
  <h1>Message threads</h1>
  <div class="sub">
    {{ filteredThreads.length }} of {{ snapshot?.threads.length || 0 }} threads shown ·
    hover a status badge · click a name to open their profile
  </div>

  <div class="thread-layout">
    <aside class="side">
      <div class="card">
        <h3>Status</h3>
        <div class="status-list">
          <label
            v-for="s in STATUS_OPTIONS"
            :key="s"
            class="status-row"
          >
            <input
              type="checkbox"
              :checked="activeStatuses.has(s)"
              @change="toggleStatus(s)"
            />
            <span :class="['badge', STATUS_CLASS[s]]">{{ s }}</span>
          </label>
        </div>
      </div>
      <div class="card">
        <h3>Participant</h3>
        <input
          v-model="participantFilter"
          type="text"
          placeholder="name or @handle"
          class="participant-input"
        />
      </div>
    </aside>

    <section>
      <div v-if="!filteredThreads.length" class="card">
        <p class="empty">No threads match the current filters.</p>
      </div>
      <article
        v-for="t in filteredThreads"
        :key="t.thread_id"
        class="thread"
      >
        <header class="thead">
          <div>
            <b>Thread #{{ t.thread_id }}</b>
            <span v-if="listingFor(t)">
              — {{ listingFor(t).title }}
              @ {{ fmtPrice(listingFor(t).price_cents) }}
            </span>
            <span
              v-if="listingFor(t)?.is_phantom"
              style="color: var(--red); margin-left: 4px;"
            >· 👻 phantom</span>
            <span :class="['badge', STATUS_CLASS[t.status]]">{{ t.status }}</span>
          </div>
          <div class="tmeta">
            <RouterLink :to="`/agents/${t.buyer_agent_id}`" class="participant">
              <Avatar :id="t.buyer_agent_id" :name="nameOf(t.buyer_agent_id)" :size="20" />
              {{ nameOf(t.buyer_agent_id) }}
            </RouterLink>
            <span style="margin: 0 6px; color: var(--ink-mute);">↔</span>
            <RouterLink
              v-if="t.seller_agent_id != null"
              :to="`/agents/${t.seller_agent_id}`"
              class="participant"
            >
              <Avatar :id="t.seller_agent_id" :name="nameOf(t.seller_agent_id)" :size="20" />
              {{ nameOf(t.seller_agent_id) }}
            </RouterLink>
            <span v-else class="participant" style="color: var(--ink-mute);">
              Phantom
            </span>
          </div>
        </header>

        <div v-if="!t.messages.length" class="empty">
          No messages in this thread.
        </div>
        <div
          v-for="m in t.messages"
          :key="m.message_id"
          :class="['msg', m.sender_agent_id === t.buyer_agent_id ? 'right' : 'left']"
        >
          <div>
            <div class="who">{{ nameOf(m.sender_agent_id) }}</div>
            <div class="bubble">{{ m.body }}</div>
            <PhotoCard
              v-if="photoFor(m.photo_id)"
              :photo="photoFor(m.photo_id)"
            />
            <div class="tick-s">
              tick {{ m.tick }}
              <span v-if="m.read_at_tick != null">· read t={{ m.read_at_tick }}</span>
            </div>
          </div>
        </div>

        <div v-if="t.offers.length" class="offers">
          <b>Offers:</b>
          <span
            v-for="o in t.offers"
            :key="o.offer_id"
            class="offer"
          >
            R{{ o.round }} {{ nameOf(o.proposer_id) }}
            {{ fmtPrice(o.price_cents) }}
            ({{ o.status }}, t{{ o.tick }})
          </span>
        </div>
      </article>
    </section>
  </div>
</template>

<style scoped>
.thread-layout {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 16px;
  align-items: start;
}
.side { display: flex; flex-direction: column; gap: 12px;
        position: sticky; top: 66px; }
.status-list { display: flex; flex-direction: column; gap: 4px; }
.status-row {
  display: flex; align-items: center; gap: 8px; font-size: 12.5px;
}
.participant-input {
  width: 100%; padding: 5px 8px; border-radius: 6px;
  border: 0.5px solid var(--border); font-size: 12.5px;
  font-family: inherit; color: var(--ink);
}

.thread {
  background: white; border: 0.5px solid var(--border);
  border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.thead {
  display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 8px;
  border-bottom: 0.5px solid var(--border);
  padding-bottom: 8px; margin-bottom: 10px;
}
.thead b { font-weight: 500; }
.tmeta { font-size: 12px; color: var(--ink-dim);
         display: flex; align-items: center; }
.participant {
  display: inline-flex; align-items: center; gap: 6px;
  color: var(--ink); padding: 2px 4px; border-radius: 4px;
}
.participant:hover { background: #fbfaf6; text-decoration: none; }

.msg { display: flex; margin: 8px 0; }
.msg.right { justify-content: flex-end; }
.msg > div { max-width: 72%; }
.bubble {
  padding: 8px 12px; border-radius: 12px; font-size: 13px;
  display: inline-block;
}
.msg.left  .bubble {
  background: var(--purple-b); color: var(--purple);
  border-top-left-radius: 4px;
}
.msg.right .bubble {
  background: var(--coral-b); color: var(--coral);
  border-top-right-radius: 4px;
}
.msg.right .who, .msg.right .tick-s { text-align: right; }
.who { font-size: 10.5px; color: var(--ink-mute); margin-bottom: 2px; }
.tick-s { font-size: 10.5px; color: var(--ink-mute); margin-top: 3px; }

.offers { margin-top: 8px; }
.offer {
  display: inline-block; margin-right: 6px; padding: 2px 8px;
  border-radius: 6px; background: var(--amber-b); color: var(--amber);
  font-size: 11px;
}

@media (max-width: 820px) {
  .thread-layout { grid-template-columns: 1fr; }
  .side { position: static; }
}
</style>
