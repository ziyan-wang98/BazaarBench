<script setup>
// V1 — marketplace feed view.
//
// Left sidebar: category filter (click a row to toggle), phantom
// toggle, sort dropdown. Right: listing grid, each card linking
// to the seller's profile.
//
// Everything is derived off `snapshot.listings`; no additional
// fetches. State (filters, sort) lives in local refs.
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { agentById, useSnapshot } from '../composables/useData.js'

const { snapshot } = useSnapshot()

// ---- filter + sort state --------------------------------------------------

const activeCategory = ref(null)   // null = "All"
const includePhantom = ref(true)
const sortMode = ref('fresh')      // 'fresh' | 'price-asc' | 'price-desc'
const zipFilter = ref('')

// ---- derivations ----------------------------------------------------------

const allListings = computed(() => {
  if (!snapshot.value) return []
  return snapshot.value.listings.filter((l) => l.status === 'active')
})

const categoryCounts = computed(() => {
  const c = {}
  for (const l of allListings.value) {
    c[l.category] = (c[l.category] || 0) + 1
  }
  return Object.entries(c).sort((a, b) => b[1] - a[1])
})

const filtered = computed(() => {
  let rows = allListings.value
  if (activeCategory.value != null) {
    rows = rows.filter((l) => l.category === activeCategory.value)
  }
  if (!includePhantom.value) {
    rows = rows.filter((l) => !l.is_phantom)
  }
  if (zipFilter.value) {
    const z = zipFilter.value.trim()
    rows = rows.filter((l) => String(l.location_zip).startsWith(z))
  }
  rows = [...rows]
  if (sortMode.value === 'fresh') {
    rows.sort((a, b) => b.created_at_tick - a.created_at_tick
                     || b.listing_id - a.listing_id)
  } else if (sortMode.value === 'price-asc') {
    rows.sort((a, b) => a.price_cents - b.price_cents)
  } else if (sortMode.value === 'price-desc') {
    rows.sort((a, b) => b.price_cents - a.price_cents)
  }
  return rows
})

function sellerName(ownerId) {
  if (ownerId == null) return 'Phantom'
  const a = agentById(snapshot.value, ownerId)
  return a ? a.display_name : `#${ownerId}`
}

function fmtPrice(cents) {
  return `$${(cents / 100).toFixed(2)}`
}

// ---- UI helpers -----------------------------------------------------------

function clearFilters() {
  activeCategory.value = null
  includePhantom.value = true
  sortMode.value = 'fresh'
  zipFilter.value = ''
}

const anyFilter = computed(
  () => activeCategory.value != null
     || !includePhantom.value
     || zipFilter.value !== ''
     || sortMode.value !== 'fresh',
)
</script>

<template>
  <h1>Marketplace feed</h1>
  <div class="sub">
    {{ allListings.length }} active listings ·
    {{ filtered.length }} after filters ·
    click a seller name to open their profile
  </div>

  <div class="feed-layout">
    <!-- sidebar -->
    <aside class="side">
      <div class="card">
        <h3>Categories</h3>
        <div class="cat-list">
          <button
            class="cat"
            :class="{ active: activeCategory == null }"
            @click="activeCategory = null"
          >
            <span>All</span>
            <span class="tag">{{ allListings.length }}</span>
          </button>
          <button
            v-for="[name, count] in categoryCounts"
            :key="name"
            class="cat"
            :class="{ active: activeCategory === name }"
            @click="activeCategory = name"
          >
            <span>{{ name }}</span>
            <span class="tag">{{ count }}</span>
          </button>
        </div>
      </div>

      <div class="card">
        <h3>Filters</h3>
        <label class="row">
          <input v-model="includePhantom" type="checkbox" />
          <span>include 👻 phantoms</span>
        </label>
        <label class="row">
          <span class="lab">ZIP starts with</span>
          <input
            v-model="zipFilter"
            type="text"
            placeholder="e.g. 94"
            class="zip-input"
          />
        </label>
        <label class="row">
          <span class="lab">sort by</span>
          <select v-model="sortMode" class="select">
            <option value="fresh">newest tick</option>
            <option value="price-asc">price ↑</option>
            <option value="price-desc">price ↓</option>
          </select>
        </label>
        <button
          class="btn-clear"
          :disabled="!anyFilter"
          @click="clearFilters"
        >
          clear filters
        </button>
      </div>
    </aside>

    <!-- grid -->
    <section>
      <div v-if="!filtered.length" class="card">
        <p class="empty">No listings match the current filters.</p>
      </div>
      <div v-else class="grid grid-3">
        <article
          v-for="l in filtered"
          :key="l.listing_id"
          class="listing"
          :class="{ phantom: l.is_phantom }"
        >
          <div class="price">{{ fmtPrice(l.price_cents) }}</div>
          <div class="title">
            {{ l.title }}
            <span v-if="l.is_phantom" class="phantom-tag">👻</span>
          </div>
          <div class="meta">
            <span class="badge b-teal">{{ l.category }}</span>
            <span class="badge b-gray">{{ l.condition }}</span>
            <span class="badge b-gray">ZIP {{ l.location_zip }}</span>
          </div>
          <div class="meta">
            seller:
            <RouterLink
              v-if="l.owner_agent_id != null"
              :to="`/agents/${l.owner_agent_id}`"
            >
              {{ sellerName(l.owner_agent_id) }}
            </RouterLink>
            <span v-else class="empty" style="font-style:normal;">
              {{ sellerName(l.owner_agent_id) }}
            </span>
            · t={{ l.created_at_tick }}
          </div>
          <div class="counters">
            👁 {{ l.view_count }} · 💬 {{ l.inquiry_count }}
          </div>
        </article>
      </div>
    </section>
  </div>
</template>

<style scoped>
.feed-layout {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 16px;
  align-items: start;
}
.side { display: flex; flex-direction: column; gap: 12px; position: sticky; top: 66px; }
.cat-list { display: flex; flex-direction: column; gap: 2px; }
.cat {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 8px; border: none; border-radius: 6px; background: transparent;
  text-align: left; font-size: 12.5px; color: var(--ink); cursor: pointer;
  font-family: inherit;
}
.cat:hover { background: #fbfaf6; }
.cat.active {
  background: var(--accent-b); color: var(--accent); font-weight: 500;
}
.cat .tag {
  font-size: 11px; color: var(--ink-mute);
  background: #ececea; border-radius: 10px; padding: 1px 7px;
}
.cat.active .tag { background: white; color: var(--accent); }

.row {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 0; font-size: 12.5px; color: var(--ink-dim);
}
.row .lab { flex-shrink: 0; min-width: 90px; }
.zip-input, .select {
  flex: 1; padding: 4px 6px; border-radius: 6px;
  border: 0.5px solid var(--border); font-size: 12.5px;
  background: white; font-family: inherit; color: var(--ink);
}
.btn-clear {
  margin-top: 6px; width: 100%; padding: 6px 8px; border-radius: 6px;
  border: 0.5px solid var(--border); background: white;
  color: var(--ink-dim); font-size: 12px; cursor: pointer;
  font-family: inherit;
}
.btn-clear:not(:disabled):hover { background: #fbfaf6; color: var(--ink); }
.btn-clear:disabled { opacity: 0.5; cursor: default; }

.listing {
  background: white; border: 0.5px solid var(--border);
  border-radius: 10px; padding: 12px 14px;
  display: flex; flex-direction: column; gap: 4px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.listing.phantom { border-color: var(--red-b); }
.listing .price { font-weight: 600; color: var(--accent); font-size: 14.5px; }
.listing .title { font-weight: 500; color: var(--ink); font-size: 13.5px; }
.listing .title .phantom-tag {
  color: var(--red); margin-left: 4px; font-size: 12px;
}
.listing .meta {
  color: var(--ink-mute); font-size: 11.5px;
  display: flex; flex-wrap: wrap; gap: 4px; align-items: center;
}
.listing .counters {
  color: var(--ink-mute); font-size: 11px;
  border-top: 0.5px dashed var(--border); padding-top: 4px; margin-top: 4px;
}

@media (max-width: 820px) {
  .feed-layout { grid-template-columns: 1fr; }
  .side { position: static; }
}
</style>
