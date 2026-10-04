<script setup>
import { computed } from 'vue'
import { RouterLink, RouterView } from 'vue-router'
import { useSnapshot } from './composables/useData.js'

const { snapshot, loading, error } = useSnapshot()

const dbName = computed(() => snapshot.value?.meta?.db_path || '')
</script>

<template>
  <nav class="nav">
    <RouterLink to="/" class="brand">
      BazaarBench <span class="v">v0.2.0-alpha</span>
    </RouterLink>
    <RouterLink to="/feed">Feed</RouterLink>
    <RouterLink to="/threads">Threads</RouterLink>
    <RouterLink to="/agents">Agents</RouterLink>
    <RouterLink to="/sandbox">Sandbox</RouterLink>
    <RouterLink to="/timeline">Timeline</RouterLink>
    <RouterLink to="/metrics">Metrics</RouterLink>
    <span class="spacer" />
    <span class="dbname">{{ dbName }}</span>
  </nav>

  <main class="page">
    <div v-if="loading" class="loading">Loading snapshot…</div>
    <div v-else-if="error" class="error">
      Could not load <code>data.json</code> — {{ error.message }}
      <div style="margin-top: 8px; color: var(--ink-dim); font-size: 11.5px;">
        Run <code>bazaar export-data &lt;db&gt; --out &lt;dir&gt;/data.json</code>
        or use <code>bazaar dashboard-vue</code> which wires everything up.
      </div>
    </div>
    <RouterView v-else />
  </main>
</template>
