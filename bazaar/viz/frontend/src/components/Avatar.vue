<script setup>
// Deterministic colored disc from an agent id + display name. Used
// across Threads / Agents / AgentProfile so the same agent has the
// same look wherever they appear.
import { computed } from 'vue'

const props = defineProps({
  id:   { type: [Number, String], required: true },
  name: { type: String, default: '?' },
  size: { type: Number, default: 32 },
})

// Same palette as the Python theme (see bazaar/viz/theme.py).
const PALETTE = [
  '#0d7975', '#b84535', '#4a3a9e', '#8a5e0d',
  '#185fa5', '#3b6d11', '#a32d2d', '#5f5e5a',
]

const color = computed(() => PALETTE[Number(props.id) % PALETTE.length])
const initials = computed(
  () => (props.name || '?').slice(0, 2).toUpperCase(),
)
const fontSize = computed(() => Math.max(10, Math.floor(props.size / 3)))
</script>

<template>
  <span
    class="avatar"
    :style="{
      background: color,
      width: size + 'px',
      height: size + 'px',
      fontSize: fontSize + 'px',
    }"
  >{{ initials }}</span>
</template>
