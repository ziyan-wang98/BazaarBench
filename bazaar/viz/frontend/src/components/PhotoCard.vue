<script setup>
// Renders a single symbolic Photo row (Type A/B/C), colour-coded,
// with background + metadata leak fields highlighted as ⚠. Shared
// between the thread viewer (inline under a message) and the agent
// profile (grid of photos the agent authored).
import { computed } from 'vue'

const props = defineProps({
  photo: { type: Object, required: true },
})

const typeLabel = computed(() => ({
  A: 'Type A · honest',
  B: 'Type B · crafted',
  C: 'Type C · stock',
}[props.photo.photo_type] || `Type ${props.photo.photo_type}`))

const hasGroundTruth = computed(() => !!props.photo.ground_truth)

function pairs(obj) {
  return Object.entries(obj || {})
}
</script>

<template>
  <div :class="['photo', 'type-' + photo.photo_type]">
    <div class="head">
      📷 {{ typeLabel }}
      <span v-if="photo.is_stock" class="tag"> · stock</span>
      <span v-if="photo.created_at_tick != null" class="tag">
        · t={{ photo.created_at_tick }}
      </span>
    </div>

    <!-- subject (always visible) -->
    <div
      v-for="[k, v] in pairs(photo.subject_attrs)"
      :key="'s-' + k"
      class="field"
    >
      <span class="k">{{ k }}:</span> {{ v }}
    </div>

    <!-- background leaks (red) -->
    <div
      v-for="[k, v] in pairs(photo.background_leaks)"
      :key="'b-' + k"
      class="field leak"
    >
      ⚠ bg · <span class="k">{{ k }}:</span> {{ v }}
    </div>

    <!-- metadata leaks (red) -->
    <div
      v-for="[k, v] in pairs(photo.metadata_leaks)"
      :key="'m-' + k"
      class="field leak"
    >
      ⚠ exif · <span class="k">{{ k }}:</span> {{ v }}
    </div>

    <!-- Type-B silent ground-truth tag (researcher-only) -->
    <div v-if="hasGroundTruth" class="truth">
      ground_truth (offline only) ·
      <span
        v-for="[k, v] in pairs(photo.ground_truth)"
        :key="'gt-' + k"
      >
        <span class="k">{{ k }}</span>={{ v }}&nbsp;
      </span>
    </div>
  </div>
</template>

<style scoped>
.photo {
  margin-top: 6px;
  padding: 8px 12px;
  border-radius: 8px;
  border: 0.5px solid var(--border);
  font-size: 12px;
  color: var(--ink-dim);
}
.photo.type-A { background: var(--green-b);  border-color: var(--green); }
.photo.type-B { background: var(--coral-b);  border-color: var(--coral); }
.photo.type-C { background: #ececea;         border-color: var(--ink-mute); }
.head { font-weight: 600; color: var(--ink); margin-bottom: 4px; }
.head .tag { color: var(--ink-mute); font-weight: 400; }
.field { margin: 2px 0; }
.k { color: var(--ink-mute); }
.leak { color: var(--red); font-weight: 500; }
.truth {
  margin-top: 6px; padding-top: 4px; font-size: 11px;
  border-top: 0.5px dashed var(--border); color: var(--purple);
}
</style>
