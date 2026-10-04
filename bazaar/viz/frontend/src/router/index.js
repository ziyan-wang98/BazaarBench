// Router: one route per inspection surface: six core surfaces plus
// two additions (social graph, timeline).
//
// Views are code-split so the initial bundle stays slim when we
// add the heavier d3-force graph in V3.
import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  { path: '/',            name: 'home',     component: () => import('../views/Home.vue') },
  { path: '/feed',        name: 'feed',     component: () => import('../views/Feed.vue') },
  { path: '/threads',     name: 'threads',  component: () => import('../views/Threads.vue') },
  { path: '/agents',      name: 'agents',   component: () => import('../views/Agents.vue') },
  { path: '/agents/:id',  name: 'profile',  component: () => import('../views/AgentProfile.vue'), props: true },
  { path: '/sandbox',     name: 'sandbox',  component: () => import('../views/Sandbox.vue') },
  // Backwards-compat redirects — old URLs still resolve.
  { path: '/map',         redirect: '/sandbox' },
  { path: '/graph',       redirect: '/sandbox' },
  { path: '/metrics',     name: 'metrics',  component: () => import('../views/Metrics.vue') },
  { path: '/timeline',    name: 'timeline', component: () => import('../views/Timeline.vue') },
]

// Hash history so the built SPA works from a file:// URL. Ordinary
// history mode would need a server.
export default createRouter({
  history: createWebHashHistory(),
  routes,
})
