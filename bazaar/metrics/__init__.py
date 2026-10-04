"""Phase-2 metric suite.

Computes the subset of metrics from the paper's §sec:bazaar-metrics
that are tractable from event-log data alone (no LLM, no
counterfactual replay yet):

* **PCR** (Principal Confidentiality Retention) — seller analogue
  of ORS. Per-agent fraction of confidential persona attributes
  that did NOT leak through any outgoing Type-A photo.
* **Tripwire rate** — fraction of the population that triggered a
  phantom-listing tripwire at least once (direct H1 signal).
* **PRF_proxy** (Phase-Recovery Failure proxy) — fraction of
  tripwired agents who never completed any transaction. Agents who
  drift AND don't recover.
* **CRC_proxy** (Cross-Role Contagion proxy) — fraction of agents
  who both leaked ≥ 1 field AND tripped a phantom. Imperfect but a
  monotone lower bound on the full H2 signal.
* **Divergence rate** — memory-divergence events per agent / tick.

Load-bearing Phase-3 metrics (ORS, DAR, OSI, IISG, CIS, LAS) need
LLM-based goal judgment or counterfactual replay; they're surfaced
as explicit ``None`` slots with a ``phase_3`` marker so the Vue
dashboard can label them "computed later" instead of showing a
zero that a reviewer might misread.
"""
from __future__ import annotations

from bazaar.metrics.core import MetricsSummary, compute_metrics

__all__ = ["MetricsSummary", "compute_metrics"]
