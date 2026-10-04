"""Paper-facing BazaarBench measurement-contract implementation.

The package is intentionally separate from the legacy aggregate exporters.  It
implements the opportunity -> consideration -> attempt -> exposure ->
engagement -> realisation -> subsequent-outcome contract used by the revised
experiments.
"""

from .contract import (
    CHANNELS,
    TREATED_AGENT_IDS,
    Channel,
    EvidenceBasis,
    LinkConfidence,
    Perspective,
    Severity,
)

__all__ = [
    "CHANNELS",
    "TREATED_AGENT_IDS",
    "Channel",
    "EvidenceBasis",
    "LinkConfidence",
    "Perspective",
    "Severity",
]
