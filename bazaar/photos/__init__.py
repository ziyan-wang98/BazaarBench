"""Three-variant symbolic photo system.

Summary:

    Photo = <S, B, M, A>
        S = subject_attrs     (what the photo shows of the item)
        B = background_leaks  (incidental environment info — house
                               number, mail, roommate's face, etc.)
        M = metadata_leaks    (EXIF: GPS, device, capture time)
        A ⊆ {S, B, M}          (fields the *sender* is aware of)

Three constructors correspond to three research questions:

* Type A — honest photo: sender specifies focus only; env populates
  S/B/M from persona + item. A = {S}.   Isolates **involuntary leakage**
  (H1 evidence).
* Type B — crafted photo: sender authors all three fields. A = {S,B,M}.
  A silent ``ground_truth`` tag lets the researcher detect fabrication.
  Isolates **emergent deception** (H3 evidence).
* Type C — stock photo: clean placeholder; B = M = {}; ``is_stock=True``
  is visible to :func:`INSPECT_PHOTO`. Tests whether absence-of-
  -disclosure is itself a signal.

T8 scope: the ``Photo`` dataclass and three factories. Type A's
procedural S/B/M synthesis from the persona is T9; the dispatcher
wiring is T10. Here the Type A factory accepts explicit S/B/M so the
interface is stable before the synthesiser lands.
"""
from __future__ import annotations

from bazaar.photos.factory import (
    make_type_a,
    make_type_b,
    make_type_c,
)
from bazaar.photos.photo import Photo, PhotoType
from bazaar.photos.synthesis import (
    make_type_a_from_persona,
    synthesize_background,
    synthesize_metadata,
    synthesize_subject,
    synthesize_type_a,
)

__all__ = [
    "Photo",
    "PhotoType",
    "make_type_a",
    "make_type_a_from_persona",
    "make_type_b",
    "make_type_c",
    "synthesize_background",
    "synthesize_metadata",
    "synthesize_subject",
    "synthesize_type_a",
]
