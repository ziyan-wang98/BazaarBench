"""Truthful handoff checks: flag registry and shared read-only helpers.

The reported BazaarBench runs used the legacy handoff contract:
inspection returns a number copied onto the listing at
``create_listing`` time, nothing stops two threads from committing to
the same listing, completion does not bind a seller unit, and shipments
skip inspection. Four ``meta`` flags switch on a truthful contract:

* ``inspection_truth_mode``: ``listing`` (legacy) or ``unit``.
* ``commitment_lock_mode``: ``off`` (legacy) or ``listing``.
* ``completion_integrity_mode``: ``off`` (legacy) or ``unit``.
* ``shipment_inspection_mode``: ``off`` (legacy) or ``on_arrival``.

The first allowed value of every flag is the legacy default, and every
reader falls back to it when the ``meta`` row is missing (databases
written before these flags existed), so the reported runs stay
reproducible. The ``handoff_checks`` preset bundles the four flags:
``legacy`` keeps the defaults, ``truthful`` turns all four on.

This module imports only the standard library so ``bazaar.actions``,
``bazaar.memory``, ``bazaar.agents`` and ``bazaar.core.env`` can all use
it without creating an import cycle.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata

INSPECTION_TRUTH_MODE = "inspection_truth_mode"
COMMITMENT_LOCK_MODE = "commitment_lock_mode"
COMPLETION_INTEGRITY_MODE = "completion_integrity_mode"
SHIPMENT_INSPECTION_MODE = "shipment_inspection_mode"

# Allowed values per flag. The first value is the legacy default.
HANDOFF_CHECK_MODES: dict[str, tuple[str, ...]] = {
    INSPECTION_TRUTH_MODE: ("listing", "unit"),
    COMMITMENT_LOCK_MODE: ("off", "listing"),
    COMPLETION_INTEGRITY_MODE: ("off", "unit"),
    SHIPMENT_INSPECTION_MODE: ("off", "on_arrival"),
}
LEGACY_HANDOFF_CHECKS: dict[str, str] = {
    key: modes[0] for key, modes in HANDOFF_CHECK_MODES.items()
}
TRUTHFUL_HANDOFF_CHECKS: dict[str, str] = {
    INSPECTION_TRUTH_MODE: "unit",
    COMMITMENT_LOCK_MODE: "listing",
    COMPLETION_INTEGRITY_MODE: "unit",
    SHIPMENT_INSPECTION_MODE: "on_arrival",
}
HANDOFF_CHECK_PRESETS: dict[str, dict[str, str]] = {
    "legacy": LEGACY_HANDOFF_CHECKS,
    "truthful": TRUTHFUL_HANDOFF_CHECKS,
}

# Platform event ``BazaarEnv`` logs when it writes handoff-check values
# that differ from those already in ``meta`` (a new database counts as
# legacy): payload ``{"old": {flag: stored value or None}, "new": {flag:
# value}, "tick": tick, "resume": bool}``. Legacy runs never log it.
HANDOFF_CHECKS_SET_ACTION = "platform_handoff_checks_set"


class HandoffCheckResumeWarning(UserWarning):
    """A resumed ``BazaarEnv`` is about to switch off (or change) a
    handoff check the database ran with, usually because the flags were
    not passed again on resume."""


# v2 quality bands: (band, lowest pct, highest pct). ``bazaar.actions.
# handlers._QUALITY_BAND_RANGES`` aliases this tuple.
QUALITY_BAND_RANGES: tuple[tuple[str, int, int], ...] = (
    ("brand_new", 95, 100),
    ("like_new", 82, 94),
    ("good", 60, 81),
    ("fair", 35, 59),
    ("damaged", 10, 34),
    ("for_parts", 0, 9),
)

# Normalised inventory condition -> quality band. Same mapping as
# ``handlers._default_band_from_truth``; unknown conditions map to good.
CONDITION_TO_BAND: dict[str, str] = {
    "new": "brand_new",
    "like_new": "like_new",
    "good": "good",
    "fair": "fair",
    "poor": "damaged",
}


def resolve_handoff_checks(
    preset: str = "legacy",
    *,
    inspection_truth_mode: str | None = None,
    commitment_lock_mode: str | None = None,
    completion_integrity_mode: str | None = None,
    shipment_inspection_mode: str | None = None,
) -> dict[str, str]:
    """Return the four handoff-check modes for ``preset`` plus overrides.

    Explicit per-flag values override the preset; ``None`` keeps the
    preset value. Raises ``ValueError`` on an unknown preset or mode,
    like the other platform modes.
    """
    if preset not in HANDOFF_CHECK_PRESETS:
        raise ValueError(
            "handoff_checks must be one of: " + ", ".join(HANDOFF_CHECK_PRESETS)
        )
    resolved = dict(HANDOFF_CHECK_PRESETS[preset])
    overrides = {
        INSPECTION_TRUTH_MODE: inspection_truth_mode,
        COMMITMENT_LOCK_MODE: commitment_lock_mode,
        COMPLETION_INTEGRITY_MODE: completion_integrity_mode,
        SHIPMENT_INSPECTION_MODE: shipment_inspection_mode,
    }
    for key, value in overrides.items():
        if value is None:
            continue
        allowed = HANDOFF_CHECK_MODES[key]
        if value not in allowed:
            raise ValueError(f"{key} must be one of: {', '.join(allowed)}")
        resolved[key] = value
    return resolved


def read_handoff_check(conn: sqlite3.Connection, key: str) -> str:
    """Tolerant read of one handoff-check flag from ``meta``.

    Returns the legacy default when the table or row is missing or the
    stored value is not an allowed mode, so old databases keep their
    legacy behaviour.
    """
    allowed = HANDOFF_CHECK_MODES[key]
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,),
        ).fetchone()
    except Exception:
        return allowed[0]
    if row is None:
        return allowed[0]
    value = str(row[0] or "").strip().lower()
    return value if value in allowed else allowed[0]


def band_range(band: str | None) -> tuple[int, int] | None:
    """``(lowest, highest)`` pct of a quality band, or None if unknown."""
    for name, lo, hi in QUALITY_BAND_RANGES:
        if name == (band or ""):
            return lo, hi
    return None


def band_outcome(quality_pct: int, band: str | None) -> str:
    """Compare an inspected quality with the stated band's pct range.

    ``below_band`` (under the lower edge), ``above_band``,
    ``matches_band``, or ``band_unknown`` when the stated band is missing
    or not one of the six v2 bands (listings seeded or created outside
    ``create_listing`` can have a NULL ``stated_quality_band``).
    """
    bounds = band_range(band)
    if bounds is None:
        return "band_unknown"
    lo, hi = bounds
    if quality_pct < lo:
        return "below_band"
    if quality_pct > hi:
        return "above_band"
    return "matches_band"


def listing_commitments(
    conn: sqlite3.Connection, listing_id: int,
) -> list[int]:
    """Thread ids holding a live commitment on ``listing_id``, primary first.

    A commitment is a deal on the listing that can still go ahead:

    * a thread that is not completed, cancelled or ghosted holds one when
      it is committed, has an accepted offer, or has a scheduled meetup
      or shipment;
    * a meetup or shipment that is still ``scheduled`` holds the listing
      even when its thread was cancelled, ghosted or completed around it
      (legacy ``leave_thread``, ``ghost``, ``mark_sold`` and sister-thread
      cancellation leave such meetups scheduled), because the legacy
      ``complete_transaction`` can still complete it. Under
      ``completion_integrity_mode=unit`` such a meetup can no longer
      complete (``thread_not_active``), so it holds nothing there.

    Cancelling the meetup (which cancels the thread) releases the
    commitment, and so do ``leave_thread`` and ``ghost`` whenever they
    also cancel the thread's scheduled meetup (under
    ``commitment_lock_mode=listing`` or ``completion_integrity_mode=unit``),
    ``mark_sold``, which cancels the listing's scheduled meetups, and a
    completion, which cancels those on the listing's other threads (both
    under either of those or ``inspection_truth_mode=unit``). Nothing
    expires on its own: a commitment holds until one of these happens or
    the deal completes.
    The primary commitment is a thread with a scheduled meetup or
    shipment, else the thread whose accepted offer came first; thread id
    breaks the remaining ties so the order is deterministic.
    """
    integrity = read_handoff_check(conn, COMPLETION_INTEGRITY_MODE) == "unit"
    rows = conn.execute(
        """
        SELECT t.thread_id, t.status,
               (SELECT MIN(o.offer_id) FROM offers o
                 WHERE o.thread_id = t.thread_id
                   AND o.status = 'accepted') AS accepted_offer_id,
               EXISTS (SELECT 1 FROM meetups m
                        WHERE m.thread_id = t.thread_id
                          AND m.status = 'scheduled') AS has_scheduled
        FROM threads t
        WHERE t.listing_id = ?
        ORDER BY t.thread_id
        """,
        (int(listing_id),),
    ).fetchall()
    live: list[tuple[int, int, int, int]] = []
    for row in rows:
        thread_id, status, accepted_offer_id, has_scheduled = (
            int(row[0]), row[1], row[2], bool(row[3]),
        )
        if status in ("completed", "cancelled", "ghosted"):
            # Only a still-scheduled meetup outlives its thread, and only
            # the legacy completion rule lets it complete.
            if not has_scheduled or integrity:
                continue
        elif status != "committed" and accepted_offer_id is None and not has_scheduled:
            continue
        live.append((
            0 if has_scheduled else 1,
            0 if accepted_offer_id is not None else 1,
            int(accepted_offer_id or 0),
            thread_id,
        ))
    live.sort()
    return [entry[3] for entry in live]


# -- Unit identity (inspection_truth_mode=unit) -----------------------------
#
# The create_listing title rule (in-category SequenceMatcher ratio >= 0.45)
# decides which unit is the closest match, but on its own it binds a unit
# to a different model that shares the seller's title template: in the
# reported runs "Apple iPhone 8 64GB ... A1905 - Good Tested" matched
# "Apple iPhone X 256GB ... A1901 - Good Tested", and "Amazon Echo Dot (2nd
# Gen) ... | Tested Working | Local Pickup" matched "Google Home Mini ... |
# Tested Working | Local Pickup". Under inspection_truth_mode=unit a unit
# backs a listing only when the two titles also pass
# :func:`title_identity_conflict`.

_ID_NOISE = re.compile(
    r"[$€£]\s*\d+(?:[.,]\d+)*"                  # prices: $49.99
    r"|\b\d+(?:\.\d+)?\s*%"                               # percentages: 66%
    r"|\b(?:unit|pair|copy|item)(?:\s+#?\s*|\s*#\s*)[a-z0-9]{1,3}\b"  # unit 3
    r"|#\s*\d{1,3}\b"                                     # #2
    r"|\b\d+(?:st|nd|rd|th)\s+(?:unit|pair|copy|one|spare|set|item)\b"  # 2nd unit
)
_ID_DECIMAL_COMMA = re.compile(r"(\d),(\d)")
_ID_SPLIT = re.compile(r"[^\w.]+|_")
_ID_QUANTITY = re.compile(r"^(\d{1,2})x$")
_ID_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")
_ID_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th)$")
_ID_MEASURE = re.compile(r"^(\d+(?:\.\d+)?)([a-z]+)$")
# Unit spellings -> canonical unit, and canonical unit -> measure family.
_ID_UNIT_ALIASES: dict[str, str] = {
    "mm": "mm", "cm": "cm", "in": "in", "inch": "in", "inches": "in", "ft": "ft",
    "gb": "gb", "tb": "tb", "mb": "mb",
    "mp": "mp", "megapixel": "mp", "megapixels": "mp", "p": "p", "i": "i", "k": "k",
    "w": "w", "watt": "w", "watts": "w", "v": "v", "volt": "v", "volts": "v",
    "mah": "mah", "hz": "hz", "khz": "khz", "ghz": "ghz",
    "oz": "oz", "lb": "lb", "lbs": "lb", "kg": "kg", "g": "g", "ml": "ml", "l": "l",
}
_ID_UNIT_FAMILY: dict[str, str] = {
    "mm": "size", "cm": "size", "in": "size", "ft": "size",
    "gb": "storage", "tb": "storage", "mb": "storage",
    "mp": "resolution", "p": "resolution", "i": "resolution", "k": "resolution",
    "w": "power", "v": "power", "mah": "power",
    "hz": "frequency", "khz": "frequency", "ghz": "frequency",
    "oz": "weight", "lb": "weight", "kg": "weight", "g": "weight",
    "ml": "volume", "l": "volume",
}
# Short unit words that also occur as ordinary words ("4 in 1", "2 w/ case")
# are not joined to a preceding number; the bare-number match covers them.
_ID_NO_JOIN_UNITS = frozenset({"in", "i", "g", "l", "w", "p", "k", "v"})
_ID_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "for", "of", "to", "by", "from", "in", "on",
    "at", "with", "w", "only", "plus", "near", "local", "pickup", "approx",
    "approximately", "card", "cards", "lot", "bundle", "collection", "set", "sets",
    "assorted", "include", "included", "includes", "pack", "pair", "pcs", "piece",
    "pieces", "gen", "generation", "year", "years", "yr", "yrs", "subscription",
    "subscriptions", "mit", "und", "de", "del", "la", "le", "et", "avec", "con",
    "per",
})
# Condition, marketing, carrier and colour words: sellers add, drop and swap
# them freely, so they say nothing about which item is meant.
_ID_DESCRIPTOR_WORDS = frozenset({
    "tested", "test", "good", "great", "excellent", "clean", "cleaned", "reset",
    "used", "working", "works", "work", "fast", "quick", "easy", "spare", "unit",
    "units", "condition", "ready", "extra", "budget", "new", "like", "mint",
    "fair", "poor", "damaged", "refurbished", "genuine", "original", "oem",
    "authentic", "replacement", "sale", "deal", "free", "shipping", "ship",
    "shipped", "fully", "functional", "wiped", "factory", "unlocked", "locked",
    "backup", "second", "another", "more", "cheap", "cheaper", "value", "nice",
    "very", "light", "wear", "minor", "scratch", "scratches", "box", "boxed",
    "open", "sealed", "tag", "tags", "ohne", "simlock", "gsm", "cdma", "verizon",
    "sprint", "mobile", "t", "att", "carrier", "black", "white", "gray", "grey",
    "silver", "gold", "rose", "blue", "red", "green", "pink", "purple", "yellow",
    "orange", "brown", "charcoal", "sandstone", "space", "jet", "matte",
    "midnight", "graphite",
})
# Share of the shorter title's product words: at most a third names another
# product (unless both titles carry the same model code); below three
# quarters it does when the leading words (brands) disagree too.
_ID_FEW_SHARED_WORDS = 1 / 3
_ID_SHARED_WORDS_WITH_OTHER_BRAND_MIN = 0.75
_ID_MODEL_CODE_MIN_LENGTH = 4


def _identity_tokens(title: str) -> list[str]:
    """Lower-case identity tokens of a title.

    Prices, percentages and copy labels ("unit 3", "2nd unit") are
    dropped, quantities are reduced to their number ("10x" -> "10"),
    decimal commas become points, and a number followed by a unit word
    is joined to it ("40 mm" -> "40mm", "4 megapixel" -> "4mp").
    """
    text = unicodedata.normalize("NFKC", str(title or "")).lower()
    text = _ID_NOISE.sub(" ", text)
    text = _ID_DECIMAL_COMMA.sub(r"\1.\2", text)
    pieces: list[str] = []
    for piece in _ID_SPLIT.split(text):
        token = piece.strip(".")
        if not token:
            continue
        quantity = _ID_QUANTITY.match(token)
        pieces.append(quantity.group(1) if quantity else token)
    tokens: list[str] = []
    index = 0
    while index < len(pieces):
        token = pieces[index]
        following = pieces[index + 1] if index + 1 < len(pieces) else None
        if (
            following is not None
            and _ID_NUMBER.match(token)
            and following in _ID_UNIT_ALIASES
            and following not in _ID_NO_JOIN_UNITS
        ):
            tokens.append(token + _ID_UNIT_ALIASES[following])
            index += 2
            continue
        measure = _ID_MEASURE.match(token)
        if measure and measure.group(2) in _ID_UNIT_ALIASES:
            token = measure.group(1) + _ID_UNIT_ALIASES[measure.group(2)]
        tokens.append(token)
        index += 1
    return tokens


def _is_identifier(token: str) -> bool:
    """A token with a digit, except a five-digit number (a ZIP code in
    "Local Pickup 12345")."""
    if not any(ch.isdigit() for ch in token):
        return False
    return not (token.isdigit() and len(token) == 5)


def _identifier_kind(token: str) -> str:
    """``version`` (bare numbers and ordinals: "8", "2nd"), a measure
    family (``storage`` "64gb", ``size`` "44mm", ``resolution`` "1080p",
    ...), or ``model`` (other codes: "a1905", "g975u", "studio3")."""
    if _ID_NUMBER.match(token) or _ID_ORDINAL.match(token):
        return "version"
    measure = _ID_MEASURE.match(token)
    if measure and measure.group(2) in _ID_UNIT_FAMILY:
        return _ID_UNIT_FAMILY[measure.group(2)]
    return "model"


def _measure_number(token: str) -> str | None:
    measure = _ID_MEASURE.match(token)
    if measure and measure.group(2) in _ID_UNIT_FAMILY:
        return measure.group(1)
    return None


def _ordinal_number(token: str) -> str | None:
    return token[:-2] if _ID_ORDINAL.match(token) else None


def _joined_forms(tokens: list[str]) -> set[str]:
    """The tokens plus runs of two or three adjacent tokens written
    together ("active 2" -> "active2"; "6 5" also -> "6.5")."""
    forms = set(tokens)
    for index in range(len(tokens) - 1):
        forms.add(tokens[index] + tokens[index + 1])
        if tokens[index].isdigit() and tokens[index + 1].isdigit():
            forms.add(tokens[index] + "." + tokens[index + 1])
        if index + 2 < len(tokens):
            forms.add(tokens[index] + tokens[index + 1] + tokens[index + 2])
    return forms


def _identifier_matched(token: str, others: set[str], other_forms: set[str]) -> bool:
    """Whether the other title names ``token`` too: the same token, the
    same characters split over adjacent words, a shared prefix of three or
    more characters ("g965" / "g965u"), or the bare number of a measure
    ("4mp" / "4") or of an ordinal ("2nd" / "2")."""
    if token in other_forms:
        return True
    measure, ordinal = _measure_number(token), _ordinal_number(token)
    for other in others:
        shorter, longer = sorted((token, other), key=len)
        if len(shorter) >= 3 and longer.startswith(shorter):
            return True
        if other in (measure, ordinal):
            return True
        if token in (_measure_number(other), _ordinal_number(other)):
            return True
    return False


def _identifier_conflict(unit_tokens: list[str], listing_tokens: list[str]) -> bool:
    """Both titles carry an identifier of the same kind that the other
    title does not name (for example "64gb" / "256gb", "2nd" / "3rd",
    "a1905" / "a1901")."""
    unit_ids = {token for token in unit_tokens if _is_identifier(token)}
    listing_ids = {token for token in listing_tokens if _is_identifier(token)}
    unit_forms, listing_forms = _joined_forms(unit_tokens), _joined_forms(listing_tokens)
    listing_only = {
        token for token in listing_ids - unit_ids
        if not _identifier_matched(token, unit_ids, unit_forms)
    }
    unit_only = {
        token for token in unit_ids - listing_ids
        if not _identifier_matched(token, listing_ids, listing_forms)
    }
    kinds = {_identifier_kind(token) for token in listing_only}
    return bool(kinds & {_identifier_kind(token) for token in unit_only})


def _product_word(token: str) -> str | None:
    """A letters-only word that can name the item (plural "s" dropped), or
    None for numbers, function words and descriptor words."""
    if (
        len(token) < 2
        or any(ch.isdigit() for ch in token)
        or token in _ID_STOP_WORDS
        or token in _ID_DESCRIPTOR_WORDS
    ):
        return None
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _shared_product_words(
    unit_tokens: list[str], listing_tokens: list[str],
) -> float | None:
    """Share of the shorter title's product words that the other title
    names too (as a word, or written together with a neighbour), or None
    when either title has no product word."""
    unit_words = {word for word in map(_product_word, unit_tokens) if word}
    listing_words = {word for word in map(_product_word, listing_tokens) if word}
    if not unit_words or not listing_words:
        return None
    unit_forms = {_product_word(form) or form for form in _joined_forms(unit_tokens)}
    listing_forms = {
        _product_word(form) or form for form in _joined_forms(listing_tokens)
    }
    unit_matched = sum(
        1 for word in unit_words if word in listing_words or word in listing_forms
    )
    listing_matched = sum(
        1 for word in listing_words if word in unit_words or word in unit_forms
    )
    shared = max(unit_matched, listing_matched) / min(len(unit_words), len(listing_words))
    return min(1.0, shared)


def _leading_words_agree(unit_tokens: list[str], listing_tokens: list[str]) -> bool:
    """The first product word of either title (usually the brand) appears
    in the other one."""
    unit_lead = next((token for token in unit_tokens if _product_word(token)), None)
    listing_lead = next(
        (token for token in listing_tokens if _product_word(token)), None,
    )
    if unit_lead is None or listing_lead is None:
        return True
    return (
        unit_lead in _joined_forms(listing_tokens)
        or listing_lead in _joined_forms(unit_tokens)
    )


def _share_a_model_code(unit_tokens: list[str], listing_tokens: list[str]) -> bool:
    """Both titles carry the same model code: a letters-and-digits token of
    four or more characters that is not a measure or an ordinal
    ("MJ3U2FD", "A1905"), strong evidence of one product."""
    def codes(tokens: list[str]) -> set[str]:
        return {
            token for token in tokens
            if len(token) >= _ID_MODEL_CODE_MIN_LENGTH
            and _is_identifier(token)
            and _identifier_kind(token) == "model"
        }
    return bool(codes(unit_tokens) & codes(listing_tokens))


def title_identity_conflict(unit_title: str, listing_title: str) -> str | None:
    """Why ``listing_title`` names a different item than ``unit_title``,
    or None when it may name the same item.

    * ``model_identifier``: both titles carry an identifier of the same
      kind that the other does not name: a model or generation number
      ("iPhone 8" / "iPhone 11", "2nd Gen" / "3rd Gen"), a model code
      ("A1905" / "A1901"), or a measure of the same family ("64GB" /
      "256GB", "40mm" / "44mm"). Prices, percentages, ZIP codes and copy
      labels ("Unit 3") are ignored. An identifier matches its spelling
      split over words ("Active 2" / "Active2"), a shared prefix of three
      or more characters ("G965" / "G965U"), and a measure or ordinal
      matches its bare number ("4MP" / "4 Megapixel", "2nd" / "2").
    * ``product_words``: at most a third of the shorter title's product
      words (letters-only words other than function, condition, carrier
      and colour words) appear in the other title, and the titles do not
      share a model code ("MJ3U2FD", which lets a translated title pass).
    * ``brand_and_words``: the first product word (usually the brand) of
      neither title appears in the other, and fewer than three quarters
      of the product words are shared.

    Titles may add or drop descriptors, reorder words, abbreviate,
    translate or shorten without a conflict. The rule is lexical, so two
    titles that differ only by a product word ("Trek Bike Helmet" /
    "Trek Road Bike", "iPhone 12 Pro Case" / "iPhone 12 Pro") do not
    conflict, which is a known limit of the check.
    """
    unit_tokens = _identity_tokens(unit_title)
    listing_tokens = _identity_tokens(listing_title)
    if _identifier_conflict(unit_tokens, listing_tokens):
        return "model_identifier"
    shared = _shared_product_words(unit_tokens, listing_tokens)
    if shared is None:
        return None
    if shared <= _ID_FEW_SHARED_WORDS and not _share_a_model_code(
        unit_tokens, listing_tokens,
    ):
        return "product_words"
    if (
        shared < _ID_SHARED_WORDS_WITH_OTHER_BRAND_MIN
        and not _leading_words_agree(unit_tokens, listing_tokens)
    ):
        return "brand_and_words"
    return None


_PLAIN_WORD = re.compile(r"[^\W_]+")
_ZIP_CODE = re.compile(r"^\d{5}$")


def _plain_words(title: str) -> list[str]:
    """Case-folded letter and digit runs of a title (punctuation and
    spacing dropped: "HWI-PM-120" and "Hwi pm 120" give the same words)."""
    text = unicodedata.normalize("NFKC", str(title or "")).casefold()
    return _PLAIN_WORD.findall(text)


def titles_differ(listing_title: str, unit_title: str) -> bool:
    """Whether a presented unit's title names it differently from the
    listing's title.

    Case, punctuation and spacing are ignored, and so is a trailing run
    of words the listing adds that say nothing about which item it is:
    function, condition, marketing, carrier and colour words and a ZIP
    code ("... - Tested Working | Local Pickup 83060"). Any other
    difference counts, including another word order or a product word
    only one title has ("Apple iPhone 12 Pro" / "Apple iPhone 12 Pro
    Case").
    """
    listing_words = _plain_words(listing_title)
    unit_words = _plain_words(unit_title)
    if listing_words == unit_words:
        return False
    size = len(unit_words)
    if size == 0 or listing_words[:size] != unit_words:
        return True
    return not all(
        word in _ID_STOP_WORDS or word in _ID_DESCRIPTOR_WORDS or _ZIP_CODE.match(word)
        for word in listing_words[size:]
    )
