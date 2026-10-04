"""SQLite schema for BazaarBench.

Design principles:

1. **Append-only event log.**  The ``events`` table is the single
   source of truth for everything that happened.  Other tables are
   derived views over the event log; when counterfactual replay
   re-runs from a snapshot it simply replays events.
2. **Snapshot-friendly.**  Every table has an ``created_at_tick`` and
   rows are never updated destructively; we use status columns
   instead.  This makes a world-state snapshot a single SQLite
   ``.backup``.
3. **No free-text primary keys.**  Every entity gets a typed ID
   (``agent_id``, ``listing_id``, ``thread_id``, ``photo_id``,
   ``offer_id``, ``meetup_id``) so that replay and attribution can
   reference events unambiguously.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

# ``events`` is the append-only event log.  All other tables are views
# on top of it.  ``payload`` is JSON-serialised action args and result.
EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    tick           INTEGER NOT NULL
                   CHECK (typeof(tick) = 'integer'),
    wall_time      TEXT    NOT NULL,
    agent_id       INTEGER                -- NULL for platform-emitted events
                   CHECK (agent_id IS NULL OR typeof(agent_id) = 'integer'),
    action_type    TEXT    NOT NULL
                   CHECK (length(trim(action_type)) > 0),
    payload        TEXT    NOT NULL       -- JSON object
                   CHECK (
                     CASE
                       WHEN json_valid(payload)
                       THEN json_type(payload) = 'object'
                       ELSE 0
                     END
                   ),
    result_status  TEXT    NOT NULL       -- 'ok' | 'error' | 'blocked'
                   CHECK (result_status IN ('ok', 'error', 'blocked')),
    result_payload TEXT                   -- JSON object, NULL when absent
                   CHECK (
                     result_payload IS NULL OR
                     CASE
                       WHEN json_valid(result_payload)
                       THEN json_type(result_payload) = 'object'
                       ELSE 0
                     END
                   )
);
CREATE INDEX IF NOT EXISTS idx_events_tick   ON events (tick);
CREATE INDEX IF NOT EXISTS idx_events_agent  ON events (agent_id);
CREATE INDEX IF NOT EXISTS idx_events_action ON events (action_type);
"""

# Agents (minimal persona fields; full persona JSON lives in ``persona``).
AGENTS_DDL = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id            INTEGER PRIMARY KEY,
    user_name           TEXT    NOT NULL UNIQUE,
    display_name        TEXT    NOT NULL,
    home_zip            TEXT    NOT NULL,
    home_lat            REAL    NOT NULL,
    home_lng            REAL    NOT NULL,
    activity_rate       REAL    NOT NULL DEFAULT 0.3,
    privacy_awareness   REAL    NOT NULL DEFAULT 0.5,
    device              TEXT    NOT NULL DEFAULT 'iPhone 14',
    persona_json        TEXT    NOT NULL,     -- full PersonaCard as JSON
    parent_agent_id     INTEGER REFERENCES agents(agent_id), -- sub-accounts
    created_at_tick     INTEGER NOT NULL DEFAULT 0,
    status              TEXT    NOT NULL DEFAULT 'active',  -- 'active' | 'banned'
    -- R18: reference-cohort flag for the aggressive-profit subpopulation
    -- (≈20% of generated personas). 'neutral' is the capability-neutral
    -- default; 'aggressive' agents receive a factual trait line in
    -- prompt_summary describing a profit-first disposition. Stored as
    -- a dedicated column so analysis queries can stratify without
    -- parsing every persona_json blob.
    risk_posture        TEXT    NOT NULL DEFAULT 'neutral',
    -- R19: social-learning seed flag. Non-zero for the synthetic
    -- "seller" / "buyer" accounts that back the lot-sale seed feed.
    -- Seed agents exist only to anchor historical threads/offers/
    -- ratings so recent_sales_feed surfaces them; they have no
    -- policy and never act at tick-time. Analysis queries filter
    -- WHERE is_seeded = 0 to recover real-population metrics.
    is_seeded           INTEGER NOT NULL DEFAULT 0,
    -- R20r: red-team instrumentation flag. Non-zero for agents whose
    -- LLM policy receives the adversarial RED_TEAM_SYSTEM_TEMPLATE
    -- (explicit "security researcher trying to defraud the
    -- marketplace" instructions). These are research instrumentation,
    -- NOT part of the capability-neutral benign population — analysis
    -- stratifies benign-vs-red-team and counts how many benign
    -- buyers are successfully victimised. Capability-neutral claims
    -- apply only to rows with is_redteam = 0.
    is_redteam          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_agents_parent ON agents (parent_agent_id);
CREATE INDEX IF NOT EXISTS idx_agents_risk_posture ON agents (risk_posture);
CREATE INDEX IF NOT EXISTS idx_agents_is_seeded ON agents (is_seeded);
CREATE INDEX IF NOT EXISTS idx_agents_is_redteam ON agents (is_redteam);
"""

LISTINGS_DDL = """
CREATE TABLE IF NOT EXISTS listings (
    listing_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_agent_id   INTEGER REFERENCES agents(agent_id), -- NULL = phantom
    category         TEXT    NOT NULL,
    title            TEXT    NOT NULL,
    description      TEXT    NOT NULL,
    price_cents      INTEGER NOT NULL,
    condition        TEXT    NOT NULL,   -- 'new'|'like_new'|'good'|'fair'|'poor'
    location_zip     TEXT    NOT NULL,
    location_lat     REAL    NOT NULL,
    location_lng     REAL    NOT NULL,
    is_phantom       INTEGER NOT NULL DEFAULT 0,
    view_count       INTEGER NOT NULL DEFAULT 0,
    save_count       INTEGER NOT NULL DEFAULT 0,
    inquiry_count    INTEGER NOT NULL DEFAULT 0,
    created_at_tick  INTEGER NOT NULL,
    last_bumped_tick INTEGER,
    sold_at_tick     INTEGER,
    status           TEXT    NOT NULL DEFAULT 'active',
    -- R15 Part 2: speculative-listing tag. 1 when a listing's title
    -- does NOT fuzzy-match any persona.inventory_items row in the
    -- same category at create time. Never surfaced to other agents
    -- (info asymmetry — only researchers query it directly).
    is_speculative             INTEGER NOT NULL DEFAULT 0,
    inventory_match_confidence REAL    DEFAULT NULL,
    -- R19: lot-sale seed flag. 1 for listings that back the synthetic
    -- social-learning feed (hand-authored historical "sales" of lot/
    -- bundle items at high prices). The feed queries do not filter
    -- these out — that's the whole point — but H3 analysis must
    -- stratify WHERE is_seeded = 0 to avoid counting seed rows as
    -- emergent speculative behaviour.
    is_seeded                  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_listings_owner    ON listings (owner_agent_id);
CREATE INDEX IF NOT EXISTS idx_listings_category ON listings (category);
CREATE INDEX IF NOT EXISTS idx_listings_zip      ON listings (location_zip);
CREATE INDEX IF NOT EXISTS idx_listings_status   ON listings (status);
CREATE INDEX IF NOT EXISTS idx_listings_is_seeded ON listings (is_seeded);
"""

# Threads are agent-to-listing conversations.  Multi-party threads are
# modeled by having multiple thread rows all pointing at the same listing.
THREADS_DDL = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id      INTEGER NOT NULL REFERENCES listings(listing_id),
    buyer_agent_id  INTEGER NOT NULL REFERENCES agents(agent_id),
    seller_agent_id INTEGER REFERENCES agents(agent_id), -- NULL for phantom listings
    created_at_tick INTEGER NOT NULL,
    last_msg_tick   INTEGER,
    status          TEXT    NOT NULL DEFAULT 'open'
                    -- 'open' | 'committed' | 'completed' | 'cancelled' | 'ghosted'
);
CREATE INDEX IF NOT EXISTS idx_threads_listing ON threads (listing_id);
CREATE INDEX IF NOT EXISTS idx_threads_buyer   ON threads (buyer_agent_id);
CREATE INDEX IF NOT EXISTS idx_threads_seller  ON threads (seller_agent_id);
"""

MESSAGES_DDL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id        INTEGER NOT NULL REFERENCES threads(thread_id),
    sender_agent_id  INTEGER NOT NULL REFERENCES agents(agent_id),
    tick             INTEGER NOT NULL,
    body             TEXT    NOT NULL,
    photo_id         INTEGER REFERENCES photos(photo_id),
    read_at_tick     INTEGER,
    -- Present for counterfactual replay: the deterministic content hash
    -- lets us detect whether a replay produced the same message or diverged.
    content_hash     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages (thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages (sender_agent_id);
"""

# Symbolic photos.  See bazaar.photos for the three types (A/B/C).
PHOTOS_DDL = """
CREATE TABLE IF NOT EXISTS photos (
    photo_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_type         TEXT    NOT NULL,  -- 'A' | 'B' | 'C'
    sender_agent_id    INTEGER NOT NULL REFERENCES agents(agent_id),
    listing_id         INTEGER REFERENCES listings(listing_id),
    subject_attrs      TEXT    NOT NULL,  -- JSON dict
    background_leaks   TEXT    NOT NULL,  -- JSON dict (empty for type C)
    metadata_leaks     TEXT    NOT NULL,  -- JSON dict (empty for type C)
    seller_aware_of    TEXT    NOT NULL,  -- JSON list of field names
    is_stock           INTEGER NOT NULL DEFAULT 0,
    ground_truth       TEXT,              -- JSON: true fields if type B claimed-false
    created_at_tick    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_sender  ON photos (sender_agent_id);
CREATE INDEX IF NOT EXISTS idx_photos_listing ON photos (listing_id);
"""

OFFERS_DDL = """
CREATE TABLE IF NOT EXISTS offers (
    offer_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       INTEGER NOT NULL REFERENCES threads(thread_id),
    proposer_id     INTEGER NOT NULL REFERENCES agents(agent_id),
    round           INTEGER NOT NULL,
    price_cents     INTEGER NOT NULL,
    terms_json      TEXT    NOT NULL,
    tick            INTEGER NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending'
                    -- 'pending' | 'accepted' | 'countered' | 'withdrawn' | 'rejected'
);
CREATE INDEX IF NOT EXISTS idx_offers_thread ON offers (thread_id);
"""

MEETUPS_DDL = """
CREATE TABLE IF NOT EXISTS meetups (
    meetup_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id        INTEGER NOT NULL REFERENCES threads(thread_id),
    scheduled_tick   INTEGER NOT NULL,
    location_desc    TEXT    NOT NULL,
    payment_method   TEXT    NOT NULL,  -- 'cash'|'zelle'|'venmo'|'on_platform'
    buyer_confirmed  INTEGER NOT NULL DEFAULT 0,
    seller_confirmed INTEGER NOT NULL DEFAULT 0,
    -- Hidden platform-side oracle token. Not rendered in ordinary
    -- prompts; external handoff channels can reveal it to participants.
    handoff_token    TEXT,
    status           TEXT    NOT NULL DEFAULT 'scheduled'
                     -- 'scheduled'|'completed'|'cancelled'|'no_show'
);
"""

RATINGS_DDL = """
CREATE TABLE IF NOT EXISTS ratings (
    rating_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    rater_agent_id   INTEGER NOT NULL REFERENCES agents(agent_id),
    ratee_agent_id   INTEGER NOT NULL REFERENCES agents(agent_id),
    thread_id        INTEGER REFERENCES threads(thread_id),
    stars            INTEGER NOT NULL CHECK (stars BETWEEN 1 AND 5),
    body             TEXT,
    tick             INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ratings_ratee ON ratings (ratee_agent_id);
"""

BLOCKS_DDL = """
CREATE TABLE IF NOT EXISTS blocks (
    block_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    blocker_id      INTEGER NOT NULL REFERENCES agents(agent_id),
    blocked_id      INTEGER NOT NULL REFERENCES agents(agent_id),
    tick            INTEGER NOT NULL,
    UNIQUE (blocker_id, blocked_id)
);
"""

REPORTS_DDL = """
CREATE TABLE IF NOT EXISTS reports (
    report_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    reporter_id     INTEGER NOT NULL REFERENCES agents(agent_id),
    target_kind     TEXT    NOT NULL,   -- 'listing' | 'user'
    target_id       INTEGER NOT NULL,
    reason          TEXT    NOT NULL,
    tick            INTEGER NOT NULL,
    moderator_action TEXT              -- filled by moderator dynamic (D9)
);
"""

# Structured ledger per agent.  This is NOT the raw event log; it's a
# per-agent materialized summary that gets auto-injected into the LLM
# context.  Think of it as "the things the platform remembers for you,
# that you cannot forget".
LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS ledger_entries (
    entry_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id      INTEGER NOT NULL REFERENCES agents(agent_id),
    kind          TEXT    NOT NULL,  -- 'transaction'|'rating'|'block'|'report'
    counterparty_id INTEGER REFERENCES agents(agent_id),
    ref_table     TEXT    NOT NULL,
    ref_id        INTEGER NOT NULL,
    summary       TEXT    NOT NULL,
    tick          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_agent ON ledger_entries (agent_id);
"""

# Narrative memory — fuzzy free-text.  Vector embedding in Phase 2.
# ``provenance`` (Phase 3, schema v2) is the agent_id of the source when
# this narrative was passed in from another agent (e.g. via
# QUOTE_AGENT_NOTE); NULL for self-generated impressions. This is
# load-bearing for entry-point-E "inherited drift" experiments — we need
# to trace how a narrative about agent X propagates through the graph.
NARRATIVE_DDL = """
CREATE TABLE IF NOT EXISTS narrative_memories (
    memory_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id      INTEGER NOT NULL REFERENCES agents(agent_id),
    scope         TEXT    NOT NULL,   -- 'thread'|'counterparty'|'self'
    scope_ref_id  INTEGER,
    content       TEXT    NOT NULL,
    embedding     BLOB,                -- float32 vector
    created_tick  INTEGER NOT NULL,
    decayed       INTEGER NOT NULL DEFAULT 0,
    provenance    INTEGER REFERENCES agents(agent_id)
);
CREATE INDEX IF NOT EXISTS idx_narrative_agent ON narrative_memories (agent_id);
"""

# D12 self-portraits — stored for identity-drift detection.
SELF_PORTRAITS_DDL = """
CREATE TABLE IF NOT EXISTS self_portraits (
    portrait_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id      INTEGER NOT NULL REFERENCES agents(agent_id),
    tick          INTEGER NOT NULL,
    content       TEXT    NOT NULL,
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS idx_portrait_agent ON self_portraits (agent_id);
"""

# R14a Layer-1: rolling per-agent self-summary produced by D14.
# One dedicated reflection call per firing distills the agent's
# current "what am I doing, what have I learned, what do I want next"
# into a short paragraph. The most recent row per agent is injected
# into every LLMPolicy prompt as the ## MY CURRENT STATE block so the
# agent's behaviour stays coherent across many ticks without blowing
# the context window. Append-only — older rows kept for drift diffing.
AGENT_SUMMARY_DDL = """
CREATE TABLE IF NOT EXISTS agent_summary (
    summary_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    INTEGER NOT NULL REFERENCES agents(agent_id),
    tick        INTEGER NOT NULL,
    content     TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'D14'  -- 'D14' | 'fallback'
);
CREATE INDEX IF NOT EXISTS idx_agent_summary_agent ON agent_summary (agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_summary_tick  ON agent_summary (tick);
"""

# D13 snapshots — pointer table; the actual SQLite backup lives on disk.
SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    tick          INTEGER NOT NULL,
    wall_time     TEXT    NOT NULL,
    backup_path   TEXT    NOT NULL,
    rng_state     BLOB    NOT NULL,
    llm_cache_hash TEXT
);
"""

META_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# llm_calls — Phase-3 (schema v2) addition. Every real LLM invocation
# produces a row here: prompt_hash + response text + sampling params +
# seed. Together these operationalize R9 and R14 of the design study:
# CIS replay needs prompt_hash to decide whether a replayed prompt
# should hit cache, and the rolling hash of (prompt_hash, response)
# over all rows up to tick T is the ``llm_cache_hash`` stored in the
# D13 snapshot. Rows are append-only, never updated.
LLM_CALLS_DDL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tick               INTEGER NOT NULL
                       CHECK (typeof(tick) = 'integer'),
    agent_id           INTEGER NOT NULL REFERENCES agents(agent_id)
                       CHECK (typeof(agent_id) = 'integer'),
    model              TEXT    NOT NULL
                       CHECK (length(trim(model)) > 0),
    backend            TEXT    NOT NULL
                       CHECK (length(trim(backend)) > 0),
    prompt_hash        TEXT    NOT NULL
                       CHECK (length(trim(prompt_hash)) > 0),
    prompt_text        TEXT,
    sampling_params    TEXT    NOT NULL
                       CHECK (
                         CASE
                           WHEN json_valid(sampling_params)
                           THEN json_type(sampling_params) = 'object'
                           ELSE 0
                         END
                       ),
    response_text      TEXT    NOT NULL,
    tool_calls_json    TEXT
                       CHECK (
                         tool_calls_json IS NULL OR
                         CASE
                           WHEN json_valid(tool_calls_json)
                           THEN json_type(tool_calls_json) = 'array'
                           ELSE 0
                         END
                       ),
    reasoning_summary  TEXT,
    seed               INTEGER,
    cache_hit          INTEGER NOT NULL DEFAULT 0
                       CHECK (cache_hit IN (0, 1)),
    latency_ms         INTEGER
                       CHECK (latency_ms IS NULL OR latency_ms >= 0),
    wall_time          TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_tick  ON llm_calls (tick);
CREATE INDEX IF NOT EXISTS idx_llm_calls_agent ON llm_calls (agent_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_hash  ON llm_calls (prompt_hash);
"""

REQUIRED_SCHEMA_CONSTRAINTS: dict[str, dict[str, str]] = {
    "events": {
        "events.tick_integer": "typeof(tick)='integer'",
        "events.agent_id_integer": "agent_idisnullortypeof(agent_id)='integer'",
        "events.action_type_nonempty": "length(trim(action_type))>0",
        "events.payload_json_object": "json_type(payload)='object'",
        "events.result_status_enum": "result_statusin('ok','error','blocked')",
        "events.result_payload_json_object": "json_type(result_payload)='object'",
    },
    "llm_calls": {
        "llm_calls.tick_integer": "typeof(tick)='integer'",
        "llm_calls.agent_id_integer": "typeof(agent_id)='integer'",
        "llm_calls.model_nonempty": "length(trim(model))>0",
        "llm_calls.backend_nonempty": "length(trim(backend))>0",
        "llm_calls.prompt_hash_nonempty": "length(trim(prompt_hash))>0",
        "llm_calls.sampling_params_json_object": "json_type(sampling_params)='object'",
        "llm_calls.tool_calls_json_array": "json_type(tool_calls_json)='array'",
        "llm_calls.cache_hit_boolean": "cache_hitin(0,1)",
        "llm_calls.latency_ms_nonnegative": "latency_msisnullorlatency_ms>=0",
    },
}

# mental_prices — R14b Part A. Each row is one private walkaway-price
# probe: asking the LLM what price it would pay/accept *right now* for
# a specific listing, under its persona + goal + stress state. Stages
# (initial / after_chat / after_compare / final) are the measurement
# points that let the drift metric (Part B) compute how much the
# agent's private valuation shifted after chat + final commitment.
# Append-only — older probes kept so a single (listing, agent, role)
# can have its full mental-price trajectory reconstructed.
MENTAL_PRICES_DDL = """
CREATE TABLE IF NOT EXISTS mental_prices (
    entry_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id            INTEGER NOT NULL,
    agent_id              INTEGER NOT NULL,
    role                  TEXT    NOT NULL,  -- 'buyer' | 'seller'
    stage                 TEXT    NOT NULL,  -- 'initial' | 'after_chat'
                                              -- | 'after_compare' | 'final'
    mental_price_cents    INTEGER NOT NULL,
    market_baseline_cents INTEGER,
    rationale             TEXT,
    tick                  INTEGER NOT NULL,
    created_at            TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mental_prices_listing
  ON mental_prices (listing_id, agent_id, role);
CREATE INDEX IF NOT EXISTS idx_mental_prices_tick
  ON mental_prices (tick);
"""

# R14b Part B — one row per accepted offer, snapshotting both agents'
# mental-price trajectory + stress state at commitment so drift can be
# analysed post-hoc without replaying the run.
#
#   buyer_drift  = final_price - buyer_initial       (positive = paid more)
#   seller_drift = seller_initial - final_price      (positive = accepted less)
#   market_premium = final_price - market_baseline   (positive = above market)
#
# All mental-price columns nullable because the probe may fail or the
# relevant stage may not have fired (e.g. seller never held a chat).
TRANSACTION_UTILITY_DDL = """
CREATE TABLE IF NOT EXISTS transaction_utility (
    entry_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_id                INTEGER NOT NULL,
    thread_id               INTEGER NOT NULL,
    listing_id              INTEGER NOT NULL,
    buyer_agent_id          INTEGER NOT NULL,
    seller_agent_id         INTEGER,
    final_price_cents       INTEGER NOT NULL,
    market_baseline_cents   INTEGER,
    buyer_initial_mental    INTEGER,
    buyer_after_chat        INTEGER,
    buyer_final             INTEGER,
    seller_initial_mental   INTEGER,
    seller_after_chat       INTEGER,
    buyer_drift             INTEGER,
    seller_drift            INTEGER,
    market_premium          INTEGER,
    buyer_stress_at_commit  TEXT,
    seller_stress_at_commit TEXT,
    time_to_deadline_buyer  INTEGER,
    accept_tick             INTEGER NOT NULL,
    created_at              TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transaction_utility_offer
  ON transaction_utility (offer_id);
CREATE INDEX IF NOT EXISTS idx_transaction_utility_listing
  ON transaction_utility (listing_id);
CREATE INDEX IF NOT EXISTS idx_transaction_utility_tick
  ON transaction_utility (accept_tick);
"""

ALL_DDL = [
    META_DDL,
    EVENTS_DDL,
    AGENTS_DDL,
    LISTINGS_DDL,
    # photos must come before messages (FK dependency)
    PHOTOS_DDL,
    THREADS_DDL,
    MESSAGES_DDL,
    OFFERS_DDL,
    MEETUPS_DDL,
    RATINGS_DDL,
    BLOCKS_DDL,
    REPORTS_DDL,
    LEDGER_DDL,
    NARRATIVE_DDL,
    SELF_PORTRAITS_DDL,
    AGENT_SUMMARY_DDL,
    SNAPSHOTS_DDL,
    LLM_CALLS_DDL,
    MENTAL_PRICES_DDL,
    TRANSACTION_UTILITY_DDL,
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def initialize_db(db_path: str | Path) -> sqlite3.Connection:
    """Create a fresh BazaarBench database and stamp metadata.

    If the file already exists, it is replaced; callers are expected to
    manage their own on-disk lifecycle (smoke-test runs recreate every
    time; real experiments use a separate run directory per seed).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        for ddl in ALL_DDL:
            conn.executescript(ddl)
        # R11: idempotent migration — for any path that ends up at this
        # function with a pre-existing llm_calls table missing the new
        # column, ALTER it in. ``initialize_db`` itself wipes the file
        # first so this is a defensive no-op for fresh dbs; the same
        # check also runs in ``connect()`` below for legacy on-disk dbs.
        _migrate_llm_calls_reasoning_summary(conn)
        _migrate_agents_risk_posture(conn)
        _migrate_is_seeded_columns(conn)
        _migrate_agents_is_redteam(conn)
        _migrate_meetups_handoff_token(conn)
        _migrate_listings_quality(conn)
        _migrate_meetups_delivery_method(conn)
        _migrate_handoff_check_columns(conn)
        _stamp_schema_version(conn)
    return conn


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open an existing BazaarBench DB with the same pragmas."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    with conn:
        # R11: legacy dbs created before reasoning_summary existed need the
        # column added at open time so writers don't fail on INSERT.
        _migrate_llm_calls_reasoning_summary(conn)
        # R15 Part 2: legacy dbs created before the speculative-tagging
        # columns existed need them added at open time so create_listing
        # writes don't fail.
        _migrate_listings_speculative(conn)
        # R18: legacy dbs created before the reference-cohort flag existed
        # need agents.risk_posture added at open time so add_agent writes
        # don't fail.
        _migrate_agents_risk_posture(conn)
        # R19: legacy dbs created before lot-sale seeding need both
        # is_seeded columns before seed_lot_sales_feed / analysis tooling
        # tries to read them.
        _migrate_is_seeded_columns(conn)
        # R20r: red-team instrumentation column.
        _migrate_agents_is_redteam(conn)
        _migrate_meetups_handoff_token(conn)
        # v2 environment: quality model + delivery method.
        _migrate_listings_quality(conn)
        _migrate_meetups_delivery_method(conn)
        # Truthful handoff checks: nullable unit binding + inspection
        # outcome columns. They stay NULL under the legacy defaults.
        _migrate_handoff_check_columns(conn)
        # R14a: legacy dbs created before agent_summary existed need the
        # table created at open time so D14 reflection writes don't fail.
        # All DDL is ``CREATE TABLE IF NOT EXISTS`` so this is idempotent
        # and safe to run unconditionally on every connect.
        for ddl in ALL_DDL:
            conn.executescript(ddl)
        _stamp_schema_version(conn)
    return conn


def _stamp_schema_version(conn: sqlite3.Connection) -> None:
    version = 0 if missing_required_schema_constraints(conn) else SCHEMA_VERSION
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("schema_version", str(version)),
    )


def missing_required_schema_constraints(
    conn: sqlite3.Connection,
) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for table, required_fragments in REQUIRED_SCHEMA_CONSTRAINTS.items():
        row = conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table,),
        ).fetchone()
        if row is None:
            missing.append((table, f"{table}.table_exists"))
            continue
        raw_sql = row["sql"] if isinstance(row, sqlite3.Row) else row[0]
        compact_sql = "".join(str(raw_sql or "").lower().split())
        for label, fragment in required_fragments.items():
            if fragment not in compact_sql:
                missing.append((table, label))
    return missing


def _migrate_llm_calls_reasoning_summary(
    conn: sqlite3.Connection,
) -> None:
    """Add ``llm_calls.reasoning_summary`` if missing (R11).

    Tolerates the case where the table doesn't exist yet (very old
    snapshots predating Phase 3) by silently skipping.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(llm_calls)")}
    except sqlite3.OperationalError:
        return
    if not cols:
        return
    if "reasoning_summary" not in cols:
        try:
            conn.execute(
                "ALTER TABLE llm_calls ADD COLUMN reasoning_summary TEXT"
            )
        except sqlite3.OperationalError:
            # Another writer added the column between PRAGMA and ALTER,
            # or the db is locked. Readers tolerate the missing column
            # via SQL (SELECT name FROM PRAGMA at use-site).
            return


def _migrate_agents_risk_posture(
    conn: sqlite3.Connection,
) -> None:
    """Add ``agents.risk_posture`` if missing (R18). Idempotent.

    Applied at ``initialize_db`` time (defensive no-op for fresh DBs)
    and at ``connect`` time so legacy DBs resumed for a new run get
    the column before any ``add_agent`` INSERT tries to write it.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(agents)")}
    except sqlite3.OperationalError:
        return
    if not cols:
        return
    if "risk_posture" not in cols:
        try:
            conn.execute(
                "ALTER TABLE agents ADD COLUMN "
                "risk_posture TEXT NOT NULL DEFAULT 'neutral'"
            )
        except sqlite3.OperationalError:
            return


def _migrate_agents_is_redteam(
    conn: sqlite3.Connection,
) -> None:
    """Add ``agents.is_redteam`` if missing (R20r). Idempotent."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(agents)")}
    except sqlite3.OperationalError:
        return
    if not cols:
        return
    if "is_redteam" not in cols:
        try:
            conn.execute(
                "ALTER TABLE agents ADD COLUMN "
                "is_redteam INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            return


def _migrate_is_seeded_columns(
    conn: sqlite3.Connection,
) -> None:
    """Add ``agents.is_seeded`` + ``listings.is_seeded`` if missing (R19).
    Idempotent. Tolerates the very-old-snapshot case where a target
    table doesn't yet exist.
    """
    for table in ("agents", "listings"):
        try:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.OperationalError:
            continue
        if not cols:
            continue
        if "is_seeded" not in cols:
            try:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN "
                    "is_seeded INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                continue


def _migrate_listings_speculative(
    conn: sqlite3.Connection,
) -> None:
    """Add ``listings.is_speculative`` + ``inventory_match_confidence``
    if missing (R15 Part 2). Idempotent. Tolerates the pre-Phase-2
    snapshot case where the table doesn't yet exist.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(listings)")}
    except sqlite3.OperationalError:
        return
    if not cols:
        return
    if "is_speculative" not in cols:
        try:
            conn.execute(
                "ALTER TABLE listings ADD COLUMN "
                "is_speculative INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            return
    if "inventory_match_confidence" not in cols:
        try:
            conn.execute(
                "ALTER TABLE listings ADD COLUMN "
                "inventory_match_confidence REAL DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            return


def _migrate_meetups_handoff_token(
    conn: sqlite3.Connection,
) -> None:
    """Add hidden platform-side handoff token storage if missing."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(meetups)")}
    except sqlite3.OperationalError:
        return
    if not cols:
        return
    if "handoff_token" not in cols:
        try:
            conn.execute("ALTER TABLE meetups ADD COLUMN handoff_token TEXT")
        except sqlite3.OperationalError:
            return


def _migrate_listings_quality(conn: sqlite3.Connection) -> None:
    """v2 environment: add ground-truth quality + stated quality band +
    acquisition cost. Idempotent.

    - ``ground_truth_quality_pct``: integer 0..100 the platform records
      as the item's true condition. Hidden from other agents until
      meetup inspection or shipment delivery surfaces it.
    - ``stated_quality_band``: text bucket the seller claimed at create
      time ('brand_new'/'like_new'/'good'/'fair'/'damaged'/'for_parts').
      Visible to all agents in listing previews.
    - ``acquisition_cost_cents``: what the seller paid for the item
      (cold-start: synthesised; rollout-restock: from the restock
      dynamic). Used to compute realised profit; an agent can sell at
      a loss intentionally.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(listings)")}
    except sqlite3.OperationalError:
        return
    if not cols:
        return
    if "ground_truth_quality_pct" not in cols:
        try:
            conn.execute(
                "ALTER TABLE listings ADD COLUMN "
                "ground_truth_quality_pct INTEGER DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            return
    if "stated_quality_band" not in cols:
        try:
            conn.execute(
                "ALTER TABLE listings ADD COLUMN "
                "stated_quality_band TEXT DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            return
    if "acquisition_cost_cents" not in cols:
        try:
            conn.execute(
                "ALTER TABLE listings ADD COLUMN "
                "acquisition_cost_cents INTEGER DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            return
    if "reference_fair_price_cents" not in cols:
        try:
            # v2.23: per-listing scrape-grounded fair price prior, copied
            # from the matched inventory item's ``asking_price_cents``
            # at create_listing time. NULL for off-inventory listings.
            # Used for objective-shift detection in §4 instead of a
            # category-mean baseline.
            conn.execute(
                "ALTER TABLE listings ADD COLUMN "
                "reference_fair_price_cents INTEGER DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            return
    if "quality_adjusted_fair_price_cents" not in cols:
        try:
            # v2.23: quality-aware fair price computed post-hoc as
            # mean(asking_price) over (brand, model, storage_GB) inventory
            # items, multiplied by (ground_truth_quality_pct / 100). This
            # is the quality-adjusted "average price for this item at this
            # quality level" used as the ground-truth reference for
            # objective-shift detection. Populated by
            # ``scripts/cold_start/backfill_fair_price.py``; NULL until
            # backfill runs.
            conn.execute(
                "ALTER TABLE listings ADD COLUMN "
                "quality_adjusted_fair_price_cents INTEGER DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            return


def _migrate_meetups_delivery_method(conn: sqlite3.Connection) -> None:
    """v2: add ``delivery_method`` ('meetup'|'ship') so the same
    completion handler can serve both fulfilment paths. Default
    'meetup' preserves pre-v2 semantics. Idempotent.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(meetups)")}
    except sqlite3.OperationalError:
        return
    if not cols:
        return
    if "delivery_method" not in cols:
        try:
            conn.execute(
                "ALTER TABLE meetups ADD COLUMN "
                "delivery_method TEXT NOT NULL DEFAULT 'meetup'"
            )
        except sqlite3.OperationalError:
            return
    if "buyer_inspected_quality_pct" not in cols:
        try:
            conn.execute(
                "ALTER TABLE meetups ADD COLUMN "
                "buyer_inspected_quality_pct INTEGER DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            return
    if "delivered_at_tick" not in cols:
        try:
            conn.execute(
                "ALTER TABLE meetups ADD COLUMN "
                "delivered_at_tick INTEGER DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            return


def _migrate_handoff_check_columns(conn: sqlite3.Connection) -> None:
    """Truthful handoff checks: add nullable
    columns if missing. Idempotent; safe on old DBs.

    - ``listings.backing_unit_uid``: the seller inventory unit bound to
      the listing under ``inspection_truth_mode=unit`` (at
      ``create_listing`` or ``relist``, or at the handoff when the
      listing had none) or bound by a unit-aware sale of an unbound
      listing. NULL when no unit backs the listing and always NULL under
      the legacy default.
    - ``meetups.inspection_outcome``: ``below_band`` / ``matches_band`` /
      ``above_band`` / ``band_unknown`` (no stated band) /
      ``item_not_present`` recorded by ``inspect_at_meetup`` under
      ``inspection_truth_mode=unit``. NULL under the legacy default.
    """
    for table, column in (
        ("listings", "backing_unit_uid"),
        ("meetups", "inspection_outcome"),
    ):
        try:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.OperationalError:
            continue
        if not cols or column in cols:
            continue
        try:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} TEXT DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            continue


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    return int(row[0] if isinstance(row, tuple) else row["value"])
