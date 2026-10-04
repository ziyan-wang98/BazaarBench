"""Smoke tests for the Phase-2 multi-surface dashboard renderer."""
from __future__ import annotations

import pytest

from bazaar import BazaarEnv, MarketAgent, RandomBenignPolicy, generate_persona
from bazaar.memory import HashEncoder, NarrativeStore, install_store
from bazaar.viz import write_dashboard


@pytest.fixture
def populated_env(tmp_db):
    env = BazaarEnv(db_path=tmp_db, seed_phantom_listings=3)
    install_store(
        env.platform.conn,
        NarrativeStore(env.platform.conn, encoder=HashEncoder()),
    )
    for i in range(6):
        env.add_agent(
            MarketAgent(
                persona=generate_persona(i + 1, seed=1000 + i),
                policy=RandomBenignPolicy(seed=1000 + i),
            )
        )
    env.reset()
    env.step_many(30)
    yield env
    env.close()


def test_dashboard_writes_every_surface(populated_env, tmp_path):
    out = tmp_path / "site"
    written = write_dashboard(populated_env.platform.db_path, out)
    assert written == out

    for f in ("index.html", "threads.html", "agents.html",
              "map.html", "metrics.html"):
        assert (out / f).exists() and (out / f).stat().st_size > 500

    # One profile per agent.
    profiles = list((out / "agents").glob("agent_*.html"))
    assert len(profiles) == 6


def test_index_references_navbar_surfaces(populated_env, tmp_path):
    out = tmp_path / "site"
    write_dashboard(populated_env.platform.db_path, out)
    text = (out / "index.html").read_text(encoding="utf-8")
    for href in ("index.html", "threads.html", "agents.html",
                 "map.html", "metrics.html"):
        assert f'href="{href}"' in text or f'href="./{href}"' in text


def test_profile_pages_prefix_nav_for_subdir(populated_env, tmp_path):
    out = tmp_path / "site"
    write_dashboard(populated_env.platform.db_path, out)
    any_profile = next((out / "agents").glob("agent_*.html"))
    text = any_profile.read_text(encoding="utf-8")
    # Subdir pages must prefix the nav hrefs with ../ so Feed/Threads
    # etc. still resolve.
    assert 'href="../index.html"' in text
    assert 'href="../threads.html"' in text


def test_dashboard_survives_empty_database(tmp_path):
    from bazaar.core.schema import initialize_db
    db = tmp_path / "empty.db"
    conn = initialize_db(db)
    conn.close()
    out = tmp_path / "site"
    write_dashboard(db, out)
    for f in ("index.html", "threads.html", "agents.html",
              "map.html", "metrics.html"):
        assert (out / f).exists()
