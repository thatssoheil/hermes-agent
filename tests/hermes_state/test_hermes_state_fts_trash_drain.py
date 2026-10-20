"""#141131: the opportunistic drain of an interrupted optimize-storage's FTS trash.

`_fts_teardown_trash_step` is only reachable from the foreground `sessions optimize-storage`
command, so a process killed between the demote step and teardown left GBs of dead
`fts_v22_trash_*` pages that nothing drained and a layout stamp that never landed. These tests
cover the maintenance entry point that now drains a bounded slice per tick and settles the
layout (without VACUUM) once the trash is gone.
"""

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_STORAGE_VERSION


@pytest.fixture(autouse=True)
def _no_fts_rebuild_throttle(monkeypatch):
    """Private tmp-path store, no concurrent process: the duty-cycle sleep protects nobody."""
    monkeypatch.setattr(SessionDB, "_FTS_REBUILD_MIN_PAUSE", 0.0)
    monkeypatch.setattr(SessionDB, "_FTS_REBUILD_DUTY_FACTOR", 0.0)


def _store_with_trash(tmp_path, rows=1200):
    """The interrupted-optimize shape: demoted v22 shadow table present, layout stamp absent."""
    db = SessionDB(db_path=tmp_path / "trash.db")
    conn = db._conn
    conn.execute(
        "CREATE TABLE fts_v22_trash_messages_fts_data "
        "(docid INTEGER PRIMARY KEY, block BLOB)"
    )
    conn.executemany(
        "INSERT INTO fts_v22_trash_messages_fts_data (docid, block) VALUES (?, ?)",
        [(i, b"x" * 16) for i in range(1, rows + 1)],
    )
    conn.execute("DELETE FROM state_meta WHERE key = 'fts_storage_version'")
    conn.commit()
    return db


def test_drain_settles_after_last_chunk(tmp_path):
    """A maintenance tick drains the trash to the end and settles the layout stamp."""
    db = _store_with_trash(tmp_path)
    try:
        assert db._has_fts_trash(db._conn) is True
        assert db.get_meta("fts_storage_version") is None

        result = db.drain_fts_trash_opportunistically(max_steps=100, time_budget=30.0)

        assert result["remaining"] is False
        assert result["settled"] is True
        assert result["reason"] is None
        assert result["drained"] >= 3
        assert db._has_fts_trash(db._conn) is False
        assert db.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        assert db.fts_optimize_available() is False
    finally:
        db.close()


def test_slices_resume_across_ticks(tmp_path):
    """One tick takes a bounded slice; the high-water marker lets the next tick resume
    instead of restarting, and the layout settles only once the trash is gone."""
    db = _store_with_trash(tmp_path)
    try:
        first = db.drain_fts_trash_opportunistically(max_steps=1, time_budget=30.0)
        assert first == {"drained": 1, "remaining": True, "settled": False, "reason": None}
        assert db.get_meta("fts_teardown_fts_v22_trash_messages_fts_data_progress") == "500"
        assert db.get_meta("fts_storage_version") is None

        second = db.drain_fts_trash_opportunistically(max_steps=1, time_budget=30.0)
        assert second["drained"] == 1 and second["remaining"] is True
        assert db.get_meta("fts_teardown_fts_v22_trash_messages_fts_data_progress") == "1000"

        result = second
        while result["remaining"]:
            result = db.drain_fts_trash_opportunistically(max_steps=1, time_budget=30.0)
        assert result["settled"] is True
        assert db._has_fts_trash(db._conn) is False
        assert db.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
    finally:
        db.close()


def test_noop_without_trash(tmp_path):
    """A healthy store takes one schema read and no write."""
    db = SessionDB(db_path=tmp_path / "clean.db")
    try:
        before = db.get_meta("fts_storage_version")
        result = db.drain_fts_trash_opportunistically()
        assert result == {"drained": 0, "remaining": False, "settled": False, "reason": None}
        assert db.get_meta("fts_storage_version") == before
    finally:
        db.close()
