"""The gateway housekeeping tick drains an interrupted optimize-storage's FTS trash (#141131).

`_fts_teardown_trash_step` is otherwise only reachable from the foreground `sessions
optimize-storage` command, so a gateway that holds state.db for days never clears the
demoted v22 shadow tables a killed optimize left behind.
"""

from contextlib import contextmanager

from gateway import run
from hermes_state import SessionDB


def _store_with_trash(tmp_path, rows=600):
    db = SessionDB(db_path=tmp_path / "state.db")
    db._conn.execute(
        "CREATE TABLE fts_v22_trash_messages_fts_data (docid INTEGER PRIMARY KEY, block BLOB)"
    )
    db._conn.executemany(
        "INSERT INTO fts_v22_trash_messages_fts_data (docid, block) VALUES (?, ?)",
        [(i, b"x" * 16) for i in range(1, rows + 1)],
    )
    db._conn.commit()
    return db


def test_housekeeping_tick_drains_interrupted_optimize_trash(monkeypatch, tmp_path):
    db = _store_with_trash(tmp_path)
    try:
        assert db._has_fts_trash(db._conn) is True

        @contextmanager
        def _borrow():
            yield [db]

        monkeypatch.setattr("hermes_state_registry.borrow_live_shared_session_dbs", _borrow)
        run._housekeeping_fts_trash_drain()

        assert db._has_fts_trash(db._conn) is False
    finally:
        db.close()


def test_housekeeping_tick_is_a_noop_without_trash(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        drained = []

        @contextmanager
        def _borrow():
            yield [db]

        monkeypatch.setattr("hermes_state_registry.borrow_live_shared_session_dbs", _borrow)
        monkeypatch.setattr(
            db, "drain_fts_trash_opportunistically",
            lambda: drained.append(True) or {"drained": 0, "settled": False},
        )
        run._housekeeping_fts_trash_drain()

        assert drained == [True]
        assert db._has_fts_trash(db._conn) is False
    finally:
        db.close()
