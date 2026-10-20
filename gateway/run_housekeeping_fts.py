"""Gateway housekeeping chores for the state.db FTS index: retrying a deferred rebuild and
draining the demoted v22 shadow tables an interrupted ``optimize-storage`` left behind.

An interrupted optimize is the case neither side of the foreground command covers: the demote
step renames the v22 shadow family to ``fts_v22_trash_*`` and only ``optimize_fts_storage``
drains it, so a gateway that holds state.db for days (and every process that opened the store
afterwards) kept GBs of dead pages and never got the layout stamp. Both chores run on the
existing housekeeping tick against the shared stores this process already holds; each is
rate-limited inside SessionDB so an idle tick costs one attribute read / schema read.
"""

from __future__ import annotations

import logging

# Log-record parity with the origin module (gateway.run).
logger = logging.getLogger("gateway.run")


def _housekeeping_deferred_fts_retry() -> None:
    """A SessionDB opened while another process held the rebuild lock fails closed onto the LIKE fallback
    and the gateway stays up for days. Non-blocking, rate-limited inside SessionDB; no-op when not stale."""
    # Retry here, on the existing tick, against the shared instances this process already holds:
    # non-blocking admission, no new thread, rate-limited inside SessionDB. No-op when nothing is stale (one
    # attribute read per instance). See #100108.
    from hermes_state_registry import borrow_live_shared_session_dbs
    with borrow_live_shared_session_dbs() as _session_dbs:
        for _sdb in _session_dbs:
            _retry = getattr(_sdb, "retry_deferred_fts_recovery", None)
            if callable(_retry) and _retry():
                logger.info(
                    "Deferred state.db FTS rebuild completed in-process for %s; full-text search restored.",
                    getattr(_sdb, "db_path", "state.db"))


def _housekeeping_fts_trash_drain() -> None:
    """An interrupted ``optimize-storage`` leaves its demoted v22 FTS shadow tables in state.db and only
    the foreground command drains them; a gateway holds state.db for days, so drain a bounded slice on the
    existing tick. The step is resumable across processes (high-water marker claimed in BEGIN IMMEDIATE),
    so concurrent holders only split the work. No-op when there is no trash (one schema read per instance).
    See #141131."""
    from hermes_state_registry import borrow_live_shared_session_dbs
    with borrow_live_shared_session_dbs() as _session_dbs:
        for _sdb in _session_dbs:
            _drain = getattr(_sdb, "drain_fts_trash_opportunistically", None)
            if not callable(_drain):
                continue
            _result = _drain()
            if _result.get("drained") or _result.get("settled"):
                logger.info(
                    "FTS trash drain: %d chunk(s) from %s (remaining=%s, settled=%s).",
                    _result.get("drained", 0), getattr(_sdb, "db_path", "state.db"),
                    _result.get("remaining"), _result.get("settled"))
