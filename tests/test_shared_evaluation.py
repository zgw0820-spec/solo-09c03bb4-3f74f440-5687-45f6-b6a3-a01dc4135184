"""Shared evaluation basis: reuse, idempotent replay, version changes, legacy rows."""
from __future__ import annotations

import threading

import pytest
from sqlalchemy import func, select

from app import service
from app.models import (
    CheckCandidate,
    CheckEvaluation,
    CheckRun,
    CheckVerdict,
    SessionLocal,
)
from app.schemas import CheckRequest, PromoteRequest

from .conftest import archive, make_incident, observe

WATERMARK = "0/3000100"


def _seed(session, incident_id: int) -> None:
    observe(
        session,
        incident_id,
        [
            ("r1", 1, WATERMARK),
            ("r2", 1, "0/2000050"),
            ("r3", 1, "0/1000080"),
            ("r4", 2, "0/3000080"),
        ],
        version=1,
        request_id="shared-obs",
    )
    archive(
        session,
        incident_id,
        1,
        "000000010000000000000002",
        "000000010000000000000003",
        version=2,
    )


def _counts(session) -> tuple[int, int, int]:
    evaluations = session.scalar(select(func.count()).select_from(CheckEvaluation))
    verdicts = session.scalar(select(func.count()).select_from(CheckVerdict))
    checks = session.scalar(select(func.count()).select_from(CheckRun))
    return evaluations, verdicts, checks


def test_check_then_promote_reuses_one_shared_conclusion(session):
    iid = make_incident(session)
    _seed(session, iid)

    check = service.run_check(session, iid, CheckRequest(expected_version=3))
    promote = service.promote(
        session, iid, PromoteRequest(expected_version=3, request_id="promo-shared")
    )
    assert promote["promoted"] is True
    assert promote["check"]["check_id"] != check["check_id"]
    assert promote["check"]["outcome"] == "promoted"
    assert check["outcome"] == "evaluated"
    assert promote["check"]["candidates"] == check["candidates"]

    evaluations, verdicts, checks = _counts(session)
    assert (evaluations, verdicts, checks) == (1, 4, 2)


def test_repeated_check_gets_own_check_id_but_no_second_conclusion(session):
    iid = make_incident(session)
    _seed(session, iid)

    first = service.run_check(session, iid, CheckRequest(expected_version=3))
    second = service.run_check(session, iid, CheckRequest(expected_version=3))
    assert first["check_id"] != second["check_id"]
    assert first["candidates"] == second["candidates"]
    assert _counts(session) == (1, 4, 2)

    evaluations, _, _ = _counts(session)
    assert evaluations == 1


def test_restarted_process_reuses_persisted_conclusion(session):
    iid = make_incident(session)
    _seed(session, iid)
    first = service.run_check(session, iid, CheckRequest(expected_version=3))
    session.commit()

    # Fresh session/engine checkout = a restarted worker reading only the DB.
    other = SessionLocal()
    try:
        again = service.run_check(other, iid, CheckRequest(expected_version=3))
        other.commit()
    finally:
        other.close()

    assert again["candidates"] == first["candidates"]
    assert again["check_id"] != first["check_id"]
    refreshed = SessionLocal()
    try:
        assert _counts(refreshed) == (1, 4, 2)
    finally:
        refreshed.close()


def test_same_numeric_lsn_text_variant_reuses_basis(session):
    iid = make_incident(session)
    _seed(session, iid)
    first = service.run_check(session, iid, CheckRequest(expected_version=3))
    # Same numeric watermark, different spelling; explicit timeline override
    # matching the incident timeline is the same effective basis too.
    second = service.run_check(
        session,
        iid,
        CheckRequest(expected_version=3, primary_timeline=1, commit_lsn="0/03000100"),
    )
    assert second["candidates"] == first["candidates"]
    assert second["commit_lsn"] == first["commit_lsn"] == 0x03000100
    assert _counts(session) == (1, 4, 2)


def test_version_or_watermark_change_uses_new_basis(session):
    iid = make_incident(session)
    _seed(session, iid)
    service.run_check(session, iid, CheckRequest(expected_version=3))

    # New observation bumps the incident version: inputs may have changed.
    observe(session, iid, [("r2", 1, "0/3000100")], version=3, request_id="shared-obs-2")
    service.run_check(session, iid, CheckRequest(expected_version=4))
    assert _counts(session) == (2, 8, 2)

    # A different effective watermark is a new basis even at the same version.
    service.run_check(session, iid, CheckRequest(expected_version=4, commit_lsn="0/2000050"))
    assert _counts(session)[0] == 3


def test_completed_request_replay_returns_first_check_id(session):
    iid = make_incident(session)
    _seed(session, iid)
    req = CheckRequest(expected_version=3, client_request_id="check-once")
    first = service.run_check(session, iid, req)
    replay = service.run_check(session, iid, req)
    assert replay["idempotent_replay"] is True
    assert replay["check_id"] == first["check_id"]
    assert replay["candidates"] == first["candidates"]
    assert _counts(session) == (1, 4, 1)


def test_stale_new_request_still_rejected_but_old_replay_served(session):
    iid = make_incident(session)
    _seed(session, iid)
    first = service.run_check(
        session, iid, CheckRequest(expected_version=3, client_request_id="check-stale")
    )
    service.promote(session, iid, PromoteRequest(expected_version=3, request_id="promo-stale"))

    with pytest.raises(service.Conflict) as exc:
        service.run_check(
            session, iid, CheckRequest(expected_version=3, client_request_id="check-new-stale")
        )
    assert exc.value.code == "stale_version"

    replay = service.run_check(
        session, iid, CheckRequest(expected_version=3, client_request_id="check-stale")
    )
    assert replay["check_id"] == first["check_id"]
    assert replay["idempotent_replay"] is True


def test_legacy_check_runs_without_evaluation_remain_readable(session):
    """Rows written by an older build keep their own per-check verdicts."""
    iid = make_incident(session)
    node = service.incident_detail(session, iid)["nodes"][1]
    legacy = CheckRun(
        incident_id=iid,
        incident_version=1,
        primary_timeline=1,
        commit_lsn=0x03000100,
        commit_lsn_text=WATERMARK,
        outcome="evaluated",
        request_id=None,
        detail=None,
        evaluation_id=None,
    )
    session.add(legacy)
    session.flush()
    session.add(
        CheckCandidate(
            check_run_id=legacy.id,
            node_id=node["id"],
            timeline=1,
            replay_lsn=0x03000100,
            replay_lsn_text=WATERMARK,
            eligible=True,
            reasons_json="[]",
            required_segments_json="[]",
            missing_segment=None,
        )
    )
    session.flush()

    detail = service.incident_detail(session, iid)
    assert detail["last_check"]["check_id"] == legacy.id
    assert detail["last_check"]["candidates"][0]["eligible"] is True
    assert detail["last_check"]["candidates"][0]["replay_lsn_text"] == WATERMARK


def test_failed_check_write_rolls_back_evaluation_and_check(session):
    iid = make_incident(session)
    _seed(session, iid)
    session.commit()

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated write failure")

    worker = SessionLocal()
    original = service._persist_check
    service._persist_check = boom
    try:
        with pytest.raises(RuntimeError):
            service.run_check(worker, iid, CheckRequest(expected_version=3))
        worker.rollback()
    finally:
        service._persist_check = original
        worker.close()

    fresh = SessionLocal()
    try:
        assert _counts(fresh) == (0, 0, 0)
    finally:
        fresh.close()


def test_concurrent_same_basis_requests_share_conclusion(session):
    """Two workers submit at the same basis; only one conclusion is stored."""
    iid = make_incident(session)
    _seed(session, iid)
    session.commit()

    barrier = threading.Barrier(2)
    results: list[dict] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(request_id: str) -> None:
        db = SessionLocal()
        try:
            barrier.wait(timeout=10)
            result = service.run_check(
                db,
                iid,
                CheckRequest(expected_version=3, client_request_id=request_id),
            )
            db.commit()
            with lock:
                results.append(result)
        except BaseException as exc:  # noqa: BLE001
            db.rollback()
            with lock:
                errors.append(exc)
        finally:
            db.close()

    threads = [
        threading.Thread(target=worker, args=("concurrent-a",)),
        threading.Thread(target=worker, args=("concurrent-b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, errors
    assert len(results) == 2
    assert results[0]["check_id"] != results[1]["check_id"]
    assert results[0]["candidates"] == results[1]["candidates"]

    fresh = SessionLocal()
    try:
        assert _counts(fresh) == (1, 4, 2)
    finally:
        fresh.close()
