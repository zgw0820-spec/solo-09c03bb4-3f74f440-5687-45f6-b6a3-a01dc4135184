"""HTTP-level checks for the shared evaluation: UI flow, concurrency, replays."""
from __future__ import annotations

import threading

from sqlalchemy import func, select

from app.models import CheckEvaluation, CheckRun, CheckVerdict, SessionLocal

from .test_api import _create


def _seed(client, iid: int) -> None:
    obs = client.post(
        f"/api/incidents/{iid}/observations",
        json={
            "expected_version": 1,
            "client_request_id": "shared-api-obs",
            "observations": [
                {"node_key": "r1", "timeline": 1, "replay_lsn": "0/3000100"},
                {"node_key": "r2", "timeline": 1, "replay_lsn": "0/2000050"},
                {"node_key": "r3", "timeline": 1, "replay_lsn": "0/1000080"},
                {"node_key": "r4", "timeline": 2, "replay_lsn": "0/3000080"},
            ],
        },
    )
    assert obs.status_code == 200, obs.text


def _counts() -> tuple[int, int, int]:
    db = SessionLocal()
    try:
        return (
            db.scalar(select(func.count()).select_from(CheckEvaluation)),
            db.scalar(select(func.count()).select_from(CheckVerdict)),
            db.scalar(select(func.count()).select_from(CheckRun)),
        )
    finally:
        db.close()


def test_api_check_then_promote_share_one_conclusion(client):
    iid = _create(client)
    _seed(client, iid)
    check = client.post(
        f"/api/incidents/{iid}/checks",
        json={"expected_version": 2, "client_request_id": "api-shared-check"},
    )
    assert check.status_code == 200, check.text
    promote = client.post(
        f"/api/incidents/{iid}/promote",
        json={"expected_version": 2, "request_id": "api-shared-promote"},
    )
    assert promote.status_code == 200, promote.text
    assert promote.json()["promoted"] is True
    assert promote.json()["check"]["check_id"] != check.json()["check_id"]
    assert promote.json()["check"]["candidates"] == check.json()["candidates"]
    assert _counts() == (1, 4, 2)


def test_api_check_retry_after_completion_returns_first_receipt(client):
    iid = _create(client)
    _seed(client, iid)
    payload = {"expected_version": 2, "client_request_id": "api-idem-check"}
    first = client.post(f"/api/incidents/{iid}/checks", json=payload)
    client.post(
        f"/api/incidents/{iid}/promote",
        json={"expected_version": 2, "request_id": "api-idem-promote"},
    )
    # Page is now stale (v3) but a completed request retried by its original
    # number still returns the first conclusion and check id.
    replay = client.post(f"/api/incidents/{iid}/checks", json=payload)
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["check_id"] == first.json()["check_id"]
    assert replay.json()["candidates"] == first.json()["candidates"]

    # A new request from the stale page is rejected as usual.
    stale = client.post(
        f"/api/incidents/{iid}/checks",
        json={"expected_version": 2, "client_request_id": "api-new-stale"},
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_version"


def test_api_archive_supplement_forms_new_basis(client):
    iid = _create(client)
    _seed(client, iid)
    before = client.post(
        f"/api/incidents/{iid}/checks",
        json={"expected_version": 2, "client_request_id": "api-before-archive"},
    ).json()
    assert before["outcome"] == "evaluated"
    # No archive yet: r2/r3 behind the watermark are rejected.
    assert sum(c["eligible"] for c in before["candidates"]) == 1

    arc = client.post(
        f"/api/incidents/{iid}/archives",
        json={
            "expected_version": 2,
            "client_request_id": "api-late-archive",
            "archives": [
                {
                    "timeline": 1,
                    "start_segment": "000000010000000000000002",
                    "end_segment": "000000010000000000000003",
                }
            ],
        },
    )
    assert arc.status_code == 200, arc.text

    after = client.post(
        f"/api/incidents/{iid}/checks",
        json={"expected_version": 3, "client_request_id": "api-after-archive"},
    ).json()
    assert sum(c["eligible"] for c in after["candidates"]) == 2
    assert _counts()[:2] == (2, 8)  # a separate conclusion for the new basis


def test_api_concurrent_checks_reuse_single_conclusion(client):
    iid = _create(client)
    _seed(client, iid)
    # Make sure tables exist before threads start.
    client.get(f"/api/incidents/{iid}")

    barrier = threading.Barrier(4)
    outcomes: list[dict] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        try:
            with TestClient(app) as worker_client:
                barrier.wait(timeout=15)
                resp = worker_client.post(
                    f"/api/incidents/{iid}/checks",
                    json={
                        "expected_version": 2,
                        "client_request_id": f"api-concurrent-{n}",
                    },
                )
                with lock:
                    outcomes.append({"status": resp.status_code, "body": resp.json()})
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert all(o["status"] == 200 for o in outcomes), outcomes
    bodies = [o["body"] for o in outcomes]
    assert len({b["check_id"] for b in bodies}) == 4  # own check id per request
    assert all(b["candidates"] == bodies[0]["candidates"] for b in bodies)
    assert _counts() == (1, 4, 4)


def test_html_check_then_confirmation_shares_conclusion(client):
    iid = _create(client)
    _seed(client, iid)
    page = client.get(f"/incidents/{iid}").text
    assert "执行接任检查" in page

    check = client.post(
        f"/incidents/{iid}/checks",
        data={"expected_version": "2", "request_id": "html-shared-check"},
        follow_redirects=False,
    )
    assert check.status_code == 303
    promote = client.post(
        f"/incidents/{iid}/promote",
        data={"expected_version": "2", "request_id": "html-shared-promote", "node_key": ""},
        follow_redirects=False,
    )
    assert promote.status_code == 303
    detail = client.get(f"/api/incidents/{iid}").json()
    assert detail["active_primary"]["node_key"] == "r1"
    assert detail["last_check"]["outcome"] == "promoted"
    assert _counts() == (1, 4, 2)
