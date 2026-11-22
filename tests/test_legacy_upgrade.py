"""Legacy database upgrade: an old schema must keep working after ``init_db``."""
from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from app import service
from app.models import (
    Base,
    _upgrade_legacy_schema,
    init_db,
    sessionmaker,
)
from app.schemas import (
    CheckRequest,
    IncidentCreate,
    NodeSpec,
    ObservationBatch,
    ObservationSpec,
    PromoteRequest,
)


_LEGACY_CHECK_RUNS_DDL = """
CREATE TABLE check_runs (
    id INTEGER NOT NULL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    incident_version INTEGER NOT NULL,
    primary_timeline INTEGER NOT NULL,
    commit_lsn BIGINT NOT NULL,
    commit_lsn_text VARCHAR(32) NOT NULL,
    outcome VARCHAR(40) NOT NULL,
    request_id VARCHAR(100),
    detail TEXT,
    created_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
)
"""

_LEGACY_CHECK_CANDIDATES_DDL = """
CREATE TABLE check_candidates (
    id INTEGER NOT NULL PRIMARY KEY,
    check_run_id INTEGER NOT NULL REFERENCES check_runs(id) ON DELETE CASCADE,
    node_id INTEGER NOT NULL REFERENCES nodes(id),
    timeline INTEGER NOT NULL,
    replay_lsn BIGINT NOT NULL,
    replay_lsn_text VARCHAR(32) NOT NULL,
    eligible BOOLEAN NOT NULL,
    reasons_json TEXT NOT NULL,
    required_segments_json TEXT NOT NULL,
    missing_segment BIGINT
)
"""


def _old_schema_engine(tmp_path):
    """Create the pre-shared-evaluation schema: check_runs without evaluation_id."""
    eng = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}", future=True)
    with eng.begin() as conn:
        conn.execute(text(_LEGACY_CHECK_RUNS_DDL))
        conn.execute(text(_LEGACY_CHECK_CANDIDATES_DDL))
    Base.metadata.create_all(bind=eng)  # creates everything still missing
    return eng


def test_old_database_gets_column_and_old_rows_stay_readable(tmp_path):
    eng = _old_schema_engine(tmp_path)
    assert "evaluation_id" not in {
        c["name"] for c in inspect(eng).get_columns("check_runs")
    }

    # Seed a complete old-style incident + check run through the ORM.
    Session = sessionmaker(bind=eng, expire_on_commit=False, future=True)
    db = Session()
    try:
        incident = service.create_incident(
            db,
            IncidentCreate(
                title="旧库事件",
                primary_timeline=1,
                commit_lsn="0/3000100",
                nodes=[
                    NodeSpec(node_key="primary", name="主库", role="primary"),
                    NodeSpec(node_key="r1", name="副本1", role="candidate"),
                ],
            ),
        )
        db.flush()
        node = next(n for n in incident.nodes if n.node_key == "r1")
        iid = incident.id
        # Old code wrote these rows; insert them without the new column.
        db.execute(
            text(
                "INSERT INTO check_runs (incident_id, incident_version, "
                "primary_timeline, commit_lsn, commit_lsn_text, outcome, "
                "request_id, detail) VALUES (:iid, 1, 1, :lsn, '0/3000100', "
                "'evaluated', 'old-run', NULL)"
            ),
            {"iid": iid, "lsn": 0x03000100},
        )
        old_check = db.execute(
            text("SELECT id FROM check_runs WHERE request_id = 'old-run'")
        ).scalar_one()
        db.execute(
            text(
                "INSERT INTO check_candidates (check_run_id, node_id, timeline, "
                "replay_lsn, replay_lsn_text, eligible, reasons_json, "
                "required_segments_json, missing_segment) VALUES "
                "(:cid, :nid, 1, :lsn, '0/3000100', 1, '[]', '[]', NULL)"
            ),
            {"cid": old_check, "nid": node.id, "lsn": 0x03000100},
        )
        db.commit()

        # Upgrade exactly as application startup does.
        init_db(eng)
        assert "evaluation_id" in {
            c["name"] for c in inspect(eng).get_columns("check_runs")
        }

        detail = service.incident_detail(db, iid)
        assert detail["last_check"]["check_id"] == old_check
        assert detail["last_check"]["candidates"][0]["replay_lsn_text"] == "0/3000100"

        # A new request at the same basis computes and stores one shared
        # conclusion while leaving the legacy receipt untouched.
        service.submit_observations(
            db,
            iid,
            ObservationBatch(
                expected_version=1,
                client_request_id="legacy-obs",
                observations=[
                    ObservationSpec(node_key="r1", timeline=1, replay_lsn="0/3000100")
                ],
            ),
        )
        result = service.run_check(db, iid, CheckRequest(expected_version=2))
        assert result["check_id"] != old_check
        assert result["candidates"][0]["eligible"] is True
        db.commit()

        promote = service.promote(
            db, iid, PromoteRequest(expected_version=2, request_id="legacy-promote")
        )
        assert promote["promoted"] is True
        db.commit()

        # Old receipt still readable, and the check→promote pair shares one
        # new conclusion.
        assert service.incident_detail(db, iid)["last_check"]["outcome"] == "promoted"
        evaluations = db.execute(
            text("SELECT COUNT(*) FROM check_evaluations")
        ).scalar_one()
        assert evaluations == 1
    finally:
        db.close()
        eng.dispose()


def test_upgrade_is_idempotent(tmp_path):
    eng = _old_schema_engine(tmp_path)
    try:
        _upgrade_legacy_schema(eng)
        _upgrade_legacy_schema(eng)  # second startup must be a no-op
        assert "evaluation_id" in {
            c["name"] for c in inspect(eng).get_columns("check_runs")
        }
    finally:
        eng.dispose()
