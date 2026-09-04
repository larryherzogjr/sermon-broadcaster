import sqlite3
from datetime import datetime, timezone

import config
from cleanup_jobs import cleanup_expired_jobs
from pipeline import db


def test_cleanup_deletes_old_jobs_in_every_status_and_their_files(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    review_dir = state_dir / "review_jobs"
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "output"
    upload_dir = tmp_path / "uploads"
    for path in (state_dir, review_dir, work_dir, output_dir, upload_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "STATE_DIR", str(state_dir))
    monkeypatch.setattr(config, "DB_PATH", str(state_dir / "jobs.db"))
    monkeypatch.setattr(config, "REVIEW_DIR", str(review_dir))
    monkeypatch.setattr(config, "WORK_DIR", str(work_dir))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(config, "UPLOAD_DIR", str(upload_dir))
    db.init_schema()

    active_id = "20260801_010101_000001"
    complete_id = "20260802_010101_000002"
    recent_id = "20260825_010101_000003"
    active_upload = f"{active_id}_service.wav"
    complete_output = f"sermon_{complete_id}.mp3"

    db.create_job(active_id, active_upload, "upload", "27:18", False, False, True)
    db.create_job(complete_id, "https://youtu.be/abcdefghijk", "youtube", "27:18", False, False, True)
    db.set_result(
        complete_id,
        [{"variant": "sermon", "filename": complete_output, "note": ""}],
        {},
    )
    db.create_job(recent_id, "https://youtu.be/abcdefghijk", "youtube", "27:18", False, False, True)
    db.set_error(recent_id, "fixture")

    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("UPDATE jobs SET created_at=? WHERE id=?", ("2026-08-01T01:01:01+00:00", active_id))
        conn.execute("UPDATE jobs SET created_at=? WHERE id=?", ("2026-08-02T01:01:01+00:00", complete_id))
        conn.execute("UPDATE jobs SET created_at=? WHERE id=?", ("2026-08-25T01:01:01+00:00", recent_id))

    active_review = review_dir / active_id
    active_work = work_dir / f"review_{active_id}_fixture"
    active_review.mkdir()
    active_work.mkdir()
    (active_review / "raw_audio.wav").write_bytes(b"fixture")
    (upload_dir / active_upload).write_bytes(b"fixture")
    (output_dir / complete_output).write_bytes(b"fixture")

    deleted = cleanup_expired_jobs(
        days=14,
        now=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
    )

    assert deleted == [active_id, complete_id]
    assert db.get_job(active_id) is None
    assert db.get_job(complete_id) is None
    assert db.get_job(recent_id)["status"] == "failed"
    assert not active_review.exists()
    assert not active_work.exists()
    assert not (upload_dir / active_upload).exists()
    assert not (output_dir / complete_output).exists()


def test_cleanup_dry_run_does_not_delete(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(config, "DB_PATH", str(state_dir / "jobs.db"))
    db.init_schema()
    job_id = "20260801_010101_000004"
    db.create_job(job_id, "fixture", "youtube", "27:18", False, False, True)
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("UPDATE jobs SET created_at=? WHERE id=?", ("2026-08-01T01:01:01+00:00", job_id))

    deleted = cleanup_expired_jobs(
        days=14,
        now=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
        dry_run=True,
    )

    assert deleted == []
    assert db.get_job(job_id) is not None
