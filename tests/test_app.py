import importlib
import io
import os
import sys

import pytest

import config


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    root = tmp_path_factory.mktemp("app-state")
    config.STATE_DIR = str(root / "state")
    config.DB_PATH = str(root / "state" / "jobs.db")
    config.REVIEW_DIR = str(root / "state" / "review_jobs")
    config.WORK_DIR = str(root / "work")
    config.OUTPUT_DIR = str(root / "output")
    for path in (config.STATE_DIR, config.REVIEW_DIR, config.WORK_DIR, config.OUTPUT_DIR):
        os.makedirs(path, exist_ok=True)

    import pipeline
    for name in ("app", "pipeline.feedback", "pipeline.db"):
        sys.modules.pop(name, None)
    for attribute in ("feedback", "db"):
        if hasattr(pipeline, attribute):
            delattr(pipeline, attribute)

    module = importlib.import_module("app")
    module.UPLOAD_DIR = str(root / "uploads")
    os.makedirs(module.UPLOAD_DIR, exist_ok=True)
    module.app.config.update(TESTING=True)
    return module


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


def test_basic_pages_and_health_render(client):
    index = client.get("/")
    assert index.status_code == 200
    index_html = index.get_data(as_text=True)
    assert "localStorage" not in index_html
    assert "new URLSearchParams(location.search).get('job')" in index_html
    assert 'id="newSessionLink"' in index_html
    assert 'id="manualSelectionCheckbox"' in index_html
    assert 'id="manualTeaserCheckbox"' not in index_html
    assert 'id="manualTeaserEditorCheckbox"' in index_html
    assert 'id="applyCutBtn"' in index_html
    assert "I’ll select it manually" in index_html
    assert 'id="dynamicCheckbox" checked' in index_html
    assert 'id="analyzeBtn" disabled' in index_html

    history = client.get("/history")
    assert history.status_code == 200
    history_html = history.get_data(as_text=True)
    assert "View Progress" in history_html
    assert "View Finished Job" in history_html
    assert 'data-delete-job=' in history_html
    assert "This cannot be undone" in history_html
    assert 'const APP_TIME_ZONE = "America/Chicago"' in history_html
    assert "timeZone: APP_TIME_ZONE" in history_html
    assert client.get("/api/history").status_code == 200
    health = client.get("/api/health")
    assert health.status_code in {200, 503}
    assert health.get_json()["status"] in {"ok", "degraded"}


def test_youtube_url_validation_rejects_lookalike_hosts(app_module):
    assert app_module._is_youtube_url("https://www.youtube.com/watch?v=abcdefghijk")
    assert app_module._is_youtube_url("https://youtu.be/abcdefghijk")
    assert not app_module._is_youtube_url("https://youtube.com.evil.example/video")
    assert not app_module._is_youtube_url("https://example.com/?next=youtube.com")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), ("false", False), (1, True), (0, False)],
)
def test_boolean_normalization(app_module, value, expected):
    assert app_module._as_bool(value) is expected


def test_fully_manual_selection_does_not_require_anthropic(app_module, monkeypatch):
    monkeypatch.setattr(app_module.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(app_module.config, "TRANSCRIBE_BACKEND", "faster-whisper")
    monkeypatch.setattr(app_module.config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(
        app_module, "sermon_target_seconds", lambda *_args: (1638.0, {})
    )

    app_module._validate_processing_requirements(
        "27:18", True, False, False, manual_selection=True, manual_teaser=True
    )

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        app_module._validate_processing_requirements(
            "27:18", True, False, False, manual_selection=False
        )


def test_manual_editor_defers_transcription_requirements_until_render(app_module, monkeypatch):
    monkeypatch.setattr(app_module.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(app_module.config, "TRANSCRIBE_BACKEND", "local")
    monkeypatch.setattr(app_module.config, "WHISPER_LOCAL_URL", "")
    monkeypatch.setattr(app_module.config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(
        app_module, "sermon_target_seconds", lambda *_args: (1638.0, {})
    )

    app_module._validate_processing_requirements(
        "27:18", True, False, False, manual_selection=True, manual_teaser=True
    )

    app_module._validate_processing_requirements(
        "27:18", True, False, False, manual_selection=True, manual_teaser=False
    )


def test_invalid_requests_fail_before_processing(client):
    assert client.post("/api/analyze", json={}).status_code == 400
    assert client.post("/api/analyze", json=[]).status_code == 400
    assert client.post(
        "/api/analyze", json={"url": "https://youtube.com.evil.example/video"}
    ).status_code == 400
    assert client.post(
        "/api/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=abcdefghijk",
            "target_duration": "bad",
        },
    ).status_code == 400
    assert client.post(
        "/api/analyze",
        data={"file": (io.BytesIO(b"not media"), "payload.exe")},
        content_type="multipart/form-data",
    ).status_code == 400


def test_request_requires_a_selected_workflow_option(client):
    response = client.post(
        "/api/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=abcdefghijk",
            "include_bumpers_dynamic": False,
            "include_bumpers_stock": False,
            "sermon_only": False,
            "manual_selection": False,
        },
    )

    assert response.status_code == 400
    assert "Select at least one" in response.get_json()["error"]


def test_missing_media_error_takes_priority_over_option_error(client):
    response = client.post(
        "/api/analyze",
        json={
            "include_bumpers_dynamic": False,
            "include_bumpers_stock": False,
            "sermon_only": False,
            "manual_selection": False,
        },
    )

    assert response.status_code == 400
    assert "YouTube URL or upload a file" in response.get_json()["error"]


def test_valid_request_is_persisted_and_queued(client, app_module, monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.args, self.daemon))

    monkeypatch.setattr(app_module, "_validate_processing_requirements", lambda *args: None)
    monkeypatch.setattr(app_module.threading, "Thread", FakeThread)

    response = client.post(
        "/api/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=abcdefghijk",
            "target_duration": "27:18",
            "include_bumpers_dynamic": False,
            "include_bumpers_stock": False,
            "sermon_only": True,
        },
    )

    assert response.status_code == 200
    job_id = response.get_json()["job_id"]
    job = client.get(f"/api/status/{job_id}").get_json()
    assert job["status"] == "queued"
    assert job["source_type"] == "youtube"
    assert len(started) == 1
    assert any(item["job_id"] == job_id for item in client.get("/api/history").get_json())


def test_manual_selection_mode_is_forwarded_to_analysis(client, app_module, monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            self.args = args

        def start(self):
            started.append(self.args)

    monkeypatch.setattr(app_module, "_validate_processing_requirements", lambda *args: None)
    monkeypatch.setattr(app_module.threading, "Thread", FakeThread)

    response = client.post(
        "/api/analyze",
        json={
            "url": "https://www.youtube.com/watch?v=abcdefghijk",
            "target_duration": "27:18",
            "include_bumpers_dynamic": False,
            "include_bumpers_stock": False,
            "sermon_only": False,
            "manual_selection": True,
        },
    )

    assert response.status_code == 200
    assert started[0][0].manual_selection is True
    assert started[0][0].manual_teaser is False
    assert started[0][0].include_bumpers_dynamic is True


def test_render_claim_is_atomic(app_module):
    job_id = "20260714_120000_000001"
    app_module.db.create_job(job_id, "fixture.wav", "upload", "27:18", False, False, True)
    review = {"sermon_start": 0.0, "sermon_end": 100.0}
    app_module.db.set_analysis_ready(job_id, {"review": review})

    assert app_module.db.claim_render(job_id, review) is True
    assert app_module.db.claim_render(job_id, review) is False
    assert app_module.db.get_job(job_id)["status"] == "rendering"


def test_delete_job_removes_database_records_and_files(client, app_module):
    job_id = "20260903_130000_000001"
    upload_name = f"{job_id}_service.wav"
    output_name = f"sermon_{job_id}_dynamic.mp3"
    app_module.db.create_job(
        job_id, upload_name, "upload", "29:30", True, False, False
    )
    app_module.db.set_result(
        job_id,
        [{"variant": "dynamic", "filename": output_name, "note": ""}],
        {"review": {"title": "Delete me"}},
    )
    session_id = app_module.db.create_feedback_session(job_id)
    app_module.db.append_feedback_message(session_id, "user", "test")

    review_dir = os.path.join(app_module.config.REVIEW_DIR, job_id)
    work_dir = os.path.join(app_module.config.WORK_DIR, f"review_{job_id}_123456")
    upload_path = os.path.join(app_module.UPLOAD_DIR, upload_name)
    output_path = os.path.join(app_module.config.OUTPUT_DIR, output_name)
    os.makedirs(review_dir)
    os.makedirs(work_dir)
    for path in (upload_path, output_path, os.path.join(review_dir, "raw_audio.wav")):
        with open(path, "wb") as handle:
            handle.write(b"fixture")

    response = client.delete(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert response.get_json()["deleted"] is True
    assert app_module.db.get_job(job_id) is None
    assert app_module.db.get_feedback_session(session_id) is None
    assert not os.path.exists(review_dir)
    assert not os.path.exists(work_dir)
    assert not os.path.exists(upload_path)
    assert not os.path.exists(output_path)


def test_delete_active_job_marks_worker_cancelled(client, app_module):
    job_id = "20260903_130000_000002"
    app_module.db.create_job(
        job_id, "https://youtu.be/abcdefghijk", "youtube", "29:30", True, False, False
    )

    response = client.delete(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert app_module._job_is_cancelled(job_id) is True
    assert app_module.db.get_job(job_id) is None
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404


def test_undo_cut_restores_review_metadata(client, app_module, monkeypatch):
    metadata = {"review": {"undo_available": True}}
    restored = {"audio_duration": 100, "undo_available": False}
    monkeypatch.setattr(app_module.db, "get_job", lambda _: {"review": metadata["review"], "status": "awaiting_review"})
    monkeypatch.setattr(app_module.db, "get_metadata", lambda _: metadata)
    monkeypatch.setattr(app_module, "undo_working_cut", lambda *_: {
        "review": restored, "waveform": {"peaks": [0.5]},
        "metadata": {"transcript_summary": {"duration": 100}},
    })
    updates = []
    monkeypatch.setattr(app_module.db, "update_metadata", lambda *args: updates.append(args))
    response = client.post("/api/jobs/1/undo-cut")
    assert response.status_code == 200
    assert response.get_json()["review"] == restored
    assert updates[0][1]["review"] == restored


def test_undo_cut_rejects_rendering_jobs(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.db, "get_job", lambda _: {"review": {"undo_available": True}, "status": "rendering"})
    monkeypatch.setattr(app_module.db, "get_metadata", lambda _: {"review": {"undo_available": True}})
    assert client.post("/api/jobs/1/undo-cut").status_code == 409


def test_finished_audio_preview_is_inline(client, app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.config, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "preview.mp3").write_bytes(b"test audio")
    download = client.get("/api/download/preview.mp3")
    preview = client.get("/api/download/preview.mp3?preview=1")
    assert download.headers["Content-Disposition"].startswith("attachment")
    assert preview.headers["Content-Disposition"].startswith("inline")
