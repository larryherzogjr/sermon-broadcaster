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
    assert client.get("/").status_code == 200
    assert client.get("/history").status_code == 200
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


def test_render_claim_is_atomic(app_module):
    job_id = "20260714_120000_000001"
    app_module.db.create_job(job_id, "fixture.wav", "upload", "27:18", False, False, True)
    review = {"sermon_start": 0.0, "sermon_end": 100.0}
    app_module.db.set_analysis_ready(job_id, {"review": review})

    assert app_module.db.claim_render(job_id, review) is True
    assert app_module.db.claim_render(job_id, review) is False
    assert app_module.db.get_job(job_id)["status"] == "rendering"
