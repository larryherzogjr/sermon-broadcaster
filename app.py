"""
Sermon Broadcaster
Flask app for processing church service videos into broadcast-ready audio.
"""
import os
import logging
import threading
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import config
from pipeline import db
from pipeline.orchestrator import run_pipeline

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask App ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 * 1024  # 5 GB upload limit

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Persistence: schema + reconcile orphaned jobs from a prior crash/restart.
# Runs at import time so it executes under systemd too.
db.init_schema()
db.mark_orphans_failed()

ALLOWED_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "webm", "mp3", "wav", "m4a", "flac", "ogg"}


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Job Handle ───────────────────────────────────────────────────────

class Job:
    """Thin handle over a persisted job row. State lives in SQLite."""

    def __init__(self, job_id, target_duration, source, source_type,
                 include_bumpers_dynamic=False, include_bumpers_stock=False,
                 sermon_only=False,
                 youtube_url=None, local_file=None):
        self.job_id = job_id
        self.youtube_url = youtube_url
        self.local_file = local_file
        self.target_duration = target_duration
        self.include_bumpers_dynamic = include_bumpers_dynamic
        self.include_bumpers_stock = include_bumpers_stock
        self.sermon_only = sermon_only

        db.create_job(
            job_id=job_id,
            source=source,
            source_type=source_type,
            target_duration=target_duration,
            include_dynamic=include_bumpers_dynamic,
            include_stock=include_bumpers_stock,
            sermon_only=sermon_only,
        )

    def update_status(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        db.append_message(self.job_id, ts, message)
        logger.info(f"[Job {self.job_id}] {message}")

    def set_result(self, outputs, metadata):
        db.set_result(self.job_id, outputs, metadata)

    def set_error(self, error_str):
        db.set_error(self.job_id, error_str)


def _run_job(job: Job):
    """Run the pipeline in a background thread."""
    try:
        result = run_pipeline(
            youtube_url=job.youtube_url,
            local_file=job.local_file,
            target_duration=job.target_duration,
            include_bumpers_dynamic=job.include_bumpers_dynamic,
            include_bumpers_stock=job.include_bumpers_stock,
            sermon_only=job.sermon_only,
            status_callback=job.update_status,
        )
        metadata = {
            "boundaries": result.get("boundaries"),
            "processing": result.get("processing"),
            "timing": result.get("timing"),
            "teaser": result.get("teaser"),
            "transcript_summary": result.get("transcript"),
            "broadcast_duration": result.get("broadcast_duration"),
            "include_bumpers": result.get("include_bumpers"),
        }
        job.set_result(result.get("outputs", []), metadata)
    except Exception as e:
        job.set_error(str(e))
        logger.exception(f"Job {job.job_id} failed")


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        default_duration=config.DEFAULT_TARGET_DURATION,
        default_broadcast_duration=config.DEFAULT_BROADCAST_DURATION,
    )


@app.route("/api/process", methods=["POST"])
def start_processing():
    """Start a processing job from YouTube URL (JSON) or file upload (multipart)."""

    # Handle file upload (multipart form)
    if request.content_type and "multipart" in request.content_type:
        file = request.files.get("file")
        youtube_url = request.form.get("url", "").strip()
        include_dynamic = request.form.get("include_bumpers_dynamic", "false").lower() == "true"
        include_stock = request.form.get("include_bumpers_stock", "false").lower() == "true"
        sermon_only = request.form.get("sermon_only", "false").lower() == "true"
        any_bumpers = include_dynamic or include_stock
        default_dur = config.DEFAULT_BROADCAST_DURATION if any_bumpers else config.DEFAULT_TARGET_DURATION
        target_duration = request.form.get("target_duration", default_dur).strip()

        if file and file.filename and _allowed_file(file.filename):
            job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = secure_filename(f"{job_id}_{file.filename}")
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)
            logger.info(f"File uploaded: {filepath} ({os.path.getsize(filepath) / 1024 / 1024:.1f} MB)")

            job = Job(job_id, target_duration,
                      source=os.path.basename(filepath),
                      source_type="upload",
                      include_bumpers_dynamic=include_dynamic,
                      include_bumpers_stock=include_stock,
                      sermon_only=sermon_only,
                      local_file=filepath)

            thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
            thread.start()
            return jsonify({"job_id": job_id, "status": "queued"})

        elif youtube_url:
            pass
        else:
            return jsonify({"error": "Please provide a YouTube URL or upload a video/audio file"}), 400

    # Handle JSON request (YouTube URL)
    else:
        data = request.get_json() or {}
        youtube_url = data.get("url", "").strip()
        include_dynamic = data.get("include_bumpers_dynamic", False)
        include_stock = data.get("include_bumpers_stock", False)
        sermon_only = data.get("sermon_only", False)
        if data.get("include_bumpers") and not (include_dynamic or include_stock):
            include_dynamic = True
        any_bumpers = include_dynamic or include_stock
        default_dur = config.DEFAULT_BROADCAST_DURATION if any_bumpers else config.DEFAULT_TARGET_DURATION
        target_duration = data.get("target_duration", default_dur).strip()

    if not youtube_url:
        return jsonify({"error": "Please provide a YouTube URL or upload a file"}), 400

    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        return jsonify({"error": "Please provide a valid YouTube URL"}), 400

    parts = target_duration.split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        return jsonify({"error": "Duration must be in MM:SS or HH:MM:SS format"}), 400

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = Job(job_id, target_duration,
              source=youtube_url,
              source_type="youtube",
              include_bumpers_dynamic=include_dynamic,
              include_bumpers_stock=include_stock,
              sermon_only=sermon_only,
              youtube_url=youtube_url)

    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    j = db.get_job(job_id)
    if not j:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(j)


@app.route("/api/download/<filename>")
def download_file(filename):
    return send_from_directory(
        config.OUTPUT_DIR,
        filename,
        as_attachment=True,
        mimetype="audio/mpeg",
    )


@app.route("/api/history")
def job_history():
    """Return list of completed/failed jobs, newest first."""
    return jsonify(db.list_jobs(limit=200))


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
