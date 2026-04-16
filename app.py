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
from pipeline.orchestrator import run_pipeline

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask App ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload limit

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "webm", "mp3", "wav", "m4a", "flac", "ogg"}


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Job State ────────────────────────────────────────────────────────
jobs = {}


class Job:
    def __init__(self, job_id, target_duration,
                 include_bumpers_dynamic=False, include_bumpers_stock=False,
                 youtube_url=None, local_file=None):
        self.job_id = job_id
        self.youtube_url = youtube_url
        self.local_file = local_file
        self.target_duration = target_duration
        self.include_bumpers_dynamic = include_bumpers_dynamic
        self.include_bumpers_stock = include_bumpers_stock
        self.status = "queued"
        self.messages = []
        self.result = None
        self.error = None

    def update_status(self, message):
        self.status = "running"
        self.messages.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "text": message,
        })
        logger.info(f"[Job {self.job_id}] {message}")

    def to_dict(self):
        d = {
            "job_id": self.job_id,
            "source": self.youtube_url or (os.path.basename(self.local_file) if self.local_file else ""),
            "target_duration": self.target_duration,
            "status": self.status,
            "messages": self.messages,
        }
        if self.result:
            d["result"] = {
                "output_filename": self.result.get("output_filename"),
                "outputs": self.result.get("outputs", []),
                "boundaries": self.result.get("boundaries"),
                "processing": self.result.get("processing"),
                "timing": self.result.get("timing"),
                "transcript_summary": self.result.get("transcript"),
                "broadcast_duration": self.result.get("broadcast_duration"),
                "include_bumpers": self.result.get("include_bumpers"),
                "teaser": self.result.get("teaser"),
            }
        if self.error:
            d["error"] = str(self.error)
        return d


def _run_job(job: Job):
    """Run the pipeline in a background thread."""
    try:
        job.status = "running"
        result = run_pipeline(
            youtube_url=job.youtube_url,
            local_file=job.local_file,
            target_duration=job.target_duration,
            include_bumpers_dynamic=job.include_bumpers_dynamic,
            include_bumpers_stock=job.include_bumpers_stock,
            status_callback=job.update_status,
        )
        job.result = result
        job.status = "complete"
    except Exception as e:
        job.error = str(e)
        job.status = "failed"
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
                      include_bumpers_dynamic=include_dynamic,
                      include_bumpers_stock=include_stock,
                      local_file=filepath)
            jobs[job_id] = job

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
        # Support both new flags and legacy include_bumpers
        include_dynamic = data.get("include_bumpers_dynamic", False)
        include_stock = data.get("include_bumpers_stock", False)
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
              include_bumpers_dynamic=include_dynamic,
              include_bumpers_stock=include_stock,
              youtube_url=youtube_url)
    jobs[job_id] = job

    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job.to_dict())


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
    """Return list of completed jobs."""
    completed = [
        j.to_dict() for j in jobs.values()
        if j.status in ("complete", "failed")
    ]
    completed.sort(key=lambda j: j["job_id"], reverse=True)
    return jsonify(completed)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
