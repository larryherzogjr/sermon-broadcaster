"""
Sermon Broadcaster
Flask app for processing church service videos into broadcast-ready audio.
"""
import os
import math
import logging
import threading
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import config
from pipeline import db
from pipeline import feedback
from pipeline.review_workflow import (
    analyze_job,
    build_preflight,
    create_teaser_preview,
    load_transcript,
    parse_duration,
    render_job,
    review_job_dir,
)

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


def _run_analysis(job: Job):
    """Prepare persisted audio/transcript artifacts, then pause for review."""
    try:
        db.set_status(job.job_id, "analyzing")
        metadata = analyze_job(
            job.job_id,
            youtube_url=job.youtube_url,
            local_file=job.local_file,
            target_duration=job.target_duration,
            include_dynamic=job.include_bumpers_dynamic,
            include_stock=job.include_bumpers_stock,
            sermon_only=job.sermon_only,
            status_callback=job.update_status,
        )
        db.set_analysis_ready(job.job_id, metadata)
        job.update_status("Analysis complete. Review the sermon and teaser selections.")
    except Exception as e:
        job.set_error(str(e))
        logger.exception(f"Analysis for job {job.job_id} failed")


def _run_render(job_id: str, selections: dict):
    """Render confirmed selections in a background thread."""
    try:
        metadata = db.get_metadata(job_id)
        if not metadata:
            raise ValueError("Review data is no longer available for this job")
        db.set_status(job_id, "rendering")

        def update(message):
            ts = datetime.now().strftime("%H:%M:%S")
            db.append_message(job_id, ts, message)
            logger.info(f"[Job {job_id}] {message}")

        result = render_job(job_id, metadata, selections, status_callback=update)
        db.set_result(job_id, result.pop("outputs", []), result)
    except Exception as e:
        db.set_review_error(job_id, str(e))
        logger.exception(f"Render for job {job_id} failed")


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        default_duration=config.DEFAULT_TARGET_DURATION,
        default_broadcast_duration=config.DEFAULT_BROADCAST_DURATION,
    )


@app.route("/api/analyze", methods=["POST"])
@app.route("/api/process", methods=["POST"])
def start_processing():
    """Analyze a source and pause when it is ready for human review."""

    local_file = None
    if request.content_type and "multipart" in request.content_type:
        file = request.files.get("file")
        youtube_url = request.form.get("url", "").strip()
        include_dynamic = request.form.get("include_bumpers_dynamic", "true").lower() == "true"
        include_stock = request.form.get("include_bumpers_stock", "false").lower() == "true"
        sermon_only = request.form.get("sermon_only", "false").lower() == "true"
        any_bumpers = include_dynamic or include_stock
        default_dur = config.DEFAULT_BROADCAST_DURATION if any_bumpers else config.DEFAULT_TARGET_DURATION
        target_duration = request.form.get("target_duration", default_dur).strip()

        if file and file.filename:
            if not _allowed_file(file.filename):
                return jsonify({"error": "That audio/video file type is not supported"}), 400
            job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = secure_filename(f"{job_id}_{file.filename}")
            local_file = os.path.join(UPLOAD_DIR, filename)
            file.save(local_file)
            logger.info(
                "File uploaded: %s (%.1f MB)", local_file,
                os.path.getsize(local_file) / 1024 / 1024,
            )
        elif not youtube_url:
            return jsonify({"error": "Please provide a YouTube URL or upload a video/audio file"}), 400
    else:
        data = request.get_json() or {}
        youtube_url = data.get("url", "").strip()
        include_dynamic = data.get("include_bumpers_dynamic", True)
        include_stock = data.get("include_bumpers_stock", False)
        sermon_only = data.get("sermon_only", False)
        if data.get("include_bumpers") and not (include_dynamic or include_stock):
            include_dynamic = True
        any_bumpers = include_dynamic or include_stock
        default_dur = config.DEFAULT_BROADCAST_DURATION if any_bumpers else config.DEFAULT_TARGET_DURATION
        target_duration = data.get("target_duration", default_dur).strip()

    if not youtube_url and not local_file:
        return jsonify({"error": "Please provide a YouTube URL or upload a file"}), 400

    if youtube_url and "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        return jsonify({"error": "Please provide a valid YouTube URL"}), 400

    try:
        parse_duration(target_duration)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not local_file:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    source = os.path.basename(local_file) if local_file else youtube_url
    job = Job(job_id, target_duration,
              source=source,
              source_type="upload" if local_file else "youtube",
              include_bumpers_dynamic=include_dynamic,
              include_bumpers_stock=include_stock,
              sermon_only=sermon_only,
              youtube_url=None if local_file else youtube_url,
              local_file=local_file)

    thread = threading.Thread(target=_run_analysis, args=(job,), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    j = db.get_job(job_id)
    if not j:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(j)


@app.route("/api/jobs/<job_id>/transcript")
def review_transcript(job_id):
    job = db.get_job(job_id)
    if not job or not job.get("review"):
        return jsonify({"error": "Review job not found"}), 404
    try:
        transcript = load_transcript(job_id)
        # The editor only needs these three fields. Projecting the response also
        # strips optional Whisper confidence values, which may legally be NaN in
        # Python but are invalid JSON in browsers (and would break response.json()).
        segments = []
        for segment in transcript.get("segments", []):
            try:
                start = float(segment["start"])
                end = float(segment["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            segments.append({
                "start": start,
                "end": end,
                "text": str(segment.get("text") or ""),
            })

        duration = transcript.get("duration", job["review"].get("audio_duration"))
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = float(job["review"].get("audio_duration") or 0.0)
        if not math.isfinite(duration):
            duration = float(job["review"].get("audio_duration") or 0.0)

        return jsonify({
            "segments": segments,
            "duration": duration,
        })
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/api/jobs/<job_id>/waveform")
def review_waveform(job_id):
    job = db.get_job(job_id)
    if not job or not job.get("review"):
        return jsonify({"error": "Review job not found"}), 404
    return send_from_directory(review_job_dir(job_id), "waveform.json", mimetype="application/json")


@app.route("/api/jobs/<job_id>/audio")
def review_audio(job_id):
    job = db.get_job(job_id)
    if not job or not job.get("review"):
        return jsonify({"error": "Review job not found"}), 404
    return send_from_directory(
        review_job_dir(job_id), "raw_audio.wav", mimetype="audio/wav", conditional=True
    )


@app.route("/api/jobs/<job_id>/preflight", methods=["POST"])
def review_preflight(job_id):
    job = db.get_job(job_id)
    if not job or not job.get("review"):
        return jsonify({"error": "Review job not found"}), 404
    try:
        return jsonify(build_preflight(job["review"], request.get_json() or {}))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/jobs/<job_id>/preview-teaser", methods=["POST"])
def review_teaser_preview(job_id):
    job = db.get_job(job_id)
    metadata = db.get_metadata(job_id)
    if not job or not metadata or not job.get("review"):
        return jsonify({"error": "Review job not found"}), 404
    try:
        create_teaser_preview(job_id, metadata, request.get_json() or {})
        return jsonify({"url": f"/api/jobs/{job_id}/teaser-preview-audio"})
    except (ValueError, RuntimeError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/jobs/<job_id>/teaser-preview-audio")
def review_teaser_preview_audio(job_id):
    job = db.get_job(job_id)
    if not job or not job.get("review"):
        return jsonify({"error": "Review job not found"}), 404
    return send_from_directory(
        review_job_dir(job_id), "teaser_preview.wav", mimetype="audio/wav", conditional=True
    )


@app.route("/api/jobs/<job_id>/render", methods=["POST"])
def review_render(job_id):
    job = db.get_job(job_id)
    metadata = db.get_metadata(job_id)
    if not job or not metadata or not job.get("review"):
        return jsonify({"error": "Review job not found"}), 404
    if job["status"] != "awaiting_review":
        return jsonify({"error": f"Job is currently {job['status']}"}), 409

    selections = request.get_json() or {}
    try:
        preflight = build_preflight(job["review"], selections)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not preflight["ready"]:
        return jsonify({"error": " ".join(preflight["blockers"]), "preflight": preflight}), 400

    review = dict(metadata["review"])
    review.update(preflight["selections"])
    review["markers_confirmed"] = True
    metadata["review"] = review
    db.update_metadata(job_id, {"review": review})
    db.set_status(job_id, "rendering")
    thread = threading.Thread(
        target=_run_render, args=(job_id, preflight["selections"]), daemon=True
    )
    thread.start()
    return jsonify({"job_id": job_id, "status": "rendering"})


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


# ── Feedback routes ──────────────────────────────────────────────────

@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/feedback/<job_id>")
def feedback_page(job_id):
    j = db.get_job(job_id)
    if not j:
        return "Job not found", 404
    return render_template("feedback.html", job_id=job_id, job=j)


@app.route("/api/feedback/<job_id>/start", methods=["POST"])
def feedback_start(job_id):
    j = db.get_job(job_id)
    if not j:
        return jsonify({"error": "Job not found"}), 404
    if j.get("status") != "complete":
        return jsonify({"error": "Feedback is only available for completed jobs"}), 400
    try:
        return jsonify(feedback.start_interview(job_id))
    except Exception as e:
        logger.exception("Feedback start failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/<session_id>/message", methods=["POST"])
def feedback_message(session_id):
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400
    try:
        return jsonify(feedback.send_user_message(session_id, text))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Feedback message failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/<session_id>/submit", methods=["POST"])
def feedback_submit(session_id):
    data = request.get_json() or {}
    summary = data.get("summary")
    severity = data.get("severity")
    title = data.get("title")
    try:
        issue_url = feedback.submit_to_github(
            session_id,
            summary_override=summary,
            severity_override=severity,
            title_override=title,
        )
        return jsonify({"issue_url": issue_url})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.exception("Feedback submit failed")
        return jsonify({"error": str(e)}), 500


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
