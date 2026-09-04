"""Remove all persisted files owned by a job."""
import logging
import os
import shutil

import config


logger = logging.getLogger(__name__)


def _review_job_dir(job_id):
    if not job_id or any(c not in "0123456789_" for c in job_id):
        raise ValueError("Invalid job id")
    return os.path.join(config.REVIEW_DIR, job_id)


def _remove_file(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        logger.warning("Could not remove deleted-job file %s", path, exc_info=True)


def cleanup_job_files(job_id, job=None, upload_dir=None):
    """Remove persisted and temporary files belonging to exactly one job id."""
    try:
        shutil.rmtree(_review_job_dir(job_id), ignore_errors=True)
    except ValueError:
        return

    upload_dir = upload_dir or config.UPLOAD_DIR
    if job and job.get("source_type") == "upload":
        source_name = os.path.basename(str(job.get("source") or ""))
        if source_name.startswith(f"{job_id}_"):
            _remove_file(os.path.join(upload_dir, source_name))

    output_names = {
        f"sermon_{job_id}.mp3",
        f"sermon_{job_id}_dynamic.mp3",
        f"sermon_{job_id}_stock.mp3",
    }
    for output in ((job or {}).get("result") or {}).get("outputs", []):
        filename = os.path.basename(str(output.get("filename") or ""))
        if filename.startswith(f"sermon_{job_id}"):
            output_names.add(filename)
    for filename in output_names:
        _remove_file(os.path.join(config.OUTPUT_DIR, filename))

    work_prefix = f"review_{job_id}_"
    try:
        for name in os.listdir(config.WORK_DIR):
            if name.startswith(work_prefix):
                shutil.rmtree(os.path.join(config.WORK_DIR, name), ignore_errors=True)
    except OSError:
        pass
