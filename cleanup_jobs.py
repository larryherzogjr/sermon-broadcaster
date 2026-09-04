#!/usr/bin/env python3
"""Delete Job History entries older than the configured retention period."""
import argparse
import logging
from datetime import datetime, timedelta, timezone

import config
from pipeline import db
from pipeline.job_cleanup import cleanup_job_files


logger = logging.getLogger(__name__)


def cleanup_expired_jobs(days=14, now=None, dry_run=False):
    """Delete jobs created before the retention cutoff, regardless of status."""
    if days < 0:
        raise ValueError("Retention days cannot be negative")

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    cutoff = current_time.astimezone(timezone.utc) - timedelta(days=days)

    db.init_schema()
    job_ids = db.list_job_ids_created_before(cutoff.isoformat(timespec="seconds"))
    deleted = []
    for job_id in job_ids:
        job = db.get_job(job_id)
        if not job:
            continue
        if dry_run:
            logger.info("Would delete expired job %s (%s)", job_id, job["status"])
            continue
        if db.delete_job(job_id):
            cleanup_job_files(job_id, job)
            deleted.append(job_id)
            logger.info("Deleted expired job %s (%s)", job_id, job["status"])
    return deleted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14, help="retention period (default: 14)")
    parser.add_argument("--dry-run", action="store_true", help="report without deleting")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    deleted = cleanup_expired_jobs(days=args.days, dry_run=args.dry_run)
    logger.info("Expired job cleanup complete: %d job(s) deleted", len(deleted))


if __name__ == "__main__":
    main()
