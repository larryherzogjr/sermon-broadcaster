"""SQLite persistence for job state."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  completed_at TEXT,
  source TEXT NOT NULL,
  source_type TEXT NOT NULL,
  target_duration TEXT NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  include_dynamic INTEGER NOT NULL,
  include_stock INTEGER NOT NULL,
  sermon_only INTEGER NOT NULL,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS job_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  text TEXT NOT NULL,
  FOREIGN KEY (job_id) REFERENCES jobs(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_job ON job_messages(job_id);

CREATE TABLE IF NOT EXISTS job_outputs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  variant TEXT NOT NULL,
  filename TEXT NOT NULL,
  note TEXT,
  FOREIGN KEY (job_id) REFERENCES jobs(id)
);
CREATE INDEX IF NOT EXISTS idx_outputs_job ON job_outputs(job_id);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    finally:
        conn.close()


def _now():
    return datetime.now().isoformat(timespec="seconds")


def init_schema():
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)


def mark_orphans_failed():
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', error=?, completed_at=? "
            "WHERE status IN ('queued','running')",
            ("Server restarted during processing", _now()),
        )


def create_job(job_id, source, source_type, target_duration,
               include_dynamic, include_stock, sermon_only):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, created_at, source, source_type, target_duration, "
            "status, include_dynamic, include_stock, sermon_only) "
            "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
            (job_id, _now(), source, source_type, target_duration,
             1 if include_dynamic else 0,
             1 if include_stock else 0,
             1 if sermon_only else 0),
        )


def append_message(job_id, ts, text):
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='running' WHERE id=? AND status='queued'",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO job_messages (job_id, ts, text) VALUES (?, ?, ?)",
            (job_id, ts, text),
        )


def set_result(job_id, outputs, metadata):
    metadata_json = json.dumps(metadata, default=str)
    with _connect() as conn:
        conn.execute("BEGIN")
        try:
            conn.execute(
                "UPDATE jobs SET status='complete', completed_at=?, metadata_json=? "
                "WHERE id=?",
                (_now(), metadata_json, job_id),
            )
            for out in outputs:
                conn.execute(
                    "INSERT INTO job_outputs (job_id, variant, filename, note) "
                    "VALUES (?, ?, ?, ?)",
                    (job_id, out.get("variant", ""), out.get("filename", ""),
                     out.get("note") or ""),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def set_error(job_id, error_str):
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', error=?, completed_at=? WHERE id=?",
            (error_str, _now(), job_id),
        )


def _row_to_job_dict(conn, row):
    job_id = row["id"]
    status = row["status"]

    msgs = conn.execute(
        "SELECT ts, text FROM job_messages WHERE job_id=? ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    messages = [{"time": m["ts"], "text": m["text"]} for m in msgs]

    d = {
        "job_id": job_id,
        "source": row["source"],
        "target_duration": row["target_duration"],
        "status": status,
        "messages": messages,
    }

    if status == "complete":
        outs = conn.execute(
            "SELECT variant, filename, note FROM job_outputs WHERE job_id=? ORDER BY id ASC",
            (job_id,),
        ).fetchall()
        outputs = []
        for o in outs:
            filename = o["filename"]
            outputs.append({
                "filename": filename,
                "variant": o["variant"],
                "note": o["note"] or "",
                "available": os.path.exists(os.path.join(config.OUTPUT_DIR, filename)),
            })

        meta = {}
        if row["metadata_json"]:
            try:
                meta = json.loads(row["metadata_json"])
            except (ValueError, TypeError):
                meta = {}

        d["result"] = {
            "output_filename": outputs[0]["filename"] if outputs else None,
            "outputs": outputs,
            "boundaries": meta.get("boundaries"),
            "processing": meta.get("processing"),
            "timing": meta.get("timing"),
            "transcript_summary": meta.get("transcript_summary"),
            "broadcast_duration": meta.get("broadcast_duration"),
            "include_bumpers": meta.get("include_bumpers"),
            "teaser": meta.get("teaser"),
        }

    if row["error"]:
        d["error"] = row["error"]

    return d


def get_job(job_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        return _row_to_job_dict(conn, row)


def list_jobs(limit=200):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('complete','failed') "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_job_dict(conn, r) for r in rows]
