"""SQLite persistence for job state."""
import json
import os
import sqlite3
import uuid
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

CREATE TABLE IF NOT EXISTS feedback_sessions (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  submitted_at TEXT,
  status TEXT NOT NULL,
  summary TEXT,
  severity TEXT,
  github_issue_url TEXT,
  FOREIGN KEY (job_id) REFERENCES jobs(id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_job ON feedback_sessions(job_id);

CREATE TABLE IF NOT EXISTS feedback_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES feedback_sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_msgs_session ON feedback_messages(session_id);
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

    fb_row = conn.execute(
        "SELECT id, status, github_issue_url FROM feedback_sessions "
        "WHERE job_id=? ORDER BY created_at DESC LIMIT 1",
        (job_id,),
    ).fetchone()
    if fb_row:
        feedback = {
            "status": fb_row["status"],
            "session_id": fb_row["id"],
        }
        if fb_row["github_issue_url"]:
            feedback["issue_url"] = fb_row["github_issue_url"]
    else:
        feedback = {"status": "none"}

    d = {
        "job_id": job_id,
        "created_at": row["created_at"],
        "source": row["source"],
        "source_type": row["source_type"],
        "target_duration": row["target_duration"],
        "status": status,
        "messages": messages,
        "feedback": feedback,
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


# ── Feedback sessions ────────────────────────────────────────────────

def create_feedback_session(job_id):
    session_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback_sessions (id, job_id, created_at, status) "
            "VALUES (?, ?, ?, 'in_progress')",
            (session_id, job_id, _now()),
        )
    return session_id


def _row_to_session_dict(row):
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "created_at": row["created_at"],
        "submitted_at": row["submitted_at"],
        "status": row["status"],
        "summary": row["summary"],
        "severity": row["severity"],
        "github_issue_url": row["github_issue_url"],
    }


def get_feedback_session(session_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM feedback_sessions WHERE id=?", (session_id,)
        ).fetchone()
        return _row_to_session_dict(row) if row else None


def get_feedback_session_by_job(job_id):
    """Most-recent feedback session for a job, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM feedback_sessions WHERE job_id=? "
            "ORDER BY created_at DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        return _row_to_session_dict(row) if row else None


def append_feedback_message(session_id, role, content):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback_messages (session_id, ts, role, content) "
            "VALUES (?, ?, ?, ?)",
            (session_id, _now(), role, content),
        )


def get_feedback_messages(session_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, role, content FROM feedback_messages "
            "WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [
            {"ts": r["ts"], "role": r["role"], "content": r["content"]}
            for r in rows
        ]


def submit_feedback_session(session_id, summary, severity, issue_url):
    with _connect() as conn:
        conn.execute(
            "UPDATE feedback_sessions "
            "SET status='submitted', submitted_at=?, summary=?, severity=?, "
            "github_issue_url=? WHERE id=?",
            (_now(), summary, severity, issue_url, session_id),
        )
