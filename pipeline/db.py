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
            "WHERE status IN ('queued','running','analyzing')",
            ("Server restarted during processing", _now()),
        )
        conn.execute(
            "UPDATE jobs SET status='awaiting_review', error=? "
            "WHERE status='rendering' AND metadata_json IS NOT NULL",
            ("Server restarted during rendering. Your selections were preserved; render again.",),
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
            "UPDATE jobs SET status='analyzing' WHERE id=? AND status='queued'",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO job_messages (job_id, ts, text) VALUES (?, ?, ?)",
            (job_id, ts, text),
        )


def set_status(job_id, status):
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, error=NULL WHERE id=?",
            (status, job_id),
        )


def _load_metadata_value(raw):
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def get_metadata(job_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        return _load_metadata_value(row["metadata_json"]) if row else None


def set_analysis_ready(job_id, metadata):
    metadata_json = json.dumps(metadata, default=str)
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='awaiting_review', error=NULL, "
            "metadata_json=? WHERE id=?",
            (metadata_json, job_id),
        )


def update_metadata(job_id, updates):
    with _connect() as conn:
        row = conn.execute(
            "SELECT metadata_json FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if not row:
            raise ValueError("Job not found")
        metadata = _load_metadata_value(row["metadata_json"])
        metadata.update(updates or {})
        conn.execute(
            "UPDATE jobs SET metadata_json=? WHERE id=?",
            (json.dumps(metadata, default=str), job_id),
        )


def claim_render(job_id, review):
    """Atomically persist confirmed markers and claim a job for rendering."""
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT status, metadata_json FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row or row["status"] != "awaiting_review":
                conn.execute("ROLLBACK")
                return False
            metadata = _load_metadata_value(row["metadata_json"])
            metadata["review"] = review
            conn.execute(
                "UPDATE jobs SET status='rendering', error=NULL, metadata_json=? "
                "WHERE id=?",
                (json.dumps(metadata, default=str), job_id),
            )
            conn.execute("COMMIT")
            return True
        except Exception:
            conn.execute("ROLLBACK")
            raise


def set_result(job_id, outputs, metadata):
    with _connect() as conn:
        existing = conn.execute(
            "SELECT metadata_json FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        merged = _load_metadata_value(existing["metadata_json"] if existing else None)
        merged.update(metadata or {})
        metadata_json = json.dumps(merged, default=str)
        conn.execute("BEGIN")
        try:
            conn.execute(
                "UPDATE jobs SET status='complete', completed_at=?, metadata_json=? "
                "WHERE id=?",
                (_now(), metadata_json, job_id),
            )
            conn.execute("DELETE FROM job_outputs WHERE job_id=?", (job_id,))
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


def set_review_error(job_id, error_str):
    """Return a failed render to the editor so selections can be adjusted."""
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='awaiting_review', error=? WHERE id=?",
            (error_str, job_id),
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
        "include_dynamic": bool(row["include_dynamic"]),
        "include_stock": bool(row["include_stock"]),
        "sermon_only": bool(row["sermon_only"]),
        "messages": messages,
        "feedback": feedback,
    }

    meta = _load_metadata_value(row["metadata_json"])
    if meta.get("review"):
        d["review"] = meta["review"]

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

        d["result"] = {
            "output_filename": outputs[0]["filename"] if outputs else None,
            "outputs": outputs,
            "boundaries": meta.get("boundaries"),
            "processing": meta.get("processing"),
            "timing": meta.get("timing"),
            "transcript_summary": meta.get("transcript_summary"),
            "broadcast_duration": meta.get("broadcast_duration"),
            "output_durations": meta.get("output_durations"),
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
            "SELECT * FROM jobs WHERE status IN ('complete','failed','awaiting_review') "
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
