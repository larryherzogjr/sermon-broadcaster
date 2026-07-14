"""
Phase 2: Claude-powered feedback interview + GitHub issue submission.

Orchestrates a multi-turn chat where Claude asks the user about a completed
job, references the actual job details, and proposes a structured summary
once enough context has been gathered. The summary plus the full diagnostic
bundle is then submitted as a GitHub issue.
"""
import logging
import re

import requests
from anthropic import Anthropic

import config
from pipeline import db
from pipeline.review_workflow import load_transcript

logger = logging.getLogger(__name__)

SUMMARY_MARKER = "---SUMMARY---"
MAX_ISSUE_TRANSCRIPT_CHARS = 20000
MAX_GITHUB_ISSUE_BODY_CHARS = 60000


def _fmt_sec(seconds):
    if seconds is None:
        return "—"
    try:
        return f"{float(seconds):.1f}s"
    except (TypeError, ValueError):
        return str(seconds)


def _job_context_lines(job):
    """Build the JOB CONTEXT block injected into Claude's system prompt."""
    result = job.get("result") or {}
    boundaries = result.get("boundaries") or {}
    processing = result.get("processing") or {}
    teaser = result.get("teaser") or {}
    timing = result.get("timing") or {}

    title = boundaries.get("sermon_title_guess") or "(no title detected)"
    confidence = boundaries.get("confidence") or "—"
    s_start = boundaries.get("sermon_start")
    s_end = boundaries.get("sermon_end")
    s_end_with = boundaries.get("sermon_end_with_prayer")
    s_end_without = boundaries.get("sermon_end_without_prayer")

    endpoint_used = "—"
    if s_end is not None:
        if s_end_with is not None and abs(s_end - s_end_with) < 0.5:
            endpoint_used = "with-prayer"
        elif s_end_without is not None and abs(s_end - s_end_without) < 0.5:
            endpoint_used = "without-prayer"

    orig = processing.get("original_duration")
    final = processing.get("final_duration")
    silence = processing.get("silence_adjustment") or 0.0
    tempo = processing.get("tempo_factor") or 1.0

    teaser_text = (teaser.get("teaser_text") or "")[:200]

    outputs = result.get("outputs") or []
    variant_list = ", ".join(o.get("variant", "") for o in outputs) or "—"
    total_time = timing.get("total")

    return (
        f'- Sermon: "{title}" (job {job["job_id"]})\n'
        f"- Target duration: {job.get('target_duration', '—')}\n"
        f"- Detected boundaries: start={_fmt_sec(s_start)}, end={_fmt_sec(s_end)} "
        f"(endpoint used: {endpoint_used})\n"
        f"- Confidence: {confidence}\n"
        f"- Processing: original {_fmt_sec(orig)}, final {_fmt_sec(final)}, "
        f"silence adjustment {silence:+.1f}s, tempo factor {tempo:.3f}x\n"
        f'- Teaser selected: "{teaser_text}..."\n'
        f"- Outputs produced: {variant_list}\n"
        f"- Total processing time: {_fmt_sec(total_time)}"
    )


def _build_system_prompt(job, user_turn_count):
    job_ctx = _job_context_lines(job)
    return (
        "You are gathering feedback about a sermon broadcaster job for the developer.\n"
        "Be warm, curious, and specific — reference the actual job details so the\n"
        "user feels heard and so the developer gets actionable info.\n\n"
        f"JOB CONTEXT:\n{job_ctx}\n\n"
        "INTERVIEW PATTERN:\n"
        "1. Open with one open-ended question about how the output sounded or felt.\n"
        "2. Ask 1-3 specific follow-ups, referencing job details when relevant.\n"
        "3. Propose a structured summary when ready.\n\n"
        f"TURN COUNT: This is user turn {user_turn_count}.\n"
        f"- Soft cap is {config.FEEDBACK_SOFT_CAP}. If N >= {config.FEEDBACK_SOFT_CAP}, "
        "lean toward proposing the summary now.\n"
        f"- Hard cap is {config.FEEDBACK_HARD_CAP}. If N >= {config.FEEDBACK_HARD_CAP}, "
        "you MUST propose the summary in this response.\n\n"
        "SUMMARY FORMAT: When proposing a summary, structure your response as:\n\n"
        "<closing message to the user>\n\n"
        f"{SUMMARY_MARKER}\n"
        "severity: <low|medium|high>\n"
        "title: <one-line title for the GitHub issue>\n\n"
        "<2-4 paragraph summary covering what worked, what didn't, "
        "and what to investigate>\n\n"
        "SEVERITY RUBRIC:\n"
        "- low: minor/cosmetic, suggestion, edge case\n"
        "- medium: noticeable bug or quality issue but not blocking\n"
        "- high: output unusable, feature broken, major correctness problem"
    )


def _parse_summary(text):
    """Return {severity, title, body, closing} if marker found, else None."""
    if SUMMARY_MARKER not in text:
        return None
    head, _, tail = text.partition(SUMMARY_MARKER)
    severity = None
    title = None
    body_lines = []
    saw_severity = False
    saw_title = False
    for line in tail.strip().splitlines():
        if not saw_severity:
            m_sev = re.match(r"\s*severity:\s*(low|medium|high)\s*$", line, re.IGNORECASE)
            if m_sev:
                severity = m_sev.group(1).lower()
                saw_severity = True
                continue
        if not saw_title:
            m_title = re.match(r"\s*title:\s*(.+)$", line)
            if m_title:
                title = m_title.group(1).strip()
                saw_title = True
                continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return {
        "severity": severity,
        "title": title,
        "body": body,
        "closing": head.strip(),
    }


def _call_claude(system_prompt, conversation):
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.CLAUDE_MODEL_FEEDBACK,
        max_tokens=2048,
        system=system_prompt,
        messages=conversation,
    )
    return response.content[0].text.strip()


def _conversation_from_messages(messages):
    """Convert stored messages into Anthropic message format.

    The Anthropic API requires the first message to be from the user, so when
    the conversation begins with an assistant turn (Claude's opening question),
    we prepend a synthetic primer."""
    conv = [{"role": m["role"], "content": m["content"]} for m in messages]
    if conv and conv[0]["role"] == "assistant":
        conv.insert(0, {"role": "user", "content": "Please begin the feedback interview."})
    return conv


def start_interview(job_id):
    """Idempotent. Returns existing session if one exists, else creates one
    and triggers Claude's opening question."""
    existing = db.get_feedback_session_by_job(job_id)
    if existing:
        out = {
            "session_id": existing["id"],
            "status": existing["status"],
            "messages": db.get_feedback_messages(existing["id"]),
        }
        if existing.get("github_issue_url"):
            out["github_issue_url"] = existing["github_issue_url"]
        return out

    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    session_id = db.create_feedback_session(job_id)

    system_prompt = _build_system_prompt(job, user_turn_count=1)
    opening = _call_claude(
        system_prompt,
        [{"role": "user", "content": "Please begin the feedback interview."}],
    )
    db.append_feedback_message(session_id, "assistant", opening)

    return {
        "session_id": session_id,
        "status": "in_progress",
        "messages": db.get_feedback_messages(session_id),
    }


def send_user_message(session_id, user_text):
    session = db.get_feedback_session(session_id)
    if not session:
        raise ValueError(f"Session not found: {session_id}")
    if session["status"] != "in_progress":
        raise ValueError(f"Session is not in progress: {session_id}")

    db.append_feedback_message(session_id, "user", user_text)

    messages = db.get_feedback_messages(session_id)
    user_turn = sum(1 for m in messages if m["role"] == "user")

    job = db.get_job(session["job_id"])
    system_prompt = _build_system_prompt(job, user_turn_count=user_turn)
    conversation = _conversation_from_messages(messages)

    assistant_text = _call_claude(system_prompt, conversation)
    parsed = _parse_summary(assistant_text)

    # Retry once at the hard cap if Claude forgot the marker.
    if user_turn >= config.FEEDBACK_HARD_CAP and parsed is None:
        logger.warning(
            f"Session {session_id}: hard cap reached without summary marker; retrying"
        )
        retry_conv = conversation + [
            {"role": "assistant", "content": assistant_text},
            {
                "role": "user",
                "content": (
                    "We're at the hard cap and need to wrap up. Please re-send your "
                    "response, this time including the structured summary block "
                    f"starting with the '{SUMMARY_MARKER}' marker on its own line, "
                    "followed by 'severity:', 'title:', and a 2-4 paragraph body, "
                    "exactly as specified in your instructions."
                ),
            },
        ]
        assistant_text = _call_claude(system_prompt, retry_conv)
        parsed = _parse_summary(assistant_text)

    # Final fallback: synthesize a summary from Claude's response so the UI
    # always has something the user can edit and submit.
    summary_proposed = parsed is not None
    if user_turn >= config.FEEDBACK_HARD_CAP and not summary_proposed:
        sermon_title = (
            ((job.get("result") or {}).get("boundaries") or {}).get("sermon_title_guess")
            or "(no title)"
        )
        parsed = {
            "severity": "medium",
            "title": f"Feedback: {sermon_title} ({job['job_id']})",
            "body": assistant_text,
            "closing": "",
        }
        summary_proposed = True
        logger.warning(
            f"Session {session_id}: fallback summary used after hard-cap retry failed"
        )

    db.append_feedback_message(session_id, "assistant", assistant_text)

    out = {
        "assistant_text": assistant_text,
        "user_turn": user_turn,
        "summary_proposed": summary_proposed,
    }
    if summary_proposed:
        out["summary"] = {
            "severity": parsed["severity"] or "medium",
            "title": parsed["title"] or "",
            "body": parsed["body"],
        }
    return out


def _build_issue_body(job, messages, summary, severity):
    result = job.get("result") or {}
    boundaries = result.get("boundaries") or {}
    processing = result.get("processing") or {}
    timing = result.get("timing") or {}
    teaser = result.get("teaser") or {}
    transcript = result.get("transcript_summary") or {}
    outputs = result.get("outputs") or []

    out = []
    out.append("## Summary\n")
    out.append(f"{summary}\n\n")
    out.append(f"**Severity:** {severity}\n\n")

    out.append("## Interview\n\n")
    for m in messages:
        role = "Claude" if m["role"] == "assistant" else "User"
        out.append(f"**{role}:** {m['content']}\n\n")

    out.append("## Diagnostic\n\n")
    out.append(f"**Job ID:** {job['job_id']}\n")
    out.append(f"**Created:** {job.get('created_at', '—')}\n")
    out.append(f"**Source:** {job.get('source', '—')}\n")
    out.append(f"**Target duration:** {job.get('target_duration', '—')}\n")
    out.append(f"**Status:** {job.get('status', '—')}\n\n")

    out.append("### Boundaries\n")
    out.append(f"- sermon_start: {boundaries.get('sermon_start')}\n")
    out.append(f"- sermon_end (selected): {boundaries.get('sermon_end')}\n")
    out.append(f"- sermon_end_with_prayer: {boundaries.get('sermon_end_with_prayer')}\n")
    out.append(f"- sermon_end_without_prayer: {boundaries.get('sermon_end_without_prayer')}\n")
    out.append(f"- sermon_title_guess: {boundaries.get('sermon_title_guess')}\n")
    out.append(f"- confidence: {boundaries.get('confidence')}\n\n")

    out.append("### Processing\n")
    out.append(f"- original_duration: {processing.get('original_duration')}\n")
    out.append(f"- target_duration: {processing.get('target_duration')}\n")
    out.append(f"- final_duration: {processing.get('final_duration')}\n")
    out.append(f"- silence_adjustment: {processing.get('silence_adjustment')}\n")
    out.append(f"- tempo_factor: {processing.get('tempo_factor')}\n\n")

    out.append("### Timing\n")
    for key in ("download", "transcribe", "boundaries", "extract",
                "fit_to_duration", "assembly", "total"):
        if key in timing:
            out.append(f"- {key}: {timing[key]}\n")
    out.append("\n")

    out.append("### Teaser\n")
    out.append(f"- teaser_text: {teaser.get('teaser_text')}\n")
    out.append(f"- teaser_start: {teaser.get('teaser_start')}\n")
    out.append(f"- teaser_end: {teaser.get('teaser_end')}\n")
    out.append(f"- reason: {teaser.get('reason')}\n\n")

    out.append("### Outputs\n")
    for o in outputs:
        out.append(
            f"- {o.get('variant', '')}: {o.get('filename', '')} "
            f"({o.get('note') or ''})\n"
        )
    out.append("\n")

    out.append("### Status messages\n")
    for msg in job.get("messages") or []:
        out.append(f"- [{msg.get('time', '')}] {msg.get('text', '')}\n")
    out.append("\n")

    transcript_text = transcript.get("full_text") or ""
    if not transcript_text:
        try:
            transcript_text = load_transcript(job["job_id"]).get("full_text", "")
        except (OSError, ValueError, TypeError):
            transcript_text = ""
    out.append("## Sermon Transcript Excerpt\n")
    if transcript_text:
        excerpt = transcript_text[:MAX_ISSUE_TRANSCRIPT_CHARS]
        out.append(excerpt)
        if len(transcript_text) > len(excerpt):
            out.append(
                f"\n\n_(Transcript truncated after {MAX_ISSUE_TRANSCRIPT_CHARS:,} "
                "characters to stay within GitHub's issue size limit.)_"
            )
    else:
        out.append("(no transcript available)")

    body = "".join(out)
    if len(body) > MAX_GITHUB_ISSUE_BODY_CHARS:
        note = "\n\n_(Issue body truncated to stay within GitHub's size limit.)_"
        body = body[:MAX_GITHUB_ISSUE_BODY_CHARS - len(note)] + note
    return body


def submit_to_github(session_id, summary_override=None,
                     severity_override=None, title_override=None):
    session = db.get_feedback_session(session_id)
    if not session:
        raise ValueError(f"Session not found: {session_id}")
    if session["status"] == "submitted":
        return session["github_issue_url"]

    job = db.get_job(session["job_id"])
    if not job:
        raise ValueError(f"Job not found: {session['job_id']}")
    messages = db.get_feedback_messages(session_id)

    # Pull the most recent assistant message that contains a parsed summary
    # — that's our default for any field the user didn't override.
    parsed = None
    for m in reversed(messages):
        if m["role"] == "assistant":
            parsed = _parse_summary(m["content"])
            if parsed:
                break

    summary = summary_override if summary_override is not None else (
        parsed["body"] if parsed else ""
    )

    severity = severity_override or (parsed["severity"] if parsed else None) or "medium"
    severity = severity.lower()
    if severity not in ("low", "medium", "high"):
        severity = "medium"

    sermon_title = (
        ((job.get("result") or {}).get("boundaries") or {}).get("sermon_title_guess")
        or "(no title)"
    )
    default_title = f"Feedback: {sermon_title} ({job['job_id']})"
    if title_override:
        title = title_override
    elif parsed and parsed.get("title"):
        title = parsed["title"]
    else:
        title = default_title

    body = _build_issue_body(job, messages, summary, severity)

    if not config.GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is not configured")
    if not config.GITHUB_REPO:
        raise RuntimeError("GITHUB_REPO is not configured")

    api_url = f"https://api.github.com/repos/{config.GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"Bearer {config.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "title": title,
        "body": body,
        "labels": ["feedback", f"severity:{severity}"],
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(
            f"GitHub issue creation failed: {resp.status_code} {resp.text[:500]}"
        )

    issue_url = resp.json().get("html_url", "")
    db.submit_feedback_session(session_id, summary, severity, issue_url)
    return issue_url
