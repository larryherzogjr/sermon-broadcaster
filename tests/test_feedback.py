from pipeline import feedback


def test_issue_body_recovers_and_caps_persisted_transcript(monkeypatch):
    monkeypatch.setattr(
        feedback,
        "load_transcript",
        lambda _job_id: {"full_text": "transcript " * 10000},
    )
    job = {
        "job_id": "20260714_120000_000002",
        "result": {},
        "messages": [],
    }
    messages = [{"role": "assistant", "content": "detail " * 10000}]

    body = feedback._build_issue_body(job, messages, "summary", "medium")

    assert len(body) <= feedback.MAX_GITHUB_ISSUE_BODY_CHARS
    assert "Issue body truncated" in body
