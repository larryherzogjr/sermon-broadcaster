"""
Module 2 (dispatcher): Transcription backend selector + normalizer.

Exposes a single `transcribe(audio_path, status_callback)` that returns ONE
identical dict shape regardless of backend, so boundary detection, teaser
selection, and the 27:18 / 29:30 trim logic stay completely backend-agnostic:

    {
        "full_text": str,
        "segments":  [{"start": float, "end": float, "text": str, ...extras}],
        "words":     [{"start": float, "end": float, "word": str}],
        "duration":  float,
        "language":  str,
    }

Backend is chosen via config.TRANSCRIBE_BACKEND:
    - "openai"         -> OpenAI Whisper API   (pipeline.transcriber_cloud)  [default / fallback]
    - "local"          -> headless M1 mini mlx-whisper HTTP service
    - "faster-whisper" -> legacy on-CPU faster-whisper (pipeline.transcriber)

"openai" delegates verbatim to the existing cloud transcriber, so flipping back
to it is a true no-op fallback.
"""
import os
import logging

import requests

import config

logger = logging.getLogger(__name__)

# Optional per-segment fields we preserve when a backend provides them, so the
# boundary logic can rely on them on EITHER backend (parity). Purely additive —
# downstream that only reads start/end/text is unaffected.
_SEGMENT_EXTRA_FIELDS = ("no_speech_prob", "avg_logprob", "compression_ratio")


def transcribe(audio_path: str, status_callback=None) -> dict:
    """Dispatch to the configured backend and return the normalized dict."""
    backend = (config.TRANSCRIBE_BACKEND or "openai").strip().lower()

    if backend == "local":
        return _transcribe_local(audio_path, status_callback)

    if backend in ("openai", "cloud"):
        # True no-op fallback: call the existing OpenAI path unchanged.
        from pipeline.transcriber_cloud import transcribe as _openai_transcribe
        return _openai_transcribe(audio_path, status_callback)

    if backend in ("faster-whisper", "faster_whisper", "local-cpu"):
        from pipeline.transcriber import transcribe as _fw_transcribe
        return _fw_transcribe(audio_path, status_callback)

    raise RuntimeError(
        f"Unknown TRANSCRIBE_BACKEND '{backend}'. "
        f"Use 'openai' (default), 'local', or 'faster-whisper'."
    )


# ── Local M1 mini HTTP backend ───────────────────────────────────────────────

def _local_base_url() -> str:
    base = (config.WHISPER_LOCAL_URL or "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "WHISPER_LOCAL_URL is not set but TRANSCRIBE_BACKEND=local. "
            "Set it to the mini's static IP, e.g. http://10.0.0.50:5005 "
            "(use the IP, not whisper-mini.local — mDNS won't resolve from the VM)."
        )
    return base


def check_health() -> dict:
    """GET /health on the local service. Raises on non-200. For smoke tests."""
    url = _local_base_url() + "/health"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return {"status": resp.text.strip()}


def _transcribe_local(audio_path: str, status_callback=None) -> dict:
    """POST the audio file to the mini's mlx-whisper service and normalize."""
    url = _local_base_url() + "/transcribe"
    timeout = config.WHISPER_LOCAL_TIMEOUT

    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info(
        f"Posting {size_mb:.1f} MB to local Whisper service at {url} "
        f"(timeout {timeout}s) — no upload cap, sending raw audio uncompressed"
    )
    if status_callback:
        status_callback("Transcribing via local Whisper service (M1 mini)...")

    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=timeout,
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Local Whisper service error ({resp.status_code}): {resp.text[:500]}"
        )

    try:
        data = resp.json()
    except ValueError as e:
        raise RuntimeError(
            f"Local Whisper service returned non-JSON response: {resp.text[:300]}"
        ) from e

    result = _normalize_local(data)

    logger.info(
        f"Local transcription complete: {len(result['segments'])} segments, "
        f"{len(result['words'])} words, {result['duration']:.1f}s "
        f"(language {result['language']})"
    )
    if status_callback:
        status_callback(
            f"Transcription complete: {len(result['words'])} words across "
            f"{result['duration'] / 60:.1f} minutes"
        )

    # NOTE: intentionally no "transcribed_audio_path" — we sent the RAW file
    # uncompressed, so teaser timestamps align to raw_audio_path. The
    # orchestrator falls back to raw audio when this key is absent.
    return result


def _normalize_local(data: dict) -> dict:
    """Map the mlx-whisper service JSON onto the shared normalized dict.

    Service shape: {"text", "language", "segments": [{"start","end","text",
    ...maybe no_speech_prob/avg_logprob..., "words": [{"word","start","end"}]}]}
    Word timestamps are nested inside segments, so we flatten them to a
    top-level "words" list (which downstream relies on).
    """
    segments = []
    words = []

    for seg in data.get("segments", []) or []:
        if seg.get("start") is None or seg.get("end") is None:
            continue
        seg_out = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": (seg.get("text") or "").strip(),
        }
        for key in _SEGMENT_EXTRA_FIELDS:
            if key in seg and seg[key] is not None:
                seg_out[key] = seg[key]
        segments.append(seg_out)

        for w in seg.get("words", []) or []:
            w_start = w.get("start")
            w_end = w.get("end")
            if w_start is None or w_end is None:
                continue
            words.append({
                "start": float(w_start),
                "end": float(w_end),
                "word": (w.get("word") or "").strip(),
            })

    full_text = (data.get("text") or "").strip()
    if not full_text:
        full_text = " ".join(s["text"] for s in segments).strip()

    # The service may not return a duration — derive it from the last timestamp.
    duration = data.get("duration")
    if not duration:
        ends = [w["end"] for w in words] + [s["end"] for s in segments]
        duration = max(ends) if ends else 0.0

    return {
        "full_text": full_text,
        "segments": segments,
        "words": words,
        "duration": float(duration),
        "language": data.get("language", "en"),
    }


# ── Standalone smoke test ─────────────────────────────────────────────────────
# Usage: python -m pipeline.transcription /path/to/sermon_audio.wav
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.transcription <audio_path>")
        print(f"Backend: {config.TRANSCRIBE_BACKEND}")
        if (config.TRANSCRIBE_BACKEND or "").lower() == "local":
            print(f"Local URL: {config.WHISPER_LOCAL_URL!r}")
            try:
                print(f"Health: {check_health()}")
            except Exception as e:  # noqa: BLE001
                print(f"Health check FAILED: {e}")
        sys.exit(1)

    audio = sys.argv[1]
    print(f"Backend={config.TRANSCRIBE_BACKEND}  audio={audio}")
    out = transcribe(audio, status_callback=lambda m: print(f"  [status] {m}"))
    print(f"\nlanguage : {out['language']}")
    print(f"duration : {out['duration']:.1f}s ({out['duration'] / 60:.1f} min)")
    print(f"segments : {len(out['segments'])}")
    print(f"words    : {len(out['words'])}")
    print(f"full_text[:200]: {out['full_text'][:200]}")
    if out["words"]:
        w0, wN = out["words"][0], out["words"][-1]
        print(f"first word @ {w0['start']:.2f}s: {w0['word']!r}")
        print(f"last  word @ {wN['end']:.2f}s: {wN['word']!r}")
    if out["segments"]:
        s0 = out["segments"][0]
        extras = {k: s0[k] for k in _SEGMENT_EXTRA_FIELDS if k in s0}
        print(f"segment[0] extras present: {extras if extras else '(none)'}")
