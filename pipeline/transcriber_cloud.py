"""
Module 2b: Cloud Transcription (OpenAI Whisper API)
Alternative to local faster-whisper — faster, costs ~$0.50/sermon.
"""
import os
import logging
import json
import requests

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"


def transcribe(audio_path: str, status_callback=None) -> dict:
    """
    Transcribe audio using OpenAI's Whisper API.
    Returns same format as local transcriber for drop-in compatibility.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        status_callback: Optional callable for status updates

    Returns:
        dict with keys: full_text, segments, words, duration, language
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to your .env file."
        )

    if status_callback:
        status_callback("Uploading audio to OpenAI Whisper API...")

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info(f"Uploading {file_size_mb:.1f} MB to OpenAI Whisper API")

    # Compress to mono OGG/Opus for upload — WAV files are too large for OpenAI's 25MB limit.
    # Using Opus instead of MP3 because Opus has NO encoder delay, so
    # timestamps from Whisper align exactly with the source audio.
    # MP3 introduces frame-padding delays that cause timestamp drift.
    if status_callback:
        status_callback("Compressing audio for upload...")
    import subprocess
    compressed = audio_path.replace(".wav", "_compressed.ogg")
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-acodec", "libopus", "-b:a", "48k",
        "-ac", "1", "-vn",
        compressed,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compression failed: {result.stderr}")
    upload_path = compressed
    new_size = os.path.getsize(upload_path) / (1024 * 1024)
    logger.info(f"Compressed to OGG/Opus: {new_size:.1f} MB")

    # If still over 25MB, reduce bitrate
    if new_size > 24:
        compressed2 = audio_path.replace(".wav", "_compressed2.ogg")
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-acodec", "libopus", "-b:a", "24k",
            "-ac", "1", "-vn",
            compressed2,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Second compression failed: {result.stderr}")
        upload_path = compressed2
        new_size = os.path.getsize(upload_path) / (1024 * 1024)
        logger.info(f"Re-compressed to {new_size:.1f} MB")

    if status_callback:
        status_callback("Transcribing via OpenAI (typically 3-5 minutes)...")

    # Request word-level AND segment-level timestamps via verbose_json
    # Use list of tuples so requests sends the key twice (required for arrays)
    with open(upload_path, "rb") as f:
        response = requests.post(
            OPENAI_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (os.path.basename(upload_path), f)},
            data=[
                ("model", "whisper-1"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ],
        )

    if response.status_code != 200:
        error_detail = response.text
        raise RuntimeError(
            f"OpenAI API error ({response.status_code}): {error_detail}"
        )

    data = response.json()

    # Build segments from the response
    segments = []
    if "segments" in data:
        for seg in data["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })

    # Build words list
    words = []
    if "words" in data:
        for w in data["words"]:
            words.append({
                "start": w["start"],
                "end": w["end"],
                "word": w["word"].strip(),
            })

    full_text = data.get("text", "").strip()
    duration = data.get("duration", 0.0)
    language = data.get("language", "en")

    logger.info(
        f"Cloud transcription complete: {len(segments)} segments, "
        f"{len(words)} words, {duration:.1f}s"
    )

    if status_callback:
        status_callback(
            f"Transcription complete: {len(words)} words across "
            f"{duration / 60:.1f} minutes"
        )

    return {
        "full_text": full_text,
        "segments": segments,
        "words": words,
        "duration": duration,
        "language": language,
        "transcribed_audio_path": upload_path,
    }
