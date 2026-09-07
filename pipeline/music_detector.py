"""Local YAMNet audio classification and grouping into substantial sung hymns.

The fixed ONNX export includes Google's waveform frontend. Model provenance,
license, and download instructions are documented in docs/hymn-detection.md.
"""
import hashlib
import logging
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np
import requests
import soundfile as sf

import config

logger = logging.getLogger(__name__)
MODEL_REVISION = "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd"
MODEL_URL = f"https://huggingface.co/audiomagic/yamnet-onnx/resolve/{MODEL_REVISION}/yamnet.onnx"
MODEL_SHA256 = "d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e"
SAMPLE_RATE = 16000
HOP_SAMPLES = 7680  # YAMNet advances by 0.48 seconds.
PATCH_SAMPLES = 15600  # 0.975 seconds, including the final STFT window.
CHUNK_HOPS = 64
# Indices in Google's yamnet_class_map.csv, for the pinned 521-class model.
SPEECH, SINGING, CHOIR, MUSIC = 0, 24, 25, 132
_model_lock = threading.Lock()
_session = None
_session_path = None


class MusicDetectionError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_model() -> str:
    """Download once into application state; verify before loading any model."""
    path = Path(config.HYMN_MODEL_PATH)
    try:
        if path.exists():
            if _digest(path) != MODEL_SHA256:
                raise MusicDetectionError(
                    f"The hymn detector model at {path} failed verification. "
                    "Remove that file and run python -m pipeline.music_detector to reinstall it."
                )
            return str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temporary file plus atomic rename also handles separate workers
        # installing the same pinned model concurrently.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as output:
                temporary = Path(output.name)
                with requests.get(MODEL_URL, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            if _digest(temporary) != MODEL_SHA256:
                raise MusicDetectionError("The downloaded hymn detector model failed verification.")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return str(path)
    except (OSError, requests.RequestException) as exc:
        raise MusicDetectionError(
            "The hymn detector model could not be loaded. Run "
            "python -m pipeline.music_detector with network access, or select the sermon manually."
        ) from exc


def _get_session(status_callback=None):
    global _session, _session_path
    with _model_lock:
        if _session is not None and _session_path == config.HYMN_MODEL_PATH:
            return _session
        if status_callback:
            status_callback("Loading the hymn detector (first use downloads a 16 MB model)...")
        model_path = install_model()
        try:
            import onnxruntime as ort
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                model_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise MusicDetectionError(
                "The local hymn detector could not start. Check the onnxruntime installation "
                "or select the sermon manually."
            ) from exc
        _session, _session_path = session, config.HYMN_MODEL_PATH
        return session


def group_music_frames(frames: list) -> list:
    """Bridge brief verse gaps, then reject short responses and instrumentals.

    Singing must appear in at least 20% of the detected music. This includes an
    instrumental hymn introduction, but excludes instrumental preludes, special
    music, or communion music even when they occur after the last sung hymn.
    """
    events = []
    for frame in frames:
        vocal_score = max(frame["singing"], frame["choir"])
        music_score = max(frame["music"], vocal_score)
        if music_score < 0.5 or music_score <= frame["speech"]:
            continue
        if not events or frame["start"] - events[-1]["end"] > config.HYMN_MAX_GAP_SECONDS:
            events.append({"start": frame["start"], "end": frame["end"],
                           "music_seconds": 0.0, "singing_seconds": 0.0})
        event = events[-1]
        event["end"] = frame["end"]
        duration = frame["end"] - frame["start"]
        event["music_seconds"] += duration
        if vocal_score >= 0.2:
            event["singing_seconds"] += duration
    for event in events:
        duration = event["end"] - event["start"]
        event["is_hymn"] = bool(
            duration >= config.HYMN_MIN_DURATION_SECONDS
            and event["music_seconds"] >= duration * 0.6
            and event["singing_seconds"] >= event["music_seconds"] * 0.2
        )
        for key in ("start", "end", "music_seconds", "singing_seconds"):
            event[key] = round(event[key], 3)
    for index, event in enumerate(events):
        if not event["is_hymn"]:
            continue
        # A ringing final chord can drop below the music threshold while it
        # remains audible. Locate the first nearby speech instead of adding a
        # fixed delay that could cut off a pastor who starts immediately.
        search_end = event["end"] + config.HYMN_SPEECH_SEARCH_SECONDS
        if index + 1 < len(events):
            search_end = min(search_end, events[index + 1]["start"])
        event["following_speech_start"] = next((
            frame["start"] for frame in frames
            if event["end"] <= frame["start"] < search_end
            and frame["speech"] >= 0.5
            and frame["speech"] > max(frame["music"], frame["singing"], frame["choir"])
        ), None)
    return events


def _score_wav(path: str, session, status_callback=None) -> tuple:
    """Bound memory to about 31 seconds of audio, preserving the global hop grid."""
    frames = []
    with sf.SoundFile(path) as audio:
        if audio.samplerate != SAMPLE_RATE or audio.channels != 1:
            raise MusicDetectionError("Hymn analysis requires 16 kHz mono audio.")
        duration = len(audio) / SAMPLE_RATE
        stride = HOP_SAMPLES * CHUNK_HOPS
        chunk_size = stride + PATCH_SAMPLES - HOP_SAMPLES
        for offset in range(0, len(audio), stride):
            audio.seek(offset)
            data = audio.read(chunk_size, dtype="float32")
            expected = min(CHUNK_HOPS, math.ceil((len(audio) - offset) / HOP_SAMPLES))
            # The final partial chunk needs right padding to score every real
            # audio cell. Padded cells themselves are never returned.
            needed = expected * HOP_SAMPLES + PATCH_SAMPLES - HOP_SAMPLES
            if len(data) < needed:
                data = np.pad(data, (0, needed - len(data)))
            try:
                scores = session.run(["output_0"], {"waveform": data})[0][:CHUNK_HOPS]
            except Exception as exc:
                raise MusicDetectionError("The hymn detector could not classify this recording.") from exc
            if scores.ndim != 2 or len(scores) < expected or scores.shape[1] != 521 or not np.isfinite(scores).all():
                raise MusicDetectionError("Hymn analysis returned invalid audio scores.")
            for i, row in enumerate(scores[:expected]):
                start = (offset + i * HOP_SAMPLES) / SAMPLE_RATE
                frames.append({
                    "start": start, "end": min(start + HOP_SAMPLES / SAMPLE_RATE, duration),
                    "music": float(row[MUSIC]), "speech": float(row[SPEECH]),
                    "singing": float(row[SINGING]), "choir": float(row[CHOIR]),
                })
            if status_callback and (offset // stride) % 20 == 0:
                status_callback(f"Checking music: {min(100, int((offset + stride) / len(audio) * 100))}%")
    if not frames:
        raise MusicDetectionError("The recording contains no audio to analyze.")
    return frames, duration


def detect_music_events(audio_path: str, status_callback=None) -> tuple:
    session = _get_session(status_callback)
    try:
        with tempfile.TemporaryDirectory(prefix="sermon-hymns-") as directory:
            mono_path = os.path.join(directory, "mono.wav")
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", audio_path,
                 "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", mono_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=600,
            )
            if result.returncode:
                raise MusicDetectionError(f"Audio conversion for hymn analysis failed: {result.stderr[-500:]}")
            frames, duration = _score_wav(mono_path, session, status_callback)
    except (OSError, sf.LibsndfileError, subprocess.TimeoutExpired) as exc:
        raise MusicDetectionError("Could not decode the recording for hymn analysis.") from exc
    events = group_music_frames(frames)
    logger.info("Found %d substantial sung hymns in %d music events",
                sum(event["is_hymn"] for event in events), len(events))
    return events, duration


if __name__ == "__main__":
    print(f"Hymn detector model ready: {install_model()}")
