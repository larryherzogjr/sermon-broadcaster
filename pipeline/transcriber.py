"""
Module 2: Audio Transcription
Uses faster-whisper to transcribe audio with word-level timestamps.
"""
import logging
from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)

# Cache the model globally so it only loads once
_model = None


def _get_model() -> WhisperModel:
    """Load or return cached Whisper model."""
    global _model
    if _model is None:
        logger.info(
            f"Loading Whisper model '{config.WHISPER_MODEL_SIZE}' "
            f"(device={config.WHISPER_DEVICE}, compute={config.WHISPER_COMPUTE_TYPE})..."
        )
        _model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded.")
    return _model


def transcribe(audio_path: str, status_callback=None) -> dict:
    """
    Transcribe audio file with word-level timestamps.

    Args:
        audio_path: Path to WAV audio file
        status_callback: Optional callable for status updates

    Returns:
        dict with keys:
            - "full_text": Complete transcript as a single string
            - "segments": List of segment dicts with keys:
                - "start": float (seconds)
                - "end": float (seconds)
                - "text": str
            - "words": List of word dicts with keys:
                - "start": float (seconds)
                - "end": float (seconds)
                - "word": str
    """
    if status_callback:
        status_callback("Loading transcription model (this may take a moment on first run)...")

    model = _get_model()

    if status_callback:
        status_callback("Transcribing audio (this takes a while on CPU, be patient)...")

    segments_gen, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )

    logger.info(
        f"Detected language: {info.language} (probability {info.language_probability:.2f})"
    )

    segments = []
    words = []
    full_text_parts = []

    for segment in segments_gen:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        }
        segments.append(seg_data)
        full_text_parts.append(segment.text.strip())

        if segment.words:
            for w in segment.words:
                words.append(
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word.strip(),
                    }
                )

    full_text = " ".join(full_text_parts)
    logger.info(
        f"Transcription complete: {len(segments)} segments, {len(words)} words, "
        f"{info.duration:.1f}s audio"
    )

    if status_callback:
        status_callback(
            f"Transcription complete: {len(words)} words across "
            f"{info.duration / 60:.1f} minutes"
        )

    return {
        "full_text": full_text,
        "segments": segments,
        "words": words,
        "duration": info.duration,
        "language": info.language,
    }
