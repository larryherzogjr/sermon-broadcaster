"""Select the sermon between the final two substantial sung hymns.

Music establishes the range. The transcript can only shorten its ending to a
closing Amen before the hymn announcement; it never chooses the sermon itself.
"""
import logging
import math
import re

import config
from pipeline.music_detector import MusicDetectionError, detect_music_events

logger = logging.getLogger(__name__)


class BoundaryDetectionError(RuntimeError):
    """The source needs manual selection instead of an automatic suggestion."""


def _tokens(transcript: dict, start: float, end: float) -> list:
    """Use actual word times only; segment times cannot locate an exact Amen."""
    tokens = []
    for word in transcript.get("words", []):
        try:
            a, b = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(a) and math.isfinite(b) and start <= a <= b <= end):
            continue
        text = str(word.get("word", "")).lower().replace("’", "'")
        for token in re.findall(r"[a-z]+(?:'[a-z]+)?", text):
            tokens.append({"text": token, "start": a, "end": b})
    return sorted(tokens, key=lambda word: (word["start"], word["end"]))


def _is_hymn_announcement(tokens: list) -> bool:
    # Keep matching within one short utterance, never across a silence or hymn.
    phrase = []
    for word in tokens[:18]:
        if phrase and word["start"] - phrase[-1]["end"] > 3:
            break
        phrase.append(word)
    text = " ".join(word["text"] for word in phrase)
    return bool(re.match(
        r"(?:(?:please|let us|let's|would you) (?:stand|rise)\b.*\b(?:sing|hymn)\b"
        r"|(?:our|the) (?:closing|final|next) hymn\b"
        r"|(?:let us|let's|please|we will|we'll) sing\b"
        r"|(?:turn|open)\b.*\bhymn(?:als)?\b)", text
    ))


def closing_prayer_end(transcript: dict, start: float, hymn_start: float):
    """Return an Amen cut, or None to retain the before-music fallback.

    Only examine the last two minutes before the final hymn. Accept an Amen
    when the next utterance is a hymn announcement (within 30 seconds), or
    when Amen is the final transcribed word before the music. An earlier Amen
    followed by more teaching is deliberately left alone.
    """
    words = _tokens(transcript, max(start, hymn_start - config.HYMN_END_SEARCH_SECONDS), hymn_start)
    for index in range(len(words) - 1, -1, -1):
        amen = words[index]
        if amen["text"] != "amen":
            continue
        following = words[index + 1:]
        if following:
            if following[0]["start"] - amen["end"] > 30:
                continue
            if not _is_hymn_announcement(following):
                continue
            ceiling = following[0]["start"]
        else:
            if hymn_start - amen["end"] > 30:
                continue
            ceiling = hymn_start
        # Preserve the word and a little trailing room without taking any of
        # the announcement. Do not move a cut into the final hymn.
        return min(amen["end"] + 0.3, ceiling, hymn_start)
    return None


def select_sermon_range(events: list, transcript_data: dict, audio_duration: float) -> dict:
    """Apply the service-order rule, independent of broadcast target length."""
    hymns = sorted((event for event in events if event["is_hymn"]), key=lambda event: event["start"])
    if len(hymns) < 2:
        raise BoundaryDetectionError(
            "Fewer than two sung hymns of at least "
            f"{config.HYMN_MIN_DURATION_SECONDS:g} seconds were found. "
            "Select the sermon start and end manually."
        )
    preceding, following = hymns[-2:]
    start, music_start = float(preceding["end"]), float(following["start"])
    if not (0 <= start < music_start <= audio_duration):
        raise BoundaryDetectionError("The final two hymns do not bracket a usable sermon range.")
    speech_start = preceding.get("following_speech_start")
    if speech_start is not None and start <= speech_start < min(
        music_start, start + config.HYMN_SPEECH_SEARCH_SECONDS
    ):
        start = max(start, speech_start - config.HYMN_SPEECH_LEAD_SECONDS)
        start_method = "first_speech_after_hymn"
        opening_reason = "Starts just before the first speech after the second-to-last hymn"
    else:
        start_method = "after_hymn"
        opening_reason = "Starts at the detected hymn end; check for a ringing final chord"
    amen_end = closing_prayer_end(transcript_data, start, music_start)
    end = amen_end if amen_end is not None else music_start
    reason = (
        f"{opening_reason}; ends after the closing Amen."
        if amen_end is not None else
        f"{opening_reason}; ends before the final hymn's music. "
        "Check the ending for a hymn announcement."
    )
    return {
        "sermon_start": start,
        "sermon_end": end,
        "confidence": "suggested",
        "sermon_title_guess": "Sermon",
        "selection_label": "between final two hymns",
        "selection_reason": reason,
        "start_method": start_method,
        "end_method": "closing_amen" if amen_end is not None else "before_hymn",
        "music_events": events,
        "preceding_hymn": preceding,
        "following_hymn": following,
    }


def detect_boundaries(audio_path: str, transcript_data: dict, status_callback=None) -> dict:
    if status_callback:
        status_callback("Finding the final two substantial sung hymns...")
    try:
        events, duration = detect_music_events(audio_path, status_callback)
    except MusicDetectionError as exc:
        raise BoundaryDetectionError(str(exc)) from exc
    result = select_sermon_range(events, transcript_data, duration)
    logger.info("Hymn-based sermon range: %.2fs–%.2fs (%s)",
                result["sermon_start"], result["sermon_end"], result["end_method"])
    if status_callback:
        status_callback(result["selection_reason"])
    return result
