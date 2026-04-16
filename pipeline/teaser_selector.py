"""
Module 6: Teaser Selector
Uses Claude API to pick a compelling 12-15 second sermon excerpt for the radio intro.
"""
import json
import logging
from anthropic import Anthropic

import config

logger = logging.getLogger(__name__)

TEASER_PROMPT = """You are selecting a short audio teaser clip from a sermon transcript for a radio broadcast intro. The teaser plays during the introduction to hook listeners.

DURATION:
The teaser should be 13-20 seconds of spoken audio. Pick a COMPLETE, self-contained thought that naturally falls within that range — don't pad it out, and don't cut a powerful passage short just to save a few seconds. 20 seconds maximum.

Content requirements:
- Must be a single, self-contained thought — makes sense without context.
- Should be COMPELLING — a vivid illustration, powerful statement, provocative question, or key insight.
- Should make the listener want to hear the full sermon.
- Must start at the beginning of a sentence and end at the end of a sentence.
- Prefer the pastor's own teaching over direct Bible verse reading.
- Should NOT be from the very beginning or very end of the sermon.

Respond ONLY with a JSON object (no markdown, no extra text):
{
    "teaser_text": "The EXACT verbatim words from the transcript. Copy directly without paraphrasing.",
    "reason": "Brief explanation of why this clip was chosen"
}

CRITICAL: teaser_text must be copied VERBATIM from the transcript (character-for-character). The system will find the timestamps by searching for this exact text."""


def _snap_to_word_boundaries(teaser: dict, words: list) -> dict:
    """
    Snap teaser start/end timestamps to exact word boundaries
    so the clip doesn't cut mid-word.
    """
    t_start = teaser["teaser_start"]
    t_end = teaser["teaser_end"]

    # Find the word nearest to (but not after) the teaser start
    best_start_word = None
    for w in words:
        if w["start"] >= t_start - 1.0 and w["start"] <= t_start + 1.0:
            if best_start_word is None or abs(w["start"] - t_start) < abs(best_start_word["start"] - t_start):
                best_start_word = w

    # Find the word nearest to (but not before) the teaser end
    best_end_word = None
    for w in words:
        if w["end"] >= t_end - 1.0 and w["end"] <= t_end + 1.0:
            if best_end_word is None or abs(w["end"] - t_end) < abs(best_end_word["end"] - t_end):
                best_end_word = w

    if best_start_word:
        old_start = teaser["teaser_start"]
        teaser["teaser_start"] = best_start_word["start"] - 0.05  # tiny pre-roll
        logger.info(
            f"Teaser start snapped: {old_start:.2f}s -> {teaser['teaser_start']:.2f}s "
            f"(word: '{best_start_word['word']}')"
        )

    if best_end_word:
        old_end = teaser["teaser_end"]
        teaser["teaser_end"] = best_end_word["end"] + 0.15  # small buffer after last word
        logger.info(
            f"Teaser end snapped: {old_end:.2f}s -> {teaser['teaser_end']:.2f}s "
            f"(word: '{best_end_word['word']}')"
        )

    return teaser


def select_teaser(transcript_data: dict, sermon_start: float, sermon_end: float,
                  status_callback=None) -> dict:
    """
    Use Claude to select a compelling teaser clip from the sermon.

    Args:
        transcript_data: Full transcript with segments and words
        sermon_start: Sermon start timestamp (seconds)
        sermon_end: Sermon end timestamp (seconds)
        status_callback: Optional callable for status updates

    Returns:
        dict with teaser_start, teaser_end, teaser_text, reason
    """
    if status_callback:
        status_callback("Selecting teaser clip for radio intro...")

    # Build transcript of just the sermon portion (excluding first/last 2 minutes)
    buffer = 120  # avoid first and last 2 minutes
    seg_start = sermon_start + buffer
    seg_end = sermon_end - buffer

    segments = transcript_data.get("segments", [])
    sermon_segments = [
        s for s in segments
        if s["start"] >= seg_start and s["end"] <= seg_end
    ]

    if not sermon_segments:
        # Fallback: use all sermon segments
        sermon_segments = [
            s for s in segments
            if s["start"] >= sermon_start and s["end"] <= sermon_end
        ]

    timestamped_lines = []
    for seg in sermon_segments:
        minutes = int(seg["start"] // 60)
        seconds = seg["start"] % 60
        timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
        timestamped_lines.append(f"{timestamp} {seg['text']}")

    formatted_transcript = "\n".join(timestamped_lines)

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    MAX_ATTEMPTS = 3
    previous_attempts_note = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            user_message = (
                f"Here is the sermon transcript. "
                f"Sermon runs from {sermon_start:.0f}s to {sermon_end:.0f}s.\n\n"
                f"TRANSCRIPT:\n{formatted_transcript}"
            )
            if previous_attempts_note:
                user_message += f"\n\n{previous_attempts_note}"

            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=512,
                system=TEASER_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            response_text = response.content[0].text.strip()
            logger.info(f"Teaser response (attempt {attempt}): {response_text[:300]}")

            # Parse JSON (handle markdown fencing)
            if "```" in response_text:
                parts = response_text.split("```")
                if len(parts) >= 3:
                    inner = parts[1]
                    if inner.startswith("json"):
                        inner = inner[4:]
                    response_text = inner.strip()

            if not response_text.startswith("{"):
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]

            result = json.loads(response_text)

            teaser_text = result.get("teaser_text", "").strip()
            if not teaser_text:
                raise ValueError("Claude did not return teaser_text")

            words = transcript_data.get("words", [])
            if not words:
                raise ValueError("No word-level timestamps available")

            # Find the text in the word stream — require STRONG prefix match
            start_time, end_time, match_quality = _find_text_in_words(
                teaser_text, words, sermon_start, sermon_end
            )

            if start_time is None:
                logger.warning(f"Attempt {attempt}: could not locate teaser text at all")
                previous_attempts_note = (
                    f"PREVIOUS ATTEMPT FAILED: Your previous selection "
                    f"'{teaser_text[:80]}...' could not be found in the transcript. "
                    f"You MUST copy text VERBATIM from the transcript above. "
                    f"Do not paraphrase, do not combine phrases from different places. "
                    f"Select a DIFFERENT passage and copy it exactly as it appears."
                )
                continue

            # If we fell back to window matching, the result is unreliable
            if match_quality == "window" and attempt < MAX_ATTEMPTS:
                logger.warning(
                    f"Attempt {attempt}: matched via window (unreliable), retrying"
                )
                previous_attempts_note = (
                    f"PREVIOUS ATTEMPT WAS INACCURATE: Your selection "
                    f"'{teaser_text[:80]}...' did not match the transcript exactly. "
                    f"You MUST copy text VERBATIM — letter-for-letter from the transcript. "
                    f"Do not combine phrases from different locations in the sermon. "
                    f"Pick a different passage and copy it EXACTLY."
                )
                continue

            # Good match (or final attempt) — use it
            raw_duration = end_time - start_time
            logger.info(
                f"Teaser accepted (attempt {attempt}, match: {match_quality}): "
                f"{start_time:.1f}s - {end_time:.1f}s ({raw_duration:.1f}s)"
            )
            break

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt < MAX_ATTEMPTS:
                logger.warning(f"Attempt {attempt} failed: {e} — retrying")
                previous_attempts_note = (
                    f"PREVIOUS ATTEMPT FAILED: {str(e)[:100]}. "
                    f"Please respond with valid JSON containing teaser_text that is "
                    f"copied VERBATIM from the transcript."
                )
                continue
            raise

    # At this point `result`, `start_time`, `end_time` are set from the accepted attempt
    try:

        # Target: fit within the intro teaser window (typically 23s wide)
        # Cap at 22s with a 1s safety margin
        MAX_DURATION = 22.0
        MIN_DURATION = 10.0

        if raw_duration > MAX_DURATION:
            # Too long — trim at a natural word boundary
            sermon_words = [w for w in words if start_time <= w["start"] <= end_time]
            # Find the last word that fits within MAX_DURATION
            target_end = start_time + MAX_DURATION
            trimmed_end_word = None
            for w in sermon_words:
                if w["end"] <= target_end:
                    trimmed_end_word = w
                else:
                    break
            if trimmed_end_word:
                end_time = trimmed_end_word["end"] + 0.2
                logger.info(
                    f"Teaser trimmed to word boundary: {start_time:.1f}s - {end_time:.1f}s "
                    f"({end_time - start_time:.1f}s, ends at '{trimmed_end_word['word']}')"
                )
        elif raw_duration < MIN_DURATION:
            logger.warning(
                f"Teaser is short ({raw_duration:.1f}s, target 13s) — "
                f"Claude selected too little text"
            )

        result["teaser_start"] = start_time
        result["teaser_end"] = end_time

        duration = end_time - start_time
        logger.info(
            f"Teaser final: {start_time:.1f}s - {end_time:.1f}s "
            f"({duration:.1f}s): {teaser_text[:80]}..."
        )

        if status_callback:
            status_callback(
                f"Teaser selected ({duration:.0f}s): \"{teaser_text[:60]}...\""
            )

        return result

    except Exception as e:
        logger.error(f"Teaser selection failed: {e}")
        raise


def _find_text_in_words(target_text: str, words: list,
                        sermon_start: float, sermon_end: float):
    """
    Find the start and end timestamps of target_text within the word stream.

    Returns (start_time, end_time, quality) where quality is:
    - "exact": found the complete target phrase
    - "prefix": matched a leading prefix (4+ consecutive words)
    - "window": fell back to finding a smaller window anywhere (unreliable)
    - None, None, None if no match at all
    """
    import re

    def normalize(s):
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    target_words = normalize(target_text).split()
    if not target_words:
        return None, None, None

    sermon_words = [w for w in words if sermon_start <= w["start"] <= sermon_end]
    normalized_stream = [normalize(w["word"]) for w in sermon_words]

    if not sermon_words:
        return None, None, None

    def find_sequence(sequence):
        n = len(sequence)
        for i in range(len(normalized_stream) - n + 1):
            if normalized_stream[i:i + n] == sequence:
                return i
        return None

    target_len = len(target_words)

    # STRATEGY 1: Full exact match
    start_idx = find_sequence(target_words)
    if start_idx is not None:
        end_idx = start_idx + target_len - 1
        start_time = sermon_words[start_idx]["start"]
        end_time = sermon_words[end_idx]["end"] + 0.2
        logger.info(
            f"Teaser matched EXACTLY ({target_len} words): "
            f"'{sermon_words[start_idx]['word']}' -> '{sermon_words[end_idx]['word']}'"
        )
        return start_time, end_time, "exact"

    # STRATEGY 2: Prefix match (4+ words in a row from start of target)
    min_prefix = max(4, target_len // 3)
    for prefix_len in range(target_len - 1, min_prefix - 1, -1):
        start_idx = find_sequence(target_words[:prefix_len])
        if start_idx is not None:
            end_idx = _find_end_position(start_idx, target_words, normalized_stream)
            start_time = sermon_words[start_idx]["start"]
            end_time = sermon_words[end_idx]["end"] + 0.2
            logger.info(
                f"Teaser matched via PREFIX ({prefix_len}/{target_len} words): "
                f"'{sermon_words[start_idx]['word']}' -> '{sermon_words[end_idx]['word']}'"
            )
            return start_time, end_time, "prefix"

    # STRATEGY 3: Window search (last resort - unreliable)
    logger.warning("Prefix match failed, trying window search")
    for window_size in [5, 4]:
        for offset in range(target_len - window_size + 1):
            window = target_words[offset:offset + window_size]
            match_idx = find_sequence(window)
            if match_idx is not None:
                est_start_idx = max(0, match_idx - offset)
                est_end_idx = min(len(sermon_words) - 1, match_idx + (target_len - offset) - 1)
                start_time = sermon_words[est_start_idx]["start"]
                end_time = sermon_words[est_end_idx]["end"] + 0.2
                logger.info(
                    f"Teaser matched via WINDOW ({window_size} words at target offset {offset}): "
                    f"'{sermon_words[est_start_idx]['word']}' -> "
                    f"'{sermon_words[est_end_idx]['word']}'"
                )
                return start_time, end_time, "window"

    return None, None, None


def _find_end_position(start_idx, target_words, normalized_stream):
    """Find the end position given a start position and target word sequence."""
    target_len = len(target_words)
    # Try exact full match
    if start_idx + target_len <= len(normalized_stream):
        if normalized_stream[start_idx:start_idx + target_len] == target_words:
            return start_idx + target_len - 1
    # Look for the last target word within reasonable window
    last_word = target_words[-1]
    search_end = min(start_idx + target_len + 8, len(normalized_stream))
    best = start_idx + min(target_len - 1, search_end - start_idx - 1)
    for j in range(start_idx, search_end):
        if normalized_stream[j] == last_word:
            best = j
    return best
