"""
Module 6: Teaser Selector
Uses Claude API to pick a compelling 12-15 second sermon excerpt for the radio intro.
"""
import json
import logging
from anthropic import Anthropic

import config

logger = logging.getLogger(__name__)

TEASER_PROMPT = """You are choosing a short audio TEASER from a sermon transcript. It plays during the radio intro and has one job: make a listener stop and want to hear the whole sermon.

WHAT MAKES A GREAT TEASER — pick the single most compelling moment:
- A vivid image, story, or illustration at its peak.
- A provocative question, or a question-and-answer that lands.
- A striking, bold, or surprising claim.
- A moment of tension, conviction, or resolution.
- Speaks TO the listener — "you", "we", "us" — not abstract exposition.
Pick the passage you'd put on a billboard for this sermon. Favor heat over tidiness.

LENGTH — this matters:
- 13-20 seconds of speech. That is normally 2-4 sentences that BUILD — a setup and a payoff — NOT a single short sentence.
- Do NOT pick a lone one-liner; on air it plays as an abrupt fragment. If your favorite line is short, include the surrounding sentences that complete the thought.
- Never under 13 seconds. 20 seconds maximum.

AVOID:
- Dry exposition, throat-clearing, or mid-illustration fragments that need prior context to make sense.
- Plain Bible-verse reading — prefer the pastor's own words.
- The very opening or the very end of the sermon.

HOW TO QUOTE IT (so the system can find the audio):
- Copy the passage from the transcript AS IT APPEARS — including any transcription errors, odd phrasing, or repeated words. Do NOT clean it up, fix grammar, or paraphrase.
- We locate the clip by matching your quote against the transcript, then rebuild the displayed text from the audio itself — so an exact quote is only needed to FIND the passage, not for the final wording.
- Make sure your quote (especially its first sentence) appears word-for-word in the transcript above.

Respond ONLY with a JSON object (no markdown, no extra text):
{
    "teaser_text": "The passage copied from the transcript above — 2-4 sentences, ~13-20 seconds of speech, building to a payoff.",
    "reason": "One sentence: why this is the most compelling hook."
}"""


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


def _reconstruct_text_from_span(words: list, start_time: float, end_time: float,
                                tol: float = 0.05) -> str:
    """Rebuild teaser text from the words actually inside the clip span.

    Guarantees the stored/displayed text matches the extracted audio, regardless
    of which match path ran or whether the duration cap trimmed the end.
    """
    span = [
        w["word"] for w in words
        if w["start"] >= start_time - tol and w["end"] <= end_time + tol
    ]
    return " ".join(span).strip()


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

    # Build transcript of just the sermon portion.
    # Use a buffer to avoid the first/last portion of the sermon (opening
    # greetings or closing wrap-up rarely make compelling teasers).
    # Buffer scales with sermon length: 2 min for long sermons, less for short.
    sermon_duration = sermon_end - sermon_start
    if sermon_duration < 600:  # < 10 min — tiny buffer
        buffer = 30
    elif sermon_duration < 1200:  # 10-20 min — moderate buffer
        buffer = 60
    else:  # 20+ min — full 2 min buffer
        buffer = 120

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
                model=config.CLAUDE_MODEL_TEASER,
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
        # Target: fit within the intro teaser window (typically 23s wide).
        # Cap at 22s with a ~1s safety margin. The window is a HARD ceiling —
        # the assembler fade-cuts anything longer — so when a selection overruns
        # we TRIM it back to a complete thought; we never extend past the budget.
        MAX_DURATION = 22.0
        MIN_DURATION = 12.0
        TARGET_DURATION = 16.0

        def _ends_sentence(word_str):
            w = word_str.strip()
            return w.endswith((".", "?", "!"))

        if raw_duration > MAX_DURATION:
            clip_words = [w for w in words if start_time <= w["start"] <= end_time]
            budget_end = start_time + MAX_DURATION

            sentence_ends = [
                w for w in clip_words
                if w["end"] <= budget_end and _ends_sentence(w["word"])
            ]

            if sentence_ends:
                long_enough = [
                    w for w in sentence_ends
                    if (w["end"] - start_time) >= MIN_DURATION
                ]
                # Prefer the latest complete sentence that's also long enough;
                # otherwise take the latest complete sentence (a slightly short
                # but COMPLETE thought beats a 22s mid-sentence truncation).
                chosen = long_enough[-1] if long_enough else sentence_ends[-1]
                end_time = chosen["end"] + 0.2
                log_at = logger.info if long_enough else logger.warning
                log_at(
                    f"Teaser trimmed to SENTENCE boundary: {start_time:.1f}s - "
                    f"{end_time:.1f}s ({end_time - start_time:.1f}s, ends at "
                    f"'{chosen['word']}')"
                )
            else:
                # No complete sentence fits the budget (rare: first sentence is
                # itself > 22s). Trim to the last word boundary; the text is
                # re-derived below so audio and text still stay in sync.
                trimmed_end_word = None
                for w in clip_words:
                    if w["end"] <= budget_end:
                        trimmed_end_word = w
                    else:
                        break
                if trimmed_end_word:
                    end_time = trimmed_end_word["end"] + 0.2
                    logger.warning(
                        f"Teaser too long; no sentence boundary fits "
                        f"{MAX_DURATION:.0f}s — trimmed to word boundary at "
                        f"'{trimmed_end_word['word']}' ({end_time - start_time:.1f}s)"
                    )
        elif raw_duration < MIN_DURATION:
            # Too short — Claude picked a one-liner (or matching located only
            # the first sentence). Extend the END forward across following
            # sentence boundaries to reach the target length, without exceeding
            # MAX_DURATION. The displayed text is re-derived from the final span
            # below, so audio and text stay in sync.
            budget_end = start_time + MAX_DURATION
            extend_to = None
            for w in words:
                if w["start"] < end_time - 0.05:
                    continue  # still inside / before the current clip
                if w["end"] > budget_end:
                    break     # past the 22s ceiling
                if _ends_sentence(w["word"]):
                    extend_to = w
                    if (w["end"] - start_time) >= TARGET_DURATION:
                        break  # reached target on a clean sentence boundary

            if extend_to is not None and extend_to["end"] > end_time:
                new_end = extend_to["end"] + 0.2
                logger.info(
                    f"Teaser too short ({raw_duration:.1f}s); extended to "
                    f"SENTENCE boundary: {start_time:.1f}s - {new_end:.1f}s "
                    f"({new_end - start_time:.1f}s, ends at '{extend_to['word']}')"
                )
                end_time = new_end
            else:
                logger.warning(
                    f"Teaser is short ({raw_duration:.1f}s) and no sentence "
                    f"boundary fits within {MAX_DURATION:.0f}s to extend into — "
                    f"leaving as is"
                )

        result["teaser_start"] = start_time
        result["teaser_end"] = end_time

        # CRITICAL: re-derive teaser_text from the words actually inside the
        # final clip span. The duration trim (and prefix/window matching) can
        # move end_time without touching Claude's original text — that mismatch
        # is what stranded the listener mid-sentence. Re-deriving guarantees the
        # stored/displayed text equals the extracted audio.
        original_text = teaser_text
        reconstructed = _reconstruct_text_from_span(words, start_time, end_time)
        if reconstructed:
            result["teaser_text"] = reconstructed
            if reconstructed.strip() != original_text.strip():
                logger.info(
                    "Teaser text re-derived from clip span to match audio.\n"
                    f"  Claude selected: {original_text[:120]}\n"
                    f"  Clip contains  : {reconstructed[:120]}"
                )

        duration = end_time - start_time
        logger.info(
            f"Teaser final: {start_time:.1f}s - {end_time:.1f}s "
            f"({duration:.1f}s) | text-end derived from clip span: "
            f"{result['teaser_text'][:80]}..."
        )

        if status_callback:
            status_callback(
                f"Teaser selected ({duration:.0f}s): \"{result['teaser_text'][:60]}...\""
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

            # If we matched a SMALL prefix (e.g., 17 of 48 words), Claude likely
            # paraphrased the rest. The end position came from word count, which
            # is unreliable. Snap forward to the next sentence-ending punctuation
            # so we end at a natural boundary rather than mid-sentence.
            if prefix_len < target_len * 0.7:
                raw_words_in_stream = [w["word"] for w in sermon_words]

                def is_sentence_end(idx):
                    if idx < 0 or idx >= len(raw_words_in_stream):
                        return False
                    w = raw_words_in_stream[idx].strip()
                    return w.endswith(".") or w.endswith("?") or w.endswith("!")

                start_time_check = sermon_words[start_idx]["start"]
                target_max_duration = 22.0

                # Look forward from end_idx for sentence-ending punctuation
                lookahead_max = min(len(sermon_words),
                                    end_idx + 30)
                for j in range(end_idx, lookahead_max):
                    if is_sentence_end(j):
                        # Check duration constraint
                        if sermon_words[j]["end"] - start_time_check <= target_max_duration:
                            end_idx = j
                            logger.info(
                                f"Prefix match was partial ({prefix_len}/{target_len} "
                                f"words); snapped end to sentence boundary at "
                                f"'{sermon_words[j]['word']}'"
                            )
                            break

            start_time = sermon_words[start_idx]["start"]
            end_time = sermon_words[end_idx]["end"] + 0.2
            logger.info(
                f"Teaser matched via PREFIX ({prefix_len}/{target_len} words): "
                f"'{sermon_words[start_idx]['word']}' -> '{sermon_words[end_idx]['word']}'"
            )
            return start_time, end_time, "prefix"

    # STRATEGY 3: Window search (last resort - unreliable)
    # When Claude paraphrases, the start/end positions can be way off.
    # We try to find natural sentence boundaries near the matched window
    # so we at least don't cut mid-word or mid-sentence.
    logger.warning("Prefix match failed, trying window search")

    # Build a list of words with punctuation info (preserve original word
    # with its punctuation for sentence-end detection)
    raw_words_in_stream = [w["word"] for w in sermon_words]

    def is_sentence_end(idx):
        """Check if the word at idx ends with sentence-ending punctuation."""
        if idx < 0 or idx >= len(raw_words_in_stream):
            return False
        w = raw_words_in_stream[idx].strip()
        return w.endswith(".") or w.endswith("?") or w.endswith("!")

    def find_next_sentence_end(start_idx, max_lookahead=40):
        """Find the next sentence boundary forward from start_idx."""
        end = min(len(raw_words_in_stream), start_idx + max_lookahead)
        for i in range(start_idx, end):
            if is_sentence_end(i):
                return i
        return None

    def find_prev_sentence_start(end_idx, max_lookback=40):
        """Find the start of the current sentence (just after the previous
        sentence's ending punctuation)."""
        start = max(0, end_idx - max_lookback)
        for i in range(end_idx - 1, start - 1, -1):
            if is_sentence_end(i):
                return i + 1
        return max(0, end_idx - max_lookback)

    for window_size in [5, 4]:
        for offset in range(target_len - window_size + 1):
            window = target_words[offset:offset + window_size]
            match_idx = find_sequence(window)
            if match_idx is not None:
                # Naive estimate first
                naive_start = max(0, match_idx - offset)
                naive_end = min(len(sermon_words) - 1,
                                match_idx + (target_len - offset) - 1)

                # Snap start back to the beginning of its sentence
                est_start_idx = find_prev_sentence_start(naive_start + 1, max_lookback=30)

                # Snap end forward to the next sentence-ending punctuation,
                # but cap the duration at ~22 seconds worth of words
                target_max_duration = 22.0
                est_start_time = sermon_words[est_start_idx]["start"]
                next_end = find_next_sentence_end(naive_end, max_lookahead=30)
                if next_end is not None:
                    # Check duration constraint
                    if sermon_words[next_end]["end"] - est_start_time <= target_max_duration:
                        est_end_idx = next_end
                    else:
                        # Too long — use naive end and accept slight mid-sentence
                        est_end_idx = naive_end
                else:
                    est_end_idx = naive_end

                start_time = sermon_words[est_start_idx]["start"]
                end_time = sermon_words[est_end_idx]["end"] + 0.2
                logger.info(
                    f"Teaser matched via WINDOW ({window_size} words at target offset {offset}): "
                    f"'{sermon_words[est_start_idx]['word']}' -> "
                    f"'{sermon_words[est_end_idx]['word']}' "
                    f"(snapped to sentence boundaries)"
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
