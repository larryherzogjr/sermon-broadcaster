"""
Module 3: Sermon Boundary Detection
Uses Claude API to analyze the transcript and identify sermon start/end timestamps.
"""
import json
import logging
from anthropic import Anthropic

import config

logger = logging.getLogger(__name__)

BOUNDARY_PROMPT = """You are an expert at analyzing church service transcripts to identify the exact boundaries of the sermon portion for radio broadcast.

You will receive a timestamped transcript of a full church service. Your job is to identify:

1. The SERMON START — the moment the SCRIPTURE READING for the sermon begins.
   The broadcast includes the scripture reading AND the sermon body that follows.
   Just before the scripture reading, the pastor typically asks the congregation to stand:
     - "Please stand for the reading of God's Word"
     - "Let us stand for the reading of God's Word"
     - "Please rise"
   The sermon_start should be set to the START of the scripture reading itself,
   AFTER the "please stand" / "let us stand" cue ends.
   This is NOT:
   - Welcome/announcements
   - Opening prayer (before the scripture reading)
   - Hymns or worship
   - Offering
   - The "please stand" cue itself (we want to skip past this)

   IMPORTANT: Use the timestamp of the BEGINNING of the scripture reading
   (after the standing cue completes). This ensures we capture the first word
   of scripture cleanly without the standing cue.

2. SEATING CUE — between the scripture reading and the sermon body, the pastor
   typically tells the congregation to sit:
     - "You may be seated"
     - "Please be seated"
     - "You may sit down"
     - "Be seated, please"
   Identify this cue's timestamps so the system can splice it out.

   - seating_cue_start: timestamp of the BEGINNING of the seating cue phrase
   - seating_cue_end: timestamp of the END of the seating cue phrase (so audio
     resumes cleanly with the sermon body)

   If no clear seating cue is present (e.g., the pastor flows directly from
   scripture into preaching), set both to null.

3. TWO SERMON END POINTS — you must provide both:

   a) sermon_end_with_prayer — the end of the closing prayer's "Amen."
      - This is the prayer the pastor prays to conclude the sermon.
      - The timestamp should be the END of the segment containing the final "Amen" of this prayer.
      - EXCLUDE anything after: announcements, closing hymns, benediction.

   b) sermon_end_without_prayer — the end of the last substantive sermon sentence BEFORE the prayer transition.
      - This is the final teaching point, application, or concluding thought.
      - EXCLUDE "Let us pray" / "Let's pray" / "Shall we pray" / any prayer language.
      - Example: "...and that is the hope we have in Christ. Let's pray." — this timestamp is the END of the segment containing "Christ."

   The broadcast system will choose which ending to use based on time constraints.

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{
    "sermon_start": 123.45,
    "seating_cue_start": 145.20,
    "seating_cue_end": 147.80,
    "sermon_end_with_prayer": 2400.00,
    "sermon_end_without_prayer": 2345.67,
    "confidence": "high",
    "start_reason": "Brief explanation — quote the first few words of scripture reading in single quotes",
    "seating_cue_reason": "The exact phrase used (or null if no seating cue present)",
    "end_with_prayer_reason": "Brief explanation — what the closing Amen is",
    "end_without_prayer_reason": "The last substantive sentence before prayer transition",
    "sermon_title_guess": "Your best guess at the sermon title/topic based on content"
}

All times are in seconds (float). Confidence should be "high", "medium", or "low".
If no seating cue is present, set seating_cue_start and seating_cue_end to null.
If you cannot identify a sermon in the transcript, respond with:
{
    "error": "Explanation of why sermon boundaries could not be determined"
}"""


def _refine_boundaries(boundaries: dict, words: list, status_callback=None) -> dict:
    """
    Refine Claude's boundaries using word-level timestamps.
    
    Claude's timestamps can be off by a minute or more because it works from
    segment-level timestamps. We use word-level data to find exact cut points.
    """
    sermon_start = boundaries["sermon_start"]
    end_with = boundaries["sermon_end_with_prayer"]
    end_without = boundaries["sermon_end_without_prayer"]
    
    logger.info(
        f"[REFINE] Claude endpoints: start={sermon_start:.1f}, "
        f"end_with={end_with:.1f}, end_without={end_without:.1f}"
    )
    
    # ── Refine START boundary ────────────────────────────────────────
    # Claude's start can land in a hymn or silence before the sermon.
    # Strategy: extract the first words Claude quoted in start_reason,
    # find them in the word-level timestamps, and snap to just before them.
    
    start_refined = False
    start_reason = boundaries.get("start_reason", "")
    
    # Extract quoted text from start_reason (text between single quotes)
    import re
    quoted = re.findall(r"'([^']+)'", start_reason)
    if not quoted:
        # Try double quotes
        quoted = re.findall(r'"([^"]+)"', start_reason)
    
    if quoted:
        # Take the first quoted phrase and extract words for matching
        first_phrase = quoted[0]
        all_search_words = first_phrase.lower().split()[:6]
        
        logger.info(f"[REFINE] Sermon start phrase: '{first_phrase[:50]}...'")
        
        # Search for this sequence in the word timestamps near Claude's start
        search_begin = sermon_start - 60
        search_end = sermon_start + 60
        
        # Try progressively shorter matches: 5 words, 4, 3, then 2
        for match_len in range(min(5, len(all_search_words)), 1, -1):
            search_words = all_search_words[:match_len]
            
            for i, w in enumerate(words):
                if w["start"] < search_begin or w["start"] > search_end:
                    continue
                
                if i + match_len > len(words):
                    continue
                
                candidate = [
                    words[i + j]["word"].lower().strip(".,!?;:'\"")
                    for j in range(match_len)
                ]
                
                target = [sw.strip(".,!?;:'\"") for sw in search_words]
                
                if candidate == target:
                    new_start = words[i]["start"] - 0.3
                    old_start = boundaries["sermon_start"]
                    boundaries["sermon_start"] = new_start
                    start_refined = True
                    logger.info(
                        f"[REFINE] Start matched ({match_len} words): "
                        f"'{' '.join(candidate)}' at {words[i]['start']:.1f}s "
                        f"(was {old_start:.1f}s)"
                    )
                    if status_callback:
                        status_callback(
                            f"Refined start: '{first_phrase[:40]}...' at {words[i]['start']:.0f}s"
                        )
                    break
            
            if start_refined:
                break
        
        if not start_refined:
            # Log what words ARE near Claude's start for debugging
            nearby = [
                w["word"] for w in words
                if sermon_start - 5 <= w["start"] <= sermon_start + 15
            ]
            logger.info(
                f"[REFINE] No match found. Words near start: "
                f"'{' '.join(nearby[:15])}'"
            )
            logger.info("[REFINE] Trying gap detection fallback")
    
    # Fallback: gap-based detection if word matching didn't work
    if not start_refined:
        start_search_begin = sermon_start - 30
        start_search_end = sermon_start + 30
        
        start_region_words = [
            (i, w) for i, w in enumerate(words)
            if start_search_begin <= w["start"] <= start_search_end
        ]
        
        if start_region_words:
            # Look for a silence gap > 1.5s near Claude's start point
            for k in range(len(start_region_words) - 1):
                idx_a, word_a = start_region_words[k]
                idx_b, word_b = start_region_words[k + 1]
                gap = word_b["start"] - word_a["end"]
                
                if gap >= 1.5 and word_b["start"] >= sermon_start - 5:
                    new_start = words[idx_b]["start"] - 0.3
                    old_start = boundaries["sermon_start"]
                    boundaries["sermon_start"] = new_start
                    start_refined = True
                    logger.info(
                        f"[REFINE] Start gap: {gap:.1f}s silence, "
                        f"speech at {word_b['start']:.1f}s ('{word_b['word']}') "
                        f"(was {old_start:.1f}s)"
                    )
                    break
        
        if not start_refined:
            logger.warning("[REFINE] Could not refine start — keeping Claude's endpoint")
    
    # ── Refine END boundaries ────────────────────────────────────────
    # Search the last 40% of the sermon for all markers
    sermon_start = boundaries["sermon_start"]  # use refined start
    earliest_endpoint = min(end_with, end_without)
    search_from = sermon_start + (earliest_endpoint - sermon_start) * 0.6
    
    # --- Find all post-sermon transitions ---
    # These phrases must be specific enough to not match mid-sermon speech.
    # "at this time" alone is too generic — require a follow-up like "let us"
    post_sermon_phrases = [
        "at this time let us", "at this time let's", "at this time we will",
        "let us stand", "let us sing", "stand and sing",
        "closing hymn", "hymn number", "turn to hymn", "turn in your hymnals",
        "page number", "our closing", "stand together", "rise and sing",
        "let's stand", "please stand", "please rise",
    ]
    
    # Only search near Claude's endpoints — transitions more than 2 minutes
    # before the earliest endpoint are almost certainly mid-sermon false positives
    transition_search_from = min(end_with, end_without) - 120
    transition_search_from = max(transition_search_from, search_from)
    
    transitions = []
    for i, w in enumerate(words):
        if w["start"] < transition_search_from:
            continue
        context_words = []
        for j in range(i, min(i + 8, len(words))):
            context_words.append(words[j]["word"].lower().strip(".,!?;:"))
        context = " ".join(context_words)
        for phrase in post_sermon_phrases:
            if context.startswith(phrase):
                transitions.append({"idx": i, "word": w, "phrase": phrase, "context": context})
                logger.info(f"[REFINE] Post-sermon transition at {w['start']:.1f}s: '{context[:50]}'")
                break
    
    # --- Find all "Amen" instances ---
    amens = []
    for i, w in enumerate(words):
        if w["start"] < search_from:
            continue
        if w["word"].lower().strip(".,!?;:") == "amen":
            amens.append({"idx": i, "word": w})
            logger.info(f"[REFINE] 'Amen' at {w['start']:.2f}s")
    
    # --- Find all prayer transitions ---
    prayer_phrases = [
        "let's pray", "let us pray", "shall we pray",
        "would you pray", "bow our heads", "bow your heads",
        "let's bow", "let us bow",
    ]
    
    prayer_transitions = []
    for i, w in enumerate(words):
        if w["start"] < search_from:
            continue
        context_words = []
        for j in range(i, min(i + 5, len(words))):
            context_words.append(words[j]["word"].lower().strip(".,!?;:"))
        context = " ".join(context_words)
        for phrase in prayer_phrases:
            if context.startswith(phrase):
                prayer_transitions.append({"idx": i, "word": w, "context": context})
                logger.info(f"[REFINE] Prayer transition at {w['start']:.1f}s: '{context[:50]}'")
                break
    
    logger.info(
        f"[REFINE] Found: {len(transitions)} post-sermon transitions, "
        f"{len(amens)} Amens, {len(prayer_transitions)} prayer transitions"
    )
    
    # --- Choose the right instances using Claude's rough endpoints ---
    
    # For WITH-PRAYER endpoint: find the Amen closest to (but before) the first
    # post-sermon transition. If no transition found, use the Amen closest to
    # Claude's endpoint.
    if transitions and amens:
        first_transition = transitions[0]
        # Find the last Amen BEFORE the first post-sermon transition
        valid_amens = [a for a in amens if a["word"]["end"] < first_transition["word"]["start"]]
        if valid_amens:
            best_amen = valid_amens[-1]  # last one before transition
            boundaries["sermon_end_with_prayer"] = best_amen["word"]["end"] + 0.5
            logger.info(
                f"[REFINE] With-prayer: snapped to Amen at {best_amen['word']['start']:.1f}s "
                f"(before transition at {first_transition['word']['start']:.1f}s)"
            )
            if status_callback:
                status_callback(
                    f"Refined: prayer ends at Amen ({best_amen['word']['start']:.0f}s), "
                    f"before '{first_transition['phrase']}' ({first_transition['word']['start']:.0f}s)"
                )
        else:
            # No Amen before transition — cut before the transition itself
            cut_word = words[first_transition["idx"] - 1]
            boundaries["sermon_end_with_prayer"] = cut_word["end"] + 0.3
            logger.info(f"[REFINE] No Amen before transition, cutting at {cut_word['end']:.1f}s")
            if status_callback:
                status_callback(f"No Amen found — cutting before '{first_transition['phrase']}'")
    elif amens:
        # No transitions found, use Amen closest to Claude's endpoint
        best_amen = min(amens, key=lambda a: abs(a["word"]["end"] - end_with))
        boundaries["sermon_end_with_prayer"] = best_amen["word"]["end"] + 0.5
        logger.info(f"[REFINE] With-prayer: snapped to Amen at {best_amen['word']['start']:.1f}s")
        if status_callback:
            status_callback(f"Refined: prayer ends at Amen ({best_amen['word']['start']:.0f}s)")
    elif transitions:
        # No Amen found, cut before first transition
        cut_word = words[transitions[0]["idx"] - 1]
        boundaries["sermon_end_with_prayer"] = cut_word["end"] + 0.3
        logger.info(f"[REFINE] No Amen, cutting before transition at {cut_word['end']:.1f}s")
        if status_callback:
            status_callback(f"No Amen — cutting before '{transitions[0]['phrase']}'")
    else:
        logger.warning("[REFINE] No Amen or transitions found — keeping Claude's endpoint")
    
    # For WITHOUT-PRAYER endpoint: use the last word before the first prayer transition
    if prayer_transitions:
        pt = prayer_transitions[0]
        if pt["idx"] > 0:
            last_word = words[pt["idx"] - 1]
            boundaries["sermon_end_without_prayer"] = last_word["end"] + 0.3
            logger.info(
                f"[REFINE] Without-prayer: cut at {last_word['end']:.1f}s, "
                f"before prayer at {pt['word']['start']:.1f}s"
            )
    
    # HARD CEILING: without-prayer must NEVER be after the first post-sermon transition.
    # If it is, the prayer transition we found was actually after the closing hymn,
    # not the sermon-ending prayer.
    # Best cut: snap to Amen before the transition (same as with-prayer logic).
    # Fallback: cut right before the transition itself.
    if transitions:
        first_transition = transitions[0]
        ceiling = first_transition["word"]["start"] - 0.5
        if boundaries["sermon_end_without_prayer"] > ceiling:
            old_end = boundaries["sermon_end_without_prayer"]
            
            # Try to find an Amen before this transition (tightest cut)
            valid_amens = [a for a in amens if a["word"]["end"] < first_transition["word"]["start"]]
            if valid_amens:
                best_amen = valid_amens[-1]
                boundaries["sermon_end_without_prayer"] = best_amen["word"]["end"] + 0.5
                logger.info(
                    f"[REFINE] Without-prayer CLAMPED to Amen: {old_end:.1f}s -> "
                    f"{boundaries['sermon_end_without_prayer']:.1f}s "
                    f"(Amen at {best_amen['word']['start']:.1f}s, before transition at "
                    f"{first_transition['word']['start']:.1f}s)"
                )
            else:
                # No Amen — cut right before the transition
                cut_word = words[first_transition["idx"] - 1] if first_transition["idx"] > 0 else first_transition["word"]
                boundaries["sermon_end_without_prayer"] = cut_word["end"] + 0.3
                logger.info(
                    f"[REFINE] Without-prayer CLAMPED: {old_end:.1f}s -> "
                    f"{boundaries['sermon_end_without_prayer']:.1f}s "
                    f"(before transition at {first_transition['word']['start']:.1f}s)"
                )
            
            if status_callback:
                status_callback(
                    f"Without-prayer clamped before '{first_transition['phrase']}' at "
                    f"{first_transition['word']['start']:.0f}s"
                )
    
    logger.info(
        f"[REFINE] Final: end_with={boundaries['sermon_end_with_prayer']:.1f}, "
        f"end_without={boundaries['sermon_end_without_prayer']:.1f}"
    )

    # ── Refine SEATING CUE if present ────────────────────────────────
    seating_start = boundaries.get("seating_cue_start")
    seating_end = boundaries.get("seating_cue_end")
    if seating_start is not None and seating_end is not None:
        # Search for the seating cue phrase in the word stream
        # Look for variations: "you may be seated", "please be seated",
        # "you may sit", "be seated"
        seating_phrases = [
            ["you", "may", "be", "seated"],
            ["please", "be", "seated"],
            ["you", "may", "sit", "down"],
            ["you", "may", "sit"],
            ["be", "seated", "please"],
            ["be", "seated"],
        ]

        # Search window: from sermon_start to ~5 minutes after, since the
        # seating cue is between the scripture reading and the sermon body
        search_start = boundaries["sermon_start"]
        search_end = min(boundaries["sermon_start"] + 600,
                         boundaries["sermon_end_with_prayer"])

        normalized_words = []
        for w in words:
            if search_start <= w["start"] <= search_end:
                normalized_words.append({
                    "word": w["word"].lower().strip(".,!?;:'\""),
                    "start": w["start"],
                    "end": w["end"],
                })

        found = False
        for phrase in seating_phrases:
            n = len(phrase)
            for i in range(len(normalized_words) - n + 1):
                window = [normalized_words[i + j]["word"] for j in range(n)]
                if window == phrase:
                    # Found it — use these word timestamps
                    refined_start = normalized_words[i]["start"]
                    refined_end = normalized_words[i + n - 1]["end"]
                    # Add a small buffer (100ms) on each side for clean cut
                    refined_start = max(0, refined_start - 0.1)
                    refined_end = refined_end + 0.1
                    boundaries["seating_cue_start"] = refined_start
                    boundaries["seating_cue_end"] = refined_end
                    logger.info(
                        f"[REFINE] Seating cue refined: "
                        f"'{' '.join(phrase)}' at "
                        f"{refined_start:.1f}s - {refined_end:.1f}s"
                    )
                    if status_callback:
                        status_callback(
                            f"Seating cue identified: '{' '.join(phrase)}' "
                            f"at {refined_start:.0f}s - will be removed"
                        )
                    found = True
                    break
            if found:
                break

        if not found:
            logger.warning(
                "[REFINE] Could not locate seating cue phrase in word stream — "
                "using Claude's timestamps as-is"
            )

    return boundaries


def detect_boundaries(transcript_data: dict, status_callback=None) -> dict:
    """
    Use Claude to analyze transcript and find sermon boundaries.
    Returns both with-prayer and without-prayer end points.
    The orchestrator decides which to use based on target duration.
    """
    if status_callback:
        status_callback("Analyzing transcript for sermon boundaries...")

    # Build a timestamped transcript for Claude to analyze
    # Use segments (not individual words) for readability and token efficiency
    timestamped_lines = []

    segments = transcript_data.get("segments", [])

    # Fallback: if no segments, build them from words
    if not segments and transcript_data.get("words"):
        logger.info("No segments found, building from words...")
        chunk = []
        chunk_start = 0.0
        for w in transcript_data["words"]:
            if not chunk:
                chunk_start = w["start"]
            chunk.append(w["word"])
            # Break into ~15-word segments
            if len(chunk) >= 15:
                segments.append({
                    "start": chunk_start,
                    "end": w["end"],
                    "text": " ".join(chunk),
                })
                chunk = []
        if chunk:
            segments.append({
                "start": chunk_start,
                "end": transcript_data["words"][-1]["end"],
                "text": " ".join(chunk),
            })

    # Fallback: if still no segments, use full_text with estimated timing
    if not segments and transcript_data.get("full_text"):
        logger.warning("No segments or words — using full text without timestamps")
        segments = [{"start": 0.0, "end": transcript_data.get("duration", 0), "text": transcript_data["full_text"]}]

    for seg in segments:
        minutes = int(seg["start"] // 60)
        seconds = seg["start"] % 60
        timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
        timestamped_lines.append(f"{timestamp} {seg['text']}")

    formatted_transcript = "\n".join(timestamped_lines)

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL_BOUNDARY,
            max_tokens=1024,
            system=BOUNDARY_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Here is the full timestamped transcript of a church service. "
                        f"Total duration: {transcript_data['duration']:.1f} seconds "
                        f"({transcript_data['duration'] / 60:.1f} minutes).\n\n"
                        f"TRANSCRIPT:\n{formatted_transcript}"
                    ),
                }
            ],
        )

        response_text = response.content[0].text.strip()
        logger.info(f"Raw Claude response (first 500 chars): {response_text[:500]}")

        # Strip any markdown fencing if present
        if "```" in response_text:
            # Extract content between first ``` and last ```
            parts = response_text.split("```")
            # The JSON is typically in parts[1]
            if len(parts) >= 3:
                inner = parts[1]
                # Remove optional language tag (e.g., "json\n")
                if inner.startswith("json"):
                    inner = inner[4:]
                response_text = inner.strip()
            else:
                response_text = parts[-1].strip()

        # Try to find JSON object in the response if there's surrounding text
        if not response_text.startswith("{"):
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]

        result = json.loads(response_text)

        if "error" in result:
            raise RuntimeError(f"Boundary detection failed: {result['error']}")

        # Refine boundaries using word-level timestamps
        words = transcript_data.get("words", [])
        if words:
            result = _refine_boundaries(result, words, status_callback)

        # Log both options
        dur_with = result["sermon_end_with_prayer"] - result["sermon_start"]
        dur_without = result["sermon_end_without_prayer"] - result["sermon_start"]
        logger.info(
            f"Sermon boundaries: start={result['sermon_start']:.1f}s, "
            f"end_with_prayer={result['sermon_end_with_prayer']:.1f}s ({dur_with / 60:.1f} min), "
            f"end_without_prayer={result['sermon_end_without_prayer']:.1f}s ({dur_without / 60:.1f} min), "
            f"confidence: {result['confidence']}"
        )

        if status_callback:
            # If clamping forced both endpoints to the same time, show a cleaner message
            if abs(dur_with - dur_without) < 1.0:
                status_callback(
                    f"Sermon identified: \"{result.get('sermon_title_guess', 'Unknown')}\" — "
                    f"{dur_with / 60:.1f} min "
                    f"(confidence: {result['confidence']})"
                )
            else:
                status_callback(
                    f"Sermon identified: \"{result.get('sermon_title_guess', 'Unknown')}\" — "
                    f"{dur_with / 60:.1f} min with prayer, "
                    f"{dur_without / 60:.1f} min without "
                    f"(confidence: {result['confidence']})"
                )

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response. Raw text:\n{response_text}")
        raise RuntimeError(
            f"Claude did not return valid JSON. Raw response: {response_text[:300]}"
        )
    except Exception as e:
        logger.error(f"Boundary detection error: {e}")
        raise
