"""
Module 3: Sermon Boundary Detection
Uses Claude API to analyze the transcript and identify sermon start/end timestamps.
"""
import json
import logging
from anthropic import Anthropic

import config

logger = logging.getLogger(__name__)

BOUNDARY_PROMPT = """You are an expert at analyzing church service transcripts to identify the structural boundaries of the sermon portion for radio broadcast.

The broadcast prioritizes content in this order (most important first):
  1. Scripture reading (always included)
  2. Sermon body (always included)
  3. Opening prayer (included when target duration allows)
  4. Closing prayer (included when target duration allows)

You will identify boundary timestamps for each component. The broadcast system will then choose which optional components to include based on the target duration.

---

REQUIRED TIMESTAMPS:

1. SERMON BODY START — sermon_body_start
   The moment the pastor begins preaching the sermon proper (not the scripture reading, not a prayer).
   This is typically right after a "you may be seated" cue following the scripture reading,
   or after special music between the readings and the sermon. It is the first words of the
   pastor's exposition/teaching content.

2. SCRIPTURE READING — scripture_start and scripture_end (or both null if no formal reading)
   The formal Bible reading(s) that the sermon is based on. Identifiable by:
   - Stately, verse-by-verse cadence (read deliberately)
   - Announced beforehand ("Our text today is from Acts chapter 2..." or
     "Please stand for the reading of God's Word")
   - Typically follows a "please stand" / "let us stand" cue (set scripture_start to AFTER
     the stand cue, at the first word of the actual reading)
   - Concludes with a closing phrase like "Here ends our reading", "This is the Word of the
     Lord", or "Thanks be to God" (set scripture_end to the end of the LAST formal reading,
     BEFORE any congregational response)

   Multiple consecutive readings: If the service has multiple formal readings back-to-back
   (e.g., OT reading + NT reading), include ALL of them. Set scripture_start to the start
   of the FIRST reading and scripture_end to the end of the LAST reading.

   What scripture_start/scripture_end IS NOT:
   - Mid-sermon paraphrases or quotes of scripture by the pastor
   - Earlier liturgical readings (Psalm in the call to worship, etc.) that are part of
     worship liturgy but not the sermon text
   - Recap of scripture inside the sermon body

   Match to sermon: If the sermon is preached on a DIFFERENT passage than the formal
   readings (e.g., sermon on John 7 but formal readings were Numbers 11 + Acts 2), STILL
   use the formal readings as scripture_start/scripture_end. Lutheran services often have
   readings that thematically support the sermon without being the exact passage preached.

   If no formal sermon scripture reading occurs (pastor jumps straight into preaching with
   only inline paraphrasing), set both scripture_start and scripture_end to null.

3. OPENING PRAYER — opening_prayer_start and opening_prayer_end (or both null if none)
   A SHORT prayer offered by the pastor AFTER the scripture reading (if any) and
   IMMEDIATELY before the sermon body begins. Often called a "prayer of illumination" or
   simply opening prayer.

   SERVICE ORDER: scripture reading → [optional creed/music/seating cue] →
   opening prayer → sermon body. The opening prayer is always between scripture_end
   and sermon_body_start.

   Identifiable by:
   - Phrases like "Let us pray", "Bow with me in prayer", "Heavenly Father, open our hearts..."
   - Brief duration (15-60 seconds typically)
   - Located AFTER scripture_end (when scripture is present), immediately before
     sermon_body_start
   - If there is no scripture reading, it appears immediately before sermon_body_start

   What opening_prayer IS NOT:
   - The invocation/prayer at the START of the service (way back at the call to worship)
   - The prayer of confession (corporate, liturgical)
   - The pastoral prayer (typically longer, intercessory)
   - The Lord's Prayer (corporate)
   - Prayer for graduates / special people (situational, not sermon-related)

   Only mark as opening_prayer if you're confident it's the pastor's sermon-opening prayer.
   When in doubt, set to null. Including a non-sermon prayer hurts the broadcast.

4. SEATING CUE — seating_cue_start and seating_cue_end (existing behavior, unchanged)
   "You may be seated" / "Please be seated" between scripture reading and sermon body.
   The cue that occurs AFTER the sermon scripture (not after earlier liturgical content).

5. SERMON END WITH PRAYER — sermon_end_with_prayer
   End of the pastor's closing prayer "Amen" — exclude anything after (announcements,
   closing hymn, benediction).

6. SERMON END WITHOUT PRAYER — sermon_end_without_prayer
   End of the last substantive sermon sentence BEFORE the prayer transition (before
   "Let's pray" / "Let us pray" / etc.).

---

RESPONSE FORMAT — respond ONLY with a JSON object, no markdown, no preamble:

{
    "sermon_body_start": 2229.7,
    "scripture_start": 1750.0,
    "scripture_end": 1920.0,
    "opening_prayer_start": null,
    "opening_prayer_end": null,
    "seating_cue_start": 2218.0,
    "seating_cue_end": 2220.0,
    "sermon_end_with_prayer": 3436.0,
    "sermon_end_without_prayer": 3367.4,
    "confidence": "high",
    "sermon_body_reason": "First words of preaching — quote in single quotes",
    "scripture_reason": "Brief identification of the scripture reading — quote first few words of the reading itself in single quotes",
    "opening_prayer_reason": "Identification of the opening prayer if present, or 'no opening prayer detected'",
    "seating_cue_reason": "The exact phrase used (or null)",
    "end_with_prayer_reason": "What the closing Amen is",
    "end_without_prayer_reason": "The last substantive sentence before prayer transition",
    "sermon_title_guess": "Best guess at sermon title/topic"
}

All times are in seconds (float). Use null (not 0.0) for missing optional timestamps.
Confidence: "high", "medium", or "low".

If you cannot identify a sermon at all, respond with:
{"error": "Explanation"}"""


def _refine_timestamp_by_quoted_phrase(timestamp: float, reason: str, words: list,
                                        search_window: float = 120.0,
                                        is_end: bool = False) -> tuple:
    """
    Try to refine a timestamp by finding a quoted phrase from `reason` in the word stream.

    Args:
        timestamp: Claude's reported timestamp (seconds)
        reason: Claude's reason string, which may contain a quoted phrase in single or double quotes
        words: Full word-level timestamp list
        search_window: How far around Claude's timestamp to search (seconds, default ±120)
        is_end: If True, snap to the END of the matched phrase; if False, snap to the START

    Returns:
        (refined_timestamp, matched_phrase) or (None, None) if no match
    """
    import re

    quoted = re.findall(r"'([^']+)'", reason or "")
    if not quoted:
        quoted = re.findall(r'"([^"]+)"', reason or "")
    if not quoted:
        return None, None

    phrase = quoted[0]
    search_words = phrase.lower().split()[:6]
    if len(search_words) < 2:
        return None, None

    search_begin = timestamp - search_window
    search_end = timestamp + search_window

    # Try progressively shorter matches: 5 words → 2 words
    for match_len in range(min(5, len(search_words)), 1, -1):
        target = [sw.strip(".,!?;:'\"") for sw in search_words[:match_len]]
        for i, w in enumerate(words):
            if w["start"] < search_begin or w["start"] > search_end:
                continue
            if i + match_len > len(words):
                continue
            candidate = [
                words[i + j]["word"].lower().strip(".,!?;:'\"")
                for j in range(match_len)
            ]
            if candidate == target:
                if is_end:
                    return words[i + match_len - 1]["end"] + 0.2, " ".join(candidate)
                else:
                    return max(0, words[i]["start"] - 0.3), " ".join(candidate)
    return None, None


def _refine_boundaries(boundaries: dict, words: list, status_callback=None) -> dict:
    """
    Refine Claude's boundaries using word-level timestamps.

    Claude's timestamps can be off by a minute or more because it works from
    segment-level timestamps. We use word-level data to find exact cut points.
    """
    sermon_start = boundaries["sermon_body_start"]
    end_with = boundaries["sermon_end_with_prayer"]
    end_without = boundaries["sermon_end_without_prayer"]

    logger.info(
        f"[REFINE] Claude endpoints: sermon_body_start={sermon_start:.1f}, "
        f"end_with={end_with:.1f}, end_without={end_without:.1f}"
    )

    # ── Refine SERMON BODY START boundary ────────────────────────────
    # Claude's start can land in a hymn or silence before the sermon.
    # Strategy: extract the first words Claude quoted in sermon_body_reason,
    # find them in the word-level timestamps, and snap to just before them.

    start_refined = False
    start_reason = boundaries.get("sermon_body_reason") or boundaries.get("start_reason", "")
    
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

        # Search for this sequence in the word timestamps near Claude's start.
        # Use a wider window (±180s) since Claude's segment timestamps can be
        # off by minutes when there's a hymn/musical interlude near the start.
        search_begin = sermon_start - 180
        search_end = sermon_start + 180

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
                    old_start = boundaries["sermon_body_start"]
                    boundaries["sermon_body_start"] = new_start
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

    # Fallback: gap-based detection if word matching didn't work.
    # Use a wider window and look for the LARGEST gap (most likely the
    # hymn-to-speech transition).
    if not start_refined:
        start_search_begin = sermon_start - 120
        start_search_end = sermon_start + 120

        start_region_words = [
            (i, w) for i, w in enumerate(words)
            if start_search_begin <= w["start"] <= start_search_end
        ]

        if start_region_words:
            # Find the LARGEST silence gap in the region, prefer ones close to
            # but before Claude's reported start time
            best_gap = 0
            best_idx = None
            for k in range(len(start_region_words) - 1):
                idx_a, word_a = start_region_words[k]
                idx_b, word_b = start_region_words[k + 1]
                gap = word_b["start"] - word_a["end"]

                # Require at least 2s gap (longer than typical word spacing
                # even in slow speech, but shorter than a service transition)
                if gap >= 2.0 and gap > best_gap:
                    best_gap = gap
                    best_idx = idx_b

            if best_idx is not None:
                new_start = words[best_idx]["start"] - 0.3
                old_start = boundaries["sermon_body_start"]
                boundaries["sermon_body_start"] = new_start
                start_refined = True
                logger.info(
                    f"[REFINE] Start gap (widest): {best_gap:.1f}s silence, "
                    f"speech resumes at {words[best_idx]['start']:.1f}s "
                    f"('{words[best_idx]['word']}') (was {old_start:.1f}s)"
                )
                if status_callback:
                    status_callback(
                        f"Refined start via silence gap ({best_gap:.1f}s) at "
                        f"{words[best_idx]['start']:.0f}s"
                    )

        if not start_refined:
            logger.warning(
                "[REFINE] Could not refine start — keeping Claude's endpoint "
                "(may include singing or silence before the sermon)"
            )
            if status_callback:
                status_callback(
                    "WARNING: Could not pinpoint sermon start — output may "
                    "include singing or silence at the beginning"
                )
    
    # ── Refine END boundaries ────────────────────────────────────────
    # Search the last 40% of the sermon for all markers
    sermon_start = boundaries["sermon_body_start"]  # use refined start
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
        "would you pray", "will you pray",
        "would you bow", "will you bow", "please bow",
        "bow our heads", "bow your heads", "bow with me",
        "let's bow", "let us bow",
        "would you join me", "will you join me", "join me in prayer",
        "pray with me",
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
            # No Amen before transition. Two possibilities:
            #
            # 1. Some services announce the closing hymn BEFORE the closing
            #    prayer, so the actual prayer Amen comes AFTER the transition.
            # 2. There genuinely is no closing prayer.
            #
            # Heuristic: if Claude's end_with_prayer_reason explicitly mentions
            # an Amen / prayer / "Jesus' name", trust that a prayer exists and
            # find the nearest Amen AFTER the transition (within 5 minutes).
            end_reason = (boundaries.get("end_with_prayer_reason") or "").lower()
            mentions_prayer = any(
                kw in end_reason
                for kw in ["amen", "prayer", "jesus' name", "jesus name", "in his name"]
            )

            late_amens = [
                a for a in amens
                if (a["word"]["end"] > first_transition["word"]["start"]
                    and a["word"]["end"] < first_transition["word"]["start"] + 300)
            ]

            if mentions_prayer and late_amens:
                # Prayer comes AFTER hymn announcement. Use the latest Amen in
                # the 5-minute window (closing prayer Amen is typically the
                # last one before the hymn actually starts being sung).
                best_amen = late_amens[-1]
                boundaries["sermon_end_with_prayer"] = best_amen["word"]["end"] + 0.5
                logger.info(
                    f"[REFINE] With-prayer: prayer follows hymn announcement. "
                    f"Snapped to Amen at {best_amen['word']['start']:.1f}s "
                    f"(transition was at {first_transition['word']['start']:.1f}s; "
                    f"Claude's reason mentioned prayer)"
                )
                if status_callback:
                    status_callback(
                        f"Refined: prayer follows hymn announcement — Amen at "
                        f"{best_amen['word']['start']:.0f}s"
                    )
            else:
                # No Amen anywhere reasonable — cut before the transition itself
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

    # ── Refine SCRIPTURE start/end ──────────────────────────────────
    scripture_start = boundaries.get("scripture_start")
    if scripture_start is not None:
        refined, matched = _refine_timestamp_by_quoted_phrase(
            scripture_start, boundaries.get("scripture_reason", ""),
            words, search_window=180.0, is_end=False,
        )
        if refined is not None:
            logger.info(
                f"[REFINE] Scripture start: {scripture_start:.1f}s -> {refined:.1f}s "
                f"(matched: '{matched}')"
            )
            boundaries["scripture_start"] = refined
            if status_callback:
                status_callback(f"Refined scripture start at {refined:.0f}s")
        else:
            logger.warning(
                "[REFINE] Could not refine scripture_start — keeping Claude's raw value "
                f"({scripture_start:.1f}s)"
            )

    scripture_end = boundaries.get("scripture_end")
    if scripture_end is not None:
        # For scripture end, look for closing phrases like "Here ends" / "Thanks be to God"
        # OR the quoted phrase from scripture_reason (if any was at the end).
        # Simpler approach: search for any of the common closing phrases near Claude's timestamp.
        closing_phrases = [
            ["here", "ends", "our"],          # "Here ends our scripture reading"
            ["here", "ends", "the"],          # "Here ends the lesson"
            ["thanks", "be", "to", "god"],    # "Thanks be to God"
            ["this", "is", "the", "word"],    # "This is the word of the Lord"
            ["word", "of", "the", "lord"],
        ]
        found = False
        for phrase in closing_phrases:
            for i, w in enumerate(words):
                if abs(w["start"] - scripture_end) > 120:
                    continue
                if i + len(phrase) > len(words):
                    continue
                candidate = [
                    words[i + j]["word"].lower().strip(".,!?;:'\"")
                    for j in range(len(phrase))
                ]
                if candidate == phrase:
                    new_end = words[i + len(phrase) - 1]["end"] + 0.3
                    logger.info(
                        f"[REFINE] Scripture end: {scripture_end:.1f}s -> {new_end:.1f}s "
                        f"(matched closing phrase: '{' '.join(phrase)}')"
                    )
                    boundaries["scripture_end"] = new_end
                    found = True
                    break
            if found:
                break
        if not found:
            logger.info(
                f"[REFINE] No scripture closing phrase found — keeping Claude's "
                f"scripture_end={scripture_end:.1f}s"
            )

    # ── Refine OPENING PRAYER start/end ─────────────────────────────
    opening_prayer_start = boundaries.get("opening_prayer_start")
    if opening_prayer_start is not None:
        refined, matched = _refine_timestamp_by_quoted_phrase(
            opening_prayer_start, boundaries.get("opening_prayer_reason", ""),
            words, search_window=120.0, is_end=False,
        )
        if refined is not None:
            logger.info(
                f"[REFINE] Opening prayer start: {opening_prayer_start:.1f}s -> {refined:.1f}s "
                f"(matched: '{matched}')"
            )
            boundaries["opening_prayer_start"] = refined

    opening_prayer_end = boundaries.get("opening_prayer_end")
    if opening_prayer_end is not None:
        # Opening prayer typically ends with "Amen" or "in Jesus' name, Amen"
        # Look for the closest "Amen" near Claude's timestamp
        closest_amen = None
        for w in words:
            if abs(w["end"] - opening_prayer_end) > 60:
                continue
            if w["word"].lower().strip(".,!?;:'\"") == "amen":
                if closest_amen is None or abs(w["end"] - opening_prayer_end) < abs(closest_amen["end"] - opening_prayer_end):
                    closest_amen = w
        if closest_amen:
            new_end = closest_amen["end"] + 0.5
            logger.info(
                f"[REFINE] Opening prayer end: {opening_prayer_end:.1f}s -> {new_end:.1f}s "
                f"(matched 'Amen')"
            )
            boundaries["opening_prayer_end"] = new_end

    # ── Validate: prayer/scripture must be in order ─────────────────
    # If scripture_end < scripture_start (Claude messed up), null out both.
    if (boundaries.get("scripture_start") is not None
            and boundaries.get("scripture_end") is not None
            and boundaries["scripture_end"] <= boundaries["scripture_start"]):
        logger.warning("[REFINE] scripture_end ≤ scripture_start; nulling both")
        boundaries["scripture_start"] = None
        boundaries["scripture_end"] = None

    # Opening prayer must come AFTER scripture (if any) and BEFORE sermon_body_start
    op_start = boundaries.get("opening_prayer_start")
    op_end = boundaries.get("opening_prayer_end")
    scripture_end_val = boundaries.get("scripture_end")
    body_start_val = boundaries.get("sermon_body_start")
    if op_start is not None and op_end is not None:
        # Must be before sermon body
        if body_start_val is not None and op_start >= body_start_val:
            logger.warning(
                f"[REFINE] opening_prayer_start ({op_start:.1f}s) is at or after "
                f"sermon_body_start ({body_start_val:.1f}s); nulling opening prayer"
            )
            boundaries["opening_prayer_start"] = None
            boundaries["opening_prayer_end"] = None
        # Must be after scripture end (if scripture exists), with small tolerance
        elif scripture_end_val is not None and op_start < scripture_end_val - 5:
            logger.warning(
                f"[REFINE] opening_prayer_start ({op_start:.1f}s) is before "
                f"scripture_end ({scripture_end_val:.1f}s); nulling opening prayer "
                f"(likely an earlier liturgical prayer, not the sermon-opening prayer)"
            )
            boundaries["opening_prayer_start"] = None
            boundaries["opening_prayer_end"] = None

    # ── Refine SEATING CUE if present ────────────────────────────────
    seating_start = boundaries.get("seating_cue_start")
    seating_end = boundaries.get("seating_cue_end")

    # Determine earliest content boundary so we can sanity-check that the
    # seating cue falls inside the broadcasted range. The cue sits in service
    # order between scripture_end and sermon_body_start, so the earliest
    # legitimate position is the start of whatever content we may include.
    earliest_content = min(
        v for v in (
            boundaries.get("scripture_start"),
            boundaries.get("opening_prayer_start"),
            boundaries.get("sermon_body_start"),
        )
        if v is not None
    )

    # Reject seating cues that fall OUTSIDE the sermon range. Sometimes
    # Claude returns a "you may be seated" from an earlier liturgical
    # reading (Psalm, Epistle, etc.) that's not the sermon scripture.
    if (seating_start is not None and seating_end is not None
            and (seating_start < earliest_content
                 or seating_end > boundaries["sermon_end_with_prayer"])):
        logger.warning(
            f"[REFINE] Seating cue at {seating_start:.1f}s falls outside "
            f"sermon range ({earliest_content:.1f}s - "
            f"{boundaries['sermon_end_with_prayer']:.1f}s). "
            f"Likely from an earlier liturgical reading. Ignoring."
        )
        if status_callback:
            status_callback(
                "Seating cue was outside sermon range (likely from earlier "
                "reading) — ignoring"
            )
        boundaries["seating_cue_start"] = None
        boundaries["seating_cue_end"] = None
        seating_start = None
        seating_end = None

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

        # Search window: tight window around Claude's reported timestamp.
        # Claude's segment-level timestamps can be off by 30-90 seconds, so we
        # search ±2 minutes around its reported position to find the actual
        # phrase. We DO NOT start from the earliest content because there may
        # be earlier "be seated" cues during liturgical readings.
        search_center = (seating_start + seating_end) / 2.0
        search_start = max(earliest_content, search_center - 120)
        search_end = min(boundaries["sermon_end_with_prayer"], search_center + 120)

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
                "skipping splice (Claude's raw timestamps too unreliable)"
            )
            # Clear the cue so the orchestrator skips the splice entirely.
            # Better to keep the cue audible than to splice the wrong chunk.
            boundaries["seating_cue_start"] = None
            boundaries["seating_cue_end"] = None
            if status_callback:
                status_callback(
                    "Seating cue not confirmed in word stream — leaving audio unchanged"
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

        # Backward-compat: legacy code reads `sermon_start`. Default to sermon_body_start.
        # The orchestrator may override this when it selects a content combination.
        if "sermon_start" not in result:
            result["sermon_start"] = result["sermon_body_start"]

        # Log both options
        dur_with = result["sermon_end_with_prayer"] - result["sermon_body_start"]
        dur_without = result["sermon_end_without_prayer"] - result["sermon_body_start"]
        logger.info(
            f"Sermon boundaries: sermon_body_start={result['sermon_body_start']:.1f}s, "
            f"end_with_prayer={result['sermon_end_with_prayer']:.1f}s ({dur_with / 60:.1f} min), "
            f"end_without_prayer={result['sermon_end_without_prayer']:.1f}s ({dur_without / 60:.1f} min), "
            f"confidence: {result['confidence']}"
        )

        logger.info(
            f"Structural map: "
            f"opening_prayer={'%.1fs' % result['opening_prayer_start'] if result.get('opening_prayer_start') else 'none'}, "
            f"scripture={'%.1fs-%.1fs' % (result['scripture_start'], result['scripture_end']) if result.get('scripture_start') else 'none'}, "
            f"sermon_body={result['sermon_body_start']:.1f}s, "
            f"end_with_prayer={result['sermon_end_with_prayer']:.1f}s, "
            f"end_without_prayer={result['sermon_end_without_prayer']:.1f}s"
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
