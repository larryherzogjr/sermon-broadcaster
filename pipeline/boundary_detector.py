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

1. The SERMON START — the moment the SCRIPTURE READING EVENT for the sermon begins.
   The broadcast includes the scripture reading AND the sermon body that follows.

   WHAT COUNTS AS THE SERMON SCRIPTURE READING EVENT:
   The "sermon scripture reading event" includes BOTH:
   (a) The pastor's ANNOUNCEMENT of the passage being read (e.g., "Our text today
       is from John, chapter 7, reading verses 37 through 39"), if present, AND
   (b) The reading of the verses themselves (e.g., "On the last day of the feast,
       the great day, Jesus stood up and cried out...")

   sermon_start should be at the BEGINNING of whichever happens FIRST:
   - If an announcement precedes the verses → start of announcement
   - If the pastor reads verses without a preceding announcement → start of verses

   The broadcast system will use word-level matching against the transcript to
   locate the exact start, and may walk backward across any citation/announcement
   that immediately precedes the matched words. So both options work — the
   important thing is that start_reason quotes words that appear VERBATIM in the
   transcript at sermon_start.

   WHAT IS NOT THE SERMON SCRIPTURE READING:
   - The pastor PARAPHRASING or RECAPPING scripture within the sermon body
     (e.g., "So Paul shares with them the God who made the world..." is the
     pastor's own narration, NOT a scripture reading)
   - The pastor QUOTING a verse mid-sermon for illustration
   - The pastor REFERENCING a passage during preaching
   - Earlier liturgical readings (Psalm, Epistle, OT reading) that are part
     of worship but are NOT what the sermon is preached from

   IMPORTANT — DISTINGUISHING THE SERMON SCRIPTURE FROM EARLIER READINGS:
   Many services have multiple scripture readings (e.g., Old Testament, Psalm,
   Epistle, Gospel) earlier in the service. The pastor may also ask the
   congregation to stand for those readings. Those are NOT the sermon scripture.

   The SERMON scripture reading is the LAST formal Bible reading BEFORE the
   sermon proper. You can identify it by these signs:
   - It is followed (often after a "you may be seated" cue) by sustained
     expository or topical preaching on that same passage
   - The pastor frequently references the passage during the sermon
   - It typically comes AFTER any earlier liturgical readings, hymns, offering,
     pastoral prayer, etc.
   - The text being read sounds like Bible verses (formal, scriptural language),
     not the pastor's own conversational explanation

   If unsure between two candidate readings, pick the one whose content the
   pastor preaches on in the sermon body. Look at what the sermon is ABOUT and
   match it to the FORMAL reading (not to mid-sermon paraphrases).

   Just before the sermon scripture reading event, the pastor typically asks the
   congregation to stand:
     - "Please stand for the reading of God's Word"
     - "Let us stand for the reading of God's Word"
     - "Please rise"
   The sermon_start should be set to AFTER the "please stand" cue ends, at the
   start of the scripture reading event.

   IMPORTANT: start_reason MUST quote words that appear VERBATIM at sermon_start
   in the transcript. The system uses your quote to verify the exact position
   via word-level matching. If your quote isn't found, the system falls back to
   silence-gap heuristics that often produce wrong results. So:
   - Quote the actual first 5-8 words at sermon_start, copied from the transcript
   - Use single quotes around the quoted phrase
   - Don't paraphrase, don't combine, don't clean up grammar

   IMPORTANT: Use the timestamp of the BEGINNING of the scripture reading event.
   If no formal scripture reading occurs (pastor jumps straight into preaching),
   use the START of the sermon proper.

   This is NOT:
   - Welcome/announcements
   - Earlier scripture readings (OT, Psalm, Epistle, etc.) that are part of the
     liturgy but not what the sermon is preached from
   - Opening prayer (before any scripture)
   - Hymns or worship
   - Offering
   - The "please stand" cue itself (we want to skip past this)
   - Mid-sermon paraphrases or quotes of scripture by the pastor

2. TWO SERMON END POINTS — you must provide both:

   a) sermon_end_with_prayer — the end of the closing prayer's "Amen."
      - This is the prayer the pastor prays to conclude the sermon.
      - The timestamp should be the END of the segment containing the final "Amen" of this prayer.
      - EXCLUDE anything after: announcements, closing hymns, benediction.

   b) sermon_end_without_prayer — the end of the last substantive sermon sentence BEFORE the prayer transition.
      - This is the final teaching point, application, or concluding thought.
      - EXCLUDE "Let us pray" / "Let's pray" / "Shall we pray" / any prayer language.
      - Example: "...and that is the hope we have in Christ. Let's pray." — this timestamp is the END of the segment containing "Christ."

   The broadcast system will choose which ending to use based on time constraints.

3. ADDITIONAL STRUCTURAL FIELDS (used for fine-grained content selection)

   Provide each of these IF clearly present, or null otherwise. These are
   SECONDARY — the primary identifications above are the most important.
   Use null freely if a component isn't clearly there; do NOT invent boundaries.

   a) scripture_end — timestamp where the formal sermon scripture reading
      concludes. This is the END of the last formal reading, BEFORE any
      congregational response. Typical markers: "Here ends our reading",
      "This is the Word of the Lord", "Thanks be to God". If there are
      multiple consecutive sermon-related readings (e.g., OT then NT), this
      is the end of the LAST one. Null if no formal scripture reading.

   b) sermon_body_start — timestamp where the pastor's preached content begins.
      This is AFTER scripture reading, AFTER any opening prayer, AFTER any
      special music. It is the first words of the pastor's actual
      teaching/exposition. If the pastor jumps straight into preaching with
      no scripture reading, this equals sermon_start.

   c) opening_prayer_start and opening_prayer_end — A SHORT prayer offered
      by the pastor AFTER the scripture reading (if any) and IMMEDIATELY
      before the sermon body begins. Often called a "prayer of illumination."
      Typical phrases: "Let us pray", "Heavenly Father, open our hearts...".
      Brief duration (15-60 seconds typically).

      This is NOT:
      - The invocation/prayer at the START of the service
      - The corporate prayer of confession
      - The pastoral prayer (typically longer, intercessory)
      - The Lord's Prayer
      - Prayer for graduates / special people
      - The CLOSING prayer at the end of the sermon

      The opening prayer ALWAYS sits between the scripture reading (or
      sermon_start if no scripture) and the sermon body. When in doubt, set
      to null.

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{
    "sermon_start": 123.45,
    "sermon_end_with_prayer": 2400.00,
    "sermon_end_without_prayer": 2345.67,
    "scripture_end": 256.10,
    "sermon_body_start": 280.50,
    "opening_prayer_start": 258.00,
    "opening_prayer_end": 275.30,
    "confidence": "high",
    "start_reason": "Brief explanation — quote 5-8 first words at sermon_start VERBATIM from the transcript in single quotes. These will be used for word-level matching.",
    "end_with_prayer_reason": "Brief explanation — what the closing Amen is",
    "end_without_prayer_reason": "The last substantive sentence before prayer transition",
    "scripture_end_reason": "Brief description (or null)",
    "sermon_body_reason": "Quote the first few words of the sermon body in single quotes",
    "opening_prayer_reason": "Brief description (or null if no opening prayer)",
    "sermon_title_guess": "Your best guess at the sermon title/topic based on content"
}

All times are in seconds (float). Confidence should be "high", "medium", or "low".
Use null for any of the section-3 secondary fields you cannot identify with confidence.

If you cannot identify a sermon in the transcript at all, respond with:
{
    "error": "Explanation of why sermon boundaries could not be determined"
}"""


def _refine_boundaries(boundaries: dict, words: list, status_callback=None) -> dict:
    """
    Refine Claude's boundaries using word-level timestamps.

    Claude's timestamps can be off by a minute or more because it works from
    segment-level timestamps. We use word-level data to find exact cut points.
    """
    # ── Defensive validation and field normalization ────────────────
    # Accept either sermon_start (legacy) or sermon_body_start (new). Normalize
    # so downstream code can read either key. Fail with a clear, actionable
    # error if Claude returned neither (e.g., low-confidence "no sermon detected").
    sermon_start_val = boundaries.get("sermon_start")
    body_start_val = boundaries.get("sermon_body_start")

    if sermon_start_val is None and body_start_val is None:
        reason = (boundaries.get("sermon_body_reason")
                  or boundaries.get("start_reason")
                  or "no explanation given")
        confidence = boundaries.get("confidence", "unknown")
        raise RuntimeError(
            f"Boundary detection failed: Claude returned null sermon start "
            f"(confidence: {confidence}). Reason: {reason[:250]}"
        )

    # Mirror missing fields so both keys are populated
    if body_start_val is None:
        body_start_val = sermon_start_val
        boundaries["sermon_body_start"] = sermon_start_val
    if sermon_start_val is None:
        sermon_start_val = body_start_val
        boundaries["sermon_start"] = body_start_val

    # NOTE: scripture_start derivation moved to the END of this function so it
    # uses REFINED sermon_start (word-level snapped), not Claude's raw timestamp.

    if not boundaries.get("sermon_end_with_prayer"):
        raise RuntimeError("Boundary detection returned no sermon_end_with_prayer")
    if not boundaries.get("sermon_end_without_prayer"):
        boundaries["sermon_end_without_prayer"] = boundaries["sermon_end_with_prayer"]

    sermon_start = sermon_start_val
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
                    # Back up to capture any preceding citation/announcement.
                    # Two-pass approach:
                    # 1. Collect sentence-boundary indices going backward.
                    # 2. Walk through them, checking whether the PREVIOUS sentence
                    #    (the one we'd cross INTO) looks like a Bible citation.
                    #    Include it if yes; stop if no.
                    BIBLE_BOOKS = {
                        "genesis", "exodus", "leviticus", "numbers", "deuteronomy",
                        "joshua", "judges", "ruth", "samuel", "kings", "chronicles",
                        "ezra", "nehemiah", "esther", "job", "psalm", "psalms",
                        "proverbs", "ecclesiastes", "song", "isaiah", "jeremiah",
                        "lamentations", "ezekiel", "daniel", "hosea", "joel",
                        "amos", "obadiah", "jonah", "micah", "nahum", "habakkuk",
                        "zephaniah", "haggai", "zechariah", "malachi", "matthew",
                        "mark", "luke", "john", "acts", "romans", "corinthians",
                        "galatians", "ephesians", "philippians", "colossians",
                        "thessalonians", "timothy", "titus", "philemon",
                        "hebrews", "james", "peter", "jude", "revelation",
                    }
                    CHAPTER_VERSE_KEYWORDS = {"chapter", "verse", "verses"}

                    def _looks_like_citation(word_list):
                        """A scripture citation either names a Bible book
                        explicitly, or includes a chapter/verse keyword
                        alongside a numeric token. Bare keywords like
                        'reading' or 'text' aren't enough — 'please stand
                        for the reading' would otherwise match."""
                        text_words = [
                            w.get("word", "").lower().strip(".,!?;:'\"")
                            for w in word_list
                        ]
                        has_book = any(w in BIBLE_BOOKS for w in text_words)
                        has_chapter_verse = any(
                            w in CHAPTER_VERSE_KEYWORDS for w in text_words
                        )
                        has_number = any(
                            any(c.isdigit() for c in w) for w in text_words
                        )
                        return has_book or (has_chapter_verse and has_number)

                    # Pass 1: collect sentence boundaries going backward.
                    # boundary_starts[k] = (idx, label) where idx is the FIRST
                    # word of a sentence above a boundary.
                    look_back_limit = words[i]["start"] - 30.0
                    boundary_starts = []
                    for j in range(i - 1, -1, -1):
                        if words[j]["end"] < look_back_limit:
                            break

                        is_boundary = False
                        boundary_label = None

                        if j + 1 < len(words):
                            gap = words[j + 1]["start"] - words[j]["end"]
                            if gap > 1.5:
                                is_boundary = True
                                boundary_label = f"silence ({gap:.1f}s)"

                        if not is_boundary:
                            prev_word_raw = words[j].get("word", "").strip()
                            if prev_word_raw.endswith((".", "?", "!")):
                                is_boundary = True
                                boundary_label = (
                                    f"sentence end '{prev_word_raw}'"
                                )

                        if is_boundary:
                            boundary_starts.append((j + 1, boundary_label))

                    # Pass 2: decide how far back to go.
                    sentence_start_idx = i
                    backed_up_reason = None

                    if boundary_starts:
                        # Default: stop at first boundary (start of current
                        # sentence — the one containing the match).
                        sentence_start_idx = boundary_starts[0][0]
                        backed_up_reason = boundary_starts[0][1]

                        # Walk further back across citation-like sentences.
                        # boundary_starts[k+1] is the start of the sentence
                        # ABOVE boundary_starts[k]. Check that "above" sentence;
                        # if citation, include it.
                        for k in range(len(boundary_starts) - 1):
                            sent_start = boundary_starts[k + 1][0]
                            sent_end_excl = boundary_starts[k][0]
                            if sent_end_excl <= sent_start:
                                break
                            sentence_words = words[sent_start:sent_end_excl]
                            sent_dur = (
                                words[sent_end_excl - 1]["end"]
                                - words[sent_start]["start"]
                            )
                            if (sent_dur <= 12.0
                                    and _looks_like_citation(sentence_words)):
                                sentence_start_idx = sent_start
                                backed_up_reason = (
                                    f"citation across "
                                    f"{boundary_starts[k + 1][1]}"
                                )
                            else:
                                break

                    new_start = words[sentence_start_idx]["start"] - 0.3
                    old_start = boundaries["sermon_start"]
                    boundaries["sermon_start"] = new_start
                    start_refined = True
                    if sentence_start_idx != i:
                        logger.info(
                            f"[REFINE] Start matched ({match_len} words) "
                            f"'{' '.join(candidate)}' at {words[i]['start']:.1f}s; "
                            f"backed up to '{words[sentence_start_idx]['word']}' at "
                            f"{words[sentence_start_idx]['start']:.1f}s "
                            f"({backed_up_reason}; was {old_start:.1f}s)"
                        )
                    else:
                        logger.info(
                            f"[REFINE] Start matched ({match_len} words): "
                            f"'{' '.join(candidate)}' at {words[i]['start']:.1f}s "
                            f"(was {old_start:.1f}s)"
                        )
                    if status_callback:
                        status_callback(
                            f"Refined start: '{first_phrase[:40]}...' at "
                            f"{words[sentence_start_idx]['start']:.0f}s"
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
                old_start = boundaries["sermon_start"]
                boundaries["sermon_start"] = new_start
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

    # ── Final structural pass: sanity check + scripture_start derivation ─
    # This happens AFTER all refinement so we use word-level snapped timestamps.

    # Opening prayer validation: position (after scripture, before sermon body)
    # and duration (10-90s typical pre-sermon prayer).
    op_start = boundaries.get("opening_prayer_start")
    op_end = boundaries.get("opening_prayer_end")
    scripture_end_check = boundaries.get("scripture_end")
    body_start_check = boundaries.get("sermon_body_start")
    if op_start is not None and op_end is not None:
        # Duration sanity. A pre-sermon prayer is roughly 10-90 seconds. A
        # 4-second "prayer" is almost certainly Claude latching onto an "Amen"
        # by mistake; a multi-minute "prayer" is Claude absorbing the sermon
        # body.
        op_dur = op_end - op_start
        if op_dur < 10 or op_dur > 90:
            logger.warning(
                f"[REFINE] Opening prayer duration {op_dur:.1f}s out of "
                f"plausible range (10-90s); nulling — likely misidentified"
            )
            boundaries["opening_prayer_start"] = None
            boundaries["opening_prayer_end"] = None
            op_start = None
            op_end = None

    if op_start is not None and op_end is not None:
        # Must come BEFORE sermon body
        if body_start_check is not None and op_start >= body_start_check:
            logger.warning(
                f"[REFINE] opening_prayer_start ({op_start:.1f}s) is at or "
                f"after sermon_body_start ({body_start_check:.1f}s); nulling"
            )
            boundaries["opening_prayer_start"] = None
            boundaries["opening_prayer_end"] = None
        # Must come AFTER scripture end (if scripture exists), with tolerance
        elif scripture_end_check is not None and op_start < scripture_end_check - 5:
            logger.warning(
                f"[REFINE] opening_prayer_start ({op_start:.1f}s) is before "
                f"scripture_end ({scripture_end_check:.1f}s); nulling — likely "
                f"an earlier liturgical prayer, not the sermon-opening prayer"
            )
            boundaries["opening_prayer_start"] = None
            boundaries["opening_prayer_end"] = None

    # Sanity check: if Claude's structural breakdown is implausible (e.g.,
    # scripture longer than sermon body, or sermon body < 5 min), discard the
    # structural fields. Common cause: pastor reads scripture inline within the
    # sermon, and Claude can't tell where the "reading" ends from where the
    # sermon body begins.
    scripture_start_val = boundaries.get("scripture_start")
    scripture_end_val = boundaries.get("scripture_end")
    scripture_dur = 0
    if scripture_start_val is not None and scripture_end_val is not None:
        scripture_dur = scripture_end_val - scripture_start_val

    body_start_for_check = (boundaries.get("sermon_body_start")
                            or boundaries["sermon_start"])
    body_dur = boundaries["sermon_end_with_prayer"] - body_start_for_check

    structure_invalid = (
        (scripture_dur > 0 and scripture_dur > body_dur)
        or (boundaries.get("sermon_body_start") is not None and 0 < body_dur < 300)
    )

    if structure_invalid:
        logger.warning(
            f"[REFINE] Implausible structure: scripture={scripture_dur:.0f}s, "
            f"sermon_body={body_dur:.0f}s. Discarding structural fields; "
            f"falling back to single-block sermon_start..sermon_end_with_prayer."
        )
        if status_callback:
            status_callback(
                f"WARNING: Claude's structural breakdown looked wrong "
                f"(scripture {scripture_dur:.0f}s vs body {body_dur:.0f}s) — "
                f"treating sermon as single block"
            )
        boundaries["scripture_start"] = None
        boundaries["scripture_end"] = None
        boundaries["opening_prayer_start"] = None
        boundaries["opening_prayer_end"] = None
        boundaries["sermon_body_start"] = None  # orchestrator falls back to sermon_start

    # Derive scripture_start from REFINED sermon_start if structure is still valid
    # and sermon_body_start indicates scripture content exists between them.
    if (boundaries.get("scripture_start") is None
            and boundaries.get("sermon_body_start") is not None
            and boundaries["sermon_body_start"] > boundaries["sermon_start"] + 30):
        boundaries["scripture_start"] = boundaries["sermon_start"]
        logger.info(
            f"[REFINE] Derived scripture_start={boundaries['sermon_start']:.1f}s "
            f"(sermon_body_start={boundaries['sermon_body_start']:.1f}s is "
            f"{boundaries['sermon_body_start'] - boundaries['sermon_start']:.0f}s later)"
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

        # Structural map for diagnostic visibility (null-safe formatting)
        def _fmt_ts(v):
            return f"{v:.1f}s" if v is not None else "none"
        def _fmt_range(a, b):
            if a is None or b is None:
                return "none"
            return f"{a:.1f}s-{b:.1f}s"
        logger.info(
            f"Structural map: "
            f"scripture={_fmt_range(result.get('scripture_start'), result.get('scripture_end'))}, "
            f"opening_prayer={_fmt_range(result.get('opening_prayer_start'), result.get('opening_prayer_end'))}, "
            f"sermon_body={_fmt_ts(result.get('sermon_body_start'))}, "
            f"end_with={_fmt_ts(result.get('sermon_end_with_prayer'))}, "
            f"end_without={_fmt_ts(result.get('sermon_end_without_prayer'))}"
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
