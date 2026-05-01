"""
Module 5: Pipeline Orchestrator
Coordinates the full sermon processing pipeline.
"""
import os
import re
import json
import hashlib
import shutil
import logging
import time
from datetime import datetime

from pipeline.downloader import download_audio
from pipeline.boundary_detector import detect_boundaries
from pipeline.audio_processor import extract_segment, fit_to_duration
from pipeline.teaser_selector import select_teaser
from pipeline.assembler import mix_teaser_into_intro, assemble_broadcast, get_bumper_durations

import config

logger = logging.getLogger(__name__)


def _check_audio(filepath, label, status_callback=None):
    """Diagnostic: check audio levels of a file and log them."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", filepath, "-af", "volumedetect", "-f", "null", "/dev/null"],
            capture_output=True, text=True
        )
        output = result.stderr
        mean_vol = "unknown"
        max_vol = "unknown"
        duration = "unknown"
        for line in output.split("\n"):
            if "mean_volume" in line:
                mean_vol = line.split("mean_volume:")[1].strip()
            if "max_volume" in line:
                max_vol = line.split("max_volume:")[1].strip()
            if "Duration" in line and "bitrate" in line:
                duration = line.split("Duration:")[1].split(",")[0].strip()
        msg = f"[DIAG] {label}: duration={duration}, mean={mean_vol}, max={max_vol}"
        logger.info(msg)
        if status_callback:
            status_callback(msg)
    except Exception as e:
        logger.error(f"[DIAG] {label}: check failed: {e}")


def _parse_target(duration_str: str) -> float:
    """Parse duration string (MM:SS or HH:MM:SS) to seconds."""
    parts = duration_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

# Select transcriber based on config
if config.TRANSCRIBER == "cloud":
    from pipeline.transcriber_cloud import transcribe
    logger.info("Using cloud transcriber (OpenAI Whisper API)")
else:
    from pipeline.transcriber import transcribe
    logger.info("Using local transcriber (faster-whisper)")

# ── Cache helpers ────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _video_id(url: str) -> str:
    """Extract YouTube video ID from URL for cache keying."""
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    return m.group(1) if m else hashlib.md5(url.encode()).hexdigest()[:12]


def _cache_path(video_id: str, suffix: str) -> str:
    return os.path.join(CACHE_DIR, f"{video_id}_{suffix}")


def run_pipeline(youtube_url: str = None, local_file: str = None,
                 target_duration: str = None,
                 include_bumpers: bool = False,
                 include_bumpers_dynamic: bool = False,
                 include_bumpers_stock: bool = False,
                 sermon_only: bool = False,
                 status_callback=None, boundary_override: dict = None) -> dict:
    """
    Run the full sermon processing pipeline.

    Args:
        youtube_url: YouTube video URL (provide this OR local_file)
        local_file: Path to a local audio/video file (provide this OR youtube_url)
        target_duration: Total target duration string (MM:SS or HH:MM:SS).
        include_bumpers: Legacy flag — if True, equivalent to include_bumpers_dynamic
        include_bumpers_dynamic: If True, produce variant with AI-selected teaser
        include_bumpers_stock: If True, produce variant with pre-mixed stock intro
        sermon_only: If True, skip boundary detection — audio IS the sermon.
                     Use when uploading a pre-trimmed sermon or manually edited audio.
        status_callback: Optional callable(str) for progress updates
        boundary_override: Optional dict with "sermon_start" and "sermon_end"

    Returns:
        dict with pipeline results (may include multiple output files)
    """
    if not youtube_url and not local_file:
        raise ValueError("Either youtube_url or local_file must be provided")

    # Backward compat: if old flag is set, treat as dynamic
    if include_bumpers and not (include_bumpers_dynamic or include_bumpers_stock):
        include_bumpers_dynamic = True

    any_bumpers = include_bumpers_dynamic or include_bumpers_stock

    if target_duration is None:
        if any_bumpers:
            target_duration = config.DEFAULT_BROADCAST_DURATION
        else:
            target_duration = config.DEFAULT_TARGET_DURATION

    # Calculate the sermon-only target duration
    total_target_seconds = _parse_target(target_duration)

    if any_bumpers:
        bumper_durations = get_bumper_durations()
        intro_dur = bumper_durations["intro"]
        outro_dur = bumper_durations["outro"]
        sermon_target_seconds = total_target_seconds - intro_dur - outro_dur
        # Add 1s overshoot to compensate for MP3 frame padding loss during encoding
        sermon_target_seconds += 1.0
        sermon_target_str = f"{int(sermon_target_seconds // 60)}:{int(sermon_target_seconds % 60):02d}"

        if status_callback:
            status_callback(
                f"Broadcast mode: {target_duration} total — "
                f"intro {intro_dur:.0f}s + sermon {sermon_target_seconds:.0f}s + outro {outro_dur:.0f}s"
            )
        logger.info(
            f"Broadcast mode: total={total_target_seconds:.0f}s, "
            f"intro={intro_dur:.1f}s, outro={outro_dur:.1f}s, "
            f"sermon_target={sermon_target_seconds:.1f}s ({sermon_target_str})"
        )
    else:
        # Add 1s overshoot to compensate for MP3 frame padding loss
        sermon_target_seconds = total_target_seconds + 1.0
        sermon_target_str = f"{int(sermon_target_seconds // 60)}:{int(sermon_target_seconds % 60):02d}"

    # Create a unique working directory for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(config.WORK_DIR, run_id)
    os.makedirs(work_dir, exist_ok=True)

    # Generate cache key from URL or file hash
    if youtube_url:
        vid = _video_id(youtube_url)
        source_label = youtube_url
    else:
        import hashlib as _hl
        with open(local_file, "rb") as f:
            vid = _hl.md5(f.read(1024 * 1024)).hexdigest()[:12]
        source_label = os.path.basename(local_file)

    result = {
        "run_id": run_id,
        "source": source_label,
        "target_duration": target_duration,
        "timing": {},
    }

    try:
        # Stage 1: Get audio (download from YT, or convert local file)
        t0 = time.time()
        cached_audio = _cache_path(vid, "audio.wav")
        if os.path.exists(cached_audio):
            raw_audio_path = os.path.join(work_dir, "raw_audio.wav")
            shutil.copy2(cached_audio, raw_audio_path)
            if status_callback:
                status_callback("Using cached audio...")
        elif youtube_url:
            raw_audio_path = download_audio(youtube_url, work_dir, status_callback)
            shutil.copy2(raw_audio_path, cached_audio)
        else:
            # Convert uploaded file to WAV
            if status_callback:
                status_callback(f"Converting uploaded file...")
            raw_audio_path = os.path.join(work_dir, "raw_audio.wav")
            import subprocess as _sp
            cmd = [
                "ffmpeg", "-y", "-i", local_file,
                "-acodec", "pcm_s16le", "-ar", "48000",
                raw_audio_path,
            ]
            conv = _sp.run(cmd, capture_output=True, text=True)
            if conv.returncode != 0:
                raise RuntimeError(f"Failed to convert uploaded file: {conv.stderr[-200:]}")
            shutil.copy2(raw_audio_path, cached_audio)
            if status_callback:
                status_callback("File converted to audio.")
            # Clean up the uploaded file — it's now cached as WAV
            try:
                os.remove(local_file)
                logger.info(f"Cleaned up uploaded file: {local_file}")
            except OSError:
                pass
        result["timing"]["download"] = time.time() - t0

        # Stage 2: Transcribe
        t0 = time.time()
        transcript_data = transcribe(raw_audio_path, status_callback)

        # Save the path to the audio file Whisper actually transcribed.
        # Teaser extraction MUST use this file so timestamps align exactly.
        transcribed_audio = transcript_data.pop("transcribed_audio_path", None)
        if transcribed_audio and os.path.exists(transcribed_audio):
            # Copy to work dir preserving the original extension
            ext = os.path.splitext(transcribed_audio)[1] or ".audio"
            teaser_source_path = os.path.join(work_dir, f"transcribed_audio{ext}")
            shutil.copy2(transcribed_audio, teaser_source_path)
        else:
            # Fallback: use raw audio (local transcriber doesn't compress)
            teaser_source_path = raw_audio_path

        result["transcript"] = {
            "full_text": transcript_data["full_text"],
            "segment_count": len(transcript_data["segments"]),
            "word_count": len(transcript_data["words"]),
            "duration": transcript_data["duration"],
        }
        result["timing"]["transcribe"] = time.time() - t0

        # Stage 3: Detect boundaries (or skip if audio is sermon-only)
        t0 = time.time()
        if sermon_only:
            # Audio IS the sermon — no boundary detection needed
            full_duration = transcript_data["duration"]
            boundaries = {
                "sermon_start": 0.0,
                "sermon_end": full_duration,
                "sermon_end_with_prayer": full_duration,
                "sermon_end_without_prayer": full_duration,
                "confidence": "manual",
                "sermon_title_guess": "Uploaded sermon (pre-trimmed)",
            }
            if status_callback:
                status_callback(
                    f"Sermon-only mode: using full audio ({full_duration/60:.1f} min) "
                    f"as the sermon"
                )
            logger.info(
                f"Sermon-only mode: skipping boundary detection, "
                f"using full audio ({full_duration:.1f}s)"
            )
        elif boundary_override:
            boundaries = boundary_override
            if status_callback:
                status_callback("Using manually specified sermon boundaries...")
        else:
            boundaries = detect_boundaries(transcript_data, status_callback)

        # Choose which end point to use (skip if sermon_only — already set)
        if not sermon_only and "sermon_end_with_prayer" in boundaries:
            dur_with = boundaries["sermon_end_with_prayer"] - boundaries["sermon_start"]
            dur_without = boundaries["sermon_end_without_prayer"] - boundaries["sermon_start"]

            # Calculate how well each option fits the sermon target
            # (using sermon_target_seconds which accounts for bumpers if enabled)
            delta_with = dur_with - sermon_target_seconds
            delta_without = dur_without - sermon_target_seconds

            # Check if each option is achievable within tempo limits
            # Too long: can we speed up enough (after ~60s of silence trimming)?
            max_fittable = sermon_target_seconds * config.MAX_SPEEDUP + 60
            # Too short: can we slow down enough (after ~small silence expansion)?
            min_fittable = sermon_target_seconds * config.MAX_SLOWDOWN - 10

            with_fits = min_fittable <= dur_with <= max_fittable
            without_fits = min_fittable <= dur_without <= max_fittable

            logger.info(
                f"Endpoint selection: with_prayer={dur_with:.0f}s (delta={delta_with:+.0f}s, fits={with_fits}), "
                f"without_prayer={dur_without:.0f}s (delta={delta_without:+.0f}s, fits={without_fits}), "
                f"target={sermon_target_seconds:.0f}s"
            )

            if with_fits and without_fits:
                # Both fit — prefer with-prayer (more complete), unless without-prayer
                # is closer to the target and requires less adjustment
                if abs(delta_with) <= abs(delta_without):
                    boundaries["sermon_end"] = boundaries["sermon_end_with_prayer"]
                    prayer_decision = "including closing prayer (better fit)"
                else:
                    boundaries["sermon_end"] = boundaries["sermon_end_with_prayer"]
                    prayer_decision = "including closing prayer (preferred)"
            elif with_fits:
                boundaries["sermon_end"] = boundaries["sermon_end_with_prayer"]
                prayer_decision = "including closing prayer"
            elif without_fits:
                boundaries["sermon_end"] = boundaries["sermon_end_without_prayer"]
                prayer_decision = "excluding closing prayer (with-prayer too short to fit)"
            else:
                # Neither fits perfectly — pick whichever is closer to target
                if abs(delta_with) <= abs(delta_without):
                    boundaries["sermon_end"] = boundaries["sermon_end_with_prayer"]
                    prayer_decision = "including closing prayer (closest to target, may exceed tempo limits)"
                else:
                    boundaries["sermon_end"] = boundaries["sermon_end_without_prayer"]
                    prayer_decision = "excluding closing prayer (closest to target, may exceed tempo limits)"

            if status_callback:
                chosen_dur = boundaries["sermon_end"] - boundaries["sermon_start"]
                status_callback(
                    f"End point selected: {prayer_decision} "
                    f"({chosen_dur / 60:.1f} min)"
                )
            logger.info(f"Prayer decision: {prayer_decision}")
        elif "sermon_end" not in boundaries:
            raise RuntimeError("Boundary detection returned no end point")

        result["boundaries"] = boundaries
        result["timing"]["boundaries"] = time.time() - t0

        # Stage 4: Extract sermon segment
        t0 = time.time()
        sermon_audio_path = os.path.join(work_dir, "sermon_raw.wav")
        logger.info(
            f"[EXTRACT] Using boundaries: start={boundaries['sermon_start']:.2f}, "
            f"end={boundaries['sermon_end']:.2f} "
            f"(duration={boundaries['sermon_end'] - boundaries['sermon_start']:.1f}s)"
        )
        if status_callback:
            status_callback(
                f"[DEBUG] Extracting: {boundaries['sermon_start']:.1f}s to "
                f"{boundaries['sermon_end']:.1f}s"
            )
        extract_segment(
            raw_audio_path,
            boundaries["sermon_start"],
            boundaries["sermon_end"],
            sermon_audio_path,
        )

        if status_callback:
            sermon_dur = boundaries["sermon_end"] - boundaries["sermon_start"]
            status_callback(
                f"Extracted sermon segment: {sermon_dur / 60:.1f} minutes"
            )

        # Diagnostics: check audio at each stage
        _check_audio(raw_audio_path, "raw_audio", status_callback)
        _check_audio(sermon_audio_path, "after_extract", status_callback)

        result["timing"]["extract"] = time.time() - t0

        # Stage 4.5: Splice out seating cue if detected
        # ("you may be seated" / "please be seated" between scripture and sermon)
        seating_start_orig = boundaries.get("seating_cue_start")
        seating_end_orig = boundaries.get("seating_cue_end")

        if seating_start_orig is not None and seating_end_orig is not None:
            # The extracted sermon audio added a 500ms pre-roll buffer at the start
            # (see extract_segment in audio_processor.py). So the time mapping is:
            #   extracted_time = original_time - (sermon_start - 0.5)
            extract_pre_roll = 0.5
            extract_origin = boundaries["sermon_start"] - extract_pre_roll
            seating_start_in_extract = seating_start_orig - extract_origin
            seating_end_in_extract = seating_end_orig - extract_origin

            from pipeline.audio_processor import remove_segment

            sermon_spliced_path = os.path.join(work_dir, "sermon_spliced.wav")
            try:
                remove_segment(
                    sermon_audio_path,
                    seating_start_in_extract,
                    seating_end_in_extract,
                    sermon_spliced_path,
                )
                # Use the spliced audio going forward
                sermon_audio_path = sermon_spliced_path
                removed_dur = seating_end_orig - seating_start_orig
                if status_callback:
                    status_callback(
                        f"Removed seating cue: {removed_dur:.1f}s spliced out"
                    )
                logger.info(
                    f"Seating cue spliced out: "
                    f"{seating_start_in_extract:.2f}s - {seating_end_in_extract:.2f}s "
                    f"in extracted audio ({removed_dur:.2f}s removed)"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to splice seating cue (non-fatal, continuing): {e}"
                )

        # Stage 5: Fit sermon to target duration
        t0 = time.time()
        sermon_fitted_path = os.path.join(work_dir, "sermon_fitted.mp3")

        processing_details = fit_to_duration(
            sermon_audio_path, sermon_target_str, sermon_fitted_path, status_callback
        )
        result["processing"] = processing_details
        result["timing"]["fit_to_duration"] = time.time() - t0

        # Stage 6: Assemble output(s)
        t0 = time.time()
        outputs = []  # list of {"filename": str, "path": str, "variant": str, "note": str}

        def _build_dynamic_variant():
            """Build the variant with AI-selected dynamic teaser."""
            if status_callback:
                status_callback("Selecting teaser clip for intro...")
            teaser_info = select_teaser(
                transcript_data,
                boundaries["sermon_start"],
                boundaries["sermon_end"],
                status_callback,
            )
            result["teaser"] = teaser_info

            # Extract teaser from the same audio file Whisper transcribed.
            # Add a 250ms pre-roll before the start — Whisper word timestamps
            # mark where a word is DETECTED (mid-word onset), which is often
            # slightly after the actual audio onset. Without this pre-roll,
            # the first word of the teaser can be clipped.
            import subprocess as _sp
            teaser_audio_path = os.path.join(work_dir, "teaser.wav")
            PRE_ROLL = 0.25
            t_start = max(0, teaser_info["teaser_start"] - PRE_ROLL)
            t_end = teaser_info["teaser_end"]
            teaser_dur = t_end - t_start

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{t_start:.3f}",
                "-i", teaser_source_path,
                "-t", f"{teaser_dur:.3f}",
                "-acodec", "pcm_s16le",
                "-ar", "48000", "-ac", "1",
                teaser_audio_path,
            ]
            ex = _sp.run(cmd, capture_output=True, text=True)
            if ex.returncode != 0:
                raise RuntimeError(f"Teaser extraction failed: {ex.stderr[-200:]}")

            import soundfile as sf
            logger.info(
                f"Teaser extracted from {os.path.basename(teaser_source_path)}: "
                f"{t_start:.1f}s - {t_end:.1f}s ({teaser_dur:.1f}s, with {PRE_ROLL*1000:.0f}ms pre-roll)"
            )

            # Mix teaser into intro
            teaser_data, teaser_sr = sf.read(teaser_audio_path, dtype="float64")
            mixed_intro_path = os.path.join(work_dir, "intro_with_teaser.wav")
            mix_teaser_into_intro(
                config.INTRO_PATH, teaser_data, teaser_sr, mixed_intro_path,
            )

            # Assemble
            dyn_filename = f"sermon_{run_id}_dynamic.mp3"
            dyn_path = os.path.join(config.OUTPUT_DIR, dyn_filename)
            assemble_broadcast(
                mixed_intro_path, sermon_fitted_path,
                config.OUTRO_PATH, dyn_path, status_callback,
            )
            return {"filename": dyn_filename, "path": dyn_path, "variant": "dynamic", "note": ""}

        def _build_stock_variant():
            """Build the variant with the pre-mixed stock intro."""
            stock_intro_path = os.path.join(config.ASSETS_DIR, "intro_stock.mp3")
            if not os.path.exists(stock_intro_path):
                raise RuntimeError(
                    f"Stock intro file not found at {stock_intro_path}. "
                    f"Add a pre-mixed intro_stock.mp3 to the assets folder."
                )
            if status_callback:
                status_callback("Assembling broadcast with stock intro...")
            stock_filename = f"sermon_{run_id}_stock.mp3"
            stock_path = os.path.join(config.OUTPUT_DIR, stock_filename)
            assemble_broadcast(
                stock_intro_path, sermon_fitted_path,
                config.OUTRO_PATH, stock_path, status_callback,
            )
            return {"filename": stock_filename, "path": stock_path, "variant": "stock", "note": ""}

        if include_bumpers_dynamic:
            try:
                outputs.append(_build_dynamic_variant())
            except Exception as e:
                logger.warning(f"Dynamic teaser variant failed: {e}")
                if status_callback:
                    status_callback(f"Dynamic teaser failed: {e}")
                # Auto-fallback: if dynamic failed and stock wasn't already requested, build stock
                if not include_bumpers_stock:
                    if status_callback:
                        status_callback("Falling back to stock teaser variant...")
                    try:
                        stock_out = _build_stock_variant()
                        stock_out["note"] = "Auto-generated as fallback (dynamic teaser failed)"
                        outputs.append(stock_out)
                    except Exception as e2:
                        logger.error(f"Stock fallback also failed: {e2}")
                        raise

        if include_bumpers_stock:
            outputs.append(_build_stock_variant())

        if not any_bumpers:
            # Sermon only
            sermon_filename = f"sermon_{run_id}.mp3"
            sermon_output_path = os.path.join(config.OUTPUT_DIR, sermon_filename)
            shutil.copy2(sermon_fitted_path, sermon_output_path)
            outputs.append({
                "filename": sermon_filename,
                "path": sermon_output_path,
                "variant": "sermon-only",
                "note": "",
            })

        # Set primary output (first in list) for backward compat
        if outputs:
            result["output_path"] = outputs[0]["path"]
            result["output_filename"] = outputs[0]["filename"]
        result["outputs"] = outputs
        result["include_bumpers"] = any_bumpers
        result["timing"]["assembly"] = time.time() - t0

        # Get duration of the primary output (first variant)
        import subprocess as _sp
        primary_path = outputs[0]["path"] if outputs else None
        final_duration = 0.0
        if primary_path:
            dur_result = _sp.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", primary_path],
                capture_output=True, text=True,
            )
            if dur_result.returncode == 0 and dur_result.stdout.strip():
                final_duration = float(dur_result.stdout.strip())
            else:
                from pipeline.audio_processor import get_audio_duration
                final_duration = get_audio_duration(primary_path)
        result["broadcast_duration"] = final_duration

        # Total time
        result["timing"]["total"] = sum(result["timing"].values())

        if status_callback:
            total = result["timing"]["total"]
            final_min = int(final_duration // 60)
            final_sec = int(final_duration % 60)
            variant_count = len(outputs)
            files_msg = "file ready" if variant_count == 1 else f"{variant_count} files ready"
            status_callback(
                f"Pipeline complete in {total / 60:.1f} minutes. "
                f"Final duration: {final_min}m {final_sec}s. "
                f"{files_msg} for download."
            )

        logger.info(
            f"Pipeline complete. Outputs: {[o['filename'] for o in outputs]}"
        )
        # Success — clear cache so next run gets a fresh teaser
        for suffix in ["audio.wav", "transcript.json"]:
            cache_file = _cache_path(vid, suffix)
            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            except OSError:
                pass

        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if status_callback:
            status_callback(f"ERROR: {e}")
        # Preserve cache on failure so retries don't re-download/re-transcribe
        logger.info(
            "Cache preserved for retry — audio and transcript will be reused "
            "on next run with the same URL/file"
        )
        raise

    finally:
        # Always clean up the working directory
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
