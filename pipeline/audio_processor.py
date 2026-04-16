"""
Module 4: Audio Processor
Handles sermon extraction, silence manipulation, and tempo adjustment
to hit the exact target duration.
"""
import os
import logging
import subprocess
import tempfile

import numpy as np
import soundfile as sf

import config

logger = logging.getLogger(__name__)


def _diag_check(filepath, label, status_callback=None):
    """Diagnostic: check audio levels and log."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", filepath, "-af", "volumedetect", "-f", "null", "/dev/null"],
            capture_output=True, text=True
        )
        output = result.stderr
        mean_vol = max_vol = duration = "?"
        for line in output.split("\n"):
            if "mean_volume" in line:
                mean_vol = line.split("mean_volume:")[1].strip()
            if "max_volume" in line:
                max_vol = line.split("max_volume:")[1].strip()
            if "Duration" in line and "bitrate" in line:
                duration = line.split("Duration:")[1].split(",")[0].strip()
        msg = f"[DIAG] {label}: dur={duration}, mean={mean_vol}, max={max_vol}"
        logger.info(msg)
        if status_callback:
            status_callback(msg)
    except Exception as e:
        logger.error(f"[DIAG] {label}: failed: {e}")


def _seconds_to_ms(seconds: float) -> int:
    return int(seconds * 1000)


def _parse_duration(duration_str: str) -> float:
    """Parse duration string (MM:SS or HH:MM:SS) to seconds."""
    parts = duration_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        raise ValueError(f"Invalid duration format: {duration_str}")


def extract_segment(audio_path: str, start: float, end: float, output_path: str) -> str:
    """Extract a time segment from the audio file.
    Uses soundfile for reliable sample-accurate extraction.
    Adds a small pre-roll buffer and fade-in/fade-out for clean transitions.
    """
    # Read the full audio to get sample rate
    info = sf.info(audio_path)
    sr = info.samplerate

    # Pad start by 500ms to avoid clipping into the first word
    pre_roll = 0.5
    padded_start = max(0, start - pre_roll)

    # Calculate sample positions
    start_sample = int(padded_start * sr)
    end_sample = int(end * sr)

    # Read just the segment we need
    data, sr = sf.read(audio_path, start=start_sample, stop=end_sample, dtype='float64')

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Apply fade-in (300ms) and fade-out (500ms)
    fade_in_samples = int(0.3 * sr)
    fade_out_samples = int(0.5 * sr)

    if fade_in_samples > 0 and fade_in_samples < len(data):
        fade_in = np.linspace(0.0, 1.0, fade_in_samples)
        data[:fade_in_samples] *= fade_in

    if fade_out_samples > 0 and fade_out_samples < len(data):
        fade_out = np.linspace(1.0, 0.0, fade_out_samples)
        data[-fade_out_samples:] *= fade_out

    # Write as 16-bit PCM WAV at target sample rate
    sf.write(output_path, data, sr, subtype='PCM_16')

    duration = len(data) / sr
    logger.info(f"Extracted segment: {padded_start:.1f}s - {end:.1f}s ({duration:.1f}s) from {audio_path}")
    return output_path


def detect_pauses(audio_path: str, min_silence_ms: int = 300, silence_thresh_db: float = -30.0) -> list:
    """
    Detect pauses/silences in the audio using energy-based detection.
    Uses median RMS as reference (robust against loud transients).

    Returns list of dicts: {"start": float, "end": float, "duration": float} in seconds.
    """
    data, sr = sf.read(audio_path)

    # Convert to mono if needed
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Frame-based energy detection
    frame_length = int(sr * 0.02)  # 20ms frames
    hop_length = int(sr * 0.01)    # 10ms hop

    # Calculate RMS energy per frame
    num_frames = (len(data) - frame_length) // hop_length + 1
    rms = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        frame = data[start : start + frame_length]
        rms[i] = np.sqrt(np.mean(frame**2))

    # Use the 75th percentile of non-zero RMS as reference
    # This is robust against both loud transients and long silences
    nonzero_rms = rms[rms > 0]
    if len(nonzero_rms) == 0:
        logger.warning("Audio appears to be completely silent")
        return []

    ref_rms = np.percentile(nonzero_rms, 75)
    thresh = ref_rms * (10 ** (silence_thresh_db / 20.0))

    logger.info(
        f"Pause detection: ref_rms={ref_rms:.6f}, thresh={thresh:.6f}, "
        f"max_rms={np.max(rms):.6f}, median_rms={np.median(nonzero_rms):.6f}"
    )

    # Find silent frames
    is_silent = rms < thresh

    # Group consecutive silent frames into pauses
    pauses = []
    in_pause = False
    pause_start = 0

    for i, silent in enumerate(is_silent):
        time_sec = i * hop_length / sr
        if silent and not in_pause:
            pause_start = time_sec
            in_pause = True
        elif not silent and in_pause:
            pause_end = time_sec
            duration = pause_end - pause_start
            if duration >= min_silence_ms / 1000.0:
                pauses.append({
                    "start": pause_start,
                    "end": pause_end,
                    "duration": duration,
                })
            in_pause = False

    # Handle trailing silence
    if in_pause:
        pause_end = num_frames * hop_length / sr
        duration = pause_end - pause_start
        if duration >= min_silence_ms / 1000.0:
            pauses.append({
                "start": pause_start,
                "end": pause_end,
                "duration": duration,
            })

    total_pause_time = sum(p["duration"] for p in pauses)
    logger.info(
        f"Detected {len(pauses)} pauses totaling {total_pause_time:.1f}s "
        f"(longest: {max((p['duration'] for p in pauses), default=0):.1f}s)"
    )
    return pauses


def trim_silences(audio_path: str, pauses: list, output_path: str,
                  max_trim: float = None,
                  max_pause_ms: int = None, target_pause_ms: int = None) -> tuple:
    """
    Trim long silences, but only as much as needed.

    Args:
        audio_path: Input audio file
        pauses: List of detected pauses
        output_path: Output file path
        max_trim: Maximum total seconds to trim (None = trim all eligible)
        max_pause_ms: Only trim pauses longer than this (ms)
        target_pause_ms: Trim them down to this (ms)

    Returns: (output_path, time_saved_seconds)
    """
    if max_pause_ms is None:
        max_pause_ms = config.MAX_PAUSE_DURATION_MS
    if target_pause_ms is None:
        target_pause_ms = config.TARGET_PAUSE_DURATION_MS

    max_pause_s = max_pause_ms / 1000.0
    target_pause_s = target_pause_ms / 1000.0

    data, sr = sf.read(audio_path)

    # Find eligible pauses, sorted by duration descending (trim longest first)
    long_pauses = [p for p in pauses if p["duration"] > max_pause_s]
    long_pauses.sort(key=lambda p: p["duration"], reverse=True)

    if not long_pauses:
        logger.info("No pauses long enough to trim")
        sf.write(output_path, data, sr, subtype='PCM_16')
        return output_path, 0.0

    # Plan how much to trim from each pause
    trim_plan = []
    total_planned = 0.0

    for pause in long_pauses:
        available = pause["duration"] - target_pause_s
        if available <= 0:
            continue

        if max_trim is not None:
            remaining_budget = max_trim - total_planned
            if remaining_budget <= 0:
                break
            trim_here = min(available, remaining_budget)
        else:
            trim_here = available

        trim_plan.append({
            "start": pause["start"],
            "end": pause["end"],
            "duration": pause["duration"],
            "trim_amount": trim_here,
        })
        total_planned += trim_here

    logger.info(
        f"Trim plan: {len(trim_plan)} pauses, {total_planned:.1f}s total "
        f"(budget: {max_trim if max_trim else 'unlimited'})"
    )

    # Execute trims in reverse chronological order (to preserve offsets)
    trim_plan.sort(key=lambda p: p["start"], reverse=True)
    time_saved = 0.0

    for t in trim_plan:
        # Calculate how much of the pause to keep
        keep_duration = t["duration"] - t["trim_amount"]
        half_keep = keep_duration / 2

        keep_start = int((t["start"] + half_keep) * sr)
        cut_end = int((t["end"] - half_keep) * sr)

        if cut_end > keep_start:
            data = np.concatenate([data[:keep_start], data[cut_end:]])
            time_saved += t["trim_amount"]

    sf.write(output_path, data, sr, subtype='PCM_16')
    logger.info(f"Trimmed silences: saved {time_saved:.2f}s")
    return output_path, time_saved


def expand_silences(audio_path: str, pauses: list, time_to_add: float,
                    output_path: str) -> tuple:
    """
    Expand existing pauses to add time to the audio.
    Distributes added silence proportionally across existing pauses.

    Returns: (output_path, time_added_seconds)
    """
    min_pause_s = config.MIN_PAUSE_FOR_INSERT_MS / 1000.0
    max_insert_s = config.MAX_PAUSE_INSERT_MS / 1000.0

    # Find eligible pauses
    eligible = [p for p in pauses if p["duration"] >= min_pause_s]
    if not eligible:
        logger.warning("No eligible pauses found for expansion")
        return audio_path, 0.0

    # Calculate how much to add at each pause point
    per_pause = time_to_add / len(eligible)
    per_pause = min(per_pause, max_insert_s)

    data, sr = sf.read(audio_path)

    # Sort descending so inserts don't shift later indices
    eligible.sort(key=lambda p: p["start"], reverse=True)

    time_added = 0.0

    for pause in eligible:
        if time_added >= time_to_add:
            break

        add_here = min(per_pause, time_to_add - time_added)
        insert_samples = int(add_here * sr)

        # Insert silence at the midpoint of the pause
        insert_pos = int(((pause["start"] + pause["end"]) / 2) * sr)
        silence = np.zeros(insert_samples)
        data = np.concatenate([data[:insert_pos], silence, data[insert_pos:]])
        time_added += add_here

    sf.write(output_path, data, sr, subtype='PCM_16')
    logger.info(f"Expanded silences: added {time_added:.2f}s")
    return output_path, time_added


def adjust_tempo(audio_path: str, factor: float, output_path: str) -> str:
    """
    Adjust audio tempo using ffmpeg's atempo filter (time-stretch without pitch change).
    factor > 1.0 speeds up, factor < 1.0 slows down.
    """
    # ffmpeg atempo accepts 0.5 to 100.0
    # factor maps directly: 1.05 = play 5% faster (shorter), 0.95 = play 5% slower (longer)
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-af", f"atempo={factor}",
        "-acodec", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg tempo adjustment failed: {result.stderr}")

    # Validate output duration
    input_dur = get_audio_duration(audio_path)
    output_dur = get_audio_duration(output_path)
    expected_dur = input_dur / factor
    if abs(output_dur - expected_dur) > 5.0:
        raise RuntimeError(
            f"Tempo adjustment produced unexpected duration: "
            f"expected ~{expected_dur:.1f}s, got {output_dur:.1f}s"
        )

    logger.info(f"Tempo adjusted by factor {factor:.4f}: {input_dur:.1f}s -> {output_dur:.1f}s")
    return output_path


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    data, sr = sf.read(audio_path)
    return len(data) / sr


def encode_final(audio_path: str, output_path: str) -> str:
    """Encode final audio to broadcast-spec MP3."""
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-acodec", "libmp3lame",
        "-b:a", config.OUTPUT_BITRATE,
        "-ar", str(config.OUTPUT_SAMPLE_RATE),
        "-ac", "1",  # mono for AM/FM broadcast
        # Loudness normalization for broadcast
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed: {result.stderr}")

    logger.info(f"Final encoding complete: {output_path}")
    return output_path


def fit_to_duration(sermon_audio_path: str, target_duration_str: str,
                    output_path: str, status_callback=None) -> dict:
    """
    Main function: fit sermon audio to exact target duration.

    Strategy:
    1. Detect pauses in the sermon audio
    2. If too long: trim silences first, then speed up if needed
    3. If too short: expand silences first, then slow down if needed
    4. Encode to broadcast MP3

    Returns dict with processing details.
    """
    target_seconds = _parse_duration(target_duration_str)
    current_duration = get_audio_duration(sermon_audio_path)
    delta = current_duration - target_seconds

    logger.info(
        f"Current: {current_duration:.1f}s, Target: {target_seconds:.1f}s, "
        f"Delta: {delta:+.1f}s ({delta / 60:+.1f} min)"
    )

    if status_callback:
        if delta > 0:
            status_callback(
                f"Sermon is {abs(delta):.0f}s ({abs(delta)/60:.1f} min) too long — "
                f"trimming silences and adjusting..."
            )
        else:
            status_callback(
                f"Sermon is {abs(delta):.0f}s ({abs(delta)/60:.1f} min) too short — "
                f"expanding pauses and adjusting..."
            )

    # Detect pauses
    pauses = detect_pauses(sermon_audio_path)
    working_path = sermon_audio_path

    processing_log = {
        "original_duration": current_duration,
        "target_duration": target_seconds,
        "initial_delta": delta,
        "silence_adjustment": 0.0,
        "tempo_factor": 1.0,
        "final_duration": 0.0,
    }

    if abs(delta) < 0.5:
        # Close enough, just encode
        logger.info("Duration already within tolerance, encoding directly")
    elif delta > 0:
        # Too long — trim silences first (only as much as needed)
        trimmed_path = sermon_audio_path.replace(".wav", "_trimmed.wav")
        _, time_saved = trim_silences(working_path, pauses, trimmed_path, max_trim=delta)
        working_path = trimmed_path
        processing_log["silence_adjustment"] = -time_saved
        _diag_check(working_path, "after_trim", status_callback)
    else:
        # Too short — expand silences first
        time_needed = abs(delta)
        expanded_path = sermon_audio_path.replace(".wav", "_expanded.wav")
        _, time_added = expand_silences(working_path, pauses, time_needed, expanded_path)
        working_path = expanded_path
        processing_log["silence_adjustment"] = time_added
        _diag_check(working_path, "after_expand", status_callback)

    # After silence manipulation, check if tempo adjustment is needed
    current_duration = get_audio_duration(working_path)
    remaining_delta = current_duration - target_seconds

    if abs(remaining_delta) > 0.5:
        # Need tempo adjustment to hit target exactly
        factor = current_duration / target_seconds

        # Clamp to safe range
        if factor > config.MAX_SPEEDUP:
            logger.warning(
                f"Required speedup {factor:.3f}x exceeds max {config.MAX_SPEEDUP}x. "
                f"Output will be longer than target."
            )
            factor = config.MAX_SPEEDUP
        elif factor < config.MAX_SLOWDOWN:
            logger.warning(
                f"Required slowdown {factor:.3f}x exceeds max {config.MAX_SLOWDOWN}x. "
                f"Output will be shorter than target."
            )
            factor = config.MAX_SLOWDOWN

        direction = "up" if factor > 1.0 else "down"
        if status_callback:
            status_callback(
                f"Adjusting tempo {direction} by {abs(factor - 1.0) * 100:.1f}%..."
            )

        tempo_path = working_path.replace(".wav", "_tempo.wav")
        if tempo_path == working_path:
            tempo_path = working_path + "_tempo.wav"
        adjust_tempo(working_path, factor, tempo_path)
        working_path = tempo_path
        processing_log["tempo_factor"] = factor
        _diag_check(working_path, "after_tempo", status_callback)

    # Final encode
    if status_callback:
        status_callback("Encoding final broadcast audio...")

    encode_final(working_path, output_path)
    processing_log["final_duration"] = get_audio_duration(output_path)

    if status_callback:
        final_dur = processing_log["final_duration"]
        target_dur = processing_log["target_duration"]
        diff = final_dur - target_dur
        status_callback(
            f"Complete! Final duration: {final_dur / 60:.0f}m {final_dur % 60:.0f}s "
            f"(target delta: {diff:+.1f}s)"
        )

    return processing_log
