"""
Module 7: Audio Assembler
Mixes the teaser clip into the intro bumper and concatenates
intro + sermon + outro into the final broadcast file.
"""
import os
import logging
import subprocess

import numpy as np
import soundfile as sf

import config

logger = logging.getLogger(__name__)

# Teaser window — read from config
TEASER_WINDOW_START = config.TEASER_WINDOW_START
TEASER_WINDOW_END = config.TEASER_WINDOW_END
TEASER_WINDOW_DURATION = TEASER_WINDOW_END - TEASER_WINDOW_START

# How much to duck the intro music during the teaser
MUSIC_DUCK_LEVEL = 0.15  # 15% volume during teaser
DUCK_FADE_DURATION = 0.5  # fade down/up over 500ms


def _load_as_mono(filepath: str, target_sr: int = 44100) -> tuple:
    """Load an audio file as mono at the target sample rate."""
    # First convert to WAV at target sample rate using ffmpeg
    temp_wav = filepath + ".tmp.wav"
    cmd = [
        "ffmpeg", "-y", "-i", filepath,
        "-ar", str(target_sr), "-ac", "1",
        "-acodec", "pcm_s16le", temp_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to convert {filepath}: {result.stderr}")

    data, sr = sf.read(temp_wav, dtype="float64")
    os.remove(temp_wav)
    return data, sr


def mix_teaser_into_intro(intro_path: str, teaser_audio: np.ndarray,
                          teaser_sr: int, output_path: str) -> str:
    """
    Mix a teaser clip into the intro bumper at the teaser window.

    The intro music is ducked (lowered) during the window and the
    teaser is overlaid on top.

    Args:
        intro_path: Path to the intro MP3/WAV
        teaser_audio: Numpy array of the teaser audio (mono)
        teaser_sr: Sample rate of the teaser audio
        output_path: Path to write the mixed intro

    Returns:
        Path to the mixed intro WAV
    """
    # Load intro as mono at the teaser's sample rate
    intro_data, sr = _load_as_mono(intro_path, target_sr=teaser_sr)

    # Calculate sample positions for the teaser window
    window_start_sample = int(TEASER_WINDOW_START * sr)
    window_end_sample = int(TEASER_WINDOW_END * sr)
    window_length = window_end_sample - window_start_sample

    # Trim or pad teaser to fit the window
    if len(teaser_audio) > window_length:
        # Trim teaser to fit, with fade-out at the end
        teaser_clip = teaser_audio[:window_length].copy()
        fade_samples = int(0.3 * sr)
        if fade_samples < len(teaser_clip):
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            teaser_clip[-fade_samples:] *= fade_out
    elif len(teaser_audio) < window_length:
        # Place teaser at the START of the window, pad silence after
        pad_after = window_length - len(teaser_audio)
        teaser_clip = np.concatenate([
            teaser_audio,
            np.zeros(pad_after),
        ])
    else:
        teaser_clip = teaser_audio.copy()

    # Add fade-in to teaser
    fade_in_samples = int(0.2 * sr)
    if fade_in_samples < len(teaser_clip):
        fade_in = np.linspace(0.0, 1.0, fade_in_samples)
        teaser_clip[:fade_in_samples] *= fade_in

    # Mix teaser on top of intro at the teaser window start.
    # NOTE: We do NOT apply ducking here — the intro.mp3 file is already
    # pre-ducked in the teaser window. Adding more ducking would make
    # the music inaudibly quiet (15% × 15% = 2.25% volume).
    intro_mixed = intro_data.copy()
    end_pos = min(window_start_sample + len(teaser_clip), len(intro_mixed))
    intro_mixed[window_start_sample:end_pos] += teaser_clip[:end_pos - window_start_sample]

    # Clip to prevent distortion
    intro_mixed = np.clip(intro_mixed, -1.0, 1.0)

    sf.write(output_path, intro_mixed, sr, subtype="PCM_16")
    logger.info(f"Mixed teaser into intro: {output_path}")
    return output_path


def assemble_broadcast(intro_path: str, sermon_path: str, outro_path: str,
                       output_path: str, status_callback=None) -> str:
    """
    Concatenate intro + sermon + outro into the final broadcast file.
    
    IMPORTANT: No audio processing (loudnorm, tempo, etc.) is applied here.
    The intro and outro are used exactly as-is. The sermon already has
    loudnorm applied from the fitting stage.

    Args:
        intro_path: Path to the mixed intro (with teaser) WAV
        sermon_path: Path to the fitted sermon MP3/WAV
        outro_path: Path to the outro bumper MP3
        output_path: Path for the final output MP3

    Returns:
        Path to the final broadcast MP3
    """
    if status_callback:
        status_callback("Assembling final broadcast: intro + sermon + outro...")

    work_dir = os.path.dirname(intro_path)

    # Convert all three pieces to identical WAV format (44100 Hz, mono, PCM16)
    # so ffmpeg concat demuxer can join them cleanly
    pieces = []
    for i, (src, label) in enumerate([
        (intro_path, "intro"),
        (sermon_path, "sermon"),
        (outro_path, "outro"),
    ]):
        dst = os.path.join(work_dir, f"concat_{i}_{label}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le",
            dst,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert {label}: {result.stderr}")

        # Log duration of each piece
        dur_result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", dst],
            capture_output=True, text=True,
        )
        if dur_result.returncode == 0:
            dur = float(dur_result.stdout.strip())
            logger.info(f"[ASSEMBLE] {label}: {dur:.1f}s")

        pieces.append(dst)

    # Create concat list
    list_path = os.path.join(work_dir, "concat_list.txt")
    with open(list_path, "w") as f:
        for p in pieces:
            f.write(f"file '{os.path.abspath(p)}'\n")

    # Concatenate and encode to MP3 — NO audio filters
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-acodec", "libmp3lame",
        "-b:a", config.OUTPUT_BITRATE,
        "-ar", str(config.OUTPUT_SAMPLE_RATE),
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Broadcast assembly failed: {result.stderr}")

    # Clean up temp files
    for f_path in pieces + [list_path]:
        try:
            os.remove(f_path)
        except OSError:
            pass

    # Log final duration
    dur_result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", output_path],
        capture_output=True, text=True,
    )
    if dur_result.returncode == 0:
        total = float(dur_result.stdout.strip())
        logger.info(f"[ASSEMBLE] Final broadcast: {total:.1f}s ({total/60:.1f} min)")
        if status_callback:
            status_callback(f"Broadcast assembled: {total/60:.0f}m {total%60:.0f}s total")

    return output_path


def get_bumper_durations() -> dict:
    """Get the durations of the intro and outro bumpers."""
    intro_dur = 0.0
    outro_dur = 0.0

    for path, name in [(config.INTRO_PATH, "intro"), (config.OUTRO_PATH, "outro")]:
        if os.path.exists(path):
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                dur = float(result.stdout.strip())
                if name == "intro":
                    intro_dur = dur
                else:
                    outro_dur = dur

    return {"intro": intro_dur, "outro": outro_dur}
