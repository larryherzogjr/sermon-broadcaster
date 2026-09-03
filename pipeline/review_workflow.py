"""Two-stage, human-reviewed sermon broadcast workflow.

The expensive source preparation (download, conversion, transcription, and AI
suggestions) is persisted per job. A user can then review exact sermon and
teaser boundaries before the existing fitting and assembly machinery runs.
"""
import json
import logging
import math
import os
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import soundfile as sf

import config
from pipeline.assembler import (
    assemble_broadcast,
    get_bumper_durations,
    mix_teaser_into_intro,
)
from pipeline.audio_processor import (
    extract_segment,
    fit_to_duration,
    get_audio_duration,
    remove_segment,
)
from pipeline.boundary_detector import detect_boundaries
from pipeline.downloader import download_audio
from pipeline.orchestrator import select_content_combination
from pipeline.teaser_selector import select_teaser
from pipeline.transcription import transcribe

logger = logging.getLogger(__name__)


def parse_duration(value: str) -> float:
    parts = str(value or "").strip().split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError("Duration must be in MM:SS or HH:MM:SS format")
    numbers = [int(p) for p in parts]
    if numbers[-1] >= 60 or (len(numbers) == 3 and numbers[-2] >= 60):
        raise ValueError("Seconds and the final minutes field must be below 60")
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def review_job_dir(job_id: str) -> str:
    if not job_id or any(c not in "0123456789_" for c in job_id):
        raise ValueError("Invalid job id")
    return os.path.join(config.REVIEW_DIR, job_id)


def _write_json(path: str, value) -> None:
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
    os.replace(temp_path, path)


def load_transcript(job_id: str) -> dict:
    path = os.path.join(review_job_dir(job_id), "transcript.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _time_after_cut(value: float, cut_start: float, cut_end: float,
                    actual_reduction: float) -> float:
    """Map a marker from the old working audio onto the newly spliced audio."""
    value = float(value)
    if value <= cut_start:
        return value
    if value >= cut_end:
        return max(cut_start, value - actual_reduction)
    return cut_start


def _transcript_after_cut(transcript: dict, cut_start: float, cut_end: float,
                          actual_reduction: float, new_duration: float) -> dict:
    """Drop cut transcript entries and shift later timestamps for editor display."""
    updated = dict(transcript or {})
    for collection_name in ("segments", "words"):
        kept = []
        for item in updated.get(collection_name, []) or []:
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < cut_end and end > cut_start:
                continue
            adjusted = dict(item)
            adjusted["start"] = _time_after_cut(
                start, cut_start, cut_end, actual_reduction
            )
            adjusted["end"] = _time_after_cut(
                end, cut_start, cut_end, actual_reduction
            )
            kept.append(adjusted)
        updated[collection_name] = kept
    updated["duration"] = new_duration
    if "segments" in updated:
        updated["full_text"] = " ".join(
            str(segment.get("text") or "").strip()
            for segment in updated["segments"]
            if str(segment.get("text") or "").strip()
        )
    return updated


def apply_working_cut(job_id: str, metadata: dict, selections: dict,
                      cut_start: float, cut_end: float) -> dict:
    """Permanently splice one range from a review job's editable working audio."""
    review = dict(metadata.get("review") or {})
    if not review:
        raise ValueError("Review data is no longer available for this job")
    try:
        cut_start = float(cut_start)
        cut_end = float(cut_end)
    except (TypeError, ValueError):
        raise ValueError("Cut boundaries must be valid times") from None
    if not math.isfinite(cut_start) or not math.isfinite(cut_end):
        raise ValueError("Cut boundaries must be finite times")

    normalized = normalize_selections(
        selections, manual_teaser_default=bool(review.get("manual_teaser"))
    )
    sermon_start = normalized["sermon_start"]
    sermon_end = normalized["sermon_end"]
    old_duration = float(review.get("audio_duration") or 0.0)
    if (
        cut_start < sermon_start or cut_end > sermon_end
        or cut_start < 0 or cut_end > old_duration or cut_end <= cut_start
    ):
        raise ValueError("Choose a valid cut range inside the selected sermon")
    if cut_end - cut_start < 0.05:
        raise ValueError("Choose at least 0.05 seconds to cut")
    if cut_start <= sermon_start and cut_end >= sermon_end:
        raise ValueError("A cut cannot remove the entire selected sermon")

    artifact_dir = review_job_dir(job_id)
    audio_path = os.path.join(artifact_dir, "raw_audio.wav")
    edited_path = os.path.join(artifact_dir, "raw_audio.editing.wav")
    waveform_path = os.path.join(artifact_dir, "waveform.json")
    edited_waveform_path = os.path.join(artifact_dir, "waveform.editing.json")
    try:
        remove_segment(audio_path, cut_start, cut_end, edited_path)
        new_duration = get_audio_duration(edited_path)
        actual_reduction = old_duration - new_duration
        if actual_reduction <= 0:
            raise RuntimeError("The selected cut did not shorten the working audio")
        waveform = _generate_waveform(edited_path, edited_waveform_path)
        os.replace(edited_path, audio_path)
        os.replace(edited_waveform_path, waveform_path)
    finally:
        for temporary_path in (edited_path, edited_waveform_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass

    for key in ("sermon_start", "sermon_end", "teaser_start", "teaser_end"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = _time_after_cut(
                value, cut_start, cut_end, actual_reduction
            )
    teaser_overlapped = (
        normalized.get("teaser_start") is not None
        and selections.get("teaser_start") is not None
        and selections.get("teaser_end") is not None
        and float(selections["teaser_start"]) < cut_end
        and float(selections["teaser_end"]) > cut_start
    )
    if teaser_overlapped:
        normalized["teaser_start"] = None
        normalized["teaser_end"] = None

    transcript = _transcript_after_cut(
        load_transcript(job_id), cut_start, cut_end, actual_reduction, new_duration
    )
    _write_json(os.path.join(artifact_dir, "transcript.json"), transcript)

    review.update(normalized)
    review["cuts"] = []
    review["audio_duration"] = new_duration
    review["manual_teaser"] = normalized["manual_teaser"]
    review["edit_count"] = int(review.get("edit_count") or 0) + 1
    review["markers_confirmed"] = False
    if teaser_overlapped or not normalized["manual_teaser"]:
        review["teaser_text"] = ""
    metadata["review"] = review
    metadata["transcript_summary"] = {
        "segment_count": len(transcript.get("segments", [])),
        "word_count": len(transcript.get("words", [])),
        "duration": new_duration,
    }
    return {
        "review": review,
        "waveform": waveform,
        "transcript": transcript,
        "removed_duration": actual_reduction,
        "teaser_cleared": teaser_overlapped,
        "metadata": metadata,
    }


def _convert_upload(local_file: str, raw_audio_path: str, status_callback=None) -> None:
    if status_callback:
        status_callback("Converting uploaded file to review audio...")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", local_file, "-vn",
            "-acodec", "pcm_s16le", "-ar", "48000", raw_audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to convert uploaded file: {result.stderr[-300:]}")
    if status_callback:
        status_callback("Upload converted to audio.")


def _generate_waveform(audio_path: str, output_path: str, bins: int = 1600) -> dict:
    """Generate a compact peak envelope without loading the full service into RAM."""
    info = sf.info(audio_path)
    frames_per_bin = max(1, math.ceil(info.frames / bins))
    peaks = []
    with sf.SoundFile(audio_path) as source:
        while len(peaks) < bins:
            block = source.read(frames_per_bin, dtype="float32", always_2d=True)
            if not len(block):
                break
            mono = block.mean(axis=1)
            peaks.append(round(float(np.max(np.abs(mono))) if len(mono) else 0.0, 5))
    payload = {
        "duration": float(info.frames / info.samplerate),
        "sample_rate": info.samplerate,
        "peaks": peaks,
    }
    _write_json(output_path, payload)
    return payload


def sermon_target_seconds(target_duration: str, include_dynamic: bool,
                          include_stock: bool = False) -> tuple[float, dict]:
    total = parse_duration(target_duration)
    variants = {}
    if include_dynamic:
        variants["dynamic"] = get_bumper_durations()
    if include_stock:
        stock_intro = os.path.join(config.ASSETS_DIR, "intro_stock.mp3")
        variants["stock"] = get_bumper_durations(stock_intro, config.OUTRO_PATH)

    if not variants:
        bumper_durations = {"intro": 0.0, "outro": 0.0, "variants": {}}
    else:
        totals = {
            name: durations["intro"] + durations["outro"]
            for name, durations in variants.items()
        }
        if len(totals) > 1 and max(totals.values()) - min(totals.values()) > 0.25:
            raise ValueError(
                "The dynamic and stock intros have different lengths. Select one "
                "variant or install bumper files with matching total durations."
            )
        bumper_durations = dict(next(iter(variants.values())))
        bumper_durations["variants"] = variants

    sermon_target = total - bumper_durations["intro"] - bumper_durations["outro"]
    if sermon_target <= 0:
        if variants:
            raise ValueError("Target duration is shorter than the intro and outro")
        raise ValueError("Target duration must be greater than zero")
    return sermon_target, bumper_durations


def _initial_boundaries(transcript_data: dict, use_full_source: bool,
                        sermon_target: float, audio_duration: float,
                        status_callback=None) -> dict:
    if use_full_source:
        return {
            "sermon_start": 0.0,
            # The decoded WAV is the file the editor and renderer actually use.
            # Transcriber duration can drift beyond it by a fraction of a second.
            "sermon_end": float(audio_duration),
            "confidence": "manual",
            "sermon_title_guess": "Manual sermon selection",
        }

    boundaries = detect_boundaries(transcript_data, status_callback)
    if "sermon_end_with_prayer" in boundaries:
        selection = select_content_combination(boundaries, sermon_target, status_callback)
        boundaries["sermon_start"] = selection["start"]
        boundaries["sermon_end"] = selection["end"]
        boundaries["selection_label"] = selection["label"]
    if "sermon_start" not in boundaries or "sermon_end" not in boundaries:
        raise RuntimeError("Boundary analysis did not return a usable sermon range")
    return boundaries


def analyze_job(job_id: str, *, youtube_url: str = None, local_file: str = None,
                target_duration: str, include_dynamic: bool, include_stock: bool,
                sermon_only: bool, manual_selection: bool = False,
                manual_teaser: bool = False,
                status_callback=None) -> dict:
    """Prepare and persist everything required by the review editor."""
    started = time.time()
    artifact_dir = review_job_dir(job_id)
    os.makedirs(artifact_dir, exist_ok=True)
    raw_audio_path = os.path.join(artifact_dir, "raw_audio.wav")

    if youtube_url:
        downloaded = download_audio(youtube_url, artifact_dir, status_callback)
        if os.path.abspath(downloaded) != os.path.abspath(raw_audio_path):
            shutil.move(downloaded, raw_audio_path)
    elif local_file:
        _convert_upload(local_file, raw_audio_path, status_callback)
        try:
            os.remove(local_file)
        except OSError:
            pass
    else:
        raise ValueError("A YouTube URL or uploaded file is required")

    if manual_selection:
        if status_callback:
            status_callback("Manual selection enabled; skipping transcription.")
        transcript_data = {"segments": [], "words": [], "full_text": ""}
    else:
        if status_callback:
            status_callback("Transcribing audio for the review editor...")
        transcript_data = transcribe(raw_audio_path, status_callback)

    # Manual markers are chosen while listening to raw_audio.wav, so that same
    # file must be authoritative for teaser preview and final extraction. The
    # cloud transcriber's compressed helper file is intentionally not reused;
    # doing so could make a reviewed marker sound slightly different on output.
    transcript_data.pop("transcribed_audio_path", None)
    for helper_name in ("raw_audio_compressed.ogg", "raw_audio_compressed2.ogg"):
        try:
            os.remove(os.path.join(artifact_dir, helper_name))
        except OSError:
            pass
    teaser_source_name = "raw_audio.wav"

    if status_callback:
        status_callback("Building the waveform preview...")
    waveform = _generate_waveform(raw_audio_path, os.path.join(artifact_dir, "waveform.json"))
    if manual_selection:
        transcript_data["duration"] = float(waveform["duration"])
    _write_json(os.path.join(artifact_dir, "transcript.json"), transcript_data)

    target_seconds, bumpers = sermon_target_seconds(
        target_duration, include_dynamic, include_stock
    )
    boundaries = _initial_boundaries(
        transcript_data, sermon_only or manual_selection, target_seconds,
        float(waveform["duration"]), status_callback
    )

    teaser = None
    if include_dynamic and not manual_selection:
        try:
            teaser = select_teaser(
                transcript_data,
                float(boundaries["sermon_start"]),
                float(boundaries["sermon_end"]),
                status_callback,
            )
        except Exception as exc:  # A suggestion is helpful, but never blocks review.
            logger.warning("Teaser suggestion failed for %s: %s", job_id, exc)
            if status_callback:
                status_callback("No teaser suggestion was available; please select one manually.")

    review = {
        "audio_duration": float(waveform["duration"]),
        "target_duration": target_duration,
        "sermon_target_seconds": target_seconds,
        "bumper_durations": bumpers,
        "include_dynamic": bool(include_dynamic),
        "include_stock": bool(include_stock),
        "sermon_only": bool(sermon_only),
        "manual_selection": bool(manual_selection),
        "manual_teaser": bool(manual_teaser),
        "teaser_window_seconds": config.TEASER_WINDOW_END - config.TEASER_WINDOW_START,
        "suggested_sermon_start": float(boundaries["sermon_start"]),
        "suggested_sermon_end": float(boundaries["sermon_end"]),
        "sermon_start": float(boundaries["sermon_start"]),
        "sermon_end": float(boundaries["sermon_end"]),
        "teaser_start": float(teaser["teaser_start"]) if teaser else None,
        "teaser_end": float(teaser["teaser_end"]) if teaser else None,
        "teaser_text": teaser.get("teaser_text", "") if teaser else "",
        "title": boundaries.get("sermon_title_guess") or "Sermon",
        "confidence": boundaries.get("confidence") or "—",
        "analysis_seconds": round(time.time() - started, 1),
    }
    return {
        "review": review,
        "artifacts": {
            "audio": "raw_audio.wav",
            "transcript": "transcript.json",
            "waveform": "waveform.json",
            "teaser_source": teaser_source_name,
        },
        "boundaries": boundaries,
        "transcript_summary": {
            "segment_count": len(transcript_data.get("segments", [])),
            "word_count": len(transcript_data.get("words", [])),
            "duration": transcript_data.get("duration", waveform["duration"]),
        },
    }


def normalize_selections(selections: dict, manual_teaser_default: bool = False) -> dict:
    manual_teaser = selections.get("manual_teaser", manual_teaser_default)
    if not isinstance(manual_teaser, bool):
        raise ValueError("Manual teaser selection must be true or false")
    marker_values = [
        selections.get("sermon_start"), selections.get("sermon_end"),
        selections.get("teaser_start"), selections.get("teaser_end"),
    ]
    if any(isinstance(value, bool) for value in marker_values if value is not None):
        raise ValueError("Selection markers must be valid times")
    try:
        normalized = {
            "sermon_start": float(selections.get("sermon_start")),
            "sermon_end": float(selections.get("sermon_end")),
            "teaser_start": None,
            "teaser_end": None,
            "manual_teaser": manual_teaser,
            "cuts": [],
        }
        if selections.get("teaser_start") is not None:
            normalized["teaser_start"] = float(selections["teaser_start"])
        if selections.get("teaser_end") is not None:
            normalized["teaser_end"] = float(selections["teaser_end"])
        raw_cuts = selections.get("cuts") or []
        if not isinstance(raw_cuts, list) or len(raw_cuts) > 3:
            raise ValueError("Up to three cut selections are allowed")
        for cut in raw_cuts:
            if not isinstance(cut, dict):
                raise ValueError("Cut selections must have start and end times")
            if isinstance(cut.get("start"), bool) or isinstance(cut.get("end"), bool):
                raise ValueError("Cut selections must have valid times")
            normalized["cuts"].append({
                "start": float(cut.get("start")),
                "end": float(cut.get("end")),
            })
    except (TypeError, ValueError):
        raise ValueError("Selection markers must be valid times") from None
    if any(
        value is not None and not math.isfinite(value)
        for value in [
            normalized["sermon_start"], normalized["sermon_end"],
            normalized["teaser_start"], normalized["teaser_end"],
            *(point for cut in normalized["cuts"] for point in cut.values()),
        ]
    ):
        raise ValueError("Selection markers must be finite times")
    return normalized


def build_preflight(review: dict, selections: dict) -> dict:
    selections = normalize_selections(
        selections, manual_teaser_default=bool(review.get("manual_teaser"))
    )
    start = selections["sermon_start"]
    end = selections["sermon_end"]
    audio_duration = float(review.get("audio_duration") or 0.0)
    # Media containers and browser decoders can disagree by a few frames at
    # either edge. Treat an endpoint within one second of the decoded WAV edge
    # as that exact edge instead of making the user nudge valid full-file edits.
    if abs(start) <= 0.01:
        start = 0.0
    if abs(end - audio_duration) <= 1.0:
        end = audio_duration
    selections["sermon_start"] = start
    selections["sermon_end"] = end
    if start < 0 or end <= start or end > audio_duration:
        raise ValueError("Sermon start and end markers are outside the source audio")

    cuts = sorted(selections["cuts"], key=lambda cut: cut["start"])
    cut_duration = 0.0
    previous_end = None
    for cut in cuts:
        if cut["start"] < start or cut["end"] > end or cut["end"] <= cut["start"]:
            raise ValueError("Cut selections must be valid ranges inside the sermon")
        if previous_end is not None and cut["start"] < previous_end:
            raise ValueError("Cut selections cannot overlap")
        cut_duration += cut["end"] - cut["start"]
        previous_end = cut["end"]
    selections["cuts"] = cuts

    selected_duration = end - start - cut_duration
    if selected_duration <= 0:
        raise ValueError("Cut selections cannot remove the entire sermon")
    target = float(review["sermon_target_seconds"])
    difference = selected_duration - target
    blockers = []
    warnings = []

    if difference < -config.MAX_AUTOMATIC_SHORTFALL_SECONDS:
        blockers.append(
            f"The sermon selection is {format_duration(abs(difference))} shorter than "
            "the available broadcast time. Extend the selection rather than stretching it."
        )
    elif difference < -5:
        warnings.append(
            f"The selection is {format_duration(abs(difference))} short; pauses and tempo "
            "will be adjusted conservatively."
        )
    elif difference > 0 and selected_duration / target > config.MAX_SPEEDUP:
        warnings.append(
            f"The selection is {format_duration(difference)} long. Long pauses will be "
            "compressed first; rendering will stop if the safe tempo limit is not enough."
        )

    teaser_duration = None
    if review.get("include_dynamic"):
        teaser_start = selections.get("teaser_start")
        teaser_end = selections.get("teaser_end")
        automatic_manual_teaser = (
            review.get("manual_selection") and not selections["manual_teaser"]
        )
        if (teaser_start is None or teaser_end is None) and automatic_manual_teaser:
            pass
        elif teaser_start is None or teaser_end is None:
            blockers.append("Select a teaser start and end.")
        else:
            teaser_duration = teaser_end - teaser_start
            max_teaser = float(review["teaser_window_seconds"])
            if teaser_start < start or teaser_end > end:
                blockers.append("The teaser must be inside the selected sermon.")
            if any(teaser_start < cut["end"] and teaser_end > cut["start"] for cut in cuts):
                blockers.append("The teaser cannot overlap a section marked for cutting.")
            if teaser_duration < 3:
                blockers.append("The teaser is too short; select at least 3 seconds.")
            elif teaser_duration > max_teaser:
                blockers.append(
                    f"The teaser is {teaser_duration:.1f}s, but the intro has only "
                    f"{max_teaser:.1f}s available."
                )

    return {
        "ready": not blockers,
        "selected_duration": selected_duration,
        "sermon_target_seconds": target,
        "difference_seconds": difference,
        "teaser_duration": teaser_duration,
        "cut_duration": cut_duration,
        "blockers": blockers,
        "warnings": warnings,
        "selections": selections,
    }


def _extract_teaser(source_path: str, start: float, end: float, output_path: str) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", source_path,
            "-t", f"{end - start:.3f}", "-acodec", "pcm_s16le",
            "-ar", "48000", "-ac", "1", output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Teaser extraction failed: {result.stderr[-300:]}")


def create_teaser_preview(job_id: str, metadata: dict, selections: dict) -> str:
    review = metadata["review"]
    preflight = build_preflight(review, selections)
    teaser_errors = [b for b in preflight["blockers"] if "teaser" in b.lower()]
    if teaser_errors:
        raise ValueError(teaser_errors[0])
    selected = preflight["selections"]
    artifact_dir = review_job_dir(job_id)
    source_name = metadata["artifacts"].get("teaser_source", "raw_audio.wav")
    source_path = os.path.join(artifact_dir, os.path.basename(source_name))
    teaser_path = os.path.join(artifact_dir, "teaser_preview_clip.wav")
    preview_path = os.path.join(artifact_dir, "teaser_preview.wav")
    _extract_teaser(
        source_path, selected["teaser_start"], selected["teaser_end"], teaser_path
    )
    teaser_audio, sample_rate = sf.read(teaser_path, dtype="float64")
    mix_teaser_into_intro(config.INTRO_PATH, teaser_audio, sample_rate, preview_path)
    return preview_path


def _teaser_text(transcript: dict, start: float, end: float) -> str:
    words = [
        w.get("word", "").strip()
        for w in transcript.get("words", [])
        if float(w.get("start", 0)) >= start - 0.35 and float(w.get("end", 0)) <= end + 0.2
    ]
    return " ".join(w for w in words if w).strip()


def _validate_output_durations(outputs: list, requested_seconds: float) -> dict:
    durations = {}
    for output in outputs:
        duration = get_audio_duration(output["path"])
        variant = output.get("variant") or output.get("filename") or "output"
        durations[variant] = duration
        output["duration"] = duration
        difference = duration - requested_seconds
        if abs(difference) > config.FINAL_DURATION_TOLERANCE_SECONDS:
            direction = "long" if difference > 0 else "short"
            raise ValueError(
                f"The {variant} output is {format_duration(abs(difference))} too "
                f"{direction} after assembly. Verify the bumper files and render again."
            )
    return durations


def render_job(job_id: str, metadata: dict, selections: dict,
               status_callback=None) -> dict:
    """Render confirmed selections using the existing production audio stages."""
    review = metadata["review"]
    preflight = build_preflight(review, selections)
    if not preflight["ready"]:
        raise ValueError(" ".join(preflight["blockers"]))
    selected = preflight["selections"]
    transcript = load_transcript(job_id)

    artifact_dir = review_job_dir(job_id)
    raw_audio_path = os.path.join(artifact_dir, "raw_audio.wav")
    source_name = metadata["artifacts"].get("teaser_source", "raw_audio.wav")
    teaser_source_path = os.path.join(artifact_dir, os.path.basename(source_name))
    work_dir = os.path.join(
        config.WORK_DIR, f"review_{job_id}_{datetime.now().strftime('%H%M%S')}"
    )
    os.makedirs(work_dir, exist_ok=True)
    started = time.time()
    output_paths = []

    try:
        if status_callback:
            status_callback("Extracting the confirmed sermon selection...")
        sermon_raw = os.path.join(work_dir, "sermon_raw.wav")
        extract_segment(
            raw_audio_path, selected["sermon_start"], selected["sermon_end"],
            sermon_raw, cuts=selected["cuts"]
        )

        teaser_transcript = transcript
        rendered_teaser_source = teaser_source_path
        teaser_reason = "Manually confirmed in the review editor"
        if (
            review.get("include_dynamic")
            and review.get("manual_selection")
            and not selected["manual_teaser"]
        ):
            if status_callback:
                status_callback("Transcribing the edited sermon to select a teaser...")
            teaser_transcript = transcribe(sermon_raw, status_callback)
            teaser_transcript.pop("transcribed_audio_path", None)
            edited_duration = get_audio_duration(sermon_raw)
            if status_callback:
                status_callback("Selecting a teaser from the edited sermon...")
            generated_teaser = select_teaser(
                teaser_transcript, 0.0, edited_duration, status_callback
            )
            selected["teaser_start"] = float(generated_teaser["teaser_start"])
            selected["teaser_end"] = float(generated_teaser["teaser_end"])
            rendered_teaser_source = sermon_raw
            teaser_reason = generated_teaser.get("reason") or "Selected after the manual sermon edit"

        target_seconds = float(review["sermon_target_seconds"])
        target_string = format_duration(target_seconds + 1.0)
        sermon_fitted = os.path.join(work_dir, "sermon_fitted.mp3")
        processing = fit_to_duration(
            sermon_raw, target_string, sermon_fitted, status_callback
        )
        final_sermon_duration = float(processing.get("final_duration") or 0.0)
        if abs(final_sermon_duration - (target_seconds + 1.0)) > config.FINAL_DURATION_TOLERANCE_SECONDS:
            over_by = final_sermon_duration - (target_seconds + 1.0)
            if over_by > 0:
                raise ValueError(
                    f"The selection is still {format_duration(over_by)} too long after safe "
                    "pause and tempo adjustments. Shorten the sermon selection and render again."
                )
            raise ValueError(
                "The selected sermon could not be safely expanded to the target duration. "
                "Extend the sermon selection and render again."
            )

        outputs = []
        teaser_info = None
        if review.get("include_dynamic"):
            if status_callback:
                status_callback("Mixing the confirmed teaser into the intro...")
            teaser_path = os.path.join(work_dir, "teaser.wav")
            _extract_teaser(
                rendered_teaser_source,
                selected["teaser_start"],
                selected["teaser_end"],
                teaser_path,
            )
            teaser_audio, teaser_sr = sf.read(teaser_path, dtype="float64")
            mixed_intro = os.path.join(work_dir, "intro_with_teaser.wav")
            mix_teaser_into_intro(config.INTRO_PATH, teaser_audio, teaser_sr, mixed_intro)
            filename = f"sermon_{job_id}_dynamic.mp3"
            output_path = os.path.join(config.OUTPUT_DIR, filename)
            output_paths.append(output_path)
            assemble_broadcast(
                mixed_intro, sermon_fitted, config.OUTRO_PATH, output_path, status_callback
            )
            outputs.append({"filename": filename, "path": output_path, "variant": "dynamic", "note": ""})
            teaser_info = {
                "teaser_start": selected["teaser_start"],
                "teaser_end": selected["teaser_end"],
                "teaser_text": _teaser_text(
                    teaser_transcript, selected["teaser_start"], selected["teaser_end"]
                ),
                "reason": teaser_reason,
            }

        if review.get("include_stock"):
            stock_intro = os.path.join(config.ASSETS_DIR, "intro_stock.mp3")
            if not os.path.exists(stock_intro):
                raise RuntimeError("Stock intro file is not installed")
            filename = f"sermon_{job_id}_stock.mp3"
            output_path = os.path.join(config.OUTPUT_DIR, filename)
            output_paths.append(output_path)
            assemble_broadcast(
                stock_intro, sermon_fitted, config.OUTRO_PATH, output_path, status_callback
            )
            outputs.append({"filename": filename, "path": output_path, "variant": "stock", "note": ""})

        if not review.get("include_dynamic") and not review.get("include_stock"):
            filename = f"sermon_{job_id}.mp3"
            output_path = os.path.join(config.OUTPUT_DIR, filename)
            output_paths.append(output_path)
            shutil.copy2(sermon_fitted, output_path)
            outputs.append({"filename": filename, "path": output_path, "variant": "sermon-only", "note": ""})

        requested_duration = parse_duration(review["target_duration"])
        output_durations = _validate_output_durations(outputs, requested_duration)
        broadcast_duration = next(iter(output_durations.values()), 0.0)
        boundaries = dict(metadata.get("boundaries") or {})
        boundaries.update({
            "sermon_start": selected["sermon_start"],
            "sermon_end": selected["sermon_end"],
            "cuts": selected["cuts"],
            "confidence": "manual",
            "selection_label": "manually confirmed",
        })
        review_result = dict(review)
        review_result.update(selected)
        review_result["rendered"] = True

        return {
            "outputs": outputs,
            "boundaries": boundaries,
            "processing": processing,
            "teaser": teaser_info,
            "broadcast_duration": broadcast_duration,
            "output_durations": output_durations,
            "include_bumpers": bool(review.get("include_dynamic") or review.get("include_stock")),
            "timing": {"render": round(time.time() - started, 2)},
            "review": review_result,
            "transcript_summary": metadata.get("transcript_summary"),
        }
    except Exception:
        for output_path in output_paths:
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
