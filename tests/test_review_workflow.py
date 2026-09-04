import math
import os

import numpy as np
import pytest
import soundfile as sf

from pipeline import review_workflow


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("27:18", 1638), ("1:02:03", 3723), ("0:00", 0)],
)
def test_parse_duration(value, seconds):
    assert review_workflow.parse_duration(value) == seconds


@pytest.mark.parametrize("value", ["27", "1:60", "1:60:00", "hello", ""])
def test_parse_duration_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        review_workflow.parse_duration(value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_normalize_selections_rejects_non_finite_markers(value):
    with pytest.raises(ValueError, match="finite"):
        review_workflow.normalize_selections(
            {"sermon_start": value, "sermon_end": 120.0}
        )


def test_normalize_selections_rejects_boolean_markers():
    with pytest.raises(ValueError, match="valid times"):
        review_workflow.normalize_selections(
            {"sermon_start": True, "sermon_end": 120.0}
        )


def test_apply_working_cut_splices_audio_and_remaps_editor_state(tmp_path, monkeypatch):
    monkeypatch.setattr(review_workflow.config, "REVIEW_DIR", str(tmp_path))
    job_id = "20260903_120000_000001"
    artifact_dir = tmp_path / job_id
    artifact_dir.mkdir()
    audio_path = artifact_dir / "raw_audio.wav"
    sf.write(audio_path, np.full(192000, 0.2), 48000, subtype="PCM_16")
    review_workflow._write_json(
        str(artifact_dir / "transcript.json"),
        {
            "duration": 4.0,
            "segments": [
                {"start": 0.2, "end": 0.8, "text": "before"},
                {"start": 1.1, "end": 1.4, "text": "remove"},
                {"start": 2.0, "end": 2.5, "text": "after"},
            ],
            "words": [],
        },
    )
    metadata = {
        "review": {
            "audio_duration": 4.0,
            "sermon_start": 0.2,
            "sermon_end": 3.8,
            "manual_selection": True,
            "manual_teaser": True,
        }
    }

    result = review_workflow.apply_working_cut(
        job_id,
        metadata,
        {
            "sermon_start": 0.2,
            "sermon_end": 3.8,
            "teaser_start": 2.0,
            "teaser_end": 2.4,
            "manual_teaser": True,
        },
        1.0,
        1.5,
    )

    assert result["review"]["audio_duration"] == pytest.approx(3.45, abs=0.002)
    assert result["review"]["sermon_end"] == pytest.approx(3.25, abs=0.002)
    assert result["review"]["teaser_start"] == pytest.approx(1.45, abs=0.002)
    assert result["review"]["edit_count"] == 1
    assert [segment["text"] for segment in result["transcript"]["segments"]] == ["before", "after"]
    assert sf.info(audio_path).duration == pytest.approx(3.45, abs=0.002)

    before_second_cut = audio_path.read_bytes()
    second = review_workflow.apply_working_cut(
        job_id,
        result["metadata"],
        {
            "sermon_start": result["review"]["sermon_start"],
            "sermon_end": result["review"]["sermon_end"],
            "teaser_start": result["review"]["teaser_start"],
            "teaser_end": result["review"]["teaser_end"],
            "manual_teaser": True,
        },
        0.5,
        0.7,
    )
    assert second["review"]["audio_duration"] == pytest.approx(3.2, abs=0.002)
    assert second["review"]["edit_count"] == 2

    restored = review_workflow.undo_working_cut(job_id, second["metadata"])
    assert audio_path.read_bytes() == before_second_cut
    assert restored["review"]["audio_duration"] == pytest.approx(3.45, abs=0.002)
    assert sf.info(audio_path).duration == pytest.approx(3.45, abs=0.002)
    assert restored["review"]["sermon_end"] == pytest.approx(3.25, abs=0.002)
    assert restored["review"]["teaser_start"] == pytest.approx(1.45, abs=0.002)
    assert restored["review"]["edit_count"] == 1
    assert restored["review"]["undo_available"] is False
    assert restored["review"]["markers_confirmed"] is False
    assert review_workflow.load_transcript(job_id) == result["transcript"]
    with pytest.raises(ValueError, match="no cut to undo"):
        review_workflow.undo_working_cut(job_id, restored["metadata"])


def test_preflight_accepts_exact_target():
    review = {
        "audio_duration": 2000.0,
        "sermon_target_seconds": 1638.0,
        "include_dynamic": False,
    }

    result = review_workflow.build_preflight(
        review, {"sermon_start": 100.0, "sermon_end": 1738.0}
    )

    assert result["ready"] is True
    assert result["difference_seconds"] == 0.0


def test_manual_full_source_uses_decoded_audio_duration():
    boundaries = review_workflow._initial_boundaries(
        {"duration": 1800.8}, True, 1800.0, 1800.0
    )

    assert boundaries["sermon_start"] == 0.0
    assert boundaries["sermon_end"] == 1800.0


def test_manual_analysis_skips_transcription(tmp_path, monkeypatch):
    monkeypatch.setattr(review_workflow.config, "REVIEW_DIR", str(tmp_path))
    monkeypatch.setattr(
        review_workflow, "download_audio",
        lambda _url, output_dir, _status: f"{output_dir}/raw_audio.wav",
    )
    monkeypatch.setattr(
        review_workflow, "transcribe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual mode must not transcribe")
        ),
    )
    monkeypatch.setattr(
        review_workflow, "_generate_waveform",
        lambda *_args, **_kwargs: {"duration": 1800.0, "sample_rate": 48000, "peaks": []},
    )
    monkeypatch.setattr(
        review_workflow, "sermon_target_seconds",
        lambda *_args, **_kwargs: (1638.0, {"intro": 0.0, "outro": 0.0}),
    )

    result = review_workflow.analyze_job(
        "123456",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        target_duration="27:18",
        include_dynamic=False,
        include_stock=False,
        sermon_only=False,
        manual_selection=True,
    )

    assert result["review"]["manual_selection"] is True
    assert result["transcript_summary"]["segment_count"] == 0
    assert result["transcript_summary"]["word_count"] == 0


def test_preflight_snaps_small_end_drift_to_audio_boundary():
    review = {
        "audio_duration": 1800.0,
        "sermon_target_seconds": 1800.0,
        "include_dynamic": False,
    }

    result = review_workflow.build_preflight(
        review, {"sermon_start": 0.0, "sermon_end": 1800.7}
    )

    assert result["ready"] is True
    assert result["selections"]["sermon_end"] == 1800.0


def test_preflight_subtracts_interior_cuts_from_selected_duration():
    review = {
        "audio_duration": 2000.0,
        "sermon_target_seconds": 1600.0,
        "include_dynamic": False,
    }

    result = review_workflow.build_preflight(
        review,
        {
            "sermon_start": 100.0,
            "sermon_end": 1720.0,
            "cuts": [{"start": 300.0, "end": 320.0}],
        },
    )

    assert result["ready"] is True
    assert result["selected_duration"] == 1600.0
    assert result["cut_duration"] == 20.0


def test_preflight_rejects_overlapping_cuts():
    review = {
        "audio_duration": 2000.0,
        "sermon_target_seconds": 1600.0,
        "include_dynamic": False,
    }

    with pytest.raises(ValueError, match="cannot overlap"):
        review_workflow.build_preflight(
            review,
            {
                "sermon_start": 100.0,
                "sermon_end": 1700.0,
                "cuts": [
                    {"start": 300.0, "end": 330.0},
                    {"start": 320.0, "end": 340.0},
                ],
            },
        )


def test_manual_mode_can_defer_teaser_selection_until_render():
    review = {
        "audio_duration": 1700.0,
        "sermon_target_seconds": 1600.0,
        "include_dynamic": True,
        "manual_selection": True,
        "manual_teaser": False,
        "teaser_window_seconds": 23.0,
    }

    result = review_workflow.build_preflight(
        review, {"sermon_start": 50.0, "sermon_end": 1650.0}
    )

    assert result["ready"] is True
    assert result["teaser_duration"] is None


def test_editor_manual_teaser_toggle_requires_visible_teaser_markers():
    review = {
        "audio_duration": 100.0,
        "sermon_target_seconds": 90.0,
        "include_dynamic": True,
        "manual_selection": True,
        "manual_teaser": False,
        "teaser_window_seconds": 23.0,
    }

    automatic = review_workflow.build_preflight(
        review,
        {
            "sermon_start": 5.0,
            "sermon_end": 95.0,
            "manual_teaser": False,
        },
    )
    manual = review_workflow.build_preflight(
        review,
        {
            "sermon_start": 5.0,
            "sermon_end": 95.0,
            "manual_teaser": True,
        },
    )

    assert automatic["ready"] is True
    assert any("Select a teaser" in blocker for blocker in manual["blockers"])


def test_manual_render_generates_teaser_and_assembles_broadcast(tmp_path, monkeypatch):
    review_dir = tmp_path / "reviews"
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "output"
    for path in (review_dir / "123456", work_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(review_workflow.config, "REVIEW_DIR", str(review_dir))
    monkeypatch.setattr(review_workflow.config, "WORK_DIR", str(work_dir))
    monkeypatch.setattr(review_workflow.config, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(review_workflow, "load_transcript", lambda _job_id: {})

    calls = []

    def fake_extract(_source, _start, _end, output, cuts=None):
        calls.append(("extract", cuts))
        open(output, "wb").close()

    def fake_transcribe(path, _status):
        calls.append(("transcribe", os.path.basename(path)))
        return {
            "duration": 1600.0,
            "segments": [],
            "words": [{"word": "Hope", "start": 10.0, "end": 11.0}],
        }

    def fake_fit(_source, _target, output, _status):
        open(output, "wb").close()
        return {"final_duration": 1601.0, "original_duration": 1600.0}

    def fake_assemble(_intro, _sermon, _outro, output, _status):
        calls.append(("assemble", os.path.basename(output)))
        open(output, "wb").close()

    monkeypatch.setattr(review_workflow, "extract_segment", fake_extract)
    monkeypatch.setattr(review_workflow, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        review_workflow, "select_teaser",
        lambda *_args: {"teaser_start": 10.0, "teaser_end": 20.0, "reason": "Strong hook"},
    )
    monkeypatch.setattr(review_workflow, "get_audio_duration", lambda _path: 1600.0)
    monkeypatch.setattr(review_workflow, "fit_to_duration", fake_fit)
    monkeypatch.setattr(
        review_workflow, "_extract_teaser",
        lambda _source, _start, _end, output: open(output, "wb").close(),
    )
    monkeypatch.setattr(
        review_workflow.sf, "read",
        lambda _path, dtype=None: (review_workflow.np.zeros(48000), 48000),
    )
    monkeypatch.setattr(
        review_workflow, "mix_teaser_into_intro",
        lambda _intro, _audio, _rate, output: open(output, "wb").close(),
    )
    monkeypatch.setattr(review_workflow, "assemble_broadcast", fake_assemble)
    monkeypatch.setattr(
        review_workflow, "_validate_output_durations",
        lambda outputs, _requested: {"dynamic": 1770.0},
    )

    metadata = {
        "review": {
            "audio_duration": 1700.0,
            "target_duration": "29:30",
            "sermon_target_seconds": 1600.0,
            "include_dynamic": True,
            "include_stock": False,
            "manual_selection": True,
            "manual_teaser": False,
            "teaser_window_seconds": 23.0,
        },
        "artifacts": {"teaser_source": "raw_audio.wav"},
        "boundaries": {},
    }
    result = review_workflow.render_job(
        "123456", metadata,
        {"sermon_start": 50.0, "sermon_end": 1650.0, "cuts": []},
    )

    assert ("transcribe", "sermon_raw.wav") in calls
    assert any(call[0] == "assemble" for call in calls)
    assert result["outputs"][0]["variant"] == "dynamic"
    assert result["teaser"]["reason"] == "Strong hook"


def test_preflight_still_rejects_materially_out_of_bounds_end():
    review = {
        "audio_duration": 1800.0,
        "sermon_target_seconds": 1800.0,
        "include_dynamic": False,
    }

    with pytest.raises(ValueError, match="outside the source audio"):
        review_workflow.build_preflight(
            review, {"sermon_start": 0.0, "sermon_end": 1801.1}
        )


def test_sermon_target_uses_selected_stock_intro(monkeypatch):
    def fake_durations(intro_path=None, outro_path=None):
        if intro_path and intro_path.endswith("intro_stock.mp3"):
            return {"intro": 70.0, "outro": 30.0}
        return {"intro": 80.0, "outro": 30.0}

    monkeypatch.setattr(review_workflow, "get_bumper_durations", fake_durations)

    target, bumpers = review_workflow.sermon_target_seconds("20:00", False, True)

    assert target == 1100.0
    assert bumpers["variants"]["stock"]["intro"] == 70.0


def test_sermon_target_must_be_positive():
    with pytest.raises(ValueError, match="greater than zero"):
        review_workflow.sermon_target_seconds("0:00", False, False)


def test_mixed_variants_require_matching_bumper_lengths(monkeypatch):
    def fake_durations(intro_path=None, outro_path=None):
        if intro_path and intro_path.endswith("intro_stock.mp3"):
            return {"intro": 70.0, "outro": 30.0}
        return {"intro": 80.0, "outro": 30.0}

    monkeypatch.setattr(review_workflow, "get_bumper_durations", fake_durations)

    with pytest.raises(ValueError, match="different lengths"):
        review_workflow.sermon_target_seconds("20:00", True, True)


def test_output_duration_validation_checks_every_variant(monkeypatch):
    durations = {"dynamic.mp3": 1770.4, "stock.mp3": 1771.0}
    monkeypatch.setattr(
        review_workflow,
        "get_audio_duration",
        lambda path: durations[path],
    )
    outputs = [
        {"path": "dynamic.mp3", "variant": "dynamic"},
        {"path": "stock.mp3", "variant": "stock"},
    ]

    result = review_workflow._validate_output_durations(outputs, 1770.0)

    assert result == {"dynamic": 1770.4, "stock": 1771.0}
    assert outputs[1]["duration"] == 1771.0


def test_output_duration_validation_rejects_drift(monkeypatch):
    monkeypatch.setattr(review_workflow, "get_audio_duration", lambda _path: 1760.0)

    with pytest.raises(ValueError, match="too short"):
        review_workflow._validate_output_durations(
            [{"path": "bad.mp3", "variant": "dynamic"}], 1770.0
        )
