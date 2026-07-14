import math

import pytest

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
