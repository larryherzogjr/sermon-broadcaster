import hashlib
import math

import numpy as np
import pytest
import soundfile as sf

from pipeline import music_detector as detector


def music(start, end, *, singing=True):
    return [{"start": float(t), "end": min(float(t + 1), end),
             "music": 0.9, "speech": 0.01,
             "singing": 0.7 if singing else 0.0, "choir": 0.0}
            for t in range(start, math.ceil(end))]


@pytest.mark.parametrize(("duration", "is_hymn"), [(5, False), (45, False), (89.999, False), (90, True), (150, True)])
def test_ninety_second_cutoff(duration, is_hymn):
    assert detector.group_music_frames(music(0, duration))[0]["is_hymn"] is is_hymn


def test_back_to_back_short_responses_are_still_excluded():
    result = detector.group_music_frames(music(0, 40) + music(43, 83))
    assert len(result) == 1
    assert result[0]["end"] == 83
    assert result[0]["is_hymn"] is False


def test_verse_gaps_are_joined_but_separate_events_are_not():
    result = detector.group_music_frames(music(0, 50) + music(55, 105) + music(111, 161))
    assert [(e["start"], e["end"], e["is_hymn"]) for e in result] == [
        (0, 105, True), (111, 161, False)
    ]


def test_hymn_includes_instrumental_intro_but_instrumental_event_is_not_hymn():
    result = detector.group_music_frames(music(0, 20, singing=False) + music(20, 120)
                                        + music(200, 500, singing=False))
    assert result[0]["start"] == 0
    assert result[0]["is_hymn"] is True
    assert result[1]["is_hymn"] is False


def test_scattered_music_predictions_cannot_make_a_hymn():
    frames = [music(t, t + 1)[0] for t in range(0, 150, 5)]
    result = detector.group_music_frames(frames)
    assert len(result) == 1
    assert result[0]["is_hymn"] is False


def test_speech_dominates_background_music():
    frames = music(0, 150)
    for frame in frames:
        frame["speech"] = 0.95
    assert detector.group_music_frames(frames) == []


def speech(start, end):
    frames = music(start, end, singing=False)
    for frame in frames:
        frame["speech"], frame["music"] = 0.95, 0.0
    return frames


def test_hymn_retains_first_speech_after_fading_chord():
    frames = music(0, 90)
    tail = music(90, 94, singing=False)
    for frame in tail:
        frame["music"] = 0.3  # Still audible, but no longer confidently music.
    # Do not skip a short initial word in favor of later sustained speech.
    result = detector.group_music_frames(frames + tail + speech(95, 96) + speech(100, 110))
    assert result[0]["end"] == 90
    assert result[0]["following_speech_start"] == 95


@pytest.mark.parametrize("speech_start", [90, 94, 104])
def test_speech_search_follows_actual_onset_not_fixed_delay(speech_start):
    result = detector.group_music_frames(music(0, 90) + speech(speech_start, speech_start + 2))
    assert result[0]["following_speech_start"] == speech_start


def test_speech_search_does_not_jump_to_distant_speech_or_across_more_music():
    result = detector.group_music_frames(music(0, 90) + speech(110, 115))
    assert result[0]["following_speech_start"] is None
    result = detector.group_music_frames(music(0, 90) + music(96, 98) + speech(100, 110))
    assert result[0]["following_speech_start"] is None


@pytest.mark.parametrize("seconds", [0.1, 0.48, 1.0, 30.72, 31.0, 61.8])
def test_chunk_scoring_covers_tail_once_without_padding_the_recording(tmp_path, seconds):
    path = tmp_path / "audio.wav"
    samples = int(seconds * detector.SAMPLE_RATE)
    sf.write(path, np.zeros(samples), detector.SAMPLE_RATE)
    calls = []

    class Session:
        def run(self, _outputs, inputs):
            size = len(inputs["waveform"])
            calls.append(size)
            count = 1 + math.ceil(max(0, size - detector.PATCH_SAMPLES) / detector.HOP_SAMPLES)
            return [np.zeros((count, 521), dtype=np.float32)]

    frames, duration = detector._score_wav(str(path), Session())
    assert len(frames) == math.ceil(samples / detector.HOP_SAMPLES)
    assert duration == seconds
    assert frames[0]["start"] == 0
    assert frames[-1]["end"] == seconds
    assert all(a["end"] == pytest.approx(b["start"]) for a, b in zip(frames, frames[1:]))
    assert max(calls) <= detector.CHUNK_HOPS * detector.HOP_SAMPLES + detector.PATCH_SAMPLES - detector.HOP_SAMPLES


def test_installed_model_is_verified_without_network(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test model")
    monkeypatch.setattr(detector.config, "HYMN_MODEL_PATH", str(model))
    monkeypatch.setattr(detector, "MODEL_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    monkeypatch.setattr(detector.requests, "get", lambda *_a, **_k: pytest.fail("must not download"))
    assert detector.install_model() == str(model)
    model.write_bytes(b"corrupt model")
    with pytest.raises(detector.MusicDetectionError, match="failed verification"):
        detector.install_model()


def test_bad_download_is_not_installed_and_temporary_file_is_removed(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    monkeypatch.setattr(detector.config, "HYMN_MODEL_PATH", str(model))

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def raise_for_status(self): pass
        def iter_content(self, _size): return iter([b"bad download"])

    monkeypatch.setattr(detector.requests, "get", lambda *_a, **_k: Response())
    with pytest.raises(detector.MusicDetectionError, match="failed verification"):
        detector.install_model()
    assert list(tmp_path.iterdir()) == []
