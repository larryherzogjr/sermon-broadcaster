import shutil

import numpy as np
import soundfile as sf

from pipeline import audio_processor


def test_short_audio_without_pauses_falls_back_to_tempo(tmp_path, monkeypatch):
    source = tmp_path / "speech.wav"
    output = tmp_path / "output.mp3"
    sf.write(source, np.full(48000, 0.25), 48000, subtype="PCM_16")

    def copy_audio(input_path, _factor_or_output, maybe_output=None):
        output_path = maybe_output or _factor_or_output
        shutil.copy2(input_path, output_path)
        return str(output_path)

    monkeypatch.setattr(audio_processor, "adjust_tempo", copy_audio)
    monkeypatch.setattr(audio_processor, "encode_final", copy_audio)
    monkeypatch.setattr(audio_processor, "_diag_check", lambda *args, **kwargs: None)

    result = audio_processor.fit_to_duration(str(source), "0:02", str(output))

    assert output.exists()
    assert result["silence_adjustment"] == 0.0
    assert result["tempo_factor"] == audio_processor.config.MAX_SLOWDOWN


def test_get_audio_duration_uses_file_metadata(tmp_path):
    source = tmp_path / "tone.wav"
    sf.write(source, np.zeros(24000), 48000, subtype="PCM_16")

    assert audio_processor.get_audio_duration(str(source)) == 0.5


def test_extract_segment_clamps_end_to_decoded_audio_boundary(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "selection.wav"
    sf.write(source, np.full(48000, 0.25), 48000, subtype="PCM_16")

    audio_processor.extract_segment(str(source), 0.0, 1.01, str(output))

    assert sf.info(output).frames == 48000


def test_extract_segment_does_not_reintroduce_audio_before_reviewed_start(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "selection.wav"
    # Loud hymn before the marker, quieter spoken material after it.
    sf.write(source, np.concatenate([np.full(48000, 0.8), np.full(48000, 0.2)]), 48000)
    audio_processor.extract_segment(str(source), 1.0, 2.0, str(output))
    data, _ = sf.read(output)
    assert len(data) == 48000
    assert np.max(data) < 0.21


def test_internal_cut_stays_aligned_with_nonzero_selection_start(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "selection.wav"
    sf.write(source, np.full(144000, 0.25), 48000)
    audio_processor.extract_segment(str(source), 1.0, 3.0, str(output),
                                    cuts=[{"start": 1.5, "end": 2.0}])
    assert sf.info(output).frames == 69600


def test_extract_segment_removes_manual_cut(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "selection.wav"
    sf.write(source, np.full(96000, 0.25), 48000, subtype="PCM_16")

    audio_processor.extract_segment(
        str(source), 0.0, 2.0, str(output), cuts=[{"start": 0.5, "end": 1.0}]
    )

    # The 50 ms crossfade overlaps the retained sides in addition to the cut.
    assert sf.info(output).frames == 69600


def test_expansion_caps_insertions_and_resulting_pauses(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "expanded.wav"
    sf.write(source, np.zeros(5000), 1000)
    pauses = [
        {"start": 0.0, "end": 0.3, "duration": 0.3},
        {"start": 1.0, "end": 2.3, "duration": 1.3},
        {"start": 3.0, "end": 4.6, "duration": 1.6},
    ]
    _, added = audio_processor.expand_silences(str(source), pauses, 10, str(output))
    # Short pause receives 0.5s; 1.3s pause gets only the room left below
    # the 1.5s cap after maximum slowdown. Already-long pause is untouched.
    expected = 0.5 + (1.5 * audio_processor.config.MAX_SLOWDOWN - 1.3)
    assert abs(added - expected) < 1e-9
    assert abs(sf.info(output).duration - (5 + expected)) < 0.002


def test_short_sermon_balances_pauses_and_slowdown(monkeypatch, tmp_path):
    source = str(tmp_path / "sermon.wav")
    durations = {source: 1455.0}
    requested_pause_time = []
    monkeypatch.setattr(audio_processor, "get_audio_duration", lambda path: durations[path])
    monkeypatch.setattr(audio_processor, "detect_pauses", lambda path: [])
    monkeypatch.setattr(audio_processor, "_diag_check", lambda *args: None)

    def expand(path, pauses, amount, output):
        requested_pause_time.append(amount)
        durations[output] = durations[path] + amount
        return output, amount

    def tempo(path, factor, output):
        durations[output] = durations[path] / factor
        return output

    def encode(path, output):
        durations[output] = durations[path]

    monkeypatch.setattr(audio_processor, "expand_silences", expand)
    monkeypatch.setattr(audio_processor, "adjust_tempo", tempo)
    monkeypatch.setattr(audio_processor, "encode_final", encode)
    result = audio_processor.fit_to_duration(source, "27:20", str(tmp_path / "out.mp3"))
    assert abs(result["tempo_factor"] - 0.96) < 1e-9
    assert abs(requested_pause_time[0] - 119.4) < 1e-9
    assert abs(result["final_duration"] - 1640) < 1e-9
