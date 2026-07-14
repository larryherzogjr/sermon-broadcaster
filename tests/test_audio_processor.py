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
