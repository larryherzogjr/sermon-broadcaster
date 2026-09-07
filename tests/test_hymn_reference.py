"""Optional real-model regression against the user-provided reference service.

HYMN_REFERENCE_AUDIO must point to the complete, untrimmed kbz86frmaYc recording.
The short word fixture isolates boundary refinement from transcription variance.
"""
import os

import pytest

from pipeline.boundary_detector import detect_boundaries


@pytest.mark.skipif(not os.getenv("HYMN_REFERENCE_AUDIO"), reason="Reference service audio not configured")
def test_september_six_reference_service():
    transcript = {"words": [
        {"word": "Amen.", "start": 3643.94, "end": 3644.08},
        {"word": "Our", "start": 3648.68, "end": 3648.8},
        {"word": "closing", "start": 3648.8, "end": 3649.1},
        {"word": "hymn", "start": 3649.1, "end": 3649.42},
    ]}
    result = detect_boundaries(os.environ["HYMN_REFERENCE_AUDIO"], transcript)
    assert result["sermon_start"] == pytest.approx(1973.46, abs=0.5)
    assert result["start_method"] == "first_speech_after_hymn"
    assert result["sermon_end"] == pytest.approx(3644.38, abs=0.1)
    assert result["end_method"] == "closing_amen"
    hymns = [event for event in result["music_events"] if event["is_hymn"]]
    assert len(hymns) == 3
    assert hymns[-1]["start"] == pytest.approx(3669.12, abs=1.0)
    closing_responses = [event for event in result["music_events"] if event["start"] > 4700]
    assert closing_responses
    assert all(not event["is_hymn"] for event in closing_responses)
