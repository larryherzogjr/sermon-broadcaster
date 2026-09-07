import pytest

from pipeline import boundary_detector as detector
from pipeline.music_detector import group_music_frames


def music(start, end, *, singing=True):
    return [{"start": float(t), "end": min(float(t + 1), end),
             "music": 0.9, "speech": 0.01,
             "singing": 0.7 if singing else 0.0, "choir": 0.0}
            for t in range(start, int(end) + (end % 1 > 0))]


def words(text, start, step=0.4):
    return [{"word": token, "start": start + i * step,
             "end": start + i * step + step / 2}
            for i, token in enumerate(text.split())]


def service_events():
    return group_music_frames(
        music(0, 150, singing=False)  # prelude
        + music(400, 550)  # hymn 1
        + music(900, 1080)  # special music may itself be another hymn
        + music(1500, 1650)  # hymn 2
        + music(3400, 3590)  # hymn 3
        + music(3700, 4100, singing=False)  # later instrumental music
        + music(4300, 4335) + music(4337, 4372)  # short closing responses
    )


def test_selects_last_two_hymns_ignoring_special_music_and_short_responses():
    result = detector.select_sermon_range(service_events(), {}, 4400)
    assert result["sermon_start"] == 1650
    assert result["sermon_end"] == 3400
    assert result["end_method"] == "before_hymn"
    assert "scripture_start" not in result
    assert "sermon_end_without_prayer" not in result


def test_closing_prayer_is_kept_and_hymn_announcement_excluded():
    transcript = {"words": (
        words("Amen. Would you bow with me in prayer?", 3300)
        + words("We pray this in Jesus name. Amen.", 3370)
        + words("Our closing hymn today is number 450.", 3380)
        + words("Let's stand together and sing.", 3385)
    )}
    result = detector.select_sermon_range(service_events(), transcript, 4400)
    assert result["end_method"] == "closing_amen"
    assert result["sermon_end"] == pytest.approx(3372.9)
    assert result["sermon_start"] == 1650


@pytest.mark.parametrize("announcement", [
    "Please stand as we sing our final hymn.",
    "Let us stand together and sing.",
    "Let's rise to sing.",
    "Our closing hymn is number 450.",
    "Please sing with us.",
    "Turn in your hymnals to number 450.",
])
def test_amen_before_hymn_announcements(announcement):
    transcript = {"words": words("Amen.", 3370) + words(announcement, 3380)}
    assert detector.closing_prayer_end(transcript, 1650, 3400) == pytest.approx(3370.5)


def test_missing_closing_amen_does_not_use_the_amen_before_prayer():
    transcript = {"words": words("Amen. Let us pray. Lord help us.", 3370)
                  + words("Our closing hymn is number 450.", 3380)}
    result = detector.select_sermon_range(service_events(), transcript, 4400)
    assert result["sermon_end"] == 3400
    assert result["end_method"] == "before_hymn"


def test_no_word_timestamps_falls_back_instead_of_guessing_inside_segment():
    transcript = {"segments": [{"start": 3370, "end": 3390,
                                "text": "Amen. Our closing hymn is number 450."}]}
    assert detector.closing_prayer_end(transcript, 1650, 3400) is None


def test_final_spoken_amen_before_music_is_usable_without_announcement():
    assert detector.closing_prayer_end({"words": words("Amen.", 3390)}, 1650, 3400) == pytest.approx(3390.5)


def test_search_does_not_reach_earlier_amens_or_cross_long_gaps():
    assert detector.closing_prayer_end({"words": words("Amen.", 2000)}, 1650, 3400) is None
    transcript = {"words": words("Amen.", 3340) + words("Our closing hymn", 3380)}
    assert detector.closing_prayer_end(transcript, 1650, 3400) is None


def test_amen_padding_cannot_include_announcement():
    transcript = {"words": words("Amen.", 3380) + words("Please stand as we sing", 3380.25)}
    assert detector.closing_prayer_end(transcript, 1650, 3400) == 3380.25


def test_missing_or_overlapping_hymns_require_manual_selection():
    with pytest.raises(detector.BoundaryDetectionError, match="Fewer than two"):
        detector.select_sermon_range([], {}, 4400)
    events = [{"start": 0, "end": 180, "is_hymn": True},
              {"start": 170, "end": 300, "is_hymn": True}]
    with pytest.raises(detector.BoundaryDetectionError, match="do not bracket"):
        detector.select_sermon_range(events, {}, 4400)
