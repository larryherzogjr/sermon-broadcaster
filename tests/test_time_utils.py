from datetime import datetime

import config
from pipeline import time_utils


def test_local_timestamp_observes_central_daylight_time(monkeypatch):
    monkeypatch.setattr(config, "APP_TIMEZONE", "America/Chicago")

    assert time_utils.format_local_timestamp("2026-07-14T17:00:00+00:00") == (
        "2026-07-14 12:00:00 PM CDT"
    )


def test_local_timestamp_observes_central_standard_time(monkeypatch):
    monkeypatch.setattr(config, "APP_TIMEZONE", "America/Chicago")

    assert time_utils.format_local_timestamp("2026-01-14T18:00:00Z") == (
        "2026-01-14 12:00:00 PM CST"
    )


def test_legacy_naive_timestamp_is_interpreted_as_utc(monkeypatch):
    monkeypatch.setattr(config, "APP_TIMEZONE", "America/Chicago")

    assert time_utils.format_local_timestamp("2026-09-03T12:55:03") == (
        "2026-09-03 07:55:03 AM CDT"
    )


def test_persisted_timestamp_is_timezone_aware():
    value = datetime.fromisoformat(time_utils.utc_now_iso())

    assert value.utcoffset() is not None
