"""Timezone-safe timestamp helpers.

Persist instants in UTC and convert them only at presentation boundaries. Older
database rows were written by a UTC production host without an offset, so naive
ISO strings are treated as UTC for backward compatibility.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import config


def utc_now_iso():
    """Return an unambiguous ISO-8601 UTC timestamp for persistence."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_now():
    """Return the current time in the configured application timezone."""
    return datetime.now(ZoneInfo(config.APP_TIMEZONE))


def format_local_timestamp(value):
    """Format an ISO timestamp in the configured timezone.

    Timezone-less values are legacy values created on the UTC production host.
    """
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(ZoneInfo(config.APP_TIMEZONE)).strftime(
        "%Y-%m-%d %I:%M:%S %p %Z"
    )
