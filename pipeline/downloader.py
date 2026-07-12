"""
Module 1: YouTube Audio Downloader
Downloads audio from a YouTube URL and saves as WAV for processing.

Hardened against the common server-side failure modes:
  - transient network / throttling hiccups  -> retry with exponential backoff
  - YouTube bot-detection ("Sign in to confirm you're not a bot")
        -> rotate player clients + optional cookies file
  - unavailable / private / members-only / not-yet-live videos
        -> fail fast with a clear, actionable error (no pointless retries)

Every failure raises RuntimeError with a human-readable message; that string is
what lands in the job's `error` column, so keep it specific and actionable.
"""
import os
import time
import logging

import yt_dlp

import config

logger = logging.getLogger(__name__)


class DownloadRetryableError(RuntimeError):
    """Transient failure (throttling, network, 429) — worth retrying."""


class DownloadPermanentError(RuntimeError):
    """Failure that retrying won't fix (private, removed, members-only, etc.)."""


def _classify(raw: str):
    """
    Map a yt-dlp error string to (retryable: bool, friendly: str).

    The friendly message is written to the job's error field, so it names the
    likely cause and — where relevant — the operator fix (cookies file, etc.).
    """
    low = raw.lower()

    # ── Permanent: retrying is pointless ────────────────────────────────
    # Age check comes first: "Sign in to confirm your age" would otherwise be
    # swallowed by the looser bot-detection phrasing below.
    if "confirm your age" in low or "inappropriate for some users" in low \
            or ("age" in low and "restrict" in low):
        return False, (
            "This video is age-restricted. Provide YTDLP_COOKIES_FILE from a "
            "logged-in, age-verified account to download it."
        )
    if "not a bot" in low or "confirm you're not a bot" in low:
        return False, (
            "YouTube is blocking this server with a bot-detection challenge "
            "(\"Sign in to confirm you're not a bot\"). Set YTDLP_COOKIES_FILE "
            "to a cookies.txt exported from a logged-in browser, and/or adjust "
            "YTDLP_PLAYER_CLIENTS."
        )
    if "private video" in low:
        return False, "This is a private video — it can't be downloaded."
    if "members-only" in low or "members only" in low or "join this channel" in low:
        return False, (
            "This video is members-only. A cookies.txt from an account with "
            "membership (YTDLP_COOKIES_FILE) is required to download it."
        )
    if "removed by the uploader" in low or "video unavailable" in low \
            or "no longer available" in low or "account associated with this video has been terminated" in low:
        return False, "This video is unavailable (removed, deleted, or terminated)."
    if "available in your country" in low or ("geo" in low and "restrict" in low):
        return False, "This video is geo-blocked and not available from this server's region."
    if "premieres in" in low or "this live event will begin" in low \
            or "will begin in" in low or ("live event" in low and "begin" in low):
        return False, "This is a scheduled premiere/livestream that hasn't started yet."
    if "requested format is not available" in low or "no video formats found" in low:
        return False, "No downloadable audio/video formats were found for this URL."

    # ── Retryable: transient conditions ─────────────────────────────────
    if "429" in low or "too many requests" in low or "throttl" in low:
        return True, "YouTube rate-limited this server (HTTP 429 / throttling)."
    if any(s in low for s in (
        "timed out", "timeout", "connection reset", "connection aborted",
        "temporary failure", "network is unreachable", "read operation",
        "unable to download", "http error 5", "giving up after",
        "fragment", "incomplete",
    )):
        return True, "Transient network/download error while fetching from YouTube."

    # Unknown — treat as retryable once; the raw text goes in the message.
    return True, f"Download failed: {raw.strip()[:300]}"


def _build_opts(output_path: str):
    """Assemble yt-dlp options with the hardening knobs from config."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        # Don't fan out to a whole playlist if the URL carries a list= param.
        "noplaylist": True,
        # Let yt-dlp handle its own internal retries per request, on top of our
        # outer attempt loop.
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "socket_timeout": 30,
        # Best-effort geo bypass for region-locked content.
        "geo_bypass": True,
    }

    clients = [c.strip() for c in (config.YTDLP_PLAYER_CLIENTS or "").split(",") if c.strip()]
    if clients:
        # yt-dlp tries these player clients in order; rotating them is the main
        # lever against bot-detection (the "web" client is blocked most often).
        opts["extractor_args"] = {"youtube": {"player_client": clients}}

    cookies = (config.YTDLP_COOKIES_FILE or "").strip()
    if cookies:
        if os.path.exists(cookies):
            opts["cookiefile"] = cookies
            logger.info(f"Using YouTube cookies file: {cookies}")
        else:
            logger.warning(
                f"YTDLP_COOKIES_FILE is set to '{cookies}' but the file does not "
                f"exist — continuing without cookies."
            )

    return opts


def download_audio(youtube_url: str, output_dir: str, status_callback=None) -> str:
    """
    Download audio from a YouTube video and save as WAV.

    Args:
        youtube_url: Full YouTube URL
        output_dir: Directory to save the downloaded audio
        status_callback: Optional callable for status updates

    Returns:
        Path to the downloaded WAV file

    Raises:
        RuntimeError: with an actionable message describing why the download
                      failed (this string is persisted to the job's error field).
    """
    if status_callback:
        status_callback("Downloading audio from YouTube...")

    output_path = os.path.join(output_dir, "raw_audio")
    wav_path = output_path + ".wav"
    max_attempts = max(1, config.YTDLP_MAX_ATTEMPTS)

    last_friendly = None
    for attempt in range(1, max_attempts + 1):
        try:
            with yt_dlp.YoutubeDL(_build_opts(output_path)) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                title = info.get("title", "Unknown")
                duration = info.get("duration", 0) or 0
                logger.info(f"Downloaded: {title} ({duration}s)")

            # Verify the postprocessor actually produced a usable file.
            if not os.path.exists(wav_path):
                raise DownloadRetryableError(
                    f"Expected WAV file not found at {wav_path} after download"
                )
            if os.path.getsize(wav_path) == 0:
                os.remove(wav_path)
                raise DownloadRetryableError("Downloaded audio file is empty (0 bytes)")

            if status_callback:
                status_callback(f"Downloaded: {title} ({duration // 60}m {duration % 60}s)")
            return wav_path

        except Exception as e:  # noqa: BLE001 — classify below, don't swallow
            retryable, friendly = _classify(str(e))
            last_friendly = friendly
            logger.warning(
                f"Download attempt {attempt}/{max_attempts} failed "
                f"({'retryable' if retryable else 'permanent'}): {e}"
            )

            if not retryable:
                if status_callback:
                    status_callback(f"Download failed: {friendly}")
                raise DownloadPermanentError(friendly) from e

            if attempt < max_attempts:
                backoff = 2 ** (attempt - 1) * 3  # 3s, 6s, 12s, ...
                if status_callback:
                    status_callback(
                        f"{friendly} Retrying in {backoff}s "
                        f"(attempt {attempt + 1}/{max_attempts})..."
                    )
                time.sleep(backoff)

    # Exhausted all attempts on a retryable error.
    msg = (
        f"Failed to download audio from {youtube_url} after {max_attempts} "
        f"attempts. Last error: {last_friendly}"
    )
    logger.error(msg)
    if status_callback:
        status_callback(f"Download failed: {last_friendly}")
    raise RuntimeError(msg)
