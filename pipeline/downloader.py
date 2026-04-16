"""
Module 1: YouTube Audio Downloader
Downloads audio from a YouTube URL and saves as WAV for processing.
"""
import os
import logging
import yt_dlp

logger = logging.getLogger(__name__)


def download_audio(youtube_url: str, output_dir: str, status_callback=None) -> str:
    """
    Download audio from a YouTube video and save as WAV.

    Args:
        youtube_url: Full YouTube URL
        output_dir: Directory to save the downloaded audio
        status_callback: Optional callable for status updates

    Returns:
        Path to the downloaded WAV file
    """
    if status_callback:
        status_callback("Downloading audio from YouTube...")

    output_path = os.path.join(output_dir, "raw_audio")

    ydl_opts = {
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
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            title = info.get("title", "Unknown")
            duration = info.get("duration", 0)
            logger.info(f"Downloaded: {title} ({duration}s)")

        wav_path = output_path + ".wav"
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Expected WAV file not found at {wav_path}")

        if status_callback:
            status_callback(f"Downloaded: {title} ({duration // 60}m {duration % 60}s)")

        return wav_path

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise RuntimeError(f"Failed to download audio from {youtube_url}: {e}")
