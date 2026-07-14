import os
from dotenv import load_dotenv

load_dotenv()

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Per-task model selection — different tasks have different needs.
# Boundary detection is mechanical (find start/end), Sonnet handles fine.
# Teaser selection requires more "taste" — Opus can pick more compelling clips
# and copy text more reliably verbatim, reducing retries.
# NOTE: claude-sonnet-4-20250514 was retired 2026-06-15 (API returns 404).
# claude-sonnet-5 is the replacement. The boundary call disables thinking
# (see boundary_detector.py) because Sonnet 5 enables adaptive thinking by
# default, which would put a thinking block first in response.content.
_legacy_claude_model = os.getenv("CLAUDE_MODEL", "").strip()
CLAUDE_MODEL_BOUNDARY = os.getenv(
    "CLAUDE_MODEL_BOUNDARY", _legacy_claude_model or "claude-sonnet-5"
)
CLAUDE_MODEL_TEASER = os.getenv("CLAUDE_MODEL_TEASER", "claude-opus-4-8")

# Legacy fallback — if old code references CLAUDE_MODEL, use boundary model
CLAUDE_MODEL = CLAUDE_MODEL_BOUNDARY

# Legacy transcription toggle (superseded by TRANSCRIBE_BACKEND below).
# "local" = faster-whisper on CPU, "cloud" = OpenAI Whisper API.
TRANSCRIBER = os.getenv("TRANSCRIBER", "cloud")

# Transcription backend selector. Supersedes TRANSCRIBER.
#   "openai"         -> OpenAI Whisper API           (default / fallback)
#   "local"          -> headless M1 mini mlx-whisper HTTP service
#   "faster-whisper" -> legacy on-CPU faster-whisper
# Back-compat: if TRANSCRIBE_BACKEND is unset we derive it from TRANSCRIBER
# (cloud -> openai, local -> faster-whisper) so existing .env files keep working.
_legacy_transcriber = (TRANSCRIBER or "").strip().lower()
_default_backend = "faster-whisper" if _legacy_transcriber == "local" else "openai"
TRANSCRIBE_BACKEND = os.getenv("TRANSCRIBE_BACKEND", _default_backend).strip().lower()

# Local M1 mini Whisper service (only used when TRANSCRIBE_BACKEND=local).
# Use the pfSense-reserved static IP, NOT whisper-mini.local (mDNS won't resolve
# from the Linux VM). Base URL only — "/transcribe" and "/health" are appended.
WHISPER_LOCAL_URL = os.getenv("WHISPER_LOCAL_URL", "")
# ~75-min sermon is a 5-8 min synchronous call, so allow a generous timeout.
WHISPER_LOCAL_TIMEOUT = int(os.getenv("WHISPER_LOCAL_TIMEOUT", "600"))
# Optional: if set, the local backend dumps the raw mini JSON response here for
# debugging (off by default). One file per audio basename. Never affects output.
WHISPER_LOCAL_DEBUG_DIR = os.getenv("WHISPER_LOCAL_DEBUG_DIR", "")

# faster-whisper settings (only used when TRANSCRIBE_BACKEND=faster-whisper)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# OpenAI API (only used when TRANSCRIBE_BACKEND=openai)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TRANSCRIPTION_TIMEOUT = int(os.getenv("OPENAI_TRANSCRIPTION_TIMEOUT", "900"))

# ── YouTube download (yt-dlp) ─────────────────────────────────────────
# Optional path to a Netscape-format cookies.txt exported from a logged-in
# browser. This is the single most effective fix for YouTube's server-side
# "Sign in to confirm you're not a bot" block. Empty = run without cookies.
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "")
# Comma-separated yt-dlp player clients to try, in order. Rotating clients is
# the primary workaround for bot-detection — "tv"/mobile clients are often
# reachable when the "web" client is blocked. Empty = yt-dlp defaults.
YTDLP_PLAYER_CLIENTS = os.getenv("YTDLP_PLAYER_CLIENTS", "tv,web_safari,ios,android,web")
# How many times to attempt the download when the error looks transient
# (throttling, 429, network blips). Non-retryable errors (private/removed/
# members-only) fail immediately regardless of this.
YTDLP_MAX_ATTEMPTS = int(os.getenv("YTDLP_MAX_ATTEMPTS", "3"))

# Audio defaults
DEFAULT_TARGET_DURATION = os.getenv("DEFAULT_TARGET_DURATION", "27:18")
DEFAULT_BROADCAST_DURATION = os.getenv("DEFAULT_BROADCAST_DURATION", "29:30")
OUTPUT_FORMAT = "mp3"
OUTPUT_BITRATE = "128k"
OUTPUT_SAMPLE_RATE = 44100

# Web server. The app is intended for a trusted private network unless an
# authenticating reverse proxy is placed in front of it.
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "5003"))
MAX_UPLOAD_GB = int(os.getenv("MAX_UPLOAD_GB", "5"))

# Bumper paths
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
INTRO_PATH = os.path.join(ASSETS_DIR, "intro.mp3")
OUTRO_PATH = os.path.join(ASSETS_DIR, "outro.mp3")

# Teaser window in the intro (seconds)
TEASER_WINDOW_START = 12.0
TEASER_WINDOW_END = 35.0

# Silence manipulation thresholds
MAX_PAUSE_DURATION_MS = 1500      # Trim pauses longer than this (ms)
TARGET_PAUSE_DURATION_MS = 800    # Trim them down to this (ms)
MIN_PAUSE_FOR_INSERT_MS = 300     # Minimum existing pause to expand when adding time
MAX_PAUSE_INSERT_MS = 2000        # Don't expand any single pause beyond this

# Tempo adjustment limits
MAX_SPEEDUP = 1.08   # Don't speed up more than 8%
MAX_SLOWDOWN = 0.93  # Don't slow down more than 7%

# Paths
WORK_DIR = os.path.join(os.path.dirname(__file__), "work")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
STATE_DIR = os.path.join(os.path.dirname(__file__), "state")
DB_PATH = os.path.join(STATE_DIR, "jobs.db")
# Persisted source audio, transcripts, waveform summaries, and previews for the
# human review step. These are intentionally separate from WORK_DIR, whose
# contents are disposable render intermediates.
REVIEW_DIR = os.path.join(STATE_DIR, "review_jobs")

# Human-review guardrails. Small timing corrections are safe to automate; a
# large shortfall usually means the selected sermon boundaries are wrong.
MAX_AUTOMATIC_SHORTFALL_SECONDS = 45
FINAL_DURATION_TOLERANCE_SECONDS = 2.0

# Feedback / GitHub issue submission
GITHUB_REPO = os.getenv("GITHUB_REPO", "larryherzogjr/sermon-broadcaster")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
FEEDBACK_SOFT_CAP = 5
FEEDBACK_HARD_CAP = 8
CLAUDE_MODEL_FEEDBACK = os.getenv("CLAUDE_MODEL_FEEDBACK", "claude-opus-4-8")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)
